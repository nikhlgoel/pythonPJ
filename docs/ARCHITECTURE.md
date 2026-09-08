# Architecture

```
                         ┌──────────────────────────────────────────┐
   client (py / http)    │              annex.Database              │
        │                └──────────────────────────────────────────┘
        ▼                                   │
┌───────────────┐                           ▼
│  Transaction  │──stage──▶┌────────────────────────────────────────┐
└───────────────┘          │             Collection                 │
                           │  ┌──────────────────────────────────┐  │
        search()  ────────▶│  │   QueryPlanner  (EXPLAIN)        │  │
                           │  └──────────────────────────────────┘  │
                           │      │            │            │       │
                           │      ▼            ▼            ▼       │
                           │  exact scan   HNSW graph   BM25 index  │
                           │      └────── RRF fusion ─────┘         │
                           │                                        │
                           │  MVCC version table (born, dead)       │
                           └────────────────┬───────────────────────┘
                                            ▼
                           ┌────────────────────────────────────────┐
                           │              LSMStore                  │
                           │   WAL  ──flush──▶ segments ──compact──▶│
                           └────────────────────────────────────────┘
```

## Layers

### 1. Storage (`annex/storage`)

| File | Responsibility |
|---|---|
| `codec.py` | Binary frame format (magic, version, type, seq, length, CRC32) and record payload encoding |
| `wal.py` | Append-only log; replay with torn-tail truncation; rotation after a flush |
| `segment.py` | Immutable segment files, written to `*.tmp` and `os.replace`-d into place |
| `lsm.py` | Memtable, flush, full compaction, manifest, recovery |

**Write path.** `put()` → append a CRC-framed `PUT` frame to the WAL → update the
memtable and the merged in-memory view → return. Once the memtable exceeds
`memtable_max_records` it is flushed to a segment and the WAL is rotated.

**Recovery.** Read `MANIFEST.json`, replay segments oldest-first, then replay the
WAL tail. Higher sequence numbers win. A frame that fails its CRC ends replay and
the file is truncated at that offset — a partially written frame was never
acknowledged, so discarding it is correct.

### 2. Vectors (`annex/index`)

Two orthogonal abstractions:

* **`VectorSpace`** — *where vectors live and how distances are computed*.
  `ExactSpace` keeps `float32`; `PQSpace` keeps `m`-byte product-quantised codes
  (optionally with originals for reranking).
* **`VectorIndex`** — *how to avoid computing most distances*. `FlatIndex`
  (exhaustive, exact) and `HNSWIndex` (graph).

Because `HNSWIndex` only talks to `VectorSpace`, the identical graph code runs
over exact vectors *and* over 32×-compressed codes.

Internal keys are dense integers. That is deliberate: it lets the graph, the
BM25 index and the MVCC table share one address space and lets every distance
computation be a NumPy fancy-index rather than a dict lookup.

### 3. Concurrency (`annex/mvcc.py`)

An update never mutates a vector. It allocates a new key and stamps the previous
version `dead = commit_seq`:

```
key 7   doc="a"  born=3  dead=9        old version
key 12  doc="a"  born=9  dead=inf      current
```

A snapshot is an integer. Visibility is `born <= snap < dead` — cheap enough to
evaluate inside the graph traversal, which is why readers never block writers and
why an old snapshot keeps returning the old vector. `vacuum()` reclaims versions
below the oldest open snapshot (the PostgreSQL horizon rule).

### 4. Query (`annex/query`)

1. `compile_filter()` turns the filter DSL into a closure plus selectivity stats.
2. `QueryPlanner` estimates the cost of an exact scan versus a graph traversal
   (see [ALGORITHMS.md](ALGORITHMS.md)) and picks a strategy.
3. The dense stage runs; the lexical stage runs if a query string was supplied.
4. `reciprocal_rank_fusion()` merges the two ranked lists.
5. The chosen plan is returned to the caller in `SearchResult.plan`.

### 5. Replication (`annex/cluster`)

`RaftNode` is a pure state machine: `tick(now)` and `handle(message, now)` return
the messages to send. No threads, no clocks, no sockets. `ClusterSimulator`
supplies a virtual network with latency, loss and partitions, so
*"the cluster keeps exactly one leader across a partition and reconverges after
healing"* is an ordinary, deterministic unit test.

`CollectionStateMachine` applies committed commands to a local collection, making
a collection the replicated state machine.

## Concurrency model

* One writer per collection (an `RLock` around the write path).
* Unlimited concurrent readers; reads take no lock, only a sequence number.
* The HTTP server is async; engine calls are short and CPU-bound in NumPy.

## Failure model

| Failure | Behaviour |
|---|---|
| Crash between WAL append and memtable update | Replay reconstructs it |
| Crash mid-`write` (torn frame) | CRC mismatch → truncate; the write was never acknowledged |
| Crash during segment flush | `*.tmp` is orphaned; the manifest still points at the previous set |
| Corrupted segment vector block | CRC check fails at open with `CorruptionError` |
| Leader loss (cluster) | New election after a randomised timeout; committed entries survive by the leader-completeness rule |
