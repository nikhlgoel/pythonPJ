<h1 align="center">Embra</h1>

<p align="center">
  <b>An embedded, transactional, vector-native search engine — written from scratch in Python.</b><br>
  HNSW · product quantisation · LSM storage with WAL &amp; crash recovery · MVCC snapshots · BM25 hybrid search · cost-based query planner · Raft replication
</p>

---

## Why this exists

Vector databases are usually consumed as a black box: you call `add()` and `query()`
and hope the recall is good. Embra is the opposite — it is a *readable*
implementation of every layer that a production vector database actually needs,
built without FAISS, without hnswlib, without SQLite, and without a single
`pip install` beyond NumPy.

It is a real engine, not a demo:

| Layer | What is implemented | Where |
|---|---|---|
| **ANN index** | HNSW with exponential level assignment, the Algorithm-4 neighbour-selection heuristic, bidirectional pruning, batched distance evaluation, tombstoned deletes, predicate-filtered traversal | [`index/hnsw.py`](src/embra/index/hnsw.py) |
| **Compression** | Product quantisation with k-means++ codebooks, asymmetric (ADC) and symmetric (SDC) distance computation, exact reranking | [`index/pq.py`](src/embra/index/pq.py) |
| **Storage** | Write-ahead log with CRC32 framing and torn-tail truncation, immutable segments written atomically, memtable flush, full compaction | [`storage/`](src/embra/storage) |
| **Concurrency** | MVCC — every update creates a new version; snapshot reads are an integer comparison; vacuum uses an oldest-snapshot horizon | [`mvcc.py`](src/embra/mvcc.py) |
| **Transactions** | Optimistic, buffered, first-committer-wins conflict detection | [`db.py`](src/embra/db.py) |
| **Lexical search** | Incremental BM25 inverted index with deletes and a deterministic analyzer | [`text/bm25.py`](src/embra/text/bm25.py) |
| **Query layer** | MongoDB-style filter DSL compiled to a closure, reciprocal rank fusion, **cost-based planner that picks between exact scan / graph traversal / hybrid** | [`query/`](src/embra/query) |
| **Distribution** | Raft — leader election, log matching, the leader-completeness commit rule — implemented as a *pure, tick-driven* state machine so a 5-node partition is a unit test | [`cluster/raft.py`](src/embra/cluster/raft.py) |
| **Serving** | FastAPI HTTP API, Typer CLI, benchmark harness reporting recall@k + p50/p95/p99 | [`server/`](src/embra/server), [`cli.py`](src/embra/cli.py), [`bench/`](src/embra/bench) |

## Install

```bash
git clone https://github.com/nikhlgoel/pythonpj.git && cd pythonpj
make install          # creates .venv and installs embra[all]
make test             # full test suite
make bench            # recall / latency benchmark
```

## 60-second tour

```python
import numpy as np
import embra

db = embra.Database("./data")
papers = db.create_collection("papers", dim=384, metric="cosine", index="hnsw")

papers.upsert(
    "2306.01234",
    np.random.rand(384),
    metadata={"year": 2023, "venue": "NeurIPS", "tags": ["retrieval", "ann"]},
    text="Efficient approximate nearest neighbour search with learned partitions",
)

# dense + lexical, fused with reciprocal rank fusion, filtered on metadata
result = papers.search(
    query_embedding,
    text="approximate nearest neighbour",
    k=10,
    filter={"$and": [{"year": {"$gte": 2020}}, {"tags": {"$contains": "ann"}}]},
)

for hit in result:
    print(f"{hit.score:6.3f}  {hit.id}  {hit.text}")

print(result.plan)   # EXPLAIN: which strategy the planner chose, and why
```

### Snapshots and transactions

```python
with papers.snapshot() as snap:          # pinned, consistent view
    before = papers.search(q, k=5, snapshot=snap)

with papers.transaction() as txn:        # atomic; rolls back on exception
    txn.upsert("a", vec_a, {"status": "published"})
    txn.delete("b")
```

### The query plan is a first-class object

```python
>>> papers.search(q, k=10, filter={"venue": "NeurIPS"}).plan
{'strategy': 'exact_scan', 'k': 10, 'ef': 0, 'fetch_k': 10, 'use_filter': True,
 'live': 1200, 'selectivity': 0.1, 'estimated_survivors': 120,
 'scan_cost': 10.0, 'graph_cost': 1030.4, 'index': 'hnsw',
 'reason': 'selective filter makes a pre-filtered scan cheaper'}
```

That decision is the point: with a 10 %-selective filter a graph walk wastes
most of its distance computations on rejected nodes, so Embra pre-filters and
scans — and tells you it did.

## CLI

```bash
embra create papers --dim 384 --index hnsw
embra ingest papers corpus.jsonl
embra query  papers --text "graph neural networks" -k 5
embra stats  papers
embra bench  --n 50000 --dim 128
embra serve  --port 8080
```

## HTTP API

```bash
make serve
curl -s localhost:8080/health
curl -s -X POST localhost:8080/v1/collections -d '{"name":"docs","dim":8}' -H 'content-type: application/json'
curl -s -X POST localhost:8080/v1/collections/docs/search \
     -H 'content-type: application/json' \
     -d '{"vector":[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8],"k":3}'
```

Interactive OpenAPI docs at `http://localhost:8080/docs`.

## Documentation

* [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) — layer-by-layer design and data flow
* [`docs/ALGORITHMS.md`](docs/ALGORITHMS.md) — HNSW, PQ, BM25, RRF and the planner cost model, with the maths
* [`docs/STORAGE.md`](docs/STORAGE.md) — file formats, durability and recovery guarantees
* [`docs/BENCHMARKS.md`](docs/BENCHMARKS.md) — how to reproduce the numbers
* [`docs/ROADMAP.md`](docs/ROADMAP.md) — what a production/commercial version needs next

## Guarantees (and honest limits)

**Guaranteed**

* An acknowledged write is in the WAL; a crash at any point recovers to the last
  acknowledged write, and a torn tail is detected by CRC and truncated.
* Readers never block writers, and a snapshot sees a consistent point in time.
* `flat` results are exact; `hnsw` recall is tunable via `ef` and measured by
  the benchmark suite against exact ground truth.

**Not claimed**

* This is pure Python + NumPy. Throughput is one to two orders of magnitude
  below a SIMD C++ engine; the design, not the constant factor, is the artefact.
* Raft is implemented and simulated deterministically, but the HTTP transport
  binding it to a live multi-process cluster is future work
  ([roadmap](docs/ROADMAP.md)).
* Single-writer per collection (a lock); concurrent *readers* scale freely.

## Licence

Apache-2.0.
