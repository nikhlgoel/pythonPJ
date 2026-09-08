# Algorithms

## 1. HNSW

Hierarchical Navigable Small World graphs (Malkov & Yashunin, 2016) give
`O(log N)` expected search by stacking navigable small-world graphs into a
skip-list-like hierarchy.

**Level assignment.** A node's top level is drawn as

$$\ell = \lfloor -\ln(U) \cdot m_L \rfloor, \qquad m_L = \frac{1}{\ln M}$$

so level occupancy decays geometrically and the top layer holds `O(1)` nodes.

**Search.** Greedy descent with beam width 1 from the entry point down to layer 1,
then a beam search of width `ef` on layer 0. Increasing `ef` monotonically trades
latency for recall; it is *the* tuning dial.

**Neighbour selection (Algorithm 4).** Keeping the `M` closest candidates is the
naive choice and it fails on clustered data: every link points into the same
dense blob and the graph loses long-range navigability. Instead a candidate `c`
at distance `d(q,c)` is kept only if

$$d(q, c) < \min_{s \in \text{selected}} d(c, s)$$

i.e. `c` is closer to the new node than to anything already selected. This
*diversifies* the neighbourhood. Rejected candidates are used to top the degree
back up to `M` (`keepPrunedConnections`).

**Pruning.** Layer 0 allows `M_max0 = 2M` links, upper layers `M`. When a
neighbour exceeds its budget its list is re-selected with the same heuristic.

**Deletes.** Tombstoned: the node stays in the graph (so connectivity is
preserved) but is excluded from the result heap. `rebuild_index()` compacts.

**Filtering.** The predicate is applied when a node would enter the result heap,
*not* when it is traversed. Filtering traversal would disconnect the graph and
collapse recall.

**Complexity.** `O(log N)` hops, `O(M)` distance computations per hop; every
expansion is evaluated as one vectorised NumPy call.

## 2. Product quantisation

Split `x ∈ R^d` into `m` sub-vectors and quantise each with its own
`K = 2^bits` k-means codebook. Storage drops from `4d` bytes to `m` bytes
(32× at `d=128, m=8`).

Both distance families decompose additively:

$$\|q-x\|^2 = \sum_s \|q_s - x_s\|^2, \qquad \langle q,x\rangle = \sum_s \langle q_s, x_s\rangle$$

**ADC (asymmetric).** Build an `(m, K)` table of query-to-centroid partial
distances once per query; scoring a candidate is then `m` lookups and `m-1` adds
instead of `d` multiply-adds.

**SDC (symmetric).** An `(m, K, K)` centroid-to-centroid table lets two *codes*
be compared directly — used during graph construction when originals are absent.

**Reranking.** The graph returns `k × rerank_factor` candidates using approximate
distances; those are re-scored exactly, which recovers most of the recall PQ
costs.

**Codebooks.** k-means++ seeding, Lloyd iterations, empty clusters re-seeded on
the worst-represented point so no code word is wasted.

## 3. BM25

$$\text{score}(d,q) = \sum_{t \in q} \text{idf}(t)\cdot\frac{f_{t,d}\,(k_1+1)}{f_{t,d} + k_1\left(1-b+b\frac{|d|}{\text{avgdl}}\right)}$$

$$\text{idf}(t) = \ln\!\left(1 + \frac{N - df_t + 0.5}{df_t + 0.5}\right)$$

`k1 = 1.2` controls term-frequency saturation; `b = 0.75` controls length
normalisation. The index is incremental: postings carry document keys, so deletes
and re-indexing are `O(terms in document)` rather than a full rebuild.

## 4. Hybrid fusion

Cosine similarity ∈ `[-1, 1]` and BM25 ∈ `[0, ∞)` are not comparable, and BM25's
scale shifts with corpus statistics. Annex therefore fuses **ranks**:

$$\text{RRF}(d) = \sum_r \frac{w_r}{K + \text{rank}_r(d)}, \qquad K = 60$$

RRF needs no calibration, degrades gracefully when one retriever returns nothing,
and empirically beats tuned linear interpolation on heterogeneous retrievers.
`weighted_fusion()` is available when scores *are* calibrated.

## 5. The query planner

Cost is measured in distance-computation equivalents.

```
survivors   = live × selectivity(filter)
scan_cost   = survivors / scan_speedup            # vectorised, cache friendly
graph_cost  = graph_factor × ef × log2(live)
if filter:  graph_cost /= selectivity             # most visited nodes get rejected
```

Choose the exact scan when the index is `flat`, when `live` is below the
small-collection threshold, or when `scan_cost ≤ graph_cost`.

Selectivity uses a simple prior — each equality clause ≈ 10× reduction, each
range clause ≈ 3× — clamped to `[0.001, 1]`. That is coarse on purpose: the
decision boundary is broad, and the alternative (per-field histograms) is on the
[roadmap](ROADMAP.md), not in the critical path.

The chosen plan, its costs and a human-readable `reason` are returned with every
result, which makes the planner debuggable rather than magical.

## 6. Raft

Implemented per figure 2 of Ongaro & Ousterhout (2014):

* **Election safety** — one vote per term, granted only to a candidate whose log
  is at least as up-to-date (`lastTerm`, then `lastIndex`).
* **Log matching** — `AppendEntries` carries `(prevIndex, prevTerm)`; a follower
  rejects anything that would leave a hole and truncates a conflicting suffix
  before appending. Rejections carry a `conflict_index` back-off hint so a lagging
  follower converges in a couple of round trips rather than one entry per RPC.
* **Leader completeness** — the leader advances `commitIndex` only to an entry
  **from its own term** replicated on a quorum. Skipping that rule is the classic
  way to lose a committed entry after a leader change.
