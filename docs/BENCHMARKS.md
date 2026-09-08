# Benchmarks

## Running

```bash
make bench
# or
python -m embra.bench --n 50000 --dim 128 --queries 500 --out results.json
embra bench --n 20000 --dim 128
```

## What is measured

* **recall@k** against exact brute-force ground truth on the same data
* **latency** p50 / p95 / p99, single-threaded, per query
* **throughput** queries per second
* **build time** and resident index memory

## Method

* Data is a **mixture of Gaussians** (`clustered_dataset`), not uniform noise.
  Uniform vectors are the easy case for ANN and flatter the graph; real
  embeddings are clustered, which is precisely where a naive neighbour-selection
  heuristic collapses.
* Queries are drawn near existing points with small jitter — the realistic
  retrieval regime, rather than random points in a space where nothing is close.
* Ground truth is exact k-NN under the same metric; recall@k is
  `|found ∩ true| / k` averaged over the query set.
* One process, one thread, no warm-up tricks. Numbers are from pure Python +
  NumPy, so treat them as *relative* comparisons between strategies, not as
  competition with a SIMD C++ engine.

## Reading the results

Checked-in runs live in [`../benchmarks/`](../benchmarks). The shape to expect:

| Configuration | recall@10 | Notes |
|---|---|---|
| `flat (exact)` | 1.000 | The ground truth. Fastest option on small collections. |
| `hnsw ef=32` | ~0.90 | Lowest latency graph setting |
| `hnsw ef=128` | ~0.98 | The usual production point |
| `hnsw ef=256` | ~0.99 | Diminishing returns |
| `hnsw+pq` | lower, ~32× less vector memory | Reranking recovers most of the loss |

Two results are worth internalising because they justify design decisions in the
engine:

1. **Below a few thousand vectors, `flat` beats `hnsw` outright.** A single
   BLAS-backed scan is faster than a Python graph walk. This is exactly why the
   query planner exists and why `small_collection_threshold` defaults to 2000.
2. **`ef` is a monotone recall/latency dial.** Doubling `ef` roughly doubles the
   distance computations, and recall saturates well before latency does — which
   is why `ef` is exposed per query, not just per collection.

## Reproducibility

Every dataset and index is seeded (`--seed`). The same command on the same
machine produces the same recall to the digit; latency varies with the host.
