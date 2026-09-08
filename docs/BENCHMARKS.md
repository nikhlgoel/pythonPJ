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

## Measured results

`n=20,000`, `dim=128`, 200 queries, `k=10`, clustered data, single thread,
Python 3.11 + NumPy on one container vCPU. Raw output:
[`benchmarks/results-n20000-d128.txt`](../benchmarks/results-n20000-d128.txt),
machine-readable JSON alongside it.

| Configuration | recall@10 | QPS | p50 | p95 | vector memory |
|---|---:|---:|---:|---:|---:|
| `flat (exact)` | **1.000** | 403 | 2.37 ms | 2.70 ms | 10.2 MB |
| `hnsw ef=32` | 0.977 | **1308** | 0.77 ms | 1.02 ms | 10.2 MB |
| `hnsw ef=64` | 0.987 | 979 | 1.00 ms | 1.24 ms | 10.2 MB |
| `hnsw ef=128` | 0.994 | 499 | 1.90 ms | 2.54 ms | 10.2 MB |
| `hnsw ef=256` | 0.997 | 274 | 3.64 ms | 4.07 ms | 10.2 MB |
| `hnsw+pq ef=128` (rerank on originals) | 0.986 | 131 | 7.54 ms | 8.49 ms | 10.9 MB |
| `hnsw+pq ef=128` (codes only) | 0.636 | 114 | 8.64 ms | 10.44 ms | **0.6 MB** |

### What these numbers say

1. **`ef` is a clean recall/latency dial.** 0.977 → 0.997 recall costs 4.8× the
   latency. Recall saturates long before latency does, which is why `ef` is
   exposed per query and not just per collection.
2. **HNSW at `ef=32` is 3.2× the throughput of an exact scan at 97.7% recall** —
   and the crossover is real: at `n=4,000` the exact scan *wins outright*. That
   crossover is the entire justification for the query planner and for
   `small_collection_threshold = 2000`.
3. **PQ buys 17× less vector memory (10.2 MB → 0.6 MB) at 0.64 recall on codes
   alone; keeping originals for the rerank stage restores 0.986.** That is the
   honest PQ trade-off: the codes are for *navigation*, and exactness has to come
   from somewhere. The middle ground — codes in RAM, originals on disk, fetched
   only for the `k × rerank_factor` finalists — is the production shape and is on
   the [roadmap](ROADMAP.md).
4. **Sub-space granularity dominates PQ recall.** With `dim=128`, moving from
   8 sub-vectors (16 dims each) to 32 (4 dims each) took recall from 0.61 to
   0.97 at essentially the same memory. The default is `dim // 4`.

Build times (121 s for 20k HNSW inserts) are the pure-Python tax: construction
runs the neighbour-selection heuristic in Python for every insert. Query time is
vectorised and therefore far healthier than build time. Native construction
kernels are roadmap item #1.

## Reproducibility

Every dataset and index is seeded (`--seed`). The same command on the same
machine produces the same recall to the digit; latency varies with the host.
