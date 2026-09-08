"""Reproducible benchmark suite.

Measures, for each configuration:

* **recall@k** against exact brute-force ground truth,
* **latency** p50 / p95 / p99 (ms) over the query set,
* **throughput** (queries/second, single-threaded),
* **build time** and resident index memory.

Run it with ``python -m annex.bench.suite --n 50000 --dim 128``.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from ..config import CollectionConfig, HNSWConfig, PQConfig
from ..index import build_index
from ..index.base import VectorIndex
from ..types import IndexKind, Metric
from .datasets import clustered_dataset, query_set


@dataclass
class BenchmarkResult:
    name: str
    n: int
    dim: int
    k: int
    recall: float
    build_seconds: float
    qps: float
    p50_ms: float
    p95_ms: float
    p99_ms: float
    memory_mb: float
    params: dict = field(default_factory=dict)

    def row(self) -> str:
        return (
            f"{self.name:<22} recall@{self.k}={self.recall:6.3f}  "
            f"qps={self.qps:8.1f}  p50={self.p50_ms:6.2f}ms  p95={self.p95_ms:6.2f}ms  "
            f"build={self.build_seconds:6.2f}s  mem={self.memory_mb:6.1f}MB"
        )


def exact_ground_truth(data: np.ndarray, queries: np.ndarray, k: int, metric: Metric) -> np.ndarray:
    """``(len(queries), k)`` matrix of true nearest-neighbour ids."""
    out = np.empty((queries.shape[0], k), dtype=np.int64)
    for i, q in enumerate(queries):
        if metric is Metric.L2:
            diff = data - q
            dist = np.einsum("ij,ij->i", diff, diff)
        else:
            dist = -(data @ q)
        part = np.argpartition(dist, k - 1)[:k]
        out[i] = part[np.argsort(dist[part])]
    return out


def recall_at_k(found: list[list[int]], truth: np.ndarray, k: int) -> float:
    total = 0.0
    for got, want in zip(found, truth, strict=True):
        total += len(set(got[:k]) & set(want[:k].tolist())) / k
    return total / len(found) if found else 0.0


def benchmark_index(
    name: str,
    index: VectorIndex,
    data: np.ndarray,
    queries: np.ndarray,
    truth: np.ndarray,
    k: int,
    *,
    ef: int | None = None,
    params: dict | None = None,
) -> BenchmarkResult:
    start = time.perf_counter()
    space: Any = getattr(index, "space", None)
    if space is not None and hasattr(space, "train") and not getattr(space, "is_trained", True):
        space.train(data)
    for i, vec in enumerate(data):
        index.add(i, vec)
    build_seconds = time.perf_counter() - start

    latencies: list[float] = []
    found: list[list[int]] = []
    for q in queries:
        t0 = time.perf_counter()
        hits = index.search(q, k, ef=ef)
        latencies.append((time.perf_counter() - t0) * 1000.0)
        found.append([key for key, _ in hits])

    latencies.sort()
    def pct(p: float) -> float:
        return latencies[min(len(latencies) - 1, int(len(latencies) * p))]

    total_s = sum(latencies) / 1000.0
    return BenchmarkResult(
        name=name,
        n=int(data.shape[0]),
        dim=int(data.shape[1]),
        k=k,
        recall=recall_at_k(found, truth, k),
        build_seconds=build_seconds,
        qps=len(queries) / total_s if total_s else float("inf"),
        p50_ms=statistics.median(latencies),
        p95_ms=pct(0.95),
        p99_ms=pct(0.99),
        memory_mb=(space.memory_bytes() / 1e6) if space is not None else 0.0,
        params=params or {},
    )


def run_suite(
    n: int = 20_000,
    dim: int = 128,
    queries: int = 200,
    k: int = 10,
    *,
    seed: int = 0,
    include_flat: bool = True,
    ef_values: tuple[int, ...] = (32, 64, 128, 256),
) -> list[BenchmarkResult]:
    data = clustered_dataset(n, dim, seed=seed)
    qs = query_set(data, queries, seed=seed + 1)
    truth = exact_ground_truth(data, qs, k, Metric.COSINE)

    results: list[BenchmarkResult] = []
    if include_flat:
        cfg = CollectionConfig(name="bench", dim=dim, index=IndexKind.FLAT)
        results.append(
            benchmark_index("flat (exact)", build_index(cfg, n), data, qs, truth, k)
        )

    hnsw_cfg = CollectionConfig(
        name="bench", dim=dim, index=IndexKind.HNSW,
        hnsw=HNSWConfig(m=16, ef_construction=200),
    )
    index = build_index(hnsw_cfg, n)
    built = False
    for ef in ef_values:
        if not built:
            results.append(
                benchmark_index(
                    f"hnsw ef={ef}", index, data, qs, truth, k, ef=ef,
                    params={"m": 16, "ef_construction": 200, "ef_search": ef},
                )
            )
            built = True
        else:
            res = benchmark_index_prebuilt(f"hnsw ef={ef}", index, qs, truth, k, ef, n, dim)
            results.append(res)

    subvectors = max(1, dim // 4)
    if dim % subvectors == 0:
        for label, keep in (("hnsw+pq", True), ("hnsw+pq (codes only)", False)):
            pq_cfg = CollectionConfig(
                name="bench", dim=dim, index=IndexKind.HNSW_PQ,
                hnsw=HNSWConfig(m=16, ef_construction=200),
                pq=PQConfig(subvectors=subvectors, bits=8, keep_originals=keep),
            )
            pq_index = build_index(pq_cfg, n)
            res = benchmark_index(
                f"{label} ef=128", pq_index, data, qs, truth, k, ef=128,
                params={"subvectors": subvectors, "bits": 8, "keep_originals": keep},
            )
            space: Any = pq_index.space
            res.params["code_mb"] = round(space.code_bytes() / 1e6, 2)
            res.params["compression_ratio"] = space.compression_ratio()
            results.append(res)
    return results


def benchmark_index_prebuilt(
    name: str,
    index: VectorIndex,
    queries: np.ndarray,
    truth: np.ndarray,
    k: int,
    ef: int,
    n: int,
    dim: int,
) -> BenchmarkResult:
    latencies: list[float] = []
    found: list[list[int]] = []
    for q in queries:
        t0 = time.perf_counter()
        hits = index.search(q, k, ef=ef)
        latencies.append((time.perf_counter() - t0) * 1000.0)
        found.append([key for key, _ in hits])
    latencies.sort()
    total_s = sum(latencies) / 1000.0
    space: Any = getattr(index, "space", None)
    return BenchmarkResult(
        name=name, n=n, dim=dim, k=k,
        recall=recall_at_k(found, truth, k),
        build_seconds=0.0,
        qps=len(queries) / total_s if total_s else float("inf"),
        p50_ms=statistics.median(latencies),
        p95_ms=latencies[min(len(latencies) - 1, int(len(latencies) * 0.95))],
        p99_ms=latencies[min(len(latencies) - 1, int(len(latencies) * 0.99))],
        memory_mb=(space.memory_bytes() / 1e6) if space is not None else 0.0,
        params={"ef_search": ef},
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Annex benchmark suite")
    parser.add_argument("--n", type=int, default=20_000)
    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument("--queries", type=int, default=200)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-flat", action="store_true")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)

    results = run_suite(
        args.n, args.dim, args.queries, args.k,
        seed=args.seed, include_flat=not args.no_flat,
    )
    print(f"\nAnnex benchmark - n={args.n} dim={args.dim} queries={args.queries} k={args.k}\n")
    for res in results:
        print(res.row())
    if args.out:
        args.out.write_text(json.dumps([asdict(r) for r in results], indent=2))
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
