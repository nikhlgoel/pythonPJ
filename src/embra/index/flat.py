"""Exhaustive (brute-force) index.

Exact by construction, and therefore the ground truth against which the HNSW
recall numbers in :mod:`embra.bench` are measured.  It is also the right index
for small collections, where a single vectorised pass beats any graph walk.
"""

from __future__ import annotations

import numpy as np

from ..types import Metric
from .base import Predicate, VectorIndex
from .space import ExactSpace, PQSpace, VectorSpace


class FlatIndex(VectorIndex):
    """Linear scan over every live vector."""

    def __init__(self, space: VectorSpace) -> None:
        self.space = space
        self.metric: Metric = space.metric
        self._live: set[int] = set()

    @property
    def size(self) -> int:
        return len(self._live)

    def add(self, key: int, vector: np.ndarray) -> None:
        self.space.set(key, np.ascontiguousarray(vector, dtype=np.float32))
        self._live.add(key)

    def remove(self, key: int) -> None:
        self._live.discard(key)

    def search(
        self,
        query: np.ndarray,
        k: int,
        *,
        ef: int | None = None,
        allow: Predicate | None = None,
    ) -> list[tuple[int, float]]:
        if k <= 0 or not self._live:
            return []
        keys = np.fromiter(
            (kk for kk in sorted(self._live) if allow is None or allow(kk)),
            dtype=np.int64,
        )
        if keys.size == 0:
            return []
        query = np.ascontiguousarray(query, dtype=np.float32)
        if isinstance(self.space, PQSpace) and self.space.is_trained:
            dists = self.space.exact_distances(query, keys)
        else:
            dists = self.space.distances(query, keys)
        k = min(k, keys.size)
        part = np.argpartition(dists, k - 1)[:k]
        order = part[np.argsort(dists[part], kind="stable")]
        return [(int(keys[i]), float(dists[i])) for i in order]

    def stats(self) -> dict:
        return {
            "type": "flat",
            "size": self.size,
            "memory_bytes": self.space.memory_bytes(),
            "exact": True,
        }


def exact_knn(data: np.ndarray, query: np.ndarray, k: int, metric: Metric) -> list[tuple[int, float]]:
    """Standalone exact k-NN helper used by the benchmark harness."""
    space = ExactSpace(data.shape[1], metric, capacity=max(len(data), 1))
    for i, row in enumerate(data):
        space.set(i, row)
    idx = FlatIndex(space)
    idx._live = set(range(len(data)))
    return idx.search(query, k)
