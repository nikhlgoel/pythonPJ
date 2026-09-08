"""Vector spaces: the storage + distance backend an index sits on top of.

Separating *how vectors are stored* from *how the search graph is traversed*
means the HNSW implementation is written once and reused verbatim for the exact
and the product-quantised variants.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from ..config import PQConfig
from ..errors import IndexError_
from ..types import Metric
from ..util.math import distance_matrix, distance_pair
from .pq import ProductQuantizer

_GROWTH = 1.6


class VectorSpace(ABC):
    """Growable, dense storage addressed by internal key == row index."""

    def __init__(self, dim: int, metric: Metric, capacity: int = 1024) -> None:
        self.dim = dim
        self.metric = metric
        self._capacity = max(capacity, 1)
        self._count = 0

    @property
    def count(self) -> int:
        return self._count

    def _ensure(self, key: int) -> None:
        if key < 0:
            raise IndexError_("keys must be non-negative")
        if key + 1 > self._capacity:
            new_cap = max(int(self._capacity * _GROWTH) + 1, key + 1)
            self._grow(new_cap)
            self._capacity = new_cap
        self._count = max(self._count, key + 1)

    @abstractmethod
    def _grow(self, new_cap: int) -> None: ...

    @abstractmethod
    def set(self, key: int, vector: np.ndarray) -> None:
        """Store ``vector`` at ``key``."""

    @abstractmethod
    def distances(self, query: np.ndarray, keys: np.ndarray) -> np.ndarray:
        """Distances from a full-precision query to each of ``keys``."""

    @abstractmethod
    def distance_between(self, a: int, b: int) -> float:
        """Distance between two stored vectors."""

    @abstractmethod
    def query_context(self, query: np.ndarray):
        """Return an opaque per-query object reused across ``distances_ctx``."""

    @abstractmethod
    def distances_ctx(self, ctx, keys: np.ndarray) -> np.ndarray:
        """Distances using a prepared context (avoids re-building ADC tables)."""

    def memory_bytes(self) -> int:
        return 0


class ExactSpace(VectorSpace):
    """Full-precision ``float32`` storage; distances are exact."""

    def __init__(self, dim: int, metric: Metric, capacity: int = 1024) -> None:
        super().__init__(dim, metric, capacity)
        self._data = np.zeros((self._capacity, dim), dtype=np.float32)

    def _grow(self, new_cap: int) -> None:
        buf = np.zeros((new_cap, self.dim), dtype=np.float32)
        buf[: self._data.shape[0]] = self._data
        self._data = buf

    @property
    def data(self) -> np.ndarray:
        return self._data[: self._count]

    def set(self, key: int, vector: np.ndarray) -> None:
        self._ensure(key)
        self._data[key] = vector

    def get(self, key: int) -> np.ndarray:
        return self._data[key]

    def distances(self, query: np.ndarray, keys: np.ndarray) -> np.ndarray:
        if len(keys) == 0:
            return np.empty(0, dtype=np.float32)
        return distance_matrix(query, self._data[keys], self.metric)

    def distance_between(self, a: int, b: int) -> float:
        return distance_pair(self._data[a], self._data[b], self.metric)

    def query_context(self, query: np.ndarray):
        return np.ascontiguousarray(query, dtype=np.float32)

    def distances_ctx(self, ctx, keys: np.ndarray) -> np.ndarray:
        return self.distances(ctx, keys)

    def memory_bytes(self) -> int:
        return int(self._data.nbytes)


class PQSpace(VectorSpace):
    """Product-quantised storage: ``m`` bytes per vector.

    Optionally keeps the full-precision vectors around for *reranking*; the
    collection turns this on when exact top-k ordering matters more than RAM.
    """

    def __init__(
        self,
        dim: int,
        metric: Metric,
        config: PQConfig,
        *,
        capacity: int = 1024,
        keep_originals: bool = True,
    ) -> None:
        super().__init__(dim, metric, capacity)
        self.pq = ProductQuantizer(dim=dim, config=config, metric=metric)
        self._codes = np.zeros((self._capacity, config.subvectors), dtype=np.uint8)
        self.keep_originals = keep_originals
        self._raw = np.zeros((self._capacity, dim), dtype=np.float32) if keep_originals else None
        self._pending: dict[int, np.ndarray] = {}

    # -- lifecycle -----------------------------------------------------
    @property
    def is_trained(self) -> bool:
        return self.pq.is_trained

    def train(self, sample: np.ndarray) -> None:
        self.pq.train(sample)
        if self._pending:
            keys = np.fromiter(self._pending.keys(), dtype=np.int64)
            mat = np.stack([self._pending[int(k)] for k in keys])
            self._codes[keys] = self.pq.encode(mat)
            self._pending.clear()

    def _grow(self, new_cap: int) -> None:
        codes = np.zeros((new_cap, self.pq.m), dtype=np.uint8)
        codes[: self._codes.shape[0]] = self._codes
        self._codes = codes
        if self._raw is not None:
            raw = np.zeros((new_cap, self.dim), dtype=np.float32)
            raw[: self._raw.shape[0]] = self._raw
            self._raw = raw

    def set(self, key: int, vector: np.ndarray) -> None:
        self._ensure(key)
        if self._raw is not None:
            self._raw[key] = vector
        if self.pq.is_trained:
            self._codes[key] = self.pq.encode(vector.reshape(1, -1))[0]
        else:
            self._pending[key] = np.array(vector, dtype=np.float32, copy=True)

    # -- distances -----------------------------------------------------
    def query_context(self, query: np.ndarray):
        if not self.pq.is_trained:
            return ("exact", np.ascontiguousarray(query, dtype=np.float32))
        table, const = self.pq.distance_table(query)
        return ("adc", table, const)

    def distances_ctx(self, ctx, keys: np.ndarray) -> np.ndarray:
        if len(keys) == 0:
            return np.empty(0, dtype=np.float32)
        if ctx[0] == "exact":
            if self._raw is None:
                raise IndexError_("PQ space is untrained and keeps no originals")
            return distance_matrix(ctx[1], self._raw[keys], self.metric)
        return self.pq.adc(ctx[1], ctx[2], self._codes[keys])

    def distances(self, query: np.ndarray, keys: np.ndarray) -> np.ndarray:
        return self.distances_ctx(self.query_context(query), keys)

    def exact_distances(self, query: np.ndarray, keys: np.ndarray) -> np.ndarray:
        """Full-precision distances used by the rerank stage."""
        if self._raw is None:
            recon = self.pq.decode(self._codes[keys])
            return distance_matrix(query, recon, self.metric)
        return distance_matrix(query, self._raw[keys], self.metric)

    def distance_between(self, a: int, b: int) -> float:
        if self._raw is not None:
            return distance_pair(self._raw[a], self._raw[b], self.metric)
        if not self.pq.is_trained:
            raise IndexError_("cannot compare codes before training")
        return self.pq.sdc(self._codes[a], self._codes[b])

    def memory_bytes(self) -> int:
        total = int(self._codes.nbytes)
        if self._raw is not None:
            total += int(self._raw.nbytes)
        return total
