"""Product quantisation.

A ``dim``-dimensional vector is split into ``m`` contiguous sub-vectors; each
sub-space gets its own codebook of ``2**bits`` centroids learned with k-means.
A vector is then stored as ``m`` bytes instead of ``4 * dim`` bytes - a 32x
reduction at ``dim=128, m=8``.

Query time uses *asymmetric distance computation* (ADC): the query stays in full
precision, and a ``(m, K)`` lookup table is built once per query so that scoring
a candidate costs ``m`` table lookups and ``m-1`` additions instead of ``dim``
multiply-adds.

Both supported distance families decompose additively over the sub-spaces::

    ||q - x||^2 = sum_s ||q_s - x_s||^2
    <q, x>      = sum_s <q_s, x_s>

which is exactly why PQ works for L2, dot and (normalised) cosine alike.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from ..config import PQConfig
from ..errors import IndexError_
from ..types import Metric


def kmeans(
    data: np.ndarray,
    k: int,
    *,
    iters: int = 25,
    seed: int = 0,
    tol: float = 1e-6,
) -> np.ndarray:
    """Lloyd's algorithm with k-means++ seeding.

    Returns a ``(k, d)`` centroid matrix.  Empty clusters are re-seeded on the
    currently worst-represented point, which keeps every code word useful.
    """
    rng = np.random.default_rng(seed)
    n, d = data.shape
    if n == 0:
        raise IndexError_("cannot train k-means on an empty sample")
    if n <= k:
        pad = np.repeat(data[-1:], k - n, axis=0) if n < k else np.empty((0, d), data.dtype)
        return np.vstack([data, pad]).astype(np.float32)

    # --- k-means++ initialisation
    centers = np.empty((k, d), dtype=np.float32)
    centers[0] = data[rng.integers(n)]
    closest = np.sum((data - centers[0]) ** 2, axis=1)
    for i in range(1, k):
        total = float(closest.sum())
        if total <= 0:
            centers[i] = data[rng.integers(n)]
        else:
            centers[i] = data[rng.choice(n, p=closest / total)]
        closest = np.minimum(closest, np.sum((data - centers[i]) ** 2, axis=1))

    prev_inertia = np.inf
    for _ in range(iters):
        # (n, k) squared distances via the ||a||^2 - 2ab + ||b||^2 expansion
        d2 = (
            np.einsum("ij,ij->i", data, data)[:, None]
            - 2.0 * data @ centers.T
            + np.einsum("ij,ij->i", centers, centers)[None, :]
        )
        assign = np.argmin(d2, axis=1)
        inertia = float(d2[np.arange(n), assign].sum())
        counts = np.bincount(assign, minlength=k)
        new = np.zeros_like(centers)
        np.add.at(new, assign, data)
        nonempty = counts > 0
        new[nonempty] /= counts[nonempty, None]
        if np.any(~nonempty):
            worst = np.argsort(-d2[np.arange(n), assign])[: int((~nonempty).sum())]
            new[~nonempty] = data[worst]
        centers = new
        if abs(prev_inertia - inertia) <= tol * max(prev_inertia, 1.0):
            break
        prev_inertia = inertia
    return centers


@dataclass
class ProductQuantizer:
    """Trained product quantiser for a fixed dimensionality and metric."""

    dim: int
    config: PQConfig
    metric: Metric = Metric.COSINE
    codebooks: np.ndarray | None = None  # (m, K, dsub)

    def __post_init__(self) -> None:
        if self.dim % self.config.subvectors:
            raise IndexError_("dim must be divisible by pq.subvectors")
        self.dsub = self.dim // self.config.subvectors
        self._sym_tables: np.ndarray | None = None

    # ------------------------------------------------------------------
    @property
    def is_trained(self) -> bool:
        return self.codebooks is not None

    @property
    def m(self) -> int:
        return self.config.subvectors

    @property
    def code_dtype(self) -> np.dtype:
        return np.dtype(np.uint8)  # bits<=8 by config validation

    def train(self, data: np.ndarray) -> ProductQuantizer:
        """Learn one codebook per sub-space from a (sub)sample of ``data``."""
        data = np.ascontiguousarray(data, dtype=np.float32)
        if data.ndim != 2 or data.shape[1] != self.dim:
            raise IndexError_(f"training data must have shape (n, {self.dim})")
        rng = np.random.default_rng(self.config.seed)
        if data.shape[0] > self.config.train_sample:
            idx = rng.choice(data.shape[0], self.config.train_sample, replace=False)
            data = data[idx]
        k = self.config.centroids
        books = np.empty((self.m, k, self.dsub), dtype=np.float32)
        for s in range(self.m):
            sub = data[:, s * self.dsub : (s + 1) * self.dsub]
            books[s] = kmeans(sub, k, iters=self.config.train_iters, seed=self.config.seed + s)
        self.codebooks = books
        self._sym_tables = None
        return self

    # ------------------------------------------------------------------
    def encode(self, vectors: np.ndarray) -> np.ndarray:
        """Encode ``(n, dim)`` float vectors into ``(n, m)`` uint8 codes."""
        if self.codebooks is None:
            raise IndexError_("quantizer is not trained")
        vectors = np.atleast_2d(np.asarray(vectors, dtype=np.float32))
        n = vectors.shape[0]
        codes = np.empty((n, self.m), dtype=self.code_dtype)
        for s in range(self.m):
            sub = vectors[:, s * self.dsub : (s + 1) * self.dsub]
            book = self.codebooks[s]
            d2 = (
                np.einsum("ij,ij->i", sub, sub)[:, None]
                - 2.0 * sub @ book.T
                + np.einsum("ij,ij->i", book, book)[None, :]
            )
            codes[:, s] = np.argmin(d2, axis=1)
        return codes

    def decode(self, codes: np.ndarray) -> np.ndarray:
        """Reconstruct approximate float vectors from codes."""
        if self.codebooks is None:
            raise IndexError_("quantizer is not trained")
        codes = np.atleast_2d(codes)
        out = np.empty((codes.shape[0], self.dim), dtype=np.float32)
        for s in range(self.m):
            out[:, s * self.dsub : (s + 1) * self.dsub] = self.codebooks[s][codes[:, s]]
        return out

    # ------------------------------------------------------------------
    def distance_table(self, query: np.ndarray) -> tuple[np.ndarray, float]:
        """Build the ADC lookup table for one query.

        Returns ``(table, const)`` such that the distance of a code ``c`` is
        ``const + sum_s table[s, c[s]]``.
        """
        if self.codebooks is None:
            raise IndexError_("quantizer is not trained")
        q = np.asarray(query, dtype=np.float32).reshape(-1)
        table = np.empty((self.m, self.config.centroids), dtype=np.float32)
        for s in range(self.m):
            qs = q[s * self.dsub : (s + 1) * self.dsub]
            book = self.codebooks[s]
            if self.metric is Metric.L2:
                diff = book - qs
                table[s] = np.einsum("ij,ij->i", diff, diff)
            else:
                table[s] = -(book @ qs)
        const = 1.0 if self.metric is Metric.COSINE else 0.0
        return table, const

    def adc(self, table: np.ndarray, const: float, codes: np.ndarray) -> np.ndarray:
        """Score a block of codes against a prepared table."""
        codes = np.atleast_2d(codes)
        acc = np.zeros(codes.shape[0], dtype=np.float32)
        for s in range(self.m):
            acc += table[s][codes[:, s]]
        return acc + np.float32(const)

    # ------------------------------------------------------------------
    def symmetric_tables(self) -> np.ndarray:
        """``(m, K, K)`` centroid-to-centroid distances, cached lazily."""
        if self._sym_tables is None:
            if self.codebooks is None:
                raise IndexError_("quantizer is not trained")
            k = self.config.centroids
            tables = np.empty((self.m, k, k), dtype=np.float32)
            for s in range(self.m):
                book = self.codebooks[s]
                if self.metric is Metric.L2:
                    sq = np.einsum("ij,ij->i", book, book)
                    tables[s] = sq[:, None] - 2.0 * book @ book.T + sq[None, :]
                else:
                    tables[s] = -(book @ book.T)
            self._sym_tables = tables
        return self._sym_tables

    def sdc(self, a: np.ndarray, b: np.ndarray) -> float:
        """Symmetric distance between two codes (used for graph construction)."""
        tables = self.symmetric_tables()
        const = 1.0 if self.metric is Metric.COSINE else 0.0
        return float(sum(tables[s, a[s], b[s]] for s in range(self.m)) + const)

    # ------------------------------------------------------------------
    def state_dict(self) -> dict:
        return {
            "dim": self.dim,
            "metric": self.metric.value,
            "config": asdict(self.config),
            "codebooks": None if self.codebooks is None else self.codebooks.tolist(),
        }
