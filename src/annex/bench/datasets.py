"""Synthetic datasets.

Uniform Gaussian vectors are the *easy* case for ANN indexes; real embeddings
are strongly clustered, which is exactly where a naive neighbour selection
collapses.  ``clustered_dataset`` therefore generates a mixture of Gaussians
with controllable intrinsic dimensionality so the benchmark reports numbers that
transfer to real corpora.
"""

from __future__ import annotations

import numpy as np

from ..util.math import l2_normalize


def random_dataset(n: int, dim: int, *, seed: int = 0, normalize: bool = True) -> np.ndarray:
    rng = np.random.default_rng(seed)
    data = rng.normal(size=(n, dim)).astype(np.float32)
    return l2_normalize(data) if normalize else data


def clustered_dataset(
    n: int,
    dim: int,
    *,
    clusters: int = 64,
    spread: float = 0.15,
    seed: int = 0,
    normalize: bool = True,
) -> np.ndarray:
    """Mixture-of-Gaussians embedding-like data."""
    rng = np.random.default_rng(seed)
    centers = rng.normal(size=(clusters, dim)).astype(np.float32)
    centers = l2_normalize(centers)
    assign = rng.integers(0, clusters, size=n)
    data = centers[assign] + rng.normal(scale=spread, size=(n, dim)).astype(np.float32)
    return l2_normalize(data) if normalize else data


def query_set(data: np.ndarray, count: int, *, seed: int = 1, jitter: float = 0.05) -> np.ndarray:
    """Queries drawn near existing points - the realistic retrieval regime."""
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, data.shape[0], size=count)
    q = data[idx] + rng.normal(scale=jitter, size=(count, data.shape[1])).astype(np.float32)
    return l2_normalize(q)
