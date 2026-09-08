"""Vectorised distance kernels.

All index implementations funnel through these helpers so that metric semantics
(especially the *lower-is-better distance* vs *higher-is-better score*
convention) are defined in exactly one place.

Convention
----------
* ``distance_*`` returns a value where **smaller means more similar**.
* ``score_from_distance`` converts that into the user-facing score where
  **larger means more similar**, and for cosine lands in ``[-1, 1]``.
"""

from __future__ import annotations

import numpy as np

from ..errors import SchemaError
from ..types import Metric

_EPS = 1e-12


def l2_normalize(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Return ``x`` scaled to unit L2 norm along ``axis`` (zero-safe)."""
    norm = np.linalg.norm(x, axis=axis, keepdims=True)
    return x / np.maximum(norm, _EPS)


def validate_vector(vec, dim: int, dtype: str = "float32") -> np.ndarray:
    """Coerce ``vec`` into a contiguous 1-D array of ``dim`` elements."""
    arr = np.asarray(vec, dtype=dtype)
    if arr.ndim != 1:
        arr = arr.reshape(-1)
    if arr.shape[0] != dim:
        raise SchemaError(f"expected vector of dim {dim}, got {arr.shape[0]}")
    if not np.all(np.isfinite(arr)):
        raise SchemaError("vector contains NaN or Inf")
    return np.ascontiguousarray(arr)


def coerce_vectors(vecs, dim: int, dtype: str = "float32") -> np.ndarray:
    """Coerce a batch into a contiguous ``(n, dim)`` matrix."""
    arr = np.asarray(vecs, dtype=dtype)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.ndim != 2 or arr.shape[1] != dim:
        raise SchemaError(f"expected matrix of shape (n, {dim}), got {arr.shape}")
    if not np.all(np.isfinite(arr)):
        raise SchemaError("vector batch contains NaN or Inf")
    return np.ascontiguousarray(arr)


def distance_matrix(query: np.ndarray, data: np.ndarray, metric: Metric) -> np.ndarray:
    """Distances from one query vector to every row of ``data``.

    For cosine and dot metrics vectors are assumed pre-normalised where the
    collection requires it; the engine enforces that at ingest time.
    """
    if data.size == 0:
        return np.empty(0, dtype=np.float32)
    if metric is Metric.L2:
        diff = data - query
        return np.einsum("ij,ij->i", diff, diff, dtype=np.float32)
    inner = data @ query
    if metric is Metric.COSINE:
        return (1.0 - inner).astype(np.float32, copy=False)
    return (-inner).astype(np.float32, copy=False)  # DOT


def distance_pair(a: np.ndarray, b: np.ndarray, metric: Metric) -> float:
    """Distance between two single vectors."""
    if metric is Metric.L2:
        d = a - b
        return float(d @ d)
    inner = float(a @ b)
    return 1.0 - inner if metric is Metric.COSINE else -inner


def score_from_distance(dist: float | np.ndarray, metric: Metric):
    """Convert an engine-internal distance into a user-facing similarity score."""
    if metric is Metric.L2:
        return -dist  # keep it monotone; callers may sqrt for display
    if metric is Metric.COSINE:
        return 1.0 - dist
    return -dist  # DOT: distance was -inner


def prepare_for_metric(mat: np.ndarray, metric: Metric) -> np.ndarray:
    """Apply the ingest-time transform a metric requires."""
    return l2_normalize(mat) if metric.normalizes else mat
