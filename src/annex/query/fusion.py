"""Rank fusion for hybrid (dense + lexical) retrieval.

Dense cosine similarities and BM25 scores live on incomparable scales, so Annex
fuses *ranks*, not scores.  Reciprocal Rank Fusion (Cormack et al., 2009) is the
default because it needs no tuning, no score calibration, and is robust when one
retriever returns nothing at all::

    RRF(d) = sum_r w_r / (K + rank_r(d))

``weighted_fusion`` is offered for the case where you *do* have calibrated
scores and want linear interpolation after min-max normalisation.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence


def reciprocal_rank_fusion(
    rankings: Sequence[Iterable[tuple]],
    *,
    weights: Sequence[float] | None = None,
    k_constant: float = 60.0,
    top_k: int | None = None,
) -> list[tuple[object, float]]:
    """Fuse several ranked lists of ``(key, score)`` into one."""
    weights = list(weights or [1.0] * len(rankings))
    if len(weights) != len(rankings):
        raise ValueError("weights and rankings must have the same length")
    fused: dict[object, float] = {}
    for w, ranking in zip(weights, rankings, strict=True):
        for rank, item in enumerate(ranking):
            key = item[0] if isinstance(item, tuple) else item
            fused[key] = fused.get(key, 0.0) + w / (k_constant + rank + 1.0)
    out = sorted(fused.items(), key=lambda kv: -kv[1])
    return out[:top_k] if top_k else out


def _minmax(values: list[float]) -> list[float]:
    if not values:
        return []
    lo, hi = min(values), max(values)
    if hi - lo < 1e-12:
        return [1.0] * len(values)
    return [(v - lo) / (hi - lo) for v in values]


def weighted_fusion(
    rankings: Sequence[Sequence[tuple[object, float]]],
    *,
    weights: Sequence[float] | None = None,
    top_k: int | None = None,
) -> list[tuple[object, float]]:
    """Min-max normalise each list, then take a weighted sum."""
    weights = list(weights or [1.0] * len(rankings))
    fused: dict[object, float] = {}
    for w, ranking in zip(weights, rankings, strict=True):
        norm = _minmax([s for _, s in ranking])
        for (key, _), value in zip(ranking, norm, strict=True):
            fused[key] = fused.get(key, 0.0) + w * value
    out = sorted(fused.items(), key=lambda kv: -kv[1])
    return out[:top_k] if top_k else out
