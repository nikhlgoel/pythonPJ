"""Index construction from a :class:`CollectionConfig`."""

from __future__ import annotations

from ..config import CollectionConfig
from ..types import IndexKind
from .base import VectorIndex
from .flat import FlatIndex
from .hnsw import HNSWIndex
from .space import ExactSpace, PQSpace, VectorSpace


def build_index(config: CollectionConfig, capacity: int = 1024) -> VectorIndex:
    """Instantiate the index described by ``config``."""
    if config.index is IndexKind.HNSW_PQ:
        pq_space = PQSpace(
            config.dim,
            config.metric,
            config.pq,
            capacity=capacity,
            keep_originals=config.pq.keep_originals,
        )
        return HNSWIndex(pq_space, config.hnsw, rerank_factor=config.pq.rerank_factor)
    space: VectorSpace = ExactSpace(config.dim, config.metric, capacity=capacity)
    if config.index is IndexKind.FLAT:
        return FlatIndex(space)
    return HNSWIndex(space, config.hnsw)
