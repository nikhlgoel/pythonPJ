"""Vector index implementations.

The package is organised around two orthogonal abstractions:

``VectorSpace``
    Knows how to compute distances between stored vectors and a query
    (:mod:`embra.index.space`).  It may hold exact ``float32`` vectors or
    product-quantised codes.

``VectorIndex``
    Knows how to *avoid* computing most of those distances
    (:mod:`embra.index.flat`, :mod:`embra.index.hnsw`).

Because the graph never touches raw vectors directly, the very same HNSW
implementation runs over exact vectors or over 32x-compressed PQ codes.
"""

from .base import VectorIndex
from .factory import build_index
from .flat import FlatIndex
from .hnsw import HNSWIndex
from .pq import ProductQuantizer
from .space import ExactSpace, PQSpace, VectorSpace

__all__ = [
    "ExactSpace",
    "FlatIndex",
    "HNSWIndex",
    "PQSpace",
    "ProductQuantizer",
    "VectorIndex",
    "VectorSpace",
    "build_index",
]
