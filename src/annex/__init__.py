"""Annex - an embedded, transactional, vector-native search engine.

Annex is a from-scratch implementation of the machinery behind a modern vector
database: a hierarchical navigable small-world graph, product quantisation, a
log-structured storage engine with write-ahead logging and crash recovery,
snapshot isolation via MVCC, a BM25 inverted index for hybrid retrieval, and a
cost-based query planner that chooses between them.

Quick start
-----------
>>> import numpy as np, annex
>>> db = annex.Database("/tmp/annex-doctest")
>>> books = db.create_collection("books", dim=4)
>>> _ = books.upsert("b1", [0.1, 0.2, 0.3, 0.4], {"year": 2021}, text="vector search")
>>> res = books.search([0.1, 0.2, 0.3, 0.4], k=1)
>>> res.hits[0].id
'b1'
"""

from .config import CollectionConfig, HNSWConfig, PQConfig, StorageConfig
from .db import Collection, Database, Transaction
from .errors import (
    AnnexError,
    ConfigError,
    ConflictError,
    CorruptionError,
    NotFoundError,
    QueryError,
    SchemaError,
    TransactionError,
)
from .types import Hit, IndexKind, Metric, Record, SearchResult

__version__ = "0.1.0"

__all__ = [
    "Collection",
    "CollectionConfig",
    "ConfigError",
    "ConflictError",
    "CorruptionError",
    "Database",
    "AnnexError",
    "HNSWConfig",
    "Hit",
    "IndexKind",
    "Metric",
    "NotFoundError",
    "PQConfig",
    "QueryError",
    "Record",
    "SchemaError",
    "SearchResult",
    "StorageConfig",
    "Transaction",
    "TransactionError",
    "__version__",
]
