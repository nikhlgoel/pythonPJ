"""Core value types shared across the engine.

The engine is deliberately built around a handful of small, immutable value
objects.  Everything on the hot path (vectors, distance matrices) is a raw
``numpy`` array; everything on the control path is a frozen dataclass.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np

VectorLike = Any  # np.ndarray | Sequence[float]
Metadata = dict[str, Any]

#: Sequence numbers are the engine's logical clock. Every mutation bumps it.
Seq = int
DocId = str


class Metric(str, Enum):
    """Supported vector similarity metrics."""

    COSINE = "cosine"
    L2 = "l2"
    DOT = "dot"

    @property
    def normalizes(self) -> bool:
        """Whether vectors are L2-normalised at ingest for this metric."""
        return self is Metric.COSINE


class IndexKind(str, Enum):
    FLAT = "flat"
    HNSW = "hnsw"
    HNSW_PQ = "hnsw_pq"


@dataclass(frozen=True, slots=True)
class Record:
    """A stored document: id, vector, metadata, optional raw text."""

    id: DocId
    vector: np.ndarray
    metadata: Metadata = field(default_factory=dict)
    text: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.id, str) or not self.id:
            raise ValueError("record id must be a non-empty string")


@dataclass(frozen=True, slots=True)
class Hit:
    """A single search result."""

    id: DocId
    score: float
    metadata: Metadata = field(default_factory=dict)
    text: str | None = None
    vector: np.ndarray | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "score": float(self.score),
            "metadata": self.metadata,
            "text": self.text,
        }


@dataclass(frozen=True, slots=True)
class SearchResult:
    """Result set plus the observability payload the planner produced."""

    hits: list[Hit]
    took_ms: float = 0.0
    candidates_scanned: int = 0
    plan: dict[str, Any] = field(default_factory=dict)

    def __iter__(self):  # pragma: no cover - trivial
        return iter(self.hits)

    def __len__(self) -> int:
        return len(self.hits)

    def __getitem__(self, i: int) -> Hit:
        return self.hits[i]

    def ids(self) -> list[DocId]:
        return [h.id for h in self.hits]
