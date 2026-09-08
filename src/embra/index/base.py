"""Index interface.

Keys are *dense internal ids* assigned by the owning collection (0, 1, 2, ...).
Densification is what lets the whole engine use plain ``numpy`` row indexing
instead of Python dict lookups on the hot path.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable

import numpy as np

Predicate = Callable[[int], bool]


class VectorIndex(ABC):
    """Approximate or exact nearest-neighbour index over dense internal keys."""

    @abstractmethod
    def add(self, key: int, vector: np.ndarray) -> None:
        """Insert (or replace) the vector stored under ``key``."""

    @abstractmethod
    def remove(self, key: int) -> None:
        """Soft-delete ``key``; it must stop appearing in results."""

    @abstractmethod
    def search(
        self,
        query: np.ndarray,
        k: int,
        *,
        ef: int | None = None,
        allow: Predicate | None = None,
    ) -> list[tuple[int, float]]:
        """Return up to ``k`` ``(key, distance)`` pairs, best (smallest) first."""

    @property
    @abstractmethod
    def size(self) -> int:
        """Number of live vectors."""

    def __len__(self) -> int:
        return self.size

    def stats(self) -> dict:
        return {"type": type(self).__name__, "size": self.size}
