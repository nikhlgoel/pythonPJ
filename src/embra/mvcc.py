"""Multi-version concurrency control.

Embra never mutates a stored vector in place.  An update allocates a **new
internal key** and marks the previous one dead at the committing sequence
number::

    key 7  doc="a"  born=3  dead=9      <- old version, still readable at snap<9
    key 12 doc="a"  born=9  dead=inf    <- current version

A *snapshot* is just an integer sequence number.  A version is visible to a
reader at snapshot ``s`` iff ``born <= s < dead``.  Because visibility is a pure
function of two integers, it can be evaluated inside the HNSW traversal at
essentially zero cost, which is what lets long-running scans coexist with
concurrent writers without any locking on the read path.

Dead versions stay in the index until :meth:`Collection.vacuum` removes those no
longer visible to any live snapshot - the same "oldest active snapshot"
horizon rule PostgreSQL uses.
"""

from __future__ import annotations

import itertools
import threading
from dataclasses import dataclass, field

from .types import DocId, Metadata, Seq

INF: Seq = 1 << 62


@dataclass(slots=True)
class Version:
    """One immutable version of a document."""

    key: int
    doc_id: DocId
    born: Seq
    dead: Seq = INF
    metadata: Metadata = field(default_factory=dict)
    text: str | None = None

    def visible_at(self, snapshot: Seq) -> bool:
        return self.born <= snapshot < self.dead

    @property
    def is_live(self) -> bool:
        return self.dead == INF


class SnapshotRegistry:
    """Tracks open read snapshots so vacuum knows what it may reclaim."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._counter = itertools.count(1)
        self._open: dict[int, Seq] = {}

    def acquire(self, seq: Seq) -> int:
        with self._lock:
            handle = next(self._counter)
            self._open[handle] = seq
            return handle

    def release(self, handle: int) -> None:
        with self._lock:
            self._open.pop(handle, None)

    def horizon(self, current: Seq) -> Seq:
        """The oldest sequence any open reader can still see."""
        with self._lock:
            return min(self._open.values(), default=current)

    def __len__(self) -> int:
        with self._lock:
            return len(self._open)


class Snapshot:
    """Context manager pinning a consistent point-in-time view."""

    __slots__ = ("_registry", "_handle", "seq", "collection")

    def __init__(self, registry: SnapshotRegistry, seq: Seq, collection=None) -> None:
        self._registry = registry
        self.seq = seq
        self.collection = collection
        self._handle = registry.acquire(seq)

    def close(self) -> None:
        self._registry.release(self._handle)

    def __enter__(self) -> Snapshot:
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"<Snapshot seq={self.seq}>"
