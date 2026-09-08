"""Lightweight timing helpers used by the query planner and benchmarks."""

from __future__ import annotations

import time
from types import TracebackType


class Stopwatch:
    """Context manager measuring wall-clock milliseconds.

    >>> with Stopwatch() as sw:
    ...     pass
    >>> sw.ms >= 0
    True
    """

    __slots__ = ("_start", "ms")

    def __init__(self) -> None:
        self._start = 0.0
        self.ms = 0.0

    def __enter__(self) -> Stopwatch:
        self._start = time.perf_counter()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.ms = (time.perf_counter() - self._start) * 1000.0
