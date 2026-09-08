"""Append-only write-ahead log with crash-safe recovery."""

from __future__ import annotations

import os
from collections.abc import Iterator
from pathlib import Path

from ..errors import CorruptionError
from .codec import Frame, decode_frame, encode_frame


class WriteAheadLog:
    """A single WAL file.

    The log is opened in append mode and never rewritten in place.  On
    :meth:`replay` we stop at the first frame that fails its CRC and *truncate*
    the file there: a frame that was only partially written before a crash was,
    by definition, never acknowledged to the caller.
    """

    def __init__(self, path: str | os.PathLike[str], *, sync: bool = False) -> None:
        self.path = Path(path)
        self.sync = sync
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = open(self.path, "ab+")  # noqa: SIM115 - lifetime owned by this object
        self._bytes_written = self.path.stat().st_size

    # ------------------------------------------------------------------
    @property
    def size_bytes(self) -> int:
        return self._bytes_written

    def append(self, frame: Frame) -> int:
        """Append one frame; returns the byte offset it was written at."""
        blob = encode_frame(frame)
        offset = self._bytes_written
        self._fh.write(blob)
        self._fh.flush()
        if self.sync:
            os.fsync(self._fh.fileno())
        self._bytes_written += len(blob)
        return offset

    def flush(self, *, fsync: bool = True) -> None:
        self._fh.flush()
        if fsync:
            os.fsync(self._fh.fileno())

    def replay(self, *, truncate_corrupt: bool = True) -> Iterator[Frame]:
        """Yield every intact frame, truncating a torn tail if present."""
        data = self.path.read_bytes()
        offset = 0
        while offset < len(data):
            try:
                frame, offset = decode_frame(data, offset)
            except CorruptionError:
                if not truncate_corrupt:
                    raise
                self._truncate(offset)
                return
            yield frame

    def _truncate(self, offset: int) -> None:
        self._fh.close()
        with open(self.path, "r+b") as fh:
            fh.truncate(offset)
            fh.flush()
            os.fsync(fh.fileno())
        self._fh = open(self.path, "ab+")  # noqa: SIM115
        self._bytes_written = offset

    def rotate(self) -> None:
        """Discard the log (called after a successful memtable flush)."""
        self._fh.close()
        with open(self.path, "wb"):
            pass
        self._fh = open(self.path, "ab+")  # noqa: SIM115
        self._bytes_written = 0

    def close(self) -> None:
        if not self._fh.closed:
            self._fh.flush()
            if self.sync:
                os.fsync(self._fh.fileno())
            self._fh.close()

    def __enter__(self) -> WriteAheadLog:
        return self

    def __exit__(self, *exc) -> None:
        self.close()
