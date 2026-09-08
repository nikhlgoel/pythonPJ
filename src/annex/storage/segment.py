"""Immutable on-disk segments.

A segment is the flushed, sorted image of one memtable.  The format is
deliberately simple and mmap-friendly::

    [ json header ][ vectors: float32 (n, dim) ][ json lines: one per record ]

The header records offsets, the record count and a CRC of the vector block.
Segments are written to ``<name>.tmp`` and ``os.replace``-d into place, so a
crash never leaves a half-visible segment.
"""

from __future__ import annotations

import json
import os
import struct
import zlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..errors import CorruptionError

_MAGIC = b"ASEG"
_HDR_LEN = struct.Struct("<4sI")


@dataclass(frozen=True, slots=True)
class SegmentEntry:
    id: str
    seq: int
    deleted: bool
    metadata: dict
    text: str | None
    row: int  # index into the vector block (-1 for tombstones)


class SegmentWriter:
    """Builds one segment file."""

    def __init__(self, path: str | os.PathLike[str], dim: int) -> None:
        self.path = Path(path)
        self.dim = dim
        self._vectors: list[np.ndarray] = []
        self._entries: list[dict] = []

    def add(
        self,
        rec_id: str,
        seq: int,
        vector: np.ndarray | None,
        metadata: dict,
        text: str | None,
        deleted: bool = False,
    ) -> None:
        row = -1
        if vector is not None and not deleted:
            row = len(self._vectors)
            self._vectors.append(np.ascontiguousarray(vector, dtype=np.float32))
        self._entries.append(
            {
                "id": rec_id,
                "seq": seq,
                "deleted": deleted,
                "metadata": metadata,
                "text": text,
                "row": row,
            }
        )

    def __len__(self) -> int:
        return len(self._entries)

    def finish(self) -> Path:
        """Atomically materialise the segment; returns its final path."""
        block = (
            np.stack(self._vectors)
            if self._vectors
            else np.empty((0, self.dim), dtype=np.float32)
        )
        body = b"\n".join(json.dumps(e, separators=(",", ":")).encode("utf-8") for e in self._entries)
        header = {
            "version": 1,
            "dim": self.dim,
            "records": len(self._entries),
            "vectors": int(block.shape[0]),
            "vector_crc": zlib.crc32(block.tobytes()) & 0xFFFFFFFF,
            "min_seq": min((e["seq"] for e in self._entries), default=0),
            "max_seq": max((e["seq"] for e in self._entries), default=0),
        }
        hdr = json.dumps(header, separators=(",", ":")).encode("utf-8")
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        with open(tmp, "wb") as fh:
            fh.write(_HDR_LEN.pack(_MAGIC, len(hdr)))
            fh.write(hdr)
            fh.write(block.tobytes())
            fh.write(body)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, self.path)
        return self.path


class Segment:
    """Read side of a segment file (lazily loaded)."""

    def __init__(self, path: str | os.PathLike[str]) -> None:
        self.path = Path(path)
        raw = self.path.read_bytes()
        magic, hlen = _HDR_LEN.unpack_from(raw, 0)
        if magic != _MAGIC:
            raise CorruptionError(f"{self.path}: not an Annex segment")
        off = _HDR_LEN.size
        self.header = json.loads(raw[off : off + hlen])
        off += hlen
        n, dim = self.header["vectors"], self.header["dim"]
        nbytes = n * dim * 4
        blob = raw[off : off + nbytes]
        if zlib.crc32(blob) & 0xFFFFFFFF != self.header["vector_crc"]:
            raise CorruptionError(f"{self.path}: vector block CRC mismatch")
        self.vectors = np.frombuffer(blob, dtype=np.float32).reshape(n, dim).copy()
        off += nbytes
        tail = raw[off:]
        self.entries = [
            SegmentEntry(**json.loads(line)) for line in tail.split(b"\n") if line
        ]

    @property
    def max_seq(self) -> int:
        return int(self.header["max_seq"])

    def __len__(self) -> int:
        return len(self.entries)

    def vector_of(self, entry: SegmentEntry) -> np.ndarray | None:
        return None if entry.row < 0 else self.vectors[entry.row]

    def __iter__(self):
        return iter(self.entries)
