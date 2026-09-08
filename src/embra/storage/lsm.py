"""LSM-style store: memtable + WAL + immutable segments + compaction."""

from __future__ import annotations

import json
import os
import time
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..config import StorageConfig
from ..errors import CorruptionError
from ..types import Metadata, Seq
from .codec import Frame, FrameType, decode_delete, decode_put, encode_delete, encode_put
from .segment import Segment, SegmentWriter
from .wal import WriteAheadLog

MANIFEST = "MANIFEST.json"


@dataclass(slots=True)
class StoredRecord:
    """A record as the storage layer sees it, including MVCC bookkeeping."""

    id: str
    vector: np.ndarray | None
    metadata: Metadata = field(default_factory=dict)
    text: str | None = None
    seq: Seq = 0
    deleted: bool = False


class LSMStore:
    """Durable key/vector store with log-structured writes.

    The merged view of all live records is kept in memory (``self._table``)
    because the vector index needs the vectors resident anyway; segments provide
    durability, bounded WAL size and fast restart, exactly as in an LSM tree.
    """

    def __init__(
        self,
        directory: str | os.PathLike[str],
        dim: int,
        config: StorageConfig | None = None,
    ) -> None:
        self.dir = Path(directory)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.dim = dim
        self.config = config or StorageConfig()

        self._table: dict[str, StoredRecord] = {}
        self._memtable: dict[str, StoredRecord] = {}
        self._segments: list[str] = []
        self._seq: Seq = 0
        self._mutations_since_checkpoint = 0

        self.wal = WriteAheadLog(self.dir / "wal.log", sync=self.config.wal_sync)
        self._recover()

    # ------------------------------------------------------------------
    # recovery
    # ------------------------------------------------------------------
    @property
    def manifest_path(self) -> Path:
        return self.dir / MANIFEST

    def _read_manifest(self) -> dict:
        if not self.manifest_path.exists():
            return {"segments": [], "seq": 0, "dim": self.dim}
        try:
            return json.loads(self.manifest_path.read_text())
        except json.JSONDecodeError as exc:
            raise CorruptionError(f"unreadable manifest at {self.manifest_path}") from exc

    def _write_manifest(self) -> None:
        payload = {
            "segments": self._segments,
            "seq": self._seq,
            "dim": self.dim,
            "updated_at": time.time(),
        }
        tmp = self.manifest_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload, indent=2))
        os.replace(tmp, self.manifest_path)

    def _recover(self) -> None:
        manifest = self._read_manifest()
        if manifest.get("dim", self.dim) != self.dim:
            raise CorruptionError(
                f"store dim {manifest['dim']} != requested dim {self.dim}"
            )
        self._segments = list(manifest.get("segments", []))
        self._seq = int(manifest.get("seq", 0))

        for name in self._segments:  # oldest first: later segments overwrite
            seg = Segment(self.dir / name)
            for entry in seg:
                self._apply_entry(
                    StoredRecord(
                        id=entry.id,
                        vector=seg.vector_of(entry),
                        metadata=entry.metadata,
                        text=entry.text,
                        seq=entry.seq,
                        deleted=entry.deleted,
                    )
                )

        for frame in self.wal.replay():
            if frame.type is FrameType.PUT:
                rid, vec, meta, text = decode_put(frame.payload)
                rec = StoredRecord(rid, vec, meta, text, frame.seq, False)
            elif frame.type is FrameType.DELETE:
                rid = decode_delete(frame.payload)
                rec = StoredRecord(rid, None, {}, None, frame.seq, True)
            else:
                self._seq = max(self._seq, frame.seq)
                continue
            self._apply_entry(rec)
            self._memtable[rec.id] = rec
            self._seq = max(self._seq, frame.seq)

    def _apply_entry(self, rec: StoredRecord) -> None:
        current = self._table.get(rec.id)
        if current is None or rec.seq >= current.seq:
            self._table[rec.id] = rec

    # ------------------------------------------------------------------
    # writes
    # ------------------------------------------------------------------
    @property
    def seq(self) -> Seq:
        return self._seq

    def next_seq(self) -> Seq:
        self._seq += 1
        return self._seq

    def put(
        self,
        rec_id: str,
        vector: np.ndarray,
        metadata: Metadata | None = None,
        text: str | None = None,
        *,
        seq: Seq | None = None,
    ) -> StoredRecord:
        seq = seq if seq is not None else self.next_seq()
        self._seq = max(self._seq, seq)
        metadata = metadata or {}
        self.wal.append(Frame(FrameType.PUT, seq, encode_put(rec_id, vector, metadata, text)))
        rec = StoredRecord(rec_id, np.asarray(vector, dtype=np.float32), metadata, text, seq, False)
        self._apply_entry(rec)
        self._memtable[rec_id] = rec
        self._after_mutation()
        return rec

    def delete(self, rec_id: str, *, seq: Seq | None = None) -> StoredRecord | None:
        existing = self._table.get(rec_id)
        if existing is None or existing.deleted:
            return None
        seq = seq if seq is not None else self.next_seq()
        self._seq = max(self._seq, seq)
        self.wal.append(Frame(FrameType.DELETE, seq, encode_delete(rec_id)))
        rec = StoredRecord(rec_id, None, {}, None, seq, True)
        self._apply_entry(rec)
        self._memtable[rec_id] = rec
        self._after_mutation()
        return rec

    def _after_mutation(self) -> None:
        self._mutations_since_checkpoint += 1
        if len(self._memtable) >= self.config.memtable_max_records:
            self.flush()

    # ------------------------------------------------------------------
    # reads
    # ------------------------------------------------------------------
    def get(self, rec_id: str) -> StoredRecord | None:
        rec = self._table.get(rec_id)
        return None if rec is None or rec.deleted else rec

    def __contains__(self, rec_id: object) -> bool:
        return isinstance(rec_id, str) and self.get(rec_id) is not None

    def __len__(self) -> int:
        return sum(1 for r in self._table.values() if not r.deleted)

    def iter_live(self) -> Iterator[StoredRecord]:
        for rec in self._table.values():
            if not rec.deleted:
                yield rec

    # ------------------------------------------------------------------
    # flush / compaction
    # ------------------------------------------------------------------
    def flush(self) -> str | None:
        """Persist the memtable as a new immutable segment, then rotate the WAL."""
        if not self._memtable:
            return None
        name = f"seg-{len(self._segments):06d}-{int(time.time() * 1000)}.eseg"
        writer = SegmentWriter(self.dir / name, self.dim)
        for rec in sorted(self._memtable.values(), key=lambda r: r.id):
            writer.add(rec.id, rec.seq, rec.vector, rec.metadata, rec.text, rec.deleted)
        writer.finish()
        self._segments.append(name)
        self._memtable.clear()
        self._write_manifest()
        self.wal.rotate()
        if len(self._segments) >= self.config.compaction_trigger:
            self.compact()
        return name

    def compact(self) -> str | None:
        """Merge every segment into one, dropping shadowed rows and tombstones.

        Correctness rule: a tombstone may only be dropped by a *full* compaction
        (one covering every segment), which is what this method performs.
        """
        if len(self._segments) < 2:
            return None
        merged: dict[str, StoredRecord] = {}
        for name in self._segments:
            seg = Segment(self.dir / name)
            for entry in seg:
                prev = merged.get(entry.id)
                if prev is None or entry.seq >= prev.seq:
                    merged[entry.id] = StoredRecord(
                        entry.id,
                        seg.vector_of(entry),
                        entry.metadata,
                        entry.text,
                        entry.seq,
                        entry.deleted,
                    )
        name = f"seg-{len(self._segments):06d}-compact-{int(time.time() * 1000)}.eseg"
        writer = SegmentWriter(self.dir / name, self.dim)
        for rec in sorted(merged.values(), key=lambda r: r.id):
            if rec.deleted:
                continue  # full compaction: tombstone has no older row to hide
            writer.add(rec.id, rec.seq, rec.vector, rec.metadata, rec.text, False)
        writer.finish()
        old = self._segments
        self._segments = [name]
        self._write_manifest()
        for stale in old:
            (self.dir / stale).unlink(missing_ok=True)
        return name

    def checkpoint(self) -> None:
        self.wal.append(Frame(FrameType.CHECKPOINT, self._seq, b""))
        self.wal.flush()
        self._write_manifest()
        self._mutations_since_checkpoint = 0

    def stats(self) -> dict:
        return {
            "records": len(self),
            "tombstones": sum(1 for r in self._table.values() if r.deleted),
            "memtable": len(self._memtable),
            "segments": len(self._segments),
            "seq": self._seq,
            "wal_bytes": self.wal.size_bytes,
        }

    def close(self) -> None:
        self._write_manifest()
        self.wal.close()

    def __enter__(self) -> LSMStore:
        return self

    def __exit__(self, *exc) -> None:
        self.close()
