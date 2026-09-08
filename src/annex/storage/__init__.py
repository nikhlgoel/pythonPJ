"""Durable storage: write-ahead log, immutable segments, LSM-style store.

Durability model
----------------
1. Every mutation is appended to the **WAL** first (``PUT`` / ``DELETE``),
   framed with a CRC32 so a torn tail from a crash mid-``write`` is detected and
   truncated instead of silently corrupting the database.
2. Mutations accumulate in an in-memory **memtable**.
3. When the memtable is full it is flushed into an immutable **segment** file
   (written to a temp path and ``rename``-d, which is atomic on POSIX), and the
   WAL is rotated.
4. Segments are periodically **compacted**: newer versions and tombstones win,
   dead space is reclaimed.

Recovery replays segments oldest-first, then the WAL tail, reconstructing the
exact pre-crash state up to the last durable frame.
"""

from .codec import Frame, FrameType, decode_frame, encode_frame
from .lsm import LSMStore, StoredRecord
from .segment import Segment, SegmentWriter
from .wal import WriteAheadLog

__all__ = [
    "Frame",
    "FrameType",
    "LSMStore",
    "Segment",
    "SegmentWriter",
    "StoredRecord",
    "WriteAheadLog",
    "decode_frame",
    "encode_frame",
]
