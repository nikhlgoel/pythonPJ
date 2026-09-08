"""Binary framing and record serialisation.

Frame layout (little endian)::

    +--------+--------+------+------+-----------+-----------+---------+
    | magic  | ver    | type | flag | seq (u64) | len (u32) | crc(u32)|
    | 4 B    | 2 B    | 1 B  | 1 B  | 8 B       | 4 B       | 4 B     |
    +--------+--------+------+------+-----------+-----------+---------+
    | payload (len bytes)                                            |
    +----------------------------------------------------------------+

The CRC32 covers the payload *and* the header prefix, so both a truncated write
and a bit-flip in the header are caught.  ``MAGIC`` lets the reader resynchronise
and lets us reject files that are not Annex WALs at all.
"""

from __future__ import annotations

import json
import struct
import zlib
from dataclasses import dataclass
from enum import IntEnum

import numpy as np

from ..errors import CorruptionError

MAGIC = b"AWAL"
VERSION = 1
_HEADER = struct.Struct("<4sHBBQII")
HEADER_SIZE = _HEADER.size


class FrameType(IntEnum):
    PUT = 1
    DELETE = 2
    CHECKPOINT = 3
    TXN_COMMIT = 4


@dataclass(frozen=True, slots=True)
class Frame:
    type: FrameType
    seq: int
    payload: bytes
    flags: int = 0


def encode_frame(frame: Frame) -> bytes:
    """Serialise a frame, including its integrity check."""
    prefix = _HEADER.pack(
        MAGIC, VERSION, int(frame.type), frame.flags, frame.seq, len(frame.payload), 0
    )
    crc = zlib.crc32(prefix[:-4]) & 0xFFFFFFFF
    crc = zlib.crc32(frame.payload, crc) & 0xFFFFFFFF
    return prefix[:-4] + struct.pack("<I", crc) + frame.payload


def decode_frame(buf: bytes, offset: int = 0) -> tuple[Frame, int]:
    """Decode one frame starting at ``offset``.

    Returns ``(frame, next_offset)``.  Raises :class:`CorruptionError` for a
    torn or corrupted frame - the WAL reader turns that into a clean truncation.
    """
    if offset + HEADER_SIZE > len(buf):
        raise CorruptionError("truncated frame header")
    magic, version, ftype, flags, seq, length, crc = _HEADER.unpack_from(buf, offset)
    if magic != MAGIC:
        raise CorruptionError(f"bad magic {magic!r}")
    if version != VERSION:
        raise CorruptionError(f"unsupported frame version {version}")
    end = offset + HEADER_SIZE + length
    if end > len(buf):
        raise CorruptionError("truncated frame payload")
    payload = buf[offset + HEADER_SIZE : end]
    expect = zlib.crc32(buf[offset : offset + HEADER_SIZE - 4]) & 0xFFFFFFFF
    expect = zlib.crc32(payload, expect) & 0xFFFFFFFF
    if expect != crc:
        raise CorruptionError("crc mismatch")
    try:
        kind = FrameType(ftype)
    except ValueError as exc:  # pragma: no cover - defensive
        raise CorruptionError(f"unknown frame type {ftype}") from exc
    return Frame(kind, seq, payload, flags), end


# ----------------------------------------------------------------------
# record payloads
# ----------------------------------------------------------------------
_U32 = struct.Struct("<I")


def _pack_bytes(b: bytes) -> bytes:
    return _U32.pack(len(b)) + b


def _unpack_bytes(buf: bytes, off: int) -> tuple[bytes, int]:
    (n,) = _U32.unpack_from(buf, off)
    off += 4
    return buf[off : off + n], off + n


def encode_put(rec_id: str, vector: np.ndarray, metadata: dict, text: str | None) -> bytes:
    vec = np.ascontiguousarray(vector, dtype=np.float32)
    return b"".join(
        (
            _pack_bytes(rec_id.encode("utf-8")),
            _U32.pack(vec.shape[0]),
            vec.tobytes(),
            _pack_bytes(json.dumps(metadata, separators=(",", ":")).encode("utf-8")),
            _pack_bytes((text or "").encode("utf-8")),
            struct.pack("<B", 1 if text is not None else 0),
        )
    )


def decode_put(payload: bytes) -> tuple[str, np.ndarray, dict, str | None]:
    off = 0
    raw_id, off = _unpack_bytes(payload, off)
    (dim,) = _U32.unpack_from(payload, off)
    off += 4
    vec = np.frombuffer(payload, dtype=np.float32, count=dim, offset=off).copy()
    off += dim * 4
    raw_meta, off = _unpack_bytes(payload, off)
    raw_text, off = _unpack_bytes(payload, off)
    (has_text,) = struct.unpack_from("<B", payload, off)
    return (
        raw_id.decode("utf-8"),
        vec,
        json.loads(raw_meta) if raw_meta else {},
        raw_text.decode("utf-8") if has_text else None,
    )


def encode_delete(rec_id: str) -> bytes:
    return _pack_bytes(rec_id.encode("utf-8"))


def decode_delete(payload: bytes) -> str:
    raw, _ = _unpack_bytes(payload, 0)
    return raw.decode("utf-8")
