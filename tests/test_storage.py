import struct

import numpy as np
import pytest

from annex.config import StorageConfig
from annex.errors import CorruptionError
from annex.storage import LSMStore, WriteAheadLog
from annex.storage.codec import (
    Frame,
    FrameType,
    decode_frame,
    decode_put,
    encode_frame,
    encode_put,
)
from annex.storage.segment import Segment, SegmentWriter


def test_frame_roundtrip():
    frame = Frame(FrameType.PUT, 7, b"payload")
    decoded, offset = decode_frame(encode_frame(frame))
    assert decoded == frame
    assert offset == len(encode_frame(frame))


def test_frame_detects_bitflip():
    blob = bytearray(encode_frame(Frame(FrameType.PUT, 1, b"abcdef")))
    blob[-1] ^= 0xFF
    with pytest.raises(CorruptionError):
        decode_frame(bytes(blob))


def test_frame_detects_truncation():
    blob = encode_frame(Frame(FrameType.PUT, 1, b"abcdef"))
    with pytest.raises(CorruptionError):
        decode_frame(blob[:-2])


def test_put_payload_roundtrip():
    vec = np.arange(4, dtype="float32")
    payload = encode_put("id-1", vec, {"a": 1}, "hello")
    rid, got, meta, text = decode_put(payload)
    assert (rid, meta, text) == ("id-1", {"a": 1}, "hello")
    assert np.array_equal(got, vec)


def test_put_payload_preserves_none_text():
    _, _, _, text = decode_put(encode_put("x", np.zeros(2, "float32"), {}, None))
    assert text is None


def test_wal_replays_appended_frames(tmp_path):
    wal = WriteAheadLog(tmp_path / "w.log")
    for i in range(5):
        wal.append(Frame(FrameType.PUT, i, f"p{i}".encode()))
    wal.close()
    frames = list(WriteAheadLog(tmp_path / "w.log").replay())
    assert [f.seq for f in frames] == list(range(5))


def test_wal_truncates_torn_tail(tmp_path):
    path = tmp_path / "w.log"
    wal = WriteAheadLog(path)
    wal.append(Frame(FrameType.PUT, 1, b"good"))
    wal.append(Frame(FrameType.PUT, 2, b"also-good"))
    wal.close()
    with open(path, "ab") as fh:  # simulate a crash mid-write
        fh.write(b"\x00\x01\x02garbage")
    reopened = WriteAheadLog(path)
    frames = list(reopened.replay())
    assert [f.seq for f in frames] == [1, 2]
    assert list(WriteAheadLog(path).replay())  # tail was removed, file still valid


def test_segment_roundtrip(tmp_path):
    writer = SegmentWriter(tmp_path / "s.aseg", 3)
    writer.add("a", 1, np.ones(3, "float32"), {"x": 1}, "text a")
    writer.add("b", 2, None, {}, None, deleted=True)
    writer.finish()
    seg = Segment(tmp_path / "s.aseg")
    assert len(seg) == 2
    assert seg.max_seq == 2
    entries = {e.id: e for e in seg}
    assert np.array_equal(seg.vector_of(entries["a"]), np.ones(3, "float32"))
    assert seg.vector_of(entries["b"]) is None


def test_segment_detects_corrupt_vector_block(tmp_path):
    path = tmp_path / "s.aseg"
    writer = SegmentWriter(path, 2)
    writer.add("a", 1, np.ones(2, "float32"), {}, None)
    writer.finish()
    raw = bytearray(path.read_bytes())
    _, hlen = struct.unpack_from("<4sI", raw, 0)
    raw[8 + hlen] ^= 0xFF  # flip a byte inside the vector block only
    path.write_bytes(bytes(raw))
    with pytest.raises(CorruptionError):
        Segment(path)


def test_store_put_get_delete(tmp_path):
    store = LSMStore(tmp_path, 3)
    store.put("a", np.ones(3, "float32"), {"k": 1}, "hello")
    assert store.get("a").metadata == {"k": 1}
    assert len(store) == 1
    assert store.delete("a") is not None
    assert store.get("a") is None
    assert store.delete("a") is None
    store.close()


def test_store_recovers_from_wal(tmp_path):
    store = LSMStore(tmp_path, 2)
    for i in range(10):
        store.put(f"d{i}", np.array([i, i], "float32"), {"i": i})
    store.delete("d3")
    store.close()

    reopened = LSMStore(tmp_path, 2)
    assert len(reopened) == 9
    assert reopened.get("d3") is None
    assert reopened.get("d7").metadata == {"i": 7}
    assert reopened.seq == 11
    reopened.close()


def test_store_flush_creates_segment_and_rotates_wal(tmp_path):
    store = LSMStore(tmp_path, 2, StorageConfig(memtable_max_records=1000))
    for i in range(20):
        store.put(f"d{i}", np.zeros(2, "float32"))
    name = store.flush()
    assert name is not None
    assert store.wal.size_bytes == 0
    assert store.flush() is None
    store.close()

    reopened = LSMStore(tmp_path, 2)
    assert len(reopened) == 20
    reopened.close()


def test_automatic_flush_on_full_memtable(tmp_path):
    store = LSMStore(tmp_path, 2, StorageConfig(memtable_max_records=5, compaction_trigger=2))
    for i in range(20):
        store.put(f"d{i}", np.zeros(2, "float32"))
    assert store.stats()["segments"] >= 1
    store.close()
    assert len(LSMStore(tmp_path, 2)) == 20


def test_compaction_merges_and_drops_tombstones(tmp_path):
    store = LSMStore(tmp_path, 2, StorageConfig(memtable_max_records=1000, compaction_trigger=99))
    for i in range(10):
        store.put(f"d{i}", np.zeros(2, "float32"))
    store.flush()
    store.delete("d0")
    store.put("d1", np.ones(2, "float32"), {"v": 2})
    store.flush()
    store.compact()
    assert store.stats()["segments"] == 1
    store.close()

    reopened = LSMStore(tmp_path, 2)
    assert reopened.get("d0") is None
    assert reopened.get("d1").metadata == {"v": 2}
    assert len(reopened) == 9
    reopened.close()


def test_store_rejects_dimension_mismatch(tmp_path):
    LSMStore(tmp_path, 4).close()
    with pytest.raises(CorruptionError):
        LSMStore(tmp_path, 8)
