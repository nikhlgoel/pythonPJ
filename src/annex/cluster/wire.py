"""JSON wire codec for Raft RPCs.

The consensus core is a pure state machine that speaks in dataclasses; this
module is the only place that knows how those become bytes.  Keeping the codec
separate is what makes the transport swappable - the in-process transport hands
dataclasses straight over, while the HTTP transport round-trips them through
this codec, and neither can change protocol behaviour.
"""

from __future__ import annotations

from typing import Any

from ..errors import ClusterError
from .raft import (
    AppendEntries,
    AppendEntriesReply,
    InstallSnapshot,
    InstallSnapshotReply,
    LogEntry,
    Message,
    RequestVote,
    RequestVoteReply,
)

_TYPES = {
    cls.__name__: cls
    for cls in (
        RequestVote,
        RequestVoteReply,
        AppendEntries,
        AppendEntriesReply,
        InstallSnapshot,
        InstallSnapshotReply,
    )
}


def encode_entry(entry: LogEntry) -> dict[str, Any]:
    return {"term": entry.term, "index": entry.index, "command": entry.command}


def decode_entry(raw: dict[str, Any]) -> LogEntry:
    return LogEntry(int(raw["term"]), int(raw["index"]), raw["command"])


def encode_message(msg: Message) -> dict[str, Any]:
    """Serialise an envelope into a JSON-compatible dict."""
    payload = msg.payload
    kind = type(payload).__name__
    if kind not in _TYPES:
        raise ClusterError(f"cannot encode payload of type {kind}")
    body = dict(payload.__dict__) if hasattr(payload, "__dict__") else {
        f: getattr(payload, f) for f in payload.__slots__
    }
    if isinstance(payload, AppendEntries):
        body["entries"] = [encode_entry(e) for e in payload.entries]
    if isinstance(payload, InstallSnapshot):
        body["members"] = list(payload.members)
    return {"src": msg.src, "dst": msg.dst, "kind": kind, "payload": body}


def decode_message(raw: dict[str, Any]) -> Message:
    """Rebuild an envelope produced by :func:`encode_message`."""
    kind = raw.get("kind")
    cls = _TYPES.get(kind or "")
    if cls is None:
        raise ClusterError(f"unknown payload kind {kind!r}")
    body = dict(raw.get("payload") or {})
    if cls is AppendEntries:
        body["entries"] = tuple(decode_entry(e) for e in body.get("entries", []))
    if cls is InstallSnapshot:
        body["members"] = tuple(body.get("members", []))
    try:
        payload = cls(**body)
    except TypeError as exc:
        raise ClusterError(f"malformed {kind} payload: {exc}") from exc
    return Message(str(raw["src"]), str(raw["dst"]), payload)
