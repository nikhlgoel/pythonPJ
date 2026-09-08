"""Binding a replicated Raft log to an Annex collection.

A command is a small JSON-serialisable dict; applying it to the local collection
is deterministic, so every replica that applies the same committed prefix ends
up byte-identical.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ..db import Collection


def encode_command(op: str, **kwargs: Any) -> dict:
    """Build a replicable command (vectors become plain lists)."""
    if "vector" in kwargs and kwargs["vector"] is not None:
        kwargs["vector"] = np.asarray(kwargs["vector"], dtype=np.float32).tolist()
    return {"op": op, **kwargs}


class CollectionStateMachine:
    """Applies committed commands to a local :class:`Collection` replica."""

    def __init__(self, collection: Collection) -> None:
        self.collection = collection
        self.applied = 0

    def __call__(self, command: dict) -> None:
        self.apply(command)

    # -- snapshot / restore -------------------------------------------
    def snapshot(self) -> dict:
        """Serialise every live document - the payload of ``InstallSnapshot``."""
        docs = []
        for doc_id in sorted(self.collection._current):  # noqa: SLF001 - same package
            hit = self.collection.get(doc_id, include_vector=True)
            if hit is None or hit.vector is None:
                continue
            docs.append(
                {
                    "id": hit.id,
                    "vector": np.asarray(hit.vector, dtype=np.float32).tolist(),
                    "metadata": hit.metadata,
                    "text": hit.text,
                }
            )
        return {"documents": docs, "applied": self.applied}

    def restore(self, snapshot: dict | None) -> None:
        """Replace local state with a snapshot shipped by the leader."""
        if not snapshot:
            return
        for doc_id in list(self.collection._current):  # noqa: SLF001
            self.collection.delete(doc_id)
        for doc in snapshot.get("documents", []):
            self.collection.upsert(
                doc["id"],
                np.asarray(doc["vector"], dtype=np.float32),
                doc.get("metadata") or {},
                doc.get("text"),
            )
        self.collection.vacuum()
        self.applied = int(snapshot.get("applied", self.applied))

    def apply(self, command: dict) -> None:
        op = command.get("op")
        if op == "upsert":
            self.collection.upsert(
                command["id"],
                np.asarray(command["vector"], dtype=np.float32),
                command.get("metadata") or {},
                command.get("text"),
            )
        elif op == "delete":
            self.collection.delete(command["id"])
        elif op == "noop":
            pass
        else:  # pragma: no cover - defensive
            raise ValueError(f"unknown replicated op {op!r}")
        self.applied += 1
