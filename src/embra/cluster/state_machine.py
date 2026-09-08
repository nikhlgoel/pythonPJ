"""Binding a replicated Raft log to an Embra collection.

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
