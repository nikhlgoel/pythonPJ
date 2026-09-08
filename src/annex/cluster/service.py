"""A running replica: a Raft node, a transport and a state machine.

:class:`RaftService` is the thin, thread-safe shell around the pure state
machine.  It owns the only lock in the cluster layer, pumps outbound envelopes
through the transport, and feeds replies back in - so the node itself keeps
knowing nothing about time, sockets or concurrency.

``tick`` is explicit rather than hidden behind a thread by default: a caller
that wants a background driver uses :meth:`start`, and a caller that wants a
deterministic test drives :meth:`tick` itself.
"""

from __future__ import annotations

import threading
import time
from typing import Any

from ..errors import ClusterError
from .raft import Message, RaftNode, Role
from .transport import Transport


class RaftService:
    """Drives one :class:`RaftNode` over a :class:`Transport`."""

    def __init__(
        self,
        node: RaftNode,
        transport: Transport,
        *,
        max_pump_rounds: int = 8,
    ) -> None:
        self.node = node
        self.transport = transport
        self.max_pump_rounds = max_pump_rounds
        self._lock = threading.RLock()
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._clock_ms = 0

    # ------------------------------------------------------------------
    @property
    def id(self) -> str:
        return self.node.id

    @property
    def is_leader(self) -> bool:
        return self.node.role is Role.LEADER

    def state(self) -> dict:
        with self._lock:
            return self.node.state()

    # ------------------------------------------------------------------
    def receive(self, message: Message) -> list[Message]:
        """Handle one inbound envelope; returns replies for the caller to route."""
        with self._lock:
            return self.node.handle(message, self._clock_ms)

    def tick(self, now_ms: int | None = None) -> None:
        """Advance logical time and flush whatever the node wants to send."""
        with self._lock:
            self._clock_ms = now_ms if now_ms is not None else int(time.monotonic() * 1000)
            outbound = self.node.tick(self._clock_ms)
        self._pump(outbound)

    def _pump(self, outbound: list[Message]) -> None:
        """Deliver messages and feed replies back, bounded to avoid ping-pong."""
        rounds = 0
        while outbound and rounds < self.max_pump_rounds:
            replies = self.transport.send(outbound)
            outbound = []
            for reply in replies:
                with self._lock:
                    outbound.extend(self.node.handle(reply, self._clock_ms))
            rounds += 1

    # ------------------------------------------------------------------
    def propose(self, command: Any) -> int:
        """Replicate a command.  Raises if this replica is not the leader."""
        with self._lock:
            index = self.node.propose(command)
            if index is None:
                raise ClusterError(
                    f"{self.id} is not the leader"
                    + (f"; try {self.node.leader_id}" if self.node.leader_id else "")
                )
            outbound = self.node._broadcast_append()  # noqa: SLF001 - same package
        self._pump(outbound)
        return index

    def add_server(self, node_id: str) -> int | None:
        with self._lock:
            index = self.node.add_server(node_id)
            outbound = self.node._broadcast_append() if index else []  # noqa: SLF001
        self._pump(outbound)
        return index

    def remove_server(self, node_id: str) -> int | None:
        with self._lock:
            index = self.node.remove_server(node_id)
            outbound = self.node._broadcast_append() if index else []  # noqa: SLF001
        self._pump(outbound)
        return index

    def compact(self, up_to: int | None = None) -> int:
        """Snapshot the state machine and discard the applied log prefix."""
        with self._lock:
            return self.node.compact(up_to)

    # ------------------------------------------------------------------
    def start(self, interval_ms: int = 20) -> None:  # pragma: no cover - thread driver
        """Run the tick loop on a background thread."""
        if self._thread is not None:
            return
        self._stop.clear()

        def loop() -> None:
            while not self._stop.is_set():
                self.tick()
                time.sleep(interval_ms / 1000.0)

        self._thread = threading.Thread(target=loop, name=f"raft-{self.id}", daemon=True)
        self._thread.start()

    def stop(self) -> None:  # pragma: no cover - thread driver
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        self.transport.close()
