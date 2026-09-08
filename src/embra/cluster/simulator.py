"""Deterministic cluster simulator.

Drives a set of :class:`~embra.cluster.raft.RaftNode` instances through logical
time, with controllable message loss, latency and network partitions.  This is
how the replication tests exercise leader election, log convergence and
split-brain avoidance without sleeping or spawning threads.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from .raft import Message, NodeId, RaftNode, Role


@dataclass(order=True)
class _Scheduled:
    at: int
    seq: int
    message: Message


class ClusterSimulator:
    """A virtual network plus a virtual clock."""

    def __init__(
        self,
        nodes: list[RaftNode],
        *,
        latency: tuple[int, int] = (5, 20),
        drop_rate: float = 0.0,
        seed: int = 1,
    ) -> None:
        self.nodes: dict[NodeId, RaftNode] = {n.id: n for n in nodes}
        self.time = 0
        self.latency = latency
        self.drop_rate = drop_rate
        self._rng = random.Random(seed)
        self._queue: list[_Scheduled] = []
        self._seq = 0
        self._partitions: list[set[NodeId]] = []
        self.delivered = 0
        self.dropped = 0

    # ------------------------------------------------------------------
    def partition(self, *groups: set[NodeId]) -> None:
        """Split the network; nodes can only talk within their own group."""
        self._partitions = [set(g) for g in groups]

    def heal(self) -> None:
        self._partitions = []

    def _reachable(self, src: NodeId, dst: NodeId) -> bool:
        if not self._partitions:
            return True
        return any(src in g and dst in g for g in self._partitions)

    # ------------------------------------------------------------------
    def _enqueue(self, messages: list[Message]) -> None:
        for msg in messages:
            if not self._reachable(msg.src, msg.dst):
                self.dropped += 1
                continue
            if self._rng.random() < self.drop_rate:
                self.dropped += 1
                continue
            self._seq += 1
            delay = self._rng.randint(*self.latency)
            self._queue.append(_Scheduled(self.time + delay, self._seq, msg))
        self._queue.sort()

    def step(self, dt: int = 1) -> None:
        """Advance the clock by ``dt`` ms, ticking nodes and delivering mail."""
        self.time += dt
        for node in self.nodes.values():
            self._enqueue(node.tick(self.time))
        ready = [s for s in self._queue if s.at <= self.time]
        self._queue = [s for s in self._queue if s.at > self.time]
        for scheduled in ready:
            node = self.nodes.get(scheduled.message.dst)
            if node is None:
                continue
            self.delivered += 1
            self._enqueue(node.handle(scheduled.message, self.time))

    def run(self, ms: int, dt: int = 1) -> None:
        for _ in range(0, ms, dt):
            self.step(dt)

    # ------------------------------------------------------------------
    def leader(self) -> RaftNode | None:
        leaders = [n for n in self.nodes.values() if n.role is Role.LEADER]
        if len(leaders) == 1:
            return leaders[0]
        if not leaders:
            return None
        return max(leaders, key=lambda n: n.current_term)

    def leaders(self) -> list[RaftNode]:
        return [n for n in self.nodes.values() if n.role is Role.LEADER]

    def wait_for_leader(self, max_ms: int = 3000) -> RaftNode:
        for _ in range(max_ms):
            self.step()
            leader = self.leader()
            if leader is not None and len(self.leaders()) == 1:
                return leader
        raise TimeoutError("no leader elected")

    def converged(self) -> bool:
        logs = [tuple((e.term, e.index, e.command) for e in n.log) for n in self.nodes.values()]
        return all(log == logs[0] for log in logs)
