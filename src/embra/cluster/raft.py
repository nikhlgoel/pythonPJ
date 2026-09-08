"""Raft consensus - leader election, log replication and the commit rule.

The implementation follows the Raft paper (Ongaro & Ousterhout, 2014), figure 2,
and is intentionally *pure*: :meth:`RaftNode.tick` and :meth:`RaftNode.handle`
take the current logical time and an inbound message and return the list of
messages to send.  There is no I/O, no threading and no wall clock inside the
node, so a five-node cluster with a network partition is an ordinary unit test.

Safety properties implemented here
----------------------------------
* **Election safety** - a node grants at most one vote per term, and only to a
  candidate whose log is at least as up-to-date as its own.
* **Log matching** - ``AppendEntries`` carries ``(prev_index, prev_term)``; a
  follower rejects anything that would create a hole, and conflicting suffixes
  are truncated before appending.
* **Leader completeness** - a leader only advances ``commit_index`` to an entry
  *from its own term* that is replicated on a quorum, which is the subtle rule
  that prevents committed-entry loss after a leader change.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from enum import Enum
from typing import Any

NodeId = str


class Role(str, Enum):
    FOLLOWER = "follower"
    CANDIDATE = "candidate"
    LEADER = "leader"


@dataclass(frozen=True, slots=True)
class LogEntry:
    term: int
    index: int
    command: Any


@dataclass(frozen=True, slots=True)
class RequestVote:
    term: int
    candidate_id: NodeId
    last_log_index: int
    last_log_term: int


@dataclass(frozen=True, slots=True)
class RequestVoteReply:
    term: int
    vote_granted: bool
    voter: NodeId


@dataclass(frozen=True, slots=True)
class AppendEntries:
    term: int
    leader_id: NodeId
    prev_log_index: int
    prev_log_term: int
    entries: tuple[LogEntry, ...]
    leader_commit: int


@dataclass(frozen=True, slots=True)
class AppendEntriesReply:
    term: int
    success: bool
    follower: NodeId
    match_index: int
    #: fast back-off hint so a lagging follower converges in O(1) round trips
    conflict_index: int = 0


@dataclass(frozen=True, slots=True)
class Message:
    """An envelope on the wire."""

    src: NodeId
    dst: NodeId
    payload: Any


class RaftNode:
    """One replica.  Drive it with :meth:`tick` and :meth:`handle`."""

    def __init__(
        self,
        node_id: NodeId,
        peers: list[NodeId],
        *,
        election_timeout: tuple[int, int] = (150, 300),
        heartbeat_interval: int = 50,
        seed: int | None = None,
        apply_fn=None,
    ) -> None:
        self.id = node_id
        self.peers = [p for p in peers if p != node_id]
        self.role = Role.FOLLOWER
        self.current_term = 0
        self.voted_for: NodeId | None = None
        self.log: list[LogEntry] = []
        self.commit_index = 0
        self.last_applied = 0
        self.leader_id: NodeId | None = None

        self.next_index: dict[NodeId, int] = {}
        self.match_index: dict[NodeId, int] = {}
        self._votes: set[NodeId] = set()

        self._rng = random.Random(seed if seed is not None else hash(node_id) & 0xFFFF)
        self._election_range = election_timeout
        self._heartbeat_interval = heartbeat_interval
        self._election_deadline = self._new_deadline(0)
        self._next_heartbeat = 0
        self.apply_fn = apply_fn
        self.applied_commands: list[Any] = []

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------
    @property
    def quorum(self) -> int:
        return (len(self.peers) + 1) // 2 + 1

    @property
    def last_index(self) -> int:
        return self.log[-1].index if self.log else 0

    @property
    def last_term(self) -> int:
        return self.log[-1].term if self.log else 0

    def entry_at(self, index: int) -> LogEntry | None:
        if index <= 0 or index > len(self.log):
            return None
        return self.log[index - 1]

    def term_at(self, index: int) -> int:
        entry = self.entry_at(index)
        return entry.term if entry else 0

    def _new_deadline(self, now: int) -> int:
        return now + self._rng.randint(*self._election_range)

    def _become_follower(self, term: int, now: int) -> None:
        self.role = Role.FOLLOWER
        self.current_term = term
        self.voted_for = None
        self._votes.clear()
        self._election_deadline = self._new_deadline(now)

    # ------------------------------------------------------------------
    # driving the node
    # ------------------------------------------------------------------
    def tick(self, now: int) -> list[Message]:
        """Advance logical time; returns messages to deliver."""
        if self.role is Role.LEADER:
            if now >= self._next_heartbeat:
                self._next_heartbeat = now + self._heartbeat_interval
                return self._broadcast_append()
            return []
        if now >= self._election_deadline:
            return self._start_election(now)
        return []

    def _start_election(self, now: int) -> list[Message]:
        self.role = Role.CANDIDATE
        self.current_term += 1
        self.voted_for = self.id
        self._votes = {self.id}
        self.leader_id = None
        self._election_deadline = self._new_deadline(now)
        if not self.peers:  # single-node cluster commits immediately
            self._become_leader(now)
            return []
        rv = RequestVote(self.current_term, self.id, self.last_index, self.last_term)
        return [Message(self.id, p, rv) for p in self.peers]

    def _become_leader(self, now: int) -> None:
        self.role = Role.LEADER
        self.leader_id = self.id
        self.next_index = dict.fromkeys(self.peers, self.last_index + 1)
        self.match_index = dict.fromkeys(self.peers, 0)
        self._next_heartbeat = now

    def propose(self, command: Any) -> int | None:
        """Append a client command.  Returns its log index, or ``None`` if this
        node is not the leader."""
        if self.role is not Role.LEADER:
            return None
        entry = LogEntry(self.current_term, self.last_index + 1, command)
        self.log.append(entry)
        if not self.peers:
            self.commit_index = entry.index
            self._apply_committed()
        return entry.index

    # ------------------------------------------------------------------
    # RPC handling
    # ------------------------------------------------------------------
    def handle(self, msg: Message, now: int) -> list[Message]:
        payload = msg.payload
        if isinstance(payload, RequestVote):
            return self._on_request_vote(msg.src, payload, now)
        if isinstance(payload, RequestVoteReply):
            return self._on_vote_reply(payload, now)
        if isinstance(payload, AppendEntries):
            return self._on_append(msg.src, payload, now)
        if isinstance(payload, AppendEntriesReply):
            return self._on_append_reply(payload, now)
        raise TypeError(f"unknown payload {type(payload).__name__}")

    def _on_request_vote(self, src: NodeId, rv: RequestVote, now: int) -> list[Message]:
        if rv.term > self.current_term:
            self._become_follower(rv.term, now)
        granted = False
        up_to_date = rv.last_log_term > self.last_term or (
            rv.last_log_term == self.last_term and rv.last_log_index >= self.last_index
        )
        if (
            rv.term == self.current_term
            and self.voted_for in (None, rv.candidate_id)
            and up_to_date
        ):
            granted = True
            self.voted_for = rv.candidate_id
            self._election_deadline = self._new_deadline(now)
        reply = RequestVoteReply(self.current_term, granted, self.id)
        return [Message(self.id, src, reply)]

    def _on_vote_reply(self, reply: RequestVoteReply, now: int) -> list[Message]:
        if reply.term > self.current_term:
            self._become_follower(reply.term, now)
            return []
        if self.role is not Role.CANDIDATE or reply.term != self.current_term:
            return []
        if reply.vote_granted:
            self._votes.add(reply.voter)
            if len(self._votes) >= self.quorum:
                self._become_leader(now)
                return self._broadcast_append()
        return []

    def _on_append(self, src: NodeId, ae: AppendEntries, now: int) -> list[Message]:
        if ae.term < self.current_term:
            return [
                Message(self.id, src, AppendEntriesReply(self.current_term, False, self.id, 0))
            ]
        if ae.term > self.current_term or self.role is not Role.FOLLOWER:
            self._become_follower(ae.term, now)
        self.leader_id = ae.leader_id
        self._election_deadline = self._new_deadline(now)

        if ae.prev_log_index > 0 and self.term_at(ae.prev_log_index) != ae.prev_log_term:
            # fast back-off: point the leader at the first index of the conflict
            conflict = min(ae.prev_log_index, self.last_index + 1)
            if conflict <= self.last_index and conflict > 0:
                bad_term = self.term_at(conflict)
                while conflict > 1 and self.term_at(conflict - 1) == bad_term:
                    conflict -= 1
            return [
                Message(
                    self.id,
                    src,
                    AppendEntriesReply(self.current_term, False, self.id, 0, conflict),
                )
            ]

        for entry in ae.entries:
            existing = self.entry_at(entry.index)
            if existing is not None and existing.term != entry.term:
                del self.log[entry.index - 1 :]  # truncate conflicting suffix
                existing = None
            if existing is None:
                self.log.append(entry)

        if ae.leader_commit > self.commit_index:
            self.commit_index = min(ae.leader_commit, self.last_index)
            self._apply_committed()
        return [
            Message(
                self.id,
                src,
                AppendEntriesReply(self.current_term, True, self.id, self.last_index),
            )
        ]

    def _on_append_reply(self, reply: AppendEntriesReply, now: int) -> list[Message]:
        if reply.term > self.current_term:
            self._become_follower(reply.term, now)
            return []
        if self.role is not Role.LEADER:
            return []
        if reply.success:
            self.match_index[reply.follower] = reply.match_index
            self.next_index[reply.follower] = reply.match_index + 1
            self._advance_commit()
            return []
        self.next_index[reply.follower] = max(1, reply.conflict_index or
                                              self.next_index.get(reply.follower, 1) - 1)
        return self._append_to(reply.follower)

    # ------------------------------------------------------------------
    def _broadcast_append(self) -> list[Message]:
        out: list[Message] = []
        for peer in self.peers:
            out.extend(self._append_to(peer))
        return out

    def _append_to(self, peer: NodeId) -> list[Message]:
        next_idx = self.next_index.get(peer, self.last_index + 1)
        prev_index = next_idx - 1
        entries = tuple(self.log[prev_index:])
        ae = AppendEntries(
            self.current_term,
            self.id,
            prev_index,
            self.term_at(prev_index),
            entries,
            self.commit_index,
        )
        return [Message(self.id, peer, ae)]

    def _advance_commit(self) -> None:
        """Commit the highest index replicated on a quorum *in the current term*."""
        for index in range(self.last_index, self.commit_index, -1):
            if self.term_at(index) != self.current_term:
                continue
            replicated = 1 + sum(1 for m in self.match_index.values() if m >= index)
            if replicated >= self.quorum:
                self.commit_index = index
                self._apply_committed()
                return

    def _apply_committed(self) -> None:
        while self.last_applied < self.commit_index:
            self.last_applied += 1
            entry = self.entry_at(self.last_applied)
            if entry is None:  # pragma: no cover - defensive
                break
            self.applied_commands.append(entry.command)
            if self.apply_fn is not None:
                self.apply_fn(entry.command)

    # ------------------------------------------------------------------
    def state(self) -> dict:
        return {
            "id": self.id,
            "role": self.role.value,
            "term": self.current_term,
            "leader": self.leader_id,
            "log_length": len(self.log),
            "commit_index": self.commit_index,
            "last_applied": self.last_applied,
        }
