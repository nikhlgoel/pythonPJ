"""Replication.

:mod:`annex.cluster.raft` is a deterministic, tick-driven implementation of the
Raft consensus algorithm (leader election + log replication + commit rules).
"Deterministic" is the important word: the node has no threads and no clocks of
its own, so an entire cluster can be simulated - including partitions, message
loss and reordering - inside a unit test.

:mod:`annex.cluster.state_machine` binds the replicated log to an Annex
:class:`~annex.db.Collection`, making the collection the state machine that Raft
keeps identical across replicas.
"""

from .raft import (
    AppendEntries,
    AppendEntriesReply,
    InstallSnapshot,
    InstallSnapshotReply,
    LogEntry,
    Message,
    RaftNode,
    RequestVote,
    RequestVoteReply,
    Role,
    config_command,
    is_config,
)
from .service import RaftService
from .simulator import ClusterSimulator
from .state_machine import CollectionStateMachine, encode_command
from .transport import HttpTransport, InProcessTransport, Transport
from .wire import decode_message, encode_message

__all__ = [
    "AppendEntries",
    "AppendEntriesReply",
    "ClusterSimulator",
    "CollectionStateMachine",
    "HttpTransport",
    "InProcessTransport",
    "InstallSnapshot",
    "InstallSnapshotReply",
    "LogEntry",
    "Message",
    "RaftNode",
    "RaftService",
    "RequestVote",
    "RequestVoteReply",
    "Role",
    "Transport",
    "config_command",
    "decode_message",
    "encode_command",
    "encode_message",
    "is_config",
]
