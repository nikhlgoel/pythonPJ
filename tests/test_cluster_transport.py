"""Phase 5: wire codec, transports, snapshots and membership changes."""

from __future__ import annotations

import numpy as np
import pytest

from annex import Database
from annex.cluster import (
    AppendEntries,
    ClusterSimulator,
    CollectionStateMachine,
    InProcessTransport,
    InstallSnapshot,
    LogEntry,
    Message,
    RaftNode,
    RaftService,
    RequestVote,
    Role,
    config_command,
    decode_message,
    encode_command,
    encode_message,
    is_config,
)
from annex.errors import ClusterError

IDS = ["n1", "n2", "n3"]


# ------------------------------------------------------------------ wire
@pytest.mark.parametrize(
    "payload",
    [
        RequestVote(3, "n1", 7, 2),
        AppendEntries(3, "n1", 4, 2, (LogEntry(3, 5, {"op": "noop"}),), 4),
        InstallSnapshot(4, "n1", 9, 3, ("n1", "n2"), {"documents": []}),
    ],
)
def test_wire_roundtrip(payload):
    msg = Message("n1", "n2", payload)
    assert decode_message(encode_message(msg)) == msg


def test_wire_is_json_serialisable():
    import json

    blob = json.dumps(encode_message(Message("a", "b", RequestVote(1, "a", 0, 0))))
    assert decode_message(json.loads(blob)).payload.candidate_id == "a"


def test_wire_rejects_unknown_kind():
    with pytest.raises(ClusterError):
        decode_message({"src": "a", "dst": "b", "kind": "Nope", "payload": {}})


def test_wire_rejects_malformed_payload():
    with pytest.raises(ClusterError):
        decode_message({"src": "a", "dst": "b", "kind": "RequestVote", "payload": {"term": 1}})


def test_wire_rejects_unencodable_payload():
    with pytest.raises(ClusterError):
        encode_message(Message("a", "b", object()))


# ------------------------------------------------------- in-process cluster
def build_cluster(tmp_path, dim=8, ids=IDS):
    transport = InProcessTransport()
    services, machines = {}, {}
    for i, node_id in enumerate(ids):
        coll = Database(tmp_path / node_id).create_collection("c", dim=dim)
        machine = CollectionStateMachine(coll)
        node = RaftNode(
            node_id,
            ids,
            seed=i * 19 + 3,
            apply_fn=machine,
            snapshot_fn=machine.snapshot,
            restore_fn=machine.restore,
        )
        service = RaftService(node, transport)
        transport.register(node_id, service)
        services[node_id] = service
        machines[node_id] = machine
    return transport, services, machines


def drive(services, ms=600, step=10, start=0):
    clock = start
    for _ in range(ms // step):
        clock += step
        for service in services.values():
            service.tick(clock)
    return clock


def elect(services, ms=2000):
    clock = 0
    for _ in range(ms // 10):
        clock += 10
        for service in services.values():
            service.tick(clock)
        leaders = [s for s in services.values() if s.is_leader]
        if len(leaders) == 1:
            drive(services, 200, start=clock)
            return leaders[0], clock + 200
    raise AssertionError("no leader elected")


def test_in_process_cluster_elects_and_replicates(tmp_path, rng):
    _, services, machines = build_cluster(tmp_path)
    leader, clock = elect(services)
    for i in range(5):
        leader.propose(encode_command("upsert", id=f"d{i}", vector=rng.normal(size=8)))
    drive(services, 600, start=clock)
    assert all(len(m.collection) == 5 for m in machines.values())
    assert all(s.node.commit_index == 5 for s in services.values())


def test_followers_refuse_proposals(tmp_path):
    _, services, _ = build_cluster(tmp_path)
    leader, _ = elect(services)
    follower = next(s for s in services.values() if s is not leader)
    with pytest.raises(ClusterError, match="not the leader"):
        follower.propose(encode_command("noop"))


def test_partitioned_replica_catches_up(tmp_path, rng):
    transport, services, machines = build_cluster(tmp_path)
    leader, clock = elect(services)
    lagging = next(s.id for s in services.values() if s is not leader)

    transport.isolate(lagging)
    for i in range(6):
        leader.propose(encode_command("upsert", id=f"d{i}", vector=rng.normal(size=8)))
    clock = drive(services, 600, start=clock)
    assert len(machines[lagging].collection) == 0

    transport.heal()
    drive(services, 1200, start=clock)
    assert len(machines[lagging].collection) == 6


# ------------------------------------------------------------- snapshots
def test_compaction_discards_the_applied_prefix(tmp_path, rng):
    _, services, _ = build_cluster(tmp_path)
    leader, clock = elect(services)
    for i in range(8):
        leader.propose(encode_command("upsert", id=f"d{i}", vector=rng.normal(size=8)))
    drive(services, 600, start=clock)

    discarded = leader.compact()
    assert discarded > 0
    assert leader.node.snapshot_index == leader.node.last_applied
    assert len(leader.node.log) == 0
    assert leader.compact() == 0  # nothing left to fold


def test_compaction_never_discards_unapplied_entries(tmp_path):
    node = RaftNode("solo", ["solo"])
    node.tick(1000)
    node.propose({"op": "noop"})
    node.commit_index = 0
    node.last_applied = 0
    assert node.compact(up_to=1) == 0


def test_install_snapshot_catches_up_a_far_behind_replica(tmp_path, rng):
    transport, services, machines = build_cluster(tmp_path)
    leader, clock = elect(services)
    lagging = next(s.id for s in services.values() if s is not leader)

    transport.isolate(lagging)
    for i in range(10):
        leader.propose(encode_command("upsert", id=f"d{i}", vector=rng.normal(size=8)))
    clock = drive(services, 600, start=clock)

    # fold everything the leader has applied into a snapshot, discarding the log
    leader.compact()
    assert len(leader.node.log) == 0

    transport.heal()
    drive(services, 2000, start=clock)
    replica = services[lagging].node
    assert replica.snapshot_index == leader.node.snapshot_index
    assert len(machines[lagging].collection) == 10
    hits = machines[lagging].collection.search(np.zeros(8, "float32"), k=3)
    assert len(hits) == 3


def test_snapshot_roundtrips_through_the_state_machine(tmp_path, rng):
    source = Database(tmp_path / "a").create_collection("c", dim=8)
    target = Database(tmp_path / "b").create_collection("c", dim=8)
    src, dst = CollectionStateMachine(source), CollectionStateMachine(target)
    for i in range(6):
        src(encode_command("upsert", id=f"d{i}", vector=rng.normal(size=8), metadata={"i": i}))
    dst(encode_command("upsert", id="stale", vector=rng.normal(size=8)))

    dst.restore(src.snapshot())
    assert len(target) == 6
    assert target.get("stale") is None
    assert target.get("d3").metadata == {"i": 3}
    dst.restore(None)  # a missing snapshot is a no-op
    assert len(target) == 6


# ------------------------------------------------------------ membership
def test_config_command_helpers():
    assert is_config(config_command(["b", "a"]))
    assert not is_config({"op": "upsert"})
    assert config_command(["b", "a"])["__members__"] == ["a", "b"]


def test_adding_a_server_grows_the_quorum(tmp_path):
    _, services, _ = build_cluster(tmp_path)
    leader, clock = elect(services)
    assert leader.node.quorum == 2

    assert leader.add_server("n4") is not None
    assert leader.node.quorum == 3
    assert "n4" in leader.node.members
    assert leader.add_server("n4") is None  # already a member

    drive(services, 400, start=clock)
    follower = next(s for s in services.values() if s is not leader)
    assert "n4" in follower.node.members  # adopted on append, not on commit


def test_removing_a_server_shrinks_the_quorum(tmp_path):
    _, services, _ = build_cluster(tmp_path)
    leader, clock = elect(services)
    victim = next(s.id for s in services.values() if s is not leader)

    assert leader.remove_server(victim) is not None
    assert victim not in leader.node.members
    assert leader.node.quorum == 2
    assert leader.remove_server("ghost") is None
    assert victim not in leader.node.next_index

    drive(services, 400, start=clock)
    assert leader.is_leader


def test_membership_survives_replication(tmp_path):
    _, services, _ = build_cluster(tmp_path)
    leader, clock = elect(services)
    leader.add_server("n9")
    drive(services, 600, start=clock)
    assert all("n9" in s.node.members for s in services.values())


# ------------------------------------------------------ simulator parity
def test_snapshotting_node_still_converges_in_the_simulator():
    nodes = [RaftNode(i, IDS, seed=idx * 5 + 1) for idx, i in enumerate(IDS)]
    sim = ClusterSimulator(nodes, seed=4)
    leader = sim.wait_for_leader()
    for i in range(10):
        leader.propose({"op": "noop", "i": i})
    sim.run(700)
    leader.compact()
    for i in range(10, 14):
        leader.propose({"op": "noop", "i": i})
    sim.run(900)
    assert all(n.commit_index == 14 for n in nodes)
    assert all(len(n.applied_commands) == 14 for n in nodes)
    assert leader.role is Role.LEADER
