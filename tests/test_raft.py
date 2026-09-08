import pytest

from embra.cluster import ClusterSimulator, RaftNode, Role
from embra.cluster.raft import AppendEntries, LogEntry, Message, RequestVote
from embra.cluster.state_machine import CollectionStateMachine, encode_command


def cluster(n=5, **kw):
    ids = [f"n{i}" for i in range(1, n + 1)]
    nodes = [RaftNode(i, ids, seed=7 * idx + 1, **kw) for idx, i in enumerate(ids)]
    return nodes, ClusterSimulator(nodes, seed=42)


def test_single_node_elects_itself_and_commits():
    node = RaftNode("solo", ["solo"])
    node.tick(1000)
    assert node.role is Role.LEADER
    assert node.propose({"op": "noop"}) == 1
    assert node.commit_index == 1
    assert node.applied_commands == [{"op": "noop"}]


def test_exactly_one_leader_is_elected():
    nodes, sim = cluster()
    leader = sim.wait_for_leader()
    sim.run(200)  # let the first heartbeats land
    assert len(sim.leaders()) == 1
    assert all(n.leader_id == leader.id for n in nodes if n is not leader)


def test_entries_replicate_to_every_follower():
    nodes, sim = cluster()
    leader = sim.wait_for_leader()
    for i in range(10):
        leader.propose({"op": "noop", "i": i})
    sim.run(600)
    assert sim.converged()
    assert all(n.commit_index == 10 for n in nodes)
    assert all(len(n.applied_commands) == 10 for n in nodes)


def test_followers_reject_client_proposals():
    nodes, sim = cluster()
    leader = sim.wait_for_leader()
    follower = next(n for n in nodes if n is not leader)
    assert follower.propose({"op": "noop"}) is None


def test_new_leader_after_the_old_one_is_isolated():
    nodes, sim = cluster()
    old = sim.wait_for_leader()
    for i in range(3):
        old.propose({"op": "noop", "i": i})
    sim.run(400)
    majority = {n.id for n in nodes if n is not old}
    sim.partition({old.id}, majority)
    sim.run(2000)
    new_leaders = [n for n in sim.leaders() if n.id in majority]
    assert len(new_leaders) == 1
    assert new_leaders[0].current_term > old.current_term


def test_cluster_reconverges_after_healing():
    nodes, sim = cluster()
    old = sim.wait_for_leader()
    majority = {n.id for n in nodes if n is not old}
    sim.partition({old.id}, majority)
    sim.run(2000)
    new_leader = next(n for n in sim.leaders() if n.id in majority)
    for i in range(5):
        new_leader.propose({"op": "noop", "i": i})
    sim.run(600)
    sim.heal()
    sim.run(2000)
    assert sim.converged()
    assert len(sim.leaders()) == 1
    assert all(n.commit_index == new_leader.commit_index for n in nodes)


def test_minority_partition_cannot_commit():
    nodes, sim = cluster()
    leader = sim.wait_for_leader()
    minority = {leader.id, nodes[0].id if nodes[0] is not leader else nodes[1].id}
    sim.partition(minority, {n.id for n in nodes} - minority)
    before = leader.commit_index
    leader.propose({"op": "noop"})
    sim.run(1500)
    assert leader.commit_index == before  # no quorum in a 2/5 partition


def test_lossy_network_still_converges():
    ids = [f"n{i}" for i in range(1, 4)]
    nodes = [RaftNode(i, ids, seed=3 * idx + 5) for idx, i in enumerate(ids)]
    sim = ClusterSimulator(nodes, drop_rate=0.25, seed=9)
    leader = sim.wait_for_leader(6000)
    for i in range(8):
        leader.propose({"op": "noop", "i": i})
    sim.run(6000)
    assert max(n.commit_index for n in nodes) == 8


def test_vote_is_denied_to_a_stale_log():
    node = RaftNode("a", ["a", "b"])
    node.log = [LogEntry(2, 1, "x"), LogEntry(2, 2, "y")]
    node.current_term = 2
    replies = node.handle(Message("b", "a", RequestVote(3, "b", 0, 0)), 0)
    assert replies[0].payload.vote_granted is False


def test_vote_is_granted_once_per_term():
    node = RaftNode("a", ["a", "b", "c"])
    first = node.handle(Message("b", "a", RequestVote(1, "b", 0, 0)), 0)
    second = node.handle(Message("c", "a", RequestVote(1, "c", 0, 0)), 0)
    assert first[0].payload.vote_granted is True
    assert second[0].payload.vote_granted is False


def test_append_entries_truncates_a_conflicting_suffix():
    node = RaftNode("a", ["a", "b"])
    node.current_term = 2
    node.log = [LogEntry(1, 1, "x"), LogEntry(1, 2, "bad")]
    ae = AppendEntries(2, "b", 1, 1, (LogEntry(2, 2, "good"),), 2)
    reply = node.handle(Message("b", "a", ae), 0)[0].payload
    assert reply.success
    assert node.log[1].command == "good"
    assert node.commit_index == 2


def test_append_entries_from_a_stale_leader_is_rejected():
    node = RaftNode("a", ["a", "b"])
    node.current_term = 5
    ae = AppendEntries(3, "b", 0, 0, (), 0)
    assert node.handle(Message("b", "a", ae), 0)[0].payload.success is False


def test_gap_in_the_log_is_rejected_with_a_backoff_hint():
    node = RaftNode("a", ["a", "b"])
    node.current_term = 1
    ae = AppendEntries(1, "b", 5, 1, (), 0)
    reply = node.handle(Message("b", "a", ae), 0)[0].payload
    assert reply.success is False
    assert reply.conflict_index >= 1


def test_unknown_payload_raises():
    node = RaftNode("a", ["a"])
    with pytest.raises(TypeError):
        node.handle(Message("b", "a", object()), 0)


def test_state_machine_applies_commands_to_a_collection(db, vectors):
    coll = db.create_collection("replica", dim=32)
    machine = CollectionStateMachine(coll)
    machine(encode_command("upsert", id="a", vector=vectors[0], metadata={"n": 1}, text="hi"))
    machine(encode_command("upsert", id="b", vector=vectors[1]))
    machine(encode_command("delete", id="b"))
    machine(encode_command("noop"))
    assert len(coll) == 1
    assert coll.get("a").metadata == {"n": 1}
    assert machine.applied == 4


def test_state_machine_rejects_unknown_ops(db):
    machine = CollectionStateMachine(db.create_collection("r2", dim=4))
    with pytest.raises(ValueError):
        machine({"op": "nope"})


def test_replicated_writes_reach_every_replica(tmp_path, vectors):
    from embra import Database

    ids = ["n1", "n2", "n3"]
    machines = {}
    nodes = []
    for idx, node_id in enumerate(ids):
        coll = Database(tmp_path / node_id).create_collection("c", dim=32)
        machines[node_id] = CollectionStateMachine(coll)
        nodes.append(RaftNode(node_id, ids, seed=idx * 11 + 2, apply_fn=machines[node_id]))
    sim = ClusterSimulator(nodes, seed=5)
    leader = sim.wait_for_leader()
    for i in range(6):
        leader.propose(encode_command("upsert", id=f"d{i}", vector=vectors[i], metadata={"i": i}))
    sim.run(800)
    assert all(len(m.collection) == 6 for m in machines.values())
    for machine in machines.values():
        assert machine.collection.search(vectors[3], k=1).hits[0].id == "d3"
