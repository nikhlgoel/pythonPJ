"""Simulate a 5-node Embra cluster: elect a leader, replicate writes, survive a
network partition, and verify every replica converges.

    python examples/replicated_cluster.py
"""

from __future__ import annotations

import shutil
import tempfile

import numpy as np

import embra
from embra.cluster import ClusterSimulator, RaftNode
from embra.cluster.state_machine import CollectionStateMachine, encode_command
from embra.util.math import l2_normalize

DIM = 16
NODE_IDS = ["n1", "n2", "n3", "n4", "n5"]


def main() -> None:
    workdir = tempfile.mkdtemp(prefix="embra-cluster-")
    rng = np.random.default_rng(0)

    machines: dict[str, CollectionStateMachine] = {}
    nodes: list[RaftNode] = []
    for i, node_id in enumerate(NODE_IDS):
        coll = embra.Database(f"{workdir}/{node_id}").create_collection("kb", dim=DIM)
        machines[node_id] = CollectionStateMachine(coll)
        nodes.append(RaftNode(node_id, NODE_IDS, seed=i * 17 + 3, apply_fn=machines[node_id]))

    sim = ClusterSimulator(nodes, latency=(5, 20), seed=1)
    leader = sim.wait_for_leader()
    sim.run(200)
    print(f"elected leader: {leader.id} (term {leader.current_term})")

    for i in range(12):
        vec = l2_normalize(rng.normal(size=DIM).astype("float32"))
        leader.propose(encode_command("upsert", id=f"d{i}", vector=vec, metadata={"i": i}))
    sim.run(800)
    print(f"replicated 12 writes -> commit indexes: {[n.commit_index for n in nodes]}")
    print(f"documents per replica: {[len(m.collection) for m in machines.values()]}")

    minority = {leader.id}
    majority = set(NODE_IDS) - minority
    sim.partition(minority, majority)
    print(f"\npartitioning {sorted(minority)} away from {sorted(majority)} ...")
    sim.run(2500)
    survivors = [n for n in sim.leaders() if n.id in majority]
    new_leader = survivors[0]
    print(f"majority elected a new leader: {new_leader.id} (term {new_leader.current_term})")

    for i in range(12, 18):
        vec = l2_normalize(rng.normal(size=DIM).astype("float32"))
        new_leader.propose(encode_command("upsert", id=f"d{i}", vector=vec, metadata={"i": i}))
    sim.run(800)
    print(f"minority could not commit: old leader index={leader.commit_index}, "
          f"new leader index={new_leader.commit_index}")

    sim.heal()
    sim.run(2500)
    print(f"\nhealed. logs identical across replicas: {sim.converged()}")
    print(f"documents per replica: {[len(m.collection) for m in machines.values()]}")
    print(f"messages delivered={sim.delivered} dropped={sim.dropped}")

    shutil.rmtree(workdir, ignore_errors=True)


if __name__ == "__main__":
    main()
