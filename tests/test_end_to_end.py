"""A full lifecycle: ingest -> search -> update -> crash -> recover -> replicate."""

from __future__ import annotations

import numpy as np

from annex import Database
from annex.cluster import ClusterSimulator, RaftNode
from annex.cluster.state_machine import CollectionStateMachine, encode_command
from annex.config import StorageConfig
from annex.storage.codec import Frame, FrameType, encode_frame
from annex.util.math import l2_normalize

CORPUS = [
    ("nn", "neural networks learn hierarchical representations from data"),
    ("ann", "approximate nearest neighbour search scales similarity queries"),
    ("db", "log structured merge trees power modern storage engines"),
    ("raft", "raft is a consensus algorithm for replicated state machines"),
    ("bm25", "bm25 ranks documents by term frequency and inverse document frequency"),
]


def embed(text: str, dim: int = 24, seed: int = 0) -> np.ndarray:
    """Deterministic hashing embedding - good enough to exercise the engine."""
    vec = np.zeros(dim, dtype="float32")
    for token in text.lower().split():
        vec[hash(token) % dim] += 1.0
    return l2_normalize(vec + 1e-3)


def test_full_lifecycle(tmp_path):
    db = Database(tmp_path / "db")
    coll = db.create_collection(
        "kb", dim=24, storage=StorageConfig(memtable_max_records=2, compaction_trigger=2)
    )

    with coll.transaction() as txn:
        for doc_id, text in CORPUS:
            txn.upsert(doc_id, embed(text), {"topic": doc_id, "length": len(text)}, text=text)
    assert len(coll) == len(CORPUS)

    # dense
    assert coll.search(embed(CORPUS[1][1]), k=1).hits[0].id == "ann"
    # lexical
    assert "raft" in coll.search(text="consensus algorithm replicated", k=3).ids()
    # hybrid + filter
    res = coll.search(
        embed("storage engines"),
        text="storage engines",
        k=3,
        filter={"length": {"$gte": 40}},
    )
    assert res.plan["strategy"] == "hybrid"
    assert all(h.metadata["length"] >= 40 for h in res)

    # update under a snapshot
    with coll.snapshot() as snap:
        coll.upsert("bm25", embed("okapi bm25 revisited"), {"topic": "bm25", "length": 20},
                    text="okapi bm25 revisited")
        assert coll.search(embed(CORPUS[4][1]), k=1, snapshot=snap).hits[0].id == "bm25"
    assert coll.get("bm25").metadata["length"] == 20

    coll.delete("nn")
    coll.flush()
    stats = coll.stats()
    assert stats["documents"] == 4
    assert stats["storage"]["segments"] >= 1
    coll.close()

    # simulate a crash: append garbage to the WAL, then reopen
    wal_path = tmp_path / "db" / "kb" / "data" / "wal.log"
    with open(wal_path, "ab") as fh:
        fh.write(encode_frame(Frame(FrameType.PUT, 999, b"torn"))[:12])

    recovered = Database(tmp_path / "db").open_collection("kb")
    assert len(recovered) == 4
    assert recovered.get("nn") is None
    assert recovered.search(embed(CORPUS[1][1]), k=1).hits[0].id == "ann"
    assert recovered.search(text="okapi", k=1).hits[0].id == "bm25"

    # replicate the recovered state onto a 3-node cluster
    ids = ["a", "b", "c"]
    machines = {}
    nodes = []
    for idx, node_id in enumerate(ids):
        replica = Database(tmp_path / node_id).create_collection("kb", dim=24)
        machines[node_id] = CollectionStateMachine(replica)
        nodes.append(RaftNode(node_id, ids, seed=idx * 13 + 1, apply_fn=machines[node_id]))
    sim = ClusterSimulator(nodes, seed=17)
    leader = sim.wait_for_leader()
    for hit in recovered.search(embed("everything"), k=10, include_vectors=True):
        leader.propose(
            encode_command("upsert", id=hit.id, vector=hit.vector,
                           metadata=hit.metadata, text=hit.text)
        )
    sim.run(900)
    assert sim.converged()
    for machine in machines.values():
        assert len(machine.collection) == 4
        assert machine.collection.search(embed(CORPUS[1][1]), k=1).hits[0].id == "ann"
