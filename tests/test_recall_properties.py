"""Statistical / property-style guarantees for the whole engine."""

from __future__ import annotations

import numpy as np
import pytest

from embra import Database
from embra.bench.datasets import clustered_dataset, query_set, random_dataset
from embra.bench.suite import exact_ground_truth, recall_at_k
from embra.types import IndexKind, Metric


@pytest.fixture(scope="module")
def corpus():
    return clustered_dataset(3000, 48, clusters=24, seed=11)


def _recall(coll, data, queries, k, **kw):
    truth = exact_ground_truth(data, queries, k, Metric.COSINE)
    found = [[int(h.id[1:]) for h in coll.search(q, k=k, **kw)] for q in queries]
    return recall_at_k(found, truth, k)


def test_hnsw_recall_above_90_percent(tmp_path, corpus):
    coll = Database(tmp_path / "r").create_collection("c", dim=48)
    coll.planner.small_collection_threshold = 0
    coll.planner.scan_speedup = 0.001  # force the graph path
    for i, v in enumerate(corpus):
        coll.upsert(f"d{i}", v)
    queries = query_set(corpus, 60, seed=3)
    assert _recall(coll, corpus, queries, 10, ef=128) > 0.9


def test_recall_increases_with_ef(tmp_path):
    data = random_dataset(2000, 32, seed=5)
    coll = Database(tmp_path / "e").create_collection("c", dim=32)
    coll.planner.small_collection_threshold = 0
    coll.planner.scan_speedup = 0.001
    for i, v in enumerate(data):
        coll.upsert(f"d{i}", v)
    queries = random_dataset(40, 32, seed=6)
    low = _recall(coll, data, queries, 10, ef=10)
    high = _recall(coll, data, queries, 10, ef=256)
    assert high >= low


def test_exact_scan_path_is_exact(tmp_path):
    data = random_dataset(800, 16, seed=2)
    coll = Database(tmp_path / "x").create_collection("c", dim=16, index=IndexKind.FLAT)
    for i, v in enumerate(data):
        coll.upsert(f"d{i}", v)
    queries = random_dataset(20, 16, seed=4)
    assert _recall(coll, data, queries, 10) == pytest.approx(1.0)


def test_filtered_results_are_always_exact_subsets(tmp_path):
    data = random_dataset(1500, 16, seed=8)
    coll = Database(tmp_path / "f").create_collection("c", dim=16)
    for i, v in enumerate(data):
        coll.upsert(f"d{i}", v, {"g": i % 10})
    for group in range(10):
        hits = coll.search(data[0], k=20, filter={"g": group})
        assert all(h.metadata["g"] == group for h in hits)


def test_engine_survives_a_write_delete_update_storm(tmp_path):
    rng = np.random.default_rng(99)
    data = random_dataset(600, 12, seed=12)
    coll = Database(tmp_path / "s").create_collection("c", dim=12)
    live: set[str] = set()
    for i, vec in enumerate(data):
        doc = f"d{i % 200}"
        action = rng.integers(0, 3)
        if action == 2 and live:
            victim = rng.choice(sorted(live))
            coll.delete(str(victim))
            live.discard(str(victim))
        else:
            coll.upsert(doc, vec, {"i": i})
            live.add(doc)
        if i % 97 == 0:
            coll.vacuum()
    assert len(coll) == len(live)
    for doc in sorted(live)[:20]:
        assert coll.get(doc) is not None
    hits = coll.search(data[0], k=10)
    assert all(h.id in live for h in hits)
