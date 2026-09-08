import numpy as np
import pytest

from embra.config import HNSWConfig
from embra.index import HNSWIndex
from embra.index.flat import FlatIndex, exact_knn
from embra.index.space import ExactSpace
from embra.types import Metric
from embra.util.math import l2_normalize


def build(data, **kw):
    space = ExactSpace(data.shape[1], Metric.COSINE, capacity=len(data))
    index = HNSWIndex(space, HNSWConfig(**{"m": 16, "ef_construction": 120, **kw}))
    for i, v in enumerate(data):
        index.add(i, v)
    return index


@pytest.fixture(scope="module")
def data():
    rng = np.random.default_rng(7)
    return l2_normalize(rng.normal(size=(1500, 24)).astype("float32"))


def test_recall_is_high(data):
    index = build(data)
    rng = np.random.default_rng(11)
    hit = 0
    for _ in range(40):
        q = l2_normalize(rng.normal(size=24).astype("float32"))
        got = {k for k, _ in index.search(q, 10, ef=96)}
        want = {k for k, _ in exact_knn(data, q, 10, Metric.COSINE)}
        hit += len(got & want)
    assert hit / 400 > 0.9


def test_higher_ef_never_hurts_recall_much(data):
    index = build(data)
    rng = np.random.default_rng(3)
    queries = l2_normalize(rng.normal(size=(20, 24)).astype("float32"))

    def recall(ef):
        total = 0
        for q in queries:
            got = {k for k, _ in index.search(q, 10, ef=ef)}
            want = {k for k, _ in exact_knn(data, q, 10, Metric.COSINE)}
            total += len(got & want)
        return total / (20 * 10)

    assert recall(200) >= recall(16) - 1e-9


def test_results_are_sorted_by_distance(data):
    index = build(data)
    hits = index.search(data[0], 20, ef=64)
    dists = [d for _, d in hits]
    assert dists == sorted(dists)


def test_self_query_returns_self(data):
    index = build(data)
    top = index.search(data[42], 1, ef=64)
    assert top[0][0] == 42


def test_delete_removes_from_results(data):
    index = build(data)
    index.remove(42)
    assert 42 not in {k for k, _ in index.search(data[42], 5, ef=64)}
    assert index.size == len(data) - 1


def test_update_replaces_vector():
    rng = np.random.default_rng(5)
    data = l2_normalize(rng.normal(size=(200, 8)).astype("float32"))
    index = build(data)
    replacement = l2_normalize(rng.normal(size=8).astype("float32"))
    index.add(3, replacement)
    top = index.search(replacement, 1, ef=64)
    assert top[0][0] == 3


def test_filter_predicate_is_applied(data):
    index = build(data)
    allowed = set(range(0, len(data), 7))
    hits = index.search(data[0], 10, ef=200, allow=lambda k: k in allowed)
    assert {k for k, _ in hits} <= allowed


def test_empty_index_returns_nothing():
    space = ExactSpace(4, Metric.COSINE)
    index = HNSWIndex(space)
    assert index.search(np.zeros(4, "float32"), 5) == []
    assert index.size == 0


def test_k_zero_returns_empty(data):
    assert build(data[:50]).search(data[0], 0) == []


def test_graph_stats(data):
    stats = build(data[:300]).stats()
    assert stats["type"] == "hnsw"
    assert stats["levels"] >= 1
    assert stats["avg_degree_l0"] > 0


def test_flat_index_is_exact(data):
    space = ExactSpace(data.shape[1], Metric.COSINE, capacity=len(data))
    flat = FlatIndex(space)
    for i, v in enumerate(data):
        flat.add(i, v)
    got = [k for k, _ in flat.search(data[9], 5)]
    want = [k for k, _ in exact_knn(data, data[9], 5, Metric.COSINE)]
    assert got == want
    flat.remove(got[0])
    assert got[0] not in [k for k, _ in flat.search(data[9], 5)]
