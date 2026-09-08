import numpy as np
import pytest

from annex.config import PQConfig
from annex.errors import IndexError_
from annex.index.pq import ProductQuantizer, kmeans
from annex.index.space import PQSpace
from annex.types import Metric
from annex.util.math import l2_normalize


@pytest.fixture(scope="module")
def data():
    rng = np.random.default_rng(2)
    centers = l2_normalize(rng.normal(size=(20, 32)).astype("float32"))
    idx = rng.integers(0, 20, size=2000)
    return l2_normalize(centers[idx] + rng.normal(scale=0.05, size=(2000, 32)).astype("float32"))


def test_kmeans_converges_to_separated_clusters():
    rng = np.random.default_rng(0)
    a = rng.normal(loc=-5, size=(100, 2))
    b = rng.normal(loc=+5, size=(100, 2))
    centers = kmeans(np.vstack([a, b]).astype("float32"), 2, seed=1)
    assert centers.shape == (2, 2)
    assert abs(centers[:, 0].max() - centers[:, 0].min()) > 5


def test_kmeans_handles_fewer_points_than_k():
    centers = kmeans(np.zeros((2, 3), "float32"), 8)
    assert centers.shape == (8, 3)


def test_encode_decode_roundtrip_is_close(data):
    pq = ProductQuantizer(32, PQConfig(subvectors=8, bits=8, train_iters=15)).train(data)
    codes = pq.encode(data[:50])
    assert codes.shape == (50, 8)
    err = np.linalg.norm(pq.decode(codes) - data[:50], axis=1).mean()
    assert err < 0.25


def test_adc_approximates_true_distance(data):
    pq = ProductQuantizer(32, PQConfig(subvectors=8, train_iters=15), Metric.COSINE).train(data)
    codes = pq.encode(data)
    q = data[0]
    table, const = pq.distance_table(q)
    approx = pq.adc(table, const, codes)
    exact = 1.0 - data @ q
    assert np.corrcoef(approx, exact)[0, 1] > 0.95


def test_untrained_quantizer_raises(data):
    pq = ProductQuantizer(32, PQConfig())
    with pytest.raises(IndexError_):
        pq.encode(data[:1])


def test_pq_space_defers_encoding_until_trained(data):
    space = PQSpace(32, Metric.COSINE, PQConfig(subvectors=8, train_iters=5))
    for i, v in enumerate(data[:100]):
        space.set(i, v)
    assert not space.is_trained
    space.train(data[:500])
    assert space.is_trained
    keys = np.arange(10)
    assert space.distances(data[0], keys).shape == (10,)
    assert space.exact_distances(data[0], keys).shape == (10,)


def test_symmetric_distance_is_non_negative_for_l2(data):
    pq = ProductQuantizer(32, PQConfig(subvectors=8, train_iters=5), Metric.L2).train(data)
    codes = pq.encode(data[:5])
    assert pq.sdc(codes[0], codes[1]) >= 0.0
    assert pq.sdc(codes[0], codes[0]) == pytest.approx(0.0, abs=1e-5)


def test_pq_space_reports_compression(data):
    from annex.config import PQConfig as _PQ

    space = PQSpace(32, Metric.COSINE, _PQ(subvectors=8, train_iters=5), keep_originals=False)
    space.train(data[:500])
    for i, v in enumerate(data[:100]):
        space.set(i, v)
    assert space.compression_ratio() == 16.0
    assert space.code_bytes() < 32 * 4 * 100
    assert space.memory_bytes() == space.code_bytes()


def test_pq_rerank_improves_recall(tmp_path):
    from annex import Database
    from annex.bench.datasets import clustered_dataset, query_set
    from annex.bench.suite import exact_ground_truth, recall_at_k
    from annex.config import PQConfig as _PQ

    data = clustered_dataset(1500, 32, clusters=12, seed=4)
    queries = query_set(data, 25, seed=5)
    truth = exact_ground_truth(data, queries, 10, Metric.COSINE)

    def recall(rerank: int) -> float:
        db = Database(tmp_path / f"pq{rerank}")
        coll = db.create_collection(
            "c", dim=32, index="hnsw_pq", pq=_PQ(subvectors=8, rerank_factor=rerank)
        )
        coll.planner.small_collection_threshold = 0
        coll.planner.scan_speedup = 0.001
        for i, v in enumerate(data):
            coll.upsert(f"d{i}", v)
        found = [[int(h.id[1:]) for h in coll.search(q, k=10, ef=128)] for q in queries]
        return recall_at_k(found, truth, 10)

    assert recall(8) >= recall(1)
