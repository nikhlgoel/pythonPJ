import pytest

from annex.config import CollectionConfig, HNSWConfig, PQConfig, StorageConfig
from annex.errors import ConfigError
from annex.types import IndexKind, Metric


def test_roundtrip():
    cfg = CollectionConfig(name="a", dim=8, metric=Metric.L2, index=IndexKind.FLAT)
    assert CollectionConfig.from_dict(cfg.to_dict()) == cfg


@pytest.mark.parametrize(
    "kwargs",
    [
        {"name": "bad name!", "dim": 4},
        {"name": "ok", "dim": 0},
        {"name": "ok", "dim": 4, "dtype": "int8"},
    ],
)
def test_invalid_collection_config(kwargs):
    with pytest.raises(ConfigError):
        CollectionConfig(**kwargs)


def test_pq_divisibility_enforced():
    with pytest.raises(ConfigError):
        CollectionConfig(name="c", dim=10, index=IndexKind.HNSW_PQ, pq=PQConfig(subvectors=3))


def test_hnsw_bounds():
    with pytest.raises(ConfigError):
        HNSWConfig(m=1)
    with pytest.raises(ConfigError):
        HNSWConfig(m=16, ef_construction=4)
    assert HNSWConfig(m=16).m_max0 == 32


def test_storage_bounds():
    with pytest.raises(ConfigError):
        StorageConfig(compaction_trigger=1)


def test_metric_normalisation_flag():
    assert Metric.COSINE.normalizes
    assert not Metric.L2.normalizes
