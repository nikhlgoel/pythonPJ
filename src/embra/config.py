"""Configuration objects with validation.

Configuration is explicit and immutable: a collection's on-disk manifest stores
exactly this structure, so a reopened collection can never silently change its
metric, dimension or index parameters.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from .errors import ConfigError
from .types import IndexKind, Metric


@dataclass(frozen=True, slots=True)
class HNSWConfig:
    """Parameters of the hierarchical navigable small-world graph.

    ``M`` controls graph degree (memory / recall trade-off), ``ef_construction``
    the width of the beam used while inserting, ``ef_search`` the default beam
    width at query time.
    """

    m: int = 16
    ef_construction: int = 200
    ef_search: int = 64
    level_multiplier: float | None = None  # defaults to 1/ln(M)
    seed: int = 42

    def __post_init__(self) -> None:
        if self.m < 2:
            raise ConfigError("hnsw.m must be >= 2")
        if self.ef_construction < self.m:
            raise ConfigError("hnsw.ef_construction must be >= m")
        if self.ef_search < 1:
            raise ConfigError("hnsw.ef_search must be >= 1")

    @property
    def m_max0(self) -> int:
        """Max degree of layer 0 (double the upper layers, as in the paper)."""
        return self.m * 2


@dataclass(frozen=True, slots=True)
class PQConfig:
    """Product-quantisation parameters used by the compressed index."""

    subvectors: int = 8
    bits: int = 8
    train_iters: int = 25
    train_sample: int = 50_000
    seed: int = 7

    def __post_init__(self) -> None:
        if self.subvectors < 1:
            raise ConfigError("pq.subvectors must be >= 1")
        if self.bits not in (4, 8):
            raise ConfigError("pq.bits must be 4 or 8")

    @property
    def centroids(self) -> int:
        return 1 << self.bits


@dataclass(frozen=True, slots=True)
class StorageConfig:
    """Durability and compaction knobs of the LSM storage engine."""

    memtable_max_records: int = 10_000
    wal_sync: bool = False           # fsync on every append (slow, safest)
    wal_segment_bytes: int = 64 << 20
    compaction_trigger: int = 4      # number of L0 segments before merge
    checkpoint_every: int = 50_000   # mutations between manifest checkpoints

    def __post_init__(self) -> None:
        if self.memtable_max_records < 1:
            raise ConfigError("storage.memtable_max_records must be >= 1")
        if self.compaction_trigger < 2:
            raise ConfigError("storage.compaction_trigger must be >= 2")


@dataclass(frozen=True, slots=True)
class CollectionConfig:
    """Immutable description of a collection."""

    name: str
    dim: int
    metric: Metric = Metric.COSINE
    index: IndexKind = IndexKind.HNSW
    hnsw: HNSWConfig = field(default_factory=HNSWConfig)
    pq: PQConfig = field(default_factory=PQConfig)
    storage: StorageConfig = field(default_factory=StorageConfig)
    enable_text_index: bool = True
    dtype: str = "float32"

    def __post_init__(self) -> None:
        if not self.name or not self.name.replace("_", "").replace("-", "").isalnum():
            raise ConfigError(f"invalid collection name: {self.name!r}")
        if self.dim < 1:
            raise ConfigError("dim must be >= 1")
        if self.index is IndexKind.HNSW_PQ and self.dim % self.pq.subvectors:
            raise ConfigError("dim must be divisible by pq.subvectors")
        if self.dtype not in ("float32", "float64"):
            raise ConfigError("dtype must be float32 or float64")

    # -- serialisation -------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["metric"] = self.metric.value
        d["index"] = self.index.value
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> CollectionConfig:
        d = dict(d)
        return cls(
            name=d["name"],
            dim=int(d["dim"]),
            metric=Metric(d.get("metric", "cosine")),
            index=IndexKind(d.get("index", "hnsw")),
            hnsw=HNSWConfig(**d.get("hnsw", {})),
            pq=PQConfig(**d.get("pq", {})),
            storage=StorageConfig(**d.get("storage", {})),
            enable_text_index=bool(d.get("enable_text_index", True)),
            dtype=d.get("dtype", "float32"),
        )
