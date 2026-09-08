"""The public engine API: :class:`Database`, :class:`Collection`, transactions.

This module wires together every other layer:

    Database  ->  Collection  ->  { LSMStore, VectorIndex, BM25Index, MVCC }

Design notes
------------
*Internal keys are dense integers.*  Every document version gets one; the vector
index, the BM25 index and the MVCC version table are all keyed by it, so the
three subsystems stay in lock-step without any string hashing on the hot path.

*Visibility is a predicate, not a rebuild.*  Updating a document does not touch
the graph; it allocates a new key and marks the old version dead.  Reads pass a
visibility predicate into the index, so old snapshots keep working and writers
never block readers.

*Everything is crash-safe.*  A mutation is durable in the WAL before it is
acknowledged, and reopening a collection replays segments then the WAL tail and
rebuilds the in-memory indexes deterministically.
"""

from __future__ import annotations

import json
import shutil
import threading
from collections.abc import Iterable, Iterator, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from .config import CollectionConfig, StorageConfig
from .errors import ConflictError, NotFoundError, SchemaError, TransactionError
from .index import FlatIndex, HNSWIndex, build_index
from .index.space import PQSpace
from .mvcc import INF, Snapshot, SnapshotRegistry, Version
from .query.filter import Filter, compile_filter
from .query.fusion import reciprocal_rank_fusion
from .query.planner import PlannedQuery, QueryPlanner, Strategy
from .storage import LSMStore
from .text import BM25Index, Tokenizer
from .types import DocId, Hit, IndexKind, Metadata, Metric, Record, SearchResult, Seq
from .util.math import distance_matrix, l2_normalize, score_from_distance, validate_vector
from .util.timing import Stopwatch

CONFIG_FILE = "collection.json"


class Transaction:
    """Buffered, atomically-applied group of writes with conflict detection.

    Writes are staged in memory and applied under the collection lock at commit
    time.  If any document staged by this transaction was modified by another
    committed transaction after this one started, the commit raises
    :class:`ConflictError` (optimistic concurrency control, first-committer-wins).
    """

    def __init__(self, collection: Collection) -> None:
        self.collection = collection
        self.start_seq: Seq = collection.current_seq
        self._ops: list[tuple[str, tuple[Any, ...]]] = []
        self._touched: set[DocId] = set()
        self._done = False

    # -- staging --------------------------------------------------------
    def upsert(
        self,
        doc_id: DocId,
        vector,
        metadata: Metadata | None = None,
        text: str | None = None,
    ) -> Transaction:
        self._check_open()
        self._ops.append(("upsert", (doc_id, vector, metadata or {}, text)))
        self._touched.add(doc_id)
        return self

    def delete(self, doc_id: DocId) -> Transaction:
        self._check_open()
        self._ops.append(("delete", (doc_id,)))
        self._touched.add(doc_id)
        return self

    def _check_open(self) -> None:
        if self._done:
            raise TransactionError("transaction already finished")

    # -- lifecycle ------------------------------------------------------
    def commit(self) -> int:
        self._check_open()
        applied = self.collection._commit_transaction(self)
        self._done = True
        return applied

    def rollback(self) -> None:
        self._ops.clear()
        self._done = True

    def __len__(self) -> int:
        return len(self._ops)

    def __enter__(self) -> Transaction:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if exc_type is not None:
            self.rollback()
        elif not self._done:
            self.commit()


class Collection:
    """A set of vectors sharing one dimensionality, metric and index."""

    def __init__(self, directory: str | Path, config: CollectionConfig) -> None:
        self.dir = Path(directory)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.config = config
        self._lock = threading.RLock()
        self._snapshots = SnapshotRegistry()
        self.planner = QueryPlanner()

        self._store = LSMStore(self.dir / "data", config.dim, config.storage)
        self._index = build_index(config, capacity=max(1024, len(self._store) * 2))
        self._bm25 = BM25Index(Tokenizer()) if config.enable_text_index else None

        self._versions: dict[int, Version] = {}
        self._current: dict[DocId, int] = {}
        self._next_key = 0
        self._rebuild_from_store()

    # ------------------------------------------------------------------
    # construction / persistence
    # ------------------------------------------------------------------
    @property
    def name(self) -> str:
        return self.config.name

    @property
    def metric(self) -> Metric:
        return self.config.metric

    @property
    def current_seq(self) -> Seq:
        return self._store.seq

    def __len__(self) -> int:
        return len(self._current)

    def __contains__(self, doc_id: object) -> bool:
        return doc_id in self._current

    def _write_config(self) -> None:
        (self.dir / CONFIG_FILE).write_text(json.dumps(self.config.to_dict(), indent=2))

    def _rebuild_from_store(self) -> None:
        """Reconstruct MVCC tables and in-memory indexes after a restart."""
        records = sorted(self._store.iter_live(), key=lambda r: r.seq)
        if not records:
            return
        if isinstance(getattr(self._index, "space", None), PQSpace):
            sample = np.stack([r.vector for r in records if r.vector is not None])
            self._index.space.train(sample)  # type: ignore[union-attr]
        for rec in records:
            if rec.vector is None:
                continue
            key = self._next_key
            self._next_key += 1
            vec = self._prepare(rec.vector)
            self._index.add(key, vec)
            if self._bm25 is not None and rec.text:
                self._bm25.add(key, rec.text)
            self._versions[key] = Version(key, rec.id, rec.seq, INF, rec.metadata, rec.text)
            self._current[rec.id] = key

    def _prepare(self, vector) -> np.ndarray:
        vec = validate_vector(vector, self.config.dim, self.config.dtype)
        return l2_normalize(vec) if self.metric.normalizes else vec

    # ------------------------------------------------------------------
    # writes
    # ------------------------------------------------------------------
    def upsert(
        self,
        doc_id: DocId,
        vector,
        metadata: Metadata | None = None,
        text: str | None = None,
    ) -> Seq:
        """Insert or replace one document; returns the commit sequence number."""
        with self._lock:
            return self._apply_upsert(doc_id, vector, metadata or {}, text)

    def _apply_upsert(
        self, doc_id: DocId, vector, metadata: Metadata, text: str | None
    ) -> Seq:
        if not isinstance(doc_id, str) or not doc_id:
            raise SchemaError("document id must be a non-empty string")
        vec = self._prepare(vector)
        seq = self._store.next_seq()
        self._store.put(doc_id, vec, metadata, text, seq=seq)

        old_key = self._current.get(doc_id)
        if old_key is not None:
            self._versions[old_key].dead = seq

        key = self._next_key
        self._next_key += 1
        self._maybe_train(vec)
        self._index.add(key, vec)
        if self._bm25 is not None and text:
            self._bm25.add(key, text)
        self._versions[key] = Version(key, doc_id, seq, INF, dict(metadata), text)
        self._current[doc_id] = key
        return seq

    def _maybe_train(self, vec: np.ndarray) -> None:
        """Train the product quantiser once enough vectors have accumulated."""
        space = getattr(self._index, "space", None)
        if not isinstance(space, PQSpace) or space.is_trained:
            return
        pending = len(space._pending)  # noqa: SLF001 - same package
        if pending + 1 >= min(self.config.pq.train_sample, 256 * 4):
            sample = np.stack([*space._pending.values(), vec])  # noqa: SLF001
            space.train(sample)

    def upsert_many(self, records: Iterable[Record | dict]) -> int:
        """Batch upsert.  Returns the number of documents written."""
        count = 0
        with self._lock:
            for rec in records:
                if isinstance(rec, dict):
                    rec = Record(
                        id=rec["id"],
                        vector=np.asarray(rec["vector"], dtype=self.config.dtype),
                        metadata=rec.get("metadata", {}),
                        text=rec.get("text"),
                    )
                self._apply_upsert(rec.id, rec.vector, rec.metadata, rec.text)
                count += 1
            self._store.checkpoint()
        return count

    def delete(self, doc_id: DocId) -> bool:
        """Delete a document.  Returns ``False`` if it was not present."""
        with self._lock:
            key = self._current.pop(doc_id, None)
            if key is None:
                return False
            seq = self._store.next_seq()
            self._store.delete(doc_id, seq=seq)
            self._versions[key].dead = seq
            return True

    def transaction(self) -> Transaction:
        """Open an optimistic, atomically-applied write transaction."""
        return Transaction(self)

    def _commit_transaction(self, txn: Transaction) -> int:
        with self._lock:
            for doc_id in txn._touched:  # noqa: SLF001 - same module
                key = self._current.get(doc_id)
                if key is not None and self._versions[key].born > txn.start_seq:
                    raise ConflictError(
                        f"document {doc_id!r} was modified concurrently "
                        f"(born={self._versions[key].born} > snapshot={txn.start_seq})"
                    )
            applied = 0
            for op, args in txn._ops:  # noqa: SLF001
                if op == "upsert":
                    doc_id, vector, metadata, text = args
                    self._apply_upsert(doc_id, vector, metadata, text)
                else:
                    doc_id = args[0]
                    key = self._current.pop(doc_id, None)
                    if key is None:
                        continue
                    seq = self._store.next_seq()
                    self._store.delete(doc_id, seq=seq)
                    self._versions[key].dead = seq
                applied += 1
            self._store.checkpoint()
            return applied

    # ------------------------------------------------------------------
    # reads
    # ------------------------------------------------------------------
    def get(self, doc_id: DocId, *, include_vector: bool = False) -> Hit | None:
        key = self._current.get(doc_id)
        if key is None:
            return None
        ver = self._versions[key]
        vec = None
        if include_vector:
            rec = self._store.get(doc_id)
            vec = None if rec is None else rec.vector
        return Hit(doc_id, 1.0, ver.metadata, ver.text, vec)

    def snapshot(self) -> Snapshot:
        """Pin a consistent read view of the collection."""
        return Snapshot(self._snapshots, self.current_seq, self)

    def _visibility(self, snap: Seq, filt: Filter):
        versions = self._versions

        def allow(key: int) -> bool:
            ver = versions.get(key)
            if ver is None or not ver.visible_at(snap):
                return False
            return filt.predicate(ver.metadata)

        return allow

    def _live_keys(self, snap: Seq, allow) -> np.ndarray:
        return np.fromiter(
            (k for k in self._versions if allow(k)), dtype=np.int64
        )

    def _exact_search(self, query: np.ndarray, k: int, keys: np.ndarray) -> list[tuple[int, float]]:
        if keys.size == 0:
            return []
        space = self._index.space  # type: ignore[attr-defined]
        if isinstance(space, PQSpace):
            dists = space.exact_distances(query, keys)
        else:
            dists = distance_matrix(query, space.data[keys] if hasattr(space, "data") else space._data[keys], self.metric)  # noqa: SLF001
        kk = min(k, keys.size)
        part = np.argpartition(dists, kk - 1)[:kk]
        order = part[np.argsort(dists[part], kind="stable")]
        return [(int(keys[i]), float(dists[i])) for i in order]

    def search(
        self,
        vector=None,
        *,
        text: str | None = None,
        k: int = 10,
        filter: dict | None = None,  # noqa: A002 - mirrors the query DSL name
        ef: int | None = None,
        snapshot: Snapshot | None = None,
        rrf_k: float = 60.0,
        weights: Sequence[float] = (1.0, 1.0),
        oversample: float = 1.0,
        include_vectors: bool = False,
        explain: bool = True,
    ) -> SearchResult:
        """Dense, lexical or hybrid retrieval with metadata filtering.

        Parameters
        ----------
        vector:
            Query embedding.  Omit it to run a pure BM25 query.
        text:
            Query string.  Supplying both ``vector`` and ``text`` runs a hybrid
            query fused with reciprocal rank fusion.
        filter:
            Metadata filter in the DSL of :mod:`embra.query.filter`.
        ef:
            Override the HNSW beam width for this query (recall/latency dial).
        snapshot:
            Read at a pinned point in time instead of "now".
        """
        with Stopwatch() as sw:
            snap = snapshot.seq if snapshot is not None else self.current_seq
            filt = compile_filter(filter)
            allow = self._visibility(snap, filt)
            has_vec = vector is not None
            plan = self.planner.plan(
                k=k,
                live=len(self._current),
                filt=filt,
                has_vector=has_vec,
                has_text=bool(text) and self._bm25 is not None,
                ef=ef,
                default_ef=self.config.hnsw.ef_search,
                index_kind=self.config.index.value,
                oversample=oversample,
            )
            scanned = 0
            dense: list[tuple[int, float]] = []
            if has_vec:
                q = self._prepare(vector)
                if plan.strategy is Strategy.EXACT_SCAN or (
                    plan.strategy is Strategy.HYBRID
                    and plan.explain.get("vector_stage") == "exact_scan"
                ):
                    keys = self._live_keys(snap, allow)
                    scanned = int(keys.size)
                    dense = self._exact_search(q, plan.fetch_k, keys)
                else:
                    dense = self._index.search(q, plan.fetch_k, ef=plan.ef, allow=allow)
                    scanned = plan.ef

            lexical: list[tuple[int, float]] = []
            if text and self._bm25 is not None:
                lexical = self._bm25.search(text, plan.fetch_k, allow=allow)

            ranked = self._merge(plan, dense, lexical, k, rrf_k, weights)
            hits = [self._to_hit(key, score, include_vectors) for key, score in ranked]

        return SearchResult(
            hits=hits,
            took_ms=sw.ms,
            candidates_scanned=scanned,
            plan=plan.as_dict() if explain else {},
        )

    def _merge(
        self,
        plan: PlannedQuery,
        dense: list[tuple[int, float]],
        lexical: list[tuple[int, float]],
        k: int,
        rrf_k: float,
        weights: Sequence[float],
    ) -> list[tuple[int, float]]:
        if plan.strategy is Strategy.LEXICAL_ONLY:
            return lexical[:k]
        if not lexical:
            return [(key, float(score_from_distance(d, self.metric))) for key, d in dense[:k]]
        if not dense:
            return lexical[:k]
        fused = reciprocal_rank_fusion(
            [dense, lexical], weights=list(weights), k_constant=rrf_k, top_k=k
        )
        return [(int(key), float(score)) for key, score in fused]

    def _to_hit(self, key: int, score: float, include_vector: bool) -> Hit:
        ver = self._versions[key]
        vec = None
        if include_vector:
            space = self._index.space  # type: ignore[attr-defined]
            vec = np.array(space._data[key] if hasattr(space, "_data") else space._raw[key])  # noqa: SLF001
        return Hit(ver.doc_id, score, ver.metadata, ver.text, vec)

    # ------------------------------------------------------------------
    # maintenance
    # ------------------------------------------------------------------
    def vacuum(self) -> int:
        """Drop dead versions no open snapshot can still see.

        Returns the number of versions reclaimed.
        """
        with self._lock:
            horizon = self._snapshots.horizon(self.current_seq)
            dead = [k for k, v in self._versions.items() if v.dead <= horizon]
            for key in dead:
                try:
                    self._index.remove(key)
                except Exception:  # noqa: BLE001 - index may not know the key
                    pass
                if self._bm25 is not None:
                    self._bm25.remove(key)
                del self._versions[key]
            return len(dead)

    def rebuild_index(self) -> None:
        """Rebuild the ANN index from scratch (compacts graph tombstones)."""
        with self._lock:
            new_index = build_index(self.config, capacity=max(1024, len(self._current) * 2))
            space = getattr(new_index, "space", None)
            live = [(key, self._versions[key]) for key in self._current.values()]
            if isinstance(space, PQSpace) and live:
                mat = np.stack([self._vector_of(k) for k, _ in live])
                space.train(mat)
            remap: dict[int, int] = {}
            for new_key, (old_key, ver) in enumerate(live):
                new_index.add(new_key, self._vector_of(old_key))
                remap[old_key] = new_key
            versions: dict[int, Version] = {}
            bm25 = BM25Index(Tokenizer()) if self.config.enable_text_index else None
            for old_key, new_key in remap.items():
                ver = self._versions[old_key]
                versions[new_key] = Version(new_key, ver.doc_id, ver.born, INF, ver.metadata, ver.text)
                if bm25 is not None and ver.text:
                    bm25.add(new_key, ver.text)
            self._index = new_index
            self._versions = versions
            self._bm25 = bm25
            self._current = {v.doc_id: k for k, v in versions.items()}
            self._next_key = len(versions)

    def _vector_of(self, key: int) -> np.ndarray:
        space = self._index.space  # type: ignore[attr-defined]
        if isinstance(space, PQSpace):
            if space._raw is not None:  # noqa: SLF001
                return space._raw[key]  # noqa: SLF001
            return space.pq.decode(space._codes[key : key + 1])[0]  # noqa: SLF001
        return space._data[key]  # noqa: SLF001

    def flush(self) -> None:
        with self._lock:
            self._store.flush()

    def close(self) -> None:
        with self._lock:
            self._store.close()

    def stats(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "documents": len(self._current),
            "versions": len(self._versions),
            "open_snapshots": len(self._snapshots),
            "config": self.config.to_dict(),
            "index": self._index.stats(),
            "storage": self._store.stats(),
            "text": self._bm25.stats() if self._bm25 else None,
        }

    def __repr__(self) -> str:  # pragma: no cover
        return f"<Collection {self.name!r} docs={len(self)} dim={self.config.dim}>"


class Database:
    """A directory of collections."""

    def __init__(self, path: str | Path = ".embra-data") -> None:
        self.path = Path(path)
        self.path.mkdir(parents=True, exist_ok=True)
        self._collections: dict[str, Collection] = {}

    # ------------------------------------------------------------------
    def _dir_for(self, name: str) -> Path:
        return self.path / name

    def list_collections(self) -> list[str]:
        return sorted(
            p.name for p in self.path.iterdir() if (p / CONFIG_FILE).exists()
        )

    def create_collection(
        self,
        name: str,
        dim: int,
        *,
        metric: Metric | str = Metric.COSINE,
        index: IndexKind | str = IndexKind.HNSW,
        exist_ok: bool = True,
        **kwargs: Any,
    ) -> Collection:
        """Create (or open, when ``exist_ok``) a collection."""
        directory = self._dir_for(name)
        cfg_path = directory / CONFIG_FILE
        if cfg_path.exists():
            if not exist_ok:
                raise SchemaError(f"collection {name!r} already exists")
            return self.open_collection(name)
        storage = kwargs.pop("storage", None) or StorageConfig()
        config = CollectionConfig(
            name=name,
            dim=dim,
            metric=Metric(metric),
            index=IndexKind(index),
            storage=storage,
            **kwargs,
        )
        directory.mkdir(parents=True, exist_ok=True)
        coll = Collection(directory, config)
        coll._write_config()  # noqa: SLF001
        self._collections[name] = coll
        return coll

    def open_collection(self, name: str) -> Collection:
        if name in self._collections:
            return self._collections[name]
        cfg_path = self._dir_for(name) / CONFIG_FILE
        if not cfg_path.exists():
            raise NotFoundError(f"no such collection: {name!r}")
        config = CollectionConfig.from_dict(json.loads(cfg_path.read_text()))
        coll = Collection(self._dir_for(name), config)
        self._collections[name] = coll
        return coll

    def drop_collection(self, name: str) -> None:
        coll = self._collections.pop(name, None)
        if coll is not None:
            coll.close()
        directory = self._dir_for(name)
        if not directory.exists():
            raise NotFoundError(f"no such collection: {name!r}")
        shutil.rmtree(directory)

    def __getitem__(self, name: str) -> Collection:
        return self.open_collection(name)

    def __iter__(self) -> Iterator[str]:
        return iter(self.list_collections())

    def close(self) -> None:
        for coll in self._collections.values():
            coll.close()
        self._collections.clear()

    def __enter__(self) -> Database:
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def stats(self) -> dict[str, Any]:
        return {
            "path": str(self.path),
            "collections": {n: c.stats() for n, c in self._collections.items()},
        }
