import numpy as np
import pytest

from embra import Database
from embra.errors import ConflictError, NotFoundError, SchemaError, TransactionError
from embra.types import IndexKind, Metric
from embra.util.math import l2_normalize


def fill(coll, vectors, *, texts=True):
    for i, vec in enumerate(vectors):
        coll.upsert(
            f"d{i}",
            vec,
            {"group": i % 5, "rank": i, "nested": {"flag": i % 2 == 0}},
            text=f"document number {i} about vectors" if texts else None,
        )


# ----------------------------------------------------------------- basics
def test_upsert_and_get(collection, vectors):
    collection.upsert("a", vectors[0], {"x": 1}, text="hello")
    hit = collection.get("a")
    assert hit.id == "a" and hit.metadata == {"x": 1} and hit.text == "hello"
    assert "a" in collection and len(collection) == 1


def test_get_missing_returns_none(collection):
    assert collection.get("nope") is None


def test_get_with_vector(collection, vectors):
    collection.upsert("a", vectors[0])
    assert collection.get("a", include_vector=True).vector.shape == (32,)


def test_upsert_rejects_bad_dimension(collection):
    with pytest.raises(SchemaError):
        collection.upsert("a", np.ones(5, "float32"))


def test_upsert_rejects_empty_id(collection, vectors):
    with pytest.raises(SchemaError):
        collection.upsert("", vectors[0])


def test_update_replaces_document(collection, vectors):
    collection.upsert("a", vectors[0], {"v": 1})
    collection.upsert("a", vectors[1], {"v": 2})
    assert len(collection) == 1
    assert collection.get("a").metadata == {"v": 2}
    assert collection.search(vectors[1], k=1).hits[0].id == "a"


def test_delete(collection, vectors):
    collection.upsert("a", vectors[0])
    assert collection.delete("a") is True
    assert collection.delete("a") is False
    assert collection.get("a") is None
    assert collection.search(vectors[0], k=5).hits == []


def test_upsert_many(collection, vectors):
    n = collection.upsert_many(
        [{"id": f"x{i}", "vector": v, "metadata": {"i": i}} for i, v in enumerate(vectors[:20])]
    )
    assert n == 20 and len(collection) == 20


# ----------------------------------------------------------------- search
def test_search_finds_exact_match(collection, vectors):
    fill(collection, vectors[:100])
    res = collection.search(vectors[7], k=1)
    assert res.hits[0].id == "d7"
    assert res.hits[0].score == pytest.approx(1.0, abs=1e-4)
    assert res.took_ms >= 0


def test_search_scores_are_descending(collection, vectors):
    fill(collection, vectors[:100])
    scores = [h.score for h in collection.search(vectors[3], k=10)]
    assert scores == sorted(scores, reverse=True)


def test_search_respects_k(collection, vectors):
    fill(collection, vectors[:50])
    assert len(collection.search(vectors[0], k=7)) == 7


def test_search_with_filter(collection, vectors):
    fill(collection, vectors[:100])
    res = collection.search(vectors[0], k=10, filter={"group": 3})
    assert res.hits and all(h.metadata["group"] == 3 for h in res)


def test_search_with_nested_filter(collection, vectors):
    fill(collection, vectors[:100])
    res = collection.search(vectors[0], k=5, filter={"nested.flag": True})
    assert all(h.metadata["nested"]["flag"] for h in res)


def test_search_with_range_filter(collection, vectors):
    fill(collection, vectors[:100])
    res = collection.search(vectors[0], k=10, filter={"rank": {"$gte": 90}})
    assert all(h.metadata["rank"] >= 90 for h in res)


def test_filter_matching_nothing_returns_empty(collection, vectors):
    fill(collection, vectors[:50])
    assert collection.search(vectors[0], k=5, filter={"group": 99}).hits == []


def test_lexical_only_search(collection, vectors):
    fill(collection, vectors[:30])
    res = collection.search(text="document number 5", k=3)
    assert res.plan["strategy"] == "lexical_only"
    assert res.hits


def test_hybrid_search_uses_rrf(collection, vectors):
    fill(collection, vectors[:100])
    res = collection.search(vectors[9], text="document number 9", k=5)
    assert res.plan["strategy"] == "hybrid"
    assert "d9" in res.ids()


def test_search_empty_collection(collection, vectors):
    assert collection.search(vectors[0], k=5).hits == []


def test_search_include_vectors(collection, vectors):
    fill(collection, vectors[:20])
    hit = collection.search(vectors[0], k=1, include_vectors=True).hits[0]
    assert hit.vector is not None and hit.vector.shape == (32,)


def test_explain_can_be_disabled(collection, vectors):
    fill(collection, vectors[:20])
    assert collection.search(vectors[0], k=1, explain=False).plan == {}


def test_ef_override_is_reported(db, vectors):
    coll = db.create_collection("big", dim=32)
    coll.planner.small_collection_threshold = 0
    coll.planner.scan_speedup = 0.01  # force the planner onto the graph path
    for i, v in enumerate(vectors):
        coll.upsert(f"d{i}", v)
    res = coll.search(vectors[0], k=5, ef=200)
    assert res.plan["ef"] == 200
    assert res.plan["strategy"] == "ann_graph"


# --------------------------------------------------------------- metrics
@pytest.mark.parametrize("metric", ["cosine", "l2", "dot"])
def test_all_metrics_find_the_nearest_point(db, rng, metric):
    coll = db.create_collection(f"m{metric}", dim=16, metric=metric)
    data = l2_normalize(rng.normal(size=(200, 16)).astype("float32"))
    for i, v in enumerate(data):
        coll.upsert(f"d{i}", v)
    assert coll.search(data[11], k=1).hits[0].id == "d11"


def test_flat_index_collection(db, vectors):
    coll = db.create_collection("flat", dim=32, index=IndexKind.FLAT)
    for i, v in enumerate(vectors[:100]):
        coll.upsert(f"d{i}", v)
    res = coll.search(vectors[5], k=3)
    assert res.hits[0].id == "d5"
    assert res.plan["strategy"] == "exact_scan"


def test_pq_index_collection(db, rng):
    coll = db.create_collection("pq", dim=32, index=IndexKind.HNSW_PQ)
    data = l2_normalize(rng.normal(size=(1200, 32)).astype("float32"))
    for i, v in enumerate(data):
        coll.upsert(f"d{i}", v)
    assert coll.search(data[100], k=1).hits[0].id == "d100"


# ------------------------------------------------------------------ MVCC
def test_snapshot_isolates_reads(collection, vectors):
    fill(collection, vectors[:50])
    with collection.snapshot() as snap:
        collection.upsert("d0", vectors[49], {"group": 0, "rank": 0})
        collection.delete("d1")
        old = collection.search(vectors[0], k=5, snapshot=snap)
        assert "d1" in {h.id for h in collection.search(vectors[1], k=5, snapshot=snap)}
        assert old.hits[0].id == "d0"
    assert "d1" not in collection.search(vectors[1], k=5).ids()


def test_snapshot_sees_old_metadata(collection, vectors):
    collection.upsert("a", vectors[0], {"v": 1})
    with collection.snapshot() as snap:
        collection.upsert("a", vectors[0], {"v": 2})
        assert collection.search(vectors[0], k=1, snapshot=snap).hits[0].metadata == {"v": 1}
    assert collection.search(vectors[0], k=1).hits[0].metadata == {"v": 2}


def test_vacuum_reclaims_only_invisible_versions(collection, vectors):
    fill(collection, vectors[:20])
    with collection.snapshot():
        collection.upsert("d0", vectors[400], {"group": 0, "rank": 0})
        assert collection.vacuum() == 0  # an open snapshot still needs the old row
    assert collection.vacuum() == 1
    assert collection.search(vectors[400], k=1).hits[0].id == "d0"


def test_vacuum_after_delete(collection, vectors):
    fill(collection, vectors[:10])
    collection.delete("d4")
    assert collection.vacuum() == 1
    assert "d4" not in collection.search(vectors[4], k=5).ids()


# ---------------------------------------------------------- transactions
def test_transaction_commits_atomically(collection, vectors):
    with collection.transaction() as txn:
        txn.upsert("t1", vectors[0], {"n": 1})
        txn.upsert("t2", vectors[1], {"n": 2})
    assert len(collection) == 2


def test_transaction_rolls_back_on_exception(collection, vectors):
    with pytest.raises(RuntimeError), collection.transaction() as txn:
        txn.upsert("t1", vectors[0])
        raise RuntimeError("boom")
    assert len(collection) == 0


def test_explicit_rollback(collection, vectors):
    txn = collection.transaction()
    txn.upsert("t1", vectors[0])
    txn.rollback()
    assert len(collection) == 0
    with pytest.raises(TransactionError):
        txn.upsert("t2", vectors[1])


def test_transaction_delete(collection, vectors):
    collection.upsert("a", vectors[0])
    with collection.transaction() as txn:
        txn.delete("a")
    assert collection.get("a") is None


def test_conflicting_transactions_raise(collection, vectors):
    collection.upsert("a", vectors[0], {"v": 0})
    first = collection.transaction()
    first.upsert("a", vectors[1], {"v": 1})
    collection.upsert("a", vectors[2], {"v": 2})  # concurrent committed write
    with pytest.raises(ConflictError):
        first.commit()
    assert collection.get("a").metadata == {"v": 2}


def test_commit_after_commit_raises(collection, vectors):
    txn = collection.transaction()
    txn.upsert("a", vectors[0])
    txn.commit()
    with pytest.raises(TransactionError):
        txn.commit()


# ------------------------------------------------------------ durability
def test_reopen_preserves_data(tmp_path, vectors):
    db = Database(tmp_path / "db")
    coll = db.create_collection("c", dim=32)
    fill(coll, vectors[:60])
    coll.delete("d3")
    coll.close()

    reopened = Database(tmp_path / "db").open_collection("c")
    assert len(reopened) == 59
    assert reopened.get("d3") is None
    assert reopened.search(vectors[7], k=1).hits[0].id == "d7"
    assert reopened.search(text="document number 7", k=1).hits


def test_reopen_preserves_config(tmp_path):
    db = Database(tmp_path / "db")
    db.create_collection("c", dim=8, metric=Metric.L2, index=IndexKind.FLAT).close()
    cfg = Database(tmp_path / "db").open_collection("c").config
    assert cfg.metric is Metric.L2 and cfg.index is IndexKind.FLAT


def test_rebuild_index_preserves_results(collection, vectors):
    fill(collection, vectors[:80])
    collection.delete("d5")
    before = collection.search(vectors[9], k=5).ids()
    collection.rebuild_index()
    assert collection.search(vectors[9], k=5).ids() == before
    assert "d5" not in collection.search(vectors[5], k=5).ids()
    assert collection.search(text="document number 9", k=1).hits


def test_stats_shape(collection, vectors):
    fill(collection, vectors[:10])
    stats = collection.stats()
    assert stats["documents"] == 10
    assert stats["index"]["type"] in ("hnsw", "flat")
    assert stats["storage"]["records"] == 10
    assert stats["text"]["docs"] == 10


# -------------------------------------------------------------- database
def test_database_lifecycle(tmp_path, vectors):
    db = Database(tmp_path / "db")
    db.create_collection("a", dim=4)
    db.create_collection("b", dim=4)
    assert db.list_collections() == ["a", "b"]
    assert list(db) == ["a", "b"]
    db.drop_collection("a")
    assert db.list_collections() == ["b"]
    with pytest.raises(NotFoundError):
        db.open_collection("a")
    with pytest.raises(NotFoundError):
        db.drop_collection("ghost")
    db.close()


def test_create_collection_is_idempotent(db):
    first = db.create_collection("c", dim=4)
    assert db.create_collection("c", dim=4) is first
    with pytest.raises(SchemaError):
        db.create_collection("c", dim=4, exist_ok=False)


def test_getitem_opens_collection(db):
    db.create_collection("c", dim=4)
    assert db["c"].name == "c"
