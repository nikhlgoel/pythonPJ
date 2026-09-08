import pytest

from embra.query.filter import compile_filter
from embra.query.fusion import reciprocal_rank_fusion, weighted_fusion
from embra.query.planner import QueryPlanner, Strategy


def test_rrf_rewards_agreement():
    dense = [("a", 0.1), ("b", 0.2), ("c", 0.3)]
    lexical = [("c", 9.0), ("a", 4.0)]
    fused = reciprocal_rank_fusion([dense, lexical])
    assert fused[0][0] == "a"  # ranked 1st and 2nd in the two lists
    assert {k for k, _ in fused} == {"a", "b", "c"}


def test_rrf_handles_one_empty_list():
    fused = reciprocal_rank_fusion([[("a", 1.0)], []])
    assert fused[0][0] == "a"


def test_rrf_respects_weights():
    dense = [("a", 1.0), ("b", 1.0)]
    lexical = [("b", 1.0), ("a", 1.0)]
    assert reciprocal_rank_fusion([dense, lexical], weights=[5.0, 1.0])[0][0] == "a"
    assert reciprocal_rank_fusion([dense, lexical], weights=[1.0, 5.0])[0][0] == "b"


def test_rrf_top_k():
    assert len(reciprocal_rank_fusion([[("a", 1), ("b", 1), ("c", 1)]], top_k=2)) == 2


def test_rrf_rejects_mismatched_weights():
    with pytest.raises(ValueError):
        reciprocal_rank_fusion([[("a", 1)]], weights=[1.0, 2.0])


def test_weighted_fusion_normalises_scales():
    dense = [("a", 0.9), ("b", 0.1)]
    lexical = [("b", 100.0), ("a", 1.0)]
    fused = dict(weighted_fusion([dense, lexical]))
    assert pytest.approx(fused["a"], abs=1e-6) == 1.0
    assert pytest.approx(fused["b"], abs=1e-6) == 1.0


def plan(**kw):
    defaults = {
        "k": 10, "live": 100_000, "filt": compile_filter(None), "has_vector": True,
        "has_text": False, "ef": None, "default_ef": 64, "index_kind": "hnsw",
    }
    return QueryPlanner().plan(**{**defaults, **kw})


def test_large_unfiltered_collection_uses_the_graph():
    assert plan().strategy is Strategy.ANN_GRAPH


def test_small_collection_uses_exact_scan():
    assert plan(live=500).strategy is Strategy.EXACT_SCAN


def test_flat_index_always_scans():
    assert plan(index_kind="flat").strategy is Strategy.EXACT_SCAN


def test_selective_filter_prefers_a_prefiltered_scan():
    filt = compile_filter({"a": 1, "b": 2, "c": 3})
    assert plan(filt=filt).strategy is Strategy.EXACT_SCAN


def test_text_query_without_vector_is_lexical():
    assert plan(has_vector=False, has_text=True).strategy is Strategy.LEXICAL_ONLY


def test_vector_and_text_is_hybrid():
    planned = plan(has_text=True)
    assert planned.strategy is Strategy.HYBRID
    assert planned.explain["vector_stage"] in ("ann_graph", "exact_scan")


def test_ef_is_never_below_k():
    assert plan(k=128, ef=8).ef >= 128


def test_plan_serialises_with_a_reason():
    payload = plan().as_dict()
    assert payload["strategy"] == "ann_graph"
    assert "reason" in payload
