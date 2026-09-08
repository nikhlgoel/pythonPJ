import pytest

from annex.errors import QueryError
from annex.query.filter import compile_filter

META = {"year": 2021, "venue": "NeurIPS", "tags": ["ml", "ann"], "author": {"name": "ada"}}


@pytest.mark.parametrize(
    "spec,expected",
    [
        (None, True),
        ({}, True),
        ({"venue": "NeurIPS"}, True),
        ({"venue": "ICML"}, False),
        ({"year": {"$gte": 2020}}, True),
        ({"year": {"$lt": 2000}}, False),
        ({"year": {"$between": [2020, 2022]}}, True),
        ({"tags": {"$contains": "ann"}}, True),
        ({"tags": {"$contains": "nlp"}}, False),
        ({"venue": {"$in": ["ICML", "NeurIPS"]}}, True),
        ({"venue": {"$nin": ["ICML"]}}, True),
        ({"author.name": "ada"}, True),
        ({"missing": {"$exists": False}}, True),
        ({"venue": {"$regex": "^Neur"}}, True),
        ({"$and": [{"year": {"$gte": 2020}}, {"venue": "NeurIPS"}]}, True),
        ({"$or": [{"venue": "ICML"}, {"venue": "NeurIPS"}]}, True),
        ({"$not": {"venue": "ICML"}}, True),
        ({"venue": "NeurIPS", "year": 2021}, True),
        ({"venue": "NeurIPS", "year": 1999}, False),
    ],
)
def test_filter_semantics(spec, expected):
    assert compile_filter(spec)(META) is expected


def test_missing_field_is_not_equal():
    assert compile_filter({"nope": {"$ne": 3}})(META) is True
    assert compile_filter({"nope": 3})(META) is False


def test_type_mismatch_is_false_not_error():
    assert compile_filter({"venue": {"$gt": 5}})(META) is False


def test_unknown_operator_raises():
    with pytest.raises(QueryError):
        compile_filter({"year": {"$bogus": 1}})(META)


def test_unknown_top_level_operator_raises():
    with pytest.raises(QueryError):
        compile_filter({"$xor": []})


def test_non_dict_node_raises():
    with pytest.raises(QueryError):
        compile_filter({"$and": ["nope"]})


def test_statistics_and_selectivity():
    filt = compile_filter({"$and": [{"venue": "NeurIPS"}, {"year": {"$gte": 2020}}]})
    assert filt.clauses == 2
    assert "venue" in filt.fields
    assert 0.0 < filt.estimated_selectivity() < 1.0
    assert compile_filter(None).is_trivial
    assert compile_filter(None).estimated_selectivity() == 1.0


def test_more_clauses_means_lower_selectivity():
    one = compile_filter({"venue": "NeurIPS"}).estimated_selectivity()
    two = compile_filter({"venue": "NeurIPS", "year": 2021}).estimated_selectivity()
    assert two < one
