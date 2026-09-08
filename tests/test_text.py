import pytest

from embra.text import BM25Index, Tokenizer
from embra.text.tokenizer import stem


def test_tokenizer_lowercases_and_splits():
    assert Tokenizer(use_stemmer=False, stopwords=None).tokenize("Hello, World!") == [
        "hello",
        "world",
    ]


def test_tokenizer_removes_stopwords():
    assert "the" not in Tokenizer().tokenize("the cat sat on the mat")


def test_tokenizer_is_deterministic():
    tok = Tokenizer()
    assert tok.tokenize("Running quickly") == tok.tokenize("Running quickly")


def test_stemmer_conflates_forms():
    assert stem("running") == stem("runner")[:len(stem("running"))] or stem("running") == "runn"
    assert stem("studies") == "study"
    assert stem("cat") == "cat"


def test_empty_text_is_empty():
    assert Tokenizer().tokenize("") == []


@pytest.fixture
def index():
    idx = BM25Index()
    idx.add(0, "the quick brown fox jumps over the lazy dog")
    idx.add(1, "vector databases index embeddings for similarity search")
    idx.add(2, "approximate nearest neighbour search with graph indexes")
    idx.add(3, "a quick brown dog")
    return idx


def test_search_ranks_relevant_documents_first(index):
    top = index.search("quick brown dog", k=2)
    assert {key for key, _ in top} == {0, 3}
    assert top[0][1] >= top[1][1]


def test_rarer_terms_score_higher(index):
    assert index.idf("embedding") > index.idf("search")


def test_unknown_term_returns_nothing(index):
    assert index.search("quantum chromodynamics") == []


def test_remove_document(index):
    index.remove(3)
    assert 3 not in {key for key, _ in index.search("quick brown dog", k=5)}
    assert index.num_docs == 3


def test_reindex_replaces_old_content(index):
    index.add(1, "completely different subject matter")
    assert index.search("embeddings", k=5) == []
    assert index.search("subject", k=5)[0][0] == 1


def test_allow_predicate_filters(index):
    hits = index.search("search", k=5, allow=lambda key: key == 2)
    assert [key for key, _ in hits] == [2]


def test_stats(index):
    stats = index.stats()
    assert stats["docs"] == 4
    assert stats["terms"] > 0
    assert stats["avg_doc_len"] > 0


def test_empty_index_returns_nothing():
    assert BM25Index().search("anything") == []
