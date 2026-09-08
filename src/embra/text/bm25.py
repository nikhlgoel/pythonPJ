"""Incremental BM25 over an inverted index.

Scoring (Robertson/Sparck-Jones with the Lucene-style IDF floor)::

    idf(q)   = ln(1 + (N - df + 0.5) / (df + 0.5))
    score(d) = sum_q idf(q) * (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * |d| / avgdl))

Unlike a static BM25 build, this index supports deletes and updates: postings
carry the document key, ``df`` is maintained incrementally, and removal rewrites
only the affected postings lists.
"""

from __future__ import annotations

import math
from collections import defaultdict

from .tokenizer import Tokenizer


class BM25Index:
    """Inverted index with incremental add/remove and top-k retrieval."""

    def __init__(
        self,
        tokenizer: Tokenizer | None = None,
        *,
        k1: float = 1.2,
        b: float = 0.75,
    ) -> None:
        self.tokenizer = tokenizer or Tokenizer()
        self.k1 = k1
        self.b = b
        self._postings: dict[str, dict[int, int]] = defaultdict(dict)
        self._doc_len: dict[int, int] = {}
        self._doc_terms: dict[int, tuple[str, ...]] = {}
        self._total_len = 0

    # ------------------------------------------------------------------
    @property
    def num_docs(self) -> int:
        return len(self._doc_len)

    @property
    def avg_doc_len(self) -> float:
        return self._total_len / self.num_docs if self.num_docs else 0.0

    def __contains__(self, key: object) -> bool:
        return key in self._doc_len

    def add(self, key: int, text: str) -> None:
        """Index (or re-index) the document stored under internal ``key``."""
        if key in self._doc_len:
            self.remove(key)
        tokens = self.tokenizer.tokenize(text)
        if not tokens:
            self._doc_len[key] = 0
            self._doc_terms[key] = ()
            return
        tf: dict[str, int] = defaultdict(int)
        for tok in tokens:
            tf[tok] += 1
        for term, count in tf.items():
            self._postings[term][key] = count
        self._doc_len[key] = len(tokens)
        self._doc_terms[key] = tuple(tf)
        self._total_len += len(tokens)

    def remove(self, key: int) -> None:
        length = self._doc_len.pop(key, None)
        if length is None:
            return
        self._total_len -= length
        for term in self._doc_terms.pop(key, ()):  # noqa: B007
            postings = self._postings.get(term)
            if postings is not None:
                postings.pop(key, None)
                if not postings:
                    del self._postings[term]

    # ------------------------------------------------------------------
    def idf(self, term: str) -> float:
        df = len(self._postings.get(term, ()))
        if df == 0:
            return 0.0
        return math.log(1.0 + (self.num_docs - df + 0.5) / (df + 0.5))

    def search(
        self,
        query: str,
        k: int = 10,
        *,
        allow=None,
    ) -> list[tuple[int, float]]:
        """Return the top-``k`` ``(key, score)`` pairs, highest score first."""
        terms = self.tokenizer.tokenize(query)
        if not terms or self.num_docs == 0:
            return []
        avgdl = self.avg_doc_len or 1.0
        scores: dict[int, float] = defaultdict(float)
        for term in terms:
            postings = self._postings.get(term)
            if not postings:
                continue
            idf = self.idf(term)
            for key, tf in postings.items():
                if allow is not None and not allow(key):
                    continue
                dl = self._doc_len.get(key, 0)
                denom = tf + self.k1 * (1.0 - self.b + self.b * dl / avgdl)
                scores[key] += idf * (tf * (self.k1 + 1.0)) / denom
        return sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))[:k]

    def stats(self) -> dict:
        return {
            "docs": self.num_docs,
            "terms": len(self._postings),
            "postings": sum(len(p) for p in self._postings.values()),
            "avg_doc_len": round(self.avg_doc_len, 2),
        }
