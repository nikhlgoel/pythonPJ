"""A small, dependency-free analyzer.

Deliberately simple and *deterministic*: lowercase, unicode-aware word split,
optional stopword removal and a light suffix stemmer.  Determinism matters more
than linguistic sophistication here - the same analyzer must run at index time
and at query time, forever, or scores stop matching.
"""

from __future__ import annotations

import re

_WORD = re.compile(r"[\w']+", re.UNICODE)

DEFAULT_STOPWORDS = frozenset(
    ["a", "an", "and", "are", "as", "at", "be", "but", "by", "for", "from", "has", "have", "he", "her", "his", "i", "if", "in", "into", "is", "it", "its", "of", "on", "or", "she", "that", "the", "their", "them", "there", "these", "they", "this", "to", "was", "were", "will", "with", "you", "your"]
)

_SUFFIXES = ("ational", "iveness", "fulness", "ousness", "ization", "ation", "ingly",
             "edly", "ing", "ies", "ied", "ess", "ers", "er", "ly", "es", "s")


def stem(token: str) -> str:
    """Very light suffix stripper (a pragmatic subset of Porter step 1)."""
    if len(token) <= 3:
        return token
    for suf in _SUFFIXES:
        if token.endswith(suf) and len(token) - len(suf) >= 3:
            base = token[: -len(suf)]
            if suf in ("ies", "ied"):
                return base + "y"
            return base
    return token


class Tokenizer:
    """Configurable analyzer used by :class:`~annex.text.bm25.BM25Index`."""

    __slots__ = ("lowercase", "stopwords", "use_stemmer", "min_length")

    def __init__(
        self,
        *,
        lowercase: bool = True,
        stopwords: frozenset[str] | None = DEFAULT_STOPWORDS,
        use_stemmer: bool = True,
        min_length: int = 1,
    ) -> None:
        self.lowercase = lowercase
        self.stopwords = stopwords or frozenset()
        self.use_stemmer = use_stemmer
        self.min_length = min_length

    def __call__(self, text: str) -> list[str]:
        return self.tokenize(text)

    def tokenize(self, text: str) -> list[str]:
        if not text:
            return []
        raw = _WORD.findall(text.lower() if self.lowercase else text)
        out: list[str] = []
        for tok in raw:
            if len(tok) < self.min_length or tok in self.stopwords:
                continue
            out.append(stem(tok) if self.use_stemmer else tok)
        return out
