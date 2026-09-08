"""End-to-end retrieval-augmented-generation style pipeline on Annex.

Runs with no external services and no embedding model: documents are embedded
with a deterministic hashing vectoriser so the example is reproducible offline.
Swap ``HashingEmbedder`` for a sentence-transformer and the rest is unchanged.

    python examples/rag_pipeline.py
"""

from __future__ import annotations

import shutil
import tempfile
from dataclasses import dataclass

import numpy as np

import annex
from annex.text import Tokenizer
from annex.util.math import l2_normalize

CORPUS = [
    ("hnsw", "graph-index", 2016,
     "Hierarchical navigable small world graphs give logarithmic approximate "
     "nearest neighbour search by stacking navigable small world layers."),
    ("pq", "compression", 2011,
     "Product quantisation compresses vectors into short codes by quantising "
     "sub-spaces independently, enabling billion scale similarity search."),
    ("lsm", "storage", 1996,
     "The log structured merge tree turns random writes into sequential ones "
     "using a memtable, immutable segments and background compaction."),
    ("raft", "consensus", 2014,
     "Raft is a consensus algorithm for replicated logs designed to be "
     "understandable, with leader election and log matching."),
    ("bm25", "lexical", 1994,
     "BM25 ranks documents using term frequency saturation and document length "
     "normalisation over an inverted index."),
    ("mvcc", "storage", 1981,
     "Multiversion concurrency control lets readers see a consistent snapshot "
     "while writers create new versions instead of overwriting rows."),
    ("rrf", "fusion", 2009,
     "Reciprocal rank fusion combines ranked lists from different retrievers "
     "without score calibration and is robust to scale mismatch."),
    ("colbert", "reranking", 2020,
     "Late interaction retrieval keeps token level embeddings and scores query "
     "document pairs with a cheap max similarity operator."),
]


@dataclass
class HashingEmbedder:
    """Deterministic bag-of-words hashing vectoriser (stand-in for a real model)."""

    dim: int = 128
    tokenizer: Tokenizer = Tokenizer()

    def __call__(self, text: str) -> np.ndarray:
        vec = np.zeros(self.dim, dtype="float32")
        tokens = self.tokenizer.tokenize(text)
        for token in tokens:
            # two hashes -> fewer collisions, signed to keep the space isotropic
            h1 = hash(token) % self.dim
            h2 = hash(token + "#") % self.dim
            vec[h1] += 1.0
            vec[h2] += 0.5 if h2 % 2 else -0.5
        return l2_normalize(vec + 1e-4)


def build_prompt(question: str, hits) -> str:
    context = "\n".join(f"[{i}] ({h.id}) {h.text}" for i, h in enumerate(hits, 1))
    return (
        "Answer the question using only the numbered context.\n"
        f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"
    )


def main() -> None:
    workdir = tempfile.mkdtemp(prefix="annex-rag-")
    embed = HashingEmbedder(dim=128)

    db = annex.Database(workdir)
    kb = db.create_collection("knowledge", dim=embed.dim, metric="cosine", index="hnsw")

    print(f"indexing {len(CORPUS)} documents ...")
    with kb.transaction() as txn:
        for doc_id, topic, year, text in CORPUS:
            txn.upsert(doc_id, embed(text), {"topic": topic, "year": year}, text=text)

    question = "how do systems keep replicas of a log consistent?"
    q_vec = embed(question)

    print("\n--- dense only -------------------------------------------------")
    show(kb.search(q_vec, k=3))

    print("\n--- lexical only -----------------------------------------------")
    show(kb.search(text="consensus replicated log leader election", k=3))

    print("\n--- hybrid (RRF) -----------------------------------------------")
    hybrid = kb.search(q_vec, text="consensus replicated log", k=3)
    show(hybrid)

    print("\n--- hybrid + metadata filter (storage papers after 1990) -------")
    filtered = kb.search(
        embed("how is data written durably to disk?"),
        text="storage engine writes",
        k=3,
        filter={"$and": [{"topic": {"$in": ["storage", "compression"]}}, {"year": {"$gte": 1990}}]},
    )
    show(filtered)

    print("\n--- prompt handed to the generator ------------------------------")
    print(build_prompt(question, hybrid.hits))

    print("\n--- snapshot isolation ------------------------------------------")
    with kb.snapshot() as snap:
        kb.upsert("raft", embed("retracted"), {"topic": "consensus", "year": 2014}, text="retracted")
        old = kb.search(q_vec, k=1, snapshot=snap).hits[0]
        new = kb.search(q_vec, k=1).hits[0]
        print(f"  snapshot sees : {old.id} -> {old.text[:48]}")
        print(f"  live sees     : {new.id} -> {new.text[:48]}")
    print(f"  vacuum reclaimed {kb.vacuum()} dead version(s)")

    print("\n--- engine statistics -------------------------------------------")
    stats = kb.stats()
    print(f"  documents={stats['documents']}  versions={stats['versions']}")
    print(f"  index={stats['index']}")
    print(f"  storage={stats['storage']}")

    db.close()
    shutil.rmtree(workdir, ignore_errors=True)


def show(result) -> None:
    print(f"  plan: {result.plan.get('strategy')} - {result.plan.get('reason', '')}")
    for i, hit in enumerate(result.hits, 1):
        print(f"  {i}. {hit.score:6.3f}  {hit.id:<8} {hit.text[:64]}")


if __name__ == "__main__":
    main()
