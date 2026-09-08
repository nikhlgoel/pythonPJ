# Contributing

```bash
make install     # .venv + editable install with all extras
make test        # pytest
make lint        # ruff
make type        # mypy
make bench       # recall / latency suite
```

## Ground rules

* **Every behavioural change needs a test.** The suite is the specification;
  coverage is enforced at 80% in CI and currently sits near 90%.
* **Correctness claims must be executable.** "HNSW recall is high" is a test that
  measures recall against exact ground truth, not a comment.
* **Keep the layers separate.** `index/` must not import `storage/`; `query/`
  must not know which index it is planning for. `db.py` is the only place the
  layers meet.
* **Docstrings explain *why*.** The *what* is readable from the code; the design
  reasoning is not.
* **No new runtime dependencies** in the core (`numpy` only). Server and CLI
  extras may add theirs.

## Where things live

| Path | Contents |
|---|---|
| `src/annex/index/` | HNSW, flat, PQ, vector spaces |
| `src/annex/storage/` | WAL, segments, LSM store, codec |
| `src/annex/query/` | Filter DSL, fusion, planner |
| `src/annex/text/` | Tokeniser, BM25 |
| `src/annex/cluster/` | Raft, simulator, state machine |
| `src/annex/server/` | FastAPI app and schemas |
| `src/annex/bench/` | Datasets and benchmark harness |
| `docs/` | Architecture, algorithms, storage format, benchmarks, roadmap |
