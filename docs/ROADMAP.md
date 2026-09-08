# Roadmap

Where this goes next, ordered by leverage. Each item names the engineering work
and the reason it matters commercially.

## Near term — performance

1. **Native distance kernels.** Move the layer-0 beam search into Cython/Rust
   with explicit SIMD. The graph structure stays; only the inner loop moves.
   Expected 20–50× on query latency, which is the single biggest gap to a
   production engine.
2. **`mmap`-backed segments.** Today segments are read fully into RAM at open.
   Memory-mapping them makes startup `O(1)` and lets a collection exceed RAM.
3. **Concurrent readers with a real epoch GC.** Snapshot visibility is already
   lock-free; vacuum still takes the writer lock.
4. **Filter-aware graph entry points.** Maintain per-field posting lists so a
   selective filter can seed the traversal from inside the surviving set instead
   of pre-filtering and scanning.

## Medium term — capability

5. **Histogram statistics for the planner.** Replace the fixed selectivity prior
   with per-field histograms/HLL sketches collected during flush. The planner's
   cost model is already the right shape; only the estimate is coarse.
6. **IVF-PQ and hybrid coarse quantisation** for billion-scale collections where
   a graph does not fit in memory.
7. **Live HTTP Raft transport.** The consensus core is implemented and tested
   deterministically; binding it to a real multi-process cluster is mostly
   plumbing plus snapshot/install-snapshot support for a lagging follower.
8. **Sharding by key range** with the planner scattering and gathering.
9. **Late-interaction (ColBERT-style) reranking** as an optional third fusion
   stage.

## Longer term — product

10. **Managed RAG service.** The engine already exposes the pieces a retrieval
    service needs: hybrid search, metadata filtering, snapshots for reproducible
    evaluation, and an inspectable query plan. The missing pieces are embedding
    ingestion pipelines, multi-tenancy and usage metering.
11. **Evaluation harness as a product surface.** Snapshots make "which index
    configuration would have answered last week's queries better?" a
    *reproducible* question. Most vector databases cannot answer it at all.
12. **Compliance-grade deletes.** Tombstones plus a scheduled full compaction
    already give a defensible "the bytes are gone" story; it needs an audit log
    and a certificate of erasure to be sellable into regulated industries.

## Deliberately out of scope

* Being faster than FAISS in pure Python. The artefact here is the design and its
  testability, not the constant factor.
* An embedding model. Annex indexes vectors; producing them is someone else's
  job.
