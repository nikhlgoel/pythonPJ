"""Cost-based query planner.

Most vector databases hard-code "always use the graph".  That is the wrong call
in two common regimes:

* **Tiny collections.** Below a few thousand vectors a single BLAS-backed scan
  beats a Python graph walk outright.
* **Highly selective filters.** If a filter keeps 0.5% of the corpus, the graph
  wastes almost all of its distance computations on rejected nodes, and a
  pre-filtered exact scan is both faster *and* exact.

So Embra estimates the cost of each strategy and picks one, then reports the
decision back to the caller in ``SearchResult.plan`` - the query plan is a
first-class, inspectable object, like ``EXPLAIN`` in a relational database.

Cost units are "distance-computation equivalents".
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from .filter import Filter


class Strategy(str, Enum):
    EXACT_SCAN = "exact_scan"
    ANN_GRAPH = "ann_graph"
    HYBRID = "hybrid"
    LEXICAL_ONLY = "lexical_only"


@dataclass(slots=True)
class PlannedQuery:
    strategy: Strategy
    k: int
    ef: int
    fetch_k: int
    use_filter: bool
    explain: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "strategy": self.strategy.value,
            "k": self.k,
            "ef": self.ef,
            "fetch_k": self.fetch_k,
            "use_filter": self.use_filter,
            **self.explain,
        }


@dataclass(slots=True)
class QueryPlanner:
    """Chooses an execution strategy from collection statistics."""

    #: below this many live vectors an exact scan always wins
    small_collection_threshold: int = 2_000
    #: graph traversal constant: distances evaluated ~= graph_factor * ef * log2(n)
    graph_factor: float = 1.4
    #: a scan is this many times cheaper per candidate (vectorised, cache friendly)
    scan_speedup: float = 12.0

    def plan(
        self,
        *,
        k: int,
        live: int,
        filt: Filter,
        has_vector: bool,
        has_text: bool,
        ef: int | None,
        default_ef: int,
        index_kind: str,
        oversample: float = 1.0,
    ) -> PlannedQuery:
        if not has_vector and has_text:
            return PlannedQuery(
                Strategy.LEXICAL_ONLY, k, 0, k, not filt.is_trivial,
                {"reason": "no query vector supplied", "live": live},
            )

        selectivity = filt.estimated_selectivity()
        survivors = max(1.0, live * selectivity)
        ef_eff = max(ef or default_ef, k)
        fetch_k = max(k, int(math.ceil(k * oversample)))

        scan_cost = survivors / self.scan_speedup
        graph_cost = self.graph_factor * ef_eff * math.log2(max(live, 2))
        # a filter the graph cannot exploit inflates its cost by 1/selectivity,
        # because most visited nodes are rejected before entering the result heap
        if not filt.is_trivial:
            graph_cost /= max(selectivity, 0.02)

        exact = (
            index_kind == "flat"
            or live <= self.small_collection_threshold
            or scan_cost <= graph_cost
        )
        explain: dict[str, Any] = {
            "live": live,
            "selectivity": round(selectivity, 4),
            "estimated_survivors": int(survivors),
            "scan_cost": round(scan_cost, 1),
            "graph_cost": round(graph_cost, 1),
            "index": index_kind,
        }
        if has_text:
            return PlannedQuery(
                Strategy.HYBRID, k, ef_eff, fetch_k, not filt.is_trivial,
                {**explain, "reason": "vector + text query -> RRF hybrid",
                 "vector_stage": "exact_scan" if exact else "ann_graph"},
            )
        if exact:
            reason = (
                "flat index" if index_kind == "flat"
                else "collection below small-collection threshold"
                if live <= self.small_collection_threshold
                else "selective filter makes a pre-filtered scan cheaper"
            )
            return PlannedQuery(
                Strategy.EXACT_SCAN, k, 0, fetch_k, not filt.is_trivial,
                {**explain, "reason": reason},
            )
        return PlannedQuery(
            Strategy.ANN_GRAPH, k, ef_eff, fetch_k, not filt.is_trivial,
            {**explain, "reason": "graph traversal cheaper than a scan"},
        )
