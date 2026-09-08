"""Hierarchical Navigable Small World graph - implemented from scratch.

This is a faithful implementation of Malkov & Yashunin (2016), including the
parts most toy implementations skip:

* exponentially decaying **layer assignment** (``floor(-ln(U) * mL)``), which
  gives the skip-list-like logarithmic search;
* the **neighbour selection heuristic** (Algorithm 4) rather than plain
  top-``M`` selection - this is what keeps the graph navigable in clustered
  data instead of collapsing into disconnected hubs;
* **bidirectional link pruning** with ``M_max0 = 2M`` on the base layer;
* **batched distance evaluation**: every graph expansion computes the distances
  to all unvisited neighbours in a single vectorised call, which is what makes
  a pure-Python graph walk competitive;
* **tombstoned deletes** with predicate-filtered traversal, so metadata filters
  are applied *inside* the search rather than as a lossy post-filter.

Complexity is ``O(log N)`` expected hops per query with ``O(M)`` work per hop.
"""

from __future__ import annotations

import heapq
import math
import random

import numpy as np

from ..config import HNSWConfig
from ..errors import IndexError_
from ..types import Metric
from .base import Predicate, VectorIndex
from .space import PQSpace, VectorSpace


class HNSWIndex(VectorIndex):
    """Graph-based ANN index over an arbitrary :class:`VectorSpace`."""

    def __init__(
        self,
        space: VectorSpace,
        config: HNSWConfig | None = None,
        *,
        rerank_factor: int = 3,
    ) -> None:
        self.space = space
        self.cfg = config or HNSWConfig()
        self.metric: Metric = space.metric
        self.rerank_factor = max(1, rerank_factor)

        self._levels: dict[int, int] = {}
        self._links: list[dict[int, list[int]]] = []
        self._deleted: set[int] = set()
        self._entry: int | None = None
        self._max_level: int = -1
        self._rng = random.Random(self.cfg.seed)
        self._mL = self.cfg.level_multiplier or (1.0 / math.log(max(self.cfg.m, 2)))
        self._distance_calls = 0

    # ------------------------------------------------------------------
    # properties
    # ------------------------------------------------------------------
    @property
    def size(self) -> int:
        return len(self._levels) - len(self._deleted)

    @property
    def entry_point(self) -> int | None:
        return self._entry

    @property
    def max_level(self) -> int:
        return self._max_level

    def levels_of(self, key: int) -> int:
        return self._levels[key]

    def neighbors(self, key: int, level: int = 0) -> list[int]:
        if level >= len(self._links):
            return []
        return list(self._links[level].get(key, ()))

    # ------------------------------------------------------------------
    # construction
    # ------------------------------------------------------------------
    def _random_level(self) -> int:
        return int(-math.log(max(self._rng.random(), 1e-12)) * self._mL)

    def _ensure_levels(self, level: int) -> None:
        while len(self._links) <= level:
            self._links.append({})

    def add(self, key: int, vector: np.ndarray) -> None:
        vector = np.ascontiguousarray(vector, dtype=np.float32)
        if key in self._levels:
            self._unlink(key)
        self.space.set(key, vector)
        self._deleted.discard(key)

        level = self._random_level()
        self._levels[key] = level
        self._ensure_levels(level)
        for lv in range(level + 1):
            self._links[lv].setdefault(key, [])

        if self._entry is None:
            self._entry = key
            self._max_level = level
            return

        ctx = self.space.query_context(vector)
        ep = self._entry
        ep_dist = float(self.space.distances_ctx(ctx, np.array([ep], dtype=np.int64))[0])

        # Phase 1: zoom in through the layers above this node's top level.
        for lv in range(self._max_level, level, -1):
            ep, ep_dist = self._greedy_descend(ctx, ep, ep_dist, lv)

        # Phase 2: insert with a beam search on every layer we belong to.
        m_at = lambda lv: self.cfg.m_max0 if lv == 0 else self.cfg.m  # noqa: E731
        for lv in range(min(level, self._max_level), -1, -1):
            candidates = self._search_layer(ctx, [(ep_dist, ep)], self.cfg.ef_construction, lv)
            selected = self._select_neighbors(key, candidates, self.cfg.m)
            self._links[lv][key] = list(selected)
            for nb in selected:
                nb_links = self._links[lv].setdefault(nb, [])
                if key not in nb_links:
                    nb_links.append(key)
                if len(nb_links) > m_at(lv):
                    pruned = self._select_neighbors(
                        nb,
                        [(self._dist_pair(nb, o), o) for o in nb_links],
                        m_at(lv),
                    )
                    self._links[lv][nb] = list(pruned)
            if candidates:
                ep_dist, ep = min(candidates)

        if level > self._max_level:
            self._max_level = level
            self._entry = key

    def _unlink(self, key: int) -> None:
        """Detach ``key`` from the graph entirely (used on overwrite)."""
        top = self._levels.pop(key, None)
        if top is None:
            return
        for lv in range(min(top, len(self._links) - 1) + 1):
            layer = self._links[lv]
            for nb in layer.pop(key, []):
                lst = layer.get(nb)
                if lst and key in lst:
                    lst.remove(key)
        if self._entry == key:
            self._entry = None
            self._max_level = -1
            for cand, lvl in self._levels.items():
                if self._entry is None or lvl > self._max_level:
                    self._entry, self._max_level = cand, lvl

    def remove(self, key: int) -> None:
        if key not in self._levels:
            raise IndexError_(f"unknown key {key}")
        self._deleted.add(key)

    # ------------------------------------------------------------------
    # search
    # ------------------------------------------------------------------
    def _dist_pair(self, a: int, b: int) -> float:
        self._distance_calls += 1
        return self.space.distance_between(a, b)

    def _greedy_descend(
        self, ctx, ep: int, ep_dist: float, level: int
    ) -> tuple[int, float]:
        """Single-entry greedy walk used on the upper (sparse) layers."""
        layer = self._links[level]
        improved = True
        while improved:
            improved = False
            nbs = layer.get(ep)
            if not nbs:
                break
            keys = np.fromiter(nbs, dtype=np.int64, count=len(nbs))
            dists = self.space.distances_ctx(ctx, keys)
            self._distance_calls += len(keys)
            j = int(np.argmin(dists))
            if float(dists[j]) < ep_dist:
                ep_dist = float(dists[j])
                ep = int(keys[j])
                improved = True
        return ep, ep_dist

    def _search_layer(
        self,
        ctx,
        entry_points: list[tuple[float, int]],
        ef: int,
        level: int,
        allow: Predicate | None = None,
        skip_deleted: bool = False,
    ) -> list[tuple[float, int]]:
        """Beam search on one layer.

        ``candidates`` is a min-heap on distance (frontier), ``results`` a
        max-heap (stored negated) capped at ``ef``.  Filtering is applied to the
        *result set only* - deleted or filtered-out nodes are still traversed so
        the graph never becomes disconnected under selective filters.
        """
        layer = self._links[level]
        visited: set[int] = {k for _, k in entry_points}
        candidates: list[tuple[float, int]] = list(entry_points)
        heapq.heapify(candidates)
        results: list[tuple[float, int]] = []
        for d, k in entry_points:
            if self._acceptable(k, allow, skip_deleted):
                heapq.heappush(results, (-d, k))
        while len(results) > ef:
            heapq.heappop(results)

        while candidates:
            dist, node = heapq.heappop(candidates)
            if results and len(results) >= ef and dist > -results[0][0]:
                break
            nbs = layer.get(node)
            if not nbs:
                continue
            fresh = [n for n in nbs if n not in visited]
            if not fresh:
                continue
            visited.update(fresh)
            keys = np.fromiter(fresh, dtype=np.int64, count=len(fresh))
            dists = self.space.distances_ctx(ctx, keys)
            self._distance_calls += len(keys)
            worst = -results[0][0] if results else math.inf
            for kk, dd in zip(keys.tolist(), dists.tolist(), strict=True):
                if len(results) >= ef and dd >= worst:
                    continue
                heapq.heappush(candidates, (dd, kk))
                if self._acceptable(kk, allow, skip_deleted):
                    heapq.heappush(results, (-dd, kk))
                    if len(results) > ef:
                        heapq.heappop(results)
                    worst = -results[0][0]
        return [(-d, k) for d, k in results]

    def _acceptable(self, key: int, allow: Predicate | None, skip_deleted: bool) -> bool:
        if skip_deleted and key in self._deleted:
            return False
        return not (allow is not None and not allow(key))

    def _select_neighbors(
        self, base: int, candidates: list[tuple[float, int]], m: int
    ) -> list[int]:
        """Algorithm 4: keep a candidate only if it is closer to the base node
        than to any already-selected neighbour.

        This *diversifies* the neighbourhood: instead of ``M`` mutually close
        points on one side of the base node, we keep links pointing in different
        directions, which is what preserves long-range navigability.
        """
        pool = sorted((d, k) for d, k in candidates if k != base)
        selected: list[int] = []
        discarded: list[int] = []
        for dist, cand in pool:
            if len(selected) >= m:
                break
            keep = True
            for chosen in selected:
                if self._dist_pair(cand, chosen) < dist:
                    keep = False
                    break
            if keep:
                selected.append(cand)
            else:
                discarded.append(cand)
        # keepPrunedConnections: top up with the best rejects to hit degree M
        for cand in discarded:
            if len(selected) >= m:
                break
            selected.append(cand)
        return selected

    def search(
        self,
        query: np.ndarray,
        k: int,
        *,
        ef: int | None = None,
        allow: Predicate | None = None,
    ) -> list[tuple[int, float]]:
        if k <= 0:
            return []
        if self._entry is None:
            return []
        query = np.ascontiguousarray(query, dtype=np.float32)
        ef = max(ef or self.cfg.ef_search, k)
        ctx = self.space.query_context(query)

        ep = self._entry
        ep_dist = float(self.space.distances_ctx(ctx, np.array([ep], dtype=np.int64))[0])
        for lv in range(self._max_level, 0, -1):
            ep, ep_dist = self._greedy_descend(ctx, ep, ep_dist, lv)

        fetch = k * self.rerank_factor if isinstance(self.space, PQSpace) else k
        found = self._search_layer(
            ctx, [(ep_dist, ep)], max(ef, fetch), 0, allow=allow, skip_deleted=True
        )
        found.sort()
        found = found[: max(fetch, k)]

        if isinstance(self.space, PQSpace) and found:
            keys = np.array([kk for _, kk in found], dtype=np.int64)
            exact = self.space.exact_distances(query, keys)
            order = np.argsort(exact, kind="stable")[:k]
            return [(int(keys[i]), float(exact[i])) for i in order]
        return [(kk, float(d)) for d, kk in found[:k]]

    # ------------------------------------------------------------------
    def stats(self) -> dict:
        degrees = [len(v) for v in self._links[0].values()] if self._links else []
        return {
            "type": "hnsw",
            "size": self.size,
            "nodes": len(self._levels),
            "deleted": len(self._deleted),
            "levels": self._max_level + 1,
            "avg_degree_l0": (sum(degrees) / len(degrees)) if degrees else 0.0,
            "distance_calls": self._distance_calls,
            "memory_bytes": self.space.memory_bytes(),
            "ef_search": self.cfg.ef_search,
            "m": self.cfg.m,
        }
