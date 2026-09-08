"""A MongoDB-flavoured metadata filter DSL, compiled to a Python closure.

Example::

    {"$and": [{"year": {"$gte": 2020}},
              {"tags": {"$contains": "ml"}},
              {"$not": {"status": "draft"}}]}

Compilation happens once per query; evaluation is then a plain function call per
candidate, which keeps the predicate cheap enough to run *inside* the HNSW
traversal rather than as a post-filter.

The compiler also reports **selectivity hints** (how many fields are
constrained, whether an equality on an indexed field exists) which the planner
uses to choose between pre-filtering and graph traversal.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from ..errors import QueryError
from ..types import Metadata

Predicate = Callable[[Metadata], bool]

_MISSING = object()


def _get(meta: Metadata, path: str) -> Any:
    """Resolve a dotted path such as ``author.name``."""
    cur: Any = meta
    for part in path.split("."):
        if isinstance(cur, dict) and part in cur:
            cur = cur[part]
        else:
            return _MISSING
    return cur


def _cmp(op: str, left: Any, right: Any) -> bool:
    if op == "$exists":
        return bool(right) is (left is not _MISSING)
    if left is _MISSING:
        return op in ("$ne", "$nin")
    try:
        if op == "$eq":
            return bool(left == right)
        if op == "$ne":
            return bool(left != right)
        if op == "$gt":
            return left > right
        if op == "$gte":
            return left >= right
        if op == "$lt":
            return left < right
        if op == "$lte":
            return left <= right
        if op == "$in":
            return left in right
        if op == "$nin":
            return left not in right
        if op == "$contains":
            return right in left
        if op == "$regex":
            return re.search(right, str(left)) is not None
        if op == "$between":
            lo, hi = right
            return lo <= left <= hi
    except TypeError:
        return False
    raise QueryError(f"unsupported filter operator {op!r}")


@dataclass(slots=True)
class Filter:
    """A compiled filter plus the statistics the planner cares about."""

    predicate: Predicate
    fields: frozenset[str] = field(default_factory=frozenset)
    equality_fields: frozenset[str] = field(default_factory=frozenset)
    clauses: int = 0

    def __call__(self, metadata: Metadata) -> bool:
        return self.predicate(metadata)

    @property
    def is_trivial(self) -> bool:
        return self.clauses == 0

    def estimated_selectivity(self) -> float:
        """Crude prior: each equality clause cuts the candidate set by ~10x,
        each range clause by ~3x.  Clamped into ``[0.001, 1.0]``."""
        eq = len(self.equality_fields)
        other = max(self.clauses - eq, 0)
        return max(0.001, min(1.0, (0.1**eq) * (0.33**other)))


def _always_true(_meta: Metadata) -> bool:
    return True


def _all_of(subs: list[Predicate]) -> Predicate:
    def run(meta: Metadata) -> bool:
        return all(sub(meta) for sub in subs)

    return run


def _any_of(subs: list[Predicate]) -> Predicate:
    def run(meta: Metadata) -> bool:
        return any(sub(meta) for sub in subs)

    return run


def _negate(sub: Predicate) -> Predicate:
    def run(meta: Metadata) -> bool:
        return not sub(meta)

    return run


def _ops_on(field: str, ops: list[tuple[str, Any]]) -> Predicate:
    def run(meta: Metadata) -> bool:
        value = _get(meta, field)
        return all(_cmp(op, value, rhs) for op, rhs in ops)

    return run


def compile_filter(spec: dict | None) -> Filter:
    """Compile a filter specification into a :class:`Filter`."""
    if not spec:
        return Filter(_always_true)
    fields: set[str] = set()
    eq_fields: set[str] = set()
    clauses = [0]

    def build(node: Any) -> Predicate:  # noqa: C901 - a compiler is a dispatch table
        if not isinstance(node, dict):
            raise QueryError(f"filter node must be a dict, got {type(node).__name__}")
        parts: list[Predicate] = []
        for key, value in node.items():
            if key == "$and":
                parts.append(_all_of([build(v) for v in value]))
            elif key == "$or":
                parts.append(_any_of([build(v) for v in value]))
            elif key == "$not":
                parts.append(_negate(build(value)))
            elif key.startswith("$"):
                raise QueryError(f"unknown top-level operator {key!r}")
            else:
                fields.add(key)
                clauses[0] += 1
                if isinstance(value, dict) and any(k.startswith("$") for k in value):
                    ops = list(value.items())
                    for op, _ in ops:
                        if op == "$eq":
                            eq_fields.add(key)
                    parts.append(_ops_on(key, ops))
                else:
                    eq_fields.add(key)
                    parts.append(_ops_on(key, [("$eq", value)]))
        if not parts:
            return _always_true
        if len(parts) == 1:
            return parts[0]
        return _all_of(parts)

    predicate = build(spec)
    return Filter(predicate, frozenset(fields), frozenset(eq_fields), clauses[0])
