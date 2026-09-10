"""RNP-inspired query decomposition — recursive reuse of retrieval primitives.

Fisher & Rao (2023, PMC10637337): RNP trees recursively compose the same
primitives at different levels of reference frames.  In retrieval, a complex
user query decomposes into simpler sub-queries that each invoke the same
search primitive, then results are merged and re-ranked — recursively.

Provides ``decompose_query`` (synchronous) and ``decompose_and_search``
(asynchronous, parallel retrieval per sub-query).
"""
from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass, field

from core.retrieval.adaptive import classify_query
from core.retrieval.search import HierarchicalResult, SearchEngine

_SENT_SPLIT = re.compile(
    r"(?<=[.!?])\s+|(?<=;)\s+|(?<=\band\b\s)|(?<=\bbut\b\s)",
    re.IGNORECASE,
)


def _split_clauses(message: str) -> list[str]:
    """Split a message into candidate sub-queries on punctuation / conjunctions."""
    raw = _SENT_SPLIT.split(message.strip())
    parts = [s.strip().rstrip(".;?").strip() for s in raw if s.strip()]
    return [p for p in parts if len(p) > 3 and p.split()[0].lower() not in ("and", "or", "but")]
    return [p for p in parts if len(p) > 3 and p.split()[0].lower() not in ("and", "or", "but")]


_FILENAME_RE = re.compile(r"\b[\w\-./\\]+\.(py|java|ts|tsx|js|jsx|go|rs|c|cpp|h|cs|sql|yaml|yml|json|md|toml)\b", re.IGNORECASE)
_SYMBOL_RE = re.compile(r"\b(get_?[\w]+|set_?[\w]+|is_?[\w]+|has_?[\w]+|find_?[\w]+|search_?[\w]+|create_?[\w]+|delete_?[\w]+|update_?[\w]+|validate_?[\w]+|parse_?[\w]+|load_?[\w]+|save_?[\w]+|handle_?[\w]+|process_?[\w]+)\b", re.IGNORECASE)


def _infer_scope(clause: str) -> str:
    """Infer a scope hint from a clause (RNP reference-frame narrowing).

    Scope mirrors the part-whole hierarchy: ``all`` (workspace reference
    frame) → ``file:...`` (file frame) → ``symbol:...`` (symbol frame).
    """
    m = _FILENAME_RE.search(clause)
    if m:
        return f"file:{m.group(0)}"
    s = _SYMBOL_RE.search(clause)
    if s:
        return f"symbol:{s.group(0).lower()}"
    return "all"


@dataclass
class SubQuery:
    query: str
    intent: str = "search"
    scope: str = "all"
    depth: int = 0
    results: list[HierarchicalResult] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {"query": self.query, "intent": self.intent, "scope": self.scope,
                "depth": self.depth, "result_count": len(self.results)}


@dataclass
class DecomposedQuery:
    original: str
    subqueries: list[SubQuery] = field(default_factory=list)
    merged_results: list[HierarchicalResult] = field(default_factory=list)

    @property
    def needs_decomposition(self) -> bool:
        return len(self.subqueries) > 1




def decompose_query(message: str, max_depth: int = 3) -> DecomposedQuery:
    """Split ``message`` into classified sub-queries (no retrieval — pure planning)."""
    dq = DecomposedQuery(original=message)
    clauses = _split_clauses(message)

    if len(clauses) <= 1:
        dq.subqueries.append(SubQuery(
            query=message.strip(),
            intent=classify_query(message),
            scope=_infer_scope(message),
            depth=0,
        ))
    else:
        for clause in clauses[:6]:
            dq.subqueries.append(SubQuery(
                query=clause,
                intent=classify_query(clause),
                scope=_infer_scope(clause),
                depth=1,
            ))
    return dq


async def decompose_and_search(
    search_engine: SearchEngine,
    message: str,
    *,
    workspace: str | None = None,
    max_depth: int = 3,
    max_results_per_subq: int = 6,
) -> DecomposedQuery:
    """Decompose a complex query and search each sub-query in parallel (RNP recursion)."""
    dq = decompose_query(message, max_depth=max_depth)

    if not dq.needs_decomposition:
        sq = dq.subqueries[0]
        sq.results = search_engine.search(
            sq.query, workspace=workspace, limit=max_results_per_subq
        )
        dq.merged_results = sq.results
        return dq

    # Complex: parallel search per sub-query
    async def _do_subq(sq: SubQuery) -> SubQuery:
        sq.results = search_engine.search(
            sq.query, workspace=workspace, limit=max_results_per_subq
        )
        return sq

    tasks = [_do_subq(sq) for sq in dq.subqueries]
    done = await asyncio.gather(*tasks, return_exceptions=True)

    all_results: list[HierarchicalResult] = []
    for i, outcome in enumerate(done):
        if isinstance(outcome, Exception):
            continue
        sq = dq.subqueries[i]
        for r in sq.results:
            key = (r.path, r.start_line, r.end_line)
            if not any((e.path, e.start_line, e.end_line) == key for e in all_results):
                all_results.append(r)

    dq.merged_results = sorted(all_results, key=lambda r: r.combined_score, reverse=True)
    dq.merged_results = dq.merged_results[:max_results_per_subq]
    return dq


__all__ = ["SubQuery", "DecomposedQuery", "decompose_query", "decompose_and_search"]
