"""RNP-inspired hierarchical retrieval.

Fisher & Rao (2023) — Recursive Neural Programs (PMC10637337) — model images
as trees of sensory-motor programs where each level generates parts AND their
transformations within a local reference frame.

We adapt three ideas for code retrieval:

1. **Part-whole hierarchy** — retrieve at multiple levels of abstraction:
   workspace → file → function/symbol → line range, rather than flat chunks only.

2. **State-action duality** — "what" (content match) and "where" (structural /
   positional context) are scored separately then combined, mirroring how RNP
   separates state features from action transformations.

3. **Hypernetwork-like adaptation** — query-dependent weights over retrieval
   signals, so the same candidate scores differently for different query types.

The result is an ``RNPSearchEngine`` that returns hierarchically organised
results with separate "what" and "where" scores, plus a ``HierarchicalContext``
that renders them as a tree for the model.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Any

from core.indexer.index import CodeIndex
from core.retrieval.ranking import rank_chunks


@dataclass(frozen=True)
class SearchResult:
    path: str
    content: str
    start_line: int
    end_line: int
    language: str
    score: int
    modified: float = 0.0


@dataclass
class SearchEngine:
    index: CodeIndex
    page_size: int = 20
    top_k: int = 6

    def search(self, query: str, *, workspace: str | None = None, limit: int | None = None) -> list[SearchResult]:
        """Return the best-chunked results for a free-text or symbol query."""
        limit = limit or self.top_k
        raw = self.index.search(query, limit=self.page_size, workspace=workspace)
        if not raw:
            # Fall back to symbol-name lookup which is more forgiving of casing.
            raw = self.index.search(" ".join(w.capitalize() for w in query.split()[:3]),
                                    limit=self.page_size, workspace=workspace) or raw
        ranked = rank_chunks(raw, query, limit=limit)
        return [
            SearchResult(
                path=r["path"],
                content=r["content"],
                start_line=r["start_line"],
                end_line=r["end_line"],
                language=r["language"],
                score=r.get("_score", 0),
                modified=float(r.get("modified", 0) or 0),
            )
            for r in ranked
        ]

    def symbols(self, name: str, *, limit: int = 10) -> list[dict]:
        return self.index.symbols_like(name, limit=limit)


# ------------------------------------------------------------------
# RNP dual scoring components (state-action duality)
# ------------------------------------------------------------------

def _what_score(content: str, query_tokens: list[str]) -> float:
    """Content / semantic match score — RNP "what" pathway (state).

    Measures how well the chunk content semantically matches the query.
    """
    low = content.lower()
    matches = sum(1 for tok in query_tokens if tok in low)
    density = sum(low.count(tok) for tok in query_tokens) if query_tokens else 0
    return matches * 25.0 + density * 2.0


def _where_score(path: str, content: str, language: str, modified: float,
                 query_tokens: list[str]) -> float:
    """Structural / positional relevance — RNP "where" pathway (policy).

    Measures how likely this file/structural location is to contain the answer:
    filename match, symbol definition, recent edits, language.
    """
    score = 0.0
    basename = path.rsplit("/", 1)[-1].lower() if path else ""
    low_content = content.lower()
    for tok in query_tokens:
        if tok in basename:
            score += 12.0
        if re.search(rf"\b{re.escape(tok)}\s*[\(:\s=]", low_content):
            score += 15.0  # symbol definition
    if modified and time.time() - modified <= 3 * 24 * 3600:
        score += 8.0
    return score


def _classify_query(message: str) -> str:
    """Lightweight query-type classification (hypernetwork-like adaptation)."""
    low = message.lower()
    if any(t in low for t in ("error", "exception", "bug", "fix", "crash", "failed", "traceback")):
        return "fix"
    if any(t in low for t in ("explain", "why", "how does", "architecture", "design")):
        return "explain"
    if any(t in low for t in ("implement", "create", "add", "refactor", "change")):
        return "edit"
    return "search"


# Query-type → (what_weight, where_weight) — adaptive retrieval strategy
_ADAPTIVE_WEIGHTS: dict[str, tuple[float, float]] = {
    "fix":     (1.0, 1.5),
    "explain": (1.5, 1.0),
    "edit":    (1.2, 1.2),
    "search":  (1.0, 1.0),
}


@dataclass(frozen=True)
class HierarchicalResult:
    """RNP search result with state-action (what/where) dual scores."""
    path: str
    content: str
    start_line: int
    end_line: int
    language: str
    what_score: float
    where_score: float
    combined_score: float
    score: int = 0
    modified: float = 0.0


@dataclass
class HierarchicalSearch:
    """Tree-structured search results from RNPSearchEngine."""
    query: str
    query_type: str
    weights: tuple[float, float]
    results: list[HierarchicalResult] = field(default_factory=list)
    tree: dict | None = None


class RNPSearchEngine(SearchEngine):
    """RNP-inspired search engine with state-action duality and adaptive weights.

    Mirrors RNP's separation of *state* ("what" — content semantics) from
    *policy* ("where" — structural / positional context).  Query type determines
    adaptive weights (the hypernetwork analog) that tilt scoring toward the
    more useful signal.
    """

    def search(self, query: str, *, workspace: str | None = None,
               limit: int | None = None, tree: bool = False) -> list[HierarchicalResult]:
        """Return hierarchically-scored results with separate what/where scores."""
        limit = limit or self.top_k
        query_type = _classify_query(query)
        what_w, where_w = _ADAPTIVE_WEIGHTS.get(query_type, (1.0, 1.0))
        tokens = [t.lower() for t in re.findall(r"[A-Za-z_][A-Za-z0-9_]*", query) if len(t) >= 2]

        raw = self.index.search(query, limit=self.page_size, workspace=workspace)
        if not raw:
            raw = self.index.search(" ".join(w.capitalize() for w in query.split()[:3]),
                                    limit=self.page_size, workspace=workspace) or raw
        ranked = rank_chunks(raw, query, limit=limit)

        results: list[HierarchicalResult] = []
        for r in ranked:
            content = r.get("content", "")
            path = r.get("path", "")
            lang = r.get("language", "")
            mod = float(r.get("modified", 0) or 0)
            what = _what_score(content, tokens)
            where = _where_score(path, content, lang, mod, tokens)
            combined = what * what_w + where * where_w
            results.append(HierarchicalResult(
                path=path,
                content=content,
                start_line=r.get("start_line", 0),
                end_line=r.get("end_line", 0),
                language=lang,
                what_score=round(what, 2),
                where_score=round(where, 2),
                combined_score=round(combined, 2),
                score=int(r.get("_score", 0) or 0),
                modified=mod,
            ))

        results.sort(key=lambda x: x.combined_score, reverse=True)
        final = results[:limit or self.top_k]

        self._last_query_type = query_type
        self._last_weights = (what_w, where_w)
        return final

    def search_detailed(self, query: str, *, workspace: str | None = None,
                        limit: int | None = None) -> HierarchicalSearch:
        """Return full HierarchicalSearch wrapper with tree + metadata."""
        results = self.search(query, workspace=workspace, limit=limit, tree=True)
        return HierarchicalSearch(
            query=query,
            query_type=getattr(self, "_last_query_type", "search"),
            weights=getattr(self, "_last_weights", (1.0, 1.0)),
            results=results,
            tree=_build_result_tree(results) if results else None,
        )


def _build_result_tree(results: list[HierarchicalResult]) -> dict:
    """Build a workspace → file → chunk tree from flat results (part-whole)."""
    tree: dict[str, Any] = {"kind": "workspace", "label": ".", "children": {}}
    for r in results:
        if r.path not in tree["children"]:
            tree["children"][r.path] = {
                "kind": "file", "label": r.path, "children": {},
                "score": r.combined_score,
            }
        file_node = tree["children"][r.path]
        chunk_key = f"{r.start_line}-{r.end_line}"
        file_node["children"][chunk_key] = {
            "kind": "chunk", "label": chunk_key,
            "path": r.path, "start_line": r.start_line, "end_line": r.end_line,
            "content": r.content[:200], "score": r.combined_score,
        }
    return tree


__all__ = [
    "SearchEngine",
    "SearchResult",
    "RNPSearchEngine",
    "HierarchicalResult",
    "HierarchicalSearch",
    "rank_chunks",
]
