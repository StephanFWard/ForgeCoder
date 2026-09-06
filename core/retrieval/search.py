"""Search orchestration: FTS5 + symbol lookup -> ranking -> top candidates."""
from __future__ import annotations

from dataclasses import dataclass

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
