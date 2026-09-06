"""Tests for repository context assembly (batch chunking + dedupe)."""
from core.retrieval.context import ContextBuilder
from core.retrieval.search import SearchResult


class _StubEngine:
    """Returns ranked results, including duplicates, like the real pipeline."""

    def __init__(self, results: list[SearchResult]):
        self._results = results

    def search(self, query: str, *, workspace: str | None = None, limit: int | None = None):
        return self._results[: limit or len(self._results)]


def _result(path: str, start: int, end: int, content: str) -> SearchResult:
    return SearchResult(path=path, content=content, start_line=start, end_line=end,
                        language="python", score=10)


def test_retrieve_dedupes_ranges_and_content():
    dup = _result("a.py", 1, 5, "def one():\n    return 1\n")
    engine = _StubEngine([
        dup,
        _result("a.py", 1, 5, "def one():\n    return 1\n"),          # same range
        _result("b.py", 10, 12, "def one():\n    return 1\n"),        # same content
        _result("c.py", 20, 22, "x = 3\n"),
    ])
    builder = ContextBuilder(search_engine=engine)
    pool = builder._retrieve("question")
    assert len(pool) == 2
    assert {(p["path"]) for p in pool} == {"a.py", "c.py"}


def test_build_appends_chunks_within_budget():
    results = [_result(f"f{i}.py", 1, 3, f"value_{i} = {i}\n") for i in range(10)]
    builder = ContextBuilder(search_engine=_StubEngine(results))
    built = builder.build("explain these values", budget=4096)
    repo = next(s for s in built.sections if s["type"] == "repository")
    # All 10 small chunks should be appended, not just the first 6.
    assert repo["text"].count("f") >= 10
    assert built.total_tokens < 4096
