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


def test_active_selection_wins_over_stale_index_under_budget(tmp_path):
    from core.retrieval.budget import estimate_tokens
    from core.retrieval.context import load_prompt

    (tmp_path / "active.py").write_text(
        "\n".join(f"value_{i} = {i}" for i in range(1, 201)), encoding="utf-8",
    )
    results = [_result("active.py", 180, 182, "STALE INDEX CONTENT")]
    results += [_result(f"other{i}.py", 1, 100, (f"other_{i} = 1\n" * 100)) for i in range(20)]
    budget = estimate_tokens(load_prompt("edit")) + 400
    built = ContextBuilder(search_engine=_StubEngine(results)).build(
        "Update selected values", workspace=str(tmp_path), file="active.py",
        selection=(180, 182), budget=budget, behavior="edit",
    )
    assert built.sections[0]["type"] == "file"
    text = "\n\n".join(s["text"] for s in built.sections)
    for i in range(180, 183):
        assert f"{i}: value_{i} = {i}" in text
    assert "183: value_183" not in text
    assert "STALE INDEX CONTENT" not in text
    assert built.total_tokens <= budget


def test_line_fitting_never_splits_source():
    from core.retrieval.context import _fit_lines

    assert _fit_lines("short\n" + "x" * 100, 10) == "short"
    assert _fit_lines("source", 0) == ""
