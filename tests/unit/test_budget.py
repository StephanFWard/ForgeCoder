"""Tests for token budgeting."""
from core.retrieval.budget import estimate_tokens, fit_to_budget, truncate_to_tokens


def test_estimate_scales_with_length():
    assert estimate_tokens("") == 0
    short = estimate_tokens("hello")
    long_text = estimate_tokens("hello " * 100)
    assert long_text > short
    assert estimate_tokens("x" * 13) == 4  # ceil(13/4)


def test_truncate_short_text_is_passthrough():
    text = "short snippet"
    assert truncate_to_tokens(text, 10_000) == text


def test_truncate_long_text_marks_cut():
    text = "".join(f"line {i}\n" for i in range(500))
    cut = truncate_to_tokens(text, 20)
    assert len(cut) < len(text)
    assert "truncated" in cut


def test_fit_to_budget_greedy():
    chunks = [
        {"content": "word " * 40},   # ~40 tokens
        {"content": "word " * 400},  # ~400 tokens
        {"content": "word " * 40},
    ]
    picked = fit_to_budget(chunks, 100, min_chunks=1)
    # The first chunk fits; the second is partially/not included.
    assert len(picked) >= 1
    for candidate in picked:
        assert candidate["content"] in ("word " * 40, "word " * 400) or "truncated" in candidate["content"]


def test_min_chunks_keeps_at_least_one():
    big_only = [{"content": "x" * 4000}]  # ~1000 tokens
    picked = fit_to_budget(big_only, 50, min_chunks=1)
    assert len(picked) == 1
