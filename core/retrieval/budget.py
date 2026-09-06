"""Token budget helpers.

We do not run a tokenizer on the CPU runtime (that would violate the
single-model, no-extra-ML memory rule). For planning purposes 1 token ≈ 4
characters of code is a solid heuristic for a 1.5B coder model's tokenizer.
"""
from __future__ import annotations

import math

TOKENS_PER_CHAR = 4.0


def estimate_tokens(text: str) -> int:
    """Rough token estimate for arbitrary text."""
    if not text:
        return 0
    return max(1, math.ceil(len(text) / TOKENS_PER_CHAR))


def truncate_to_tokens(text: str, limit: int) -> str:
    """Truncate ``text`` to roughly ``limit`` tokens, at a newline boundary."""
    if estimate_tokens(text) <= limit:
        return text
    allowed_chars = max(1, limit * int(TOKENS_PER_CHAR))
    cut = text[:allowed_chars]
    nl = cut.rfind("\n")
    if nl > allowed_chars // 2:
        cut = cut[:nl]
    return cut + "\n... (context truncated)"


def fit_to_budget(chunks: list[dict], budget_tokens: int, *, min_chunks: int = 1) -> list[dict]:
    """Greedily select ranked chunks that fit in ``budget_tokens``.

    ``chunks`` must already be in preference order (ranked). At least
    ``min_chunks`` is always kept; each chunk is truncated to fit if needed.
    """
    used = 0
    selected: list[dict] = []
    for i, chunk in enumerate(chunks):
        est = estimate_tokens(chunk.get("content", ""))
        if i < min_chunks or used + est <= budget_tokens:
            selected.append(chunk)
            used += est
        else:
            remaining = max(budget_tokens - used, 0)
            if remaining > 0:
                chunk = dict(chunk)
                chunk["content"] = truncate_to_tokens(chunk["content"], remaining)
                selected.append(chunk)
                used += estimate_tokens(chunk["content"])
            break
    return selected


# The master plan's 4K chat budget breakdown.
CHAT_BUDGET_BREAKDOWN = {
    "system": 300,
    "request": 200,
    "repository_context": 1500,
    "relevant_code": 1000,
    "history": 500,
    "safety": 300,
    "total": 3800,
}
