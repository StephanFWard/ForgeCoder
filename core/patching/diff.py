"""Unified diff generation (stdlib difflib)."""
from __future__ import annotations

import difflib

CONTEXT_LINES = 3


def make_diff(original: str, modified: str, path: str = "file", *, context: int = CONTEXT_LINES) -> str:
    """Return a unified diff between two file contents."""
    before = original.splitlines(keepends=True)
    after = modified.splitlines(keepends=True)
    if not before and not after:
        return ""
    diff = difflib.unified_diff(
        before,
        after,
        fromfile=f"a/{path}",
        tofile=f"b/{path}",
        n=context,
    )
    return "".join(diff)


def diff_stat(diff_text: str) -> dict[str, int]:
    """Count added/removed lines for UI badges."""
    added = removed = 0
    for line in diff_text.splitlines():
        if line.startswith("+") and not line.startswith("+++"):
            added += 1
        elif line.startswith("-") and not line.startswith("---"):
            removed += 1
    return {"added": added, "removed": removed}
