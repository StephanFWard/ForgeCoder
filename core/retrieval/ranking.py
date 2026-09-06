"""Deterministic, cheap ranking over FTS5 candidates.

Scoring rules (from the master plan):

    exact symbol match       +50
    filename match           +30
    function match           +25
    error text match         +25
    import/dependency match  +10
    recently edited          +10

The goal is not to be perfect — it is to reliably surface the 1–6 chunks that
matter before a 1.5B model has to look.
"""
from __future__ import annotations

import re
import time

_WORD = re.compile(r"[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*")
_ERROR_TERMS = {"error", "exception", "null", "undefined", "failed", "failure",
                "bug", "fix", "throws", "traceback", "stack", "compile", "crash"}
_IMPORT_PATTERNS = re.compile(r"^\s*(import|from|using|require|#include|#import)\b", re.MULTILINE)
_RECENT_SECONDS = 3 * 24 * 3600


def _tokens(query: str) -> list[str]:
    return [t.lower() for t in _WORD.findall(query) if len(t) >= 2]


def rank_chunks(candidates: list[dict], query: str, *, limit: int = 6,
                now: float | None = None) -> list[dict]:
    """Score ``candidates`` (rows from CodeIndex.search) and return top ``limit``."""
    tokens = _tokens(query)
    now = now if now is not None else time.time()
    scored: list[dict] = []

    for c in candidates:
        score = 0
        content = c.get("content", "") or ""
        path = c.get("path", "") or ""
        low_content = content.lower()
        low_path = path.lower()
        basename = path.rsplit("/", 1)[-1].rsplit(".", 1)[0].lower()

        for tok in tokens:
            # exact symbol match: query token appears as a standalone identifier
            if re.search(rf"\b{re.escape(tok)}\b", low_content):
                score += 10
            if tok in low_path:
                score += 8
            if tok == basename:
                score += 30  # filename match

        # symbol match bonus: query token equals a name on its own line context
        for name in _lined_symbols(content):
            if name in tokens:
                score += 50
                break

        if "function" in (c.get("language") or "") and low_content:
            pass
        # function match: word 'def/fn/function/fun' + token in first 3 lines
        head = low_content.splitlines()[:3]
        if any(re.search(r"\b(def|fn|func|function|fun)\b", h) for h in head):
            if any(tok in head_text for tok in tokens for head_text in head):
                score += 25

        # error text match
        if any(t in _ERROR_TERMS for t in tokens) and any(t in low_content for t in tokens):
            score += 25

        # import/dependency match
        if _IMPORT_PATTERNS.match(content) and any(tok in low_content for tok in tokens):
            score += 10

        # recently edited
        modified = float(c.get("modified") or 0)
        if modified and now - modified <= _RECENT_SECONDS:
            score += 10

        c["_score"] = score
        scored.append(c)

    scored.sort(key=lambda r: (-r["_score"], r.get("rank", 0)))
    return scored[:limit]


def _lined_symbols(content: str) -> set[str]:
    """Extract candidate symbol names from line starts (heuristic)."""
    names: set[str] = set()
    for line in content.splitlines()[:60]:
        m = re.match(r"^\s*(?:public|private|protected|static|final|export|default|async)?\s*(?:class|interface|enum|def|fn|func|function|fun|struct|trait|type|const)\s+([A-Za-z_][A-Za-z0-9_]*)", line)
        if m:
            names.add(m.group(1).lower())
    return names
