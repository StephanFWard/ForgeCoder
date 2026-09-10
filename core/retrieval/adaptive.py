"""RNP-inspired query-adaptive retrieval weights (hypernetwork analog).

Fisher & Rao (2023, PMC10637337) describe hypernetworks: a high-level latent
vector generates the parameters of a lower-level network.  In retrieval, the
*query intent* serves as that high-level signal, and the adaptive weights over
retrieval features serve as the generated parameters.

This module provides:
- ``QueryType`` — classifies queries into intent categories
- ``AdaptiveWeights`` — query-type → feature-weight mapping (the hypernetwork output)
- ``classify_query()`` — lightweight classifier (heuristic, zero ML)
"""
from __future__ import annotations

import re
from enum import StrEnum


class QueryType(StrEnum):
    """Supported query intent categories for adaptive retrieval."""
    FIX = "fix"
    EXPLAIN = "explain"
    EDIT = "edit"
    PLAN = "plan"
    SEARCH = "search"

    @classmethod
    def all(cls) -> list[str]:
        return [e.value for e in cls]


# Query-type → (what_weight, where_weight, symbol_weight, recent_weight)
_DEFAULT_WEIGHTS: dict[str, tuple[float, float, float, float]] = {
    # Debugging / fixing — structural & recent context matter most
    "fix":     (0.8, 1.5, 1.2, 1.8),
    # Explaining — content semantics matter most
    "explain": (1.6, 0.8, 0.9, 0.8),
    # Editing / implementing — balanced
    "edit":    (1.3, 1.3, 1.5, 1.2),
    # Architectural / design — symbol structure matters most
    "plan":    (1.0, 1.2, 1.8, 1.0),
    # Generic search — baseline
    "search":  (1.0, 1.0, 1.0, 1.0),
}

_FIX_TERMS = {"error", "exception", "bug", "fix", "crash", "failed",
              "failure", "traceback", "stack", "compile", "wrong", "issue"}
_EXPLORE_TERMS = {"explain", "why", "how", "what does", "architecture",
                  "design", "how does", "purpose", "how to"}
_EDIT_TERMS = {"implement", "create", "add", "refactor", "change",
               "write", "modify", "remove", "delete", "update"}
_PLAN_TERMS = {"plan", "approach", "strategy", "steps", "roadmap",
               "workflow", "implement", "design pattern"}


def classify_query(message: str) -> str:
    """Classify a query into one of the predefined intent categories.

    This is a zero-shot heuristic classifier — no model needed, which keeps
    the single-model memory budget intact.
    """
    low = message.lower()
    tokens = set(re.findall(r"\b\w+\b", low))

    # Check fix first — errors need structural context most urgently
    if tokens & _FIX_TERMS:
        # But "how to fix" is more explanatory
        if "how" in low or "explain" in low:
            return "explain"
        return "fix"

    if tokens & _PLAN_TERMS and len(message.split()) > 5:
        return "plan"

    if tokens & _EDIT_TERMS:
        return "edit"

    if tokens & _EXPLORE_TERMS:
        return "explain"

    return "search"


class AdaptiveWeights:
    """Query-adaptive retrieval weights (RNP hypernetwork analog).

    The query intent vector modulates how each retrieval signal contributes
    to the final score.  For example, a ``"fix"`` query boosts structural
    (``where``) and recency signals, while an ``"explain"`` query boosts
    content (``what``) signals.
    """

    def __init__(self, weights: dict[str, tuple[float, float, float, float]] | None = None):
        self._weights = weights or dict(_DEFAULT_WEIGHTS)

    def get(self, query_type: str) -> tuple[float, float, float, float]:
        """Return (what_w, where_w, symbol_w, recent_w) for the given query type."""
        return self._weights.get(query_type, self._weights["search"])

    def describe(self, query_type: str) -> dict[str, float]:
        """Human-readable weight breakdown for logging/debugging."""
        w = self.get(query_type)
        return {"what": w[0], "where": w[1], "symbol": w[2], "recent": w[3]}

    def supported_types(self) -> list[str]:
        return list(self._weights.keys())


# Shared singleton for convenience
DEFAULT_ADAPTIVE = AdaptiveWeights()


__all__ = [
    "QueryType",
    "classify_query",
    "AdaptiveWeights",
    "DEFAULT_ADAPTIVE",
]
