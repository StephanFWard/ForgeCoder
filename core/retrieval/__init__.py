"""Retrieval layer: FTS5 search, ranking, repository context assembly, token budgets."""

from core.retrieval.budget import estimate_tokens, fit_to_budget, truncate_to_tokens
from core.retrieval.context import ContextBuilder
from core.retrieval.hierarchical import HierarchicalContextBuilder
from core.retrieval.ranking import rank_chunks
from core.retrieval.search import SearchEngine

__all__ = [
    "ContextBuilder",
    "HierarchicalContextBuilder",
    "SearchEngine",
    "estimate_tokens",
    "fit_to_budget",
    "rank_chunks",
    "truncate_to_tokens",
]
