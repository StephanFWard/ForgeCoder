"""Weighted reranking of retrieval candidates with System One score questions.

The heuristic scorer in :mod:`core.retrieval.ranking` counts matches with fixed
credits (``+10`` per token, ``+50`` for a symbol). That is fast and can be
surprising: a token that appears in every candidate is worth as much as a token
unique to the right one, and the result is a bare integer with no statement of
how sure it is.

This module adds one batched decision on top, using the same
``(state, questions)`` contract as Jev: every candidate becomes a ``score``
question in a **single** call ("adding questions barely changes the response
time"), each answer comes back as an expected level plus probabilities and a
confidence, and the final order is an explicit weighted blend of the two
signals:

    weighted = (1 - system_one_weight) * heuristic_normalised
             +        system_one_weight  * expected_level_normalised

Deterministic by default: with the default backend the whole rerank is offline
arithmetic, so the same candidates and query always produce the same order.
"""
from __future__ import annotations

from typing import Any

from core.retrieval.ranking import rank_chunks
from core.system_one import lexical
from core.system_one.decide import Decider
from core.system_one.primitives import ScoreAnswer, ScoreQuestion, state_text

# Rubric for the per-candidate relevance question. Concrete situations rather
# than adjectives: a level description is the question.
RELEVANCE_LEVELS: tuple[str, ...] = (
    "irrelevant to the request",
    "background context that the request does not need",
    "code the request depends on or refers to",
    "the exact code the request is about",
)

RELEVANCE_INSTRUCTIONS = "How relevant is retrieved chunk {index} to the request?"
# Characters of each candidate shown to the decision layer.
_CHUNK_PREVIEW = 400


def _state(query: str, candidates: list[dict]) -> str:
    """Query plus numbered chunk previews, the state all questions see."""
    lines = [f"REQUEST: {query}", "", "RETRIEVED CHUNKS:"]
    for index, candidate in enumerate(candidates):
        content = (candidate.get("content") or "").replace("\r", "").strip()
        preview = " ".join(content[:_CHUNK_PREVIEW].split())
        lines.append(f"[{index}] {candidate.get('path', '?')}: {preview}")
    return "\n".join(lines)


def relevance_questions(count: int) -> dict[str, ScoreQuestion]:
    """One ``score`` question per candidate, named ``chunk_<index>``."""
    return {
        f"chunk_{index}": ScoreQuestion(
            instructions=RELEVANCE_INSTRUCTIONS.format(index=index),
            criteria=list(RELEVANCE_LEVELS),
        )
        for index in range(count)
    }


def _normalised_level(answer: ScoreAnswer) -> float:
    """Expected level as 0..1 (0 = irrelevant, 1 = the exact code)."""
    levels = max(1, len(answer.probabilities) - 1)
    return lexical.clamp01(answer.score / levels)


def rank_chunks_weighted(candidates: list[dict], query: str, *, limit: int = 6,
                         system_one_weight: float = 0.35, backend: Any = None,
                         client: Any = None, now: float | None = None) -> list[dict]:
    """Rank candidates by a weighted blend of heuristic and System One scores.

    Each returned row keeps the heuristic ``_score`` and gains ``_weighted``
    (the blended 0..1 score), ``_system_one`` (expected level, 0..1) and
    ``_system_one_confidence`` (top-level probability). Ordering is by
    ``_weighted`` descending with the original order as the tie-break, so the
    result is stable across runs.
    """
    if not candidates:
        return []
    weight = lexical.clamp01(float(system_one_weight))
    heuristic = rank_chunks(list(candidates), query, limit=len(candidates), now=now)
    base = lexical.normalize([float(row.get("_score", 0) or 0) for row in heuristic])

    decision = Decider(backend=backend, client=client).decide(
        state_text(_state(query, heuristic)), relevance_questions(len(heuristic)),
    )

    scored: list[tuple[float, int, dict]] = []
    for index, row in enumerate(heuristic):
        answer = decision.answers.get(f"chunk_{index}")
        level = _normalised_level(answer) if isinstance(answer, ScoreAnswer) else 0.0
        confidence = answer.confidence if isinstance(answer, ScoreAnswer) else 0.0
        blended = (1.0 - weight) * base[index] + weight * level
        row["_weighted"] = round(blended, 6)
        row["_system_one"] = round(level, 6)
        row["_system_one_confidence"] = round(confidence, 6)
        row["_system_one_backend"] = decision.backend
        scored.append((blended, index, row))

    scored.sort(key=lambda entry: (-entry[0], entry[1]))
    return [row for _, _, row in scored[:limit]]


__all__ = [
    "RELEVANCE_INSTRUCTIONS",
    "RELEVANCE_LEVELS",
    "rank_chunks_weighted",
    "relevance_questions",
]
