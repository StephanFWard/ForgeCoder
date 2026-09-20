"""System One decision layer: Jev-shaped typed questions, free by default.

TypeSafe AI's **Jev** is a "System One" model: no text generation, just typed
questions (``choice`` / ``score`` / ``noul``) answered against a state with
probabilities and a confidence, fast enough to call from code. ForgeCoder
adopts the contract — and the parts of it that are free:

* :mod:`core.system_one.primitives` — the request/response shapes, limits, and
  the envelope served by ``POST /v1/decide``.
* :mod:`core.system_one.backends` — ``deterministic`` (default: offline,
  reproducible, no model), ``local`` (free, on-device llama.cpp verdicts blended
  into the same distribution shape), ``jev`` (the hosted, billed API; opt-in
  only).
* :mod:`core.system_one.decide` — :class:`Decider` plus the weighted
  composition helpers (:func:`combine`, :func:`gate`, :func:`answer_value`).
* :mod:`core.system_one.jev_compat` — the upstream ``jev`` decorator API
  (``@fn``, ``decide``, ``BaseModel``), re-implemented for Python 3.10+ because
  the PyPI package requires 3.14+ and a paid key.
* :mod:`core.system_one.rerank` — weighted retrieval reranking.

Everything in this package is local and deterministic unless a caller asks for
``backend="jev"``; see ``docs/system-one.md``.
"""

from core.system_one.backends import (
    BACKEND_ENV,
    DEFAULT_BACKEND,
    BackendUnavailable,
    DecisionBackend,
    DeterministicBackend,
    JevApiBackend,
    LocalModelBackend,
    backend_catalog,
    resolve_backend,
)
from core.system_one.decide import (
    CombinedDecision,
    Decider,
    answer_value,
    combine,
    expected_score,
    gate,
    top_choice,
)
from core.system_one.primitives import (
    MAX_CHOICE_OPTIONS,
    MAX_SCORE_LEVELS,
    MIN_SCORE_LEVELS,
    Answer,
    ChoiceAnswer,
    ChoiceQuestion,
    DecisionRequest,
    DecisionResponse,
    NoulAnswer,
    NoulQuestion,
    Question,
    ScoreAnswer,
    ScoreQuestion,
    Usage,
    answers_payload,
    questions_payload,
    state_text,
)
from core.system_one.rerank import rank_chunks_weighted, relevance_questions

__all__ = [
    "BACKEND_ENV",
    "DEFAULT_BACKEND",
    "MAX_CHOICE_OPTIONS",
    "MAX_SCORE_LEVELS",
    "MIN_SCORE_LEVELS",
    "Answer",
    "BackendUnavailable",
    "ChoiceAnswer",
    "ChoiceQuestion",
    "CombinedDecision",
    "DecisionBackend",
    "DecisionRequest",
    "DecisionResponse",
    "Decider",
    "DeterministicBackend",
    "JevApiBackend",
    "LocalModelBackend",
    "NoulAnswer",
    "NoulQuestion",
    "Question",
    "ScoreAnswer",
    "ScoreQuestion",
    "Usage",
    "answer_value",
    "answers_payload",
    "backend_catalog",
    "combine",
    "expected_score",
    "gate",
    "questions_payload",
    "rank_chunks_weighted",
    "relevance_questions",
    "resolve_backend",
    "state_text",
    "top_choice",
]
