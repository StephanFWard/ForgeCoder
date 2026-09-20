"""Answer confidence: the statistical probability shown at the start of every reply.

ForgeCoder answers begin with a marker — ``[p=0.61]`` — so a reader can see,
before reading a word, how much calibrated confidence the System One layer
assigns to the upcoming answer. The number is not a vibe: it is the blend of a
``noul`` question ("Can this user message be answered accurately from the
information supplied?") asked through the same free System One backends as
everything else — the on-device Qwen model when llama.cpp is up, the
deterministic lexical view when it is not — combined with a neutral prior at
the documented ``SYSTEM_ONE_WEIGHT`` share.

What the number means, stated plainly: it is the probability that the supplied
context is sufficient for an accurate answer. It is computed *before* the
answer is generated (so streaming is not delayed), it is free, and it degrades
to the deterministic view when the model is down. A low ``p`` is ForgeCoder
admitting up front that the answer may need evidence the turn did not have.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from core.system_one.backends import BackendUnavailable
from core.system_one.decide import CombinedDecision, Decider, combine
from core.system_one.primitives import NoulAnswer, NoulQuestion, state_text

# Share of the blend taken by the System One verdict; the rest is the neutral
# 0.5 prior (there is no cheap lexical prior for "is the evidence sufficient").
SYSTEM_ONE_WEIGHT = 0.6

# Words stripped when comparing a question's subject to its predicate.
_SUBJECT_STOPWORDS = frozenset({
    "a", "an", "the", "this", "that", "these", "those", "my", "your", "his",
    "her", "its", "our", "their", "some", "any", "each", "every", "no",
})
_IDENTITY_LEADERS = frozenset({"is", "are", "was", "were"})


def _identity_subject(message: str) -> str | None:
    """Subject of an "Is X a X?" style tautology, else None.

    One rule, checked before any model call: a question whose subject equals
    its own predicate ("Is a sandwich a sandwich?") is answered by the meaning
    of its own words, so the confidence must be 1.0 — not a lexical overlap
    score, and never something a previous turn's statistics can dilute.
    """
    words = re.findall(r"[A-Za-z]+", message.lower())
    if len(words) < 3 or words[0] not in _IDENTITY_LEADERS:
        return None
    core = [w for w in words[1:] if w not in _SUBJECT_STOPWORDS]
    if len(core) >= 2 and core[0] == core[-1]:
        return core[0]
    return None

ANSWER_QUESTION = NoulQuestion(
    instructions="Can this user message be answered accurately from the information supplied?",
    criteria={
        "true": ("the message is answerable: it is a tautology or definition true by "
                 "the meaning of its own words, or the supplied context holds the "
                 "evidence needed for an accurate answer"),
        "false": "an accurate answer would need information that was not supplied",
    },
)


def format_probability(value: float) -> str:
    """The two-decimal marker prepended to answers: ``[p=0.61]``."""
    return f"[p={max(0.0, min(1.0, float(value))):.2f}]"


@dataclass(frozen=True)
class AnswerConfidence:
    """The probability and the receipt for where it came from."""

    probability: float
    parts: dict[str, float]
    weights: dict[str, float]
    backend: str
    model: str
    free: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "probability": round(self.probability, 6),
            "parts": self.parts,
            "weights": self.weights,
            "backend": self.backend,
            "model": self.model,
            "free": self.free,
        }


def _blend(answer: NoulAnswer, backend: str, model: str, free: bool) -> AnswerConfidence:
    combined: CombinedDecision = combine(
        {"prior": NoulAnswer(noul=0.5), "system_one": answer},
        weights={"prior": 1.0 - SYSTEM_ONE_WEIGHT, "system_one": SYSTEM_ONE_WEIGHT},
    )
    return AnswerConfidence(
        probability=combined.total,
        parts=combined.parts,
        weights=combined.weights,
        backend=backend,
        model=model,
        free=free,
    )


def _state(message: str, context_text: str) -> str:
    """Message plus (truncated) evidence — the state the question is asked over."""
    parts = [f"MESSAGE: {message}"]
    if context_text.strip():
        parts.append(f"SUPPLIED CONTEXT:\n{context_text}")
    else:
        parts.append("SUPPLIED CONTEXT: (none — a general-knowledge turn)")
    return state_text("\n\n".join(parts))


async def answer_confidence(message: str, context_text: str = "",
                            *, client: Any = None) -> AnswerConfidence:
    """System One probability that the message is answerable from the context.

    Fresh per question: the state is rebuilt from *this* message and *this*
    turn's evidence, so no previous turn's statistics can leak in. A tautology
    ("Is a sandwich a sandwich?") short-circuits to 1.0 before any backend —
    it is answered by the meaning of its own words.
    """
    subject = _identity_subject(message)
    if subject is not None:
        return AnswerConfidence(probability=1.0, parts={"tautology": 1.0},
                                weights={"tautology": 1.0}, backend="tautology",
                                model="forge-identity-rule", free=True)
    if client is None:
        # Free path, no model round-trip: per-question state, freshly decided.
        answer = _deterministic_answer(message, context_text)
        return _blend(answer, "deterministic", "forge-system-one-deterministic", True)
    decider = Decider(backend="local", client=client)
    try:
        decision = await decider.adecide(_state(message, context_text), {"answer": ANSWER_QUESTION})
        answer = decision.answers["answer"]
        if not isinstance(answer, NoulAnswer):  # defensive: backend contract
            answer = NoulAnswer(noul=0.5)
        return _blend(answer, decision.backend, decision.model, decision.free)
    except (BackendUnavailable, OSError, RuntimeError, ValueError, TypeError, KeyError):
        answer = _deterministic_answer(message, context_text)
        return _blend(answer, "local", "forge-system-one-local-llama", True)


def answer_confidence_sync(message: str, context_text: str = "",
                           *, client: Any = None) -> AnswerConfidence:
    """Synchronous twin of :func:`answer_confidence` (no running loop allowed)."""
    subject = _identity_subject(message)
    if subject is not None:
        return AnswerConfidence(probability=1.0, parts={"tautology": 1.0},
                                weights={"tautology": 1.0}, backend="tautology",
                                model="forge-identity-rule", free=True)
    if client is None:
        answer = _deterministic_answer(message, context_text)
        return _blend(answer, "deterministic", "forge-system-one-deterministic", True)
    decider = Decider(backend="local", client=client)
    try:
        decision = decider.decide(_state(message, context_text), {"answer": ANSWER_QUESTION})
        answer = decision.answers["answer"]
        if not isinstance(answer, NoulAnswer):
            answer = NoulAnswer(noul=0.5)
        return _blend(answer, decision.backend, decision.model, decision.free)
    except (BackendUnavailable, OSError, RuntimeError, ValueError, TypeError, KeyError):
        answer = _deterministic_answer(message, context_text)
        return _blend(answer, "local", "forge-system-one-local-llama", True)


def _deterministic_answer(message: str, context_text: str) -> NoulAnswer:
    """Answer the question without a model (free, reproducible, per-question)."""
    decision = Decider().decide(_state(message, context_text), {"answer": ANSWER_QUESTION})
    answer = decision.answers["answer"]
    if not isinstance(answer, NoulAnswer):  # defensive: backend contract
        return NoulAnswer(noul=0.5)
    return answer


__all__ = [
    "ANSWER_QUESTION",
    "AnswerConfidence",
    "SYSTEM_ONE_WEIGHT",
    "answer_confidence",
    "answer_confidence_sync",
    "format_probability",
]
