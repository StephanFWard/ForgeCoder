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

from dataclasses import dataclass
from typing import Any

from core.system_one.backends import BackendUnavailable
from core.system_one.decide import CombinedDecision, Decider, combine
from core.system_one.primitives import NoulAnswer, NoulQuestion, state_text

# Share of the blend taken by the System One verdict; the rest is the neutral
# 0.5 prior (there is no cheap lexical prior for "is the evidence sufficient").
SYSTEM_ONE_WEIGHT = 0.6

ANSWER_QUESTION = NoulQuestion(
    instructions="Can this user message be answered accurately from the information supplied?",
    criteria={
        "true": "the supplied context contains enough correct evidence to answer accurately",
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
    """System One probability that the message is answerable from the context."""
    decider = Decider(backend="local" if client is not None else None, client=client)
    try:
        decision = await decider.adecide(_state(message, context_text), {"answer": ANSWER_QUESTION})
        answer = decision.answers["answer"]
        if not isinstance(answer, NoulAnswer):  # defensive: backend contract
            answer = NoulAnswer(noul=0.5)
        return _blend(answer, decision.backend, decision.model, decision.free)
    except (BackendUnavailable, OSError, RuntimeError, ValueError, TypeError, KeyError):
        return _blend(NoulAnswer(noul=0.5), "heuristic-only", "none", True)


def answer_confidence_sync(message: str, context_text: str = "",
                           *, client: Any = None) -> AnswerConfidence:
    """Synchronous twin of :func:`answer_confidence` (no running loop allowed)."""
    decider = Decider(backend="local" if client is not None else None, client=client)
    try:
        decision = decider.decide(_state(message, context_text), {"answer": ANSWER_QUESTION})
        answer = decision.answers["answer"]
        if not isinstance(answer, NoulAnswer):
            answer = NoulAnswer(noul=0.5)
        return _blend(answer, decision.backend, decision.model, decision.free)
    except (BackendUnavailable, OSError, RuntimeError, ValueError, TypeError, KeyError):
        return _blend(NoulAnswer(noul=0.5), "heuristic-only", "none", True)


__all__ = [
    "ANSWER_QUESTION",
    "AnswerConfidence",
    "SYSTEM_ONE_WEIGHT",
    "answer_confidence",
    "answer_confidence_sync",
    "format_probability",
]
