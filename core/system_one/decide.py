"""The decision facade: one call in, typed weighted answers out.

``Decider`` is what the rest of ForgeCoder talks to. It owns backend selection
(free by default), wraps the answer payload in the Jev-shaped envelope, and
exposes the composition helpers that make a set of atomic answers useful:
:func:`answer_value` (every answer as a 0..1 weight), :func:`combine` (weighted
sum with the coefficients in *your* code — TypeSafe's documented
"composite scoring" pattern), :func:`gate` (confidence threshold on a yes/no)
and :func:`top_choice` / :func:`expected_score`.

Nothing here is random or time-dependent, so a decision can be replayed in a
test and produce identical numbers.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from core.system_one.backends import (
    DEFAULT_BACKEND,
    BackendUnavailable,
    DecisionBackend,
    resolve_backend,
)
from core.system_one.primitives import (
    Answer,
    ChoiceAnswer,
    DecisionResponse,
    NoulAnswer,
    Question,
    ScoreAnswer,
    Usage,
    estimate_input_tokens,
    questions_payload,
    state_text,
)

# Model labels reported in the envelope, so a caller can see where the numbers
# came from. The deterministic label is deliberately explicit: no weights model.
_MODEL_LABELS = {
    DEFAULT_BACKEND: "forge-system-one-deterministic",
    "local": "forge-system-one-local-llama",
    "jev": "jev-hosted",
}


def answer_value(answer: Answer) -> float:
    """Read any answer as a 0..1 weight.

    * ``choice`` -> probability of the winning option (its confidence)
    * ``score``  -> expected level, normalised by the top level
    * ``noul``   -> the calibrated p(yes)
    """
    if isinstance(answer, ChoiceAnswer):
        return max(0.0, min(1.0, answer.confidence))
    if isinstance(answer, ScoreAnswer):
        levels = max(1, len(answer.probabilities) - 1)
        return max(0.0, min(1.0, answer.score / levels))
    return max(0.0, min(1.0, answer.noul))


def top_choice(answer: Answer) -> tuple[str, float]:
    """Winning option and its probability, for choice answers."""
    if not isinstance(answer, ChoiceAnswer):
        raise TypeError(f"top_choice needs a choice answer, got {type(answer).__name__}")
    return answer.choice, answer.probabilities.get(answer.choice, answer.confidence)


def expected_score(answer: Answer) -> float:
    """The (possibly fractional) expected level of a score answer."""
    if not isinstance(answer, ScoreAnswer):
        raise TypeError(f"expected_score needs a score answer, got {type(answer).__name__}")
    return answer.score


def gate(answer: Answer, threshold: float = 0.7) -> bool:
    """Explicit confidence gate for a yes/no: ``p(yes) >= threshold``."""
    if not isinstance(answer, NoulAnswer):
        raise TypeError(f"gate needs a noul answer, got {type(answer).__name__}")
    return answer.noul >= threshold


@dataclass(frozen=True)
class CombinedDecision:
    """A weighted composition of several answers (all values in 0..1)."""

    total: float
    parts: dict[str, float]
    weights: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"total": round(self.total, 6), "parts": self.parts, "weights": self.weights}


def combine(answers: dict[str, Answer], weights: dict[str, float] | None = None) -> CombinedDecision:
    """Weighted mean of :func:`answer_value` across answers.

    Weights live in the caller's code, not in a prompt, so a priority change is
    a coefficient change. Unweighted questions default to ``1.0``; questions
    whose weight is ``0`` are still reported in ``parts`` but cannot move the
    total. A total weight of ``0`` yields ``0.0`` rather than dividing by zero.
    """
    coefficients = {name: float((weights or {}).get(name, 1.0)) for name in answers}
    parts = {name: round(answer_value(answer), 6) for name, answer in answers.items()}
    total_weight = sum(coefficients.values())
    if total_weight <= 0.0:
        return CombinedDecision(total=0.0, parts=parts, weights=coefficients)
    total = sum(coefficients[name] * parts[name] for name in answers) / total_weight
    return CombinedDecision(total=round(total, 6), parts=parts, weights=coefficients)


class Decider:
    """Backend-selecting front door for System One decisions.

    The default backend is the free deterministic one. ``backend="local"`` uses
    the local llama.cpp server (also free); ``backend="jev"`` needs
    ``allow_paid=True`` **and** a key, so a paid model is never chosen by
    accident.
    """

    def __init__(self, backend: DecisionBackend | str | None = None, *,
                 client: Any = None, allow_paid: bool = False):
        self._backend = backend
        self._client = client
        self._allow_paid = allow_paid

    # ------------------------------------------------------------- selection
    def backend(self, name: DecisionBackend | str | None = None) -> DecisionBackend:
        """Resolve the backend for this call (per-call override wins)."""
        chosen = name if name is not None else self._backend
        if isinstance(chosen, str) or chosen is None:
            return resolve_backend(chosen, client=self._client, allow_paid=self._allow_paid)
        return chosen

    # -------------------------------------------------------------- deciding
    def decide(self, state: str | dict[str, Any] | list[Any],
               questions: dict[str, Question],
               backend: DecisionBackend | str | None = None) -> DecisionResponse:
        """Answer ``questions`` about ``state`` and wrap them in the envelope."""
        chosen = self.backend(backend)
        return self._envelope(chosen, state, questions, chosen.decide(state, questions))

    async def adecide(self, state: str | dict[str, Any] | list[Any],
                      questions: dict[str, Question],
                      backend: DecisionBackend | str | None = None) -> DecisionResponse:
        """Async form; uses a backend's ``adecide`` when it has one."""
        chosen = self.backend(backend)
        adecide = getattr(chosen, "adecide", None)
        if adecide is None:
            answers = chosen.decide(state, questions)
        else:
            answers = await adecide(state, questions)
        return self._envelope(chosen, state, questions, answers)

    # --------------------------------------------------------------- helpers
    def _envelope(self, chosen: DecisionBackend, state: Any, questions: dict[str, Question],
                  answers: dict[str, Answer]) -> DecisionResponse:
        """Wrap answers with model/cost metadata (zero cost for free backends)."""
        text = state_text(state)
        estimated = estimate_input_tokens(text + str(questions_payload(questions)))
        reported = dict(getattr(chosen, "last_usage", None) or {})
        usage = Usage(
            input_tokens=int(reported.get("input_tokens") or estimated),
            output_tokens=int(reported.get("output_tokens") or 0),
            cost_usd=float(reported.get("cost_usd") or 0.0),
            credits_remaining_usd=reported.get("credits_remaining_usd"),
        )
        return DecisionResponse(
            model=_MODEL_LABELS.get(chosen.name, chosen.name),
            answers=answers,
            usage=usage,
            backend=chosen.name,
            free=chosen.free,
        )


__all__ = [
    "BackendUnavailable",
    "CombinedDecision",
    "Decider",
    "answer_value",
    "combine",
    "expected_score",
    "gate",
    "top_choice",
]
