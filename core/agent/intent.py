"""Intent gate: is this request a code-change task, or a plain question?

A message like "Is a hot dog a sandwich?" is not a task. But ForgeCoder used to
frame *every* chat turn as one, so a plain question was decorated with edit
bounds and unknowables — "no acceptance command was stated; name the command
that proves the change" — and the 1.5B model echoed those back as
``[warn] no-unverified-test-claims`` lines instead of answering. The frame layer
already had one escape hatch (`is_creation_request`); this module adds the other
side of the same decision: *is this a code-change request at all?*

The decision is a weighted blend of two answers, both produced through the
System One (Jev) contract so the result is a calibrated distribution rather
than a bare boolean:

* a deterministic heuristic (interrogative form, edit verbs, code/file tokens)
  — the fast prior, the same role the heuristic ranker plays in the rerank;
* a ``noul`` question answered by the System One layer: with a llama.cpp
  client the free on-device model gives the meaning-level verdict (blended
  over the deterministic distribution by ``LocalModelBackend``), and without
  one the free deterministic backend answers lexically.

    weighted = (1 - w) * heuristic + w * system_one     (w = SYSTEM_ONE_WEIGHT)

``code_change`` is ``weighted >= CODE_CHANGE_THRESHOLD``. Everything here is
free, local, and never raises: if the model is down the gate degrades to the
deterministic view and still answers.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from core.system_one.backends import BackendUnavailable
from core.system_one.decide import CombinedDecision, Decider, combine
from core.system_one.primitives import NoulAnswer, NoulQuestion, state_text

# The weighted blend above this line means "treat the message as a task".
CODE_CHANGE_THRESHOLD = 0.5
# The System One answer's share of the blend. The model corrects the lexical
# prior, it does not own the decision — a downed server must not flip intents.
SYSTEM_ONE_WEIGHT = 0.6

INTENT_QUESTION = NoulQuestion(
    instructions="Is this user message a request to change, fix, or create code or files?",
    criteria={
        "true": "a software-change task: edit, fix, refactor, rename, or create code or files",
        "false": "a conversational or factual question, or general discussion with no code change",
    },
)

# Requests to change code. Broad by design: a missed verb misroutes a real
# task into chatty mode, which is the worse failure here.
_EDIT_VERB_RE = re.compile(
    r"\b(add|append|fix|repair|refactor|restructure|change|update|modify|edit|"
    r"remove|delete|drop|rename|move|implement|optimi[sz]e|simplify|clean ?up|"
    r"write|create|generate|make|build|scaffold|install|configure|set ?up|"
    r"replace|extract|inline|split|merge|migrate|port|backport|revert|"
    r"bump|pin|mock|stub|wire|hook|integrate|extend|support)\b",
    re.IGNORECASE,
)
# Interrogative form: a question about something, not a change to something.
_QUESTION_RE = re.compile(
    r"^\s*(is|are|was|were|am|do|does|did|can|could|would|should|will|shall|"
    r"what|which|who|whom|whose|when|where|why|how)\b[^?]*\??\s*$",
    re.IGNORECASE,
)
# Code-shaped tokens: file extensions, declarations, tooling nouns.
_CODE_RE = re.compile(
    r"\b(function|method|class|variable|parameter|argument|return|import|"
    r"api|endpoint|schema|grammar|patch|diff|commit|branch|build|compile|lint|"
    r"test|tests|bug|error|exception|traceback|null|none|undefined|memory|"
    r"index|indexer|database|db|cache|thread|loop|async)\b"
    r"|[\w.-]+\.(py|js|ts|tsx|jsx|java|go|rs|cs|cpp|c|h|hpp|rb|php|sql|json|"
    r"yaml|yml|toml|md|html|css|gbnf)\b",
    re.IGNORECASE,
)


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def heuristic_code_change(message: str, *, has_file: bool = False,
                          has_selection: bool = False) -> float:
    """Deterministic 0..1 prior that the message asks for a code change.

    Pure string arithmetic — no model, no network, identical output for the
    same input, so tests can pin it. ``0.35`` is the neutral prior: chat inside
    a coding tool is *usually* about code, and it takes contrary evidence (an
    interrogative form with no change verb) to pull below the line.
    """
    text = (message or "").strip()
    if not text:
        return 0.0
    lower = text.lower()
    score = 0.35
    if _EDIT_VERB_RE.search(lower):
        score += 0.30
    if _CODE_RE.search(lower):
        score += 0.20
    if has_file or has_selection:
        score += 0.25
    if _QUESTION_RE.match(text) and not _EDIT_VERB_RE.search(lower):
        score -= 0.45
    return _clamp01(score)


@dataclass(frozen=True)
class IntentDecision:
    """The blended verdict, with the numbers that produced it for inspection."""

    code_change: bool
    confidence: float
    parts: dict[str, float]
    weights: dict[str, float]
    system_one_backend: str
    system_one_model: str
    system_one_free: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "code_change": self.code_change,
            "confidence": round(self.confidence, 6),
            "parts": self.parts,
            "weights": self.weights,
            "backend": self.system_one_backend,
            "model": self.system_one_model,
            "free": self.system_one_free,
        }


def _system_one_answer(message: str, client: Any) -> tuple[NoulAnswer, str, str, bool]:
    """Ask the System One layer one ``noul`` question about the message.

    With a llama.cpp client the free on-device model gives the meaning-level
    verdict (a failure degrades, inside the backend, to the deterministic
    answer); without one the deterministic backend answers lexically. Returns
    the answer plus backend metadata for the receipt.
    """
    decider = Decider(backend="local" if client is not None else None, client=client)
    decision = decider.decide(
        state_text({"message": message}),
        {"code_change": INTENT_QUESTION},
    )
    answer = decision.answers["code_change"]
    if not isinstance(answer, NoulAnswer):  # defensive: backend contract
        answer = NoulAnswer(noul=0.5)
    return answer, decision.backend, decision.model, decision.free


async def _asystem_one_answer(message: str, client: Any) -> tuple[NoulAnswer, str, str, bool]:
    """Async twin of :func:`_system_one_answer` for FastAPI paths."""
    decider = Decider(backend="local" if client is not None else None, client=client)
    decision = await decider.adecide(
        state_text({"message": message}),
        {"code_change": INTENT_QUESTION},
    )
    answer = decision.answers["code_change"]
    if not isinstance(answer, NoulAnswer):
        answer = NoulAnswer(noul=0.5)
    return answer, decision.backend, decision.model, decision.free


def _blend(heuristic: float, answer: NoulAnswer, backend: str, model: str,
           free: bool) -> IntentDecision:
    """Weighted mean of the heuristic prior and the System One verdict."""
    combined: CombinedDecision = combine(
        {"heuristic": NoulAnswer(noul=heuristic), "system_one": answer},
        weights={"heuristic": 1.0 - SYSTEM_ONE_WEIGHT, "system_one": SYSTEM_ONE_WEIGHT},
    )
    return IntentDecision(
        code_change=combined.total >= CODE_CHANGE_THRESHOLD,
        confidence=combined.total,
        parts=combined.parts,
        weights=combined.weights,
        system_one_backend=backend,
        system_one_model=model,
        system_one_free=free,
    )


async def classify_intent(message: str, *, client: Any = None, has_file: bool = False,
                          has_selection: bool = False) -> IntentDecision:
    """Decide whether ``message`` is a code-change task. Never raises."""
    prior = heuristic_code_change(message, has_file=has_file, has_selection=has_selection)
    try:
        answer, backend, model, free = await _asystem_one_answer(message, client)
    except (BackendUnavailable, OSError, RuntimeError, ValueError, TypeError, KeyError):
        # No backend could answer at all: the heuristic prior alone decides,
        # recorded honestly as a heuristic-only decision.
        return _blend(prior, NoulAnswer(noul=prior), "heuristic-only", "none", True)
    return _blend(prior, answer, backend, model, free)


def classify_intent_sync(message: str, *, client: Any = None, has_file: bool = False,
                         has_selection: bool = False) -> IntentDecision:
    """Synchronous twin of :func:`classify_intent` (for non-loop callers/tests)."""
    prior = heuristic_code_change(message, has_file=has_file, has_selection=has_selection)
    try:
        answer, backend, model, free = _system_one_answer(message, client)
    except (BackendUnavailable, OSError, RuntimeError, ValueError, TypeError, KeyError):
        return _blend(prior, NoulAnswer(noul=prior), "heuristic-only", "none", True)
    return _blend(prior, answer, backend, model, free)


__all__ = [
    "CODE_CHANGE_THRESHOLD",
    "INTENT_QUESTION",
    "SYSTEM_ONE_WEIGHT",
    "IntentDecision",
    "classify_intent",
    "classify_intent_sync",
    "heuristic_code_change",
]

