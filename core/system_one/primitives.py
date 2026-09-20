"""Jev-compatible decision primitives: state in, typed weighted answers out.

TypeSafe AI's **Jev** ("System One") model does not generate text. It answers
typed questions about a `state` and returns structured values with probability
distributions and a confidence, so calling code branches on numbers instead of
parsing prose (typesafe.ai, docs.typesafe.ai).

The request/response shapes in this module are deliberately a copy of Jev's
public contract — `state` plus a map of `choice` / `score` / `noul` questions,
answered with `choice` / `score` / `noul` values — so that a ForgeCoder
deployment speaks the same protocol as the hosted model. Two consequences:

* Anything built against this module ports to the hosted API by swapping the
  backend (`core.system_one.backends.JevApiBackend`), no schema work required.
* The same shapes are served locally by the **free** backends, which is
  ForgeCoder's default: no key, no quota, no network.

Documented limits from Jev's API are enforced here rather than discovered at
call time: 255 options per `choice`, 2-10 levels per `score`.

Everything here is pure data: no model, no network, no I/O.
"""
from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

# Documented Jev limits (docs: "255 options per choice", score levels 2-10).
MAX_CHOICE_OPTIONS = 255
MIN_SCORE_LEVELS = 2
MAX_SCORE_LEVELS = 10
# Name of the optional `noul` criteria pair understood by the API.
NOUL_CRITERIA_KEYS = ("true", "false")

_STRICT = ConfigDict(extra="forbid")



# ---------------------------------------------------------------------------
# Questions
# ---------------------------------------------------------------------------

class ChoiceQuestion(BaseModel):
    """Pick one labelled option out of up to 255."""

    model_config = _STRICT

    type: Literal["choice"] = "choice"
    instructions: str = Field(min_length=1, max_length=2000)
    # A value may be None for a label with no description, as in the API.
    criteria: dict[str, str | None] = Field(min_length=1, max_length=MAX_CHOICE_OPTIONS)

    @field_validator("criteria")
    @classmethod
    def _keys_present(cls, value: dict[str, str | None]) -> dict[str, str | None]:
        if any(not key.strip() for key in value):
            raise ValueError("choice option keys must be non-empty")
        return value


class ScoreQuestion(BaseModel):
    """Rate the state on an ordered rubric, low to high."""

    model_config = _STRICT

    type: Literal["score"] = "score"
    instructions: str = Field(min_length=1, max_length=2000)
    criteria: list[str] = Field(min_length=MIN_SCORE_LEVELS, max_length=MAX_SCORE_LEVELS)

    @field_validator("criteria")
    @classmethod
    def _levels_described(cls, value: list[str]) -> list[str]:
        if any(not level.strip() for level in value):
            raise ValueError("every score level needs a description")
        return value


class NoulQuestion(BaseModel):
    """A calibrated yes/no: `noul` is p(yes) in 0..1."""

    model_config = _STRICT

    type: Literal["noul"] = "noul"
    instructions: str = Field(min_length=1, max_length=2000)
    # Optional spelling-out of what yes and no mean.
    criteria: dict[str, str] | None = None

    @field_validator("criteria")
    @classmethod
    def _known_criteria(cls, value: dict[str, str] | None) -> dict[str, str] | None:
        if value is None:
            return None
        unknown = set(value) - set(NOUL_CRITERIA_KEYS)
        if unknown:
            raise ValueError(f"noul criteria may only define {NOUL_CRITERIA_KEYS}, got {sorted(unknown)}")
        return value


Question = Annotated[
    ChoiceQuestion | ScoreQuestion | NoulQuestion,
    Field(discriminator="type"),
]


# ---------------------------------------------------------------------------
# Answers
# ---------------------------------------------------------------------------

class ChoiceAnswer(BaseModel):
    """Winning option plus the full probability mass over every option."""

    model_config = _STRICT

    type: Literal["choice"] = "choice"
    choice: str
    confidence: float = Field(ge=0.0, le=1.0)
    probabilities: dict[str, float] = Field(default_factory=dict)


class ScoreAnswer(BaseModel):
    """Expected level (0-based, possibly fractional) plus per-level mass.

    ``legend`` is what the official Jev API sends back — the level
    descriptions keyed by level index — and is optional so the free local
    backends (which do not repeat the question's rubric) stay unchanged.
    """

    model_config = _STRICT

    type: Literal["score"] = "score"
    score: float = Field(ge=0.0)
    confidence: float = Field(ge=0.0, le=1.0)
    probabilities: dict[str, float] = Field(default_factory=dict)
    legend: dict[str, str] | None = None


class NoulAnswer(BaseModel):
    """Calibrated p(yes) in 0..1. Jev reports no separate confidence here."""

    model_config = _STRICT

    type: Literal["noul"] = "noul"
    noul: float = Field(ge=0.0, le=1.0)


Answer = ChoiceAnswer | ScoreAnswer | NoulAnswer


def new_answer(question: Question, **kwargs: Any) -> Answer:
    """Build the answer type matching ``question`` (keeps backend code typed)."""
    if isinstance(question, ChoiceQuestion):
        return ChoiceAnswer(**kwargs)
    if isinstance(question, ScoreQuestion):
        return ScoreAnswer(**kwargs)
    return NoulAnswer(**kwargs)


def answers_payload(answers: dict[str, Answer]) -> dict[str, dict]:
    """Serialise answers exactly as the Jev API returns them."""
    return {name: answer.model_dump() for name, answer in answers.items()}


def questions_payload(questions: dict[str, Question]) -> dict[str, dict]:
    """Serialise questions exactly as the Jev API expects them."""
    return {name: question.model_dump(exclude_none=True) for name, question in questions.items()}


# ---------------------------------------------------------------------------
# Endpoint envelope (mirrors `POST /v1/decide`)
# ---------------------------------------------------------------------------

class Usage(BaseModel):
    """Token accounting. Local/free backends always report ``cost_usd == 0.0``."""

    model_config = _STRICT

    input_tokens: int = Field(ge=0)
    output_tokens: int = Field(default=0, ge=0)
    cost_usd: float = Field(default=0.0, ge=0.0)
    credits_remaining_usd: float | None = None


class DecisionRequest(BaseModel):
    """A Jev-shaped decision request: state plus typed questions."""

    model_config = _STRICT

    state: str | dict[str, Any] | list[Any]
    questions: dict[str, Question] = Field(min_length=1)
    # ForgeCoder extension: which backend answers ("deterministic", "local",
    # "jev", "auto"). Omitted -> the configured default, which is free.
    backend: str | None = None


class DecisionResponse(BaseModel):
    """A Jev-shaped decision response plus ForgeCoder backend metadata."""

    model_config = _STRICT

    model: str
    answers: dict[str, Answer]
    usage: Usage
    backend: str
    free: bool


def state_text(state: str | dict[str, Any] | list[Any]) -> str:
    """Render any accepted state form as the text a backend evaluates.

    Strings pass through; objects/arrays become JSON so key names and values
    are both visible to the lexical weighting. Anything else is stringified
    rather than raising, so a caller that hands over a model or a dataclass
    still gets an answer instead of a serializer crash.
    """
    if isinstance(state, str):
        return state
    import json

    return json.dumps(state, ensure_ascii=False, sort_keys=True, default=str)


def estimate_input_tokens(text: str) -> int:
    """Cheap, deterministic token estimate (~4 chars/token), never zero."""
    return max(1, len(text) // 4)


__all__ = [
    "MAX_CHOICE_OPTIONS",
    "MAX_SCORE_LEVELS",
    "MIN_SCORE_LEVELS",
    "NOUL_CRITERIA_KEYS",
    "Answer",
    "ChoiceAnswer",
    "ChoiceQuestion",
    "DecisionRequest",
    "DecisionResponse",
    "NoulAnswer",
    "NoulQuestion",
    "Question",
    "ScoreAnswer",
    "ScoreQuestion",
    "Usage",
    "answers_payload",
    "estimate_input_tokens",
    "new_answer",
    "questions_payload",
    "state_text",
]
