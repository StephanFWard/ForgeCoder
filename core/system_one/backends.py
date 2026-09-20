"""Decision backends: the free local ones (default) and the paid hosted one.

ForgeCoder's System One layer is **free by default**: the
:class:`DeterministicBackend` answers typed questions with IDF-weighted
arithmetic over the state, on-device, with no model, no network, no key, and no
quota. It is byte-for-byte reproducible, which is what makes its weights safe
to threshold and to unit-test.

Two optional backends exist:

* :class:`LocalModelBackend` — free as well (weights run locally through
  llama.cpp), used when a verdict from the 1.5B model is worth a round trip.
  The model's verdict is *blended into* the deterministic distribution, so the
  answer keeps the distribution/confidence shape and degrades to the
  deterministic answer instead of failing when the server is down.
* :class:`JevApiBackend` — the real Jev API (TypeSafe AI), which is **billed
  per input token** and therefore never selected implicitly. It exists so a
  deployment that already pays for Jev can switch with one environment
  variable, using the identical request/response contract.

Backend selection is explicit: ``FORGECODER_SYSTEM_ONE_BACKEND`` or the
``backend`` field on ``POST /v1/decide``. Nothing here reaches the network
unless the caller asked for ``"jev"`` **and** supplied a key.
"""
from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from core.system_one import lexical
from core.system_one.primitives import (
    Answer,
    ChoiceAnswer,
    ChoiceQuestion,
    NoulAnswer,
    NoulQuestion,
    Question,
    ScoreAnswer,
    ScoreQuestion,
    questions_payload,
    state_text,
)

DEFAULT_BACKEND = "deterministic"
BACKEND_ENV = "FORGECODER_SYSTEM_ONE_BACKEND"
JEV_API_KEY_ENV = "JEV_API_KEY"
TYPESAFE_API_KEY_ENV = "TYPESAFE_API_KEY"
JEV_API_URL_ENV = "JEV_API_URL"
JEV_MODEL_ENV = "JEV_MODEL"
# The hosted, TypeSafe-compatible endpoint documented at jevtypesafeai.com.
DEFAULT_JEV_URL = "https://jevtypesafeai.com/api/v1/decide"
# The official TypeSafe Decision API endpoint (Bearer key, `model` required).
DEFAULT_TYPESAFE_URL = "https://api.typesafe.ai/v1/systemone"
_DEFAULT_MODEL = "jev-latest"

# Environment names the optional local secret file (.env.local, gitignored)
# is allowed to populate. Anything else in the file is ignored, so a stray
# line cannot quietly export unrelated variables into the process.
_LOCAL_ENV_NAMES = (JEV_API_KEY_ENV, TYPESAFE_API_KEY_ENV, JEV_API_URL_ENV,
                    JEV_MODEL_ENV)



def parse_env_file(path: Any) -> dict[str, str]:
    """Parse ``KEY=VALUE`` lines; ``#`` comments, blank lines and unknown
    quoting tolerated. Missing file -> empty dict."""
    values: dict[str, str] = {}
    try:
        text = Path(path).read_text(encoding="utf-8")
    except OSError:
        return values
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, _, raw = stripped.partition("=")
        key = key.strip()
        if key not in _LOCAL_ENV_NAMES:
            continue
        raw = raw.strip()
        if len(raw) >= 2 and raw[0] == raw[-1] and raw[0] in {'"', "'"}:
            raw = raw[1:-1]
        if key and raw:
            values[key] = raw
    return values


def load_local_env() -> None:
    """Fill unset key/url variables from ``.env.local`` (never overrides real
    environment variables, never raises). Searched next to the repo root and
    in the working directory."""
    here = Path(__file__).resolve().parent
    for directory in (here.parent.parent, Path.cwd()):
        for key, value in parse_env_file(Path(directory) / ".env.local").items():
            os.environ.setdefault(key, value)


load_local_env()



class BackendUnavailable(RuntimeError):
    """Raised when the requested backend cannot run (missing key/client)."""


@runtime_checkable
class DecisionBackend(Protocol):
    """Anything that can turn (state, typed questions) into typed answers."""

    name: str
    free: bool

    def decide(self, state: str | dict[str, Any] | list[Any],
               questions: dict[str, Question]) -> dict[str, Answer]:
        ...  # pragma: no cover - protocol declaration


def _round_mass(probabilities: list[float]) -> list[float]:
    """Round probability mass without losing the total (largest gets the drift)."""
    rounded = [round(value, 6) for value in probabilities]
    drift = round(1.0 - sum(rounded), 6)
    if drift and rounded:
        top = max(range(len(rounded)), key=lambda index: rounded[index])
        rounded[top] = round(rounded[top] + drift, 6)
    return rounded


class DeterministicBackend:
    """Free, offline, reproducible answers from IDF-weighted state overlap.

    No model, no network, no clock, no randomness: the same ``(state,
    questions)`` pair always produces the same probabilities, so callers can
    pin thresholds on them. ``temperature`` controls how sharply the IDF
    weights separate the options; ``prior`` is the similarity at which a yes/no
    sits at 0.5; ``evidence_scale`` controls how fast it moves from there.

    Known limit, stated plainly: this backend has no world knowledge. A state
    that means "billing" without saying it ("I was charged twice") cannot be
    routed to a ``billing`` option lexically — use the ``local`` backend (free,
    on-device) or the hosted model when the decision needs meaning, and the
    deterministic backend for gates over the text in front of it.
    """

    name = DEFAULT_BACKEND
    free = True

    def __init__(self, *, temperature: float = 0.35, evidence_scale: float = 4.0,
                 prior: float = 0.3):
        self.temperature = max(1e-6, float(temperature))
        self.evidence_scale = float(evidence_scale)
        if not 0.0 < float(prior) < 1.0:
            raise ValueError("prior must be between 0 and 1")
        self.prior = float(prior)

    # ------------------------------------------------------------ questions
    def _candidate_texts(self, question: ChoiceQuestion | ScoreQuestion) -> tuple[list[str], list[str]]:
        """Option labels (in order) and the texts their weights are computed over."""
        if isinstance(question, ChoiceQuestion):
            labels = list(question.criteria)
            texts = [
                f"{question.instructions} {label}: {question.criteria[label] or ''}".strip()
                for label in labels
            ]
        else:
            labels = list(question.criteria)
            texts = [f"{question.instructions} {level}".strip() for level in labels]
        return labels, texts

    def _answer_choice(self, question: ChoiceQuestion, state_tokens: list[str]) -> ChoiceAnswer:
        labels, texts = self._candidate_texts(question)
        mass = _round_mass(lexical.softmax(lexical.similarity_weights(state_tokens, texts), self.temperature))
        best = max(range(len(labels)), key=lambda index: mass[index])
        return ChoiceAnswer(
            choice=labels[best],
            confidence=mass[best],
            probabilities=dict(zip(labels, mass, strict=True)),
        )

    def _answer_score(self, question: ScoreQuestion, state_tokens: list[str]) -> ScoreAnswer:
        labels, texts = self._candidate_texts(question)
        mass = _round_mass(lexical.softmax(lexical.similarity_weights(state_tokens, texts), self.temperature))
        expected = sum(index * probability for index, probability in enumerate(mass))
        return ScoreAnswer(
            score=round(expected, 6),
            confidence=max(mass),
            probabilities={str(index): probability for index, probability in enumerate(mass)},
        )

    def _answer_noul(self, question: NoulQuestion, state_tokens: list[str]) -> NoulAnswer:
        criteria = question.criteria or {}
        if criteria.get("true") and criteria.get("false"):
            weights = lexical.similarity_weights(
                state_tokens,
                [f"{question.instructions} {criteria['true']}",
                 f"{question.instructions} {criteria['false']}"],
            )
            evidence = weights[0] - weights[1]
        else:
            similarity = lexical.similarity_weights(state_tokens, [question.instructions])[0]
            evidence = similarity - self.prior
        return NoulAnswer(noul=round(lexical.logistic(self.evidence_scale * evidence), 6))

    # ---------------------------------------------------------------- public
    def decide(self, state: str | dict[str, Any] | list[Any],
               questions: dict[str, Question]) -> dict[str, Answer]:
        """Answer every question against the same state, in one pass."""
        state_tokens = lexical.tokenize(state_text(state))
        answers: dict[str, Answer] = {}
        for name, question in questions.items():
            if isinstance(question, ChoiceQuestion):
                answers[name] = self._answer_choice(question, state_tokens)
            elif isinstance(question, ScoreQuestion):
                answers[name] = self._answer_score(question, state_tokens)
            else:
                answers[name] = self._answer_noul(question, state_tokens)
        return answers
class LocalModelBackend:
    """Free local verdicts from llama.cpp, blended into the weighted answer.

    The 1.5B model is asked one grammar-constrained question per decision (in
    parallel, mirroring Jev's "questions are evaluated in isolation" rule), and
    its verdict is blended with the deterministic distribution at
    ``1 - blend`` / ``blend`` weights. The blend is what keeps the output a
    real distribution with a real confidence, and lets a model timeout fall
    back to a deterministic answer instead of an error.
    """

    name = "local"
    free = True

    def __init__(self, client: Any, *, blend: float = 0.6, max_tokens: int = 96,
                 temperature: float = 0.0, fallback: DeterministicBackend | None = None):
        self.client = client
        self.blend = min(1.0, max(0.0, float(blend)))
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.fallback = fallback or DeterministicBackend()
        # Names of questions whose model call failed and used the fallback.
        self.degraded: set[str] = set()

    # ------------------------------------------------------------- prompting
    @staticmethod
    def _schema(question: Question) -> dict:
        confidence = {"type": "number", "minimum": 0, "maximum": 1}
        if isinstance(question, ChoiceQuestion):
            properties = {"choice": {"type": "string", "enum": list(question.criteria)}, "confidence": confidence}
            required = ["choice", "confidence"]
        elif isinstance(question, ScoreQuestion):
            properties = {
                "level": {"type": "integer", "minimum": 0, "maximum": len(question.criteria) - 1},
                "confidence": confidence,
            }
            required = ["level", "confidence"]
        else:
            properties = {"yes": {"type": "boolean"}, "confidence": confidence}
            required = ["yes", "confidence"]
        return {
            "type": "object",
            "properties": properties,
            "required": required,
            "additionalProperties": False,
        }

    @staticmethod
    def _messages(question: Question, text: str) -> list[dict]:
        if isinstance(question, ChoiceQuestion):
            options = "\n".join(
                f"- {label}: {description or label}" for label, description in question.criteria.items()
            )
        elif isinstance(question, ScoreQuestion):
            options = "\n".join(f"- {index}: {level}" for index, level in enumerate(question.criteria))
        else:
            options = "Answer yes or no."
        system = (
            "You answer one typed question about a state. "
            "Reply with JSON matching the schema. Do not explain."
        )
        user = (
            f"STATE:\n{text}\n\nQUESTION: {question.instructions}\nOPTIONS:\n{options}\n\n"
            "Give your verdict and a confidence between 0 and 1."
        )
        return [{"role": "system", "content": system}, {"role": "user", "content": user}]

    # --------------------------------------------------------------- calling
    @staticmethod
    def _parse(question: Question, raw: str) -> tuple[Any, float]:
        """Pull the verdict and confidence out of the model's JSON reply."""
        data = json.loads(raw)
        if not isinstance(data, dict):
            raise ValueError("model reply was not a JSON object")
        confidence = lexical.clamp01(float(data.get("confidence", 0.0)))
        if isinstance(question, ChoiceQuestion):
            choice = str(data.get("choice", ""))
            if choice not in question.criteria:
                raise ValueError(f"model chose unknown option {choice!r}")
            return choice, confidence
        if isinstance(question, ScoreQuestion):
            level = int(data.get("level", -1))
            if not 0 <= level < len(question.criteria):
                raise ValueError(f"model chose invalid level {level!r}")
            return level, confidence
        return bool(data.get("yes", False)), confidence

    async def _ask(self, question: Question, text: str) -> tuple[Any, float]:
        raw = await self.client.chat(
            self._messages(question, text),
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            schema=self._schema(question),
        )
        return self._parse(question, raw)

    @staticmethod
    def _blend(one_hot: list[float], deterministic: list[float], confidence: float) -> list[float]:
        return [
            confidence * hot + (1.0 - confidence) * base
            for hot, base in zip(one_hot, deterministic, strict=True)
        ]

    # ---------------------------------------------------------------- public
    async def adecide(self, state: str | dict[str, Any] | list[Any],
                      questions: dict[str, Question]) -> dict[str, Answer]:
        """Answer every question, falling back per question when the model fails."""
        text = state_text(state)
        deterministic = self.fallback.decide(text, questions)
        self.degraded = set()

        async def one(name: str, question: Question) -> tuple[str, Answer]:
            base = deterministic[name]
            try:
                verdict, confidence = await self._ask(question, text)
            except (json.JSONDecodeError, ValueError, KeyError, TypeError, OSError, RuntimeError, asyncio.TimeoutError):
                self.degraded.add(name)
                return name, base
            return name, self._merge(question, base, verdict, confidence)

        results = await asyncio.gather(*(one(name, question) for name, question in questions.items()))
        return dict(results)

    def _merge(self, question: Question, base: Answer, verdict: Any, confidence: float) -> Answer:
        """Blend a model verdict into the deterministic distribution."""
        weight = self.blend * confidence
        if isinstance(question, ChoiceQuestion):
            labels = list(question.criteria)
            deterministic_mass = [base.probabilities[label] for label in labels]
            one_hot = [1.0 if label == verdict else 0.0 for label in labels]
            mass = _round_mass(self._blend(one_hot, deterministic_mass, weight))
            best = max(range(len(labels)), key=lambda index: mass[index])
            return ChoiceAnswer(choice=labels[best], confidence=mass[best],
                                probabilities=dict(zip(labels, mass, strict=True)))
        if isinstance(question, ScoreQuestion):
            deterministic_mass = [base.probabilities[str(index)] for index in range(len(question.criteria))]
            one_hot = [1.0 if index == verdict else 0.0 for index in range(len(question.criteria))]
            mass = _round_mass(self._blend(one_hot, deterministic_mass, weight))
            expected = sum(index * probability for index, probability in enumerate(mass))
            return ScoreAnswer(score=round(expected, 6), confidence=max(mass),
                               probabilities={str(index): probability for index, probability in enumerate(mass)})
        yes = 1.0 if verdict else 0.0
        base_noul = base.noul if isinstance(base, NoulAnswer) else 0.5
        blended = weight * yes + (1.0 - weight) * base_noul
        return NoulAnswer(noul=round(lexical.clamp01(blended), 6))

    def decide(self, state: str | dict[str, Any] | list[Any],
               questions: dict[str, Question]) -> dict[str, Answer]:
        """Synchronous entry point; use ``adecide`` inside a running loop."""
        return asyncio.run(self.adecide(state, questions))


class JevApiBackend:
    """The hosted Jev API — **paid**, opt-in, never selected implicitly.

    TypeSafe bills per input token against a prepaid balance, so this backend
    is only constructed when a key is present and the caller asked for it. Its
    only advantage over the free backends is model quality; the request and
    response payloads are identical, which is the point of speaking the same
    contract.
    """

    name = "jev"
    free = False

    def __init__(self, api_key: str, *, base_url: str | None = None, timeout: float = 30.0,
                 model: str | None = None):
        if not api_key:
            raise BackendUnavailable("a Jev API key is required (JEV_API_KEY or TYPESAFE_API_KEY)")
        self.api_key = api_key
        # Two endpoints speak this contract: the managed wrapper (keys look
        # like ``jv_live_...``) and the official TypeSafe API, which requires
        # a ``model`` field in the body. The key prefix tells them apart.
        self.base_url = (base_url or os.environ.get(JEV_API_URL_ENV)
                         or (DEFAULT_JEV_URL if api_key.startswith("jv_")
                             else DEFAULT_TYPESAFE_URL)).rstrip("/")
        self.timeout = timeout
        self.model = model or os.environ.get(JEV_MODEL_ENV) or _DEFAULT_MODEL
        self.last_usage: dict[str, Any] = {}


    # --------------------------------------------------------------- helpers
    @staticmethod
    def api_key_from_env() -> str | None:
        return os.environ.get(JEV_API_KEY_ENV) or os.environ.get(TYPESAFE_API_KEY_ENV) or None

    @classmethod
    def available(cls) -> bool:
        """True when a key is configured (the backend is still opt-in)."""
        return cls.api_key_from_env() is not None

    # ---------------------------------------------------------------- calling
    async def adecide(self, state: str | dict[str, Any] | list[Any],
                      questions: dict[str, Question]) -> dict[str, Answer]:
        import httpx
        from pydantic import TypeAdapter

        payload: dict[str, Any] = {"state": state, "questions": questions_payload(questions)}
        if self.model:
            payload["model"] = self.model
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(self.base_url, json=payload, headers=headers)
        except httpx.HTTPError as exc:
            raise BackendUnavailable(f"Jev API request failed: {exc}") from exc
        if response.status_code != 200:
            raise BackendUnavailable(f"Jev API returned HTTP {response.status_code}: {response.text[:300]}")
        data = response.json()
        self.last_usage = dict(data.get("usage") or {})
        answers = TypeAdapter(dict[str, Answer]).validate_python(data.get("answers") or {})
        missing = [name for name in questions if name not in answers]
        if missing:
            raise BackendUnavailable(f"Jev API omitted answers for {missing}")
        return answers

    def decide(self, state: str | dict[str, Any] | list[Any],
               questions: dict[str, Question]) -> dict[str, Answer]:
        return asyncio.run(self.adecide(state, questions))


# ---------------------------------------------------------------------------
# Selection
# ---------------------------------------------------------------------------

def backend_catalog() -> list[dict[str, Any]]:
    """Describe every backend so a client can explain the choices it offers."""
    return [
        {"name": DEFAULT_BACKEND, "free": True, "available": True,
         "requires": "nothing (offline, deterministic, reproducible)"},
        {"name": "local", "free": True, "available": True,
         "requires": "a running llama.cpp server (FORGECODER_LLAMA_URL)"},
        {"name": "jev", "free": False, "available": JevApiBackend.available(),
         "requires": f"paid Jev API key in {JEV_API_KEY_ENV} or {TYPESAFE_API_KEY_ENV}"
                     f" (jv_live_... -> {DEFAULT_JEV_URL}, otherwise the official"
                     f" TypeSafe API at {DEFAULT_TYPESAFE_URL})"},
    ]


def resolve_backend(name: str | None = None, *, client: Any = None,
                    allow_paid: bool = False) -> DecisionBackend:
    """Pick a backend. Defaults to the free deterministic one, always.

    ``allow_paid`` exists so a caller can veto the paid backend even when the
    environment asks for it; with the default (``False``) an explicit
    ``backend="jev"`` request fails loudly rather than silently billing.
    """
    requested = (name if name is not None else os.environ.get(BACKEND_ENV)) or DEFAULT_BACKEND
    requested = requested.strip().lower()

    if requested in {DEFAULT_BACKEND, "offline"}:
        return DeterministicBackend()
    if requested == "local":
        existing = client or _default_inference_client()
        return LocalModelBackend(existing)
    if requested == "jev":
        if not allow_paid:
            raise BackendUnavailable(
                "the Jev API backend is billed per token; pass allow_paid=True, or "
                f"set {BACKEND_ENV}=local for a free model-backed verdict"
            )
        key = JevApiBackend.api_key_from_env()
        if not key:
            raise BackendUnavailable(
                f"the 'jev' backend requires {JEV_API_KEY_ENV} or {TYPESAFE_API_KEY_ENV}"
            )
        return JevApiBackend(key)
    raise BackendUnavailable(
        f"unknown backend {requested!r}; expected one of "
        f"{[entry['name'] for entry in backend_catalog()]}"
    )


def _default_inference_client() -> Any:
    """Build a client for the configured local llama.cpp server."""
    from core.inference.client import DEFAULT_BASE_URL, InferenceClient

    base_url = os.environ.get("FORGECODER_LLAMA_URL") or os.environ.get("LLAMA_URL")
    return InferenceClient(base_url or DEFAULT_BASE_URL)


__all__ = [
    "BACKEND_ENV",
    "DEFAULT_BACKEND",
    "DEFAULT_JEV_URL",
    "DEFAULT_TYPESAFE_URL",
    "JEV_API_KEY_ENV",
    "JEV_API_URL_ENV",
    "JEV_MODEL_ENV",
    "TYPESAFE_API_KEY_ENV",
    "BackendUnavailable",
    "DecisionBackend",
    "DeterministicBackend",
    "JevApiBackend",
    "LocalModelBackend",
    "backend_catalog",
    "load_local_env",
    "parse_env_file",
    "resolve_backend",
]

