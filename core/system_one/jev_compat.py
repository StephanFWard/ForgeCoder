"""The upstream `jev` decorator API, on Python 3.10+ and free by default.

PyPI's ``jev`` package turns a Python function definition into a Jev query:
the signature is the specification, the return annotation says what shape the
answer takes, and the docstring supplies the judgment. It is a good API, and
this module re-implements its public surface on top of
:mod:`core.system_one`, with two deliberate differences:

* **Python 3.10+ instead of >= 3.14.** Upstream imports ``typing.TypeIs``,
  which does not exist before 3.13, so it cannot be installed into
  ForgeCoder's 3.11 runtime venv at all (``pip`` refuses on
  ``Requires-Python >=3.14``).
* **Free backends by default instead of a paid key.** Upstream requires
  ``TYPESAFE_API_KEY`` and bills per input token. Here the same call is
  answered by the deterministic or local backend unless the caller explicitly
  opts into the hosted API.

The field mapping is upstream's, field for field:

======================  ================  ===================================
Annotation              Jev question      Coerced back as
======================  ================  ===================================
``bool``                noul              ``p(yes) >= threshold`` (0.5)
``Literal[...]``        choice            the selected label
``Enum``                choice            the selected member
``int`` (ge/le)         score             ``lo + round(expected)``
``float`` (ge/le)       score             linear interpolation over levels
======================  ================  ===================================

``Field(description=...)`` becomes the question instructions (without one the
field name is humanized: ``is_urgent`` -> "is urgent"). Score levels default to
the numbers in range and can be overridden with
``Field(..., json_schema_extra={"levels": [...]})``. Anything else (``str``,
lists, nested models, ``Optional``) raises ``TypeError`` at decoration time,
because Jev cannot produce those values.

Usage is the upstream usage::

    import core.system_one.jev_compat as jev

    class Triage(jev.BaseModel):
        department: Literal["billing", "technical"]
        is_urgent: bool

    Triage.decide("I was charged twice and nobody replied.")

Difference worth knowing: upstream renders docstrings with Jinja2; this module
uses a small ``{{ name }}`` substitution so the runtime needs no new
dependency.
"""
from __future__ import annotations

import asyncio
import inspect
import json
import re
import typing
from enum import Enum
from typing import Any, ClassVar, Literal, get_args, get_origin

from pydantic import BaseModel as PydanticBaseModel

from core.system_one.decide import Decider
from core.system_one.primitives import (
    MAX_CHOICE_OPTIONS,
    MAX_SCORE_LEVELS,
    MIN_SCORE_LEVELS,
    ChoiceQuestion,
    NoulQuestion,
    Question,
    ScoreQuestion,
)

DEFAULT_BOOL_THRESHOLD = 0.5
_TEMPLATE_RE = re.compile(r"\{\{\s*([A-Za-z_][A-Za-z0-9_]*)\s*\}\}")


class _StateMarker:
    """Returned by ``fn.state()`` inside a decorated body: 'this is the state'."""

    __slots__ = ()

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return "<jev state>"


_STATE = _StateMarker()


def render_state(template: str, values: dict[str, Any]) -> str:
    """Substitute ``{{ name }}`` with bound arguments (missing names -> "")."""
    return _TEMPLATE_RE.sub(lambda match: str(values.get(match.group(1), "")), template)


def humanize(name: str) -> str:
    """``is_urgent`` -> ``is urgent`` (upstream's fallback instructions)."""
    return name.replace("_", " ").strip()


# ---------------------------------------------------------------------------
# Annotation resolution
# ---------------------------------------------------------------------------

def _frame_locals(target: Any, max_depth: int = 6) -> dict[str, Any]:
    """Merge the calling frames' locals, nearest frame winning.

    Needed for models declared inside a function (or a notebook cell): a string
    annotation from ``from __future__ import annotations`` names a class that
    is in nobody's module globals. Frames are read, never kept.
    """
    merged: dict[str, Any] = {}
    frame = inspect.currentframe()
    try:
        frame = frame.f_back if frame else None
        for _ in range(max_depth):
            if frame is None:
                break
            for name, value in frame.f_locals.items():
                merged.setdefault(name, value)
            frame = frame.f_back
    finally:
        del frame
    return merged


def resolve_return_annotation(target: Any) -> Any:
    """Resolve a return annotation that may still be a string.

    Order: the object itself, then ``typing.get_type_hints`` (module-level
    definitions — the normal case), then an evaluation against the merged
    caller frames so a locally defined model still works. An unresolvable
    annotation is returned as-is, which the caller reports as a TypeError.
    """
    annotation = inspect.signature(target).return_annotation
    if isinstance(annotation, type) or annotation is inspect.Signature.empty:
        return annotation
    try:
        resolved = typing.get_type_hints(target).get("return")
        if resolved is not None:
            return resolved
    except Exception:  # NameError/TypeError for locals, SyntaxError for odd hints
        pass
    if isinstance(annotation, str):
        namespace = dict(getattr(target, "__globals__", {}) or {})
        namespace.update(_frame_locals(target))
        try:
            return eval(annotation, namespace)  # noqa: S307 - repo-owned annotation text
        except Exception:
            return annotation
    return annotation


# ---------------------------------------------------------------------------
# Compilation: pydantic field -> typed question (+ the inverse coercion)
# ---------------------------------------------------------------------------

def _bounds(info: Any) -> tuple[float | None, float | None]:
    """Read ``ge``/``le`` out of a field's constraint metadata."""
    ge = le = None
    for constraint in getattr(info, "metadata", ()) or ():
        if hasattr(constraint, "ge"):
            ge = constraint.ge
        if hasattr(constraint, "le"):
            le = constraint.le
    return ge, le


def _levels(name: str, info: Any, low: float, high: float) -> list[str]:
    """Score levels: explicit ``json_schema_extra['levels']`` or the range."""
    extra = getattr(info, "json_schema_extra", None) or {}
    explicit = extra.get("levels") if isinstance(extra, dict) else None
    if explicit:
        levels = [str(level) for level in explicit]
    else:
        if not float(low).is_integer() or not float(high).is_integer():
            raise TypeError(
                f"{name}: a fractional score range needs explicit levels, e.g. "
                f"Field(ge={low}, le={high}, json_schema_extra={{'levels': [...]}})"
            )
        levels = [str(number) for number in range(int(low), int(high) + 1)]
    if not MIN_SCORE_LEVELS <= len(levels) <= MAX_SCORE_LEVELS:
        raise TypeError(
            f"{name}: Jev scores take {MIN_SCORE_LEVELS}-{MAX_SCORE_LEVELS} levels, "
            f"got {len(levels)}; narrow ge/le or pass json_schema_extra={{'levels': [...]}}"
        )
    return levels


def compile_field(name: str, info: Any) -> tuple[Question, dict[str, Any]]:
    """Compile one model field into a question plus its coercion metadata.

    Unsupported annotations raise ``TypeError`` here rather than at the first
    call, so a bad model fails at import/decoration time (upstream behaviour).
    """
    annotation = info.annotation
    instructions = (getattr(info, "description", None) or humanize(name))[:2000]

    if annotation is bool:
        return NoulQuestion(instructions=instructions), {"kind": "bool"}

    origin = get_origin(annotation)
    if origin is Literal:
        labels = [str(label) for label in get_args(annotation)]
        if len(labels) > MAX_CHOICE_OPTIONS:
            raise TypeError(f"{name}: a choice takes at most {MAX_CHOICE_OPTIONS} options, got {len(labels)}")
        return ChoiceQuestion(instructions=instructions, criteria=dict.fromkeys(labels)), {"kind": "literal"}

    if isinstance(annotation, type) and issubclass(annotation, Enum):
        members = list(annotation)
        if len(members) > MAX_CHOICE_OPTIONS:
            raise TypeError(f"{name}: a choice takes at most {MAX_CHOICE_OPTIONS} options, got {len(members)}")
        return (
            ChoiceQuestion(instructions=instructions, criteria={member.name: None for member in members}),
            {"kind": "enum", "enum": annotation},
        )

    if annotation in {int, float}:
        low, high = _bounds(info)
        if low is None or high is None:
            raise TypeError(f"{name}: a score needs Field(ge=..., le=...) bounds")
        if high <= low:
            raise TypeError(f"{name}: a score needs le > ge")
        levels = _levels(name, info, low, high)
        return (
            ScoreQuestion(instructions=instructions, criteria=levels),
            {"kind": "int" if annotation is int else "float", "low": low, "high": high, "levels": len(levels)},
        )

    raise TypeError(
        f"{name}: {annotation!r} cannot be produced by Jev; use bool, Literal, Enum, "
        "or int/float with ge/le bounds"
    )


def compile_questions(model: type[PydanticBaseModel]) -> dict[str, Question]:
    """Compile a pydantic model into named typed questions."""
    return compile_model(model)[0]


def compile_model(model: type[PydanticBaseModel]) -> tuple[dict[str, Question], dict[str, dict[str, Any]]]:
    """Compile a model once: questions plus the coercion metadata per field."""
    questions: dict[str, Question] = {}
    metas: dict[str, dict[str, Any]] = {}
    for name, info in model.model_fields.items():
        questions[name], metas[name] = compile_field(name, info)
    return questions, metas


_COMPILED: dict[type[PydanticBaseModel], tuple[dict[str, Question], dict[str, dict[str, Any]]]] = {}


def _compiled(model: type[PydanticBaseModel]) -> tuple[dict[str, Question], dict[str, dict[str, Any]]]:
    """Memoised :func:`compile_model`; hot loops must not recompile per call."""
    if model not in _COMPILED:
        _COMPILED[model] = compile_model(model)
    return _COMPILED[model]


def _threshold(model: type[PydanticBaseModel], override: float | None) -> float:
    if override is not None:
        return float(override)
    return float(getattr(model, "__jev_bool_threshold__", DEFAULT_BOOL_THRESHOLD))


def _build(model: type[PydanticBaseModel], answers: dict[str, Any], metas: dict[str, dict[str, Any]],
           threshold: float) -> PydanticBaseModel:
    values = {
        name: _coerce(name, model.model_fields[name], metas[name], answers[name], threshold)
        for name in model.model_fields
    }
    return model(**values)


# ---------------------------------------------------------------------------
# Function form: decide(state, Model) / adecide(state, Model)
# ---------------------------------------------------------------------------

def decide(state: Any, model: type[PydanticBaseModel], *,
           bool_threshold: float | None = None, backend: Any = None,
           decider: Decider | None = None) -> PydanticBaseModel:
    """Decide ``model`` from ``state`` — upstream's ``jev.decide``.

    Pin a hosted model version by constructing the backend yourself
    (``Decider(backend=JevApiBackend(key, model="jev-1.13.0"))``); the free
    backends have nothing to pin.
    """
    questions, metas = _compiled(model)
    active = decider or Decider(backend=backend)
    response = active.decide(state, questions)
    return _build(model, response.answers, metas, _threshold(model, bool_threshold))


async def adecide(state: Any, model: type[PydanticBaseModel], *,
                  bool_threshold: float | None = None, backend: Any = None,
                  decider: Decider | None = None) -> PydanticBaseModel:
    """Async form of :func:`decide`."""
    questions, metas = _compiled(model)
    active = decider or Decider(backend=backend)
    response = await active.adecide(state, questions)
    return _build(model, response.answers, metas, _threshold(model, bool_threshold))


class BaseModel(PydanticBaseModel):
    """Upstream's ``jev.BaseModel``: a pydantic model that can decide itself.

    Subclass attributes ``__jev_model__`` (pin a model version) and
    ``__jev_bool_threshold__`` (bool cutoff) mirror the upstream knobs.
    """

    __jev_model__: ClassVar[str | None] = None
    __jev_bool_threshold__: ClassVar[float] = DEFAULT_BOOL_THRESHOLD
    __jev_questions__: ClassVar[dict[str, Question] | None] = None

    @classmethod
    def __pydantic_init_subclass__(cls, **kwargs: Any) -> None:
        super().__pydantic_init_subclass__(**kwargs)
        questions, metas = compile_model(cls)
        cls.__jev_questions__ = questions
        _COMPILED[cls] = (questions, metas)

    @classmethod
    def decide(cls, state: Any, **kwargs: Any) -> Any:
        """Decide an instance of this model from ``state``."""
        return decide(state, cls, decider=kwargs.pop("decider", None),
                      backend=kwargs.pop("backend", None),
                      bool_threshold=kwargs.pop("bool_threshold", None))

    @classmethod
    async def adecide(cls, state: Any, **kwargs: Any) -> Any:
        """Async form of :meth:`decide`."""
        return await adecide(state, cls, decider=kwargs.pop("decider", None),
                             backend=kwargs.pop("backend", None),
                             bool_threshold=kwargs.pop("bool_threshold", None))


# ---------------------------------------------------------------------------
# Function form: @fn
# ---------------------------------------------------------------------------

class JevFn:
    """A decorated function whose body contributes the state, not the answer.

    The body still runs (upstream does the same), so it can compute or fetch
    the state. Returning ``fn.state()`` marks the rendered docstring as the
    whole state; returning a model instance short-circuits the call, which is
    how tests run a "decided" path without any backend.
    """

    def __init__(self, func: Any, *, model: type[PydanticBaseModel],
                 bool_threshold: float | None = None, backend: Any = None,
                 decider: Decider | None = None):
        self._func = func
        self._model = model
        self._questions, self._metas = _compiled(model)
        self._bool_threshold = bool_threshold
        self._backend = backend
        self._decider = decider
        self.__name__ = getattr(func, "__name__", type(func).__name__)
        self.__doc__ = getattr(func, "__doc__", None)
        self.__wrapped__ = func

    # -------------------------------------------------------------- the state
    def state(self, *args: Any, **kwargs: Any) -> Any:
        """Render the state, or mark it when called bare inside the body.

        * ``fn.state()`` inside the body -> marker: "the rendered docstring is
          the state" (upstream's body-less form).
        * ``fn.state(<args>)`` -> the exact state text, for assertions.
        """
        if not args and not kwargs:
            return _STATE
        bound = self._bind(args, kwargs)
        template = (self.__doc__ or "").strip()
        if template:
            return render_state(template, bound)
        return json.dumps(bound, ensure_ascii=False, sort_keys=True, default=str)

    # `state_payload` is the explicit name; `state` is the upstream one.
    state_payload = state

    @property
    def questions(self) -> dict[str, Question]:
        """The typed questions the return annotation compiled into."""
        return dict(self._questions)

    # --------------------------------------------------------------- deciding
    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        rendered = self._run(args, kwargs)
        if isinstance(rendered, self._model):
            return rendered
        state = self.state(*args, **kwargs) if rendered is _STATE else rendered
        return decide(state, self._model, bool_threshold=self._bool_threshold,
                      backend=self._backend, decider=self._decider)

    async def adecide(self, *args: Any, **kwargs: Any) -> Any:
        """Async form of :meth:`__call__`."""
        rendered = self._run(args, kwargs)
        if isinstance(rendered, self._model):
            return rendered
        state = self.state(*args, **kwargs) if rendered is _STATE else rendered
        return await adecide(state, self._model, bool_threshold=self._bool_threshold,
                             backend=self._backend, decider=self._decider)

    def map(self, items: Any) -> list[Any]:
        """Decide once per item (upstream's ``fn.map``)."""
        return [self(item) for item in items]

    async def amap(self, items: Any) -> list[Any]:
        """Async :meth:`map`; items are decided concurrently."""
        return list(await asyncio.gather(*(self.adecide(item) for item in items)))

    # --------------------------------------------------------------- internals
    def _bind(self, args: tuple, kwargs: dict) -> dict[str, Any]:
        bound = inspect.signature(self._func).bind_partial(*args, **kwargs)
        bound.apply_defaults()
        return dict(bound.arguments)

    def _run(self, args: tuple, kwargs: dict) -> Any:
        """Run the body; a returned value must be the state marker or a model."""
        result = self._func(*args, **kwargs)
        if result is _STATE or result is None or isinstance(result, self._model):
            return result
        raise TypeError(
            f"{getattr(self, '__name__', 'fn')} must return fn.state() (body-less form), "
            f"a {self._model.__name__} instance (short circuit), or None; got {result!r}"
        )


def fn(func: Any = None, *, model: type[PydanticBaseModel] | None = None,
       bool_threshold: float | None = None, backend: Any = None,
       decider: Decider | None = None) -> Any:
    """Compile a function definition into a Jev decision (upstream's ``@jev.fn``).

    Usable bare (``@fn``) or configured (``@fn(backend="local")``). The return
    annotation must be a pydantic model — anything else raises ``TypeError`` at
    decoration time, exactly as upstream does.
    """

    def decorate(target: Any) -> JevFn:
        annotation = resolve_return_annotation(target)
        if not (isinstance(annotation, type) and issubclass(annotation, PydanticBaseModel)):
            raise TypeError(
                f"{getattr(target, '__name__', 'fn')} must be annotated with a pydantic model, "
                f"got {annotation!r}"
            )
        return JevFn(target, model=annotation, bool_threshold=bool_threshold,
                     backend=backend, decider=decider)

    if func is not None:
        return decorate(func)
    return decorate


__all__ = [
    "DEFAULT_BOOL_THRESHOLD",
    "BaseModel",
    "JevFn",
    "adecide",
    "compile_field",
    "compile_model",
    "compile_questions",
    "decide",
    "fn",
    "humanize",
    "render_state",
]


def _coerce(name: str, info: Any, meta: dict[str, Any], answer: Any, threshold: float) -> Any:
    """Turn a typed answer back into the field's declared annotation."""
    kind = meta["kind"]
    if kind == "bool":
        return answer.noul >= threshold
    if kind == "literal":
        labels = list(answer.probabilities) or [str(label) for label in get_args(info.annotation)]
        if answer.choice not in labels:
            raise ValueError(f"{name}: model chose unknown option {answer.choice!r}")
        return answer.choice
    if kind == "enum":
        try:
            return meta["enum"][answer.choice]
        except KeyError as exc:
            raise ValueError(f"{name}: model chose unknown member {answer.choice!r}") from exc
    if kind == "int":
        return int(meta["low"]) + round(answer.score)
    span = float(meta["high"]) - float(meta["low"])
    levels = max(1, int(meta["levels"]) - 1)
    return float(meta["low"]) + (answer.score / levels) * span
