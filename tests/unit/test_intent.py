"""The intent gate: plain questions must not be framed as edit tasks.

Regression root: "Is a hot dog a sandwich?" used to get the edit task frame,
whose unknowables ("no acceptance command was stated; name the command that
proves the change") came back as fabricated "[warn] no-unverified-test-claims"
lines instead of an answer.
"""
from __future__ import annotations

import asyncio
import json

from core.agent.intent import (
    CODE_CHANGE_THRESHOLD,
    classify_intent,
    classify_intent_sync,
    heuristic_code_change,
)
from core.retrieval.context import ContextBuilder

HOT_DOG = "Is a hot dog a sandwich?"
EDIT_REQUEST = "add null validation to the service"
CODED_QUESTION = "fix the null pointer bug in auth.py"


class StubClient:
    """Minimal llama.cpp stand-in for LocalModelBackend: canned JSON verdicts."""

    def __init__(self, yes: bool, confidence: float = 0.95):
        self.yes = yes
        self.confidence = confidence
        self.calls: list[list[dict]] = []

    async def chat(self, messages: list[dict], **_: object) -> str:
        self.calls.append(messages)
        return json.dumps({"yes": self.yes, "confidence": self.confidence})


# ---------------------------------------------------------------- heuristic

def test_heuristic_flags_plain_questions_low():
    assert heuristic_code_change(HOT_DOG) < 0.2
    assert heuristic_code_change("What is the capital of France?") < 0.2


def test_heuristic_flags_edit_requests_high():
    assert heuristic_code_change(EDIT_REQUEST) >= 0.5
    assert heuristic_code_change(CODED_QUESTION) >= 0.5


def test_heuristic_counts_editor_context_as_evidence():
    question = "why does this method return null?"
    assert heuristic_code_change(question, has_file=True, has_selection=True) \
        > heuristic_code_change(question)
    # An open editor is itself evidence of a coding session: never full task.
    assert heuristic_code_change(question, has_file=True) < 1.0


def test_heuristic_is_deterministic():
    first = heuristic_code_change(HOT_DOG)
    assert first == heuristic_code_change(HOT_DOG)


# ------------------------------------------------------------ blended gate

def test_sync_gate_routes_the_hot_dog_question_to_conversation():
    decision = classify_intent_sync(HOT_DOG, client=StubClient(yes=False))
    assert decision.code_change is False
    assert decision.confidence < CODE_CHANGE_THRESHOLD
    assert decision.system_one_free is True
    payload = decision.to_dict()
    assert payload["parts"]["heuristic"] < 0.2


def test_sync_gate_keeps_real_edit_requests_as_tasks():
    decision = classify_intent_sync(EDIT_REQUEST, client=StubClient(yes=True))
    assert decision.code_change is True
    decision = classify_intent_sync(CODED_QUESTION, client=StubClient(yes=True))
    assert decision.code_change is True


def test_async_gate_matches_the_sync_one():
    decision = asyncio.run(classify_intent(HOT_DOG, client=StubClient(yes=False)))
    assert decision.code_change is False
    decision = asyncio.run(classify_intent(EDIT_REQUEST, client=StubClient(yes=True)))
    assert decision.code_change is True


def test_model_verdict_moves_the_blend_in_its_direction():
    # A middling message whose lexical prior is ambiguous: the model's verdict
    # must move the blend toward its answer, whichever way it goes.
    message = "can you look at the service?"
    prior = heuristic_code_change(message)
    no_model = classify_intent_sync(message, client=StubClient(yes=False, confidence=0.99))
    yes_model = classify_intent_sync(message, client=StubClient(yes=True, confidence=0.99))
    assert no_model.confidence < yes_model.confidence
    # The blend is the documented weighted mean: with the same System One
    # answer the parts differ only by the heuristic prior's share.
    assert yes_model.parts["heuristic"] == no_model.parts["heuristic"] == round(prior, 6)
    assert yes_model.system_one_backend == "local"


def test_confident_model_yes_can_flip_an_ambiguous_task():
    # "could you update the authentication module for the new api" scores a
    # middling prior (question form) but is a real task; the model's confident
    # yes must carry it over the line.
    message = "could you update the authentication module for the new api"
    decision = classify_intent_sync(message, client=StubClient(yes=True, confidence=0.99))
    assert decision.code_change is True


def test_gate_degrades_to_heuristic_only_when_no_client():
    decision = classify_intent_sync(HOT_DOG)
    assert decision.code_change is False
    assert decision.system_one_backend in {"deterministic", "heuristic-only"}


def test_gate_never_raises_when_the_model_errors():
    class BrokenClient:
        async def chat(self, *a: object, **k: object) -> str:
            raise OSError("llama-server down")

    decision = classify_intent_sync(EDIT_REQUEST, client=BrokenClient())
    # The model failed, the local backend degraded to its deterministic view,
    # and the heuristic prior still routes the edit request correctly.
    assert decision.code_change is True


# ------------------------------------------------- ContextBuilder wiring

def test_context_builder_skips_the_task_frame_for_questions():
    built = ContextBuilder().build(HOT_DOG, code_change=False)
    assert built.frame is None
    assert built.contract is None
    assert "TASK FRAME" not in built.request
    assert "acceptance" not in built.request.lower()
    assert "not a code-change" in built.system


def test_context_builder_keeps_the_frame_for_tasks_by_default():
    built = ContextBuilder().build(EDIT_REQUEST)
    assert built.frame is not None
    assert "TASK FRAME" in built.request


def test_context_builder_code_change_true_behaves_like_default():
    framed = ContextBuilder().build(EDIT_REQUEST, code_change=True)
    assert framed.frame is not None
    assert "TASK FRAME" in framed.request


def test_question_system_prompt_has_no_rule_slugs_or_markers():
    """A 1.5B model shown "[warn] ask-dont-guess" parrots it back as its
    answer. A question turn must carry neither slugs nor markers."""
    built = ContextBuilder().build(HOT_DOG, code_change=False)
    assert "ask-dont-guess" not in built.system
    assert "no-unverified-test-claims" not in built.system
    assert "[warn]" not in built.system
    assert "[block]" not in built.system
    assert "OPERATING RULES" not in built.system
    assert "answer" in built.system.lower()


def test_chat_question_path_builds_marker_free_prompt():
    """End to end for the streaming path: the intent gate routes the hot-dog
    question to conversation, and the prompt Qwen sees has no markers."""
    decision = classify_intent_sync(HOT_DOG, client=StubClient(yes=False))
    assert decision.code_change is False
    built = ContextBuilder().build(HOT_DOG, code_change=decision.code_change)
    assert built.frame is None
    assert "ask-dont-guess" not in built.system
    assert "[warn]" not in built.system


def test_task_system_prompt_keeps_its_rules():
    framed = ContextBuilder().build(EDIT_REQUEST, code_change=True)
    assert "ask-dont-guess" in framed.system

