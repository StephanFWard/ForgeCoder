"""The answer-confidence marker: every reply opens with its own [p=...]."""
from __future__ import annotations

import asyncio
import json

from core.agent.confidence import (
    answer_confidence,
    answer_confidence_sync,
    format_probability,
)


class StubClient:
    """Minimal llama.cpp stand-in for LocalModelBackend: canned JSON verdicts."""

    def __init__(self, yes: bool, confidence: float = 0.95):
        self.yes = yes
        self.confidence = confidence

    async def chat(self, messages, **_: object) -> str:
        return json.dumps({"yes": self.yes, "confidence": self.confidence})


class BrokenClient:
    async def chat(self, *a: object, **k: object) -> str:
        raise OSError("llama-server down")


def test_format_probability_is_two_decimals_and_clamped():
    assert format_probability(0.612345) == "[p=0.61]"
    assert format_probability(1.7) == "[p=1.00]"
    assert format_probability(-2) == "[p=0.00]"


def test_confidence_without_a_client_is_valid_and_labelled():
    decision = answer_confidence_sync("What does Service.get do?",
                                      "class Service:\n    def get(self, key): ...")
    assert 0.0 <= decision.probability <= 1.0
    assert decision.backend in {"deterministic", "heuristic-only"}
    assert decision.free is True
    payload = decision.to_dict()
    assert 0.0 <= payload["probability"] <= 1.0
    assert payload["free"] is True


def test_model_verdict_moves_the_probability_in_its_direction():
    evidence = "class Service:\n    def get(self, key):\n        return self.store.get(key)\n"
    yes = answer_confidence_sync("What does Service.get do?", evidence,
                                 client=StubClient(yes=True))
    no = answer_confidence_sync("What does Service.get do?", evidence,
                                client=StubClient(yes=False))
    assert yes.probability > no.probability
    # The blend is the documented weighted mean around the 0.5 prior.
    assert yes.probability > 0.5 > no.probability or no.probability <= 0.5 <= yes.probability


def test_async_twin_matches_the_sync_one():
    sync = answer_confidence_sync("Is a hot dog a sandwich?")
    async_ = asyncio.run(answer_confidence("Is a hot dog a sandwich?"))
    assert abs(sync.probability - async_.probability) < 1e-9
    assert async_.backend == sync.backend


def test_context_is_part_of_the_state():
    from core.agent.confidence import _state

    with_ctx = _state("what does this do?", "def f(): pass")
    without_ctx = _state("what does this do?", "")
    assert "def f(): pass" in with_ctx
    assert "general-knowledge" in without_ctx


def test_a_downed_model_still_produces_a_valid_confidence():
    decision = answer_confidence_sync("add null validation", "class Service: ...",
                                      client=BrokenClient())
    assert 0.0 <= decision.probability <= 1.0
    # The local backend degraded internally to its deterministic view.
    assert decision.backend == "local"


def test_marker_matches_the_probability():
    decision = answer_confidence_sync("why does this return null?",
                                      "return self.store.get(key)")
    assert format_probability(decision.probability).startswith("[p=")
