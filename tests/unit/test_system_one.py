"""Tests for the System One decision layer (Jev-shaped, free by default)."""
from __future__ import annotations

from enum import Enum
from typing import Literal

import pytest
from pydantic import Field, ValidationError

from core.system_one import (
    ChoiceAnswer,
    ChoiceQuestion,
    Decider,
    DeterministicBackend,
    NoulAnswer,
    NoulQuestion,
    ScoreAnswer,
    ScoreQuestion,
    answer_value,
    backend_catalog,
    combine,
    gate,
    resolve_backend,
    top_choice,
)
from core.system_one.backends import BackendUnavailable, JevApiBackend, LocalModelBackend
from core.system_one.lexical import normalize, similarity_weights, softmax, tokenize
from core.system_one.primitives import (
    MAX_CHOICE_OPTIONS,
    DecisionRequest,
    answers_payload,
    state_text,
)
from core.system_one.rerank import rank_chunks_weighted, relevance_questions

# A state whose distinguishing vocabulary is shared with exactly one option.
AREA_QUESTION = ChoiceQuestion(
    instructions="Which area does this request touch?",
    criteria={
        "retrieval": "search index ranking of chunks",
        "inference": "llama cpp model generation",
        "patching": "diff apply file edits",
    },
)
RISK_QUESTION = ScoreQuestion(
    instructions="How risky is this change?",
    criteria=["trivial docs tweak", "small refactor", "touches shared code", "rewrites the retrieval index"],
)
REWRITE_STATE = "This rewrites the retrieval index and the chunk ranking."


def _candidate(path: str, content: str, score: int = 0) -> dict:
    return {
        "path": path,
        "content": content,
        "language": "python",
        "modified": 0.0,
        "start_line": 1,
        "end_line": 5,
        "rank": 0.0,
        "_score": score,
    }


# ---------------------------------------------------------------- determinism

def test_same_state_and_questions_produce_identical_answers():
    questions = {"area": AREA_QUESTION, "risk": RISK_QUESTION}
    state = "The ranking of retrieved chunks is wrong; rewrite the search index."
    assert Decider().decide(state, questions).model_dump() == Decider().decide(state, questions).model_dump()


def test_default_backend_is_free_and_offline():
    decision = Decider().decide("fix the chunk ranking", {"area": AREA_QUESTION})
    assert decision.backend == "deterministic"
    assert decision.free is True
    assert decision.usage.cost_usd == 0.0
    assert decision.usage.credits_remaining_usd is None


def test_default_backend_never_calls_the_paid_api(monkeypatch):
    # Even with a key present, the default is free: the paid path is opt-in.
    monkeypatch.setenv("JEV_API_KEY", "jv_live_test")
    assert Decider().decide("fix it", {"area": AREA_QUESTION}).backend == "deterministic"
    with pytest.raises(BackendUnavailable):
        resolve_backend("jev", allow_paid=False)


def test_jev_backend_needs_both_opt_in_and_key(monkeypatch):
    monkeypatch.delenv("JEV_API_KEY", raising=False)
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    with pytest.raises(BackendUnavailable):
        resolve_backend("jev", allow_paid=True)
    monkeypatch.setenv("JEV_API_KEY", "jv_live_test")
    backend = resolve_backend("jev", allow_paid=True)
    assert isinstance(backend, JevApiBackend)
    assert backend.free is False


def test_unknown_backend_is_rejected():
    with pytest.raises(BackendUnavailable):
        resolve_backend("gpt")


def test_backend_catalog_marks_the_paid_backend_as_not_free():
    catalog = {entry["name"]: entry for entry in backend_catalog()}
    assert catalog["deterministic"]["free"] is True
    assert catalog["local"]["free"] is True
    assert catalog["jev"]["free"] is False


# ------------------------------------------------------------------- answers

def test_choice_picks_the_option_sharing_the_state_vocabulary():
    answer = Decider().decide("Fix the chunk ranking in the search index.", {"area": AREA_QUESTION}).answers["area"]
    assert isinstance(answer, ChoiceAnswer)
    assert answer.choice == "retrieval"
    assert top_choice(answer)[1] == answer.probabilities["retrieval"]


def test_choice_probabilities_are_a_distribution():
    answer = Decider().decide("rewrite the search index", {"area": AREA_QUESTION}).answers["area"]
    assert set(answer.probabilities) == set(AREA_QUESTION.criteria)
    assert abs(sum(answer.probabilities.values()) - 1.0) < 1e-6
    assert answer.confidence == max(answer.probabilities.values())


def test_score_returns_expected_level_within_the_rubric():
    answer = Decider().decide(REWRITE_STATE, {"risk": RISK_QUESTION}).answers["risk"]
    assert isinstance(answer, ScoreAnswer)
    assert 0.0 <= answer.score <= len(RISK_QUESTION.criteria) - 1
    assert answer.score > 1.5
    assert answer.confidence == max(answer.probabilities.values())


def test_noul_separates_matching_from_unrelated_states():
    question = {"q": NoulQuestion(instructions="Did the build fail?")}
    failed = Decider().decide("the build failed with a null pointer", question).answers["q"]
    passed = Decider().decide("all checks pass and the release is signed", question).answers["q"]
    assert isinstance(failed, NoulAnswer)
    assert failed.noul > 0.7
    assert passed.noul < 0.4
    assert gate(failed, 0.7) is True
    assert gate(passed, 0.7) is False


def test_noul_criteria_spell_out_yes_and_no():
    question = {
        "gate": NoulQuestion(
            instructions="Does this request ask for a code change?",
            criteria={"true": "add fix change or remove code", "false": "only explain or ask a question"},
        )
    }
    change = Decider().decide("Add a null check before calling get_user", question).answers["gate"]
    explain = Decider().decide("please explain what this function does", question).answers["gate"]
    assert change.noul > explain.noul


def test_answer_value_normalises_every_answer_type():
    answers = Decider().decide(
        "rewrite the retrieval index ranking",
        {"area": AREA_QUESTION, "risk": RISK_QUESTION, "gate": NoulQuestion(instructions="Is this a code change?")},
    ).answers
    for name, answer in answers.items():
        assert 0.0 <= answer_value(answer) <= 1.0, name


def test_combine_uses_the_callers_coefficients():
    answers = Decider().decide(REWRITE_STATE, {"risk": RISK_QUESTION}).answers
    assert combine(answers, {"risk": 5.0}).total == combine(answers, {"risk": 1.0}).total
    both = {"risk": answers["risk"], "gate": NoulAnswer(noul=0.0)}
    weighted = combine(both, {"risk": 3.0, "gate": 1.0})
    assert weighted.total == pytest.approx(weighted.parts["risk"] * 0.75, abs=1e-6)
    assert combine(both, {"risk": 0.0, "gate": 0.0}).total == 0.0


def test_answer_value_and_gate_reject_the_wrong_answer_type():
    answer = ChoiceAnswer(choice="a", confidence=0.5, probabilities={"a": 0.5, "b": 0.5})
    with pytest.raises(TypeError):
        gate(answer)
    with pytest.raises(TypeError):
        top_choice(NoulAnswer(noul=0.5))


# ------------------------------------------------------------------ contract

def test_request_accepts_jev_shaped_json():
    request = DecisionRequest.model_validate({
        "state": "Customer: I was charged twice and nobody has replied for 3 days.",
        "questions": {
            "route": {"type": "choice", "instructions": "Where should this go?",
                      "criteria": {"billing": "money", "bug": "broken", "account": "login"}},
            "urgency": {"type": "score", "instructions": "How urgent?",
                        "criteria": ["routine", "today", "urgent", "critical"]},
            "escalate": {"type": "noul", "instructions": "Escalate to a human now?"},
        },
    })
    assert set(request.questions) == {"route", "urgency", "escalate"}
    payload = answers_payload(Decider().decide(request.state, request.questions).answers)
    assert payload["route"]["type"] == "choice"
    assert payload["escalate"]["type"] == "noul"
    assert 0.0 <= payload["escalate"]["noul"] <= 1.0


def test_choice_option_limit_is_enforced():
    with pytest.raises(ValidationError):
        ChoiceQuestion(instructions="pick", criteria={str(i): None for i in range(MAX_CHOICE_OPTIONS + 1)})


def test_score_needs_between_two_and_ten_levels():
    with pytest.raises(ValidationError):
        ScoreQuestion(instructions="rate", criteria=["only one level"])


def test_noul_criteria_cannot_invent_keys():
    with pytest.raises(ValidationError):
        NoulQuestion(instructions="gate", criteria={"maybe": "unsure"})


def test_decision_request_requires_at_least_one_question():
    with pytest.raises(ValidationError):
        DecisionRequest.model_validate({"state": "x", "questions": {}})


def test_state_text_renders_objects_deterministically():
    assert state_text("plain") == "plain"
    assert state_text({"b": 1, "a": 2}) == '{"a": 2, "b": 1}'
    assert state_text({"path": "x"}).startswith("{")


# ------------------------------------------------------------------- lexical

def test_tokenize_splits_identifiers_and_stems_them():
    tokens = tokenize("getUserById failed for policies")
    assert "get" in tokens and "user" in tokens and "id" in tokens
    assert "fail" in tokens  # failed -> fail
    assert "policy" in tokens  # policies -> policy


def test_similarity_weights_are_bounded_and_ordered():
    state = tokenize("search index ranking of chunks")
    weights = similarity_weights(state, ["search index ranking of chunks", "llama model loader"])
    assert weights[0] >= weights[1]
    assert all(0.0 <= value <= 1.0 for value in weights)


def test_softmax_and_normalize_are_stable():
    assert softmax([]) == []
    assert sum(softmax([1.0, 2.0, 3.0])) == pytest.approx(1.0)
    assert normalize([5.0, 5.0]) == [0.0, 0.0]
    assert normalize([0.0, 10.0]) == [0.0, 1.0]


def test_deterministic_backend_rejects_a_degenerate_prior():
    with pytest.raises(ValueError):
        DeterministicBackend(prior=0.0)


# ------------------------------------------------------------ local backend

class FakeClient:
    """Stands in for llama.cpp; returns one canned verdict for every call."""

    def __init__(self, payload: str):
        self.payload = payload
        self.calls = 0

    async def chat(self, messages, **kwargs) -> str:
        self.calls += 1
        return self.payload


@pytest.mark.asyncio
async def test_local_backend_blends_the_model_verdict_into_the_distribution():
    state = "apply the diff to the file"
    blend = 0.6
    client = FakeClient('{"choice": "patching", "confidence": 1.0}')
    backend = LocalModelBackend(client, blend=blend)
    decision = await Decider(backend=backend).adecide(state, {"area": AREA_QUESTION})
    answer = decision.answers["area"]
    deterministic = DeterministicBackend().decide(state, {"area": AREA_QUESTION})["area"]
    expected = blend * 1.0 + (1.0 - blend) * deterministic.probabilities["patching"]
    assert answer.probabilities["patching"] == pytest.approx(expected, abs=1e-6)
    assert decision.backend == "local"
    assert decision.free is True
    assert backend.degraded == set()


@pytest.mark.asyncio
async def test_local_backend_degrades_to_deterministic_on_bad_output():
    backend = LocalModelBackend(FakeClient("not json at all"), blend=0.6)
    decision = await Decider(backend=backend).adecide("rewrite the search index ranking",
                                                      {"area": AREA_QUESTION})
    assert backend.degraded == {"area"}
    assert isinstance(decision.answers["area"], ChoiceAnswer)
    assert abs(sum(decision.answers["area"].probabilities.values()) - 1.0) < 1e-6


@pytest.mark.asyncio
async def test_local_backend_rejects_a_choice_outside_the_criteria():
    backend = LocalModelBackend(FakeClient('{"choice": "nonsense", "confidence": 0.9}'))
    decision = await Decider(backend=backend).adecide("rewrite the search index", {"area": AREA_QUESTION})
    assert backend.degraded == {"area"}
    assert decision.answers["area"].choice in AREA_QUESTION.criteria


@pytest.mark.asyncio
async def test_local_backend_answers_score_questions():
    backend = LocalModelBackend(FakeClient('{"level": 3, "confidence": 0.8}'), blend=0.5)
    answer = (await Decider(backend=backend).adecide(REWRITE_STATE, {"risk": RISK_QUESTION})).answers["risk"]
    assert isinstance(answer, ScoreAnswer)
    assert answer.score > 1.5
    assert abs(sum(answer.probabilities.values()) - 1.0) < 1e-6


@pytest.mark.asyncio
async def test_local_backend_answers_noul_questions():
    backend = LocalModelBackend(FakeClient('{"yes": true, "confidence": 1.0}'), blend=0.5)
    answer = (await Decider(backend=backend).adecide(
        "the build failed", {"q": NoulQuestion(instructions="Did the build fail?")})).answers["q"]
    assert isinstance(answer, NoulAnswer)
    assert answer.noul > 0.5


# ------------------------------------------------------------------ envelope

def test_envelope_reports_the_model_and_backend():
    decision = Decider().decide(REWRITE_STATE, {"risk": RISK_QUESTION})
    assert decision.model == "forge-system-one-deterministic"
    assert decision.usage.input_tokens > 0
    assert decision.usage.output_tokens == 0


# -------------------------------------------------------------------- rerank

def test_rerank_keeps_the_best_candidate_and_adds_weights():
    candidates = [
        _candidate("src/model.py", "def load_model(path):\n    return llama(path)"),
        _candidate("src/ranking.py", "def rank_chunks(candidates, query):\n    return sorted(candidates)"),
        _candidate("docs/readme.md", "installation instructions"),
    ]
    ranked = rank_chunks_weighted(candidates, "fix the chunk ranking of retrieved candidates", limit=3)
    assert ranked[0]["path"] == "src/ranking.py"
    for row in ranked:
        assert 0.0 <= row["_weighted"] <= 1.0
        assert 0.0 <= row["_system_one"] <= 1.0
        assert 0.0 <= row["_system_one_confidence"] <= 1.0
        assert row["_system_one_backend"] == "deterministic"


def test_rerank_is_deterministic_and_respects_the_limit():
    candidates = [_candidate(f"src/f{index}.py", f"def handler_{index}():\n    return {index}") for index in range(6)]
    first = rank_chunks_weighted(candidates, "handler_3", limit=3)
    second = rank_chunks_weighted(candidates, "handler_3", limit=3)
    assert [row["path"] for row in first] == [row["path"] for row in second]
    assert len(first) == 3


def test_rerank_handles_no_candidates():
    assert rank_chunks_weighted([], "anything") == []


def test_weight_zero_defers_to_the_heuristic_order():
    candidates = [
        _candidate("src/ranking.py", "def rank_chunks(candidates, query): ...", score=90),
        _candidate("src/other.py", "def other(): ...", score=10),
    ]
    ranked = rank_chunks_weighted(candidates, "rank_chunks", limit=2, system_one_weight=0.0)
    assert [row["path"] for row in ranked] == ["src/ranking.py", "src/other.py"]


def test_relevance_questions_match_the_candidate_count():
    questions = relevance_questions(4)
    assert list(questions) == ["chunk_0", "chunk_1", "chunk_2", "chunk_3"]
    assert all(question.type == "score" for question in questions.values())


# ---------------------------------------------------------------- jev_compat

class Area(Enum):
    RETRIEVAL = "retrieval"
    PATCHING = "patching"


def _jev():
    import core.system_one.jev_compat as jev
    return jev


def test_jev_compat_compiles_the_upstream_field_mapping():
    jev = _jev()

    class Review(jev.BaseModel):
        area: Literal["retrieval", "inference", "patching"] = Field(description="Which area?")
        mode: Area = Area.RETRIEVAL
        risk: int = Field(ge=0, le=3, description="How risky is this change?")
        score: float = Field(ge=0.0, le=1.0, json_schema_extra={"levels": ["none", "low", "high", "certain"]})
        urgent: bool = Field(description="Is this urgent?")

    questions = Review.__jev_questions__
    assert questions["area"].type == "choice"
    assert questions["mode"].type == "choice"
    assert questions["risk"].type == "score"
    assert questions["score"].type == "score"
    assert questions["urgent"].type == "noul"
    assert questions["risk"].criteria == ["0", "1", "2", "3"]
    assert questions["score"].criteria == ["none", "low", "high", "certain"]
    assert questions["area"].instructions == "Which area?"

    decision = Review.decide("The chunk ranking is broken; fix the search index right now.")
    assert decision.area == "retrieval"
    assert 0 <= decision.risk <= 3
    assert 0.0 <= decision.score <= 1.0
    assert isinstance(decision.urgent, bool)
    assert decision.mode is Area.RETRIEVAL  # Enum member comes back, not a string


def test_jev_compat_falls_back_to_the_humanized_field_name():
    jev = _jev()

    class Gate(jev.BaseModel):
        needs_review: bool

    assert Gate.__jev_questions__["needs_review"].instructions == "needs review"


def test_jev_compat_rejects_undecidable_annotations_at_definition_time():
    jev = _jev()
    with pytest.raises(TypeError):
        class Bad(jev.BaseModel):
            name: str

    with pytest.raises(TypeError):
        @jev.fn
        def wrong(text: str) -> str:
            return text


def test_jev_compat_fn_renders_the_docstring_as_state():
    jev = _jev()

    class Answer(jev.BaseModel):
        relevant: bool = Field(description="Is this about the search index?")

    @jev.fn
    def decide_it(request: str) -> Answer:
        """A user request:

        {{ request }}
        """
        return decide_it.state()

    assert decide_it.state_payload("fix the index").endswith("fix the index")
    assert isinstance(decide_it("fix the search index ranking"), Answer)
    assert set(decide_it.questions) == {"relevant"}


def test_jev_compat_fn_short_circuits_on_a_returned_model():
    jev = _jev()

    class Answer(jev.BaseModel):
        ok: bool

    @jev.fn
    def short(text: str) -> Answer:
        return Answer(ok=True)

    assert short("anything") == Answer(ok=True)


def test_jev_compat_fn_maps_over_items():
    jev = _jev()

    class Answer(jev.BaseModel):
        risk: int = Field(ge=0, le=2, description="How risky?")

    @jev.fn
    def rate(text: str) -> Answer:
        """Rate: {{ text }}"""
        return rate.state()

    mapped = rate.map(["a trivial docs tweak", "a rewrite of the retrieval index"])
    assert len(mapped) == 2
    assert all(isinstance(item, Answer) for item in mapped)
    assert mapped[1].risk >= mapped[0].risk


def test_jev_compat_bool_threshold_is_configurable():
    jev = _jev()

    class Gate(jev.BaseModel):
        __jev_bool_threshold__ = 0.99
        yes: bool = Field(description="Did the build fail?")

    assert Gate.decide("the build failed").yes is False
    assert Gate.decide("the build failed", bool_threshold=0.1).yes is True


# ---------------------------------------------------------------------------
# .env.local secret loading (Jev API key, local only, gitignored)
# ---------------------------------------------------------------------------

def test_parse_env_file_reads_known_names_and_ignores_the_rest(tmp_path):
    from core.system_one.backends import parse_env_file

    secret = tmp_path / ".env.local"
    secret.write_text(
        "# comment\n"
        "JEV_API_KEY=apikey_test_123\n"
        'JEV_API_URL="https://example.test/api/v1/decide"\n'
        "JEV_MODEL=jev-latest\n"
        "PATH=should-not-be-touched\n"
        "\n"
        "broken line without equals\n",
        encoding="utf-8",
    )
    values = parse_env_file(secret)
    assert values == {
        "JEV_API_KEY": "apikey_test_123",
        "JEV_API_URL": "https://example.test/api/v1/decide",
        "JEV_MODEL": "jev-latest",
    }


def test_parse_env_file_missing_file_is_empty(tmp_path):
    from core.system_one.backends import parse_env_file

    assert parse_env_file(tmp_path / "absent.env") == {}


def test_load_local_env_never_overrides_real_environment(tmp_path, monkeypatch):
    from core.system_one import backends

    (tmp_path / ".env.local").write_text(
        "JEV_API_KEY=from-file\n", encoding="utf-8"
    )
    monkeypatch.setattr(
        backends, "parse_env_file", lambda path: {"JEV_API_KEY": "from-file"}
    )
    monkeypatch.setenv("JEV_API_KEY", "already-set")
    backends.load_local_env()
    assert backends.JevApiBackend.api_key_from_env() == "already-set"


def test_jev_backend_reports_available_with_key(monkeypatch):
    from core.system_one import backends

    monkeypatch.setenv("JEV_API_KEY", "apikey_test_123")
    entry = next(e for e in backends.backend_catalog() if e["name"] == "jev")
    assert entry["available"] is True


def test_score_answer_accepts_official_jev_payload_shape():
    from core.system_one.primitives import ScoreAnswer

    answer = ScoreAnswer.model_validate({
        "type": "score", "score": 1.17, "confidence": 0.57,
        "probabilities": {"0": 0.06, "1": 0.71, "2": 0.23},
        "legend": {"0": "routine", "1": "should fix today", "2": "blocking release"},
    })
    assert answer.score == 1.17
    assert answer.legend["1"] == "should fix today"


def test_score_answer_legend_stays_optional_for_free_backends():
    from core.system_one.primitives import ScoreAnswer

    answer = ScoreAnswer.model_validate(
        {"type": "score", "score": 1.0, "confidence": 0.5,
         "probabilities": {"0": 0.3, "1": 0.7}})
    assert answer.legend is None


def test_jev_backend_picks_endpoint_and_model_from_key_prefix(monkeypatch):
    from core.system_one import backends

    monkeypatch.delenv("JEV_API_URL", raising=False)
    monkeypatch.delenv("JEV_MODEL", raising=False)
    hosted = backends.JevApiBackend("jv_live_abc123")
    official = backends.JevApiBackend("apikey_abc123")
    assert hosted.base_url == backends.DEFAULT_JEV_URL
    assert official.base_url == backends.DEFAULT_TYPESAFE_URL
    # The official endpoint requires a model in the body; default is pinned.
    assert hosted.model == "jev-latest"
    assert official.model == "jev-latest"


def test_jev_backend_env_overrides_win(monkeypatch):
    from core.system_one import backends

    monkeypatch.setenv("JEV_API_URL", "https://proxy.test/api")
    monkeypatch.setenv("JEV_MODEL", "jev-1.13.0")
    backend = backends.JevApiBackend("jv_live_abc123")
    assert backend.base_url == "https://proxy.test/api"
    assert backend.model == "jev-1.13.0"

