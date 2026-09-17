"""Creation-aware task frames and contracts.

"Make the game snake in html" once came back as
``[warn] ask-dont-guess: The specific file where the change should land is not
provided. Please name the command that proves this change.`` — the model
paraphrasing the two edit-oriented unknowns the frame manufactured for a
creation request. For creation tasks the frame must answer those questions
itself: the files to create are derived, and acceptance is the smoke check.
"""
from __future__ import annotations

from core.agent.scope import NO_ACCEPTANCE, creation_acceptance
from core.retrieval.context import ContextBuilder


def _built(message: str):
    """A frame for ``message`` with no editor file and no indexed evidence."""
    return ContextBuilder(index=None).build(message)


def test_snake_request_frame_names_the_files_to_create():
    built = _built("Make the game snake in html")
    frame = built.frame
    assert frame.allowed == ["index.html", "README.md"]
    assert frame.ready is True
    # The two unknowns that caused the ask-dont-guess detour must be gone.
    assert "which file should the change land in?" not in frame.unknowns
    assert not any("acceptance command" in u for u in frame.unknowns)
    # ...and the rendered frame must not carry them into the user turn either.
    rendered = built.request
    assert "which file should the change land in?" not in rendered
    assert "name the command that proves" not in rendered


def test_snake_request_acceptance_is_the_smoke_check():
    built = _built("Make the game snake in html")
    acceptance = built.contract.acceptance_criteria
    assert acceptance and acceptance[0] != NO_ACCEPTANCE
    assert "smoke check" in acceptance[0]
    assert "index.html" in acceptance[0]


def test_minesweeper_request_frame_is_creation_ready():
    built = _built("Make a minesweeper webpage game")
    assert built.frame.allowed == ["index.html", "README.md"]
    assert built.frame.ready is True
    assert "which file should the change land in?" not in built.frame.unknowns


def test_edit_request_still_asks_which_file():
    """The creation shortcut must not mute the unknowns that edits really have."""
    built = _built("find the auth bug in login.py")
    assert "which file should the change land in?" in built.frame.unknowns


def test_creation_acceptance_lists_targets():
    text = creation_acceptance(["snake.html", "README.md"])
    assert "snake.html, README.md" in text
    assert "smoke check" in text


def test_plans_wrappers_delegate_to_core():
    """The plans.py heuristics and the frame heuristics must never diverge."""
    from apps.server.forge_server.plans import _creation_files, _is_creation_request

    request = "Make the game snake in html"
    assert _is_creation_request(request) is True
    assert _creation_files(request) == ["index.html", "README.md"]
    assert _is_creation_request("find the auth bug in login.py") is False


def test_creation_frame_note_rides_in_the_user_turn():
    """The frame states the task type explicitly, so the model cannot treat a
    creation request as an edit of an unknown file."""
    built = _built("Make the game snake in html")
    assert "Task note: Creation task" in built.request
    assert "index.html" in built.request
    assert "Do not modify existing files" in built.request


def test_unknowns_rendered_as_imperatives_not_questions():
    """Unknowns used to be rendered as literal questions, which the model
    parroted back as fabricated "[warn] ask-dont-guess" findings. The render
    must say who resolves them and forbid echoing them."""
    built = _built("find the auth bug in login.py")
    assert "Unknowns to resolve yourself" in built.request
    assert "never repeat them, rule names, or severity markers" in built.request


def test_system_prompt_forbids_printing_rule_markers():
    """The rules header told the model "warn = tell the user", which produced
    fabricated "[warn] ..." lines. It must now forbid them."""
    from core.agent.prompt import RULES_HEADER, system_prompt

    assert "Never print rule slugs" in RULES_HEADER
    prompt = system_prompt("chat")
    assert "Never print rule slugs" in prompt
    assert "tell the user" not in prompt
