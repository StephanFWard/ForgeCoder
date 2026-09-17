"""Tests for git write operations and the plan/act engine."""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from core.git import operations as git_ops
from core.git.util import is_git_repo


def _init_repo(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "init", "-q"], cwd=str(root), check=True)
    subprocess.run(["git", "config", "user.email", "forge@test"], cwd=str(root), check=True)
    subprocess.run(["git", "config", "user.name", "Forge"], cwd=str(root), check=True)
    return root


def _commit_all(root: Path, message: str) -> None:
    subprocess.run(["git", "add", "-A"], cwd=str(root), check=True)
    subprocess.run(["git", "commit", "-q", "-m", message], cwd=str(root), check=True)


STAGED_DIFF = (
    "diff --git a/src/a.py b/src/a.py\n"
    "index 1111111..2222222 100644\n"
    "--- a/src/a.py\n"
    "+++ b/src/a.py\n"
    "@@ -1,3 +1,4 @@\n"
    " def keep():\n"
    "+    added = True\n"
    "     return added\n"
)

# Same file, both sources: index and worktree line numbers are not equivalent.
BOTH_SOURCES = (
    "diff --git a/src/b.py b/src/b.py\n"
    "--- a/src/b.py\n"
    "+++ b/src/b.py\n"
    "@@ -10,2 +10,3 @@\n"
    " x = 1\n"
    "+y = 2\n"
)


def test_prepare_review_anchors_added_new_side_lines():
    from core.git.review import prepare_review

    context, anchors, truncated = prepare_review(STAGED_DIFF, "")
    assert ("staged", "src/a.py", 2) in anchors      # NEW-side line of the + line
    assert ("staged", "src/a.py", 3) not in anchors  # context lines are not anchors
    assert not truncated
    assert "SOURCE: staged" in context
    assert "+++ b/src/a.py" in context


def test_prepare_review_keeps_sources_distinct():
    from core.git.review import prepare_review

    _, anchors, _ = prepare_review(BOTH_SOURCES, BOTH_SOURCES)
    assert ("staged", "src/b.py", 11) in anchors
    assert ("unstaged", "src/b.py", 11) in anchors
    assert ("staged", "src/b.py", 10) not in anchors  # context line, not an addition


def test_prepare_review_budget_truncates_whole_lines():
    from core.git.review import prepare_review

    diff = "".join(
        f"diff --git a/f{i}.py b/f{i}.py\n--- a/f{i}.py\n+++ b/f{i}.py\n@@ -1 +1,2 @@\n x\n+y\n"
        for i in range(50)
    )
    context, anchors, truncated = prepare_review("", diff, max_chars=400)
    assert truncated
    header = "\nSOURCE: unstaged\n"
    assert context.startswith(header)
    body = context[len(header):]
    assert body.endswith("\n")            # whole lines only, never a partial line
    assert diff.startswith(body)          # exactly a prefix of the real diff
    assert anchors                        # supplied additions are still anchored


def test_prepare_review_skips_quoted_paths():
    from core.git.review import prepare_review

    quoted = (
        'diff --git "a/pa th.py" "b/pa th.py"\n'
        '--- "a/pa th.py"\n'
        '+++ "b/pa th.py"\n'
        "@@ -1 +1,2 @@\n"
        " x = 1\n"
        "+y = 2\n"
    )
    _, anchors, _ = prepare_review("", quoted)
    assert anchors == set()  # cannot be validated soundly, so it is not offered


def test_validate_review_requires_supplied_anchor():
    from core.git.review import validate_review

    anchors = {("staged", "src/a.py", 2)}
    good = (
        '{"summary": "One line added.", "findings": [{"source": "staged", '
        '"path": "src/a.py", "line": 2, "severity": "info", "message": "Flags added."}]}'
    )
    review = validate_review(good, anchors)
    assert review.findings[0].line == 2

    for bad in (
        good.replace('"line": 2', '"line": 99'),               # line not in diff
        good.replace('"src/a.py"', '"src/other.py"'),          # path not in diff
        good.replace('"staged"', '"unstaged"'),                # wrong source
        good.replace('"severity": "info"', '"severity": "high"'),
        good.replace('"findings": [', '"findings": [{"extra": 1}, '),
        "{not json",
    ):
        with pytest.raises(ValueError):
            validate_review(bad, anchors)


def test_render_review_is_readable_prose():
    from core.git.review import render_review, validate_review

    review = validate_review(
        '{"summary": "Guarded the lookup.", "findings": [{"source": "unstaged", '
        '"path": "src/a.py", "line": 3, "severity": "warning", '
        '"message": "Missing None default."}]}',
        {("unstaged", "src/a.py", 3)},
    )
    text = render_review(review)
    assert text.startswith("Guarded the lookup.")
    assert "- [warning] unstaged src/a.py:3: Missing None default." in text


def test_review_prompt_forbids_writes_and_unseen_context():
    from core.git.review import REVIEW_PROMPT

    assert "read-only" in REVIEW_PROMPT
    assert "Do not invent unseen context" in REVIEW_PROMPT


def test_edit_fix_schemas_match_the_parser_contract():
    """A schema-valid edit payload must be parseable by core.patching.parser."""
    from core.patching.contracts import EDIT_SCHEMA, FIX_SCHEMA
    from core.patching.parser import parse_patch

    payload = {
        "summary": "Guard the lookup.",
        "diagnosis": "The key may be missing.",
        "files": [{"path": "src/a.py", "operations": [
            {"type": "replace", "start_line": 3, "end_line": 3, "content": "    return None"},
        ]}],
    }
    patches = parse_patch(payload)
    assert patches[0].path == "src/a.py"
    assert patches[0].operations[0].end_line == 3
    for schema in (EDIT_SCHEMA, FIX_SCHEMA):
        assert schema["$defs"]["PatchOperationPayload"]["properties"]["type"]["enum"] == [
            "insert", "replace", "delete",
        ]
        assert set(schema["$defs"]["PatchOperationPayload"]["required"]) == {
            "type", "start_line", "end_line", "content",
        }
        # No nullable branches: the grammar stays flat for 1.5B-model reliability.
        assert "anyOf" not in json.dumps(schema)
        # An empty files list is the model's "not enough context" escape hatch.
        assert "minItems" not in schema["properties"]["files"]


def test_stage_commit_and_log(tmp_path):
    root = _init_repo(tmp_path / "repo")
    assert is_git_repo(root)
    (root / "hello.py").write_text("print('hi')\n", encoding="utf-8")

    staged = git_ops.stage(root)
    assert staged == 1
    info = git_ops.commit(root, "feat: first hello")
    assert info["hash"]

    log = git_ops.commit_log(root, limit=5)
    assert log and log[0]["subject"] == "feat: first hello"


def test_commit_empty_message_rejected(tmp_path):
    root = _init_repo(tmp_path / "repo")
    (root / "a.txt").write_text("x\n", encoding="utf-8")
    git_ops.stage(root)
    with pytest.raises(git_ops.GitOperationError):
        git_ops.commit(root, "   ")


def test_stage_nothing_is_zero(tmp_path):
    root = _init_repo(tmp_path / "repo")
    (root / "a.txt").write_text("x\n", encoding="utf-8")
    _commit_all(root, "init")
    assert git_ops.stage(root) == 0


def test_current_branch(tmp_path):
    root = _init_repo(tmp_path / "repo")
    (root / "a.txt").write_text("x\n", encoding="utf-8")
    _commit_all(root, "init")  # HEAD must exist for a branch name
    branch = git_ops.current_branch(root)
    assert branch in {"main", "master"}


def test_plan_fallback_shape():
    from apps.server.forge_server.plans import _fallback_plan

    plan = _fallback_plan("find the auth bug")
    assert plan["steps"][0]["action"] == "search"
    assert all(s["action"] in {"search", "explain", "edit", "test", "commit"}
               for s in plan["steps"])


def test_plan_fallback_creation_fps_game():
    """The FPS creation request must plan per-file creation, not investigation."""
    from apps.server.forge_server.plans import _fallback_plan

    plan = _fallback_plan("Please create a local first person shooting game to play locally.")
    actions = [s["action"] for s in plan["steps"]]
    assert actions[0] == "search"
    assert "edit" in actions and "test" in actions
    assert any("game.py" in s["detail"] for s in plan["steps"] if s["action"] == "edit")


def test_plan_fallback_creation_tictactoe():
    from apps.server.forge_server.plans import _fallback_plan

    plan = _fallback_plan("Create a new repository named 'tic tac toe'")
    assert any("index.html" in s["detail"] for s in plan["steps"] if s["action"] == "edit")


def test_is_creation_request_variants():
    from apps.server.forge_server.plans import _is_creation_request

    assert _is_creation_request("Please create a local first person shooting game to play locally.")
    assert _is_creation_request("Create a new repository named 'tic tac toe'")
    assert not _is_creation_request("find the auth bug in login.py")


def test_minesweeper_request_routes_to_creation():
    """Regression: "Make a minesweeper webpage game" once produced a bogus
    README.md line-patch instead of creating the game files. The request must
    route to the creation pipeline and plan per-file creation."""
    from apps.server.forge_server.plans import _creation_files, _fallback_plan, _is_creation_request

    request = "Make a minesweeper webpage game"
    assert _is_creation_request(request)
    assert _creation_files(request) == ["index.html", "README.md"]

    plan = _fallback_plan(request)
    actions = [s["action"] for s in plan["steps"]]
    assert "edit" in actions and "test" in actions
    # index.html must be CREATED. README.md may be created alongside (that is
    # the documented per-file plan) — but every edit step is a *creation*
    # step, never a modification of an existing file.
    assert "Create index.html" in [s["title"] for s in plan["steps"]]
    for s in plan["steps"]:
        if s["action"] == "edit":
            assert s["title"].startswith("Create "), s
            assert s["detail"].startswith("Create file: "), s


def test_parse_creation_rejects_existing_file_and_foreign_patches():
    """A creation step may only be satisfied by ``creates`` output: an
    existing-file patch (or the undocumented oldLine/newLine dialect from the
    bad minesweeper transcript) is not a creation answer and must return None
    so the caller retries with a corrective instruction."""
    from apps.server.forge_server.plans import _parse_creation

    bad_dialect = json.dumps({
        "files": ["README.md"],
        "patch": {"diff": [{"oldLine": "123", "newLine": "124", "content": "..."}]},
    })
    assert _parse_creation(bad_dialect) is None

    files_only = json.dumps({
        "summary": "edit",
        "files": [{"path": "README.md", "operations": [
            {"type": "replace", "start_line": 1, "end_line": 2, "content": "x"},
        ]}],
    })
    assert _parse_creation(files_only) is None

    good = json.dumps({
        "message": "Created minesweeper",
        "creates": {"index.html": "<!doctype html><html>... complete game ...</html>"},
    })
    multi = _parse_creation(good)
    assert multi is not None
    assert multi.creates == {"index.html": "<!doctype html><html>... complete game ...</html>"}


def test_smoke_check_accepts_real_html(tmp_path):
    """The plan's test step is the acceptance evidence for creation requests:
    it must actually validate an HTML page, not just Python files."""
    from apps.server.forge_server.plans import _syntax_smoke_check

    (tmp_path / "index.html").write_text(
        "<!doctype html><html><body><h1>Snake</h1>"
        "<script>const x = 1;</script></body></html>",
        encoding="utf-8",
    )
    report = _syntax_smoke_check(str(tmp_path))
    assert report is not None
    assert report.startswith("SMOKE-TEST PASSED")
    assert "HTML file(s) pass structure checks" in report


def test_smoke_check_rejects_placeholder_html(tmp_path):
    from apps.server.forge_server.plans import _syntax_smoke_check

    (tmp_path / "bad.html").write_text("TODO: complete code here", encoding="utf-8")
    report = _syntax_smoke_check(str(tmp_path))
    assert report is not None
    assert "SMOKE-TEST FAILED" in report
    assert "bad.html" in report


def test_smoke_check_rejects_html_without_structure(tmp_path):
    from apps.server.forge_server.plans import _syntax_smoke_check

    (tmp_path / "broken.html").write_text(
        "just some notes, <script>unbalanced",
        encoding="utf-8",
    )
    report = _syntax_smoke_check(str(tmp_path))
    assert "SMOKE-TEST FAILED" in report
    assert "no <html> document structure" in report


def test_smoke_check_none_without_smokable_files(tmp_path):
    from apps.server.forge_server.plans import _syntax_smoke_check

    (tmp_path / "notes.txt").write_text("nothing smokable", encoding="utf-8")
    assert _syntax_smoke_check(str(tmp_path)) is None


def test_plan_store_roundtrip():
    from apps.server.forge_server.plans import PlanStore

    store = PlanStore()
    pid = store.save({"summary": "s", "steps": [{"title": "t", "action": "search", "detail": "d"}],
                      "results": {}})
    assert store.get(pid) is not None
    store.record(pid, 0, {"ok": True, "output": "done"})
    assert store.get(pid)["results"]["0"]["output"] == "done"
    assert store.get("nope") is None
