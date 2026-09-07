"""Tests for git write operations and the plan/act engine."""
from __future__ import annotations

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


def test_plan_store_roundtrip():
    from apps.server.forge_server.plans import PlanStore

    store = PlanStore()
    pid = store.save({"summary": "s", "steps": [{"title": "t", "action": "search", "detail": "d"}],
                      "results": {}})
    assert store.get(pid) is not None
    store.record(pid, 0, {"ok": True, "output": "done"})
    assert store.get(pid)["results"]["0"]["output"] == "done"
    assert store.get("nope") is None
