"""Git write operations: stage, commit, push, pull.

Every function here mutates repository state, so callers must gate behind the
GIT_WRITE permission and an explicit user confirmation (v0.1 security model).
"""
from __future__ import annotations

import subprocess
from pathlib import Path

from core.git.util import DEFAULT_TIMEOUT, is_git_repo, run_git  # noqa: F401 (re-exported)


class GitOperationError(RuntimeError):
    """Raised when a git write operation fails."""


def _run(args: list[str], cwd: str | Path, *, timeout: float = DEFAULT_TIMEOUT) -> str:
    """Run a git command and raise GitOperationError on failure."""
    try:
        proc = subprocess.run(
            ["git", *args],
            cwd=str(Path(cwd)),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise GitOperationError(f"git {' '.join(args)} failed: {exc}") from exc
    if proc.returncode != 0:
        detail = (proc.stderr or proc.stdout or "").strip()
        raise GitOperationError(f"git {' '.join(args)} failed: {detail[:500]}")
    return (proc.stdout or "").strip()


def current_branch(cwd: str | Path) -> str | None:
    out = run_git(["rev-parse", "--abbrev-ref", "HEAD"], cwd)
    return out.strip() if out else None


def remote_url(cwd: str | Path, remote: str = "origin") -> str | None:
    out = run_git(["remote", "get-url", remote], cwd)
    return out.strip() if out else None


def commit_log(cwd: str | Path, limit: int = 10) -> list[dict]:
    """Recent commits: [{hash, author, subject}]."""
    out = run_git([
        "log", f"-{max(1, limit)}",
        "--pretty=format:%h%x1f%an%x1f%s",
    ], cwd)
    if not out:
        return []
    entries: list[dict] = []
    for line in out.splitlines():
        parts = line.split("\x1f")
        if len(parts) == 3:
            entries.append({"hash": parts[0], "author": parts[1], "subject": parts[2]})
    return entries


def stage(cwd: str | Path, paths: list[str] | None = None) -> int:
    """Stage explicit paths, or all changes (incl. untracked) when None."""
    if paths:
        _run(["add", "--", *paths], cwd)
    else:
        _run(["add", "-A"], cwd)
    out = run_git(["diff", "--cached", "--name-only"], cwd)
    return len(out.splitlines()) if out else 0


def commit(cwd: str | Path, message: str) -> dict:
    """Commit the staged index. Returns {hash, subject}."""
    if not message.strip():
        raise GitOperationError("Commit message must not be empty")
    _run(["commit", "-m", message.strip()], cwd)
    out = _run(["log", "-1", "--pretty=format:%h%x1f%s"], cwd)
    parts = out.split("\x1f")
    return {"hash": parts[0] if parts else "", "subject": parts[1] if len(parts) > 1 else message}


def push(cwd: str | Path, remote: str = "origin", branch: str | None = None,
         *, timeout: float = 120.0) -> dict:
    """Push the current (or given) branch to ``remote``."""
    branch = branch or current_branch(cwd)
    if not branch:
        raise GitOperationError("Cannot determine the current branch")
    _run(["push", remote, branch], cwd, timeout=timeout)
    return {"remote": remote, "branch": branch}


def pull(cwd: str | Path, remote: str = "origin", branch: str | None = None,
         *, timeout: float = 120.0) -> dict:
    """Pull from ``remote`` into the current branch (no rebase in v0.1)."""
    branch = branch or current_branch(cwd)
    if not branch:
        raise GitOperationError("Cannot determine the current branch")
    out = _run(["pull", remote, branch, "--no-edit"], cwd, timeout=timeout)
    return {"remote": remote, "branch": branch, "output": out[-2000:]}
