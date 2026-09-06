"""git status (read-only)."""
from __future__ import annotations

from pathlib import Path

from core.git.util import run_git


def git_status(cwd: str | Path, *, paths: list[str] | None = None) -> list[dict]:
    """Return parsed ``git status --porcelain`` entries."""
    args = ["status", "--porcelain", "-uall"]
    if paths:
        args.extend(paths)
    out = run_git(args, cwd)
    if out is None:
        return []
    entries: list[dict] = []
    for line in out.splitlines():
        if not line or len(line) < 3:
            continue
        code, _, path = line[:2], line[2], line[3:].strip()
        state = {
            "M": "modified", "A": "added", "D": "deleted", "R": "renamed",
            "C": "copied", "U": "unmerged", "?": "untracked", "!": "ignored",
        }
        entries.append({
            "path": path,
            "staged": state.get(code[0], code[0]),
            "worktree": state.get(code[1], code[1]),
            "summary": line,
        })
    return entries


def changed_file_count(cwd: str | Path) -> int:
    return len(git_status(cwd))
