"""git diff (read-only)."""
from __future__ import annotations

from pathlib import Path

from core.git.util import run_git


def git_diff(cwd: str | Path, *, staged: bool = False, paths: list[str] | None = None) -> str:
    """Return the unified diff for the workspace (or select paths)."""
    args = ["diff", "--no-color"]
    if staged:
        args.append("--cached")
    if paths:
        args.extend(["--"] + paths)
    out = run_git(args, cwd)
    return out or ""


def recent_file_diff(cwd: str | Path, path: str, *, context: int = 20) -> str:
    out = run_git(["diff", "-U", str(context), "--no-color", "--", path], cwd)
    return out or ""
