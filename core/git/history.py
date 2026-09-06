"""git log (read-only)."""
from __future__ import annotations

from pathlib import Path

from core.git.util import run_git


def git_log(cwd: str | Path, *, n: int = 10) -> list[dict]:
    """Return the last ``n`` commits as structured records."""
    fmt = "%H%x1f%h%x1f%aI%x1f%s%x1f%an"
    out = run_git(["log", "-n", str(n), f"--pretty=format:{fmt}"], cwd)
    if out is None:
        return []
    commits: list[dict] = []
    for line in out.splitlines():
        full, short, iso, subject, author = line.split("\x1f", 4)
        commits.append({
            "hash": full,
            "short": short,
            "date": iso,
            "subject": subject,
            "author": author,
        })
    return commits


def current_branch(cwd: str | Path) -> str:
    out = run_git(["branch", "--show-current"], cwd)
    return out.strip() if out else "HEAD"
