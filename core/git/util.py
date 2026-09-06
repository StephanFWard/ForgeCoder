"""Shared git subprocess helper."""
from __future__ import annotations

import subprocess
from pathlib import Path

DEFAULT_TIMEOUT = 10.0


def run_git(args: list[str], cwd: str | Path, *, timeout: float = DEFAULT_TIMEOUT) -> str | None:
    """Run git read-only; returns stdout (stripped) or None on any failure."""
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
    except (OSError, subprocess.TimeoutExpired):
        return None
    if proc.returncode != 0:
        return None
    return proc.stdout


def is_git_repo(cwd: str | Path) -> bool:
    out = run_git(["rev-parse", "--is-inside-work-tree"], cwd)
    return out is not None and out.strip() == "true"
