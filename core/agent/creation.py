"""Creation-task detection: which requests want NEW files, and which files?

Deterministic, no model call. A request like "make the game snake in html"
must not be treated as an edit of whatever file retrieval happened to surface,
nor interrogated with "which file should the change land in?" — the target is
derivable. The task frame (``core.agent.frame``) and the scope contract
(``core.agent.scope``) use these helpers to bound a creation request to the
files it will create and to give it a real acceptance check.
"""
from __future__ import annotations

import re

_CREATION_RE = re.compile(
    r"\b(create|write|generate|build|make|scaffold|bootstrap)\b"
    r"[^.\n]{0,60}\b(file|page|html|script|module|component|class|"
    r"app|game|repository|repo|project|site|api|endpoint|test)\b"
    r"|\bgame\b|\bapp\b|\bscript\b|\bprogram\b|\btic\b|\bfps\b"
    r"|\bshooter\b|\bfirst\s+person\b|\brepository\b|\bproject\b"
    r"|\bfrom\s+scratch\b|\bnew\s+(file|app|game|script|program|project|repo)\b",
    re.IGNORECASE,
)

# The per-file plan step format: "Create file: index.html -- complete ...".
_SINGLE_FILE_RE = re.compile(
    r"Create file:\s*([\w][\w.\-/]*[A-Za-z0-9])", re.IGNORECASE
)


def is_creation_request(instruction: str) -> bool:
    """Heuristic: does this instruction ask to *create a new file*?

    Mirrors the fallback when the workspace has no indexed context — a 1.5B
    model cannot "edit" a file it was never shown, so generation (creates)
    is the only applicable behavior.
    """
    return bool(_CREATION_RE.search(instruction or ""))


def creation_targets(message: str) -> list[str]:
    """Filenames a creation request most likely needs, cheapest heuristic first."""
    text = (message or "").lower()
    m = re.search(r"([\w.-]+\.(py|html|js|ts|java|go|rs|cs|cpp|sql))", text)
    if m:
        named = m.group(1)
        return [named, "README.md"] if named != "README.md" else [named]
    if "fps" in text or "first person" in text or "shooter" in text:
        return ["game.py", "requirements.txt", "README.md"]
    if "api" in text or "endpoint" in text or "server" in text:
        return ["app.py", "requirements.txt", "README.md"]
    if "html" in text or "web" in text or "tic" in text or "browser" in text:
        return ["index.html", "README.md"]
    return ["main.py", "README.md"]


def single_creation_target(instruction: str, plan_request: str = "") -> str | None:
    """The one file an edit step must create ("Create file: index.html -- ...")."""
    for text in (instruction, plan_request):
        m = _SINGLE_FILE_RE.search(text or "")
        if m:
            return m.group(1).strip().strip("'\"").lstrip("./")
    return None


__all__ = [
    "creation_targets",
    "is_creation_request",
    "single_creation_target",
]
