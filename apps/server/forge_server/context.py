"""Shared request models and file-context helpers."""
from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field


class Selection(BaseModel):
    """1-based inclusive editor selection (matches VS Code coordinates)."""

    start: int = Field(ge=1)
    end: int = Field(ge=1)

    def as_tuple(self) -> tuple[int, int]:
        return (self.start, max(self.start, self.end))


class ContextRequest(BaseModel):
    message: str = Field(min_length=1)
    workspace: str | None = None
    file: str | None = None
    selection: Selection | None = None
    history: list[dict] = Field(default_factory=list)


def read_selection(workspace: str | None, file: str | None,
                   selection: Selection | None) -> tuple[str, tuple[int, int] | None]:
    """Return (snippet_text, selection_tuple) from the request context."""
    if not file or not workspace:
        return "", None
    path = Path(workspace) / file
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return "", None
    if selection:
        start = max(0, selection.start - 1)
        end = min(len(lines), selection.end)
        if start < end:
            return "\n".join(lines[start:end]), (selection.start, selection.end)
    return "", None


def file_of_request(workspace: str | None, file: str | None) -> str:
    return file or ""
