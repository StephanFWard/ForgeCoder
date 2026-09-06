"""Safe application of line-numbered operations to a file.

Line numbers are 1-based inclusive, matching VS Code's editor coordinates
(which is what the model sees in context). The caller is always responsible
for showing a preview first; this module never writes anything by itself
except through ``apply_patch`` which takes an explicit path + confirmation.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from core.patching.diff import make_diff
from core.patching.parser import FilePatch, Operation


class PatchError(ValueError):
    pass


@dataclass
class PatchResult:
    path: str
    original: str
    proposed: str
    diff: str
    applied: bool = False

    @property
    def changed(self) -> bool:
        return self.original != self.proposed


def apply_operations(content: str, operations: list[Operation]) -> str:
    """Apply operations to a file's text. 1-based inclusive line numbers."""
    original_lines = content.splitlines(keepends=True)
    line_count = len(original_lines)

    # Sort from the bottom of the file so earlier line numbers stay valid.
    ops = sorted(operations, key=lambda o: (o.start_line, o.end_line or o.start_line), reverse=True)

    for op in ops:
        start = op.start_line - 1
        end = (op.end_line or op.start_line) - 1
        if start < 0 or start >= line_count:
            raise PatchError(f"start_line {op.start_line} out of range (file has {line_count} lines)")
        if op.type == "replace":
            if end < start or end >= line_count:
                raise PatchError(f"replace range {op.start_line}-{op.end_line} out of range")
            original_lines[start : end + 1] = [op.content.rstrip("\n ") + "\n"]
        elif op.type == "delete":
            if end < start or end >= line_count:
                raise PatchError(f"delete range {op.start_line}-{op.end_line} out of range")
            del original_lines[start : end + 1]
        elif op.type == "insert":
            original_lines.insert(start, op.content.rstrip("\n ") + "\n")
        else:  # pragma: no cover - validated by parser
            raise PatchError(f"unknown operation type {op.type!r}")

    return "".join(original_lines)


def preview_patch(path: str | Path, patch: FilePatch) -> PatchResult:
    """Apply ``patch`` to the file at ``path`` in memory only."""
    path = Path(path)
    try:
        original = path.read_text(encoding="utf-8")
    except OSError:
        original = ""
    proposed = apply_operations(original, patch.operations)
    return PatchResult(
        path=patch.path,
        original=original,
        proposed=proposed,
        diff=make_diff(original, proposed, patch.path),
        applied=False,
    )


def apply_patch(path: str | Path, patch: FilePatch) -> PatchResult:
    """Apply ``patch`` to disk. The caller must have confirmed the preview."""
    result = preview_patch(path, patch)
    if result.changed:
        Path(path).write_text(result.proposed, encoding="utf-8")
        result.applied = True
    return result
