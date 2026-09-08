"""Safe application of line-numbered operations to a file or across files.

Multi-file support (v0.3): MultiPatch bundles edits to existing files
with brand-new file creations and applies them atomically - if any file
fails, all already-written files are rolled back.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from core.patching.diff import make_diff
from core.patching.parser import FilePatch, MultiPatch, Operation


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


# ----------------------------------------------------------------- multi-file
@dataclass
class MultiPatch:
    """A cross-file change set: edits to existing files + brand-new files.

    ``patches`` are edits to existing files (the original system). ``creates``
    are files that do not yet exist — the model supplies their full content
    keyed by repository-relative path. Both are applied together atomically.
    """
    patches: list[FilePatch] = field(default_factory=list)
    creates: dict[str, str] = field(default_factory=dict)
    message: str = ""


def preview_multi(root: str | Path, multi: MultiPatch) -> list[PatchResult]:
    """In-memory preview of a cross-file change set (no writes)."""
    root = Path(root)
    results: list[PatchResult] = []
    for patch in multi.patches:
        results.append(preview_patch(root / patch.path, patch))
    for relpath, content in multi.creates.items():
        full = root / relpath
        try:
            original = full.read_text(encoding="utf-8")
        except OSError:
            original = ""
        results.append(PatchResult(
            path=relpath, original=original, proposed=content,
            diff=make_diff(original, content, relpath), created=original == "",
        ))
    return results


def apply_multi(root: str | Path, multi: MultiPatch) -> list[PatchResult]:
    """Apply a cross-file change set atomically with rollback.

    Every file is previewed first; if any preview raises, nothing is written.
    Files are then written one at a time. If a write fails partway through,
    every file already written is restored from its captured originals
    (created files are removed). Returns the results with ``applied=True``.
    """
    root = Path(root)
    results = preview_multi(root, multi)

    backups: list[tuple[Path, str, bool]] = []  # (path, original_or_empty, existed)
    try:
        for patch in multi.patches:
            full = root / patch.path
            if not full.exists():
                raise PatchError(f"file to patch does not exist: {patch.path}")
            existed = full.exists()
            backups.append((full, full.read_text(encoding="utf-8") if existed else "", existed))
            full.write_text(preview_patch(full, patch).proposed, encoding="utf-8")
        for relpath, content in multi.creates.items():
            full = root / relpath
            full.parent.mkdir(parents=True, exist_ok=True)
            existed = full.exists()
            backups.append((full, full.read_text(encoding="utf-8") if existed else "", existed))
            full.write_text(content, encoding="utf-8")
    except Exception:
        for full, original, existed in reversed(backups):
            if not existed:
                try:
                    full.unlink()
                except OSError:
                    pass
            else:
                try:
                    full.write_text(original, encoding="utf-8")
                except OSError:
                    pass
        raise

    for r in results:
        r.applied = True
    return results


def apply_patch(path: str | Path, patch: FilePatch) -> PatchResult:
    """Apply ``patch`` to disk. The caller must have confirmed the preview."""
    result = preview_patch(path, patch)
    if result.changed:
        Path(path).write_text(result.proposed, encoding="utf-8")
        result.applied = True
    return result
