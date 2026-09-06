"""Structured patching: diffs, JSON patch parsing, and safe application."""

from core.patching.apply import (
    PatchError,
    PatchResult,
    apply_operations,
    apply_patch,
    preview_patch,
)
from core.patching.diff import make_diff
from core.patching.parser import FilePatch, Operation, PatchParseError, parse_patch

__all__ = [
    "FilePatch",
    "Operation",
    "PatchError",
    "PatchParseError",
    "PatchResult",
    "apply_operations",
    "apply_patch",
    "make_diff",
    "parse_patch",
    "preview_patch",
]
