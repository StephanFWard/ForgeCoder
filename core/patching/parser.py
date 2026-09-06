"""Parse the model's structured patch output.

Expected format (grammar-constrained by runtime/grammars/patch.gbnf):

    {
      "summary": "Added null validation.",
      "files": [{
        "path": "src/UserService.java",
        "operations": [{
          "type": "replace",               // insert | replace | delete
          "start_line": 42,
          "end_line": 47,
          "content": "..."
        }]
      }]
    }
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass


class PatchParseError(ValueError):
    pass


@dataclass(frozen=True)
class Operation:
    type: str  # "insert" | "replace" | "delete"
    start_line: int
    end_line: int | None = None
    content: str = ""


@dataclass(frozen=True)
class FilePatch:
    path: str
    operations: list[Operation]

    def to_dict(self) -> dict:
        return {
            "path": self.path,
            "operations": [
                {"type": o.type, "start_line": o.start_line,
                 "end_line": o.end_line, "content": o.content}
                for o in self.operations
            ],
        }


_VALID_TYPES = {"insert", "replace", "delete"}


def extract_json(text: str) -> dict:
    """Best-effort extraction of a JSON object from model output."""
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # Strip markdown fences
    fenced = re.sub(r"^```(?:json)?\s*", "", text.strip())
    fenced = re.sub(r"\s*```$", "", fenced)
    try:
        return json.loads(fenced)
    except json.JSONDecodeError:
        pass
    # Find the outermost {...}
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end > start:
        try:
            return json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            pass
    raise PatchParseError("No JSON object found in model output")


def parse_patch(payload: dict | str) -> list[FilePatch]:
    """Validate a patch payload dict (or JSON text) into FilePatch list."""
    if isinstance(payload, str):
        payload = extract_json(payload)

    if not isinstance(payload, dict):
        raise PatchParseError("Patch payload must be an object")

    raw_files = payload.get("files")
    if isinstance(raw_files, dict):  # tolerate single file at top level
        raw_files = [raw_files]
    if not isinstance(raw_files, list):
        raise PatchParseError("Patch payload must contain a 'files' list")

    patches: list[FilePatch] = []
    for raw in raw_files:
        if not isinstance(raw, dict) or not raw.get("path"):
            raise PatchParseError("Each file patch needs a 'path'")
        ops: list[Operation] = []
        raw_ops = raw.get("operations") or []
        if not isinstance(raw_ops, list):
            raise PatchParseError(f"operations must be a list for {raw.get('path')}")
        for op in raw_ops:
            optype = op.get("type")
            if optype not in _VALID_TYPES:
                raise PatchParseError(f"Unknown op type {optype!r}")
            start = op.get("start_line")
            if not isinstance(start, int) or start < 1:
                raise PatchParseError("start_line must be a positive integer")
            end = op.get("end_line")
            if end is not None and (not isinstance(end, int) or end < start):
                raise PatchParseError("end_line must be an int >= start_line")
            if optype != "delete" and not isinstance(op.get("content"), str):
                raise PatchParseError("content must be a string")
            ops.append(Operation(type=optype, start_line=start, end_line=end,
                                 content=op.get("content", "")))
        if not ops:
            raise PatchParseError(f"No operations for {raw.get('path')!r}")
        patches.append(FilePatch(path=raw["path"], operations=ops))
    return patches
