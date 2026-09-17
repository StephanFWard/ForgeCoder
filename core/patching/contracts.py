"""JSON schemas for model-produced patches.

Passed to llama.cpp as ``response_format={"type": "json_schema", ...}`` so the
edit and fix calls are grammar-constrained instead of trusting the model to
follow the prose contract in ``runtime/prompts/edit.txt`` /
``fix.txt``. The shapes mirror exactly what :mod:`core.patching.parser`
accepts, so a schema-valid object is always parseable.

Deliberate choices:

* ``end_line`` and ``content`` are required and never nullable: the grammar
  stays simple (no ``anyOf``/``null`` branches) and matches the documented
  contract, where an ``insert`` uses ``end_line == start_line`` and a
  ``delete`` passes an empty ``content``. Cross-field ordering
  (``end_line >= start_line``) is still enforced by the parser, since a JSON
  schema cannot express it.
* the top-level ``files`` list may be empty: that is the model's escape hatch
  for "the supplied context is not enough to edit safely", which
  ``parser.parse_patch`` accepts as ``[]``.
* ``extra="forbid"`` keeps hallucinated keys out of the patch payload.
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class PatchOperationPayload(BaseModel):
    """One line operation, 1-based inclusive line numbers on the original file."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["insert", "replace", "delete"]
    start_line: int = Field(ge=1)
    end_line: int = Field(ge=1)
    content: str  # required: a delete passes ""


class FilePatchPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    path: str = Field(min_length=1)
    operations: list[PatchOperationPayload] = Field(min_length=1)


class EditPayload(BaseModel):
    """Shape returned by ``POST /v1/edit``."""

    model_config = ConfigDict(extra="forbid")

    summary: str = Field(min_length=1, max_length=1000)
    files: list[FilePatchPayload] = []


class FixPayload(EditPayload):
    """Shape returned by ``POST /v1/fix``."""

    diagnosis: str = Field(default="", max_length=2000)


EDIT_SCHEMA = EditPayload.model_json_schema()
FIX_SCHEMA = FixPayload.model_json_schema()
