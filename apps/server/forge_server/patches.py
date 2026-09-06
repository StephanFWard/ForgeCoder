"""Structured patch endpoints: always preview first, apply only on confirmation."""
from __future__ import annotations

from fastapi import APIRouter, Depends, Header
from pydantic import BaseModel

from core.patching.apply import PatchError, apply_patch, preview_patch
from core.patching.parser import FilePatch, PatchParseError, parse_patch
from forge_server.main import AppState, get_state
from forge_server.security import (
    PERMISSIONS_AUTO,
    ForgeSecurityError,
    check_permission,
    resolve_workspace_path,
)

router = APIRouter(prefix="/v1/patch", tags=["patch"])


class PatchPreviewRequest(BaseModel):
    workspace: str
    patch: FilePatch


class PatchApplyRequest(BaseModel):
    workspace: str
    patch: FilePatch


@router.post("/preview")
async def preview(req: PatchPreviewRequest, state: AppState = Depends(get_state)) -> dict:
    """Apply the patch in-memory and return original/proposed/diff. No writes."""
    try:
        path = resolve_workspace_path(req.workspace, req.patch.path)
    except ForgeSecurityError as exc:
        return {"ok": False, "error": str(exc)}
    try:
        result = preview_patch(path, req.patch)
    except PatchError as exc:
        return {"ok": False, "error": str(exc)}
    return {
        "ok": True,
        "path": req.patch.path,
        "diff": result.diff,
        "original": result.original,
        "proposed": result.proposed,
        "changed": result.changed,
        "operations": [o.to_dict() if hasattr(o, "to_dict") else o for o in req.patch.operations],
    }


@router.post("/apply")
async def apply(req: PatchApplyRequest, confirm: str | None = Header(default=None, alias="X-Forge-Confirm"),
                state: AppState = Depends(get_state)) -> dict:
    """Apply a patch to disk. Requires explicit confirmation (v0.1 security)."""
    if confirm != "true":
        return {"ok": False, "error": "WRITE_FILE requires confirmation (X-Forge-Confirm: true)"}
    try:
        # The confirmation header IS the user's grant for this write.
        check_permission("WRITE_FILE", granted=PERMISSIONS_AUTO | {"WRITE_FILE"})
        path = resolve_workspace_path(req.workspace, req.patch.path)
        result = apply_patch(path, req.patch)
    except (ForgeSecurityError, PatchError) as exc:
        return {"ok": False, "error": str(exc)}
    return {"ok": True, "applied": result.applied, "path": req.patch.path}


@router.post("/from-model")
async def from_model(body: dict, state: AppState = Depends(get_state)) -> dict:
    """Parse raw model output into a validated patch (used by /v1/edit)."""
    try:
        patches = parse_patch(body.get("payload", body))
    except PatchParseError as exc:
        return {"ok": False, "error": str(exc)}
    return {
        "ok": True,
        "patches": [{"path": p.path, "operations": [o.__dict__ for o in p.operations]} for p in patches],
    }
