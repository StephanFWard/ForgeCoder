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
        "proposed": result.proposed if result.proposed is not None else "",
        "changed": result.changed,
        "operations": [o.to_dict() for o in req.patch.operations],
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
        "patches": [p.to_dict() for p in patches],
    }


# ------------------------------------------------------------------ multi-file
# Multi-file operations bundle edits to existing files with new file creations
# and apply them atomically (rollback on any failure). See core.patching.apply.


class MultiPatchPreviewRequest(BaseModel):
    workspace: str
    patches: list[FilePatch] = []
    creates: dict[str, str] = {}
    message: str = ""


class MultiPatchApplyRequest(BaseModel):
    workspace: str
    patches: list[FilePatch] = []
    creates: dict[str, str] = {}
    message: str = ""


@router.post("/multi-preview")
async def multi_preview(req: MultiPatchPreviewRequest, state: AppState = Depends(get_state)) -> dict:
    """Preview a cross-file change set in memory. No writes."""
    from core.patching.apply import MultiPatch, preview_multi

    root = resolve_workspace_path(req.workspace, ".")
    try:
        multi = MultiPatch(patches=req.patches, creates=req.creates, message=req.message)
        results = preview_multi(root, multi)
    except (ForgeSecurityError, PatchError) as exc:
        return {"ok": False, "error": str(exc)}
    patch_by_path = {p.path: p for p in req.patches}
    return {
        "ok": True,
        "summary": multi.message or "",
        "files": [
            {
                "path": r.path,
                "diff": r.diff,
                "original": r.original,
                "proposed": r.proposed,
                "changed": r.changed,
                "created": bool(not r.original),
                "operations": [
                    {"type": o.type, "start_line": o.start_line, "end_line": o.end_line, "content": o.content}
                    for o in patch_by_path.get(r.path, FilePatch(path=r.path, operations=[])).operations
                ],
            }
            for r in results
        ],
    }


@router.post("/multi-apply")
async def multi_apply(req: MultiPatchApplyRequest, confirm: str | None = Header(default=None, alias="X-Forge-Confirm"),
                      state: AppState = Depends(get_state)) -> dict:
    """Apply a cross-file change set atomically with rollback. Requires confirmation."""
    from core.patching.apply import MultiPatch, apply_multi

    if confirm != "true":
        return {"ok": False, "error": "WRITE_FILE requires confirmation (X-Forge-Confirm: true)"}
    try:
        check_permission("WRITE_FILE", granted=PERMISSIONS_AUTO | {"WRITE_FILE"})
        root = resolve_workspace_path(req.workspace, ".")
        multi = MultiPatch(patches=req.patches, creates=req.creates, message=req.message)
        results = apply_multi(root, multi)
    except (ForgeSecurityError, PatchError) as exc:
        return {"ok": False, "error": str(exc)}
    return {
        "ok": True,
        "summary": multi.message or "",
        "applied": [r.path for r in results if r.applied],
        "files": [{"path": r.path, "diff": r.diff, "changed": r.changed} for r in results],
    }
