"""Git endpoints: review working changes, commit, push, pull.

Read endpoints are auto-allowed (GIT_READ); anything that mutates the
repository requires the X-Forge-Confirm header (GIT_WRITE).
"""
from __future__ import annotations

from fastapi import APIRouter, Depends, Header
from pydantic import BaseModel

from core.git import diff as git_diff_mod
from core.git import operations as git_ops
from core.git import status as git_status_mod
from core.git.review import REVIEW_PROMPT, Review, prepare_review, render_review, validate_review
from core.git.util import is_git_repo
from forge_server.main import AppState, get_state
from forge_server.security import (
    PERMISSIONS_AUTO,
    ForgeSecurityError,
    check_permission,
)

router = APIRouter(prefix="/v1/git", tags=["git"])


class WorkspaceRequest(BaseModel):
    workspace: str


class CommitRequest(WorkspaceRequest):
    message: str
    add_all: bool = True
    paths: list[str] | None = None


def _require_repo(workspace: str) -> None:
    if not is_git_repo(workspace):
        raise ForgeSecurityError(f"{workspace!r} is not a git repository")


@router.post("/status")
async def status(req: WorkspaceRequest, state: AppState = Depends(get_state)) -> dict:
    """Branch, remote, and per-file working-tree state (read-only)."""
    check_permission("GIT_READ")
    _require_repo(req.workspace)
    entries = git_status_mod.git_status(req.workspace)
    staged = sum(1 for e in entries if e["staged"] not in {" ", "untracked", "!"})
    worktree = sum(1 for e in entries if e["worktree"] not in {" ", "!"})
    untracked = sum(1 for e in entries if e["worktree"] == "untracked")
    return {
        "ok": True,
        "branch": git_ops.current_branch(req.workspace),
        "remote": git_ops.remote_url(req.workspace),
        "entries": entries,
        "counts": {"total": len(entries), "staged": staged, "worktree": worktree,
                   "untracked": untracked},
        "recent": git_ops.commit_log(req.workspace, limit=5),
    }


@router.post("/diff")
async def diff(req: WorkspaceRequest, state: AppState = Depends(get_state)) -> dict:
    """Unified diff of the working tree (staged + unstaged)."""
    check_permission("GIT_READ")
    _require_repo(req.workspace)
    return {
        "ok": True,
        "staged": git_diff_mod.git_diff(req.workspace, staged=True),
        "unstaged": git_diff_mod.git_diff(req.workspace, staged=False),
    }


@router.post("/changes")
async def changes(req: WorkspaceRequest, state: AppState = Depends(get_state)) -> dict:
    """View Changes, engineered out: diff -> model review -> next steps."""
    check_permission("GIT_READ")
    _require_repo(req.workspace)
    staged = git_diff_mod.git_diff(req.workspace, staged=True)
    unstaged = git_diff_mod.git_diff(req.workspace, staged=False)
    entries = git_status_mod.git_status(req.workspace)
    diff_text = (staged or "") + (unstaged or "")
    if not entries and not diff_text:
        return {"ok": True, "clean": True, "review": "Working tree is clean — nothing to review.",
                "entries": [], "diff": ""}

    from core.retrieval.budget import truncate_to_tokens

    context, anchors, truncated = prepare_review(staged, unstaged)
    findings: list[dict] = []
    review_status = "insufficient_context"
    review = "Changes exist, but no added text lines are available for grounded review."
    if anchors:
        try:
            raw = await state.inference.chat(
                [{"role": "system", "content": REVIEW_PROMPT},
                 {"role": "user", "content": context}],
                temperature=0.1, max_tokens=1024, schema=Review.model_json_schema(),
            )
            validated = validate_review(raw, anchors)
            review = render_review(validated)
            findings = [f.model_dump() for f in validated.findings]
            review_status = "validated"
        except ValueError:
            review_status = "invalid_output"
            review = "Model review rejected: invalid JSON or unsupported file/line references."
        except Exception:  # llama offline: still return the raw diff
            review_status = "unavailable"
            review = "Model unavailable; inspect the raw diff."
    limitations = []
    if truncated:
        limitations.append("Diff truncated; only supplied lines were reviewed.")
    if any(e["worktree"] == "untracked" for e in entries):
        limitations.append("Untracked file contents are not included in Git diffs.")
    if limitations:
        review += "\n\n" + "\n".join(limitations)

    return {
        "ok": True,
        "clean": False,
        "branch": git_ops.current_branch(req.workspace),
        "entries": entries,
        "review": review,
        "review_status": review_status,
        "findings": findings,
        "diff": truncate_to_tokens(diff_text, 6000),
    }


@router.post("/commit")
async def commit(req: CommitRequest, confirm: str | None = Header(default=None, alias="X-Forge-Confirm"),
                 state: AppState = Depends(get_state)) -> dict:
    """Stage (optionally) and commit. Requires X-Forge-Confirm: true."""
    if confirm != "true":
        return {"ok": False, "error": "Committing requires confirmation (X-Forge-Confirm: true)"}
    check_permission("GIT_WRITE", granted=PERMISSIONS_AUTO | {"GIT_WRITE"})
    _require_repo(req.workspace)
    try:
        staged = git_ops.stage(req.workspace, req.paths if req.paths else None)
        if staged == 0:
            return {"ok": False, "error": "Nothing staged to commit"}
        result = git_ops.commit(req.workspace, req.message)
    except git_ops.GitOperationError as exc:
        return {"ok": False, "error": str(exc)}
    return {"ok": True, "staged": staged, **result}


@router.post("/push")
async def push(req: WorkspaceRequest, confirm: str | None = Header(default=None, alias="X-Forge-Confirm"),
               state: AppState = Depends(get_state)) -> dict:
    """Push the current branch to origin. Requires X-Forge-Confirm: true."""
    if confirm != "true":
        return {"ok": False, "error": "Push requires confirmation (X-Forge-Confirm: true)"}
    check_permission("GIT_WRITE", granted=PERMISSIONS_AUTO | {"GIT_WRITE"})
    _require_repo(req.workspace)
    if not git_ops.remote_url(req.workspace):
        return {"ok": False, "error": "No 'origin' remote configured for this repository"}
    try:
        result = git_ops.push(req.workspace)
    except git_ops.GitOperationError as exc:
        return {"ok": False, "error": str(exc)}
    return {"ok": True, **result}


@router.post("/pull")
async def pull(req: WorkspaceRequest, confirm: str | None = Header(default=None, alias="X-Forge-Confirm"),
               state: AppState = Depends(get_state)) -> dict:
    """Pull origin into the current branch. Requires X-Forge-Confirm: true."""
    if confirm != "true":
        return {"ok": False, "error": "Pull requires confirmation (X-Forge-Confirm: true)"}
    check_permission("GIT_WRITE", granted=PERMISSIONS_AUTO | {"GIT_WRITE"})
    _require_repo(req.workspace)
    try:
        result = git_ops.pull(req.workspace)
    except git_ops.GitOperationError as exc:
        return {"ok": False, "error": str(exc)}
    return {"ok": True, **result}
