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

    review_prompt = (
        "Review these working-tree changes. For each file: what changed, why it "
        "matters, any bugs or risks introduced, and one suggested next step "
        "(test, refactor, or commit). Be specific and concise.\n\n"
        f"Changed files: {', '.join(e['path'] for e in entries[:20]) or 'none'}\n\n"
        f"Diff:\n{truncate_to_tokens(diff_text, 1800)}"
    )
    review = ""
    try:
        review = await state.inference.chat(
            [{"role": "system", "content": (
                "You are ForgeCoder reviewing uncommitted git changes. Use only "
                "the supplied diff. If the diff is empty, say the tree is clean."
            )},
             {"role": "user", "content": review_prompt}],
            temperature=0.2, max_tokens=512,
            presence_penalty=0.2, frequency_penalty=0.2,
        )
    except Exception as exc:  # llama offline: still return the raw diff
        review = f"(model unavailable: {exc})"

    return {
        "ok": True,
        "clean": False,
        "branch": git_ops.current_branch(req.workspace),
        "entries": entries,
        "review": review,
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
