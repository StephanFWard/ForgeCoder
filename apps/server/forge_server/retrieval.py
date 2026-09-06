"""Retrieval endpoints: index a workspace and search it."""
from __future__ import annotations

from fastapi import APIRouter, Depends
from pydantic import BaseModel

from forge_server.main import AppState, get_state
from forge_server.security import ForgeSecurityError, check_permission, resolve_workspace_path

router = APIRouter(prefix="/v1", tags=["retrieval"])


class IndexRequest(BaseModel):
    workspace: str


class SearchRequest(BaseModel):
    query: str
    workspace: str | None = None
    limit: int = 10


@router.post("/index")
async def index(req: IndexRequest, state: AppState = Depends(get_state)) -> dict:
    """Index (incrementally) the user's workspace. Never the whole disk."""
    check_permission("READ_FILE")
    try:
        root = resolve_workspace_path(req.workspace, ".")
    except ForgeSecurityError as exc:
        return {"ok": False, "error": str(exc)}
    stats = state.index.index_workspace(root)
    return {"ok": True, "workspace": str(root), **stats}


@router.post("/search")
async def search(req: SearchRequest, state: AppState = Depends(get_state)) -> dict:
    check_permission("SEARCH")
    rows = state.index.search(
        req.query,
        limit=min(req.limit, 50),
        workspace=req.workspace.replace("\\", "/") if req.workspace else None,
    )
    return {
        "ok": True,
        "query": req.query,
        "results": [
            {
                "path": r["path"],
                "language": r["language"],
                "content": r["content"],
                "start_line": r["start_line"],
                "end_line": r["end_line"],
            }
            for r in rows
        ],
    }
