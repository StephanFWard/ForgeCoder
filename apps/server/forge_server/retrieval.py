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
    # Opt-in weighted rerank: heuristic matches blended with System One score
    # answers (free, offline by default). Omitted -> the plain FTS5 + heuristic
    # path, unchanged.
    weighted: bool = False
    system_one_weight: float = 0.35


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
    limit = min(req.limit, 50)
    workspace = req.workspace.replace("\\", "/") if req.workspace else None
    rows = state.index.search(req.query, limit=limit, workspace=workspace)
    if not req.weighted:
        return {
            "ok": True,
            "query": req.query,
            "weighted": False,
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

    from core.system_one.rerank import rank_chunks_weighted

    ranked = rank_chunks_weighted(rows, req.query, limit=limit,
                                  system_one_weight=req.system_one_weight,
                                  client=state.inference)
    return {
        "ok": True,
        "query": req.query,
        "weighted": True,
        "backend": ranked[0].get("_system_one_backend") if ranked else None,
        "results": [
            {
                "path": r["path"],
                "language": r["language"],
                "content": r["content"],
                "start_line": r["start_line"],
                "end_line": r["end_line"],
                "score": r.get("_score", 0),
                "weighted_score": r.get("_weighted", 0.0),
                "system_one": r.get("_system_one", 0.0),
                "system_one_confidence": r.get("_system_one_confidence", 0.0),
            }
            for r in ranked
        ],
    }
