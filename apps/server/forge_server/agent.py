"""POST /v1/agent/run — one context-aware entrypoint for all agents."""
from __future__ import annotations
from fastapi import APIRouter, Depends
from pydantic import BaseModel
from forge_server.main import AppState, get_state

router = APIRouter(prefix="/v1/agent", tags=["agent"])

class AgentRunRequest(BaseModel):
    message: str = ""
    workspace: str | None = None
    file: str | None = None
    selection: dict | None = None
    history: list[dict] = []
    code: str | None = None
    error: str | None = None
    instruction: str | None = None
    confirmed: bool = False
    agent: str | None = None

@router.post("/run")
async def run(req: AgentRunRequest, state: AppState = Depends(get_state)) -> dict:
    from core.agent.context import AgentContext
    from core.agent.forge import ForgeAgent
    ctx = AgentContext.from_request(
        req.message, workspace=req.workspace, file=req.file,
        selection=req.selection, history=req.history, code=req.code,
        error=req.error, instruction=req.instruction,
        confirmed=req.confirmed,
        behavior_hint=(req.agent or "").lower() or None)
    rec = await ForgeAgent().run(ctx, index=state.index,
                                 inference=state.inference,
                                 agent=(req.agent or "").lower() or None)
    return {"ok": True, **rec.to_dict()}

@router.get("/agents")
async def agents() -> dict:
    from core.agent.forge import ForgeAgent
    forge = ForgeAgent()
    return {"ok": True, "agents": [
        {"name": a.name, "description": a.description,
         "tools": list(a.tools)} for a in forge.agents.values()]}
