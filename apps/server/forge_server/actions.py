"""Code-action endpoints: explain / edit / fix / tests.

These wrap chat with behavior-specific prompts, and for edit/fix additionally
return a machine-applicable structured patch when the model complies with the
JSON grammar in runtime/grammars/patch.gbnf.
"""
from __future__ import annotations

import json

from fastapi import APIRouter, Depends

from core.inference.chat import build_chat_messages
from core.inference.client import InferenceError
from core.patching.parser import PatchParseError, extract_json, parse_patch
from core.retrieval.context import ContextBuilder, load_prompt
from forge_server.context import ContextRequest
from forge_server.main import AppState, get_state

router = APIRouter(prefix="/v1", tags=["actions"])


class ActionRequest(ContextRequest):
    message: str = ""              # actions can be driven by code/instruction instead
    code: str | None = None        # raw selection text when provided directly
    error: str | None = None       # for /v1/fix
    instruction: str | None = None # e.g. "add null validation"


def _behavior(behavior: str) -> str:
    return load_prompt(behavior)


async def _run(state: AppState, req: ActionRequest, behavior: str) -> str:
    builder = ContextBuilder(state.index)
    built = builder.build(
        req.message or req.instruction or "",
        workspace=req.workspace,
        file=req.file,
        selection=req.selection.as_tuple() if req.selection else None,
        budget=state.config.max_chat_context,
    )
    target = req.code or built.request
    user = f"{target}\n\n{req.error if req.error else ''}".strip()
    messages = build_chat_messages(built.system, user)
    return await state.inference.chat(messages, temperature=0.1, max_tokens=1024)


@router.post("/explain")
async def explain(req: ActionRequest, state: AppState = Depends(get_state)) -> dict:
    try:
        text = await _run(state, req, "chat")
    except InferenceError as exc:
        return {"ok": False, "error": str(exc)}
    return {"ok": True, "explanation": text}


@router.post("/tests")
async def tests(req: ActionRequest, state: AppState = Depends(get_state)) -> dict:
    if not req.code and not req.file:
        return {"ok": False, "error": "Provide code or a file to generate tests for"}
    try:
        text = await _run(state, req, "test")
    except InferenceError as exc:
        return {"ok": False, "error": str(exc)}
    return {"ok": True, "tests": text}


@router.post("/edit")
async def edit(req: ActionRequest, state: AppState = Depends(get_state)) -> dict:
    """Return a structured patch (summary + files) or explain why it failed."""
    text = await _run(state, req, "edit")
    try:
        payload = extract_json(text)
        patches = parse_patch(payload)
    except (PatchParseError, json.JSONDecodeError) as exc:
        return {"ok": False, "reason": "no_structured_patch", "error": str(exc),
                "proposal": text}
    return {
        "ok": True,
        "summary": payload.get("summary", ""),
        "patches": [{"path": p.path, "operations": [o.__dict__ for o in p.operations]} for p in patches],
    }


@router.post("/fix")
async def fix(req: ActionRequest, state: AppState = Depends(get_state)) -> dict:
    """Explain + structured patch for a bug/error context."""
    if not req.error and not req.code:
        return {"ok": False, "error": "Provide error output or code to fix"}
    text = await _run(state, req, "fix")
    try:
        payload = extract_json(text)
        patches = parse_patch(payload)
    except (PatchParseError, json.JSONDecodeError) as exc:
        return {"ok": False, "reason": "no_structured_patch", "error": str(exc),
                "diagnosis": text}
    return {
        "ok": True,
        "summary": payload.get("summary", ""),
        "diagnosis": payload.get("diagnosis", ""),
        "patches": [{"path": p.path, "operations": [o.__dict__ for o in p.operations]} for p in patches],
    }
