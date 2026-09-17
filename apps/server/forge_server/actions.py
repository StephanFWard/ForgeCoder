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
from core.patching.contracts import EDIT_SCHEMA, FIX_SCHEMA
from core.patching.parser import PatchParseError, extract_json, parse_patch
from core.retrieval.budget import truncate_to_tokens
from core.retrieval.context import ContextBuilder
from forge_server.context import ContextRequest
from forge_server.main import AppState, get_state

router = APIRouter(prefix="/v1", tags=["actions"])


class ActionRequest(ContextRequest):
    message: str = ""              # actions can be driven by code/instruction instead
    code: str | None = None        # raw selection text when provided directly
    error: str | None = None       # for /v1/fix
    instruction: str | None = None # e.g. "add null validation"


async def _run(state: AppState, req: ActionRequest, behavior: str, *,
               schema: dict | None = None) -> str:
    """One model call for a code action.

    ``behavior`` selects the behavior-specific system prompt (chat/edit/fix/
    test) via ContextBuilder, and the retrieved repository context is injected
    into the user turn so the model can produce line numbers that refer to
    real files instead of hallucinating them.

    ``schema`` grammar-constrains the reply via llama.cpp's ``json_schema``
    response format, so a structured action cannot come back as prose.
    """
    builder = ContextBuilder(state.index)
    built = builder.build(
        req.message or req.instruction or "",
        workspace=req.workspace,
        file=req.file,
        selection=req.selection.as_tuple() if req.selection else None,
        budget=state.config.max_chat_context,
        behavior=behavior,
    )
    parts = [req.message, req.instruction]
    if req.code:
        parts.append(f"Selected code:\n{req.code}")
    if req.error:
        parts.append(f"Error output:\n{req.error}")
    user = "\n\n".join(part for part in parts if part)
    context_text = "\n\n".join(s["text"] for s in built.sections)
    if context_text:
        user = f"{user}\n\nRepository context:\n{truncate_to_tokens(context_text, 2000)}"
    messages = build_chat_messages(built.system, user)
    return await state.inference.chat(messages, temperature=0.1, max_tokens=1024, schema=schema)


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
    text = await _run(state, req, "edit", schema=EDIT_SCHEMA)
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
    text = await _run(state, req, "fix", schema=FIX_SCHEMA)
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
