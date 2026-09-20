"""Code-action endpoints: explain / edit / fix / tests.

These wrap chat with behavior-specific prompts, and for edit/fix additionally
return a machine-applicable structured patch when the model complies with the
JSON grammar in runtime/grammars/patch.gbnf.
"""
from __future__ import annotations

import json

from fastapi import APIRouter, Depends

from core.agent import RuleContext, RuleReport, load_rules, render_task_frame, run_rules
from core.agent.confidence import answer_confidence, format_probability
from core.inference.chat import build_chat_messages
from core.inference.client import InferenceError
from core.patching.contracts import EDIT_SCHEMA, FIX_SCHEMA
from core.patching.parser import PatchParseError, extract_json, parse_patch
from core.retrieval.budget import truncate_to_tokens
from core.retrieval.context import BuiltContext, ContextBuilder
from forge_server.context import ContextRequest
from forge_server.main import AppState, get_state

router = APIRouter(prefix="/v1", tags=["actions"])


class ActionRequest(ContextRequest):
    message: str = ""              # actions can be driven by code/instruction instead
    code: str | None = None        # raw selection text when provided directly
    error: str | None = None       # for /v1/fix
    instruction: str | None = None # e.g. "add null validation"


async def _run(state: AppState, req: ActionRequest, behavior: str, *,
               schema: dict | None = None) -> tuple[str, BuiltContext]:
    """One model call for a code action.

    ``behavior`` selects the behavior-specific system prompt (chat/edit/fix/
    test) via ContextBuilder, the task frame (goal, receipts, bounds, unknowns)
    rides in the user turn, and the retrieved repository context is injected so
    the model can produce line numbers that refer to real files instead of
    hallucinating them. ``built`` is returned with the reply so the caller can
    score it against the rules with the same context the model saw.

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
    frame_text = render_task_frame(built.frame)
    if frame_text:
        user = f"{user}\n\n{frame_text}" if user else frame_text
    context_text = "\n\n".join(s["text"] for s in built.sections)
    if context_text:
        user = f"{user}\n\nRepository context:\n{truncate_to_tokens(context_text, 2000)}"
    messages = build_chat_messages(built.system, user)
    reply = await state.inference.chat(messages, temperature=0.1, max_tokens=1024, schema=schema)
    return reply, built


def _score(behavior: str, built: BuiltContext, text: str, patches: list) -> RuleReport:
    """Score a structured reply against the rules, using the model's own view.

    The scope, line counts, and unknowns come from the same ``BuiltContext``
    that fed the prompt, so a finding means the reply disagrees with evidence
    it was actually given — never with evidence it never saw.
    """
    contract = built.contract
    ctx = RuleContext(
        behavior=behavior,
        text=text,
        patches=patches,
        has_context=bool(built.sections),
        allowed_files=list(contract.allowed_files) if contract else [],
        forbidden_files=list(contract.forbidden_files) if contract else [],
        file_lines=dict(built.file_lines),
        open_unknowns=list(built.frame.unknowns) if built.frame else [],
    )
    return run_rules(load_rules(), ctx)


def _evidence_text(built: BuiltContext) -> str:
    """Evidence-only text — the state the answer-confidence question is asked over."""
    return "\n\n".join(s["text"] for s in built.sections)


async def _confidence(req: ActionRequest, built: BuiltContext, state: AppState):
    """System One probability that this turn is answerable from its evidence."""
    message = req.message or req.instruction or ""
    return await answer_confidence(message, _evidence_text(built), client=state.inference)


@router.post("/explain")
async def explain(req: ActionRequest, state: AppState = Depends(get_state)) -> dict:
    try:
        text, built = await _run(state, req, "chat")
    except InferenceError as exc:
        return {"ok": False, "error": str(exc)}
    confidence = await _confidence(req, built, state)
    return {
        "ok": True,
        # The probability leads the answer so accuracy is visible first.
        "explanation": f"{format_probability(confidence.probability)}\n\n{text}",
        "probability": round(confidence.probability, 6),
    }


@router.post("/tests")
async def tests(req: ActionRequest, state: AppState = Depends(get_state)) -> dict:
    if not req.code and not req.file:
        return {"ok": False, "error": "Provide code or a file to generate tests for"}
    try:
        text, built = await _run(state, req, "test")
    except InferenceError as exc:
        return {"ok": False, "error": str(exc)}
    confidence = await _confidence(req, built, state)
    return {
        "ok": True,
        "tests": f"{format_probability(confidence.probability)}\n\n{text}",
        "probability": round(confidence.probability, 6),
    }


@router.post("/edit")
async def edit(req: ActionRequest, state: AppState = Depends(get_state)) -> dict:
    """Return a structured patch (summary + files) or explain why it failed."""
    text, built = await _run(state, req, "edit", schema=EDIT_SCHEMA)
    try:
        payload = extract_json(text)
        patches = parse_patch(payload)
    except (PatchParseError, json.JSONDecodeError) as exc:
        return {"ok": False, "reason": "no_structured_patch", "error": str(exc),
                "proposal": text}
    report = _score("edit", built, text, patches)
    if report.blocked:
        return {"ok": False, "reason": "rule_violation",
                "findings": [f.to_dict() for f in report.findings]}
    confidence = await _confidence(req, built, state)
    return {
        "ok": True,
        "summary": f"{format_probability(confidence.probability)} {payload.get('summary', '')}",
        "probability": round(confidence.probability, 6),
        "patches": [{"path": p.path, "operations": [o.__dict__ for o in p.operations]} for p in patches],
        "review": report.to_dict(),
    }


@router.post("/fix")
async def fix(req: ActionRequest, state: AppState = Depends(get_state)) -> dict:
    """Explain + structured patch for a bug/error context."""
    if not req.error and not req.code:
        return {"ok": False, "error": "Provide error output or code to fix"}
    text, built = await _run(state, req, "fix", schema=FIX_SCHEMA)
    try:
        payload = extract_json(text)
        patches = parse_patch(payload)
    except (PatchParseError, json.JSONDecodeError) as exc:
        return {"ok": False, "reason": "no_structured_patch", "error": str(exc),
                "diagnosis": text}
    report = _score("fix", built, text, patches)
    if report.blocked:
        return {"ok": False, "reason": "rule_violation",
                "findings": [f.to_dict() for f in report.findings]}
    confidence = await _confidence(req, built, state)
    return {
        "ok": True,
        "summary": f"{format_probability(confidence.probability)} {payload.get('summary', '')}",
        "probability": round(confidence.probability, 6),
        "diagnosis": payload.get("diagnosis", ""),
        "patches": [{"path": p.path, "operations": [o.__dict__ for o in p.operations]} for p in patches],
        "review": report.to_dict(),
    }
