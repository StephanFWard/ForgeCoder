"""POST /v1/chat — streaming chat with repository context."""
from __future__ import annotations

import json
from collections.abc import AsyncIterator
from datetime import datetime

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse

from core.agent.confidence import answer_confidence, format_probability
from core.agent.creation import is_creation_request
from core.agent.intent import classify_intent
from core.inference.chat import build_chat_messages
from core.inference.verify import verify_output
from core.retrieval.adaptive import classify_query
from core.retrieval.budget import estimate_tokens, truncate_to_tokens
from core.retrieval.context import ContextBuilder
from core.retrieval.hierarchical import HierarchicalContextBuilder
from forge_server.context import ContextRequest
from forge_server.main import AppState, get_state
from forge_server.security import check_permission

router = APIRouter(prefix="/v1", tags=["chat"])


def _sse(data: dict) -> str:
    return f"data: {json.dumps(data, ensure_ascii=False)}\n\n"


async def _chat_creation(req: ContextRequest, state: AppState) -> StreamingResponse:
    """Creation request over plain chat: prepare files, offer the patch bar.

    The heavy lifting (create-schema generation with a corrective retry,
    existing-file patches refused) is the same ``_edit_step`` the act pipeline
    uses. The client already knows how to render a ``patch`` event into its
    review/apply bar, so a creation request behaves like any other patch.
    """
    from forge_server.plans import _edit_step  # lazy: imports forge_server.main

    check_permission("READ_FILE")
    prepared = await _edit_step(state, req.workspace, req.message, {},
                                plan_request=req.message)

    async def event_stream():
        yield _sse({"type": "context", "total_tokens": 0, "sections": ["creation"]})
        if not prepared:
            yield _sse({"type": "error", "message": (
                "Creation failed: the model did not produce a valid file set. "
                "Try Plan & Act mode for a step-by-step run."
            )})
            yield _sse({"type": "done"})
            return
        patches = prepared.get("files", [])
        creates = prepared.get("creates", {})
        names = [str(f.get("path", "?")) for f in patches] + [str(k) for k in creates]
        summary = prepared.get("message") or f"Prepared {len(names)} file(s)."
        yield _sse({"type": "delta", "content": (
            summary + "\n\nReady to write: " + ", ".join(names)
            + " — review before applying."
        )})
        yield _sse({"type": "patch", "patches": patches, "creates": creates,
                    "message": summary})
        yield _sse({"type": "done"})

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@router.post("/chat")
async def chat(req: ContextRequest, state: AppState = Depends(get_state)) -> StreamingResponse:
    """Build context (retrieval + file selection + git), call llama.cpp, stream back.

    Creation requests ("make the game snake in html") are routed to the
    creation pipeline instead of free-form streaming: plain chat had no
    creation routing, so the model streamed echoes of its task frame —
    ``[warn] ask-dont-guess: ...`` lines — instead of a game. Here the reply is
    a real ``creates`` patch surfaced through the standard review/apply flow.
    """
    if is_creation_request(req.message):
        return await _chat_creation(req, state)

    # System One intent gate (free): is this a code-change task at all? A
    # plain question must be answered, not framed — the edit frame's
    # unknowables ("no acceptance command was stated; ...") are what came back
    # as fabricated "[warn]" lines when users asked "Is a hot dog a sandwich?".
    intent = await classify_intent(
        req.message, client=state.inference,
        has_file=bool(req.file), has_selection=bool(req.selection),
    )
    builder = ContextBuilder(state.index)
    built = builder.build(
        req.message,
        workspace=req.workspace,
        file=req.file,
        selection=req.selection.as_tuple() if req.selection else None,
        budget=state.config.max_chat_context,
        code_change=intent.code_change,
    )

    history_budget = max(200, state.config.max_chat_context - max(built.total_tokens, 0) - 500)
    history: list[dict] = []
    seen_history: set[str] = set()
    for m in req.history[-8:]:
        content = truncate_to_tokens(str(m.get("content", "")), 350)
        role = "assistant" if m.get("role") == "assistant" else "user"
        key = role + ":" + content
        if not content or key in seen_history:
            continue
        cost = estimate_tokens(content)
        if cost > history_budget:
            break
        history.append({"role": role, "content": content})
        seen_history.add(key)
        history_budget -= cost

    request_text = _build_user_turn(built)
    system = built.system + f"\n\nToday's date: {datetime.now():%Y-%m-%d (%A)}."
    messages = build_chat_messages(system, request_text, history)

    # The statistical probability that opens every answer: how confident the
    # System One layer is that this turn is answerable from what was supplied.
    confidence = await answer_confidence(
        req.message, truncate_to_tokens(context_turn_text(built), 2600),
        client=state.inference,
    )
    marker = format_probability(confidence.probability)

    async def event_stream() -> AsyncIterator[str]:
        yield _sse({"type": "context", "total_tokens": built.total_tokens,
                    "sections": [s["type"] for s in built.sections],
                    "intent": intent.to_dict(),
                    "confidence": confidence.to_dict()})
        # The probability leads the answer so accuracy is visible first.
        yield _sse({"type": "delta", "content": f"{marker}\n\n"})
        try:
            async for delta in state.inference.chat_stream(
                messages, temperature=0.3, max_tokens=1024,
                presence_penalty=0.2, frequency_penalty=0.2,
            ):
                yield _sse({"type": "delta", "content": delta})
        except Exception as exc:
            yield _sse({"type": "error", "message": str(exc)})
        yield _sse({"type": "done"})

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@router.post("/chat/hierarchical")
async def chat_hierarchical(req: ContextRequest, state: AppState = Depends(get_state)) -> dict:
    """Hierarchical chat using RNP tree-structured context + grammar-constrained output."""
    qtype = classify_query(req.message)
    builder = HierarchicalContextBuilder(state.index)
    built = builder.build(req.message, workspace=req.workspace, budget=3072)

    system = built.system + f"\n\nToday's date: {datetime.now():%Y-%m-%d (%A)}."
    context_text = built.as_message_text()
    messages = build_chat_messages(system, context_text + "\n\n" + req.message, req.history)

    grammar = _grammar_for_intent(qtype)

    try:
        answer = await state.inference.chat(
            messages, temperature=0.3, max_tokens=1024,
            presence_penalty=0.2, frequency_penalty=0.2,
            grammar=grammar,
        )
    except Exception as exc:
        return {"ok": False, "error": str(exc)}

    result = verify_output(answer, context_text=built.as_message_text(),
                           expected_json_keys=_json_keys_for_intent(qtype))

    return {
        "ok": True,
        "answer": answer,
        "query_type": qtype,
        "grammar_used": grammar is not None,
        "verification": result.to_dict(),
        "context_tokens": built.total_tokens,
        "tree_summary": built.tree_summary(),
    }


@router.post("/chat/verify")
async def chat_verify(req: ContextRequest, state: AppState = Depends(get_state)) -> dict:
    """Verify a model output against retrieved context (RNP predictive coding)."""
    builder = HierarchicalContextBuilder(state.index)
    built = builder.build(req.message, workspace=req.workspace, budget=3072)
    context_text = built.as_message_text()

    output_to_verify = req.message
    qtype = classify_query(req.message)

    result = verify_output(
        output_to_verify,
        context_text=context_text,
        expected_json_keys=_json_keys_for_intent(qtype),
    )

    return {
        "ok": True,
        "verification": result.to_dict(),
        "context_tokens": built.total_tokens,
    }


def _grammar_for_intent(query_type: str) -> str | None:
    """Return a GBNF grammar string for the given query intent."""
    from pathlib import Path
    base = Path(__file__).resolve().parents[3] / "runtime" / "grammars"
    grammar_map = {"fix": "fix.gbnf", "explain": "explain.gbnf",
                   "plan": "plan.gbnf", "edit": "fix.gbnf"}
    fname = grammar_map.get(query_type)
    if fname:
        path = base / fname
        try:
            return path.read_text(encoding="utf-8")
        except OSError:
            return None
    return None


def _json_keys_for_intent(query_type: str) -> list[str] | None:
    keys = {"fix": ["summary", "diagnosis", "files"],
            "explain": ["topic", "explanation", "references"],
            "plan": ["summary", "steps", "risks"],
            "edit": ["summary", "files"]}
    return keys.get(query_type)


@router.post("/ask")
async def ask(req: ContextRequest, state: AppState = Depends(get_state)) -> dict:
    """Non-streaming chat (used by the MCP server and simple clients)."""
    builder = ContextBuilder(state.index)
    built = builder.build(
        req.message,
        workspace=req.workspace,
        file=req.file,
        selection=req.selection.as_tuple() if req.selection else None,
        budget=state.config.max_chat_context,
    )
    context_text = "\n\n".join(s["text"] for s in built.sections)
    user = req.message
    if context_text:
        user += f"\n\nRepository context:\n{truncate_to_tokens(context_text, 2600)}"

    system = built.system + f"\n\nToday's date: {datetime.now():%Y-%m-%d (%A)}."
    messages = build_chat_messages(system, user, None)
    try:
        answer = await state.inference.chat(
            messages, temperature=0.3, max_tokens=1024,
            presence_penalty=0.2, frequency_penalty=0.2,
        )
    except Exception as exc:
        return {"ok": False, "error": str(exc)}
    # The probability leads the answer so accuracy is visible first.
    confidence = await answer_confidence(req.message, truncate_to_tokens(context_text, 2600),
                                         client=state.inference)
    return {
        "ok": True,
        "answer": f"{format_probability(confidence.probability)}\n\n{answer}",
        "probability": round(confidence.probability, 6),
        "context_tokens": built.total_tokens,
    }


def _build_user_turn(built) -> str:
    """User turn = request + repository sections, fitted to the request budget."""
    total = built.request or ""
    for section in built.sections:
        if section["type"] in {"repository", "file"}:
            total += "\n\n" + section["text"]
    return truncate_to_tokens(total, 3800)


def context_turn_text(built) -> str:
    """Evidence-only text — the state the answer-confidence question is asked over."""
    return "\n\n".join(
        s["text"] for s in built.sections if s["type"] in {"repository", "file"})
