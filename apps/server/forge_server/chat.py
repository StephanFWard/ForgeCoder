"""POST /v1/chat — streaming chat with repository context."""
from __future__ import annotations

import json
from collections.abc import AsyncIterator

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse

from core.inference.chat import build_chat_messages
from core.retrieval.budget import estimate_tokens, truncate_to_tokens
from core.retrieval.context import ContextBuilder
from forge_server.context import ContextRequest
from forge_server.main import AppState, get_state

router = APIRouter(prefix="/v1", tags=["chat"])


def _sse(data: dict) -> str:
    return f"data: {json.dumps(data, ensure_ascii=False)}\n\n"


@router.post("/chat")
async def chat(req: ContextRequest, state: AppState = Depends(get_state)) -> StreamingResponse:
    """Build context (retrieval + file selection + git), call llama.cpp, stream back."""
    builder = ContextBuilder(state.index)
    built = builder.build(
        req.message,
        workspace=req.workspace,
        file=req.file,
        selection=req.selection.as_tuple() if req.selection else None,
        budget=state.config.max_chat_context,
    )

    # History is fitted into the remaining chat budget.
    history_budget = max(200, state.config.max_chat_context - max(built.total_tokens, 0) - 500)
    history = []
    for m in req.history[-6:]:
        cost = estimate_tokens(m.get("content", ""))
        if cost > history_budget:
            break
        history.append({"role": "assistant" if m.get("role") == "assistant" else "user",
                        "content": m.get("content", "")})
        history_budget -= cost

    request_text = _build_user_turn(built)
    messages = build_chat_messages(built.system, request_text, history)

    async def event_stream() -> AsyncIterator[str]:
        yield _sse({"type": "context", "total_tokens": built.total_tokens,
                    "sections": [s["type"] for s in built.sections]})
        try:
            async for delta in state.inference.chat_stream(
                messages, temperature=0.2, max_tokens=1024
            ):
                yield _sse({"type": "delta", "content": delta})
        except Exception as exc:  # surface llama.cpp failures to the UI
            yield _sse({"type": "error", "message": str(exc)})
        yield _sse({"type": "done"})

    return StreamingResponse(event_stream(), media_type="text/event-stream")


def _build_user_turn(built) -> str:
    """User turn = request + repository sections, fitted to the request budget."""
    total = built.request or ""
    for section in built.sections:
        if section["type"] in {"repository", "file"}:
            total += "\n\n" + section["text"]
    return truncate_to_tokens(total, 3800)
