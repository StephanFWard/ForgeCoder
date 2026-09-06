"""POST /v1/completion — fill-in-the-middle autocomplete (fast & conservative)."""
from __future__ import annotations

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field

from core.inference.client import InferenceError
from core.inference.completion import (
    DEFAULT_MAX_TOKENS,
    DEFAULT_STOP,
    DEFAULT_TEMPERATURE,
    build_fim_prompt,
)
from forge_server.main import AppState, get_state

router = APIRouter(prefix="/v1", tags=["completion"])


class CompletionRequest(BaseModel):
    language: str
    file: str = ""
    prefix: str
    suffix: str = ""
    context: list[str] = Field(default_factory=list)  # imports + header context
    temperature: float = DEFAULT_TEMPERATURE
    max_tokens: int = DEFAULT_MAX_TOKENS


@router.post("/completion")
async def completion(req: CompletionRequest, state: AppState = Depends(get_state)) -> dict:
    """Return a single conservative completion for the cursor position."""
    prompt = build_fim_prompt(req.prefix, req.suffix,
                              budget=state.config.max_completion_context)
    try:
        text = await state.inference.complete(
            prompt,
            temperature=req.temperature,
            max_tokens=min(req.max_tokens, 128),
            stop=DEFAULT_STOP,
        )
    except InferenceError as exc:
        return {"completion": "", "error": str(exc)}
    return {"completion": text.strip(), "language": req.language, "file": req.file}
