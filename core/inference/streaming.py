"""Server-Sent-Events parsing for llama.cpp streaming responses."""
from __future__ import annotations

import json
from collections.abc import AsyncIterator

import httpx


async def iter_sse_lines(response: httpx.Response) -> AsyncIterator[str]:
    """Yield raw ``data:`` payloads from an SSE stream."""
    buffer = ""
    async for raw in response.aiter_lines():
        line = raw if not buffer else buffer + raw
        buffer = ""
        line = line.strip()
        if not line:
            continue
        if line.startswith("data:"):
            payload = line[5:].strip()
            if payload == "[DONE]":
                return
            if payload:
                yield payload
        elif line.startswith(":") or line.startswith("event:"):
            continue


def parse_delta(payload: str) -> str | None:
    """Extract the text delta from an SSE JSON payload (None on end)."""
    try:
        data = json.loads(payload)
    except json.JSONDecodeError:
        return None
    choices = data.get("choices") or []
    if not choices:
        return None
    choice = choices[0]
    finish = choice.get("finish_reason")
    delta = choice.get("delta") or {}
    text = choice.get("text")
    content = delta.get("content")
    if content is not None:
        return content
    if text is not None:
        return text
    if finish:
        return None
    return None


async def iter_chat_deltas(response: httpx.Response) -> AsyncIterator[str]:
    """Yield content deltas for /v1/chat/completions streaming."""
    async for payload in iter_sse_lines(response):
        delta = parse_delta(payload)
        if delta:
            yield delta


async def iter_completion_deltas(response: httpx.Response) -> AsyncIterator[str]:
    """Yield text deltas for /v1/completions streaming."""
    async for payload in iter_sse_lines(response):
        delta = parse_delta(payload)
        if delta:
            yield delta
