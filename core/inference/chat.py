"""Chat message assembly (OpenAI-compatible message list)."""
from __future__ import annotations

from collections.abc import Iterable


def build_chat_messages(system: str, user: str, history: Iterable[dict] | None = None) -> list[dict]:
    """Build the message list for /v1/chat/completions.

    ``history`` entries are ``{"role": "user"|"assistant", "content": str}``.
    """
    messages: list[dict] = [{"role": "system", "content": system}]
    if history:
        messages.extend(
            {"role": m.get("role", "user"), "content": m.get("content", "")}
            for m in history
            if isinstance(m, dict) and m.get("role") in {"user", "assistant"}
        )
    messages.append({"role": "user", "content": user})
    return messages


def estimate_history_tokens(history: Iterable[dict]) -> int:
    """Cheap token estimate for limiting history inside the chat budget."""
    from core.retrieval.budget import estimate_tokens

    return sum(estimate_tokens(m.get("content", "")) for m in history)
