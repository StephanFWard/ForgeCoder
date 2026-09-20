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


def sanitize_history(history: Iterable[dict] | None, *, limit: int = 8,
                     per_turn: int = 350) -> list[dict]:
    """History safe to send back to the model: one entry per question, no markers.

    ForgeCoder's own ``[p=...]`` confidence markers must never re-enter the
    prompt: they are ForgeCoder's verdict *about* a turn, not user evidence,
    and a 1.5B model shown ``[p=0.23]`` above a previous answer will anchor on
    the number and echo it as if it were the answer to the new question.
    Stored assistant turns that *are* the raw verdict JSON (``{\"yes\": ...}``)
    are dropped for the same reason. Dedup keeps the newest copy.
    """
    from core.retrieval.budget import truncate_to_tokens

    if not history:
        return []
    cleaned: list[dict] = []
    seen: set[str] = set()
    for entry in reversed(list(history)[-limit:]):
        if not isinstance(entry, dict):
            continue
        role = "assistant" if entry.get("role") == "assistant" else "user"
        content = truncate_to_tokens(str(entry.get("content", "")), per_turn)
        if not content:
            continue
        stripped = content.strip()
        if stripped.startswith("[p="):
            # Own confidence marker: strip the marker, keep the prose (if any).
            stripped = stripped.split("]", 1)[-1].strip() if "]" in stripped else ""
            if not stripped:
                continue
            content = stripped
        if role == "assistant" and stripped.startswith("{\"yes\""):
            continue  # raw verdict JSON leaked into history: never evidence
        key = role + ":" + content
        if key in seen:
            continue
        seen.add(key)
        cleaned.append({"role": role, "content": content})
    cleaned.reverse()
    return cleaned
