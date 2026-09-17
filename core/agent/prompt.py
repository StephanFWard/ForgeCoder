"""Layered prompt assembly: a small router, depth on demand.

The reason an instruction file keeps growing is that every incident adds a rule
and no incident removes one; a year in, the model reads the first screen and
acts on a fraction of what it was told. The fix is not a shorter file but a
layered one: the behavior prompt stays short, the rules that apply to *this*
behavior are listed inline (slug, severity, one line), and the full rule router
with its checks stays in ``runtime/prompts/agent-rules.md`` where it can be
reviewed and diffed.

The same layering applies to the user turn: request first, then the task frame
(goal, receipts, bounds, proof, unknowns), then the evidence. Nothing here calls
a model or the network.
"""
from __future__ import annotations

from collections.abc import Iterable

from core.agent.frame import TaskFrame, render_frame
from core.agent.rules import Rule, load_rules, rules_for_behavior

RULES_HEADER = (
    "OPERATING RULES (your reply is checked against them before it is shown). "
    "block = refuse the action and name what is missing. "
    "warn = a caveat to respect while answering. "
    "Never print rule slugs, severities, or [warn]/[block] markers in your reply."
)
RULES_POINTER = "The full router with the check behind each rule is runtime/prompts/agent-rules.md."
DEFAULT_USER_TURN_TOKENS = 3800


def _base_prompt(behavior: str) -> str:
    # Imported here because core.retrieval.context owns the on-disk prompt
    # loader and imports this module for the rule layer.
    from core.retrieval.context import load_prompt

    return load_prompt(behavior)


def system_prompt(behavior: str, rules: list[Rule] | None = None) -> str:
    """The behavior prompt plus only the rules that behavior must hold to."""
    base = _base_prompt(behavior)
    inline = rules_for_behavior(rules if rules is not None else load_rules(), behavior)
    if not inline:
        return base
    lines = [RULES_HEADER]
    lines.extend(f"- {rule.one_line()}" for rule in inline)
    lines.append(RULES_POINTER)
    return f"{base}\n\n" + "\n".join(lines)


def render_task_frame(frame: TaskFrame | None) -> str:
    """Frame block for the user turn; empty string when there is no frame."""
    return render_frame(frame) if frame is not None else ""


def compose_user_turn(parts: Iterable[str | None], *, frame: TaskFrame | None = None,
                      context_text: str = "",
                      context_label: str = "Repository context",
                      max_tokens: int = DEFAULT_USER_TURN_TOKENS) -> str:
    """Request → frame → evidence, fitted into the user-turn token allowance.

    The request comes first because it is the goal; the frame turns that goal
    into allowed paths, proof, and open questions; the evidence comes last so
    the source the model must ground line numbers in is the freshest text in the
    turn. Context is truncated by whole lines to whatever budget remains.

    ``core.retrieval.budget`` is imported at call time because the package
    ``__init__`` imports ``core.retrieval.context``, which imports this module
    for the rule layer — a module-level import here would be circular.
    """
    from core.retrieval.budget import estimate_tokens, truncate_to_tokens

    body = "\n\n".join(part for part in parts if part)
    if frame is not None:
        body = f"{body}\n\n{render_frame(frame)}" if body else render_frame(frame)
    if context_text:
        remaining = max(0, max_tokens - estimate_tokens(body))
        context = truncate_to_tokens(context_text, remaining)
        if context:
            body = f"{body}\n\n{context_label}:\n{context}" if body else f"{context_label}:\n{context}"
    return body


__all__ = [
    "DEFAULT_USER_TURN_TOKENS",
    "RULES_HEADER",
    "RULES_POINTER",
    "compose_user_turn",
    "render_task_frame",
    "system_prompt",
]
