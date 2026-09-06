"""Fill-in-the-middle (FIM) prompt building for autocomplete.

Qwen2.5-Coder uses the standard ChatML FIM tokens:

    <|fim_prefix|>PREFIX<|fim_suffix|>SUFFIX<|fim_middle|>
"""
from __future__ import annotations

FIM_PREFIX = "<|fim_prefix|>"
FIM_SUFFIX = "<|fim_suffix|>"
FIM_MIDDLE = "<|fim_middle|>"
FIM_END = "<|fim_end|>"
EOS = "<|endoftext|>"

DEFAULT_STOP: list[str] = [FIM_END, EOS, "<|im_end|>"]

# Autocomplete targets: fast and conservative, not conversational.
DEFAULT_TEMPERATURE = 0.1
DEFAULT_MAX_TOKENS = 64
DEFAULT_CONTEXT_BUDGET = 1024


def build_fim_prompt(prefix: str, suffix: str = "", *, budget: int = DEFAULT_CONTEXT_BUDGET) -> str:
    """Compose the FIM prompt, trimming prefix/suffix to the completion budget."""
    from core.retrieval.budget import truncate_to_tokens

    reserved = budget // 2
    trimmed_prefix = truncate_to_tokens(prefix or "", reserved)
    trimmed_suffix = truncate_to_tokens(suffix or "", budget - reserved)
    return f"{FIM_PREFIX}{trimmed_prefix}{FIM_SUFFIX}{trimmed_suffix}{FIM_MIDDLE}"


def build_completion_context(prior_lines: str, next_lines: str, *, budget: int = 512) -> tuple[str, str]:
    """Trim surrounding lines for a completion request (imports + function body)."""
    from core.retrieval.budget import truncate_to_tokens

    half = max(64, budget // 2)
    return truncate_to_tokens(prior_lines or "", half), truncate_to_tokens(next_lines or "", half)
