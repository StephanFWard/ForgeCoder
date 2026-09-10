"""Inference layer: a thin async client for llama.cpp's OpenAI-compatible API."""

from core.inference.chat import build_chat_messages
from core.inference.client import InferenceClient, InferenceError
from core.inference.completion import build_fim_prompt
from core.inference.streaming import iter_chat_deltas, iter_completion_deltas
from core.inference.verify import (
    VerificationError,
    VerificationResult,
    extract_citations,
    verify_or_raise,
    verify_output,
)

__all__ = [
    "InferenceClient",
    "InferenceError",
    "VerificationError",
    "VerificationResult",
    "build_chat_messages",
    "build_fim_prompt",
    "extract_citations",
    "iter_chat_deltas",
    "iter_completion_deltas",
    "verify_or_raise",
    "verify_output",
]
