"""Inference layer: a thin async client for llama.cpp's OpenAI-compatible API."""

from core.inference.chat import build_chat_messages
from core.inference.client import InferenceClient
from core.inference.completion import build_fim_prompt
from core.inference.streaming import iter_chat_deltas, iter_completion_deltas

__all__ = [
    "InferenceClient",
    "build_chat_messages",
    "build_fim_prompt",
    "iter_chat_deltas",
    "iter_completion_deltas",
]
