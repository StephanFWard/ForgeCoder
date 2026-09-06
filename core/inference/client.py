"""Async client for llama.cpp's OpenAI-compatible HTTP API.

Only uses the documented endpoints (/health, /v1/chat/completions,
/v1/completions) so the inference engine can be swapped later without
rewriting retrieval or the VS Code extension.
"""
from __future__ import annotations

from collections.abc import AsyncIterator

import httpx

DEFAULT_BASE_URL = "http://127.0.0.1:8080"
DEFAULT_TIMEOUT = httpx.Timeout(connect=5.0, read=300.0, write=60.0, pool=30.0)


class InferenceError(RuntimeError):
    """Raised when llama.cpp is unreachable or returns an error."""


class InferenceClient:
    def __init__(self, base_url: str = DEFAULT_BASE_URL, timeout: httpx.Timeout = DEFAULT_TIMEOUT):
        self.base_url = base_url.rstrip("/")
        self._timeout = timeout
        self._client: httpx.AsyncClient | None = None

    @property
    def client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(base_url=self.base_url, timeout=self._timeout)
        return self._client

    async def aclose(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    # ------------------------------------------------------------- health
    async def ping(self, timeout: float = 3.0) -> bool:
        try:
            resp = await self.client.get("/health", timeout=timeout)
            return resp.status_code == 200
        except (httpx.HTTPError, OSError):
            return False

    async def server_info(self) -> dict:
        try:
            resp = await self.client.get("/props", timeout=3.0)
            resp.raise_for_status()
            return resp.json()
        except (httpx.HTTPError, ValueError, OSError):
            return {}

    # ------------------------------------------------------------- chat
    async def chat(self, messages: list[dict], *, temperature: float = 0.2,
                   max_tokens: int = 1024, stop: list[str] | None = None,
                   stream: bool = False,
                   presence_penalty: float = 0.0, frequency_penalty: float = 0.0) -> str:
        if stream:
            text = ""
            async for delta in self.chat_stream(messages, temperature=temperature,
                                                max_tokens=max_tokens, stop=stop,
                                                presence_penalty=presence_penalty,
                                                frequency_penalty=frequency_penalty):
                text += delta
            return text
        payload: dict = {
            "messages": messages,
            "temperature": temperature,
            "max_tokens": min(max_tokens, 4096),
            "stream": False,
            "presence_penalty": presence_penalty,
            "frequency_penalty": frequency_penalty,
        }
        if stop:
            payload["stop"] = stop
        try:
            resp = await self.client.post("/v1/chat/completions", json=payload)
            resp.raise_for_status()
            data = resp.json()
        except httpx.HTTPError as exc:
            raise InferenceError(f"llama.cpp chat failed: {exc}") from exc
        try:
            return data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise InferenceError(f"Unexpected llama.cpp response: {data!r}") from exc

    async def chat_stream(self, messages: list[dict], *, temperature: float = 0.2,
                          max_tokens: int = 1024, stop: list[str] | None = None,
                          presence_penalty: float = 0.0,
                          frequency_penalty: float = 0.0) -> AsyncIterator[str]:
        from core.inference.streaming import iter_chat_deltas

        payload: dict = {
            "messages": messages,
            "temperature": temperature,
            "max_tokens": min(max_tokens, 4096),
            "stream": True,
            "presence_penalty": presence_penalty,
            "frequency_penalty": frequency_penalty,
        }
        if stop:
            payload["stop"] = stop
        try:
            async with self.client.stream("POST", "/v1/chat/completions", json=payload) as resp:
                resp.raise_for_status()
                async for delta in iter_chat_deltas(resp):
                    yield delta
        except httpx.HTTPError as exc:
            raise InferenceError(f"llama.cpp chat stream failed: {exc}") from exc

    # ------------------------------------------------------------- completion
    async def complete(self, prompt: str, *, temperature: float = 0.1,
                       max_tokens: int = 64, stop: list[str] | None = None) -> str:
        payload: dict = {
            "prompt": prompt,
            "temperature": temperature,
            "max_tokens": min(max_tokens, 1024),
            "stream": False,
        }
        if stop:
            payload["stop"] = stop
        try:
            resp = await self.client.post("/v1/completions", json=payload)
            resp.raise_for_status()
            data = resp.json()
        except httpx.HTTPError as exc:
            raise InferenceError(f"llama.cpp completion failed: {exc}") from exc
        try:
            return data["choices"][0]["text"]
        except (KeyError, IndexError, TypeError) as exc:
            raise InferenceError(f"Unexpected llama.cpp response: {data!r}") from exc

    async def complete_stream(self, prompt: str, *, temperature: float = 0.1,
                              max_tokens: int = 64, stop: list[str] | None = None) -> AsyncIterator[str]:
        from core.inference.streaming import iter_completion_deltas

        payload: dict = {
            "prompt": prompt,
            "temperature": temperature,
            "max_tokens": min(max_tokens, 1024),
            "stream": True,
        }
        if stop:
            payload["stop"] = stop
        try:
            async with self.client.stream("POST", "/v1/completions", json=payload) as resp:
                resp.raise_for_status()
                async for delta in iter_completion_deltas(resp):
                    yield delta
        except httpx.HTTPError as exc:
            raise InferenceError(f"llama.cpp completion stream failed: {exc}") from exc
