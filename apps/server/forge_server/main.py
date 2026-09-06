"""Forge server entrypoint.

    GET  /health
    POST /v1/chat
    POST /v1/completion
    POST /v1/explain | /v1/edit | /v1/fix | /v1/tests
    POST /v1/index | /v1/search
    POST /v1/patch/preview | /v1/patch/apply
"""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

from forge_server import __version__
from forge_server.config import Config
from forge_server.security import assert_local_request

log = logging.getLogger("forge_server")

# Within fair use these endpoints are always registered; keep the list in
# sync with docs/api.md.
_ACTIONS_MODULE = None


def create_app(config: Config | None = None) -> FastAPI:
    cfg = config or Config.load()
    _ensure_dirs(cfg)

    state = AppState(cfg)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        state.index.connect()
        log.info("Forge server ready on %s:%s (db=%s)", cfg.host, cfg.port, cfg.db_path_path)
        yield
        await state.inference.aclose()
        state.index.close()

    app = FastAPI(title="ForgeCoder Forge API", version=__version__, lifespan=lifespan)
    app.state.forge = state

    # The chat WebView runs in a vscode-webview origin; allow it while keeping
    # the API otherwise loopback-only.
    app.add_middleware(
        CORSMiddleware,
        allow_origin_regex=r"(?:https?://(?:127\.0\.0\.1|localhost)|vscode-webview://.*)",
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    from forge_server.chat import router as chat_router
    from forge_server.completion import router as completion_router
    from forge_server.health import router as health_router
    from forge_server.patches import router as patches_router

    app.include_router(health_router)
    app.include_router(chat_router)
    app.include_router(completion_router)
    app.include_router(patches_router)

    # Retrieval + code-action endpoints live in their own modules for clarity.
    try:
        from forge_server.actions import router as actions_router
        from forge_server.retrieval import router as retrieval_router

        app.include_router(actions_router)
        app.include_router(retrieval_router)
    except Exception:  # pragma: no cover - never crash core API on import issue
        log.exception("Failed to register optional routers")

    @app.middleware("http")
    async def loopback_guard(request: Request, call_next):
        assert_local_request(request.client.host if request.client else "127.0.0.1")
        return await call_next(request)

    return app


def _ensure_dirs(cfg: Config) -> None:
    cfg.data_dir_path.mkdir(parents=True, exist_ok=True)
    cfg.log_dir_path.mkdir(parents=True, exist_ok=True)
    cfg.db_path_path.parent.mkdir(parents=True, exist_ok=True)


class AppState:
    """Wires together config, index, and inference client for dependency injection."""

    def __init__(self, config: Config):
        from core.indexer.index import CodeIndex
        from core.inference.client import InferenceClient

        self.config = config
        self.index = CodeIndex(config.db_path_path)
        self.inference = InferenceClient(config.llama_url)


def get_state(request: Request) -> AppState:
    return request.app.state.forge


def run() -> None:
    import uvicorn

    cfg = Config.load()
    uvicorn.run("forge_server.main:app", host=cfg.host, port=cfg.port, log_level="info")


app = create_app()


if __name__ == "__main__":
    run()
