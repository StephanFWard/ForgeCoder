"""GET /health — server, llama.cpp, and database status."""
from __future__ import annotations

from fastapi import APIRouter, Depends

from forge_server import __version__
from forge_server.main import AppState, get_state

router = APIRouter(tags=["health"])


@router.get("/health")
async def health(state: AppState = Depends(get_state)) -> dict:
    llama_ok = await state.inference.ping()
    return {
        "status": "ok" if llama_ok else "degraded",
        "version": __version__,
        "llama_server": {
            "ok": llama_ok,
            "url": state.config.llama_url,
        },
        "database": {
            "ok": state.index.is_ready(),
            "path": str(state.config.db_path_path),
            **state.index.stats(),
        },
        "memory": {
            "available_gb": round(_available_gb(), 2),
        },
    }


def _available_gb() -> float:
    from core.memory.process import available_ram_gb

    return available_ram_gb()
