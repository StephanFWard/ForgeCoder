"""Function agents: one agent per Forge capability."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any
from core.agent.context import AgentContext, Recommendation

@dataclass
class FunctionAgent:
    name: str
    description: str = ""
    tools: list[str] = field(default_factory=list)
    behavior: str = "chat"

    async def run(self, ctx: AgentContext, *, inference=None) -> Recommendation:
        raise NotImplementedError

    def run_sync(self, ctx: AgentContext, *, inference=None) -> Recommendation:
        import asyncio
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self.run(ctx, inference=inference))
        raise RuntimeError("run_sync() inside a running loop; await run()")

def _prob(ctx) -> float:
    try:
        return float(getattr(ctx.confidence, "probability", 0.5))
    except Exception:
        return 0.5

def _intent_dict(ctx):
    try:
        return ctx.intent.to_dict() if ctx.intent else None
    except Exception:
        return None

def _conf_dict(ctx):
    try:
        return ctx.confidence.to_dict() if ctx.confidence else None
    except Exception:
        return None

__all__ = ["FunctionAgent", "AgentContext", "Recommendation"]
