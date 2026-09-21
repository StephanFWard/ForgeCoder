"""Agent tools: verbs every FunctionAgent can call."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Callable
@dataclass
class ForgeTool:
    name: str
    description: str
    needs_confirm: bool = False
    run: Any = None
    arun: Any = None
    meta: dict[str, Any] = field(default_factory=dict)
_REGISTRY: dict[str, ForgeTool] = {}
def tool(name, description, *, needs_confirm=False, meta=None):
    def deco(fn):
        import asyncio
        t = ForgeTool(name=name, description=description,
                      needs_confirm=needs_confirm, meta=dict(meta or {}))
        if asyncio.iscoroutinefunction(fn):
            t.arun = fn
        else:
            t.run = fn
        _REGISTRY[name] = t
        return fn
    return deco
def list_tools():
    return list(_REGISTRY.values())
def get_tool(name):
    return _REGISTRY.get(name)
__all__ = ["ForgeTool", "list_tools", "get_tool", "tool"]
