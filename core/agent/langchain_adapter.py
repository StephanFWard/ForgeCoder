"""LangChain adapter: expose Forge agents as LC tools (optional dep)."""
from __future__ import annotations
from typing import Any

def langchain_available() -> bool:
    try:
        import langchain_core  # noqa: F401
        return True
    except Exception:
        return False

def forge_tool_as_lc(tool_name: str):
    from core.agent.tools import get_tool
    t = get_tool(tool_name)
    if t is None:
        raise KeyError(f"unknown tool {tool_name!r}")
    try:
        from langchain_core.tools import StructuredTool
    except Exception as exc:
        raise ImportError("langchain-core not installed") from exc
    async def _run(**kwargs):
        ctx = kwargs.pop("_ctx", None) or kwargs.pop("ctx", None)
        if t.arun is not None:
            return await t.arun(ctx, **kwargs)
        return t.run(ctx, **kwargs)
    return StructuredTool.from_function(
        func=lambda **k: {"ok": False, "error": "use arun"},
        coroutine=_run, name=t.name, description=t.description)


def forge_agent_as_runnable(agent_name: str):
    """Wrap a Forge FunctionAgent as a LangChain Runnable.

    Each agent becomes its own runnable, allowing LangChain chains/agents
    to invoke Forge capabilities directly.
    """
    try:
        from langchain_core.runnables import RunnableLambda
    except Exception as exc:
        raise ImportError("langchain-core not installed") from exc
    from core.agent.forge import ForgeAgent
    forge = ForgeAgent()
    if agent_name not in forge.agents:
        raise KeyError(f"unknown agent {agent_name!r}; available: {list(forge.agents.keys())}")

    def _invoke(payload):
        import asyncio
        from core.agent.context import AgentContext
        ctx = payload if hasattr(payload, "effective_request") else AgentContext.from_request(
            str(payload.get("message", "")), workspace=payload.get("workspace"),
            file=payload.get("file"), selection=payload.get("selection"),
            behavior_hint=payload.get("agent") or agent_name)
        rec = asyncio.run(forge.run(ctx, agent=agent_name))
        return rec.to_dict()
    return RunnableLambda(_invoke)


def context_to_lc_messages(ctx) -> list:
    try:
        from langchain_core.messages import HumanMessage, SystemMessage
    except Exception as exc:
        raise ImportError("langchain-core not installed") from exc
    msgs = []
    if ctx.system_text:
        msgs.append(SystemMessage(content=ctx.system_text))
    msgs.append(HumanMessage(content=ctx.effective_request()
                             + ("\n\n" + ctx.evidence_text if ctx.evidence_text else "")))
    return msgs


def get_langchain_agents() -> dict[str, Any]:
    """Return all Forge agents as LangChain runnables.

    Each function is its own agent/runnable, suitable for use in LangChain
    workflows, multi-agent systems, or as tools in a LangGraph graph.
    """
    from core.agent.forge import ForgeAgent
    forge = ForgeAgent()
    agents = {}
    for name, agent in forge.agents.items():
        agents[name] = forge_agent_as_runnable(name)
    return agents


def jev_as_langchain_tool():
    """Jev as a LangChain tool with context-awareness.

    Jev is the core context-aware agent that builds context, classifies intent,
    scores confidence, and recommends actions automatically.
    """
    try:
        from langchain_core.tools import StructuredTool
    except Exception as exc:
        raise ImportError("langchain-core not installed") from exc

    async def _jev_run(**kwargs):
        import asyncio
        from core.agent.context import AgentContext
        from core.agent.jev_agent import JEV_AGENT

        ctx = kwargs.pop("_ctx", None) or kwargs.pop("ctx", None)
        if ctx is None:
            ctx = AgentContext.from_request(
                str(kwargs.get("message", "")),
                workspace=kwargs.get("workspace"),
                file=kwargs.get("file"),
                selection=kwargs.get("selection"),
                behavior_hint="jev"
            )
        rec = await JEV_AGENT.run(ctx)
        return rec.to_dict()

    return StructuredTool.from_function(
        name="jev_context_aware",
        description="Jev: context-aware agent that builds workspace context, "
                    "classifies intent, scores confidence, and recommends actions. "
                    "Returns view/diff/apply recommendations automatically.",
        coroutine=_jev_run,
        return_direct=True
    )
