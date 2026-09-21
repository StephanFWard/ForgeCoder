"""ForgeAgent: routes one AgentContext to one FunctionAgent."""
from __future__ import annotations
from typing import Any
from core.agent.base import FunctionAgent
from core.agent.context import AgentContext, Recommendation

class ForgeAgent:
    def __init__(self, agents=None):
        from core.agent import misc_agents as _m
        from core.agent import tool_impl as _t  # noqa: F401 (register tools)
        from core.agent.chat_agent import AGENT as _chat
        from core.agent.edit_agent import EDIT_AGENT as _edit, FIX_AGENT as _fix
        from core.agent.jev_agent import JEV_AGENT as _jev
        from core.agent.review_agent import AGENT as _review
        self.agents: dict[str, FunctionAgent] = {}
        for a in [_chat, _edit, _fix, _m.SEARCH_AGENT, _m.EXPLAIN_AGENT,
                  _m.TEST_AGENT, _review, _jev]:
            self.agents[a.name] = a
        for a in (agents or []):
            self.agents[a.name] = a

    def pick(self, ctx: AgentContext) -> FunctionAgent:
        hint = (ctx.behavior_hint or "").lower()
        if hint in self.agents:
            return self.agents[hint]
        if hint == "create":
            return self.agents["edit"]
        intent = getattr(ctx, "intent", None)
        if intent is not None and not bool(getattr(intent, "code_change", False)):
            return self.agents["chat"]
        msg = (ctx.message or ctx.instruction or "").lower()
        if "test" in msg and ("generat" in msg or "write" in msg):
            return self.agents["test"]
        if "review" in msg or "diff" in msg or "change" in msg:
            if not any(w in msg for w in ("fix", "edit", "refactor", "add", "update")):
                return self.agents["review"]
        return self.agents["chat"]

    async def run(self, ctx: AgentContext, *, index=None,
                  inference=None, agent: str | None = None) -> Recommendation:
        ctx._index = index  # noqa: SLF001 (tools read it for search)
        if agent and agent in self.agents:
            chosen = self.agents[agent]
        elif ctx.behavior_hint in (self.agents or {}):
            chosen = self.agents[str(ctx.behavior_hint)]
        else:
            if ctx.intent is None or not ctx.built_up_to_date:
                await ctx.build(index=index, inference=inference)
            chosen = self.pick(ctx)
        rec = await chosen.run(ctx, inference=inference)
        rec = self._autofill(ctx, rec)
        return rec

    def _autofill(self, ctx, rec: Recommendation) -> Recommendation:
        acts = list(rec.actions or [])
        def _has(*names):
            return any(a in acts for a in names)
        if rec.kind in ("patch", "multi") and rec.patches:
            for a in ("view", "diff", "apply"):
                if a not in acts:
                    acts.append(a)
        elif rec.kind == "view" and not _has("view"):
            acts.append("view")
        elif rec.kind == "review" and "view" not in acts:
            acts.append("view")
        rec.actions = acts
        return rec

DEFAULT = ForgeAgent()
