"""Search/explain/test agents: one function, one agent."""
from __future__ import annotations
from core.agent.base import FunctionAgent, _conf_dict, _intent_dict, _prob
from core.agent.context import Recommendation

class SearchAgent(FunctionAgent):
    def __init__(self):
        super().__init__(name="search", description="Repo search.",
                         tools=["search_repo", "view_file"], behavior="chat")
    async def run(self, ctx, *, inference=None):
        try:
            from core.agent import tool_impl as _impl  # noqa: F401
        except Exception:
            pass
        from core.agent.tools import get_tool
        t = get_tool("search_repo")
        if t is None or t.arun is None:
            return Recommendation(kind="error", text="search tool missing", agent="search")
        res = await t.arun(ctx, ctx.effective_request(), 8)
        if not res.get("ok"):
            return Recommendation(kind="error", text=res.get("error", "?"), agent="search")
        rows = res.get("results", [])
        text = "\n\n".join(f"# {r['path']}:{r['start_line']}-{r['end_line']}\n{r['content'][:600]}" for r in rows)
        return Recommendation(kind="view", text=text or "(no matches)",
            probability=_prob(ctx), actions=["view"], agent="search",
            confidence=_conf_dict(ctx), intent=_intent_dict(ctx),
            meta={"results": rows})

class ExplainAgent(FunctionAgent):
    def __init__(self):
        super().__init__(name="explain", description="Explain code.",
                         tools=["view_file", "search_repo"], behavior="chat")
    async def run(self, ctx, *, inference=None):
        await ctx.build(index=getattr(ctx, "_index", None), inference=inference)
        from core.agent.confidence import format_probability
        from core.inference.chat import build_chat_messages
        from core.retrieval.budget import truncate_to_tokens
        user = ctx.effective_request()
        if ctx.code:
            user += f"\n\nSelected code:\n{ctx.code}"
        if ctx.evidence_text:
            user += "\n\nRepository context:\n" + truncate_to_tokens(ctx.evidence_text, 2000)
        try:
            text = await inference.chat(build_chat_messages(ctx.system_text or "x", user))
        except Exception as exc:
            return Recommendation(kind="error", text=str(exc), agent="explain")
        p = _prob(ctx)
        return Recommendation(kind="answer", text=f"{format_probability(p)}\n\n{text}",
            probability=p, actions=[], agent="explain",
            confidence=_conf_dict(ctx), intent=_intent_dict(ctx))

class TestAgent(FunctionAgent):
    def __init__(self):
        super().__init__(name="test", description="Generate tests.",
                         tools=["view_file", "search_repo"], behavior="test")
    async def run(self, ctx, *, inference=None):
        await ctx.build(index=getattr(ctx, "_index", None), inference=inference)
        from core.agent.confidence import format_probability
        from core.inference.chat import build_chat_messages
        user = ctx.effective_request() or "Generate tests for the selected code."
        if ctx.code:
            user += f"\n\nCode:\n{ctx.code}"
        try:
            text = await inference.chat(build_chat_messages(ctx.system_text or "x", user))
        except Exception as exc:
            return Recommendation(kind="error", text=str(exc), agent="test")
        p = _prob(ctx)
        return Recommendation(kind="answer", text=f"{format_probability(p)}\n\n{text}",
            probability=p, actions=["view", "test"], agent="test",
            confidence=_conf_dict(ctx), intent=_intent_dict(ctx))

SEARCH_AGENT = SearchAgent()
EXPLAIN_AGENT = ExplainAgent()
TEST_AGENT = TestAgent()
