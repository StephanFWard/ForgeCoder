"""Chat/answer agent: plain questions, no patch branching in UI."""
from __future__ import annotations
from core.agent.base import FunctionAgent, _conf_dict, _intent_dict, _prob
from core.agent.context import Recommendation

class ChatAgent(FunctionAgent):
    def __init__(self):
        super().__init__(name="chat", description="Answer with repo context.",
                         tools=["view_file", "search_repo"], behavior="chat")

    async def run(self, ctx, *, inference=None):
        await ctx.build(index=getattr(ctx, "_index", None), inference=inference)
        from core.inference.chat import build_chat_messages
        from core.retrieval.budget import truncate_to_tokens
        user = ctx.effective_request()
        if ctx.evidence_text:
            user += "\n\nRepository context:\n" + truncate_to_tokens(ctx.evidence_text, 2600)
        msgs = build_chat_messages(ctx.system_text or "You are ForgeCoder.", user)
        try:
            text = await inference.chat(msgs, temperature=0.3, max_tokens=1024)
        except Exception as exc:
            return Recommendation(kind="error", text=str(exc), agent="chat")
        from core.agent.confidence import format_probability
        p = _prob(ctx)
        return Recommendation(kind="answer",
            text=f"{format_probability(p)}\n\n{text}", probability=p,
            confidence=_conf_dict(ctx), intent=_intent_dict(ctx),
            actions=[], agent="chat")

AGENT = ChatAgent()
