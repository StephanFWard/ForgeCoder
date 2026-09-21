"""Jev as a context-aware agent (not just a decorator)."""
from __future__ import annotations
from typing import Any
from core.agent.base import FunctionAgent, _conf_dict, _intent_dict, _prob
from core.agent.context import AgentContext, Recommendation, ACTION_VIEW, ACTION_DIFF, ACTION_APPLY
from core.agent.confidence import format_probability

class JevAgent(FunctionAgent):
    """System-One decisions grounded in AgentContext evidence.

    Jev is the core context-aware agent that:
    1. Builds context from the workspace
    2. Classifies intent
    3. Scores confidence
    4. Makes decisions based on evidence
    5. Recommends actions automatically (view, diff, apply)
    """
    def __init__(self):
        super().__init__(name="jev", description="Typed Jev decisions over context.",
                         tools=["view_file", "search_repo"], behavior="chat")

    def state_for(self, ctx: AgentContext) -> str:
        return ctx.state_text()

    async def ask(self, ctx, questions, *, inference=None, backend=None):
        from core.system_one.decide import Decider
        await ctx.build(index=getattr(ctx, "_index", None), inference=inference)
        dec = Decider(backend=backend or ("local" if inference else None),
                      client=inference)
        try:
            return await dec.adecide(ctx.state_text(), questions)
        except Exception:
            dec2 = Decider()
            return dec2.decide(ctx.state_text(), questions)

    async def run(self, ctx, *, inference=None):
        await ctx.build(index=getattr(ctx, "_index", None), inference=inference)

        from core.system_one.decide import Decider
        from core.system_one.primitives import NoulQuestion

        # Question 1: Is this answerable from context?
        q = {"answerable": NoulQuestion(
            instructions="Can this be answered from the supplied context?",
            criteria={"true": "answerable from context",
                      "false": "needs missing information"})}
        dec = await self.ask(ctx, q, inference=inference)
        ans = dec.answers.get("answerable")
        p = float(getattr(ans, "noul", 0.5)) if ans else 0.5

        # Build recommendation based on confidence and context
        actions = []
        kind = "answer"
        text = ""

        # If context shows file/code changes are involved, recommend view/diff
        has_file = bool(ctx.file)
        has_code = bool(ctx.code)
        has_selection = bool(ctx.selection)

        if has_file or has_code or has_selection:
            actions.extend([ACTION_VIEW, ACTION_DIFF])

        # If confidence is high and there's evidence of a fix needed, suggest apply
        if p >= 0.7 and has_file:
            actions.append(ACTION_APPLY)

        # Determine response kind based on analysis
        if p >= 0.8:
            kind = "answer"
            text = format_probability(p) + "\n\n" + ctx.evidence_text[:500]
        elif p >= 0.5:
            kind = "answer"
            text = format_probability(p) + "\n\nContext available but confidence is moderate."
        else:
            kind = "answer"
            text = format_probability(p) + "\n\nInsufficient context to answer confidently."

        return Recommendation(
            kind=kind,
            text=text,
            probability=p,
            actions=actions,
            agent="jev",
            confidence=_conf_dict(ctx),
            intent=_intent_dict(ctx),
            meta={"backend": dec.backend, "free": dec.free, "has_file": has_file,
                  "has_code": has_code, "p": p}
        )

JEV_AGENT = JevAgent()
