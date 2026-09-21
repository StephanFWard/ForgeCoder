"""Review/git agent: view/diff handled by agent context."""
from __future__ import annotations
from core.agent.base import FunctionAgent
from core.agent.context import Recommendation

class ReviewAgent(FunctionAgent):
    def __init__(self):
        super().__init__(name="review", description="Grounded diff review.",
                         tools=["git_status", "git_diff", "view_file"],
                         behavior="chat")
    async def run(self, ctx, *, inference=None):
        await ctx.build(index=getattr(ctx, "_index", None), inference=inference)
        from core.git.diff import git_diff
        from core.git.review import prepare_review, render_review, validate_review
        from core.inference.chat import build_chat_messages
        ws = ctx.workspace or ""
        try:
            staged, unstaged = git_diff(ws, staged=True), git_diff(ws, staged=False)
        except Exception as exc:
            return Recommendation(kind="error", text=str(exc), agent="review")
        bundle, anchors, _tr = prepare_review(staged, unstaged)
        if not anchors:
            return Recommendation(kind="view", text="clean tree",
                actions=["view"], agent="review",
                meta={"diff": staged + unstaged})
        prompt = ("Review only the supplied diff; return JSON summary+findings.\n\n" + bundle)
        try:
            raw = await inference.chat(build_chat_messages("reviewer", prompt))
            review = validate_review(raw, anchors)
            text = render_review(review)
            status = "validated"
            findings = [f.model_dump() for f in review.findings]
        except Exception:
            text, status, findings = bundle, "unavailable", []
        return Recommendation(kind="review", text=text, actions=["view", "diff"],
            agent="review", meta={"review_status": status, "findings": findings,
                                  "diff": staged + unstaged})

AGENT = ReviewAgent()
