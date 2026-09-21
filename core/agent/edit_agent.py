"""Edit/fix agents: scored patches, actions auto-filled."""
from __future__ import annotations
from core.agent.base import FunctionAgent, _conf_dict, _intent_dict, _prob
from core.agent.context import Recommendation
from core.agent.patch_flow import generate_structured, parse_and_score

def _patch_dicts(patches):
    return [{"path": x.path, "operations": [o.__dict__ for o in x.operations]} for x in patches]

class EditAgent(FunctionAgent):
    def __init__(self):
        super().__init__(name="edit", description="Structured edits.",
                         tools=["view_file", "search_repo", "preview_patch", "apply_patch"],
                         behavior="edit")
    async def run(self, ctx, *, inference=None):
        from core.agent.confidence import format_probability
        from core.patching.contracts import EDIT_SCHEMA
        try:
            text = await generate_structured(ctx, inference, EDIT_SCHEMA)
        except Exception as exc:
            return Recommendation(kind="error", text=str(exc), agent="edit")
        try:
            payload, patches, report, vr = parse_and_score(ctx.built, text)
        except Exception as exc:
            return Recommendation(kind="answer", text=text, probability=_prob(ctx),
                actions=["view"], confidence=_conf_dict(ctx),
                intent=_intent_dict(ctx), agent="edit",
                meta={"parse_error": str(exc)})
        if report.blocked:
            return Recommendation(kind="answer", text=text, probability=_prob(ctx),
                review=report.to_dict(), verification=vr.to_dict(),
                actions=["view"], confidence=_conf_dict(ctx),
                intent=_intent_dict(ctx), agent="edit")
        p = _prob(ctx)
        return Recommendation(kind="patch",
            text=f"{format_probability(p)} {payload.get('summary', '')}",
            probability=p, patches=_patch_dicts(patches),
            review=report.to_dict(), verification=vr.to_dict(),
            confidence=_conf_dict(ctx), intent=_intent_dict(ctx),
            actions=["view", "diff", "apply"], agent="edit")

class FixAgent(EditAgent):
    def __init__(self):
        super().__init__()
        self.name = "fix"
        self.behavior = "fix"
    async def run(self, ctx, *, inference=None):
        from core.agent.confidence import format_probability
        from core.patching.contracts import FIX_SCHEMA
        try:
            text = await generate_structured(ctx, inference, FIX_SCHEMA)
        except Exception as exc:
            return Recommendation(kind="error", text=str(exc), agent="fix")
        try:
            payload, patches, report, vr = parse_and_score(ctx.built, text)
        except Exception as exc:
            return Recommendation(kind="answer", text=text, probability=_prob(ctx),
                actions=["view"], confidence=_conf_dict(ctx),
                intent=_intent_dict(ctx), agent="fix",
                meta={"parse_error": str(exc)})
        if report.blocked:
            return Recommendation(kind="answer", text=text, probability=_prob(ctx),
                review=report.to_dict(), verification=vr.to_dict(),
                actions=["view"], confidence=_conf_dict(ctx),
                intent=_intent_dict(ctx), agent="fix",
                meta={"diagnosis": payload.get("diagnosis", "")})
        p = _prob(ctx)
        return Recommendation(kind="patch",
            text=f"{format_probability(p)} {payload.get('summary', '')}",
            probability=p, patches=_patch_dicts(patches),
            review=report.to_dict(), verification=vr.to_dict(),
            confidence=_conf_dict(ctx), intent=_intent_dict(ctx),
            actions=["view", "diff", "apply"], agent="fix",
            meta={"diagnosis": payload.get("diagnosis", "")})

EDIT_AGENT = EditAgent()
FIX_AGENT = FixAgent()
