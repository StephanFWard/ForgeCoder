"""Shared patch generation helper for edit/fix agents."""
from __future__ import annotations

async def generate_structured(ctx, inference, schema):
    await ctx.build(index=getattr(ctx, "_index", None), inference=inference)
    from core.inference.chat import build_chat_messages
    from core.retrieval.budget import truncate_to_tokens
    parts = [ctx.message, ctx.instruction]
    if ctx.code:
        parts.append(f"Selected code:\n{ctx.code}")
    if ctx.error:
        parts.append(f"Error output:\n{ctx.error}")
    user = "\n\n".join(p for p in parts if p)
    frame = getattr(ctx.built, "frame", None)
    if frame is not None:
        try:
            from core.agent.prompt import render_task_frame
            ft = render_task_frame(frame)
            if ft:
                user = f"{user}\n\n{ft}" if user else ft
        except Exception:
            pass
    if ctx.evidence_text:
        user += "\n\nRepository context:\n" + truncate_to_tokens(ctx.evidence_text, 2000)
    msgs = build_chat_messages(ctx.system_text or "You are ForgeCoder.", user)
    return await inference.chat(msgs, temperature=0.1, max_tokens=1024, schema=schema)

def parse_and_score(built, text):
    from core.agent import RuleContext, load_rules, run_rules
    from core.inference.verify import verify_output
    from core.patching.parser import extract_json, parse_patch
    payload = extract_json(text)
    patches = parse_patch(payload)
    frame = getattr(built, "frame", None)
    contract = getattr(built, "contract", None)
    open_u = list(frame.open_unknowns()) if frame else []
    allowed = list(getattr(contract, "allowed_files", []) or [])
    forbidden = list(getattr(contract, "forbidden_files", []) or [])
    file_lines = dict(getattr(built, "file_lines", {}) or {})
    secs = getattr(built, "sections", []) or []
    ctext = "\n\n".join(s.get("text", "") for s in secs if isinstance(s, dict))
    vr = verify_output(text, sections=secs, context_text=ctext)
    rctx = RuleContext(has_context=bool(getattr(built, "request", "")),
        allowed_files=allowed, forbidden_files=forbidden, known_files=set(),
        file_lines=file_lines,
        patches=[{"path": p.path, "operations": [
            {"type": o.type, "start_line": o.start_line,
             "end_line": o.end_line or o.start_line, "content": o.content}
            for o in p.operations]} for p in patches],
        prose=text, open_unknowns=open_u,
        write_attempted=False, write_confirmed=False)
    return payload, patches, run_rules(load_rules(), rctx), vr
