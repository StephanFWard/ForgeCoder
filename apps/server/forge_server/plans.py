"""Plan -> Act events.

A planning event turns a request into structured steps (planning); each act
event executes exactly one step and feeds its result forward (acting). Plans
live in server memory for the session — nothing is auto-executed without the
extension explicitly calling /v1/act, and mutating steps still require the
normal confirmation flow.
"""
from __future__ import annotations

import asyncio
import uuid

from fastapi import APIRouter, Depends, Header
from pydantic import BaseModel

from core.inference.chat import build_chat_messages
from core.patching.parser import PatchParseError, extract_json, parse_patch
from core.retrieval.budget import truncate_to_tokens
from core.retrieval.context import ContextBuilder, load_prompt
from forge_server.main import AppState, get_state
from forge_server.security import PERMISSIONS_AUTO, check_permission

router = APIRouter(prefix="/v1", tags=["plan"])

VALID_ACTIONS = {"search", "explain", "edit", "test", "commit"}


class PlanRequest(BaseModel):
    message: str
    workspace: str | None = None


class ActRequest(BaseModel):
    plan_id: str
    index: int
    workspace: str | None = None


class PlanStore:
    """Session-scoped plan memory (in-process, cleared on server restart)."""

    def __init__(self) -> None:
        self._plans: dict[str, dict] = {}

    def save(self, plan: dict) -> str:
        plan_id = uuid.uuid4().hex[:12]
        self._plans[plan_id] = plan
        # Keep memory bounded on the 6 GB target.
        while len(self._plans) > 20:
            self._plans.pop(next(iter(self._plans)))
        return plan_id

    def get(self, plan_id: str) -> dict | None:
        return self._plans.get(plan_id)

    def record(self, plan_id: str, index: int, result: dict) -> None:
        plan = self._plans.get(plan_id)
        if plan is not None:
            plan.setdefault("results", {})[str(index)] = result


def get_plan_store(state: AppState) -> PlanStore:
    if not hasattr(state, "plans"):
        state.plans = PlanStore()  # type: ignore[attr-defined]
    return state.plans  # type: ignore[attr-defined]


def _load_grammar(name: str) -> str | None:
    """Load a GBNF grammar file from runtime/grammars (None if missing)."""
    from pathlib import Path

    path = Path(__file__).resolve().parents[3] / "runtime" / "grammars" / f"{name}.gbnf"
    try:
        return path.read_text(encoding="utf-8")
    except OSError:
        return None


def _fallback_plan(message: str) -> dict:
    """No model / unparseable output: still return a usable plan."""
    return {
        "summary": "Default investigation plan (model output unavailable).",
        "steps": [
            {"title": "Locate relevant code", "action": "search", "detail": message[:200]},
            {"title": "Explain findings", "action": "explain", "detail": message[:200]},
        ],
    }


@router.post("/plan")
async def plan(req: PlanRequest, state: AppState = Depends(get_state)) -> dict:
    """Planning event: convert a request into executable steps."""
    check_permission("READ_FILE")
    builder = ContextBuilder(state.index)
    built = builder.build(req.message, workspace=req.workspace, budget=3072)

    context_text = "\n\n".join(s["text"] for s in built.sections) or "(no indexed context)"
    user = (
        f"Request: {req.message}\n\n"
        f"Repository context:\n{truncate_to_tokens(context_text, 1600)}"
    )
    try:
        raw = await state.inference.chat(
            build_chat_messages(load_prompt("plan"), user, None),
            temperature=0.1, max_tokens=600,
            grammar=_load_grammar("plan"),
        )
        data = extract_json(raw)
        steps = [
            {"title": str(s.get("title", f"Step {i + 1}"))[:120],
             "action": s.get("action") if s.get("action") in VALID_ACTIONS else "explain",
             "detail": str(s.get("detail", ""))[:500]}
            for i, s in enumerate(data.get("steps", []))
        ]
        summary = str(data.get("summary", ""))[:300]
        if not steps:
            raise ValueError("empty plan")
    except Exception:
        plan_data = _fallback_plan(req.message)
        plan_data["context_tokens"] = built.total_tokens
        pid = get_plan_store(state).save({**plan_data, "results": {}, "request": req.message})
        return {"ok": True, "plan_id": pid, **plan_data, "fallback": True}

    pid = get_plan_store(state).save(
        {"summary": summary, "steps": steps, "results": {}, "request": req.message}
    )
    return {"ok": True, "plan_id": pid, "summary": summary, "steps": steps}

@router.post("/act")
async def act(req: ActRequest, confirm: str | None = Header(default=None, alias="X-Forge-Confirm"),
              state: AppState = Depends(get_state)) -> dict:
    """Acting event: execute exactly one plan step and record its result."""
    plan = get_plan_store(state).get(req.plan_id)
    if plan is None:
        return {"ok": False, "error": "Unknown plan_id (plans are session-scoped)"}
    steps = plan.get("steps", [])
    if not (0 <= req.index < len(steps)):
        return {"ok": False, "error": f"Step index {req.index} out of range"}
    step = steps[req.index]
    action = step["action"]
    detail = step["detail"] or plan.get("request", "")

    result: dict = {"action": action, "title": step["title"]}
    try:
        if action == "search":
            check_permission("SEARCH")
            rows = state.index.search(
                detail, limit=8,
                workspace=req.workspace.replace("\\", "/") if req.workspace else None,
            )
            result["output"] = "\n\n".join(
                f"{r['path']}:{r['start_line']}-{r['end_line']}\n{r['content'][:300]}"
                for r in rows[:8]
            ) or "No matches found."

        elif action == "explain":
            result["output"] = await _ask(state, req.workspace, detail, plan.get("results", {}))

        elif action == "edit":
            check_permission("READ_FILE")
            patch = await _edit_step(state, req.workspace, detail, plan.get("results", {}))
            if patch:
                result["patch"] = patch
                result["output"] = f"Patch prepared for {patch['path']} — preview it before applying."
            else:
                result["output"] = "The model did not produce a valid patch."

        elif action == "test":
            if confirm != "true":
                return {"ok": False, "error": "Running tests requires confirmation (X-Forge-Confirm: true)"}
            check_permission("TEST", granted=PERMISSIONS_AUTO | {"TEST"})
            result["output"] = await asyncio.to_thread(_run_tests, req.workspace or ".")

        elif action == "commit":
            if confirm != "true":
                return {"ok": False, "error": "Committing requires confirmation (X-Forge-Confirm: true)"}
            check_permission("GIT_WRITE", granted=PERMISSIONS_AUTO | {"GIT_WRITE"})
            from core.git import operations as git_ops
            from core.git.util import is_git_repo
            if not req.workspace or not is_git_repo(req.workspace):
                result["output"] = "No git workspace — commit skipped."
            else:
                message = detail or f"ForgeCoder: {plan.get('request', 'update')[:60]}"
                staged = git_ops.stage(req.workspace)
                if staged == 0:
                    result["output"] = "Nothing to commit — working tree clean."
                else:
                    info = git_ops.commit(req.workspace, message)
                    result["output"] = f"Committed {info['hash']}: {info['subject']}"

    except Exception as exc:
        result["ok"] = False
        result["output"] = f"Step failed: {exc}"
        get_plan_store(state).record(req.plan_id, req.index, result)
        return {"ok": True, "step_index": req.index, **result,
                "next_index": req.index + 1 if req.index + 1 < len(steps) else None}

    result["ok"] = True
    get_plan_store(state).record(req.plan_id, req.index, result)
    return {"ok": True, "step_index": req.index, **result,
            "next_index": req.index + 1 if req.index + 1 < len(steps) else None}


async def _ask(state: AppState, workspace: str | None, question: str,
               prior_results: dict) -> str:
    """Non-streaming chat with retrieval context + prior step results."""
    builder = ContextBuilder(state.index)
    built = builder.build(question, workspace=workspace, budget=3072)

    prior = "\n".join(
        f"- {v.get('title', '?')}: {truncate_to_tokens(str(v.get('output', '')), 250)}"
        for v in prior_results.values() if isinstance(v, dict)
    )
    context_text = "\n\n".join(s["text"] for s in built.sections)
    user = question
    if context_text:
        user += f"\n\nRepository context:\n{truncate_to_tokens(context_text, 1800)}"
    if prior:
        user += f"\n\nEarlier step results:\n{prior}"

    return await state.inference.chat(
        build_chat_messages(load_prompt("chat"), user, None),
        temperature=0.3, max_tokens=700,
        presence_penalty=0.2, frequency_penalty=0.2,
    )


async def _edit_step(state: AppState, workspace: str | None, instruction: str,
                     prior_results: dict) -> dict | None:
    """Produce a structured patch for an edit step (preview only, never applies)."""
    builder = ContextBuilder(state.index)
    built = builder.build(instruction, workspace=workspace, budget=3072)
    context_text = "\n\n".join(s["text"] for s in built.sections) or "(no context)"
    prior = "\n".join(
        f"- {v.get('title', '?')}: {truncate_to_tokens(str(v.get('output', '')), 200)}"
        for v in prior_results.values() if isinstance(v, dict)
    )
    user = (
        f"{instruction}\n\nProduce a patch JSON (files -> operations with "
        f"type/start_line/end_line/content) using 1-based line numbers.\n\n"
        f"Repository context:\n{truncate_to_tokens(context_text, 2000)}\n"
    )
    if prior:
        user += f"\nEarlier step results:\n{prior}"
    raw = await state.inference.chat(
        build_chat_messages(load_prompt("edit"), user, None),
        temperature=0.1, max_tokens=900,
    )
    try:
        patches = parse_patch(raw)
    except PatchParseError:
        return None
    if not patches:
        return None
    return patches[0].to_dict()


def _run_tests(workspace: str) -> str:
    """Run the default test command for the workspace (pytest, then npm test)."""
    import subprocess
    from pathlib import Path

    root = Path(workspace)
    if (root / "pyproject.toml").exists() or (root / "pytest.ini").exists() or (root / "tests").is_dir():
        cmd = ["python", "-m", "pytest", "-q", "--no-header"]
    elif (root / "package.json").exists():
        cmd = ["npm", "test", "--", "--silent"]
    else:
        return "No recognizable test runner (pytest/npm) in this workspace."
    try:
        proc = subprocess.run(
            cmd, cwd=str(root), capture_output=True, text=True,
            encoding="utf-8", errors="replace", timeout=300, check=False,
        )
    except subprocess.TimeoutExpired:
        return "Tests timed out after 300s."
    tail = ((proc.stdout or "") + (proc.stderr or ""))[-2500:]
    return f"exit={proc.returncode}\n{tail}"
