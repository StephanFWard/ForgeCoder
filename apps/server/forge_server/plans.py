"""Plan -> Act events.

A planning event turns a request into structured steps (planning); each act
event executes exactly one step and feeds its result forward (acting). Plans
live in server memory for the session — nothing is auto-executed without the
extension explicitly calling /v1/act, and mutating steps still require the
normal confirmation flow.
"""
from __future__ import annotations

import asyncio
import re
import uuid

from fastapi import APIRouter, Depends, Header
from pydantic import BaseModel

from core.inference.chat import build_chat_messages
from core.inference.verify import verify_output
from core.patching.parser import (
    MultiPatch,
    PatchParseError,
    extract_json,
    parse_multi,
    parse_patch,
)
from core.retrieval.budget import truncate_to_tokens
from core.retrieval.context import load_prompt
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


# JSON schemas for grammar-constrained generation. llama.cpp converts these to
# GBNF server-side (response_format json_schema), which the GBNF text files
# cannot achieve through the HTTP API — and the constraint also guarantees
# valid JSON string escaping (newlines in file content are escaped, not raw).
_CREATE_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "message": {"type": "string"},
        "creates": {"type": "object", "additionalProperties": {"type": "string"}},
    },
    "required": ["message", "creates"],
}

_FIX_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "summary": {"type": "string"},
        "diagnosis": {"type": "string"},
        "files": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "operations": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "type": {"type": "string", "enum": ["insert", "replace", "delete"]},
                                "start_line": {"type": "integer"},
                                "end_line": {"type": "integer"},
                                "content": {"type": "string"},
                            },
                            "required": ["type", "start_line", "end_line", "content"],
                        },
                    },
                },
                "required": ["path", "operations"],
            },
        },
    },
    "required": ["summary", "diagnosis", "files"],
}

_PLAN_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "summary": {"type": "string"},
        "steps": {
            "type": "array",
            "minItems": 1,
            "maxItems": 6,
            "items": {
                "type": "object",
                "properties": {
                    "title": {"type": "string"},
                    "action": {"type": "string", "enum": sorted(VALID_ACTIONS)},
                    "detail": {"type": "string"},
                },
                "required": ["title", "action", "detail"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["summary", "steps"],
    "additionalProperties": False,
}


def _schema_for(kind: str) -> dict | None:
    return {"create": _CREATE_SCHEMA, "fix": _FIX_SCHEMA, "plan": _PLAN_SCHEMA}.get(kind)


def _fallback_plan(message: str) -> dict:
    """No model / unparseable output: still return a usable plan."""
    if _is_creation_request(message):
        files = _creation_files(message)
        steps = [
            {"title": "Survey workspace", "action": "search",
             "detail": "Check whether any of " + ", ".join(files) + " already exist"},
        ]
        for fname in files:
            steps.append({
                "title": "Create " + fname, "action": "edit",
                "detail": "Create file: " + fname + " -- complete working content for: " + message[:160],
            })
        steps.append({
            "title": "Smoke-test created files", "action": "test",
            "detail": "Byte-compile every created Python file (syntax check, no side effects)",
        })
        return {"summary": message[:80], "steps": steps}
    return {
        "summary": "Default investigation plan (model output unavailable).",
        "steps": [
            {"title": "Locate relevant code", "action": "search", "detail": message[:200]},
            {"title": "Explain findings", "action": "explain", "detail": message[:200]},
        ],
    }


def _creation_files(message: str) -> list:
    """Filenames a creation request most likely needs, cheapest heuristic first."""
    text = message.lower()
    m = re.search(r"([\w.-]+\.(py|html|js|ts|java|go|rs|cs|cpp|sql))", text)
    if m:
        named = m.group(1)
        return [named, "README.md"] if named != "README.md" else [named]
    if "fps" in text or "first person" in text or "shooter" in text:
        return ["game.py", "requirements.txt", "README.md"]
    if "api" in text or "endpoint" in text or "server" in text:
        return ["app.py", "requirements.txt", "README.md"]
    if "html" in text or "web" in text or "tic" in text or "browser" in text:
        return ["index.html", "README.md"]
    return ["main.py", "README.md"]


_CREATION_RE = None  # compiled lazily; pattern lives in _is_creation_request


def _is_creation_request(instruction: str) -> bool:
    """Heuristic: does this instruction ask to *create a new file*?

    Mirrors the fallback when the workspace has no indexed context — a 1.5B
    model cannot "edit" a file it was never shown, so generation (creates)
    is the only applicable behavior.
    """
    global _CREATION_RE
    if _CREATION_RE is None:
        _CREATION_RE = re.compile(
            r"\b(create|write|generate|build|make|scaffold|bootstrap)\b"
            r"[^.\n]{0,60}\b(file|page|html|script|module|component|class|"
            r"app|game|repository|repo|project|site|api|endpoint|test)\b"
            r"|\bgame\b|\bapp\b|\bscript\b|\bprogram\b|\btic\b|\bfps\b"
            r"|\bshooter\b|\bfirst\s+person\b|\brepository\b|\bproject\b"
            r"|\bfrom\s+scratch\b|\bnew\s+(file|app|game|script|program|project|repo)\b",
            re.IGNORECASE,
        )
    return bool(_CREATION_RE.search(instruction))


_SINGLE_FILE_RE = re.compile(r"Create file:\s*([\w][\w.\-/]*[A-Za-z0-9])", re.IGNORECASE)


def _single_creation_target(instruction: str, plan_request: str) -> str | None:
    """The one file this edit step must create (per-file plan steps)."""
    m = _SINGLE_FILE_RE.search(instruction or "")
    if m:
        return m.group(1).strip().strip("'\"").lstrip("./")
    m = _SINGLE_FILE_RE.search(plan_request or "")
    if m:
        return m.group(1).strip().strip("'\"").lstrip("./")
    return None


def _multi_to_act(multi: MultiPatch) -> dict:
    """Shape a MultiPatch for the act result / VS Code extension."""
    return {
        "kind": "multi",
        "files": [fp.to_dict() for fp in multi.patches],
        "creates": dict(multi.creates),
        "message": multi.message,
    }


_PLACEHOLDER_RE = None  # compiled lazily


def _looks_placeholder(content: str) -> bool:
    """True when created-file content looks like a stub rather than working code."""
    global _PLACEHOLDER_RE
    if _PLACEHOLDER_RE is None:
        import re

        _PLACEHOLDER_RE = re.compile(
            r"(complete code here|TODO|FIXME|<placeholder|\.\.\.\s*$|"
            r"not implemented|pass\s*#\s*implement)",
            re.IGNORECASE,
        )
    stripped = content.strip()
    # A real, useful file is rarely this short
    if len(stripped) < 40:
        return True
    return bool(_PLACEHOLDER_RE.search(stripped))


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
            schema=_schema_for("plan"),
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
            if rows:
                result["output"] = "\n\n".join(
                    f"{r['path']}:{r['start_line']}-{r['end_line']}\n{r['content'][:300]}"
                    for r in rows[:8]
                )
            elif req.workspace and _is_creation_request(detail + " " + plan.get("request", "")):
                result["output"] = (
                    "Workspace has no indexed files matching this request yet -- "
                    "this is a new-file creation task, so the following edit steps "
                    "will generate each file from scratch."
                )
            else:
                result["output"] = "No matches found."

        elif action == "explain":
            if _is_creation_request(detail + " " + plan.get("request", "")):
                files = _creation_files(plan.get("request", "") + " " + detail)
                result["output"] = (
                    "Plan: create " + ", ".join(files) + " in the workspace. "
                    "Each file is generated by its own edit step (Run each step below), "
                    "then smoke-tested before applying."
                )
            else:
                result["output"] = await _ask(state, req.workspace, detail, plan.get("results", {}))
            # RNP predictive verification: check the answer's citations against
            # the context that was actually supplied to the model.
            result["verification"] = verify_output(
                result["output"], context_text=context_text or None,
            ).to_dict()

        elif action == "edit":
            check_permission("READ_FILE")
            prepared = await _edit_step(
                state, req.workspace, detail, plan.get("results", {}),
                plan_request=plan.get("request", ""),
            )
            if prepared is None:
                result["output"] = "The model did not produce a valid patch."
            elif prepared.get("kind") == "single":
                result["patch"] = prepared["patch"]
                result["output"] = (
                    f"Patch prepared for {prepared['patch']['path']} — preview it before applying."
                )
            else:  # multi: existing-file edits + brand-new file creations
                result["multiPatch"] = prepared["files"]
                result["creates"] = prepared["creates"]
                created = list(prepared["creates"])
                edited = [f["path"] for f in prepared["files"]]
                parts = []
                if edited:
                    parts.append(f"edit {', '.join(edited)}")
                if created:
                    parts.append(f"create {', '.join(created)}")
                result["output"] = (
                    f"Change set prepared to {' and '.join(parts)} — preview it before applying."
                )
                stubs = [p for p, c in prepared["creates"].items() if _looks_placeholder(c)]
                if stubs:
                    result["warning"] = (
                        f"Generated content for {', '.join(stubs)} looks like a "
                        f"stub/placeholder — re-run the step or edit before applying."
                    )

        elif action == "test":
            check_permission("TEST", granted=PERMISSIONS_AUTO | {"TEST"})
            smoke = _syntax_smoke_check(req.workspace or ".")
            if smoke is not None:
                result["output"] = smoke
            else:
                if confirm != "true":
                    return {"ok": False, "error": "Running tests requires confirmation (X-Forge-Confirm: true)"}
                result["output"] = await asyncio.to_thread(_run_tests, req.workspace or ".")

        elif action == "commit":
            if confirm != "true":
                return {"ok": False, "error": "Committing requires confirmation (X-Forge-Confirm: true)"}
            check_permission("GIT_WRITE", granted=PERMISSIONS_AUTO | {"GIT_WRITE"})
            from core.git import operations as git_ops
            from core.git.util import is_git_repo
            if not req.workspace:
                result["output"] = "No workspace selected -- commit skipped."
            elif not is_git_repo(req.workspace):
                if confirm == "true":
                    import subprocess as _sp
                    from pathlib import Path as _P
                    try:
                        _sp.run(["git", "init", "-q"], cwd=str(_P(req.workspace)),
                                check=True, timeout=30)
                        result["output"] = (
                            "Initialized a new git repository in the workspace. "
                            "Re-run this step (with confirmation) to stage and commit."
                        )
                    except Exception as exc:
                        result["output"] = f"git init failed: {exc}"
                else:
                    result["output"] = (
                        "Workspace is not a git repository yet. "
                        "Re-run this step WITH confirmation to run 'git init' here, "
                        "then Run once more to stage and commit."
                    )
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
                     prior_results: dict, plan_request: str = "") -> dict | None:
    """Produce a structured change set for an edit step (preview only, never applies).

    Two behaviors, selected automatically:

    * **Existing-file edit** — grammar-constrained patch JSON
      (``files -> operations``), applied to the file the model was shown.
    * **New-file creation** — when the instruction asks to *create* something
      (or the workspace has no indexed context), grammar-constrained
      ``{"message", "creates": {relpath: full content}}`` output parsed by
      :func:`parse_multi`.  This is what makes requests like "create a
      tic-tac-toe page" work instead of failing with "no valid patch".

    Returns ``{"kind": "single", "patch": ...}``, ``{"kind": "multi", ...}``,
    or ``None`` when the model produced nothing parseable.
    """
    builder = ContextBuilder(state.index)
    built = builder.build(instruction, workspace=workspace, budget=3072)
    context_text = "\n\n".join(s["text"] for s in built.sections)
    prior = "\n".join(
        f"- {v.get('title', '?')}: {truncate_to_tokens(str(v.get('output', '')), 200)}"
        for v in prior_results.values() if isinstance(v, dict)
    )

    creation = (
        _is_creation_request(instruction)
        or _is_creation_request(plan_request)
        or (not context_text and _is_creation_request(f"create file: {instruction} {plan_request}"))
    )

    if creation:
        single = _single_creation_target(instruction, plan_request)
        goal = instruction if (plan_request and plan_request in instruction) else (
            f"{instruction}\n(Original request: {plan_request})"
            if plan_request else instruction
        )
        if single:
            goal = (
                f"Create ONLY the file {single!r}.\n{goal}\n"
                f"The creates object must contain EXACTLY ONE key: {single!r} "
                f"with the COMPLETE, working, self-contained file content."
            )
        user = (
            f"{goal}\n\nCreate the requested file(s) now. Respond ONLY with JSON of the "
            f"shape {{\"message\": \"...\", \"creates\": {{\"FILENAME\": \"complete file "
            f"content\"}}}}. Rules:\n"
            f"1. FILENAME must be the real file the request needs — e.g. \"index.html\" "
            f"for a web page, \"main.py\" for a Python program, \"README.md\" for docs.\n"
            f"2. The content value must be the COMPLETE, working file — every line, "
            f"never placeholders like \"...\", \"TODO\" or \"Complete code here\".\n"
            f"3. Do not copy \"rel/path.ext\" or any example placeholder from this prompt.\n"
        )
        schema = _schema_for("create")
        max_tokens = 3000  # a whole game file needs room
    else:
        user = (
            f"{instruction}\n\nProduce a patch JSON (files -> operations with "
            f"type/start_line/end_line/content) using 1-based line numbers.\n\n"
            f"Repository context:\n{truncate_to_tokens(context_text or '(no context)', 2000)}\n"
        )
        schema = _schema_for("fix")
        max_tokens = 900
    if prior:
        user += f"\nEarlier step results:\n{prior}"

    raw = await state.inference.chat(
        build_chat_messages(load_prompt("edit"), user, None),
        temperature=0.1, max_tokens=max_tokens, schema=schema,
    )

    try:
        multi = parse_multi(raw)
    except PatchParseError:
        # Legacy fallback: single-file patch shape
        try:
            patches = parse_patch(raw)
        except PatchParseError:
            return None
        if not patches:
            return None
        return {"kind": "single", "patch": patches[0].to_dict()}

    if not multi.patches and not multi.creates:
        # Model ignored the creates shape (grammar not enforced) — try patch shape
        try:
            patches = parse_patch(raw)
        except PatchParseError:
            return None
        if not patches:
            return None
        return {"kind": "single", "patch": patches[0].to_dict()}

    return {"kind": "multi", **_multi_to_act(multi)}


def _syntax_smoke_check(workspace: str) -> str | None:
    """Read-only check: byte-compile created Python files for syntax errors.

    Returns the report, or None when there is nothing to smoke-test so the
    caller falls through to the real test runner. Never executes code and
    never needs confirmation.
    """
    import py_compile
    from pathlib import Path as _P

    root = _P(workspace)
    if not root.is_dir():
        return None
    targets = sorted(root.rglob("*.py"))
    if not targets:
        return None
    ok, bad = 0, []
    for f in targets[:50]:
        try:
            py_compile.compile(str(f), doraise=True)
            ok += 1
        except py_compile.PyCompileError as exc:
            bad.append(f"{f.name}: {exc}")
    if bad:
        return "SMOKE-TEST FAILED (syntax errors):\n" + "\n".join(bad)
    return f"SMOKE-TEST PASSED: {ok} Python file(s) compile cleanly (syntax check, not executed)."


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
