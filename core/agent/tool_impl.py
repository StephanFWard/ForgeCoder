"""Concrete tool implementations (kept small to fit edit limits)."""
from __future__ import annotations
from core.agent.tools import tool

@tool("view_file", "Read a workspace file with line numbers.")
def _view_file(ctx, path="", start=1, end=200):
    from pathlib import Path
    ws = getattr(ctx, "workspace", None)
    rel = path or getattr(ctx, "file", "") or ""
    if not ws or not rel:
        return {"ok": False, "error": "workspace and path required"}
    try:
        lines = (Path(ws) / rel).read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError as exc:
        return {"ok": False, "error": str(exc)}
    s = max(1, int(start or 1)); e = min(len(lines), int(end or len(lines)))
    body = "\n".join(f"{i+1}: {lines[i]}" for i in range(s-1, e))
    return {"ok": True, "path": rel, "start": s, "end": e,
            "total_lines": len(lines), "content": body}

@tool("search_repo", "Full-text repository search over the Forge index.")
async def _search_repo(ctx, query="", limit=8):
    idx = getattr(ctx, "_index", None) or getattr(ctx, "scratch", {}).get("index")
    if idx is None:
        return {"ok": False, "error": "no index bound to context"}
    from core.retrieval.search import SearchEngine
    try:
        q = query or ctx.effective_request()
        out = SearchEngine(idx).search(q, workspace=getattr(ctx, "workspace", None),
                                       limit=int(limit or 8))
        return {"ok": True, "results": [
            {"path": r.path, "content": r.content, "start_line": r.start_line,
             "end_line": r.end_line, "language": r.language} for r in out]}
    except Exception as exc:
        return {"ok": False, "error": str(exc)}

@tool("preview_patch", "Preview a structured patch in memory.")
def _preview_patch(ctx, patch):
    from core.patching.apply import preview_patch as _pp
    from core.patching.parser import FilePatch, Operation
    from forge_server.security import resolve_workspace_path
    try:
        ops = [Operation(type=o["type"], start_line=int(o["start_line"]),
                         end_line=o.get("end_line"), content=o.get("content", ""))
               for o in patch.get("operations", [])]
        fp = FilePatch(path=patch.get("path", ""), operations=ops)
        r = _pp(resolve_workspace_path(ctx.workspace, fp.path), fp)
        return {"ok": True, "path": fp.path, "diff": r.diff,
                "original": r.original, "proposed": r.proposed, "changed": r.changed}
    except Exception as exc:
        return {"ok": False, "error": str(exc)}

@tool("apply_patch", "Apply a previewed patch (needs confirmation).",
      needs_confirm=True)
def _apply_patch(ctx, patch, expected_original=None):
    from core.patching.apply import apply_patch as _ap
    from core.patching.parser import FilePatch, Operation
    from forge_server.security import resolve_workspace_path
    if not bool(getattr(ctx, "confirmed", False)):
        return {"ok": False, "error": "WRITE_FILE requires confirmation"}
    try:
        ops = [Operation(type=o["type"], start_line=int(o["start_line"]),
                         end_line=o.get("end_line"), content=o.get("content", ""))
               for o in patch.get("operations", [])]
        fp = FilePatch(path=patch.get("path", ""), operations=ops)
        full = resolve_workspace_path(ctx.workspace, fp.path)
        if expected_original is not None:
            if full.read_text(encoding="utf-8") != expected_original:
                return {"ok": False, "error": "File changed since preview."}
        r = _ap(full, fp)
        return {"ok": True, "applied": r.applied, "path": fp.path}
    except Exception as exc:
        return {"ok": False, "error": str(exc)}

@tool("run_tests", "Run workspace tests; returns a report.", needs_confirm=True)
def _run_tests(ctx):
    import subprocess
    from pathlib import Path
    ws = getattr(ctx, "workspace", None)
    if not ws:
        return {"ok": False, "error": "workspace required"}
    root = Path(ws)
    if (root / "pyproject.toml").exists() or (root / "tests").is_dir():
        cmd = ["python", "-m", "pytest", "-q", "--no-header"]
    elif (root / "package.json").exists():
        cmd = ["npm", "test", "--", "--silent"]
    else:
        return {"ok": True, "output": "No recognizable test runner."}
    try:
        p = subprocess.run(cmd, cwd=str(root), capture_output=True,
                           text=True, timeout=300, check=False)
        tail = ((p.stdout or "") + (p.stderr or ""))[-2500:]
        return {"ok": True, "exit": p.returncode, "output": f"exit={p.returncode}\n{tail}"}
    except Exception as exc:
        return {"ok": False, "error": str(exc)}

@tool("git_status", "Branch, remote, changed files (read-only).")
def _git_status(ctx):
    from core.git.status import git_status
    try:
        return {"ok": True, **git_status(getattr(ctx, "workspace", None) or "")}
    except Exception as exc:
        return {"ok": False, "error": str(exc)}

@tool("git_diff", "Unified diff of working tree (read-only).")
def _git_diff(ctx):
    from core.git.diff import git_diff
    try:
        return {"ok": True, "diff": git_diff(getattr(ctx, "workspace", None) or "")}
    except Exception as exc:
        return {"ok": False, "error": str(exc)}
