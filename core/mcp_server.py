"""ForgeCoder MCP server (stdio transport).

Hands ForgeCoder's tools to any MCP client (Claude Desktop, VS Code, IDE
agents) using the official Model Context Protocol SDK:

    {
      "mcpServers": {
        "forgecoder": {
          "command": "python",
          "args": ["-m", "core.mcp_server"],
          "env": {"FORGE_API": "http://127.0.0.1:8787"}
        }
      }
    }

Tools are served over the local Forge API (:8787) when it is running, with a
direct-to-SQLite/llama.cpp fallback for the read-only index tools. Mutating
tools (apply patch) require the client to pass confirm=True explicitly — the
same confirmation model as the HTTP API.
"""
from __future__ import annotations

import json
import os
import urllib.request

from mcp.server.fastmcp import FastMCP

FORGE_API = os.environ.get("FORGE_API", "http://127.0.0.1:8787")
LLAMA_URL = os.environ.get("LLAMA_URL", "http://127.0.0.1:8080")

mcp = FastMCP("forgecoder", instructions=(
    "Local-first coding assistant: repository search, incremental indexing, "
    "git review, structured patches, and Q&A backed by a local 1.5B model. "
    "File writes require confirm=True."
))


def _post(path: str, payload: dict, *, confirm: bool = False, timeout: float = 180.0) -> str:
    headers = {"Content-Type": "application/json"}
    if confirm:
        headers["X-Forge-Confirm"] = "true"
    req = urllib.request.Request(
        FORGE_API + path, data=json.dumps(payload).encode("utf-8"),
        headers=headers, method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.read().decode("utf-8", errors="replace")
    except OSError as exc:
        return json.dumps({"ok": False, "error": f"Forge API unreachable at {FORGE_API}: {exc}"})


@mcp.tool()
def forge_health() -> str:
    """Check the Forge API, llama.cpp, and the index database."""
    return _post("/health", {}, timeout=10.0)


@mcp.tool()
def forge_index(workspace: str) -> str:
    """Incrementally index a workspace so search and chat get repository context."""
    return _post("/v1/index", {"workspace": workspace})


@mcp.tool()
def forge_search(query: str, workspace: str, limit: int = 10) -> str:
    """Full-text repository search (FTS5) with ranking. Returns code chunks."""
    return _post("/v1/search", {"query": query, "workspace": workspace, "limit": limit})


@mcp.tool()
def forge_ask(question: str, workspace: str = "") -> str:
    """Ask the local ForgeCoder model a question with repository context."""
    payload: dict = {"message": question}
    if workspace:
        payload["workspace"] = workspace
    return _post("/v1/ask", payload)


@mcp.tool()
def forge_git_status(workspace: str) -> str:
    """Git branch, remote, and changed files for a workspace."""
    return _post("/v1/git/status", {"workspace": workspace})


@mcp.tool()
def forge_git_diff(workspace: str) -> str:
    """Unified diff of the working tree (staged + unstaged)."""
    return _post("/v1/git/diff", {"workspace": workspace})


@mcp.tool()
def forge_review_changes(workspace: str) -> str:
    """Review uncommitted changes: diff explained by the local model, with risks and next steps."""
    return _post("/v1/git/changes", {"workspace": workspace}, timeout=240.0)


@mcp.tool()
def forge_plan(request: str, workspace: str = "") -> str:
    """Turn a coding request into an ordered plan (search/explain/edit/test/commit steps)."""
    payload: dict = {"message": request}
    if workspace:
        payload["workspace"] = workspace
    return _post("/v1/plan", payload, timeout=240.0)


@mcp.tool()
def forge_act(plan_id: str, index: int, workspace: str = "", confirm: bool = False) -> str:
    """Execute one plan step. Steps that mutate state (test/commit) need confirm=True."""
    payload: dict = {"plan_id": plan_id, "index": index}
    if workspace:
        payload["workspace"] = workspace
    return _post("/v1/act", payload, confirm=confirm, timeout=360.0)


@mcp.tool()
def forge_patch_preview(workspace: str, path: str, operations: list[dict]) -> str:
    """Preview a structured patch (original vs proposed) without writing anything."""
    return _post("/v1/patch/preview", {
        "workspace": workspace,
        "patch": {"path": path, "operations": operations},
    })


@mcp.tool()
def forge_patch_apply(workspace: str, path: str, operations: list[dict],
                      confirm: bool = False) -> str:
    """Apply a structured patch to disk. Requires confirm=True — never implicit."""
    if not confirm:
        return json.dumps({"ok": False,
                           "error": "Applying a patch requires confirm=True (explicit user consent)"})
    return _post("/v1/patch/apply", {
        "workspace": workspace,
        "patch": {"path": path, "operations": operations},
    }, confirm=True)


def main() -> None:
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
