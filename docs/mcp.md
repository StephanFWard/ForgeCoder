# MCP server (Model Context Protocol)

ForgeCoder can hand its tools to any MCP client — Claude Desktop, VS Code
agent mode, or any other MCP host — over stdio.

## Install

```powershell
pip install -e ".[mcp]"
```

## Client configuration

```json
{
  "mcpServers": {
    "forgecoder": {
      "command": "python",
      "args": ["-m", "core.mcp_server"],
      "env": { "FORGE_API": "http://127.0.0.1:8787" }
    }
  }
}
```

Run the Forge stack first (`python runtime/scripts/start_forge.py`); the MCP
server talks to it over loopback HTTP.

## Tools

| Tool                   | Mutating | Purpose                                          |
| ---------------------- | -------- | ------------------------------------------------ |
| `forge_health`         | no       | API + llama.cpp + database status                 |
| `forge_index`          | no       | Incrementally index a workspace                   |
| `forge_search`         | no       | FTS5 repository search                            |
| `forge_ask`            | no       | Question answered with repository context         |
| `forge_git_status`     | no       | Branch, remote, changed files                     |
| `forge_git_diff`       | no       | Working-tree diff                                 |
| `forge_review_changes` | no       | Model review of uncommitted changes               |
| `forge_plan`           | no       | Plan -> Act: structured steps for a request       |
| `forge_act`            | confirm  | Execute one plan step (test/commit need confirm)  |
| `forge_patch_preview`  | no       | Original vs proposed, in memory only              |
| `forge_patch_apply`    | confirm  | Write a patch to disk (`confirm=True` required)   |

Mutating tools refuse to run unless the client passes `confirm: true`
explicitly — the same consent model as the HTTP API's `X-Forge-Confirm`
header. The server binds to nothing (stdio), so there is no network surface.
