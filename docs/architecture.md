# Architecture

ForgeCoder is a **thin local platform** around a 1.5B coder model. The central
design decision: repository intelligence lives in the *tooling*, not the
model. The model is a replaceable consumable.

## Layers

```
┌────────────────────────────────────────────────────┐
│ consumers: VS Code extension   CLI (forge-index)  │
├────────────────────────────────────────────────────┤
│ Forge Local API  :8787  (apps/server/forge_server)│
│   chat · completion · actions · patches · retrieval│
├────────────────┬───────────────┬──────────────────┤
│ Index          │ Retrieval     │ Git              │
│ core/indexer   │ core/retrieval│ core/git         │
├────────────────┴───────────────┴──────────────────┤
│ Context Builder — 4K-token budget                  │
├────────────────────────────────────────────────────┤
│ llama.cpp :8080  (OpenAI-compatible HTTP API)      │
├────────────────────────────────────────────────────┤
│ ForgeCoder 1.5B Q4_K_M GGUF                        │
└────────────────────────────────────────────────────┘
```

The boundaries are load-bearing: any layer can be replaced without touching
the others (e.g. swap llama.cpp for another OpenAI-compatible inference
engine, or C++-upgrade the indexer).

## Modules

- `forge_server` (FastAPI) — thin HTTP service. Enforces loopback-only,
  resolves `%LOCALAPPDATA%\ForgeCoder\config.json`, owns the SQLite index
  handle and the inference client.
- `core/indexer` — filesystem walk (ignore rules), sha1 change detection,
  semantic chunking, regex symbol extraction, and SQLite + FTS5 persistence.
- `core/retrieval` — FTS5 candidates → deterministic ranking → context
  assembly under a token budget.
- `core/inference` — async client for `/v1/chat/completions` and
  `/v1/completions` including SSE streaming and Qwen FIM tokens.
- `core/patching` — structured JSON patches (parse → validate → preview →
  apply), always diff-first.
- `core/git` — read-only helpers (status / diff / log).
- `core/memory` — RAM/CPU/VRAM detection and throttling decisions.
- `training/` — dataset prep, QLoRA training, eval, merge, GGUF conversion.
- `apps/vscode` — extension (chat WebView, inline completion, code actions).

## Request flow — chat

```
VS Code webview
   │  POST /v1/chat {message, workspace, file, selection}
   ▼
Forge server
   │  ContextBuilder:
   │    retrieve(query) → FTS5 → rank → top-6 → fit to budget
   │    + selection snippet + system prompt (runtime/prompts)
   ▼
llama.cpp /v1/chat/completions (stream=true)
   │  SSE deltas
   ▼
webview streams tokens
```

## Request flow — edit

```
selection → POST /v1/edit
   │
   ▼
model (grammar: runtime/grammars/patch.gbnf)
   │  {summary, files:[{path, operations:[{type,start_line,end_line,content}]}]}
   ▼
forge_server/actions parse + validate
   ▼
extension → POST /v1/patch/preview  (diff, no writes)
   ▼
user confirms → POST /v1/patch/apply (X-Forge-Confirm: true)
```

## Indexing algorithm

```
workspace open
  → walk (ignore node_modules/.git/dist/build/target/.venv/binaries/generated)
  → sha1(file) vs stored hash → skip unchanged
  → chunk (semantic boundaries) → extract symbols → upsert SQLite + FTS5
  → prune deleted paths
```

Never loads the whole repository into memory; only changed files are
re-chunked.

## Versioning

0.x milestones:

| Version | Milestone                                   |
| ------- | ------------------------------------------- |
| 0.1     | Runtime: llama.cpp + model + CLI + chat     |
| 0.2     | SQLite + repository search                  |
| 0.3     | VS Code chat (streaming)                    |
| 0.4     | Editing + structured patches + diff viewer  |
| 0.5     | Inline completion (FIM)                     |
| 0.6     | Git + testing                               |
| 0.7     | ForgeCoder LoRA specialization              |
| 1.0     | Installer + all components + 6 GB tuning    |

## Security posture

- Loopback-only binds (127.0.0.1). Rejected: `0.0.0.0`.
- No telemetry, no cloud, no auto-uploads, no shell execution.
- File writes require explicit user confirmation after a diff preview.
- Permissions: `READ_FILE`/`SEARCH`/`GIT_READ` auto; `WRITE_FILE`/`BUILD`/
  `TEST`/`SHELL` always confirmed.