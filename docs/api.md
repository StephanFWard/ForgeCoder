# Forge API reference

Base URL: `http://127.0.0.1:8787` (loopback only by default).

All endpoints return JSON. Chat streams Server-Sent Events
(`text/event-stream`).

## Health

### `GET /health`

```json
{
  "status": "ok",
  "version": "0.1.0",
  "llama_server": { "ok": true, "url": "http://127.0.0.1:8080" },
  "database": { "ok": true, "path": "...", "files": 128, "chunks": 640, "symbols": 512 },
  "memory": { "available_gb": 7.2 }
}
```

`status` is `"ok"` when llama.cpp answers, `"degraded"` when it does not.

## Chat

### `POST /v1/chat`

Request:

```json
{
  "message": "Why does this method return null?",
  "workspace": "C:\\Projects\\MyApp",
  "file": "src/UserService.java",
  "selection": { "start": 42, "end": 57 },
  "history": [{ "role": "user", "content": "..." }, { "role": "assistant", "content": "..." }]
}
```

Response: SSE events:

```
data: {"type":"context","total_tokens":812,"sections":["repository"]}
data: {"type":"delta","content":"The repository lookup ..."}
data: {"type":"delta","content":" can return Optional.empty()."}
data: {"type":"done"}
```

History is clipped to fit the 4K chat budget. Retrieval + selection context
are assembled server-side.

## Completion (autocomplete)

### `POST /v1/completion`

```json
{
  "language": "java",
  "file": "src/UserService.java",
  "prefix": "public User findUser(Long id) {",
  "suffix": "\n}",
  "max_tokens": 64
}
```

Response:

```json
{ "completion": "return repository.findById(id)...", "language": "java", "file": "..." }
```

Defaults are conservative: `temperature 0.1`, `max_tokens 64`, stop on FIM/EOS
tokens. Uses Qwen FIM markers: `<|fim_prefix|>...<|fim_suffix|>...<|fim_middle|>`.

## Code actions

### `POST /v1/explain`, `POST /v1/tests`

```json
{ "code": "...", "file": "src/a.py", "workspace": "C:\\Proj" }
```

Returns `{ "ok": true, "explanation"|"tests": "..." }`.

### `POST /v1/edit`, `POST /v1/fix`

```json
{
  "code": "...",
  "error": "java.lang.NullPointerException at UserService:45",
  "instruction": "Add null validation",
  "workspace": "C:\\Projects\\MyApp"
}
```

Returns a structured patch or a fallback explanation:

```json
{
  "ok": true,
  "summary": "Added null validation.",
  "patches": [{
    "path": "src/UserService.java",
    "operations": [{ "type": "replace", "start_line": 42, "end_line": 47, "content": "..." }]
  }]
}
```

If the model does not emit valid JSON, the server returns
`{ "ok": false, "reason": "no_structured_patch", "proposal": "<model text>" }`.

`/v1/edit` and `/v1/fix` grammar-constrain the reply (`EDIT_SCHEMA` /
`FIX_SCHEMA` in `core/patching/contracts.py`, sent as llama.cpp's
`json_schema` response format), so the shape above is produced by the grammar
rather than trusted from prose. Every operation carries
`type`, `start_line`, `end_line` and `content`; an empty `files` list is the
model's explicit "not enough context to edit safely" answer. The schemas accept
exactly what `core/patching/parser.py` parses, and cross-field ordering
(`end_line >= start_line`, non-overlapping operations) is still enforced there.

## Retrieval

### `POST /v1/index`

```json
{ "workspace": "C:\\Projects\\MyApp" }
```

Incremental; returns `{ "ok": true, "added": 1, "updated": 0, "unchanged": 99,
"pruned": 0, "seconds": 0.4 }`.

### `POST /v1/search`

```json
{ "query": "authentication bug", "workspace": "C:\\Projects\\MyApp", "limit": 10 }
```

```json
{
  "ok": true,
  "results": [
    { "path": "src/auth/Login.java", "language": "java", "content": "...",
      "start_line": 12, "end_line": 45 }
  ]
}
```

**Weighted rerank** — add `"weighted": true` to blend the heuristic score with
System One relevance answers (free, offline by default; see
`docs/system-one.md`):

```json
{ "query": "authentication bug", "workspace": "C:\\Projects\\MyApp",
  "weighted": true, "system_one_weight": 0.35 }
```

```json
{
  "ok": true,
  "weighted": true,
  "backend": "deterministic",
  "results": [
    { "path": "src/auth/Login.java", "language": "java", "content": "...",
      "start_line": 12, "end_line": 45,
      "score": 120, "weighted_score": 0.78,
      "system_one": 0.83, "system_one_confidence": 0.61 }
  ]
}
```

`backend` names the decision backend that produced the weights. `score` keeps
the plain heuristic value; `weighted_score`, `system_one` and
`system_one_confidence` are the blended 0..1 score, the expected relevance
level, and the top probability. Ordering is by `weighted_score` with the
original order as tie-break, so results are stable across runs.

## Decisions (System One, free by default)

### `POST /v1/decide`

Answers typed questions about a state — the same request/response contract as
the hosted Jev API, served locally by free backends (see
`docs/system-one.md`):

```json
{
  "state": "the build failed with a null pointer in UserService:45",
  "questions": {
    "stage": {
      "type": "choice",
      "instructions": "Which pipeline stage failed?",
      "criteria": {"build": "compilation or packaging error",
                   "test": "a test assertion failed"}
    },
    "risky": {
      "type": "noul",
      "instructions": "Should this be treated as risky?",
      "criteria": {"true": "blocking rule fired", "false": "no blocking evidence"}
    }
  }
}
```

Response:

```json
{
  "ok": true,
  "model": "forge-system-one-deterministic",
  "backend": "deterministic",
  "free": true,
  "answers": {
    "stage": { "type": "choice", "choice": "build", "confidence": 0.71,
               "probabilities": {"build": 0.71, "test": 0.29} },
    "risky": { "type": "noul", "noul": 0.83 }
  },
  "usage": { "input_tokens": 88, "output_tokens": 0, "cost_usd": 0.0 }
}
```

Optional request fields: `"backend"` — `deterministic` (default), `local`
(free, uses the on-device model), `auto`, or `jev` (**paid**; additionally
needs `FORGECODER_SYSTEM_ONE_ALLOW_PAID=1` and a `JEV_API_KEY`, otherwise it
returns `{ "ok": false, "error": ... }` with the backend catalog).

### `GET /v1/decide/backends`

```json
{ "ok": true, "default": "deterministic", "allow_paid": false, "backends": [...] }
```

## Patches

### `POST /v1/patch/preview`

Applies operations **in memory only**. Returns `original`, `proposed`,
`diff` (unified):

```json
{
  "ok": true,
  "path": "src/UserService.java",
  "diff": "--- a/src/UserService.java\n+++ b/src/UserService.java\n...",
  "original": "...", "proposed": "...", "changed": true
}
```

### `POST /v1/patch/apply`

Same body plus the header `X-Forge-Confirm: true` (the VS Code extension
requests the user's confirmation first). Writes the file only when confirmed.

### `POST /v1/patch/from-model`

Parses raw model output into a validated `FilePatch` structure: body
`{ "payload": "<model text>" }` → `{ "ok": true, "patches": [...] }`.

## Errors

| Code | Meaning                                          |
| ---- | ------------------------------------------------ |
| 400  | Malformed request (Pydantic validation)          |
| 403  | Security rejection: non-loopback client or path escape |
| 422  | Schema validation failure                        |
| 500  | Unexpected server error                          |

## Config file

`%LOCALAPPDATA%\ForgeCoder\config.json`:

```json
{
  "host": "127.0.0.1",
  "port": 8787,
  "llama_url": "http://127.0.0.1:8080",
  "max_chat_context": 4096,
  "max_completion_context": 1024,
  "max_retrieved_chunks": 6,
  "completion_max_tokens": 64,
  "completion_temperature": 0.1
}
```

Environment overrides: `FORGECODER_<UPPERCASE_FIELD_NAME>` (e.g.
`FORGECODER_PORT`, `FORGECODER_LLAMA_URL`).
## Git endpoints (v0.2)

| Endpoint             | Confirm header | Purpose                                        |
| -------------------- | -------------- | ---------------------------------------------- |
| `POST /v1/git/status`   | -           | Branch, remote, changed files, recent commits  |
| `POST /v1/git/diff`     | -           | Staged + unstaged unified diff                 |
| `POST /v1/git/changes`  | -           | **View Changes**: diff reviewed by the model   |
| `POST /v1/git/commit`   | required    | Stage (`add_all`) and commit with a message    |
| `POST /v1/git/push`     | required    | Push current branch to origin                  |
| `POST /v1/git/pull`     | required    | Pull origin into the current branch            |

### `POST /v1/git/changes` — grounded review

The diff is split into labelled `staged` and `unstaged` sections; only added
(`+`) lines are offered as anchors, and their NEW-side line numbers come from
the `@@ -a,b +c,d @@` hunk header. Findings the model returns are rejected
unless they point at a supplied anchor, so a hallucinated line, an
unseen path, or a line number borrowed from the other coordinate system cannot
reach the UI:

```json
{
  "ok": true,
  "clean": false,
  "review": "Guarded the lookup.\n\n- [warning] unstaged src/service.py:3: ...",
  "review_status": "validated",
  "findings": [
    { "source": "unstaged", "path": "src/service.py", "line": 3,
      "severity": "warning", "message": "Defaulting to None can mask a bad key." }
  ],
  "diff": "..."
}
```

`review_status` is one of `validated` (findings passed anchor validation),
`invalid_output` (schema-invalid JSON or unsupported file/line references),
`insufficient_context` (no added lines to anchor to — e.g. only untracked
files) or `unavailable` (the local model is offline). The raw `diff` is always
returned, and the endpoint stays read-only: it never writes to the tree it is
reviewing.

## Plan -> Act events (v0.2)

| Endpoint          | Confirm header | Purpose                                                     |
| ----------------- | -------------- | ----------------------------------------------------------- |
| `POST /v1/plan`   | -              | Turn a request into `{summary, steps[]}` (search/explain/edit/test/commit) |
| `POST /v1/act`    | for test/commit | Execute exactly one step; result feeds the next; edit steps return a patch |

## Other (v0.2)

| Endpoint        | Purpose                                        |
| --------------- | ---------------------------------------------- |
| `POST /v1/ask`  | Non-streaming chat (MCP + simple clients)      |
