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

## Plan -> Act events (v0.2)

| Endpoint          | Confirm header | Purpose                                                     |
| ----------------- | -------------- | ----------------------------------------------------------- |
| `POST /v1/plan`   | -              | Turn a request into `{summary, steps[]}` (search/explain/edit/test/commit) |
| `POST /v1/act`    | for test/commit | Execute exactly one step; result feeds the next; edit steps return a patch |

## Other (v0.2)

| Endpoint        | Purpose                                        |
| --------------- | ---------------------------------------------- |
| `POST /v1/ask`  | Non-streaming chat (MCP + simple clients)      |
