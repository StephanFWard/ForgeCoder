# Changelog

All notable changes to ForgeCoder are documented here. Format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project
adheres to **0.x** versioning until 1.0.

## [Unreleased]

### Added — System One decision layer (Jev-shaped, free by default)

- New `core/system_one/` package: typed `choice` / `score` / `noul` questions
  answered against a state with probability distributions and a confidence
  (TypeSafe AI's Jev "System One" contract), served by **free backends**:
  `deterministic` (default — IDF-weighted lexical softmax, offline, byte-for-byte
  reproducible, no model/network/key) and `local` (free on-device llama.cpp
  verdicts blended into the same shape, degrading to deterministic when the
  server is down). The billed hosted Jev API exists as `jev` but is opt-in only:
  it requires `FORGECODER_SYSTEM_ONE_ALLOW_PAID=1` **and** a `JEV_API_KEY`.
- `Decider` facade + weighted composition helpers (`answer_value`, `combine`,
  `gate`, `top_choice`, `expected_score`) — weights live in the caller's code,
  so a priority change is a coefficient change, not a prompt change.
- `jev_compat`: the upstream `jev` decorator API (`@fn`, `decide`, `BaseModel`)
  re-implemented for Python 3.10+, because PyPI `jev` 0.3.0 requires
  Python ≥ 3.14 and a paid key; `pip install -e ".[jev]"` installs the real
  wheel only on 3.14+ (environment marker) and is safe on every supported Python.
- `POST /v1/decide` and `GET /v1/decide/backends` (`apps/server/forge_server/decide.py`):
  Jev-shaped request/response plus ForgeCoder `backend`/`free` metadata; local
  usage always reports `cost_usd: 0.0`.
- Weighted retrieval rerank (`core/system_one/rerank.py`, wired into
  `POST /v1/search` via `"weighted": true`): one batched `score` question per
  candidate, final order = `(1-w)·heuristic + w·expected_level` (default
  `w = 0.35`), each row carrying `weighted_score`, `system_one` and
  `system_one_confidence`.
- `forge-decide` CLI entry point (`core/system_one/cli.py`): one decision from
  stdin/file, `--backends` catalog, `--allow-paid` guard for the billed backend.
- Local secret integration: `.env.local` (gitignored; `.env.example` documents
  the names) supplies `JEV_API_KEY` / `JEV_API_URL` / `JEV_MODEL` to the paid
  backend without touching any committed file; a real environment variable
  always wins.
- Paid-backend endpoint selection by key prefix: `jv_live_...` keys target the
  managed wrapper (`jevtypesafeai.com/api/v1/decide`), any other key targets the
  official TypeSafe Decision API (`api.typesafe.ai/v1/systemone`), which requires
  a `model` field — sent as `JEV_MODEL` or the default `jev-latest`.
  `ScoreAnswer` now accepts the official payload's `legend` field. Live-verified
  end to end (async + sync paths) against the official API.
- Packaging fix: setuptools `package-dir`/`packages.find` now map **both** source
  roots (`apps/server` → `forge_server`, repo root → `core`), so installed entry
  points (`forge-server`, `forge-index`, `forge-mcp`, `forge-decide`) work outside
  the repository directory; previously `core.*` scripts failed with
  `No module named 'core'`.
- Unit tests (`tests/unit/test_system_one.py`) and integration tests
  (`tests/integration/test_server_api.py` decide endpoints); docs in
  `docs/system-one.md`.

### Fixed — plain chat creates files instead of echoing its own frame

- `POST /v1/chat` routes creation requests ("make the game snake in html")
  through the creation pipeline (`_edit_step`, with its corrective retry):
  the reply streams a summary plus a new `patch` SSE event carrying
  `{patches, creates, message}`, which the sidebar renders through the
  standard View Diff / Apply bar (`apps/vscode/src/chat/SidebarProvider.ts`).
  Previously plain chat had no creation routing at all, so the request was
  answered with prose echoes of the task frame.
- The task frame no longer teaches the model to fabricate warnings:
  - the rules header said "warn = tell the user", and the model obeyed —
    emitting fabricated `[warn] ask-dont-guess:` lines (even quoting the
    informational "Evidence status" line as an unresolved question). The
    header now forbids printing rule slugs, severities, or `[warn]`/`[block]`
    markers (`core/agent/prompt.py`).
  - frame unknowns were rendered as literal questions ("which file should
    the change land in?") that came back verbatim; they are now imperatives
    addressed to the model, with an explicit do-not-echo directive
    (`core/agent/frame.py`).
  - `ask-dont-guess` (`core/agent/rules.py`, `runtime/prompts/agent-rules.md`)
    resolves what is derivable from the request and asks only what is truly
    undecidable.
- Creation task frames carry an explicit note ("Creation task: generate
  complete new files (index.html, README.md). Do not modify existing files
  and do not ask which file.") so the model never re-derives the target.

### Fixed — creation requests are no longer interrogated as edits

- Task frames and scope contracts are now creation-aware
  (`core/agent/creation.py`, `core/agent/frame.py`, `core/agent/scope.py`,
  `core/retrieval/context.py`). A request like "make the game snake in html"
  used to get two edit-oriented unknowns — "which file should the change land
  in?" and "no acceptance command was stated; name the command that proves the
  change" — injected into its prompt, and the model answered with
  ``[warn] ask-dont-guess`` instead of a game. Now the frame bounds the task
  to the derived files to create (`index.html`, `README.md`), the acceptance
  evidence is the Forge smoke check of those files, and the unknowns list
  stays empty. Edit requests keep their unknowns unchanged.
- `plans.py` creation heuristics (`_is_creation_request`, `_creation_files`,
  `_single_creation_target`) now delegate to `core.agent.creation` — one
  source of truth shared with the frame layer instead of two drifting regexes.
- The plan smoke-test step now validates HTML pages (doctype/html structure,
  no placeholder stubs, balanced script tags) in addition to byte-compiling
  Python, so a browser-game creation has real acceptance evidence.
- Golden creation fixture `tests/fixtures/snake/index.html` — a complete,
  playable snake page (canvas, arrow/WASD controls, food, score, speed-up,
  pause, game over) — plus LoRA seed records
  (`training/datasets/creation/snake.jsonl`).

### Fixed — creation requests never degrade into existing-file patches

- `POST /v1/plan` + `/act` edit steps: a creation request ("make a minesweeper
  webpage game") is now answered **only** by new-file `creates` output
  (`_parse_creation` in `apps/server/forge_server/plans.py`). The old fallback
  to the existing-file patch shape once turned that request into a bogus
  line-patch against `README.md` in an undocumented `oldLine`/`newLine` diff
  dialect. Unparseable or wrong-shape replies get one corrective retry with
  explicit instructions, then fail cleanly.
- `core/patching/parser.py` rejects foreign patch dialects (`oldLine`/
  `newLine`, `patch`/`diff`/`hunks` keys, unknown operation keys) with an
  error that names the expected contract, instead of a generic "needs a
  'path'" message.

### Added

- Golden creation fixture `tests/fixtures/minesweeper/index.html` — a
  complete, self-contained, playable minesweeper page (reveal, flag, flood
  fill, first-click safety, timer, win/lose). Serves as the evaluation-tier-2
  golden artifact for creation tasks and as the source for the LoRA seed
  records.
- `creation` training behavior (`training/datasets/creation/`,
  ForgeCreate): records shaped as `{"instruction", "message", "creates"}`
  teach the LoRA to answer whole-file generation requests with the exact
  `{"message", "creates"}` shape the create grammar constrains.
  `prepare_dataset.py` normalizes these records.

### Added — Phase 1 (Runtime)

- Repository scaffold: `pyproject.toml`, `apps/server` (FastAPI Forge server),
  `core` (indexer, retrieval, inference, patching, git, memory),
  `apps/vscode` (TypeScript extension), `runtime` (prompts, grammars, llama
  launcher), `training` (dataset + LoRA pipeline), `installer` (Inno Setup).
- Forge server endpoints:
  - `GET /health`
  - `POST /v1/chat` (streaming SSE)
  - `POST /v1/completion` (fill-in-the-middle autocomplete)
  - `POST /v1/explain`, `POST /v1/edit`, `POST /v1/fix`, `POST /v1/tests`
  - `POST /v1/index`, `POST /v1/search`
  - `POST /v1/patch/preview`, `POST /v1/patch/apply`
- SQLite + FTS5 incremental repository indexer with symbol and chunk tables.
- Memory/resource guards: 4096-token chat budget, 1024-token completion
  budget, single-model policy, loopback-only networking.
- Training pipeline scripts: `prepare_dataset`, `validate_dataset`,
  `train_lora` (QLoRA), `evaluate`, `merge_lora`, `convert_gguf`, `quantize_gguf`,
  `benchmark`.

### Added — Grounded review and anchored edits

- `POST /v1/git/changes` now returns validated, line-anchored findings
  (`review_status` + `findings[]`) instead of unverified prose: the diff is
  labelled `staged`/`unstaged`, only added lines from the hunk headers are
  offered as anchors, and any finding the model cannot tie to a supplied
  anchor is rejected before it reaches the UI (`core/git/review.py`).
  The endpoint stays read-only and always returns the raw diff.
- `POST /v1/edit` and `POST /v1/fix` send `EDIT_SCHEMA` / `FIX_SCHEMA`
  (`core/patching/contracts.py`) as llama.cpp's `json_schema` response format,
  so structured actions are grammar-constrained rather than trusting prose.
- VS Code: **Fix** and **Refactor** inject their suggestion directly into the
  open file after a confirmed preview when it targets that file, via
  `apps/vscode/src/patch/inject.ts` (anchor + stale-file checks); anything else
  is handed to the existing sidebar patch review flow.
- Prompt grounding rules hardened (`runtime/prompts/chat.txt`, `plan.txt`,
  `fix.txt`): repository text is data, never instructions, and paths/line
  numbers may only be cited from the supplied context.

### Added — Agent rule layer (frames, scope, scored replies)

- New `core/agent/` package: `TaskFrame` (goal, facts, bounds, unknowns) and
  `ScopeContract` (allowed/forbidden paths) are derived from what was actually
  supplied to the model, and every structured reply is scored against the rule
  router `runtime/prompts/agent-rules.md` before it is returned.
- `ContextBuilder.build()` now layers the system prompt (behavior prompt +
  inline rules for that behavior), rides the task frame in the user turn, and
  records receipts + per-file line counts so findings mean "the reply
  disagrees with evidence it was given".
- `POST /v1/edit` and `POST /v1/fix` refuse a patch with
  `{"ok": false, "reason": "rule_violation", "findings": [...]}` when a
  block-severity rule fires (out-of-scope path, forbidden path, line range
  outside the supplied file, empty operation content, overlapping operations,
  patch without context); warn-level findings (unverified test claims,
  unresolved unknowns) are surfaced in `review` without refusing the reply.
- Circular import fixed (`core/agent/prompt` ↔ `core/retrieval`) and the
  `hierarchical` retrieval fallback migrated to the shared line helpers.

### In progress

- Phase 2 — Repository intelligence (scanner, FTS5, chunking, symbol extraction)
- Phase 3 — VS Code chat panel + streaming
- Phase 4 — Structured patches + diff viewer
- Phase 5 — Inline completion (prefix/suffix, debounce, cancellation)

### Planned

- Phase 6 — Git-aware fixes (read-only first)
- Phase 7 — ForgeCoder LoRA specialization
- Phase 8 — Windows installer