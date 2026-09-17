# Changelog

All notable changes to ForgeCoder are documented here. Format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project
adheres to **0.x** versioning until 1.0.

## [Unreleased]

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

### In progress

- Phase 2 — Repository intelligence (scanner, FTS5, chunking, symbol extraction)
- Phase 3 — VS Code chat panel + streaming
- Phase 4 — Structured patches + diff viewer
- Phase 5 — Inline completion (prefix/suffix, debounce, cancellation)

### Planned

- Phase 6 — Git-aware fixes (read-only first)
- Phase 7 — ForgeCoder LoRA specialization
- Phase 8 — Windows installer