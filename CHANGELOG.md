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

### In progress

- Phase 2 — Repository intelligence (scanner, FTS5, chunking, symbol extraction)
- Phase 3 — VS Code chat panel + streaming
- Phase 4 — Structured patches + diff viewer
- Phase 5 — Inline completion (prefix/suffix, debounce, cancellation)

### Planned

- Phase 6 — Git-aware fixes (read-only first)
- Phase 7 — ForgeCoder LoRA specialization
- Phase 8 — Windows installer