# Contributing to ForgeCoder

Thanks for considering a contribution! ForgeCoder is intentionally split into
independent layers so that any one of them can be replaced without disturbing
the others:

```
ForgeCoder Runtime | Retrieval | VS Code | LoRA | Model
```

## Getting started

Prerequisites: Python 3.10+, Node 18+, git.

```powershell
git clone <your-fork>
cd ForgeCoder

# Python runtime
python -m venv .venv
.venv\Scripts\activate
pip install -e ".[dev]"

# VS Code extension
cd apps\vscode
npm install
npm run compile
```

## Running tests

```powershell
python -m pytest                # Python unit + integration tests
cd apps\vscode && npm test      # extension smoke test (launches VS Code)
```

`tests/` contains:

- `tests/unit` — pure logic (chunking, ranking, budgeting, patching)
- `tests/integration` — API behavior with a mocked inference backend
- `tests/fixtures` — miniature golden repositories used by integration tests

## Code style

- Python: `ruff` (`python -m ruff check apps/server core tests`).
- TypeScript: strict mode, `npm run compile` must pass.
- No new required dependency unless it earns its place in the memory budget.

## Branching & commits

- Work on a branch: `git checkout -b feat/<thing>`.
- Commit messages follow Conventional Commits
  (`feat:`, `fix:`, `docs:`, `chore:`).

## Engineering rules (non-negotiable)

1. Only one model is ever loaded.
2. SQLite is the only datastore in v0.x (no vector DB, no embeddings).
3. Chat context ≤ 4096 tokens; completion context ≤ 1024 tokens.
4. Never load an entire repository into memory — index incrementally.
5. The local machine is the *inference* target. Training happens on a GPU host.
6. Never automatically overwrite a user's source file.

## Training contributions

See `docs/training.md`. Dataset contributions must include real task
instructions (bug → context → patch → verification), not raw source dumps.

## Code of conduct

Be respectful and constructive. Review each other's work with the goal of a
fast, light, local product.