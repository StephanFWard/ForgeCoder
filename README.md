# ForgeCoder

<img width="492" height="967" alt="image" src="https://github.com/user-attachments/assets/e86fed06-1e1a-46b0-8d8a-f76be86b7cbd" />

A **thin, local-first coding platform** wrapped around a 1.5B coder model.
ForgeCoder does not try to put repository intelligence into the model — it
keeps a 1.5B parameter Qwen2.5-Coder running through **llama.cpp** on your
machine, and adds the engineering around it: an incremental SQLite/FTS5
repository index, smart context budgeting, structured patch output, and a
native VS Code extension.

```
        FORGECODER
            │
    ┌───────┴────────┐
    │                │
 VS Code           CLI
    │                │
    └───────┬────────┘
            │
    Forge Local API
        :8787
            │
   ┌────────┼───────────┐
   │        │           │
 Chat/Edit Retrieval    Git
   │        │           │
   │    SQLite/FTS5     │
   │        │           │
   └────────┼───────────┘
            │
     Context Builder
      4K-token budget
            │
            ▼
     llama.cpp :8080
            │
            ▼
    ForgeCoder 1.5B Q4
```

## Why this design

- **Model size.** A 1.5B model (`Qwen2.5-Coder-1.5B-Instruct`, Apache-2.0,
  GGUF available) fits comfortably in a **6 GB RAM** dev machine at
  `Q4_K_M` quantization.
- **One model, forever.** Chat *and* completion use a single loaded model.
  No embeddings model, no vector DB in v0.x.
- **Inference is replaceable.** llama.cpp exposes an OpenAI-compatible HTTP
  API; the server, indexer, and extension never depend on llama.cpp internals.
- **The LoRA comes later.** v0.1 ships the stock Qwen2.5-Coder GGUF.
  ForgeCoder's specialization is a LoRA trained on software-engineering
  *tasks* (chat/edit/fix/test/complete), merged and quantized downstream.

## Repository layout

```
apps/server    FastAPI "Forge" API  (:8787, localhost only)
apps/vscode    TypeScript VS Code extension (chat panel, completion, diffs)
core           indexer / retrieval / inference / patching / git / memory
               system_one (deterministic weighted decisions, free)
training       dataset + QLoRA pipeline (runs on a GPU host)
runtime        llama.cpp launcher, prompts, grammars
installer      Inno Setup script (Windows)
docs           architecture, api, system-one, memory, training, vscode guides
tests          unit + integration + fixtures
models         base / adapters / merged / gguf (gitignored contents)
```

## Quick start (development)

```powershell
# 1. Python runtime
python -m venv .venv
.venv\Scripts\activate
pip install -e ".[dev]"

# 2. Model — download the Q4_K_M GGUF of Qwen2.5-Coder-1.5B-Instruct
#    and place the .gguf under models/gguf/.

# 3. Detect hardware once (writes %LOCALAPPDATA%\ForgeCoder\config.json)
python runtime/scripts/detect_hardware.py

# 4. Start the whole stack (llama-server :8080 + Forge API :8787)
python runtime/scripts/start_forge.py

# 5. Install the VS Code extension
cd apps\vscode
npm install
npm run compile
# then press F5 in VS Code, or `npm run package` + Install from VSIX
```

## Model strategy

```
Qwen2.5-Coder-1.5B (GGUF, stock) ──── v0.1 runtime
        │
ForgeCoder LoRA (chat/edit/fix/test/complete)
        │
merged HF model
        │
convert → fp16 GGUF → quantize → Q4_K_M
```

Training commands (see `docs/training.md`):

```bash
python training/scripts/prepare_dataset.py --in training/datasets --out training/lora
python training/scripts/train_lora.py --config training/configs/lora.yaml
python training/scripts/evaluate.py --adapter models/adapters/forgecoder-1.5b
python training/scripts/merge_lora.py --adapter models/adapters/forgecoder-1.5b \
    --output models/merged/forgecoder-1.5b
python training/scripts/convert_gguf.py --model models/merged/forgecoder-1.5b \
    --output models/gguf/forgecoder-fp16.gguf
python training/scripts/quantize_gguf.py --model models/gguf/forgecoder-fp16.gguf \
    --output models/gguf/forgecoder-1.5b-q4_k_m.gguf
```

## API surface (Forge server)

| Endpoint                 | Purpose                                      |
| ------------------------ | -------------------------------------------- |
| `GET /health`            | Server + llama.cpp + DB status               |
| `POST /v1/chat`          | Streaming chat with retrieval context        |
| `POST /v1/completion`    | Fill-in-the-middle autocomplete              |
| `POST /v1/explain`       | Explain a selection                          |
| `POST /v1/edit`          | Structured edit of a selection (patch)       |
| `POST /v1/fix`           | Fix code + error context                     |
| `POST /v1/tests`         | Generate tests for a selection               |
| `POST /v1/index`         | Index (or re-index) a workspace              |
| `POST /v1/search`        | FTS5 repository search (+ optional weighted rerank)|
| `POST /v1/decide`        | Typed weighted decisions (System One, free)  |
| `GET /v1/decide/backends`| Decision backends and their cost             |
| `POST /v1/patch/preview` | Apply a patch in-memory, return a diff       |
| `POST /v1/patch/apply`   | Apply a patch to disk (requires confirmation)|

Per-endpoint details: `docs/api.md`.

## System One decisions (free by default)

`core/system_one` adopts the contract of TypeSafe AI's **Jev** ("System One"
model — typed `choice` / `score` / `noul` questions answered against a state
with probability distributions and a confidence) and serves it with **free,
offline backends by default**: the `deterministic` backend is pure
IDF-weighted arithmetic — reproducible, no model, no network, no key — and the
`local` backend blends verdicts from the on-device 1.5B model into the same
shape, degrading gracefully when llama.cpp is down. The billed hosted Jev API
is opt-in only (`FORGECODER_SYSTEM_ONE_ALLOW_PAID=1` plus a key).

```powershell
pip install -e ".[jev]"          # installs the wheel on Python 3.14+; on
                                 # older interpreters core.system_one
                                 # provides the identical API for free

# one decision from a shell, free:
'{"state": "build failed", "questions": {"gate": {"type": "noul",
 "instructions": "Is this a failure?"}}}' | .venv\Scripts\forge-decide.exe
```

```python
from core.system_one import Decider, NoulQuestion, combine, gate

decider = Decider()                      # deterministic: offline, reproducible
decision = decider.decide(
    {"build": "failed", "log": "AssertionError in test_user_lookup"},
    {"risky": NoulQuestion(instructions="Is this state risky?")},
)
if gate(decision.answers["risky"], threshold=0.7):
    ...                                   # branch on a number, not prose
```

Retrieval rides the same layer: `POST /v1/search` with `"weighted": true`
blends heuristic scores with System One relevance levels (default weight
`0.35`, deterministic offline arithmetic). Full guide: `docs/system-one.md`.

## Hardware profile

On first launch the server writes `%LOCALAPPDATA%\ForgeCoder\config.json`
based on detected RAM/CPU/GPU:

```json
{ "ram_gb": 6, "threads": 4, "context": 4096,
  "batch": 128, "gpu_layers": 0, "completion_context": 1024 }
```

## License

Apache-2.0. The base model (Qwen2.5-Coder-1.5B-Instruct) is Apache-2.0 too;
llama.cpp is MIT. See `LICENSE`.
