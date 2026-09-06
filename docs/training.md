# Training

## Where training happens

**Not on the 6 GB laptop.** The laptop is the inference target. Training runs
on a machine with a suitable GPU (≥8 GB VRAM is comfortable for QLoRA on a
1.5B model). Adapter or merged GGUF weights are copied back to the laptop.

## Pipeline

```
raw datasets (training/datasets/<behavior>/*.json/.jsonl)
        │  prepare_dataset.py
        ▼
validated JSONL (training/lora/train.jsonl, eval.jsonl, manifest.json)
        │  validate_dataset.py
        ▼
train_lora.py  (QLoRA, config: training/configs/lora.yaml)
        ▼
LoRA adapter (models/adapters/forgecoder-1.5b)
        │  merge_lora.py
        ▼
merged HF model (models/merged/forgecoder-1.5b)
        │  convert_gguf.py
        ▼
FP16 GGUF
        │  quantize_gguf.py
        ▼
Q4_K_M GGUF (models/gguf/forgecoder-1.5b-q4_k_m.gguf)
```

## Dataset design

ForgeCoder is trained on **software-engineering tasks**, not raw code:

| Behavior   | Example record                                    |
| ---------- | ------------------------------------------------- |
| chat       | explain a class, answer with supplied context     |
| bugfix     | bug + context + patch + verification              |
| editing    | selection + requested transform → replacement     |
| testing    | code → unit tests                                 |
| completion | prefix/suffix → target (fill-in-the-middle)       |
| debugging  | error + logs → diagnosis + fix                    |
| architecture | files → component explanation                  |
| security   | vulnerable code → hardened response               |

Scaling/quality rules:

- Only include data you're licensed to redistribute (Apache-2.0 / MIT /
  synthetic).
- Prefer records where the output is machine-verifiable (a patch that
  applies, tests that run).
- Validate with `validate_dataset.py` before training: schema, empty outputs,
  duplicates, oversized records.

## Config

`training/configs/lora.yaml`:

```yaml
base_model: Qwen/Qwen2.5-Coder-1.5B-Instruct
output_dir: models/adapters/forgecoder-1.5b
lora:
  r: 8
  alpha: 16
  dropout: 0.05
  bias: none
  task_type: CAUSAL_LM
target_modules: [q_proj, k_proj, v_proj, o_proj]
training:
  epochs: 2
  learning_rate: 0.0002
  batch_size: 1
  gradient_accumulation_steps: 16
  max_seq_length: 2048
  gradient_checkpointing: true
  fp16: true
```

## Commands

```bash
# GPU host
pip install -e ".[train]"

python training/scripts/prepare_dataset.py --in training/datasets --out training/lora
python training/scripts/validate_dataset.py training/lora/train.jsonl training/lora/eval.jsonl
python training/scripts/train_lora.py --config training/configs/lora.yaml
python training/scripts/evaluate.py --adapter models/adapters/forgecoder-1.5b --limit 100
python training/scripts/merge_lora.py --adapter models/adapters/forgecoder-1.5b \
    --output models/merged/forgecoder-1.5b
python training/scripts/convert_gguf.py --model models/merged/forgecoder-1.5b \
    --output models/gguf/forgecoder-fp16.gguf
python training/scripts/quantize_gguf.py --model models/gguf/forgecoder-fp16.gguf \
    --output models/gguf/forgecoder-1.5b-q4_k_m.gguf
python training/scripts/benchmark.py --dry
```

## Evaluation

Three tiers:

1. **Smoke** — `evaluate.py` runs the eval set through the adapter and
   reports emptiness + latency per behavior.
2. **Golden repositories** — `tests/fixtures/` mini repos with
   machine-verifiable tasks ("find the auth bug", "add validation",
   "generate tests"). See `docs/testing` section of `CONTRIBUTING.md`.
3. **Release gates** — 100 questions / 100 fixes / 100 edits / 100 tests /
   100 completions, measuring correctness, syntax validity, patch
   applicability, test success, latency, memory.

## GGUF tooling

`convert_gguf.py` and `quantize_gguf.py` wrap llama.cpp's
`convert_hf_to_gguf.py` and `llama-quantize`. Place them in `runtime/llama/`
(or PATH) before running.