"""Merge a LoRA adapter into the base weights.

Usage:
    python training/scripts/merge_lora.py \
        --adapter models/adapters/forgecoder-1.5b \
        --output models/merged/forgecoder-1.5b
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

log = logging.getLogger("forge_merge")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--base", default=None,
                        help="Base model id/path (default: taken from adapter_config.json)")
    args = parser.parse_args()

    try:
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        print(f"Missing training deps ({exc}). Use pip install -e '.[train]'", file=sys.stderr)
        return 2

    adapter_dir = Path(args.adapter)
    import json

    base = args.base
    if not base:
        cfg_path = adapter_dir / "adapter_config.json"
        if cfg_path.exists():
            base = json.loads(cfg_path.read_text(encoding="utf-8")).get("base_model_name_or_path")
    if not base:
        print("Could not determine base model; pass --base", file=sys.stderr)
        return 2

    log.info("Loading base %s", base)
    model = AutoModelForCausalLM.from_pretrained(base, torch_dtype=torch.float16)
    log.info("Loading adapter %s", args.adapter)
    model = PeftModel.from_pretrained(model, args.adapter)
    log.info("Merging...")
    merged = model.merge_and_unload()

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(out)
    tokenizer = AutoTokenizer.from_pretrained(base)
    tokenizer.save_pretrained(out)
    log.info("Merged model saved to %s", out)
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    raise SystemExit(main())
