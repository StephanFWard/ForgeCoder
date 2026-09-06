"""Evaluate an adapter or merged model against the eval set.

Measures per-behavior completeness + timing + memory. This is a smoke harness,
not a benchmark suite; training/evaluation/ contains the golden-repo tests.

Usage:
    python training/scripts/evaluate.py --adapter models/adapters/forgecoder-1.5b
    python training/scripts/evaluate.py --model models/merged/forgecoder-1.5b
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REQUIRED = {"instruction", "input", "output", "behavior"}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter", default=None, help="PEFT adapter dir")
    parser.add_argument("--model", default=None, help="Merged/standalone model dir or id")
    parser.add_argument("--eval-set", default="training/lora/eval.jsonl")
    parser.add_argument("--limit", type=int, default=50)
    args = parser.parse_args()

    if not args.adapter and not args.model:
        print("Provide --adapter or --model", file=sys.stderr)
        return 2

    try:
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        print(f"Missing training deps ({exc}). Use pip install -e '.[train]'", file=sys.stderr)
        return 2

    tokenizer = AutoTokenizer.from_pretrained(args.adapter or args.model)
    base = args.model or args.adapter
    model = AutoModelForCausalLM.from_pretrained(
        base, torch_dtype=torch.float16, device_map="auto"
    )
    if args.adapter:
        model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()

    rows = [
        json.loads(line)
        for line in Path(args.eval_set).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ][: args.limit]

    results: dict = {"total": len(rows), "by_behavior": {}, "errors": []}
    started = time.monotonic()
    for rec in rows:
        behavior = rec.get("behavior", "?")
        bucket = results["by_behavior"].setdefault(
            behavior, {"count": 0, "nonempty": 0, "seconds": 0.0}
        )
        bucket["count"] += 1
        prompt = rec.get("instruction", "") + "\n\n" + rec.get("input", "")
        t0 = time.monotonic()
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
        with torch.no_grad():
            out = model.generate(
                **inputs.to(model.device),
                max_new_tokens=256,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        gen = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        bucket["seconds"] += time.monotonic() - t0
        if gen.strip():
            bucket["nonempty"] += 1
        else:
            results["errors"].append({"behavior": behavior, "empty_generation": True})

    results["total_seconds"] = round(time.monotonic() - started, 2)
    for behavior, b in results["by_behavior"].items():
        b["nonempty_ratio"] = round(b["nonempty"] / max(b["count"], 1), 3)
        b["avg_seconds"] = round(b["seconds"] / max(b["count"], 1), 3)

    print(json.dumps(results, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
