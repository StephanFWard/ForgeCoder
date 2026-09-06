"""Dataset preparation: raw task records -> validated JSONL for LoRA training.

Each dataset folder under training/datasets/ may contain JSON/JSONL records.
Accepted record shapes:

    ForgeChat:   {"instruction", "context", "response"}
    ForgeFix:    {"instruction", "code", "error", "response"}
    ForgeEdit:   {"instruction", "code", "response"}
    ForgeTest:   {"instruction", "code", "response"}
    completion:  {"prefix", "suffix", "target"}

Output: training/lora/train.jsonl + eval.jsonl with normalized fields:

    {"instruction", "input", "output", "behavior", "language", "source"}

Usage:
    python training/scripts/prepare_dataset.py \
        --in training/datasets --out training/lora --eval-size 2000
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

BEHAVIORS = ("chat", "bugfix", "editing", "testing", "completion",
             "debugging", "architecture", "security")


def _read_records(path: Path) -> list[dict]:
    if path.suffix == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, list) else [data]
    if path.suffix in {".jsonl", ".ndjson"}:
        out = []
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                out.append(json.loads(line))
        return out
    return []


def normalize(rec: dict, *, behavior: str, source: str) -> dict | None:
    """Map a raw record onto the unified schema, or None if unusable."""
    if "prefix" in rec and "target" in rec:
        output = rec["target"]
        instruction = rec.get("instruction")
        if not instruction:
            instruction = "Complete the code at the cursor."
        return {
            "instruction": instruction,
            "input": {"prefix": rec.get("prefix", ""), "suffix": rec.get("suffix", "")},
            "output": output,
            "behavior": "completion",
            "language": rec.get("language"),
            "source": source,
        }
    output = rec.get("response") or rec.get("output") or rec.get("patch")
    if not output:
        return None
    instruction = rec.get("instruction") or rec.get("request") or ""
    context = []
    for key in ("context", "code", "error", "logs", "selection"):
        if rec.get(key):
            context.append(f"{key}: {rec[key]}")
    return {
        "instruction": instruction,
        "input": "\n".join(context) if context else "",
        "output": output,
        "behavior": behavior,
        "language": rec.get("language"),
        "source": source,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--in", dest="in_dir", default="training/datasets")
    parser.add_argument("--out", dest="out_dir", default="training/lora")
    parser.add_argument("--eval-size", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    in_dir = Path(args.in_dir)
    out_dir = Path(args.out_dir)
    records: list[dict] = []

    for behavior_dir in sorted(p for p in in_dir.iterdir() if p.is_dir()):
        behavior = behavior_dir.name.lower()
        if behavior not in BEHAVIORS:
            continue
        for path in sorted(behavior_dir.rglob("*.json")) + sorted(behavior_dir.rglob("*.jsonl")):
            for rec in _read_records(path):
                norm = normalize(rec, behavior=behavior, source=str(path.relative_to(in_dir)))
                if norm:
                    records.append(norm)

    if not records:
        print(f"No records found under {in_dir} (expected subfolders: {', '.join(BEHAVIORS)})",
              file=sys.stderr)
        return 2

    rng = random.Random(args.seed)
    rng.shuffle(records)
    eval_size = min(args.eval_size, max(1, len(records) // 10))
    eval_records = records[:eval_size]
    train_records = records[eval_size:]

    out_dir.mkdir(parents=True, exist_ok=True)
    train_path = out_dir / "train.jsonl"
    eval_path = out_dir / "eval.jsonl"
    for path, items in ((train_path, train_records), (eval_path, eval_records)):
        with open(path, "w", encoding="utf-8") as fh:
            for rec in items:
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

    manifest = {
        "dataset_version": "0.1.0",
        "samples": len(records),
        "train": len(train_records),
        "eval": len(eval_records),
        "languages": sorted({r["language"] for r in records if r.get("language")}),
        "behaviors": sorted({r["behavior"] for r in records}),
        "seed": args.seed,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"Prepared {len(records)} records -> {train_path} / {eval_path}")
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
