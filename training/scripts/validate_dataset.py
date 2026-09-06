"""Validate prepared datasets: schema, emptiness, lengths, duplication.

Usage:
    python training/scripts/validate_dataset.py training/lora/train.jsonl [eval.jsonl ...]
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

REQUIRED = {"instruction", "input", "output", "behavior"}
BEHAVIORS = {"chat", "bugfix", "editing", "testing", "completion",
             "debugging", "architecture", "security"}
MAX_INPUT_CHARS = 16_000
MAX_OUTPUT_CHARS = 32_000


def validate(path: Path) -> dict:
    stats: dict = {"file": str(path), "lines": 0, "ok": True, "errors": []}
    seq_hashes: set[str] = set()
    behavior_counts = Counter()
    language_counts: Counter = Counter()
    length_histogram: Counter = Counter()

    with open(path, encoding="utf-8") as fh:
        for idx, raw in enumerate(fh):
            raw = raw.strip()
            if not raw:
                continue
            stats["lines"] += 1
            try:
                rec = json.loads(raw)
            except json.JSONDecodeError as exc:
                stats["ok"] = False
                stats["errors"].append(f"line {idx + 1}: invalid JSON: {exc}")
                continue
            missing = [k for k in REQUIRED if not rec.get(k)]
            if missing:
                stats["ok"] = False
                stats["errors"].append(f"line {idx + 1}: missing {missing}")
                continue
            if rec["behavior"] not in BEHAVIORS:
                stats["ok"] = False
                stats["errors"].append(f"line {idx + 1}: bad behavior {rec['behavior']!r}")
            if len(rec.get("input", "")) > MAX_INPUT_CHARS:
                stats["ok"] = False
                stats["errors"].append(f"line {idx + 1}: input too long")
            if len(rec.get("output", "")) > MAX_OUTPUT_CHARS:
                stats["ok"] = False
                stats["errors"].append(f"line {idx + 1}: output too long")
            h = hash((rec["instruction"], str(rec["input"]), rec["output"]))
            if h in seq_hashes:
                stats["errors"].append(f"line {idx + 1}: duplicate record")
                stats["ok"] = False
            seq_hashes.add(h)
            behavior_counts[rec["behavior"]] += 1
            if rec.get("language"):
                language_counts[rec["language"]] += 1
            out_len = len(rec["output"])
            bucket = "<=512" if out_len <= 512 else ("<=2048" if out_len <= 2048 else ">2048")
            length_histogram[bucket] += 1

    stats["behaviors"] = dict(behavior_counts)
    stats["languages"] = dict(language_counts)
    stats["output_lengths"] = dict(length_histogram)
    return stats


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("files", nargs="+", help="JSONL dataset files")
    args = parser.parse_args()

    all_ok = True
    for f in args.files:
        stats = validate(Path(f))
        all_ok = all_ok and stats["ok"]
        print(f"{'OK  ' if stats['ok'] else 'FAIL'} {stats['file']}: "
              f"{stats['lines']} records")
        for err in stats["errors"][:10]:
            print(f"   - {err}")
    print("VALID" if all_ok else "INVALID")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
