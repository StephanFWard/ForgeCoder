"""Benchmark llama.cpp latency across completion/chat workload shapes.

Calls the running local server (default 127.0.0.1:8080) and reports
tokens/sec and memory. Use `--dry` to print the plan without running.

Usage:
    python training/scripts/benchmark.py [--url http://127.0.0.1:8080] [--tokens 128]
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from core.inference.client import InferenceClient


async def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8080")
    parser.add_argument("--tokens", type=int, default=128)
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--dry", action="store_true")
    args = parser.parse_args()

    prompt = "def factorial(n: int) -> int:\n    " + '"""Return n!."""\n    '
    client = InferenceClient(args.url)
    if not await client.ping():
        print(f"llama.cpp not reachable at {args.url}", file=sys.stderr)
        return 2

    if args.dry:
        print(json.dumps({"target": args.url, "max_tokens": args.tokens,
                          "iterations": args.iterations}, indent=2))
        return 0

    rows = []
    for i in range(args.iterations):
        t0 = time.monotonic()
        text = await client.complete(prompt, temperature=0.0, max_tokens=args.tokens)
        elapsed = time.monotonic() - t0
        n_chars = len(text)
        rows.append({
            "iteration": i + 1,
            "seconds": round(elapsed, 3),
            "chars": n_chars,
            "approx_tokens": max(1, n_chars // 4),
            "tokens_per_sec": round(max(1, n_chars // 4) / max(elapsed, 1e-6), 2),
        })
    await client.aclose()

    import psutil  # present when benchmarking locally

    report = {
        "url": args.url,
        "max_tokens": args.tokens,
        "runs": rows,
        "system_ram_gb": round(psutil.virtual_memory().total / (1024 ** 3), 1),
        "target_context": 4096,
    }
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
