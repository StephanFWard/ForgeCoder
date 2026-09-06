"""Quantize a GGUF to Q4_K_M (or another llama.cpp quantization).

Wraps `llama-quantize`. Place the binary in runtime/llama/ or on PATH.

Usage:
    python training/scripts/quantize_gguf.py \
        --model models/gguf/forgecoder-fp16.gguf \
        --output models/gguf/forgecoder-1.5b-q4_k_m.gguf \
        --type Q4_K_M
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def find_binary() -> str | None:
    name = "llama-quantize.exe" if sys.platform == "win32" else "llama-quantize"
    local = ROOT / "runtime" / "llama" / name
    if local.exists():
        return str(local)
    return shutil.which(name)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--type", default="Q4_K_M",
                        help="llama.cpp quantization type (Q4_K_M, Q5_K_M, ...)")
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()

    binary = find_binary()
    if not binary:
        print("llama-quantize not found. Place it in runtime/llama/ or add it to PATH.",
              file=sys.stderr)
        return 2

    cmd = [binary, args.model, args.output, args.type, str(args.threads)]
    print(" ".join(cmd))
    return subprocess.run(cmd).returncode


if __name__ == "__main__":
    raise SystemExit(main())
