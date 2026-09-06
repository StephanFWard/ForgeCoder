"""Convert a HuggingFace model to GGUF (fp16 base).

Wraps llama.cpp's `convert_hf_to_gguf.py`. It must be present at
runtime/llama/convert_hf_to_gguf.py or on PATH.

Usage:
    python training/scripts/convert_gguf.py \
        --model models/merged/forgecoder-1.5b \
        --output models/gguf/forgecoder-fp16.gguf
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def find_converter() -> str | None:
    local = ROOT / "runtime" / "llama" / "convert_hf_to_gguf.py"
    if local.exists():
        return str(local)
    return shutil.which("convert_hf_to_gguf.py")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="HF model directory")
    parser.add_argument("--output", required=True, help="Output .gguf path")
    parser.add_argument("--outtype", default="f16", choices=["f32", "f16", "bf16", "q8_0", "auto"])
    args = parser.parse_args()

    converter = find_converter()
    if not converter:
        print("convert_hf_to_gguf.py not found. Place it in runtime/llama/ (from "
              "https://github.com/ggml-org/llama.cpp) or add it to PATH.", file=sys.stderr)
        return 2

    cmd = [shutil.which("python") or "python", converter, args.model, "--outfile", args.output,
           "--outtype", args.outtype]
    print(" ".join(cmd))
    return subprocess.run(cmd).returncode


if __name__ == "__main__":
    raise SystemExit(main())
