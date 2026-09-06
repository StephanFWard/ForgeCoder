"""Launch llama-server for ForgeCoder.

- Locates llama-server.exe (runtime/llama or PATH)
- Picks the GGUF model (models/gguf, prefers forgecoder or qwen q4_k_m)
- Applies the hardware profile from %LOCALAPPDATA%/ForgeCoder/config.json
- Binds 127.0.0.1:8080 only

Usage:
    python runtime/scripts/launch_llama.py [--model path.gguf] [--dry-run]
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from core.memory.budget import hardware_profile  # noqa: E402 (path setup above)

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8080


def config_path() -> Path:
    base = os.environ.get("LOCALAPPDATA") or str(Path.home())
    return Path(base) / "ForgeCoder" / "config.json"


def load_profile() -> dict:
    path = config_path()
    if path.exists():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(data, dict) and data:
                return data
        except (json.JSONDecodeError, OSError):
            pass
    return hardware_profile()


def find_binary() -> str | None:
    local = ROOT / "runtime" / "llama" / ("llama-server.exe" if os.name == "nt" else "llama-server")
    if local.exists():
        return str(local)
    return shutil.which("llama-server")


def find_model(explicit: str | None) -> str | None:
    if explicit:
        return explicit
    gguf_dir = ROOT / "models" / "gguf"
    if gguf_dir.is_dir():
        candidates = sorted(gguf_dir.glob("*.gguf"))
        prefer = [c for c in candidates if "q4_k_m" in c.name.lower()
                  and ("forge" in c.name.lower() or "qwen" in c.name.lower())]
        pick = prefer[0] if prefer else (candidates[0] if candidates else None)
        if pick:
            return str(pick)
    return None


def build_command(binary: str, model: str, profile: dict) -> list[str]:
    threads = max(1, int(profile.get("threads", 4)))
    context = max(512, int(profile.get("context", 4096)))
    batch = max(64, int(profile.get("batch", 128)))
    gpu_layers = max(0, int(profile.get("gpu_layers", 0)))
    cmd = [
        binary,
        "-m", model,
        "-c", str(context),
        "-b", str(batch),
        "-ub", str(batch // 2),
        "-t", str(threads),
        "--host", DEFAULT_HOST,
        "--port", str(DEFAULT_PORT),
    ]
    if gpu_layers > 0:
        cmd += ["-ngl", str(gpu_layers)]
    return cmd


def main() -> int:
    parser = argparse.ArgumentParser(description="Launch llama-server for ForgeCoder")
    parser.add_argument("--model", default=None, help="Path to a .gguf model file")
    parser.add_argument("--dry-run", action="store_true", help="Print the command without running")
    args = parser.parse_args()

    binary = find_binary()
    model = find_model(args.model)

    if not binary:
        print("llama-server not found. Place llama-server(.exe) in runtime/llama/ or add it to PATH.",
              file=sys.stderr)
        return 2
    if not model:
        print("No .gguf model found in models/gguf/. Download a Q4_K_M GGUF of "
              "Qwen2.5-Coder-1.5B-Instruct and drop it there.", file=sys.stderr)
        return 2

    profile = load_profile()
    cmd = build_command(binary, model, profile)
    print(" ".join(cmd))
    if args.dry_run:
        return 0

    try:
        proc = subprocess.run(cmd)
        return proc.returncode
    except KeyboardInterrupt:
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
