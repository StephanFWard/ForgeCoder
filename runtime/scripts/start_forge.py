"""Start the full local stack: llama-server + Forge API.

Both processes are children of this one; Ctrl+C (or process exit) stops them.

Usage:
    python runtime/scripts/start_forge.py [--model path.gguf] [--no-llama]
"""
from __future__ import annotations

import argparse
import signal
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LLAMA = [sys.executable, str(ROOT / "runtime" / "scripts" / "launch_llama.py")]
FORGE = [sys.executable, "-m", "uvicorn", "forge_server.main:app",
         "--host", "127.0.0.1", "--port", "8787"]

procs: list[subprocess.Popen] = []


def _stop(*_args) -> None:
    for p in procs:
        if p.poll() is None:
            p.terminate()
    sys.exit(0)


def main() -> int:
    parser = argparse.ArgumentParser(description="Start ForgeCoder stack")
    parser.add_argument("--model", default=None)
    parser.add_argument("--no-llama", action="store_true", help="Skip llama-server (assume already running)")
    args = parser.parse_args()

    signal.signal(signal.SIGINT, _stop)
    signal.signal(signal.SIGTERM, _stop)

    if args.model:
        procs.append(subprocess.Popen(LLAMA + ["--model", args.model], cwd=ROOT))
    elif not args.no_llama:
        procs.append(subprocess.Popen(LLAMA, cwd=ROOT))

    procs.append(subprocess.Popen(FORGE, cwd=ROOT))

    for p in procs:
        p.wait()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
