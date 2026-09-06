"""Start the full local stack: llama-server + Forge API.

Idempotent: if either service already answers on its port, it is *reused*
instead of started again — so running this twice never causes a bind error.

Both spawned processes are children of this one; Ctrl+C (or process exit)
stops the children this script owns.

Usage:
    python runtime/scripts/start_forge.py [--model path.gguf] [--no-llama]
"""
from __future__ import annotations

import argparse
import signal
import subprocess
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LLAMA = [sys.executable, str(ROOT / "runtime" / "scripts" / "launch_llama.py")]
FORGE = [sys.executable, "-m", "uvicorn", "forge_server.main:app",
         "--host", "127.0.0.1", "--port", "8787"]

LLAMA_URL = "http://127.0.0.1:8080"
FORGE_URL = "http://127.0.0.1:8787"

procs: list[subprocess.Popen] = []


def service_up(url: str, timeout: float = 2.0) -> bool:
    """True when <url>/health answers with ok/json."""
    try:
        with urllib.request.urlopen(url + "/health", timeout=timeout) as resp:
            body = resp.read(1024)
        return resp.status == 200 and (b"ok" in body or b"status" in body)
    except (urllib.error.URLError, OSError, ValueError):
        return False


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

    if args.no_llama:
        pass
    elif service_up(LLAMA_URL):
        print(f"llama-server already running at {LLAMA_URL} — reusing it.")
    else:
        procs.append(subprocess.Popen(LLAMA + (["--model", args.model] if args.model else []), cwd=ROOT))

    if service_up(FORGE_URL):
        print(f"Forge API already running at {FORGE_URL} — reusing it.")
    else:
        procs.append(subprocess.Popen(FORGE, cwd=ROOT))

    if not procs:
        print("ForgeCoder stack is up:")
        print(f"  llama.cpp : {LLAMA_URL}")
        print(f"  Forge API : {FORGE_URL}")
        return 0

    for p in procs:
        p.wait()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
