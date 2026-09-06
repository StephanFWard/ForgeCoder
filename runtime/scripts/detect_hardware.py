"""Detect hardware and write the ForgeCoder config file.

Writes `%LOCALAPPDATA%/ForgeCoder/config.json` with a hardware profile:

    {"ram_gb": 16, "threads": 14, "context": 4096, "batch": 256,
     "gpu_layers": 0, "completion_context": 1024}

Usage:
    python runtime/scripts/detect_hardware.py [--json]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "apps" / "server"))

from forge_server.config import write_hardware_profile

from core.memory.budget import hardware_profile


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", action="store_true", help="Print profile as JSON only")
    args = parser.parse_args()

    profile = hardware_profile()
    if args.json:
        print(json.dumps(profile, indent=2))
        return 0

    path = write_hardware_profile(profile)
    print(f"Detected hardware profile written to {path}")
    for key, value in profile.items():
        print(f"  {key:20s} {value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
