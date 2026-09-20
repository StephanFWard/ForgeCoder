"""``forge-decide``: run one System One decision from a shell or a script.

Reads a Jev-shaped request, prints a Jev-shaped response. Free by default —
nothing leaves the machine unless ``--backend jev`` is asked for *and* a key is
present (that backend is billed per token and refuses to run otherwise).

Examples
--------

    forge-decide --backends
    forge-decide --question "Is this change risky?" --state-file diff.txt
    echo '{"state": "the build failed", "questions": {"gate": {"type": "noul",
          "instructions": "Is this a failure?"}}}' | forge-decide
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from core.system_one.backends import BackendUnavailable, backend_catalog
from core.system_one.decide import Decider
from core.system_one.primitives import DecisionRequest, NoulQuestion


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="forge-decide", description=__doc__.splitlines()[0])
    parser.add_argument("--backend", default=None,
                        help="deterministic (default), local, or jev (paid, opt-in)")
    parser.add_argument("--allow-paid", action="store_true",
                        help="permit the billed Jev API backend when --backend jev")
    parser.add_argument("--backends", action="store_true", help="list backends as JSON and exit")
    parser.add_argument("--input", default=None, help="request JSON file (default: stdin)")
    parser.add_argument("--state-file", default=None, help="state text file, with --question")
    parser.add_argument("--question", default=None, help="noul question about --state-file")
    return parser


def _load_request(args: argparse.Namespace) -> DecisionRequest:
    if args.state_file:
        if not args.question:
            raise ValueError("--state-file needs --question")
        state = Path(args.state_file).read_text(encoding="utf-8")
        return DecisionRequest(state=state, questions={"question": NoulQuestion(instructions=args.question)})
    raw = Path(args.input).read_text(encoding="utf-8") if args.input else sys.stdin.read()
    if not raw.strip():
        raise ValueError("no request supplied (pass --input, or JSON on stdin)")
    return DecisionRequest.model_validate_json(raw)


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.backends:
        print(json.dumps(backend_catalog(), indent=2))
        return 0

    try:
        request = _load_request(args)
    except (OSError, ValueError) as exc:
        print(json.dumps({"ok": False, "error": str(exc)}), file=sys.stderr)
        return 2

    decider = Decider(backend=args.backend, allow_paid=args.allow_paid)
    try:
        decision = decider.decide(request.state, request.questions, backend=request.backend)
    except BackendUnavailable as exc:
        print(json.dumps({"ok": False, "error": str(exc)}), file=sys.stderr)
        return 3

    print(json.dumps({"ok": True, **decision.model_dump()}, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover - module entry point
    raise SystemExit(main())
