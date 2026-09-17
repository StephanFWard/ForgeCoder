"""Golden fixture: the snake creation task.

"Make the game snake in html" once came back as an ask-dont-guess warning
instead of a game. The golden artifact pins what a correct creation answer
looks like: a complete, playable, self-contained page.
"""
from __future__ import annotations

import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "snake" / "index.html"


def test_snake_fixture_is_complete():
    html = FIXTURE.read_text(encoding="utf-8")
    assert html.lstrip().lower().startswith("<!doctype html")
    lowered = html.lower()
    for marker in ("keydown", "setinterval", "spawnfood", "restart", "game over"):
        assert marker in lowered, f"missing game behavior: {marker}"
    # No stub markers — the create pipeline flags these as placeholders
    # (see forge_server.plans._looks_placeholder).
    assert not re.search(
        r"TODO|FIXME|complete code here|not implemented|<placeholder",
        html,
        re.IGNORECASE,
    )


@pytest.mark.skipif(shutil.which("node") is None, reason="node is not available")
def test_snake_fixture_script_is_valid_javascript(tmp_path):
    html = FIXTURE.read_text(encoding="utf-8")
    match = re.search(r"<script>(.*)</script>", html, re.DOTALL)
    assert match, "fixture must contain an inline <script> block"
    js = tmp_path / "snake.js"
    js.write_text(match.group(1), encoding="utf-8")
    proc = subprocess.run(
        ["node", "--check", str(js)], capture_output=True, text=True, timeout=30,
    )
    assert proc.returncode == 0, proc.stderr or "node --check failed"


def test_snake_fixture_matches_lora_seed_record():
    seed = (
        Path(__file__).resolve().parents[2]
        / "training" / "datasets" / "creation" / "snake.jsonl"
    )
    if not seed.is_file():
        pytest.skip("training/datasets/creation/snake.jsonl not generated yet")
    import json

    html = FIXTURE.read_text(encoding="utf-8")
    records = [
        json.loads(line)
        for line in seed.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert records, "seed dataset must contain at least one record"
    assert any(
        rec.get("creates", {}).get("index.html") == html for rec in records
    ), "a seed record must contain the exact snake fixture content"


if __name__ == "__main__":  # pragma: no cover
    sys.exit(pytest.main([__file__]))
