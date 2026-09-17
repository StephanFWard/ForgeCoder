"""Golden fixture: the minesweeper creation task.

"Make a minesweeper webpage game" is the canonical creation-request example
(docs/training.md evaluation tier 2). It must route to the ``creates``
pipeline and yield a complete, playable, self-contained page — never a
line-patch against an existing file like README.md.
"""
from __future__ import annotations

import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

FIXTURE = (
    Path(__file__).resolve().parents[1] / "fixtures" / "minesweeper" / "index.html"
)


def test_minesweeper_fixture_is_complete():
    html = FIXTURE.read_text(encoding="utf-8")
    assert html.lstrip().lower().startswith("<!doctype html")
    assert "minesweeper" in html.lower()
    # Core game behaviors must be implemented in the page itself.
    lowered = html.lower()
    for marker in ("contextmenu", "flag", "reveal", "restart", "setinterval"):
        assert marker in lowered, f"missing game behavior: {marker}"
    # No stub markers — the create pipeline flags these as placeholders
    # (see forge_server.plans._looks_placeholder).
    assert not re.search(
        r"TODO|FIXME|complete code here|not implemented|<placeholder",
        html,
        re.IGNORECASE,
    )


@pytest.mark.skipif(shutil.which("node") is None, reason="node is not available")
def test_minesweeper_fixture_script_is_valid_javascript(tmp_path):
    html = FIXTURE.read_text(encoding="utf-8")
    match = re.search(r"<script>(.*)</script>", html, re.DOTALL)
    assert match, "fixture must contain an inline <script> block"
    js = tmp_path / "minesweeper.js"
    js.write_text(match.group(1), encoding="utf-8")
    proc = subprocess.run(
        ["node", "--check", str(js)], capture_output=True, text=True, timeout=30,
    )
    assert proc.returncode == 0, proc.stderr or "node --check failed"


def test_minesweeper_fixture_matches_lora_seed_record():
    """The training/datasets/creation seed record embeds this same page, so
    the fixture and the dataset must not drift apart."""
    seed = (
        Path(__file__).resolve().parents[2]
        / "training" / "datasets" / "creation" / "minesweeper.jsonl"
    )
    if not seed.is_file():
        pytest.skip("training/datasets/creation/minesweeper.jsonl not generated yet")
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
    ), "a seed record must contain the exact minesweeper fixture content"


if __name__ == "__main__":  # pragma: no cover
    sys.exit(pytest.main([__file__]))
