"""Regression tests for the agent rule layer (core.agent.rules).

The server scores every structured reply (edit/fix) against the rule router
before returning it: a block-severity finding must refuse the reply, a warn
must surface without refusing. These tests pin both behaviors and the checks
behind them.
"""
from types import SimpleNamespace

from core.agent import RuleContext, load_rules, run_rules


def _ctx(**overrides) -> RuleContext:
    defaults = dict(behavior="edit", text="", patches=[],
                    has_context=True, allowed_files=["src/service.py"],
                    forbidden_files=[".git/**"], known_files=set(),
                    file_lines={"src/service.py": 10}, open_unknowns=[])
    defaults.update(overrides)
    return RuleContext(**defaults)


def _patch(path="src/service.py", start=2, end=2, kind="replace", content="x = 1"):
    op = SimpleNamespace(type=kind, start_line=start, end_line=end, content=content)
    return SimpleNamespace(path=path, operations=[op])


def test_clean_reply_passes_all_rules():
    report = run_rules(load_rules(), _ctx(patches=[_patch()]))
    assert report.ok
    assert report.rules_checked == len(load_rules())
    assert report.findings == []


def test_out_of_scope_path_blocks():
    report = run_rules(load_rules(), _ctx(patches=[_patch(path="src/other.py")]))
    assert report.blocked
    assert any(f.slug == "allowed-paths-only" for f in report.findings)


def test_forbidden_path_blocks():
    report = run_rules(load_rules(), _ctx(patches=[_patch(path=".git/config")]))
    assert report.blocked
    assert any(f.slug == "allowed-paths-only" for f in report.findings)


def test_line_range_outside_file_blocks():
    report = run_rules(load_rules(), _ctx(patches=[_patch(start=50, end=50)]))
    assert report.blocked
    assert any(f.slug == "line-ranges-in-file" for f in report.findings)


def test_unverified_test_claim_warns_but_does_not_block():
    report = run_rules(load_rules(), _ctx(text="All tests pass after this change."))
    assert report.ok
    assert any(f.slug == "no-unverified-test-claims" and f.severity == "warn"
               for f in report.findings)


def test_empty_patch_content_blocks():
    report = run_rules(load_rules(), _ctx(patches=[_patch(kind="replace", content="")]))
    assert report.blocked
    assert any(f.slug == "operations-well-formed" for f in report.findings)


def test_overlapping_operations_block():
    ops = [SimpleNamespace(type="replace", start_line=2, end_line=4, content="a"),
           SimpleNamespace(type="replace", start_line=4, end_line=6, content="b")]
    report = run_rules(load_rules(), _ctx(
        patches=[SimpleNamespace(path="src/service.py", operations=ops)]))
    assert report.blocked
    assert any(f.slug == "operations-non-overlapping" for f in report.findings)


def test_patch_without_any_context_blocks():
    report = run_rules(load_rules(), _ctx(patches=[_patch()], has_context=False))
    assert report.blocked
    assert any(f.slug == "fresh-context-before-edit" for f in report.findings)


def test_report_to_dict_shape():
    report = run_rules(load_rules(), _ctx(patches=[_patch(path="src/other.py")]))
    payload = report.to_dict()
    assert payload["ok"] is False
    assert payload["blocked"] and payload["findings"]
    finding = payload["findings"][0]
    assert set(finding) == {"slug", "category", "severity", "message"}
