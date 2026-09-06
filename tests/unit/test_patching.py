"""Tests for structured patch parsing and application."""
import pytest

from core.patching.apply import PatchError, apply_operations
from core.patching.diff import diff_stat, make_diff
from core.patching.parser import Operation, PatchParseError, extract_json, parse_patch


def test_parse_patch_valid():
    text = """{
      "summary": "added validation",
      "files": [{
        "path": "src/UserService.java",
        "operations": [
          {"type": "replace", "start_line": 42, "end_line": 47, "content": "if (id == null) return;\\n"}
        ]
      }]
    }"""
    patches = parse_patch(text)
    assert len(patches) == 1
    assert patches[0].path == "src/UserService.java"
    assert patches[0].operations[0].type == "replace"
    assert patches[0].operations[0].start_line == 42


def test_extract_json_from_code_fence():
    text = "```json\n{\"summary\": \"x\", \"files\": []}\n```"
    assert extract_json(text) == {"summary": "x", "files": []}


def test_operation_to_dict():
    """Regression: patches.py serialized operations via o.to_dict()."""
    op = Operation(type="replace", start_line=42, end_line=47, content="x = 1\n")
    assert op.to_dict() == {
        "type": "replace",
        "start_line": 42,
        "end_line": 47,
        "content": "x = 1\n",
    }


def test_extract_json_embedded_in_text():
    text = "Sure! Here's the patch:\n{\"summary\": \"y\", \"files\": []}\n\nHope that helps."
    assert extract_json(text)["summary"] == "y"


def test_parse_patch_rejects_unknown_type():
    with pytest.raises(PatchParseError):
        parse_patch({"files": [{"path": "a.py", "operations": [{"type": "explode"}]}]})


def test_parse_patch_rejects_bad_line():
    with pytest.raises(PatchParseError):
        parse_patch({"files": [{"path": "a.py", "operations": [{"type": "replace", "start_line": 0}]}]})


CONTENT = "line1\nline2\nline3\nline4\nline5\n"


def test_replace_operation():
    result = apply_operations(CONTENT, [Operation("replace", 2, 3, "a\nb")])
    assert result == "line1\na\nb\nline4\nline5\n"


def test_insert_operation():
    result = apply_operations(CONTENT, [Operation("insert", 2, None, "new")])
    assert result == "line1\nnew\nline2\nline3\nline4\nline5\n"


def test_delete_operation():
    result = apply_operations(CONTENT, [Operation("delete", 2, 3)])
    assert result == "line1\nline4\nline5\n"


def test_out_of_range_raises():
    with pytest.raises(PatchError):
        apply_operations(CONTENT, [Operation("replace", 99, 100, "x")])


def test_operations_applied_bottom_up():
    ops = [Operation("replace", 1, 1, "ONE"), Operation("replace", 5, 5, "FIVE")]
    result = apply_operations(CONTENT, ops)
    assert result == "ONE\nline2\nline3\nline4\nFIVE\n"


def test_make_diff_and_stat():
    diff = make_diff("a\nb\nc\n", "a\nB\nc\n", "src/x.py")
    assert diff.startswith("--- a/src/x.py")
    stat = diff_stat(diff)
    assert stat["added"] >= 1
    assert stat["removed"] >= 1


def test_empty_diff():
    assert make_diff("same\n", "same\n", "x") == ""
