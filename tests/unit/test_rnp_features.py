"""Tests for RNP-inspired retrieval features (Fisher & Rao 2023, PMC10637337).

Covers:
- adaptive weights (hypernetwork analog)          core/retrieval/adaptive.py
- query decomposition (recursive reuse)           core/retrieval/query_decompose.py
- hierarchical context (part-whole tree)          core/retrieval/hierarchical.py
- RNP dual-scoring search (what/where)            core/retrieval/search.py
- predictive verification (prediction errors)     core/inference/verify.py
- grammar files present                           runtime/grammars/
"""
from pathlib import Path

import pytest

from core.indexer.index import CodeIndex
from core.inference.verify import extract_citations, verify_output
from core.retrieval.adaptive import AdaptiveWeights, classify_query
from core.retrieval.hierarchical import (
    HierarchicalContextBuilder,
    TreeNode,
    count_subtree,
)
from core.retrieval.query_decompose import decompose_and_search, decompose_query
from core.retrieval.search import RNPSearchEngine, SearchEngine


def _write_tree(root: Path) -> Path:
    src = root / "src"
    src.mkdir(parents=True)
    (src / "UserService.java").write_text(
        "package com.example;\n\n"
        "public class UserService {\n"
        "    public String greet(String name) { return \"hi \" + name; }\n"
        "    public void validate(String email) {\n"
        "        if (email == null) throw new IllegalArgumentException(\"bad\");\n"
        "    }\n"
        "}\n",
        encoding="utf-8",
    )
    (src / "math_utils.py").write_text(
        "def add(a, b):\n"
        "    \"\"\"Add two numbers.\"\"\"\n"
        "    return a + b\n",
        encoding="utf-8",
    )
    (root / "README.md").write_text("# notes\n", encoding="utf-8")
    return root


def _indexed(tmp_path: Path) -> CodeIndex:
    index = CodeIndex(Path(tmp_path) / "forge.db")
    index.connect()
    index.index_workspace(_write_tree(tmp_path), prune=False)
    return index


# ---------------------------------------------------------------------------
# Adaptive weights (hypernetwork analog)
# ---------------------------------------------------------------------------

def test_classify_fix_query():
    assert classify_query("Fix the null pointer exception in UserService") == "fix"
    assert classify_query("crash when validate is called with null email") == "fix"


def test_classify_explain_query():
    assert classify_query("Explain how greet works") == "explain"
    assert classify_query("Why does validate throw for null?") == "explain"


def test_classify_edit_query():
    assert classify_query("Add validation to the email field") == "edit"
    assert classify_query("Refactor UserService to use Optional") == "edit"


def test_classify_fallback_search():
    assert classify_query("UserService") == "search"


def test_adaptive_weights_boost_by_query_type():
    w = AdaptiveWeights()
    fix_w = w.get("fix")
    explain_w = w.get("explain")
    # fix queries weight structure (where) above content (what)
    assert fix_w[1] > fix_w[0]
    # explain queries weight content (what) above structure (where)
    assert explain_w[0] > explain_w[1]


def test_adaptive_weights_unknown_type_falls_back():
    w = AdaptiveWeights()
    assert w.get("unknown-type") == w.get("search")


# ---------------------------------------------------------------------------
# Query decomposition (recursive reuse of the search primitive)
# ---------------------------------------------------------------------------

def test_decompose_simple_query_stays_single():
    dq = decompose_query("Fix the null check in UserService")
    assert len(dq.subqueries) == 1
    assert not dq.needs_decomposition
    assert dq.subqueries[0].query == "Fix the null check in UserService"


def test_decompose_complex_query_splits():
    dq = decompose_query("Why does validate throw; and where is greet defined. Also fix the null email path")
    assert dq.needs_decomposition, "multi-clause query should split"
    assert 2 <= len(dq.subqueries) <= 6
    for sq in dq.subqueries:
        assert sq.query
        assert sq.intent in ("fix", "explain", "edit", "plan", "search")
        assert sq.scope.startswith(("all", "file:", "symbol:"))


def test_infer_scope_detects_filenames():
    dq = decompose_query("Explain the logic in UserService.java")
    assert dq.subqueries[0].scope.startswith("file:")


def test_infer_scope_detects_symbols():
    dq = decompose_query("why does get_user return null")
    assert dq.subqueries[0].scope.startswith("symbol:")


@pytest.mark.asyncio
async def test_decompose_and_search_merges(tmp_path):
    index = _indexed(Path(tmp_path))
    try:
        engine = RNPSearchEngine(index)
        dq = await decompose_and_search(
            engine,
            "What does greet return; and how does validate handle null email",
            max_results_per_subq=4,
        )
        assert dq.needs_decomposition
        assert dq.merged_results, "parallel sub-searches should retrieve chunks"
        # merged results must be deduplicated and sorted by combined score
        keys = [(r.path, r.start_line, r.end_line) for r in dq.merged_results]
        assert len(keys) == len(set(keys))
        scores = [r.combined_score for r in dq.merged_results]
        assert scores == sorted(scores, reverse=True)
    finally:
        index.close()


# ---------------------------------------------------------------------------
# RNP dual-scoring search (what / where)
# ---------------------------------------------------------------------------

def test_rnp_search_returns_dual_scores(tmp_path):
    index = _indexed(Path(tmp_path))
    try:
        engine = RNPSearchEngine(index)
        results = engine.search("greet")
        assert results
        r = results[0]
        assert hasattr(r, "what_score") and hasattr(r, "where_score")
        assert hasattr(r, "combined_score")
        assert r.path == "src/UserService.java"
        assert r.combined_score > 0
        scores = [x.combined_score for x in results]
        assert scores == sorted(scores, reverse=True)
    finally:
        index.close()


def test_rnp_search_detailed_tree(tmp_path):
    index = _indexed(Path(tmp_path))
    try:
        engine = RNPSearchEngine(index)
        detailed = engine.search_detailed("validate email")
        assert detailed.results
        assert detailed.tree is not None
        assert detailed.tree["kind"] == "workspace"
        files = detailed.tree["children"]
        assert any(p.endswith("UserService.java") for p in files)
        for node in files.values():
            assert node["kind"] == "file"
            assert node["children"], "file node should contain chunk children"
    finally:
        index.close()


def test_plain_search_engine_still_works(tmp_path):
    index = _indexed(Path(tmp_path))
    try:
        engine = SearchEngine(index)
        results = engine.search("greet", limit=3)
        assert results
        assert results[0].path == "src/UserService.java"
    finally:
        index.close()


# ---------------------------------------------------------------------------
# Hierarchical context assembly (part-whole tree)
# ---------------------------------------------------------------------------

def test_treenode_token_counting():
    leaf = TreeNode(kind="chunk", label="c1", tokens=10)
    mid = TreeNode(kind="symbol", label="greet", tokens=5, children=[leaf])
    root = TreeNode(kind="file", label="UserService.java", tokens=3, children=[mid])
    assert count_subtree(root) == 18


def test_builder_renders_hierarchy(tmp_path):
    index = _indexed(Path(tmp_path))
    try:
        builder = HierarchicalContextBuilder(index=index)
        ctx = builder.build("How does greet work?", workspace=str(tmp_path), budget=3000)
        assert ctx.system, "system prompt should be loaded"
        assert ctx.request == "How does greet work?"
        assert ctx.root.kind == "workspace"
        assert ctx.total_tokens > 0
        text = ctx.as_message_text()
        assert "Repository Context" in text
    finally:
        index.close()


def test_builder_falls_back_to_current_file(tmp_path):
    index = _indexed(Path(tmp_path))
    try:
        builder = HierarchicalContextBuilder(index=index)
        ctx = builder.build(
            "Explain this", workspace=str(tmp_path),
            file="src/math_utils.py", budget=2000,
        )
        assert ctx.total_tokens >= 0
        assert ctx.root.kind == "workspace"
    finally:
        index.close()


def test_builder_summary_shape(tmp_path):
    index = _indexed(Path(tmp_path))
    try:
        builder = HierarchicalContextBuilder(index=index)
        ctx = builder.build("greet", workspace=str(tmp_path), budget=2000)
        summary = ctx.tree_summary()
        assert summary["kind"] == "workspace"
        assert "children" in summary and "tokens" in summary
    finally:
        index.close()


# ---------------------------------------------------------------------------
# Predictive verification (prediction-error signal)
# ---------------------------------------------------------------------------

def test_extract_citations_with_lines():
    cites = extract_citations("The bug is in src/UserService.java:5-7 near greet.")
    assert ("src/UserService.java", 5, 7) in cites


def test_extract_citations_without_lines():
    cites = extract_citations("See src/UserService.java for details.")
    assert any(c[0].endswith("UserService.java") and c[1] is None for c in cites)


def test_verify_valid_citation_passes():
    ctx = "## src/UserService.java\n```java\npublic String greet(String name) {}\n```"
    result = verify_output(
        "The problem is in src/UserService.java:3 where greet is defined.",
        context_text=ctx,
    )
    assert result.valid
    assert result.confidence == 1.0
    assert not result.issues


def test_verify_hallucinated_path_fails():
    ctx = "## src/UserService.java\n```java\nclass UserService {}\n```"
    result = verify_output(
        "Check src/NonExistent.java:42 for the bug.",
        context_text=ctx,
    )
    assert not result.valid
    assert result.issues and "hallucination" in result.issues[0]
    assert result.confidence < 1.0


def test_verify_json_structure():
    good = verify_output('{"summary": "ok", "files": []}', expected_json_keys=["summary", "files"])
    assert good.valid
    bad = verify_output('{"summary": "ok"}', expected_json_keys=["summary", "files"])
    assert not bad.valid
    assert any("Missing required JSON key" in i for i in bad.issues)
    invalid = verify_output("not json at all", expected_json_keys=["summary"])
    assert not invalid.valid


# ---------------------------------------------------------------------------
# Grammar assets
# ---------------------------------------------------------------------------

def test_grammar_files_present():
    gdir = Path(__file__).resolve().parents[2] / "runtime" / "grammars"
    for name in ("patch.gbnf", "plan.gbnf", "fix.gbnf", "explain.gbnf", "architecture.gbnf"):
        assert (gdir / name).is_file(), f"{name} should exist in runtime/grammars"


def test_prompt_files_present():
    pdir = Path(__file__).resolve().parents[2] / "runtime" / "prompts"
    for name in ("hierarchical-chat.txt", "verify.txt"):
        assert (pdir / name).is_file(), f"{name} should exist in runtime/prompts"


# ---------------------------------------------------------------------------
# Plan -> Act creation behavior (edit steps can create brand-new files)
# ---------------------------------------------------------------------------

def test_creation_request_detection():
    from forge_server.plans import _is_creation_request

    assert _is_creation_request("Create a new repository named 'tic tac toe'")
    assert _is_creation_request("write an HTML page for a game")
    assert _is_creation_request("generate a Python module for parsing")
    assert _is_creation_request("scaffold a react component")
    # Editing existing code is NOT creation
    assert not _is_creation_request("Fix the null pointer in UserService")
    assert not _is_creation_request("Add validation to the greet method")


def test_multi_to_act_shape():
    from forge_server.plans import _multi_to_act

    from core.patching.parser import FilePatch, MultiPatch, Operation

    multi = MultiPatch(
        patches=[FilePatch(path="src/a.py", operations=[
            Operation(type="replace", start_line=1, end_line=2, content="x = 1"),
        ])],
        creates={"index.html": "<html></html>"},
        message="demo",
    )
    shaped = _multi_to_act(multi)
    assert shaped["kind"] == "multi"
    assert shaped["creates"] == {"index.html": "<html></html>"}
    assert shaped["files"][0]["path"] == "src/a.py"
    assert shaped["message"] == "demo"


def test_parse_multi_creation_shape():
    from core.patching.parser import parse_multi

    raw = '{"message": "made page", "creates": {"tic/index.html": "<!DOCTYPE html>"}}'
    multi = parse_multi(raw)
    assert multi.creates == {"tic/index.html": "<!DOCTYPE html>"}
    assert multi.message == "made page"
    assert not multi.patches


def test_create_grammar_present():
    gdir = Path(__file__).resolve().parents[2] / "runtime" / "grammars"
    assert (gdir / "create.gbnf").is_file(), "create.gbnf should exist in runtime/grammars"


def test_placeholder_detection():
    from forge_server.plans import _looks_placeholder

    assert _looks_placeholder("public class X { // Complete code here }")
    assert _looks_placeholder("x")  # too short
    assert _looks_placeholder("def stub():\n    TODO")
    assert not _looks_placeholder("<!DOCTYPE html>\n<html><body><h1>Tic</h1></body></html>")
    assert not _looks_placeholder(
        "const board = [];\nlet currentPlayer = 'X';\nfunction checkWin() {\n  return false;\n}\n"
    )


def test_inference_client_accepts_schema():
    """The client's chat signature must accept a JSON schema (json_schema response_format)."""
    import inspect

    from core.inference.client import InferenceClient

    sig = inspect.signature(InferenceClient.chat)
    assert "schema" in sig.parameters
    sig_stream = inspect.signature(InferenceClient.chat_stream)
    assert "schema" in sig_stream.parameters


def test_start_forge_sets_pythonpath():
    """The stack launcher must expose both import roots to the API process."""
    from pathlib import Path

    src = (Path(__file__).resolve().parents[2] / "runtime" / "scripts" / "start_forge.py").read_text(
        encoding="utf-8"
    )
    assert "PYTHONPATH" in src
    assert 'str(ROOT / "apps" / "server")' in src  # forge_server import root
    assert "env=FORGE_ENV" in src  # the uvicorn subprocess inherits it

