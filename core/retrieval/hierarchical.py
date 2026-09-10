"""RNP-inspired hierarchical context assembly.

Builds a part-whole tree (workspace → file → symbol → chunk) instead of a flat
bag of chunks. Inspired by Fisher & Rao (2023) — Recursive Neural Programs,
PMC10637337: compositional hierarchies where primitives are recursively reused
at different reference frames.

The hierarchy gives the 1.5B model a structured view: it sees which file a
snippet came from, which function within that file, and the fine-grained chunk
inside the function — mirroring how RNPs model images as trees of programs
within nested spatial reference frames.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from core.retrieval.budget import estimate_tokens, fit_to_budget, truncate_to_tokens
from core.retrieval.context import _selection_snippet, load_prompt
from core.retrieval.search import SearchEngine

# ---------------------------------------------------------------------------
# Tree node — one unit in the retrieval hierarchy
# ---------------------------------------------------------------------------

@dataclass
class TreeNode:
    """One node in the retrieval tree.

    kind: one of ``workspace`` | ``file`` | ``symbol`` | ``chunk``
    """
    kind: str
    label: str                      # filename, symbol name, or chunk header
    content: str = ""
    children: list[TreeNode] = field(default_factory=list)
    tokens: int = 0
    score: float = 0.0
    # file/path metadata (meaningful for file/symbol/chunk nodes)
    path: str = ""
    start_line: int = 0
    end_line: int = 0


@dataclass
class HierarchicalContext:
    """A tree-structured context assembled for one inference turn."""

    system: str
    request: str
    root: TreeNode
    total_tokens: int = 0

    def as_message_text(self) -> str:
        """Render the tree to a single prompt string (depth-first)."""
        parts: list[str] = []
        _render(self.root, parts, depth=0)
        return "\n\n".join(parts)

    def tree_summary(self) -> dict:
        """Return a lightweight summary of the context tree for logging."""
        return _tree_summary(self.root)


def _render(node: TreeNode, parts: list[str], depth: int) -> None:
    indent = "  " * depth
    if node.kind == "workspace":
        parts.append(f"{indent}# Repository Context")
        for child in node.children:
            _render(child, parts, depth + 1)
    elif node.kind == "file":
        parts.append(f"{indent}## {node.label}")
        if node.content:
            parts.append(f"{indent}```")
            parts.append(truncate_to_tokens(node.content, max(200, min(400, node.tokens))))
            parts.append(f"{indent}```")
        for child in node.children:
            _render(child, parts, depth + 1)
    elif node.kind == "symbol":
        sig = node.content.splitlines()[0][:80] if node.content else node.label
        parts.append(f"{indent}### {node.label} ({node.path}:{node.start_line}-{node.end_line})")
        parts.append(f"{indent}> {sig}")
        for child in node.children:
            _render(child, parts, depth + 1)
    elif node.kind == "chunk":
        if node.content:
            header = f"{indent}- {node.label} [{node.path}:{node.start_line}-{node.end_line}]"
            parts.append(header)
            body = truncate_to_tokens(node.content, 160)
            for line in body.splitlines():
                parts.append(f"{indent}  {line}")


def _tree_summary(node: TreeNode) -> dict:
    summary: dict[str, Any] = {
        "kind": node.kind,
        "label": node.label,
        "tokens": node.tokens,
        "children": [],
    }
    for child in node.children:
        summary["children"].append(_tree_summary(child))
    return summary


def count_subtree(node: TreeNode) -> int:
    """Total token count of a node including its descendants."""
    total = node.tokens
    for child in node.children:
        total += count_subtree(child)
    return total


# ---------------------------------------------------------------------------
# Hierarchical builder — Phase 1/2/3 retrieval + budget pruning
# ---------------------------------------------------------------------------

class HierarchicalContextBuilder:
    """RNP-inspired hierarchical context assembly.

    Phase 1 -- Coarse file retrieval: which files are relevant to the query?
    Phase 2 -- Symbol expansion: for each file, which symbols/functions match?
    Phase 3 -- Chunk refinement: for each symbol, which chunks are relevant?
    Phase 4 -- Budget pruning: walk the tree, truncating leaves under budget
        while keeping the hierarchy intact (workspace dominates -> file ->
        symbol -> chunk).

    Falls back gracefully when the index is not available.
    """

    def __init__(self, index=None, search_engine: SearchEngine | None = None):
        self.index = index
        self.search_engine = search_engine
        if search_engine is None and index is not None:
            self.search_engine = SearchEngine(index)

    def build(self, message: str, *, workspace: str | None = None,
              file: str | None = None, selection: tuple[int, int] | None = None,
              budget: int = 4096, behavior: str = "chat") -> HierarchicalContext:
        """Assemble a hierarchical context tree for one user message."""
        system = load_prompt(behavior)
        used = estimate_tokens(system) + estimate_tokens(message)

        root = TreeNode(kind="workspace", label="repository", path="")

        # --- Phase 1: file-level retrieval -----------------------------------
        if self.search_engine and message.strip():
            pool = self._retrieve_files(message, workspace=workspace)
            remaining = budget - used - 400
            fitted = fit_to_budget(pool, remaining, min_chunks=1)
            for f in fitted:
                node = TreeNode(
                    kind="file",
                    label=f["path"],
                    path=f["path"],
                    content=f["content"],
                    tokens=estimate_tokens(f["content"]),
                    score=f.get("_score", 0),
                )
                root.children.append(node)
                used += node.tokens
                self._expand_file(node, message, budget - used)
                used = count_subtree(node) + used

        # --- Fallback: current-file snippet ----------------------------------
        if not root.children and file and workspace:
            snippet = _selection_snippet(Path(workspace) / file, selection)
            if snippet:
                text = f"### {file}\n```\n{snippet}\n```"
                text = truncate_to_tokens(text, max(budget - used - 200, 100))
                root.children.append(TreeNode(
                    kind="file", label=file, path=file,
                    content=text, tokens=estimate_tokens(text),
                ))

        return HierarchicalContext(system=system, request=message,
                                   root=root, total_tokens=count_subtree(root))

    # -- expansion helpers -------------------------------------------------

    def _expand_file(self, file_node: TreeNode, message: str, budget: int) -> None:
        if budget <= 0:
            return
        symbols = self._retrieve_symbols(file_node.path, message)
        sym_budget = max(budget // 3, 50)
        fitted_syms = fit_to_budget(symbols, sym_budget, min_chunks=0)
        for s in fitted_syms:
            s_node = TreeNode(
                kind="symbol",
                label=s.get("name", s["path"].rsplit("/", 1)[-1]),
                path=s.get("path", file_node.path),
                start_line=s.get("start_line", 0),
                end_line=s.get("end_line", 0),
                content=s.get("content", ""),
                tokens=estimate_tokens(s.get("content", "")),
                score=s.get("_score", 0),
            )
            file_node.children.append(s_node)
            self._expand_symbol(s_node, message, max(budget // 6, 30))

    def _expand_symbol(self, sym_node: TreeNode, message: str, budget: int) -> None:
        if budget <= 0:
            return
        chunks = self._retrieve_chunks(sym_node.path, message)
        fitted = fit_to_budget(chunks, budget, min_chunks=0)
        for c in fitted:
            c_node = TreeNode(
                kind="chunk",
                label=f"{c['path']}:{c['start_line']}-{c['end_line']}",
                path=c["path"],
                start_line=c["start_line"],
                end_line=c["end_line"],
                content=c.get("content", ""),
                tokens=estimate_tokens(c.get("content", "")),
                score=c.get("_score", 0),
            )
            sym_node.children.append(c_node)

    # -- retrieval helpers -------------------------------------------------

    def _retrieve_files(self, message: str, *, workspace: str | None = None) -> list[dict]:
        if not self.search_engine:
            return []
        raw = self.search_engine.search(message, workspace=workspace, limit=16)
        seen: dict[str, dict] = {}
        for r in raw:
            short = truncate_to_tokens(r.content, 250)
            if r.path not in seen or r.score > seen[r.path].get("_score", 0):
                seen[r.path] = {"path": r.path, "content": short, "_score": r.score}
        results = sorted(seen.values(), key=lambda x: -x["_score"])
        return results

    def _retrieve_symbols(self, file_path: str, query: str) -> list[dict]:
        if not self.index:
            return []
        try:
            file_name = file_path.rsplit("/", 1)[-1]
            symbols = self.index.symbols_like(file_name, limit=12)
            candidates: list[dict] = []
            for sym in symbols:
                path = sym.get("path", "")
                if path == file_path or path.endswith(file_path):
                    content = _symbol_chunk_content(self.index, file_path, sym)
                    candidates.append({
                        "path": file_path,
                        "name": sym.get("name", ""),
                        "start_line": sym.get("start_line", 0),
                        "end_line": sym.get("end_line", 0),
                        "content": truncate_to_tokens(content, 200),
                        "_score": 0,
                    })
            qt = set(t.lower() for t in query.split() if len(t) >= 2)
            for c in candidates:
                name_hits = sum(1 for t in qt if t in c["name"].lower())
                content_hits = sum(1 for t in qt if t in c["content"].lower())
                c["_score"] = name_hits * 12 + content_hits * 6
            candidates.sort(key=lambda x: -x["_score"])
            return candidates[:5]
        except Exception:
            return []

    def _retrieve_chunks(self, file_path: str, query: str) -> list[dict]:
        if not self.search_engine:
            return []
        raw = self.search_engine.search(query, workspace=None, limit=12)
        results: list[dict] = []
        for r in raw:
            if r.path == file_path:
                results.append({
                    "path": r.path,
                    "content": truncate_to_tokens(r.content, 250),
                    "start_line": r.start_line,
                    "end_line": r.end_line,
                    "_score": r.score,
                })
        return results


def _symbol_chunk_content(index, file_path: str, sym: dict) -> str:
    if index is None:
        return ""
    try:
        conn = index._conn
        if conn is None:
            return ""
        rows = conn.execute(
            """SELECT c.content, c.start_line, c.end_line
               FROM chunks c
               JOIN files f ON f.id = c.file_id
               WHERE f.path = ? AND c.start_line >= ? AND c.end_line <= ?
               ORDER BY c.start_line LIMIT 5""",
            (file_path.replace("\\", "/").rstrip("/"),
             sym.get("start_line", 0),
             sym.get("end_line", 999999)),
        ).fetchall()
        return "\n---\n".join(r["content"] for r in rows if r["content"])
    except Exception:
        return ""
