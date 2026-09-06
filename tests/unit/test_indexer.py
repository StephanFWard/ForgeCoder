"""End-to-end indexer tests against a real SQLite database in a temp dir."""
from pathlib import Path

from core.indexer.index import CodeIndex


def _write_tree(root: Path) -> Path:
    src = root / "src"
    src.mkdir(parents=True)
    (src / "UserService.java").write_text(
        "package com.example;\n\n"
        "public class UserService {\n"
        "    public String greet(String name) { return \"hi \" + name; }\n"
        "}\n",
        encoding="utf-8",
    )
    (root / "README.md").write_text("# notes\n", encoding="utf-8")
    # noise that must be ignored
    nm = root / "node_modules" / "pkg"
    nm.mkdir(parents=True)
    (nm / "index.js").write_text("let x = 1;\n", encoding="utf-8")
    dist = root / "dist"
    dist.mkdir(exist_ok=True)
    (dist / "bundle.js").write_text("console.log(1)\n", encoding="utf-8")
    return root


def test_index_roundtrip(tmp_path):
    root = _write_tree(Path(tmp_path))
    db = Path(tmp_path) / "forge.db"
    index = CodeIndex(db)
    index.connect()

    stats = index.index_workspace(root)
    assert stats["added"] == 2   # UserService.java + README.md (markdown is indexed)
    assert index.stats()["files"] == 2
    assert index.stats()["symbols"] >= 2  # class + method

    found = index.search("UserService", limit=5)
    assert found, "FTS5 should find the class chunk"
    assert found[0]["path"] == "src/UserService.java"

    # content is searchable too
    assert index.search("greet", limit=5)

    # re-index is idempotent (unchanged)
    stats2 = index.index_workspace(root)
    assert stats2["unchanged"] == 2
    assert stats2["added"] == 0

    # prune (after deleting a file)
    (root / "src" / "UserService.java").unlink()
    stats3 = index.index_workspace(root)
    assert stats3["pruned"] == 1
    assert index.stats()["files"] == 1  # README.md remains
    index.close()


def test_search_returns_empty_for_noise(tmp_path):
    root = _write_tree(Path(tmp_path))
    index = CodeIndex(Path(tmp_path) / "db.sqlite")
    index.connect()
    index.index_workspace(root, prune=False)
    assert index.search("console.log") == []  # dist excluded => no results
    index.close()


def test_symbol_lookup(tmp_path):
    root = _write_tree(Path(tmp_path))
    index = CodeIndex(Path(tmp_path) / "db.sqlite")
    index.connect()
    index.index_workspace(root)
    symbols = index.symbols_like("UserService")
    assert symbols
    assert symbols[0]["name"] == "UserService"
    index.close()
