"""SQLite + FTS5 repository index.

The index is intentionally incremental: files are hashed, only changed files
are re-chunked, and the whole repository is never held in memory.
"""
from __future__ import annotations

import argparse
import sqlite3
import time
from pathlib import Path

from core.indexer.chunker import Chunk, chunk_content
from core.indexer.scanner import FileRecord, iter_files
from core.indexer.symbols import Symbol, extract_symbols

SCHEMA = """
CREATE TABLE IF NOT EXISTS files (
    id INTEGER PRIMARY KEY,
    path TEXT UNIQUE NOT NULL,
    root TEXT,
    language TEXT,
    size INTEGER,
    hash TEXT,
    modified INTEGER
);

CREATE TABLE IF NOT EXISTS symbols (
    id INTEGER PRIMARY KEY,
    file_id INTEGER NOT NULL,
    name TEXT NOT NULL,
    kind TEXT,
    start_line INTEGER,
    end_line INTEGER,
    FOREIGN KEY(file_id) REFERENCES files(id)
);
CREATE INDEX IF NOT EXISTS idx_symbols_name ON symbols(name);
CREATE INDEX IF NOT EXISTS idx_symbols_file ON symbols(file_id);

CREATE TABLE IF NOT EXISTS chunks (
    id INTEGER PRIMARY KEY,
    file_id INTEGER NOT NULL,
    symbol_id INTEGER,
    path TEXT,
    language TEXT,
    content TEXT NOT NULL,
    start_line INTEGER,
    end_line INTEGER,
    FOREIGN KEY(file_id) REFERENCES files(id)
);
CREATE INDEX IF NOT EXISTS idx_chunks_file ON chunks(file_id);

CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
    content,
    path,
    language,
    content='chunks',
    content_rowid='id'
);

CREATE TRIGGER IF NOT EXISTS chunks_ai AFTER INSERT ON chunks BEGIN
    INSERT INTO chunks_fts(rowid, content, path, language)
    VALUES (new.id, new.content, new.path, new.language);
END;

CREATE TRIGGER IF NOT EXISTS chunks_ad AFTER DELETE ON chunks BEGIN
    INSERT INTO chunks_fts(chunks_fts, rowid, content, path, language)
    VALUES ('delete', old.id, old.content, old.path, old.language);
END;

CREATE TRIGGER IF NOT EXISTS chunks_au AFTER UPDATE ON chunks BEGIN
    INSERT INTO chunks_fts(chunks_fts, rowid, content, path, language)
    VALUES ('delete', old.id, old.content, old.path, old.language);
    INSERT INTO chunks_fts(rowid, content, path, language)
    VALUES (new.id, new.content, new.path, new.language);
END;
"""

# FTS5 MATCH can throw syntax errors on arbitrary text; we always emit safe
# quoted-term queries, so tokenize first.
_WORD = r"[A-Za-z0-9_*][A-Za-z0-9_.\-*]*"


class CodeIndex:
    """Owns the SQLite database and exposes incremental indexing + search."""

    def __init__(self, db_path: str | Path):
        self.db_path = Path(db_path)
        self._conn: sqlite3.Connection | None = None

    # ---------------------------------------------------------------- lifecycle
    def connect(self) -> None:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        # FastAPI runs sync handlers in a threadpool while the lifespan may
        # connect on the event loop thread; allow cross-thread use. Access is
        # internally serialized by our per-request usage patterns.
        self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._conn.executescript(SCHEMA)
        self._conn.commit()

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    @property
    def conn(self) -> sqlite3.Connection:
        if self._conn is None:
            self.connect()
        assert self._conn is not None
        return self._conn

    def is_ready(self) -> bool:
        if self._conn is None:
            return False
        try:
            self.conn.execute("SELECT COUNT(*) FROM files").fetchone()
            return True
        except sqlite3.Error:
            return False

    # ---------------------------------------------------------------- indexing
    def index_workspace(self, root: str | Path, *, prune: bool = True) -> dict[str, int]:
        """Index a workspace root; returns a small stats dict."""
        root = Path(root)
        root_key = str(root.resolve()).replace("\\", "/").rstrip("/")
        conn = self.conn
        started = time.monotonic()
        seen: set[str] = set()
        added = updated = unchanged = pruned = 0

        for rec in iter_files(root):
            seen.add(rec.path)
            existing = conn.execute(
                "SELECT id, hash, modified FROM files WHERE path = ? AND root = ?",
                (rec.path, root_key),
            ).fetchone()
            if existing and existing["hash"] == rec.sha1 and existing["modified"] == int(rec.modified):
                unchanged += 1
                continue
            self._replace_file(rec, existing["id"] if existing else None, root_key)
            if existing:
                updated += 1
            else:
                added += 1
            conn.commit()

        if prune:
            stale_sql = "SELECT path, root FROM files WHERE root = ?" if not seen else (
                "SELECT path, root FROM files WHERE root = ? AND path NOT IN ("
                + ",".join("?" * len(seen))
                + ")"
            )
            params: list[object] = [root_key]
            if seen:
                params.extend(sorted(seen))
            stale = [row["path"] for row in conn.execute(stale_sql, params).fetchall()]
            for path in stale:
                self.remove_file(path, root=root_key)
                pruned += 1
            conn.commit()

        return {
            "added": added,
            "updated": updated,
            "unchanged": unchanged,
            "pruned": pruned,
            "seconds": round(time.monotonic() - started, 3),
        }

    def _replace_file(self, rec: FileRecord, file_id: int | None, root: str) -> None:
        conn = self.conn
        if file_id is None:
            cur = conn.execute(
                "INSERT INTO files (path, root, language, size, hash, modified) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (rec.path, root, rec.language, rec.size, rec.sha1, int(rec.modified)),
            )
            file_id = int(cur.lastrowid)
        else:
            conn.execute(
                "UPDATE files SET root = ?, language = ?, size = ?, hash = ?, modified = ? "
                "WHERE id = ?",
                (root, rec.language, rec.size, rec.sha1, int(rec.modified), file_id),
            )

        conn.execute("DELETE FROM symbols WHERE file_id = ?", (file_id,))
        conn.execute("DELETE FROM chunks WHERE file_id = ?", (file_id,))

        try:
            text = Path(rec.absolute).read_text(encoding="utf-8", errors="replace")
        except OSError:
            return
        if not text:
            return

        symbols: list[Symbol] = extract_symbols(text, rec.language)
        symbol_rows: list[tuple[int, str, int, int]] = []
        for sym in symbols:
            cur = conn.execute(
                "INSERT INTO symbols (file_id, name, kind, start_line, end_line) VALUES (?, ?, ?, ?, ?)",
                (file_id, sym.name, sym.kind, sym.start_line, sym.end_line),
            )
            symbol_rows.append((int(cur.lastrowid), sym.name, sym.start_line, sym.end_line))
            if len(symbol_rows) > 50_000:  # pathological-file guard
                break

        chunks: list[Chunk] = chunk_content(text, rec.path, rec.language)
        for chunk in chunks:
            symbol_match: int | None = None
            for sym_id, name, s_line, e_line in symbol_rows:
                if chunk.start_line >= s_line and chunk.end_line <= e_line:
                    symbol_match = sym_id
                    break
            conn.execute(
                "INSERT INTO chunks (file_id, symbol_id, path, language, content, start_line, end_line) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (file_id, symbol_match, chunk.path, chunk.language, chunk.content,
                 chunk.start_line, chunk.end_line),
            )

    def remove_file(self, path: str, root: str | None = None) -> None:
        conn = self.conn
        if root is not None:
            row = conn.execute(
                "SELECT id FROM files WHERE path = ? AND root = ?", (path, root)
            ).fetchone()
        else:
            row = conn.execute("SELECT id FROM files WHERE path = ?", (path,)).fetchone()
        if row is None:
            return
        file_id = row["id"]
        conn.execute("DELETE FROM symbols WHERE file_id = ?", (file_id,))
        conn.execute("DELETE FROM chunks WHERE file_id = ?", (file_id,))
        conn.execute("DELETE FROM files WHERE id = ?", (file_id,))

    # ------------------------------------------------------------------ search
    def search(self, query: str, *, limit: int = 20, workspace: str | None = None) -> list[dict]:
        """FTS5 search over chunk text/path/language; returns dict rows."""
        tokens = self._fts_tokens(query)
        if not tokens:
            return []
        match_expr = " AND ".join(f'"{t}"' for t in tokens)
        sql = (
            "SELECT c.id, c.path, c.language, c.content, c.start_line, c.end_line, "
            "       f.modified, f.size, bm25(chunks_fts) AS rank "
            "FROM chunks c "
            "JOIN chunks_fts ON chunks_fts.rowid = c.id "
            "JOIN files f ON f.id = c.file_id "
            "WHERE chunks_fts MATCH ? "
        )
        params: list[object] = [match_expr]
        if workspace:
            sql += "AND f.root = ? "
            params.append(workspace.replace("\\", "/").rstrip("/"))
        sql += "ORDER BY rank LIMIT ?"
        params.append(limit)
        try:
            rows = self.conn.execute(sql, params).fetchall()
        except sqlite3.Error:
            return []
        return [dict(r) for r in rows]

    def _fts_tokens(self, query: str) -> list[str]:
        import re

        tokens = re.findall(_WORD, query)
        return [t for t in tokens if len(t) >= 2][:12]

    def symbols_like(self, name: str, *, limit: int = 20) -> list[dict]:
        import re

        tokens = re.findall(r"\w+", name)
        if not tokens:
            return []
        pattern = "%" + "%".join(tokens) + "%"
        rows = self.conn.execute(
            "SELECT s.name, s.kind, s.start_line, s.end_line, f.path "
            "FROM symbols s JOIN files f ON f.id = s.file_id "
            "WHERE s.name LIKE ? COLLATE NOCASE LIMIT ?",
            (pattern, limit),
        ).fetchall()
        return [dict(r) for r in rows]

    # ------------------------------------------------------------------ stats
    def stats(self) -> dict:
        try:
            files = self.conn.execute("SELECT COUNT(*) FROM files").fetchone()[0]
            chunks = self.conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
            symbols = self.conn.execute("SELECT COUNT(*) FROM symbols").fetchone()[0]
        except sqlite3.Error:
            return {"files": 0, "chunks": 0, "symbols": 0}
        return {"files": files, "chunks": chunks, "symbols": symbols}


def cli() -> None:
    parser = argparse.ArgumentParser(description="ForgeCoder indexer")
    parser.add_argument("--workspace", required=True, help="Workspace/root path to index")
    parser.add_argument("--db", default=None, help="SQLite DB path (default: %%LOCALAPPDATA%%/ForgeCoder/forge.db)")
    parser.add_argument("--no-prune", action="store_true", help="Keep stale rows instead of pruning")
    args = parser.parse_args()

    import os

    db = args.db
    if not db:
        local = os.environ.get("LOCALAPPDATA")
        db = str(Path(local or Path.home()).joinpath("ForgeCoder", "forge.db"))

    index = CodeIndex(db)
    index.connect()
    stats = index.index_workspace(args.workspace, prune=not args.no_prune)
    print(f"Indexed {args.workspace}: {stats}")
    index.close()


if __name__ == "__main__":
    cli()
