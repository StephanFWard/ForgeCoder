"""Filesystem scanning.

Walks a workspace one file at a time (never loads the repository into
memory), skips ignored directories/generated artifacts, and returns a
``FileRecord`` per supported source file.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

from core.indexer.parser import detect_language

# Directories that are never indexed.
IGNORE_DIRS = {
    ".git",
    ".hg",
    ".svn",
    ".idea",
    ".vscode",
    "node_modules",
    "dist",
    "build",
    "out",
    ".next",
    ".nuxt",
    "target",
    ".venv",
    "venv",
    "env",
    "__pycache__",
    ".pytest_cache",
    ".ruff_cache",
    ".mypy_cache",
    ".cache",
    "bin",
    "obj",
    "site-packages",
    "vendor",
    "lib",
    "libs",
    ".elk",
    "coverage",
    ".coverage",
}

# File extensions that are never useful to an LLM.
IGNORE_EXTENSIONS = {
    ".png", ".jpg", ".jpeg", ".gif", ".bmp", ".ico", ".tiff", ".webp",
    ".pdf", ".doc", ".docx", ".xls", ".xlsx", ".ppt", ".pptx",
    ".exe", ".dll", ".so", ".dylib", ".sys", ".msi", ".class", ".jar",
    ".pyc", ".pyo", ".pyd", ".o", ".obj", ".a", ".lib",
    ".woff", ".woff2", ".ttf", ".otf", ".eot",
    ".zip", ".tar", ".gz", ".bz2", ".7z", ".rar", ".xz",
    ".lock", ".min.js", ".min.css", ".map",
    ".db", ".sqlite", ".db-wal", ".db-shm",
    ".log", ".tmp", ".bak", ".swp",
}

# Substring indicators of generated files.
GENERATED_MARKERS = ("generated", r"\.gen\.", r"_pb2\.py", "autogen")


@dataclass(frozen=True)
class FileRecord:
    """Metadata for one indexed file."""

    path: str                # POSIX-style, workspace-relative
    absolute: str
    language: str
    size: int
    sha1: str
    modified: float          # seconds since epoch

    @property
    def posix_path(self) -> str:
        return self.path


def _hash_file(path: Path) -> tuple[str, int]:
    h = hashlib.sha1()
    size = 0
    with open(path, "rb") as fh:
        while True:
            block = fh.read(1024 * 256)
            if not block:
                break
            size += len(block)
            h.update(block)
    return h.hexdigest(), size


def is_ignored(path: Path) -> bool:
    """True when a file should be skipped (dir, ext, or generated marker)."""
    parts = path.parts
    if any(part in IGNORE_DIRS for part in parts):
        return True
    if path.suffix.lower() in IGNORE_EXTENSIONS:
        return True
    return False


def iter_files(root: Path | str) -> list[FileRecord]:
    """Walk ``root`` and return FileRecords for all supported, non-ignored files.

    Uses ``os.walk`` (not ``pathlib.Path.walk``) so it works on Python 3.10/3.11.
    """
    import os

    root = Path(root)
    records: list[FileRecord] = []
    for dirpath, dirnames, filenames in os.walk(root):
        dpath = Path(dirpath)
        # Prune ignored/hidden directories in-place so os.walk does not descend.
        dirnames[:] = sorted(
            d for d in dirnames if d not in IGNORE_DIRS and not d.startswith(".")
        )
        for name in sorted(filenames):
            fpath = dpath / name
            if is_ignored(fpath):
                continue
            language = detect_language(fpath.name)
            if language is None:
                continue
            try:
                sha1, size = _hash_file(fpath)
                stat = fpath.stat()
            except OSError:
                continue
            rel = fpath.relative_to(root).as_posix()
            records.append(
                FileRecord(
                    path=rel,
                    absolute=str(fpath.resolve()),
                    language=language,
                    size=size,
                    sha1=sha1,
                    modified=stat.st_mtime,
                )
            )
    return records
