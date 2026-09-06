"""Semantic chunking.

Splits a source file into chunks that respect semantic boundaries
(class, function, method, interface, ...) instead of fixed-size windows,
so each chunk is independently retrievable and cheap to embed in a prompt.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

MAX_CHUNK_LINES = 250
MIN_CHUNK_LINES = 3

# Flow-control keywords that should never be treated as symbol starts.
_CONTROL = {
    "if", "for", "while", "switch", "catch", "else", "return", "do",
    "try", "finally", "case", "default", "match", "where", "when",
    "async", "await", "yield", "with", "select", "from",
}

# Top-level declaration starters per language.
_TOP_LEVEL: dict[str, str] = {
    "python": r"^\s*(class|def|async\s+def)\s+\w+",
    "javascript": r"^(export\s+)?(default\s+)?(class|function|async\s+function)\s+\w+|^(export\s+)?(const|let|var)\s+\w+\s*=\s*(\()",
    "typescript": r"^(export\s+)?(default\s+)?(class|function|async\s+function|interface|enum|type|namespace|module)\s+\w+|^(export\s+)?(const|let|var)\s+\w+\s*=\s*(\()",
    "java": r"^\s*(public|private|protected|static|final|abstract|synchronized|default)?\s*(class|interface|enum|record|@interface)\s+\w+",
    "kotlin": r"^\s*(public|private|protected|internal|abstract|final|data|sealed|open)?\s*(class|interface|enum|object|fun)\s+\w+",
    "csharp": r"^\s*(public|private|protected|internal|static|sealed|abstract|partial|readonly|async|virtual|override|record|struct)\s*\w+",
    "c": r"^\s*(static\s+|inline\s+|extern\s+)?[\w<>*:&\s]+\s+\w+\s*\([^;]*\)\s*\{|^\s*(struct|enum|union|typedef|#define|#if|#endif|#pragma)\b",
    "cpp": r"^\s*(template\s*<.*>\s*)?(static\s+|inline\s+|virtual\s+|constexpr\s+)?[\w<>*:&\s]+\s+\w+\s*\([^;]*\)\s*\{|^\s*(class|struct|enum|namespace|typedef|using|#define|#pragma)\b",
    "go": r"^\s*(func|type|struct|interface|const|var|package|import)\b",
    "rust": r"^\s*(pub\s+)?(fn|struct|enum|trait|impl|mod|const|static|use|macro_rules)\b",
    "ruby": r"^\s*(class|module|def)\s+\w+",
    "php": r"^\s*(public|private|protected|static|function|class|interface|trait|namespace)\b",
    "swift": r"^\s*(public|private|internal|fileprivate|open|static|final|class|struct|enum|protocol|func|extension)\s+\w+",
    "scala": r"^\s*(class|object|trait|case\s+class|def|abstract\s+class)\b",
    "groovy": r"^\s*(class|interface|def|static|void|public|private|protected)\b|^\s*[\w<>,.?\[\] ]+\s+\w+\s*\(",
    "lua": r"^\s*(local\s+)?function\s+\w+|^\s*function\s+",
}

# Member/method starters that require indentation.
_MEMBER: dict[str, str] = {
    "python": r"^\s+(async\s+)?def\s+\w+",
    "javascript": r"^\s{2,}(async\s+)?\w+\s*\([^)]*\)\s*\{|^\s{2,}static\s+\w+\s*\([^)]*\)\s*\{",
    "typescript": r"^\s{2,}(public|private|protected|readonly|static|async|get|set|async)?\s*\w+\s*(<[^>]*>)?\([^)]*\)\s*\{",
    "java": r"^\s{2,}(public|private|protected|static|final|synchronized|native|abstract)?\s*[\w<>,.?\[\]\s]+\s+\w+\s*\([^)]*\)[^;]*\{",
    "kotlin": r"^\s{2,}(fun|override|operator|infix|suspend|tailrec|external)\s+\w+",
    "csharp": r"^\s{2,}(public|private|protected|internal|static|async|virtual|override|void|int|string|bool|var|Task)\s+\w+\s*\(",
    "c": r"^\s+(static\s+|inline\s+)?[\w<>*&\s]+\s+\w+\s*\([^;]*\)\s*\{",
    "cpp": r"^\s+(virtual\s+|static\s+|inline\s+|constexpr\s+)?[\w<>*&:\s]+\s+\w+\s*\([^;]*\)\s*\{",
    "go": r"^\s+func\s+\(\s*\w+\s+\*?[\w.]+\s*\)\s+\w+\([^)]*\)",
    "rust": r"^\s+(pub\s+)?fn\s+\w+",
    "ruby": r"^\s{2,}def\s+\w+",
    "php": r"^\s{2,}(public|private|protected|static|function)\s+\w+",
    "swift": r"^\s{2,}(func|init|subscript|deinit)\s+\w*",
    "groovy": r"^\s+(def\s+|void\s+|[\w<>]+[\w<>,.?\[\] ]*\]?\s+\w+\s*\()",
}

_compiled_top_cache: dict[str, list[re.Pattern]] = {}
_compiled_member_cache: dict[str, list[re.Pattern]] = {}


def _patterns(cache: dict, patterns: dict, language: str) -> list[re.Pattern]:
    if language not in cache:
        expr = patterns.get(language)
        cache[language] = [re.compile(expr)] if expr else []
    return cache[language]


def _is_symbol_start(line: str, language: str) -> bool:
    stripped = line.lstrip(" \t")
    if not stripped:
        return False
    if stripped.startswith(("#", "//", "/*", "*", "\"", "'", "<!--", "@")):
        return False
    first_word = re.split(r"\W+", stripped, 1)[0].lower() if stripped else ""
    if first_word in _CONTROL:
        return False
    indent = len(line) - len(line.lstrip(" \t"))
    if indent == 0 or _patterns(_compiled_top_cache, _TOP_LEVEL, language):
        for pat in _patterns(_compiled_top_cache, _TOP_LEVEL, language):
            if pat.match(line):
                return True
    if indent >= 2:
        for pat in _patterns(_compiled_member_cache, _MEMBER, language):
            if pat.match(line):
                return True
    return False


def _kind_of(line: str, language: str) -> str:
    low = line.lower()
    stripped = line.lstrip(" \t")
    if re.search(r"\b(class|interface|struct|enum|trait|record|namespace|module)\b", low):
        return "declaration"
    # explicit function keywords (Python/Go/Rust/Ruby/...)
    if re.search(r"\b(def|fn|func|function|fun)\b", low):
        return "function"
    # C-like method declaration:  visibility returnType name(args) {
    if re.match(r"(?:\w+\s+)*\w+\s*\([^()]*\)\s*\{$", stripped) and re.search(r"\(", stripped):
        return "function"
    if low.startswith(("import ", "from ", "import(", "require(", "using ", "#include")):
        return "imports"
    return "code"


def _split_long(start: int, end: int, lines: list[str], language: str) -> list[tuple[int, int]]:
    """Split an oversized segment at blank-line boundaries."""
    if end - start + 1 <= MAX_CHUNK_LINES:
        return [(start, end)]
    out: list[tuple[int, int]] = []
    seg_start = start
    for i in range(start, end + 1):
        if i - seg_start + 1 > MAX_CHUNK_LINES:
            out.append((seg_start, i - 1))
            seg_start = i
        elif lines[i] == "" and i - seg_start + 1 >= MIN_CHUNK_LINES:
            out.append((seg_start, i))
            seg_start = i + 1
    if seg_start <= end:
        if out and end - seg_start + 1 < MIN_CHUNK_LINES:
            tail_start, _ = out[-1]
            out[-1] = (tail_start, end)
        else:
            out.append((seg_start, end))
    return out


@dataclass(frozen=True)
class Chunk:
    path: str
    language: str
    kind: str
    content: str
    start_line: int
    end_line: int

    @property
    def line_count(self) -> int:
        return self.end_line - self.start_line + 1


def chunk_content(content: str, path: str, language: str) -> list[Chunk]:
    """Split ``content`` into semantic chunks (1-based inclusive lines)."""
    lines = content.splitlines()
    if not lines or not any(ln.strip() for ln in lines):
        return []  # empty or whitespace-only files yield no chunks
    chunks: list[Chunk] = []
    boundaries = [0]
    for i in range(1, len(lines)):
        if _is_symbol_start(lines[i], language):
            boundaries.append(i)

    segments: list[tuple[int, int]] = []
    for idx, seg_start in enumerate(boundaries):
        seg_end = boundaries[idx + 1] - 1 if idx + 1 < len(boundaries) else len(lines) - 1
        segments.extend(_split_long(seg_start, seg_end, lines, language))

    for start, end in segments:
        text = "\n".join(lines[start : end + 1])
        chunks.append(
            Chunk(
                path=path,
                language=language,
                kind=_kind_of(lines[start], language),
                content=text,
                start_line=start + 1,
                end_line=end + 1,
            )
        )
    return chunks
