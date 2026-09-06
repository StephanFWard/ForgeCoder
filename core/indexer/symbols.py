"""Lightweight symbol extraction (regex-based).

The whole point of symbols is fast, dependency-free retrieval: "what is this
symbol and where does it live?", not full AST fidelity. If tree-sitter gets
added later it can replace this module without touching anything else.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

# (label, kind, pattern-with-one-name-capture)
_SYMBOL_RULES: dict[str, list[tuple[str, str, str]]] = {
    "python": [
        ("class", "class", r"^\s*class\s+(\w+)"),
        ("def", "function", r"^\s*(?:async\s+)?def\s+(\w+)"),
    ],
    "javascript": [
        ("class", "class", r"^\s*(?:export\s+)?(?:default\s+)?class\s+(\w+)"),
        ("function", "function", r"^\s*(?:export\s+)?(?:async\s+)?function\s+(\w+)"),
        ("const", "function", r"^\s*(?:export\s+)?const\s+(\w+)\s*=\s*(?:async\s*)?(?:\(|function)"),
        ("method", "function", r"^\s{2,}(?:async\s+)?(\w+)\s*\([^)]*\)\s*\{"),
    ],
    "typescript": [
        ("class", "class", r"^\s*(?:export\s+)?(?:default\s+)?class\s+(\w+)"),
        ("interface", "interface", r"^\s*(?:export\s+)?interface\s+(\w+)"),
        ("enum", "enum", r"^\s*(?:export\s+)?(?:const\s+)?enum\s+(\w+)"),
        ("type", "type", r"^\s*(?:export\s+)?type\s+(\w+)\s*="),
        ("function", "function", r"^\s*(?:export\s+)?(?:async\s+)?function\s+(\w+)"),
        ("const", "function", r"^\s*(?:export\s+)?const\s+(\w+)\s*=\s*(?:async\s*)?(?:\(|function)"),
        ("method", "function", r"^\s{2,}(?:public|private|protected|static|readonly|get|set)?\s*(?:async\s+)?(\w+)\s*\([^)]*\)\s*\{"),
    ],
    "java": [
        ("class", "class", r"^\s*(?:public|private|protected|static|final|abstract)?\s*class\s+(\w+)"),
        ("interface", "interface", r"^\s*(?:public|private|protected|static|final|abstract)?\s*interface\s+(\w+)"),
        ("enum", "enum", r"^\s*(?:public|private|protected|static|final|abstract)?\s*enum\s+(\w+)"),
        ("record", "class", r"^\s*(?:public|private|protected|static|final|abstract)?\s*record\s+(\w+)"),
        ("method", "function", r"^\s{2,}(?:public|private|protected|static|final|synchronized|native|abstract)?\s*[\w<>,.?\[\]\s]+\s(\w+)\s*\([^)]*\)[^;]*\{"),
    ],
    "kotlin": [
        ("class", "class", r"^\s*(?:public|private|protected|internal|abstract|final|data|sealed|open)?\s*class\s+(\w+)"),
        ("interface", "interface", r"^\s*(?:public|private|protected|internal)?\s*interface\s+(\w+)"),
        ("enum", "enum", r"^\s*enum\s+(?:class\s+)?(\w+)"),
        ("fun", "function", r"^\s*(?:public|private|protected|internal|override|operator|infix|suspend|tailrec|external)?\s*fun\s+(\w+)"),
    ],
    "csharp": [
        ("class", "class", r"^\s*(?:public|private|protected|internal|static|sealed|abstract|partial)?\s*class\s+(\w+)"),
        ("interface", "interface", r"^\s*(?:public|private|protected|internal)?\s*interface\s+(\w+)"),
        ("struct", "struct", r"^\s*(?:public|private|protected|internal|readonly)?\s*struct\s+(\w+)"),
        ("enum", "enum", r"^\s*(?:public|private|protected|internal)?\s*enum\s+(\w+)"),
        ("namespace", "namespace", r"^\s*namespace\s+([\w.]+)"),
        ("method", "function", r"^\s{2,}(?:public|private|protected|internal|static|async|virtual|override)?\s*(?:[\w<>]+|Task|void|int|string|bool|var|double|decimal|DateTime)\s+(\w+)\s*\("),
    ],
    "c": [
        ("struct", "struct", r"^\s*(?:typedef\s+)?struct\s+(\w+)"),
        ("enum", "enum", r"^\s*enum\s+(\w+)"),
        ("function", "function", r"^\s*(?:static\s+|inline\s+|extern\s+)?[\w<>*&\s]+\s(\w+)\s*\([^;]*\)\s*\{"),
    ],
    "cpp": [
        ("class", "class", r"^\s*class\s+(\w+)"),
        ("struct", "struct", r"^\s*struct\s+(\w+)"),
        ("enum", "enum", r"^\s*enum\s+(?:class\s+)?(\w+)"),
        ("namespace", "namespace", r"^\s*namespace\s+(\w+)"),
        ("function", "function", r"^\s*(?:static\s+|inline\s+|virtual\s+|constexpr\s+)?[\w<>*:&\s]+\s(\w+)\s*\([^;]*\)\s*\{"),
    ],
    "go": [
        ("func", "function", r"^\s*func\s+\(\s*\w+\s+\*?[\w.]+\s*\)\s+(\w+)\s*\(|^\s*func\s+(\w+)\s*\("),
        ("type", "type", r"^\s*type\s+(\w+)\s+(?:struct|interface|=\s*[\w<>\[\]]+)"),
    ],
    "rust": [
        ("fn", "function", r"^\s*(?:pub\s+)?(?:unsafe\s+)?(?:async\s+)?fn\s+(\w+)"),
        ("struct", "struct", r"^\s*(?:pub\s+)?struct\s+(\w+)"),
        ("enum", "enum", r"^\s*(?:pub\s+)?enum\s+(\w+)"),
        ("trait", "trait", r"^\s*(?:pub\s+)?trait\s+(\w+)"),
        ("impl", "impl", r"^\s*(?:pub\s+)?impl\s+(\w+)"),
    ],
    "ruby": [
        ("class", "class", r"^\s*class\s+(\w+)"),
        ("module", "module", r"^\s*module\s+(\w+)"),
        ("def", "function", r"^\s*def\s+(\w+)"),
    ],
    "php": [
        ("class", "class", r"^\s*(?:abstract\s+|final\s+)?class\s+(\w+)"),
        ("interface", "interface", r"^\s*interface\s+(\w+)"),
        ("function", "function", r"^\s*(?:public|private|protected|static)?\s*function\s+(\w+)"),
    ],
    "swift": [
        ("class", "class", r"^\s*(?:public|private|internal|open|final)?\s*class\s+(\w+)"),
        ("struct", "struct", r"^\s*(?:public|private|internal)?\s*struct\s+(\w+)"),
        ("enum", "enum", r"^\s*(?:public|private|internal)?\s*enum\s+(\w+)"),
        ("protocol", "protocol", r"^\s*(?:public|private|internal)?\s*protocol\s+(\w+)"),
        ("func", "function", r"^\s*(?:public|private|internal|static|final|class)?\s*func\s+(\w+)"),
    ],
}

_FALLBACK = [
    ("class", "class", r"^\s*class\s+(\w+)"),
    ("function", "function", r"^\s*(?:def|fn|func|function)\s+(\w+)"),
]

_rule_cache: dict[str, list[tuple[str, str, re.Pattern]]] = {}


@dataclass(frozen=True)
class Symbol:
    name: str
    kind: str
    start_line: int
    end_line: int


def _rules(language: str) -> list[tuple[str, str, re.Pattern]]:
    if language not in _rule_cache:
        compiled = []
        for kind_name, kind, expr in _SYMBOL_RULES.get(language, _FALLBACK):
            compiled.append((kind_name, kind, re.compile(expr, re.MULTILINE)))
        _rule_cache[language] = compiled
    return _rule_cache[language]


def extract_symbols(content: str, language: str) -> list[Symbol]:
    """Return symbols (1-based inclusive start/end lines), ordered by start line."""
    lines = content.splitlines()
    match_lines: list[tuple[int, str, str]] = []  # (0-based line, kind, name)

    for kind_name, kind, pattern in _rules(language):
        for m in pattern.finditer(content):
            name = next((g for g in m.groups() if g), "")
            if not name:
                continue
            lineno = content.count("\n", 0, m.start())
            match_lines.append((lineno, kind, name))

    match_lines.sort(key=lambda t: (t[0], t[1]))
    unique: list[tuple[int, str, str]] = []
    seen: set[tuple[int, str]] = set()
    for lineno, kind, name in match_lines:
        if (lineno, name) in seen:
            continue
        seen.add((lineno, name))
        unique.append((lineno, kind, name))

    symbols: list[Symbol] = []
    for i, (lineno, kind, name) in enumerate(unique):
        end = unique[i + 1][0] - 1 if i + 1 < len(unique) else len(lines) - 1
        symbols.append(Symbol(name=name, kind=kind, start_line=lineno + 1, end_line=end + 1))
    return symbols
