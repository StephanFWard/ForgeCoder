"""Language detection and file classification."""
from __future__ import annotations

from pathlib import Path

# extension -> language name (display + chunker/symbol heuristics switch on these)
EXTENSION_LANGUAGES: dict[str, str] = {
    ".py": "python",
    ".pyw": "python",
    ".js": "javascript",
    ".jsx": "javascript",
    ".mjs": "javascript",
    ".cjs": "javascript",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".java": "java",
    ".kt": "kotlin",
    ".kts": "kotlin",
    ".c": "c",
    ".h": "c",
    ".cpp": "cpp",
    ".cc": "cpp",
    ".cxx": "cpp",
    ".hpp": "cpp",
    ".hh": "cpp",
    ".cs": "csharp",
    ".go": "go",
    ".rs": "rust",
    ".rb": "ruby",
    ".php": "php",
    ".swift": "swift",
    ".sql": "sql",
    ".html": "html",
    ".htm": "html",
    ".css": "css",
    ".scss": "css",  # treat css family the same
    ".less": "css",
    ".json": "json",
    ".yaml": "yaml",
    ".yml": "yaml",
    ".toml": "toml",
    ".ini": "ini",
    ".cfg": "ini",
    ".xml": "xml",
    ".md": "markdown",
    ".rst": "markdown",
    ".sh": "shell",
    ".bash": "shell",
    ".ps1": "powershell",
    ".bat": "batch",
    ".cmd": "batch",
    ".dockerfile": "docker",
    ".vue": "vue",
    ".svelte": "svelte",
    ".proto": "protobuf",
    ".gradle": "groovy",
    ".groovy": "groovy",
    ".lua": "lua",
    ".r": "r",
    ".asm": "assembly",
    ".clj": "clojure",
    ".scala": "scala",
    ".sol": "solidity",
    ".tf": "hcl",
    ".tfvars": "hcl",
    ".graphql": "graphql",
    ".gql": "graphql",
}


def detect_language(name: str) -> str | None:
    """Return the language name for a file name, or None if unknown."""
    lower = name.lower()
    special = {
        "dockerfile": "docker",
        "makefile": "makefile",
        "cmakelists.txt": "cmake",
        "justfile": "just",
        "rakefile": "ruby",
    }
    if lower in special:
        return special[lower]
    return EXTENSION_LANGUAGES.get(Path(name).suffix.lower())


def is_supported(name: str) -> bool:
    return detect_language(name) is not None
