"""Repository context assembly.

Builds the text handed to the model for one chat turn: system prompt +
request + retrieved chunks + current-file selection + git diff summary,
all fitted into a token budget.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from core.retrieval.budget import estimate_tokens, fit_to_budget, truncate_to_tokens
from core.retrieval.search import SearchEngine

_RUNTIME_PROMPTS = Path(__file__).resolve().parents[2] / "runtime" / "prompts"


def load_prompt(name: str) -> str:
    """Load ``runtime/prompts/<name>.txt`` with a built-in fallback."""
    path = _RUNTIME_PROMPTS / f"{name}.txt"
    try:
        return path.read_text(encoding="utf-8").strip()
    except OSError:
        return _DEFAULT_PROMPTS.get(name, "") or name


_DEFAULT_PROMPTS = {
    "chat": (
        "You are ForgeCoder, a local software engineering assistant.\n"
        "Rules:\n"
        "1. Prefer minimal changes.\n"
        "2. Never invent files.\n"
        "3. Use only the supplied repository context.\n"
        "4. Preserve existing APIs unless asked to change them.\n"
        "5. Explain uncertainty.\n"
        "6. For edits, produce a machine-applicable patch.\n"
        "7. Never claim tests passed unless they were actually executed."
    ),
}


@dataclass
class BuiltContext:
    system: str
    request: str
    sections: list[dict] = field(default_factory=list)
    total_tokens: int = 0

    def as_dict(self) -> dict:
        return {
            "system": self.system,
            "request": self.request,
            "sections": self.sections,
            "total_tokens": self.total_tokens,
        }


class ContextBuilder:
    def __init__(self, index=None, search_engine: SearchEngine | None = None):
        self.index = index
        self.search_engine = search_engine
        if search_engine is None and index is not None:
            self.search_engine = SearchEngine(index)

    def build(self, message: str, *, workspace: str | None = None, file: str | None = None,
              selection: tuple[int, int] | None = None, budget: int = 4096,
              behavior: str = "chat") -> BuiltContext:
        """Assemble a prompt context for one user message."""
        system = load_prompt(behavior)
        sections: list[dict] = []
        used = estimate_tokens(system) + estimate_tokens(message)

        # 1. Retrieved repository context — batch-append chunks until the
        #    budget is filled instead of stopping at a fixed count.
        if self.search_engine and message.strip():
            pool = self._retrieve(message, workspace=workspace)
            fitted = fit_to_budget(pool, budget - used - 400, min_chunks=1)
            if fitted:
                text = _render_sections(fitted)
                sections.append({"type": "repository", "tokens": estimate_tokens(text), "text": text})
                used += estimate_tokens(text)

        # 2. Current file + selection
        if file and workspace:
            snippet = _selection_snippet(Path(workspace) / file, selection)
            if snippet:
                text = f"### {file}\n```\n{snippet}\n```"
                text = truncate_to_tokens(text, max(budget - used - 200, 100))
                sections.append({"type": "file", "tokens": estimate_tokens(text), "text": text})
                used += estimate_tokens(text)

        return BuiltContext(system=system, request=message, sections=sections, total_tokens=used)

    def _retrieve(self, message: str, *, workspace: str | None = None,
                  max_chunks: int = 24, max_chunk_tokens: int = 400) -> list[dict]:
        """Gather a deduplicated candidate pool for the context window.

        SearchEngine already ranks candidates; we cap each chunk's size so one
        large file cannot monopolise the window, drop duplicates (same line
        range or same content), and return everything — ``fit_to_budget`` then
        appends as many as the remaining token budget allows.
        """
        results = self.search_engine.search(message, workspace=workspace, limit=max_chunks)
        pool: list[dict] = []
        seen_ranges: set[tuple[str, int, int]] = set()
        seen_content: set[str] = set()
        for r in results:
            range_key = (r.path, r.start_line, r.end_line)
            content_key = r.content.strip()
            if range_key in seen_ranges or content_key in seen_content:
                continue
            seen_ranges.add(range_key)
            seen_content.add(content_key)
            pool.append({
                "content": truncate_to_tokens(r.content, max_chunk_tokens),
                "path": r.path,
                "start_line": r.start_line,
                "end_line": r.end_line,
                "modified": r.modified,
            })
        return pool


def _selection_snippet(path: Path, selection: tuple[int, int] | None) -> str:
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return ""
    if selection:
        start, end = selection
        start = max(0, int(start) - 1)
        end = min(len(lines), int(end) + 1) if end <= len(lines) else len(lines)
        if 0 <= start < end:
            return "\n".join(lines[start:end])
    return "\n".join(lines[:150])


def _render_sections(chunks: list[dict]) -> str:
    out: list[str] = []
    for c in chunks:
        header = f"# {c['path']}:{c['start_line']}-{c['end_line']}"
        out.append(f"{header}\n{c['content']}")
    return "\n\n".join(out)
