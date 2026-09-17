"""Repository context assembly.

Builds the text handed to the model for one chat turn: system prompt +
request + retrieved chunks + current-file selection + git diff summary,
all fitted into a token budget.

The system prompt is layered (behavior prompt + the rules that apply to that
behavior) and the turn carries a task frame with line-anchored receipts, so a
4K window spends its tokens on evidence and bounds rather than on prose rules.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from core.agent.creation import creation_targets, is_creation_request
from core.agent.frame import TaskFrame, build_frame
from core.agent.prompt import render_task_frame, system_prompt
from core.agent.scope import ScopeContract, build_contract
from core.retrieval.budget import estimate_tokens
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
    frame: TaskFrame | None = None
    contract: ScopeContract | None = None
    receipts: list[dict] = field(default_factory=list)
    file_lines: dict[str, int] = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "system": self.system,
            "request": self.request,
            "sections": self.sections,
            "total_tokens": self.total_tokens,
            "frame": self.frame.to_dict() if self.frame else None,
            "scope": self.contract.to_dict() if self.contract else None,
            "receipts": self.receipts,
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
        """Assemble a prompt context for one user message.

        The system prompt is the behavior prompt plus the rules that behavior is
        held to. The user turn (``request``) is the request plus a task frame
        derived from what was actually supplied, and ``sections`` hold the
        current-file and repository evidence. ``receipts`` and ``file_lines``
        record the sources the turn really contained, so a reply can be scored
        against them and a patch's line anchors verified before it is previewed.
        """
        system = system_prompt(behavior)
        sections: list[dict] = []
        used = estimate_tokens(system) + estimate_tokens(message)

        # Prefer fresh active-file evidence over potentially stale index chunks.
        active_path = None
        file_lines: dict[str, int] = {}
        if file and workspace:
            root = Path(workspace).resolve()
            candidate = (root / file).resolve()
            if candidate.is_relative_to(root):
                active_path = candidate.relative_to(root).as_posix()
                lines = _read_lines(candidate)
                if lines is not None:
                    file_lines[active_path] = len(lines)
                    text = _fit_lines(
                        f"### {active_path}\n{_numbered_lines(lines, selection)}",
                        min(1200, max(0, budget - used - 200)),
                    )
                    if text:
                        cost = estimate_tokens(text)
                        sections.append({"type": "file", "tokens": cost, "text": text})
                        used += cost

        supplied: list[dict] = []  # chunks that actually made it into the turn
        if self.search_engine and message.strip():
            pool = self._retrieve(message, workspace=workspace)
            rendered: list[str] = []
            for chunk in pool:
                # Do not contradict the current file with stale indexed copies.
                if chunk["path"].replace("\\", "/") == active_path:
                    continue
                text = _fit_lines(_render_sections([chunk]), max(0, budget - used - 200))
                if not text:
                    continue
                rendered.append(text)
                supplied.append(chunk)
                used += estimate_tokens(text) + 1  # separator allowance
            if rendered:
                text = "\n\n".join(rendered)
                sections.append({"type": "repository", "tokens": estimate_tokens(text), "text": text})

        # Frames and contracts are derived from supplied sources only: a fact
        # enters the frame when its path/lines were really sent, and the scope
        # contract bounds edits to what the model has actually seen. A creation
        # request ("make the game snake in html") with no active editor file is
        # bounded to the files it will create instead of being interrogated
        # with "which file?" — asking is what made it reply with ask-dont-guess
        # warnings instead of a game.
        creation: list[str] | None = None
        if active_path is None and is_creation_request(message):
            creation = creation_targets(message)
        contract = build_contract(
            message, file=active_path, evidence_paths=[c["path"] for c in supplied],
            creation_targets=creation,
        )
        frame = build_frame(
            message, file=active_path, selection=selection,
            evidence=supplied, contract=contract, file_lines=file_lines,
            creation_targets=creation,
        )
        request = message
        frame_text = render_task_frame(frame)
        if frame_text:
            request = f"{message}\n\n{frame_text}" if message else frame_text

        receipts = [{"path": c["path"], "start_line": c["start_line"], "end_line": c["end_line"]}
                    for c in supplied]
        if active_path and active_path in file_lines:
            start, end = selection if selection else (1, file_lines[active_path])
            receipts.insert(0, {"path": active_path, "start_line": start, "end_line": end})

        return BuiltContext(
            system=system, request=request, sections=sections, total_tokens=used,
            frame=frame, contract=contract, receipts=receipts, file_lines=file_lines,
        )

    def _retrieve(self, message: str, *, workspace: str | None = None,
                  max_chunks: int = 24, max_chunk_tokens: int = 400) -> list[dict]:
        """Gather a deduplicated candidate pool for the context window.

        SearchEngine already ranks candidates; we cap each chunk's size so one
        large file cannot monopolise the window, drop duplicates (same line
        range or same content). Whole-line fitting then appends as much
        evidence as the remaining token budget allows.
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
                "content": _fit_lines(r.content, max_chunk_tokens),
                "path": r.path,
                "start_line": r.start_line,
                "end_line": r.end_line,
                "modified": r.modified,
            })
        return pool


def _read_lines(path: Path) -> list[str] | None:
    """Whole file as lines; ``None`` when the file cannot be read."""
    try:
        return path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return None


def _numbered_lines(lines: list[str], selection: tuple[int, int] | None) -> str:
    """1-based numbered lines, windowed to the selection when one is supplied."""
    start, end = (0, min(len(lines), 150))
    if selection:
        start = max(0, int(selection[0]) - 1)
        end = min(len(lines), int(selection[1]))
    return "\n".join(f"{i + 1}: {lines[i]}" for i in range(start, end))


def _fit_lines(text: str, budget: int) -> str:
    """Keep only whole lines within the estimated token limit; never force a chunk."""
    kept: list[str] = []
    for line in text.splitlines():
        candidate = "\n".join([*kept, line])
        if estimate_tokens(candidate) > max(0, budget):
            break
        kept.append(line)
    return "\n".join(kept)


def _render_sections(chunks: list[dict]) -> str:
    out: list[str] = []
    for c in chunks:
        header = f"# {c['path']}:{c['start_line']}-{c['end_line']}"
        out.append(f"{header}\n{c['content']}")
    return "\n\n".join(out)
