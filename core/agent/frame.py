"""Task frames: the first unit of coding-agent work is not an edit.

A request like "add null validation" is not a task until someone decides which
file it lands in, which lines it touches, what proves it works, and which
questions are still open. A capable model will fill those gaps with plausible
choices, and a plausible implementation that is incompatible with the system is
the expensive failure.

So ForgeCoder writes a frame before it asks for code: the goal, the facts it
actually verified (each one a receipt — a path and a line range that was really
supplied), the allowed and forbidden paths, the acceptance evidence, and the
unknowns. The frame is rendered into the user turn so the model edits inside
known bounds, and it is returned to the client so the user can see what the
assistant believed.

Frames are derived, never invented: a fact only enters the frame if ForgeCoder
supplied that source, and an unknown stays unknown until evidence or a human
resolves it. Local, deterministic, no model call.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from core.agent.scope import NO_ACCEPTANCE, ScopeContract, build_contract

MAX_FACTS = 8
MAX_PATHS = 8
MAX_UNKNOWNS = 4
MAX_GOAL_CHARS = 240


@dataclass
class TaskFrame:
    """Goal, verified facts, bounds, proof, and open questions for one task."""

    goal: str
    facts: list[str] = field(default_factory=list)
    allowed: list[str] = field(default_factory=list)
    forbidden: list[str] = field(default_factory=list)
    acceptance: list[str] = field(default_factory=list)
    unknowns: list[str] = field(default_factory=list)
    ready: bool = True
    note: str = ""

    def open_unknowns(self) -> list[str]:
        """Unknowns that must be resolved (or asked about) before editing."""
        return list(self.unknowns) if not self.ready or self.unknowns else []

    def to_dict(self) -> dict:
        return {
            "goal": self.goal,
            "facts": list(self.facts),
            "allowed": list(self.allowed),
            "forbidden": list(self.forbidden),
            "acceptance": list(self.acceptance),
            "unknowns": list(self.unknowns),
            "ready": self.ready,
            "note": self.note,
        }


def _clean(text: str | None, limit: int = MAX_GOAL_CHARS) -> str:
    return " ".join((text or "").split())[:limit]


def _receipts(file: str | None, selection: tuple[int, int] | None,
              evidence: list[dict] | tuple[dict, ...],
              file_lines: dict[str, int] | None) -> list[str]:
    """Line-anchored receipts for the sources that were really supplied."""
    facts: list[str] = []
    if file:
        if selection:
            facts.append(f"{file}:{selection[0]}-{selection[1]}")
        elif file_lines and file_lines.get(file):
            facts.append(f"{file}:1-{file_lines[file]}")
        else:
            facts.append(file)
    for item in evidence:
        if not isinstance(item, dict):
            continue
        path = str(item.get("path") or "")
        if not path:
            continue
        start, end = item.get("start_line"), item.get("end_line")
        receipt = f"{path}:{start}-{end}" if start and end else path
        if receipt not in facts:
            facts.append(receipt)
    return facts[:MAX_FACTS]


def _unknowns(*, file: str | None, selection: tuple[int, int] | None,
              evidence, error: str | None, acceptance: list[str],
              facts: list[str],
              creation_targets: list[str] | tuple[str, ...] | None = None) -> list[str]:
    """The gaps this frame can see in itself. Ordered by how much they block.

    A creation request already answers its own questions: the files to create
    are derived (``creation_targets``), and its acceptance is the smoke check,
    not a named command — so neither "which file?" nor "name the command" is
    an unknown. Asking them is what once made "make the game snake in html"
    come back as ``[warn] ask-dont-guess`` instead of a game.
    """
    open_questions: list[str] = []
    if not facts and not creation_targets:
        open_questions.append("no repository evidence matched this request; name the file to inspect")
    if not file:
        if creation_targets:
            pass  # the files to create ARE the answer; asking would invite a guess
        else:
            open_questions.append("which file should the change land in?")
    elif not selection:
        open_questions.append("which lines of the supplied file should change?")
    if acceptance and acceptance[0] == NO_ACCEPTANCE:
        open_questions.append("no acceptance command was stated; name the command that proves the change")
    if error and not selection:
        open_questions.append("which command produced the error output that was supplied?")
    return open_questions[:MAX_UNKNOWNS]


def build_frame(goal: str, *, file: str | None = None,
                selection: tuple[int, int] | None = None,
                evidence: list[dict] | tuple[dict, ...] = (),
                instruction: str | None = None,
                error: str | None = None,
                contract: ScopeContract | None = None,
                file_lines: dict[str, int] | None = None,
                creation_targets: list[str] | tuple[str, ...] | None = None) -> TaskFrame:
    """Derive the task frame for one request from what was actually supplied.

    ``evidence`` is the retrieved context that went into the prompt (the same
    dicts the context builder rendered), so every fact carries a real
    ``path:start-end`` receipt and a reviewer can check it. For creation
    requests (``creation_targets``), the frame bounds the task to the files it
    will create and stays ready even with no repository evidence.
    """
    clean_goal = _clean(goal) or _clean(instruction)
    facts = _receipts(file, selection, evidence, file_lines)
    scope = contract or build_contract(
        clean_goal or (instruction or ""), file=file,
        evidence_paths=[str(e.get("path")) for e in evidence if isinstance(e, dict)],
        instruction=instruction, error=error,
        creation_targets=creation_targets,
    )
    unknowns = _unknowns(file=file, selection=selection, evidence=evidence,
                         error=error, acceptance=scope.acceptance_criteria, facts=facts,
                         creation_targets=creation_targets)
    note = ""
    if creation_targets:
        note = ("Creation task: generate complete new files ("
                + ", ".join(creation_targets)
                + "). Do not modify existing files and do not ask which file.")
    return TaskFrame(
        goal=clean_goal or "unspecified",
        facts=facts,
        allowed=list(scope.allowed_files),
        forbidden=list(scope.forbidden_files),
        acceptance=list(scope.acceptance_criteria),
        unknowns=unknowns,
        ready=bool(facts) or bool(creation_targets),
        note=note,
    )


def render_frame(frame: TaskFrame, *, max_forbidden: int = 6) -> str:
    """Render the frame as the compact block that rides in the user turn.

    One screen maximum: a frame the model skims is worth less than no frame at
    all, so the forbidden list is summarised and every section is bounded.
    """
    lines = ["TASK FRAME", f"Goal: {frame.goal}"]
    if frame.note:
        lines.append("Task note: " + frame.note)

    if frame.facts:
        lines.append("Verified from supplied context: " + "; ".join(frame.facts[:MAX_FACTS]))
    if frame.allowed:
        lines.append("Allowed paths: " + ", ".join(frame.allowed[:MAX_PATHS]))
    else:
        lines.append("Allowed paths: (none declared — do not edit until one is named)")
    if frame.forbidden:
        shown = ", ".join(frame.forbidden[:max_forbidden])
        more = len(frame.forbidden) - len(frame.forbidden[:max_forbidden])
        lines.append(f"Forbidden paths: everything not listed above, including {shown}"
                     + (f", and {more} more patterns" if more > 0 else ""))
    lines.append("Acceptance evidence: " + "; ".join(frame.acceptance[:3]))
    if frame.unknowns:
        # Imperatives, not questions: a 1.5B model that reads literal questions
        # ("which file should the change land in?") replies with them verbatim —
        # once as four fabricated "[warn] ask-dont-guess" lines, questions and
        # informational lines alike. State who resolves them and what may never
        # be echoed.
        lines.append(
            "Unknowns to resolve yourself (from the request or the supplied "
            "context; ask the user only what is truly undecidable, and never "
            "repeat them, rule names, or severity markers in the reply): "
            + "; ".join(frame.unknowns[:MAX_UNKNOWNS])
        )
    lines.append("Evidence status: " + (
        "sufficient — edit only inside the allowed paths above"
        if frame.ready else
        "insufficient — return `files: []` and name the missing context instead of guessing"
    ))
    return "\n".join(lines)


__all__ = ["MAX_FACTS", "TaskFrame", "build_frame", "render_frame"]
