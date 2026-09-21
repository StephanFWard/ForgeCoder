"""Agent context: one object every ForgeCoder agent reads and writes.

The Recommendation.actions field drives UI behavior (view, diff, apply, test,
etc.) so the branching that used to live in the VS Code sidebar now lives in
the agent return value. Each Forge capability is its own FunctionAgent sharing
one AgentContext.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any

# Action kinds that agents can recommend. The UI/SidebarProvider reads these
# and performs the corresponding operation instead of hardcoding branching.
ACTION_VIEW = "view"
ACTION_DIFF = "diff"
ACTION_APPLY = "apply"
ACTION_APPLY_MULTI = "apply_multi"
ACTION_TEST = "test"
ACTION_CREATE = "create"
ACTION_REVIEW = "review"
ACTION_COMMIT = "commit"
ACTION_PUSH = "push"
ACTION_PLAN = "plan"
ACTION_ACT = "act"

@dataclass
class AgentContext:
    message: str = ""
    workspace: str | None = None
    file: str | None = None
    selection: tuple[int, int] | None = None
    history: list[dict] = field(default_factory=list)
    code: str | None = None
    error: str | None = None
    instruction: str | None = None
    confirmed: bool = False
    budget: int = 4096
    behavior_hint: str | None = None
    built: Any | None = None
    intent: Any | None = None
    confidence: Any | None = None
    evidence_text: str = ""
    system_text: str = ""
    built_up_to_date: bool = False
    scratch: dict[str, Any] = field(default_factory=dict)

    async def build(self, *, index: Any = None, inference: Any = None):
        if self.built_up_to_date and self.built is not None:
            return self
        from core.retrieval.context import ContextBuilder
        try:
            from core.agent.intent import classify_intent
            self.intent = await classify_intent(
                self.message or self.instruction or "",
                client=inference, has_file=bool(self.file),
                has_selection=bool(self.selection))
        except Exception:
            self.intent = None
        code_change = bool(getattr(self.intent, "code_change", False))
        if (self.behavior_hint or "") in {"edit", "fix", "test", "plan", "act", "create"}:
            code_change = True
        try:
            self.built = ContextBuilder(index).build(
                self.message or self.instruction or "",
                workspace=self.workspace, file=self.file,
                selection=self.selection, budget=self.budget,
                behavior=self.behavior_hint or "chat",
                code_change=code_change)
        except Exception:
            self.built = None
        secs = list(getattr(self.built, "sections", []) or [])
        self.evidence_text = "\n\n".join(str(s.get("text", "")) for s in secs if isinstance(s, dict))
        self.system_text = str(getattr(self.built, "system", "") or "")
        try:
            from core.agent.confidence import answer_confidence
            from core.retrieval.budget import truncate_to_tokens
            self.confidence = await answer_confidence(
                self.message or self.instruction or "",
                truncate_to_tokens(self.evidence_text, 2600), client=inference)
        except Exception:
            self.confidence = None
        self.built_up_to_date = True
        return self

    def state_text(self) -> str:
        from core.system_one.primitives import state_text as _st
        return _st({"message": self.message or self.instruction or "",
                    "file": self.file or "",
                    "selection": list(self.selection) if self.selection else [],
                    "evidence": self.evidence_text[:4000]})

    def effective_request(self) -> str:
        return self.message or self.instruction or self.code or ""

    def to_dict(self) -> dict[str, Any]:
        return {"message": self.message, "workspace": self.workspace,
                "file": self.file,
                "selection": list(self.selection) if self.selection else None,
                "code": self.code, "error": self.error,
                "instruction": self.instruction, "confirmed": self.confirmed,
                "behavior_hint": self.behavior_hint,
                "intent": self.intent.to_dict() if self.intent else None,
                "confidence": self.confidence.to_dict() if self.confidence else None}

    @classmethod
    def from_request(cls, message: str = "", *, workspace=None, file=None,
                     selection=None, history=None, code=None, error=None,
                     instruction=None, confirmed=False, budget=4096,
                     behavior_hint=None):
        sel = None
        try:
            if isinstance(selection, dict):
                sel = (int(selection.get("start", 1)), int(selection.get("end", 1)))
            elif isinstance(selection, (list, tuple)) and len(selection) >= 2:
                sel = (int(selection[0]), int(selection[1]))
            elif hasattr(selection, "as_tuple"):
                sel = selection.as_tuple()
        except Exception:
            sel = None
        return cls(message=message or "", workspace=workspace, file=file,
                   selection=sel, history=list(history or []), code=code,
                   error=error, instruction=instruction,
                   confirmed=bool(confirmed), budget=int(budget or 4096),
                   behavior_hint=behavior_hint)

@dataclass
class Recommendation:
    kind: str = "answer"
    text: str = ""
    probability: float = 0.5
    patches: list[dict] = field(default_factory=list)
    creates: dict[str, str] = field(default_factory=dict)
    review: dict | None = None
    verification: dict | None = None
    confidence: dict | None = None
    intent: dict | None = None
    # Actions drive UI behavior - view, diff, apply, apply_multi, test, create, etc.
    actions: list[str] = field(default_factory=list)
    agent: str = "forge"
    meta: dict[str, Any] = field(default_factory=dict)
    # For plan mode: steps to execute
    steps: list[dict] | None = None
    # For multi-file patches
    multi_patches: list[dict] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {"kind": self.kind, "text": self.text,
                "probability": round(float(self.probability), 6),
                "patches": list(self.patches or []),
                "creates": dict(self.creates or {}),
                "review": self.review, "verification": self.verification,
                "confidence": self.confidence, "intent": self.intent,
                "actions": list(self.actions or []),
                "agent": self.agent, "meta": dict(self.meta or {}),
                "steps": self.steps, "multi_patches": self.multi_patches}

    # Convenience methods for action handling
    def has_action(self, action: str) -> bool:
        return action in self.actions

    def has_any_action(self, *actions: str) -> bool:
        return any(a in self.actions for a in actions)

    def is_patch_ready(self) -> bool:
        return self.has_any_action(ACTION_VIEW, ACTION_DIFF) and self.patches

    def is_multi_patch_ready(self) -> bool:
        return self.has_action(ACTION_APPLY_MULTI) and (self.multi_patches or self.patches)


__all__ = ["ACTION_APPLY", "ACTION_APPLY_MULTI", "ACTION_ACT", "ACTION_COMMIT",
           "ACTION_CREATE", "ACTION_DIFF", "ACTION_VIEW", "ACTION_TEST",
           "ACTION_PLAN", "ACTION_PUSH", "ACTION_REVIEW",
           "AgentContext", "Recommendation"]
