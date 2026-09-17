"""Instructions as executable constraints.

Prose instructions are wishes; checkable instructions are tests. Each rule in
``runtime/prompts/agent-rules.md`` has a slug, one of five categories
(Startup / Forbidden / Definition of done / Uncertainty / Approval), a severity
(block / warn / info), and the name of a check implemented here. The markdown
is the reviewable source; this module is the part that can be run.

Rules are scored in the hot path, before a reply is shown, injected, or
applied, so a rule that no longer has a check surfaces as a stale rule instead
of silently passing. Everything is deterministic and local: string and line
arithmetic over the reply and the context ForgeCoder actually supplied.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from core.agent.scope import matches_any

_PACKAGE_ROOT = Path(__file__).resolve().parents[2]
RULES_PATH = _PACKAGE_ROOT / "runtime" / "prompts" / "agent-rules.md"

CATEGORIES = ("Startup", "Forbidden", "Definition of done", "Uncertainty", "Approval")
SEVERITIES = ("block", "warn", "info")


@dataclass(frozen=True)
class Rule:
    slug: str
    category: str
    severity: str
    check: str
    description: str

    def one_line(self) -> str:
        """The inline form carried by a system prompt (progressive disclosure)."""
        return f"[{self.severity}] {self.slug}: {self.description}"


# Fallback used when runtime/prompts/agent-rules.md is missing (e.g. a packaged
# install that shipped without it). Kept deliberately identical in spirit to the
# authored file: the same slugs, categories, severities, and checks.
DEFAULT_RULES: tuple[Rule, ...] = (
    Rule("fresh-context-before-edit", "Startup", "block", "context_present",
         "A structured edit needs the target file and its line-numbered source."),
    Rule("allowed-paths-only", "Forbidden", "block", "scope_paths",
         "Patched paths must match the task's allowed paths and no forbidden path."),
    Rule("line-ranges-in-file", "Forbidden", "block", "lines_within_file",
         "Operation line ranges must exist in the file that was supplied."),
    Rule("no-unrequested-files", "Forbidden", "block", "existing_target",
         "Only files verified to exist in the workspace may be patched."),
    Rule("operations-well-formed", "Definition of done", "block", "ops_well_formed",
         "replace/insert carry real code, delete carries an empty string."),
    Rule("operations-non-overlapping", "Definition of done", "block", "ops_non_overlapping",
         "Operations in one file must not overlap."),
    Rule("no-unverified-test-claims", "Definition of done", "warn", "no_unverified_claims",
         "Do not state that tests, lint, or builds passed unless they were run."),
    Rule("ask-dont-guess", "Uncertainty", "warn", "declared_unknowns",
         "Name unresolved unknowns instead of resolving them by a plausible choice."),
    Rule("writes-need-confirmation", "Approval", "block", "confirmed_write",
         "No write without preview, explicit confirmation, and a stale-file anchor."),
)

# Which rules a behavior is told about inline. Everything else stays in the
# router file, which keeps the prompt short enough for a 1.5B model to follow.
BEHAVIOR_RULES: dict[str, tuple[str, ...]] = {
    "chat": ("no-unverified-test-claims", "ask-dont-guess"),
    "hierarchical-chat": ("no-unverified-test-claims", "ask-dont-guess"),
    "explain": ("no-unverified-test-claims", "ask-dont-guess"),
    "test": ("no-unverified-test-claims", "ask-dont-guess", "fresh-context-before-edit"),
    "plan": ("ask-dont-guess", "no-unverified-test-claims", "writes-need-confirmation"),
    "edit": ("fresh-context-before-edit", "allowed-paths-only", "line-ranges-in-file",
             "no-unrequested-files", "operations-well-formed", "operations-non-overlapping",
             "no-unverified-test-claims", "ask-dont-guess"),
    "fix": ("fresh-context-before-edit", "allowed-paths-only", "line-ranges-in-file",
            "no-unrequested-files", "operations-well-formed", "operations-non-overlapping",
            "no-unverified-test-claims", "ask-dont-guess"),
    "completion": (),
}


_HEADING_RE = re.compile(r"^###\s+(?P<slug>[A-Za-z0-9][A-Za-z0-9_-]*)\s*$")
_FIELD_RE = re.compile(r"^(?P<key>category|severity|check)\s*:\s*(?P<value>.+?)\s*$", re.IGNORECASE)
_FENCE_RE = re.compile(r"```.*?```", re.DOTALL)


def parse_rules(text: str) -> list[Rule]:
    """Parse the rule router markdown into :class:`Rule` objects.

    One rule per ``### slug`` heading; ``category``/``severity``/``check`` lines
    follow, and the remaining lines are the description. Defaults are permissive
    (``warn`` severity, no check) so a malformed heading is reported as a stale
    rule rather than silently dropped.
    """
    rules: list[Rule] = []
    slug: str | None = None
    fields: dict[str, str] = {}
    body: list[str] = []

    def flush() -> None:
        if slug is None:
            return
        description = " ".join(" ".join(body).split())[:400]
        rules.append(Rule(
            slug=slug,
            category=fields.get("category", "Startup"),
            severity=fields.get("severity", "warn").lower(),
            check=fields.get("check", "").strip(),
            description=description or slug.replace("-", " "),
        ))

    for raw in text.splitlines():
        heading = _HEADING_RE.match(raw.strip())
        if heading:
            flush()
            slug, fields, body = heading.group("slug"), {}, []
            continue
        if slug is None:
            continue
        field_match = _FIELD_RE.match(raw.strip())
        if field_match:
            fields[field_match.group("key").lower()] = field_match.group("value")
        elif raw.strip():
            body.append(raw.strip())
    flush()
    return rules


def load_rules(path: Path | None = None) -> list[Rule]:
    """Load the rule router, falling back to :data:`DEFAULT_RULES`."""
    target = path or RULES_PATH
    try:
        parsed = parse_rules(target.read_text(encoding="utf-8"))
    except OSError:
        return list(DEFAULT_RULES)
    return parsed or list(DEFAULT_RULES)


def rules_for_behavior(rules: list[Rule], behavior: str) -> list[Rule]:
    """The subset of rules a behavior is told inline, in router order."""
    wanted = BEHAVIOR_RULES.get(behavior, ())
    return [rule for rule in rules if rule.slug in wanted]


# Checks that can be scored truthfully at write time, when the task's declared
# scope is no longer in hand: the patch itself, the file it is about to touch,
# and the confirmation that authorized the write. Scope checks are deliberately
# absent — without a contract, "allowed paths" would read as unverified and
# refuse every write.
WRITE_CHECKS = frozenset({
    "ops_well_formed",
    "ops_non_overlapping",
    "lines_within_file",
    "confirmed_write",
})


def rules_for_checks(rules: list[Rule], checks: frozenset[str] | set[str]) -> list[Rule]:
    """The subset of rules whose checks are in ``checks``, in router order."""
    return [rule for rule in rules if rule.check in checks]


@dataclass
class RuleContext:
    """Everything a check may look at: the reply plus the context it was given."""

    behavior: str = "chat"
    text: str = ""
    patches: list = field(default_factory=list)
    has_context: bool = True
    allowed_files: list[str] = field(default_factory=list)
    forbidden_files: list[str] = field(default_factory=list)
    known_files: set[str] = field(default_factory=set)
    file_lines: dict[str, int] = field(default_factory=dict)
    open_unknowns: list[str] = field(default_factory=list)
    write_attempted: bool = False
    write_confirmed: bool = True

    def paths(self) -> list[str]:
        return [getattr(p, "path", "") for p in self.patches]


def _ops(patch) -> list:
    return list(getattr(patch, "operations", []) or [])


def _line_bounds(op) -> tuple[int, int]:
    start = int(getattr(op, "start_line", 0) or 0)
    end = getattr(op, "end_line", None)
    return start, int(end) if end is not None else start


# --------------------------------------------------------------------------- checks
# Each check returns the messages that explain a violation (empty list = pass).
# The runner attaches the rule's slug, category, and severity to those messages,
# so a check never decides how loudly a rule speaks.

def check_context_present(ctx: RuleContext) -> list[str]:
    if ctx.patches and not ctx.has_context:
        return ["a structured patch was returned although no line-numbered source was supplied"]
    return []


def check_scope_paths(ctx: RuleContext) -> list[str]:
    problems: list[str] = []
    for path in ctx.paths():
        if matches_any(path, ctx.forbidden_files):
            problems.append(f"{path} is a forbidden path for this task")
        elif not ctx.allowed_files:
            problems.append(f"{path} is unverified: the task declared no allowed paths")
        elif not matches_any(path, ctx.allowed_files):
            problems.append(f"{path} is outside the declared scope ({', '.join(ctx.allowed_files)})")
    return problems


def check_lines_within_file(ctx: RuleContext) -> list[str]:
    problems: list[str] = []
    for patch in ctx.patches:
        path = getattr(patch, "path", "")
        count = ctx.file_lines.get(path)
        if not count:
            continue
        for op in _ops(patch):
            start, end = _line_bounds(op)
            if start < 1 or end > count:
                problems.append(f"{path}: lines {start}-{end} are outside the supplied file ({count} lines)")
    return problems


def check_existing_target(ctx: RuleContext) -> list[str]:
    if not ctx.known_files:
        return []
    return [f"{path} does not exist in the workspace" for path in ctx.paths() if path not in ctx.known_files]


def check_ops_well_formed(ctx: RuleContext) -> list[str]:
    problems: list[str] = []
    for patch in ctx.patches:
        path = getattr(patch, "path", "")
        for op in _ops(patch):
            kind = getattr(op, "type", "")
            content = getattr(op, "content", "") or ""
            start, end = _line_bounds(op)
            if kind in {"replace", "insert"} and not content.strip():
                problems.append(f"{path}: {kind} at line {start} carries no code")
            if kind == "insert" and end != start:
                problems.append(f"{path}: insert must use end_line == start_line ({start}), got {end}")
            if kind == "delete" and content.strip():
                problems.append(f"{path}: delete at line {start} should carry an empty content string")
    return problems


def check_ops_non_overlapping(ctx: RuleContext) -> list[str]:
    problems: list[str] = []
    for patch in ctx.patches:
        path = getattr(patch, "path", "")
        ranges = sorted(_line_bounds(op) for op in _ops(patch))
        for (prev_start, prev_end), (start, end) in zip(ranges, ranges[1:]):
            if start <= prev_end:
                problems.append(
                    f"{path}: operations overlap ({prev_start}-{prev_end} and {start}-{end}); combine them"
                )
    return problems


_CLAIM_RE = re.compile(
    r"\b(?:tests?|suite|lint|build|checks?|ci)\b[^.\n]{0,40}?"
    r"\b(?:pass(?:es|ed)?|succeed(?:s|ed)?|green|all good|clean)\b",
    re.IGNORECASE,
)


def check_no_unverified_claims(ctx: RuleContext) -> list[str]:
    """Heuristic: a claim that a run passed, in prose ForgeCoder did not run."""
    prose = _FENCE_RE.sub(" ", ctx.text or "")
    match = _CLAIM_RE.search(prose)
    if match:
        return [
            f"claims a run passed ({match.group(0).strip()!r}) without executed output; "
            "describe the change as proposed instead"
        ]
    return []


def check_declared_unknowns(ctx: RuleContext) -> list[str]:
    if ctx.patches and ctx.open_unknowns:
        return [f"a patch was produced while an unknown was unresolved: {ctx.open_unknowns[0]}"]
    return []


def check_confirmed_write(ctx: RuleContext) -> list[str]:
    if ctx.write_attempted and not ctx.write_confirmed:
        return ["a write was attempted without preview and explicit confirmation"]
    return []


CHECKS: dict[str, object] = {
    "context_present": check_context_present,
    "scope_paths": check_scope_paths,
    "lines_within_file": check_lines_within_file,
    "existing_target": check_existing_target,
    "ops_well_formed": check_ops_well_formed,
    "ops_non_overlapping": check_ops_non_overlapping,
    "no_unverified_claims": check_no_unverified_claims,
    "declared_unknowns": check_declared_unknowns,
    "confirmed_write": check_confirmed_write,
}


@dataclass
class RuleFinding:
    slug: str
    category: str
    severity: str
    message: str

    def to_dict(self) -> dict:
        return {"slug": self.slug, "category": self.category,
                "severity": self.severity, "message": self.message}


@dataclass
class RuleReport:
    """The score of one reply against the rule set."""

    rules_checked: int = 0
    findings: list[RuleFinding] = field(default_factory=list)

    @property
    def blocked(self) -> bool:
        return any(f.severity == "block" for f in self.findings)

    @property
    def ok(self) -> bool:
        return not self.blocked

    def messages(self, severity: str | None = None) -> list[str]:
        return [f.message for f in self.findings if severity is None or f.severity == severity]

    def to_dict(self) -> dict:
        return {
            "ok": self.ok,
            "rules_checked": self.rules_checked,
            "blocked": self.messages("block"),
            "warnings": self.messages("warn"),
            "findings": [f.to_dict() for f in self.findings],
        }


def run_rules(rules: list[Rule], ctx: RuleContext) -> RuleReport:
    """Score ``ctx`` against ``rules``; unknown checks are stale rules, not passes."""
    report = RuleReport()
    for rule in rules:
        check = CHECKS.get(rule.check)
        if check is None:
            report.findings.append(RuleFinding(
                slug=rule.slug, category=rule.category, severity="info",
                message=f"no check named {rule.check!r} is registered (stale rule)",
            ))
            continue
        report.rules_checked += 1
        for message in check(ctx):
            report.findings.append(RuleFinding(
                slug=rule.slug, category=rule.category,
                severity=rule.severity if rule.severity in SEVERITIES else "warn",
                message=message,
            ))
    return report


__all__ = [
    "BEHAVIOR_RULES",
    "CATEGORIES",
    "CHECKS",
    "DEFAULT_RULES",
    "RULES_PATH",
    "Rule",
    "RuleContext",
    "RuleFinding",
    "RuleReport",
    "WRITE_CHECKS",
    "load_rules",
    "parse_rules",
    "rules_for_behavior",
    "rules_for_checks",
    "run_rules",
]
