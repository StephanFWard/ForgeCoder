"""Scope contracts: the file-level bounds of one coding task.

A scope contract is the per-task half of ForgeCoder's prompt discipline. It
answers, before the model writes anything, where the change may land, what must
stay untouched, and what would prove the change worked. The model is told the
contract in its task frame; the server checks the returned patch against it
before the patch is ever shown or injected.

Globs, not raw paths: real repositories move files, so contracts are pinned to
patterns (`src/**/*.py`) and the negative space is explicit. A contract without
forbidden paths is incomplete, so :data:`DEFAULT_FORBIDDEN` is always applied.

Everything here is local and deterministic: no network, no model, no writes.
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field

# Never patch targets, regardless of what a task says: version control
# internals, dependency trees, build output, lockfiles, binary artifacts, and
# secrets. The model cannot recompute a lockfile hash or a compiled binary, and
# these are the paths where a wrong edit is expensive or unrecoverable.
DEFAULT_FORBIDDEN: tuple[str, ...] = (
    ".git/**",
    "**/node_modules/**",
    "**/.venv/**",
    "**/venv/**",
    "**/__pycache__/**",
    "**/dist/**",
    "**/build/**",
    "**/target/**",
    "**/out/**",
    "**/.env",
    "**/.env.*",
    "**/*.lock",
    "**/package-lock.json",
    "**/*.gguf",
    "**/*.dll",
    "**/*.so",
    "**/*.png",
    "**/*.jpg",
)

# Commands that count as acceptance evidence when the request names one. A
# patch is only "done" against a command someone can actually run, so an
# unnamed command is reported as stated-by-nobody rather than invented.
_ACCEPTANCE_PATTERNS: tuple[tuple[str, str], ...] = (
    (r"\bpytest\b", "pytest"),
    (r"\bnpm\s+(?:run\s+)?test\b", "npm test"),
    (r"\bnpx\s+mocha\b", "npx mocha"),
    (r"\bnode\s+--test\b", "node --test"),
    (r"\bruff\s+check\b", "ruff check"),
    (r"\bmypy\b", "mypy"),
    (r"\btsc\b", "tsc"),
    (r"\bcargo\s+test\b", "cargo test"),
    (r"\bgo\s+test\b", "go test"),
    (r"\bmvn\s+test\b", "mvn test"),
    (r"\bgradlew\s+test\b", "gradlew test"),
    (r"\bdotnet\s+test\b", "dotnet test"),
    (r"\bunittest\b", "python -m unittest"),
)

NO_ACCEPTANCE = "no acceptance command was stated; name the command that proves this change"

DEFAULT_ROLLBACK = (
    "Revert the file with `git checkout -- <path>`; ForgeCoder previews before writing and never commits."
)


def _pattern_to_regex(pattern: str) -> re.Pattern[str]:
    """Compile a gitignore-style glob into an anchored regex.

    ``**/`` matches any number of leading directories, ``**`` matches anything
    including ``/``, ``*`` matches within one path segment, and ``?`` matches a
    single non-separator character.
    """
    out: list[str] = ["^"]
    i = 0
    while i < len(pattern):
        char = pattern[i]
        if char == "*":
            if pattern.startswith("**/", i):
                out.append("(?:.*/)?")
                i += 3
                continue
            if pattern.startswith("**", i):
                out.append(".*")
                i += 2
                continue
            out.append("[^/]*")
        elif char == "?":
            out.append("[^/]")
        else:
            out.append(re.escape(char))
        i += 1
    out.append("$")
    return re.compile("".join(out))


def matches_any(path: str, patterns: tuple[str, ...] | list[str]) -> bool:
    """True when ``path`` (workspace-relative, POSIX separators) matches any glob."""
    normalized = path.replace("\\", "/").lstrip("./")
    return any(_pattern_to_regex(p).match(normalized) for p in patterns)


@dataclass
class ScopeContract:
    """The bounds of one task: goal, allowed paths, forbidden paths, proof."""

    goal: str
    allowed_files: list[str] = field(default_factory=list)
    forbidden_files: list[str] = field(default_factory=list)
    acceptance_criteria: list[str] = field(default_factory=list)
    rollback_plan: str = DEFAULT_ROLLBACK
    approvals_required: list[str] = field(default_factory=list)
    task_id: str = ""

    def allows(self, path: str) -> bool:
        """A path is writable when nothing forbids it and something allows it.

        An empty ``allowed_files`` means the scope was never declared, so it
        reads as unverified rather than as permission.
        """
        if matches_any(path, self.forbidden_files):
            return False
        return matches_any(path, self.allowed_files) if self.allowed_files else False

    def violations(self, paths: list[str]) -> list[str]:
        """Human-readable reasons why each of ``paths`` is out of scope."""
        reasons: list[str] = []
        for path in paths:
            if matches_any(path, self.forbidden_files):
                reasons.append(f"{path} matches a forbidden path in the scope contract")
            elif not self.allowed_files:
                reasons.append(f"{path} is out of scope: the task declared no allowed paths")
            elif not matches_any(path, self.allowed_files):
                reasons.append(
                    f"{path} is out of scope: allowed paths are {', '.join(self.allowed_files)}"
                )
        return reasons

    def to_dict(self) -> dict:
        return {
            "task_id": self.task_id,
            "goal": self.goal,
            "allowed_files": list(self.allowed_files),
            "forbidden_files": list(self.forbidden_files),
            "acceptance_criteria": list(self.acceptance_criteria),
            "rollback_plan": self.rollback_plan,
            "approvals_required": list(self.approvals_required),
        }


def acceptance_commands(*texts: str | None) -> list[str]:
    """Commands the request itself named as proof, in first-mentioned order."""
    found: list[str] = []
    blob = "\n".join(t for t in texts if t)
    for pattern, label in _ACCEPTANCE_PATTERNS:
        if label not in found and re.search(pattern, blob, re.IGNORECASE):
            found.append(label)
    return found


def build_contract(goal: str, *, file: str | None = None,
                   evidence_paths: list[str] | tuple[str, ...] = (),
                   instruction: str | None = None,
                   error: str | None = None) -> ScopeContract:
    """Derive the scope contract for one request.

    The target file — when the editor supplied one — is the whole allowed set: a
    request that names a file is a request to change *that* file. Without a
    file the contract falls back to the retrieved evidence, bounded and
    deduplicated, because those are the only paths the model has seen.
    """
    clean_goal = " ".join((goal or "").split())[:300]
    if file:
        allowed = [file.replace("\\", "/")]
    else:
        allowed = sorted({p.replace("\\", "/") for p in evidence_paths if p})[:8]

    acceptance = acceptance_commands(clean_goal, instruction, error) or [NO_ACCEPTANCE]

    approvals = ["WRITE_FILE requires an explicit confirmation and a diff preview"]
    haystack = " ".join(filter(None, (clean_goal, instruction, error))).lower()
    if re.search(r"\b(new file|create|scaffold|add a file|boilerplate)\b", haystack):
        approvals.append("creating a new file requires approval")
    if re.search(r"\b(dependency|dependencies|package|library|import a new)\b", haystack):
        approvals.append("adding a dependency requires approval")

    contract = ScopeContract(
        goal=clean_goal,
        allowed_files=allowed,
        forbidden_files=list(DEFAULT_FORBIDDEN),
        acceptance_criteria=acceptance,
        approvals_required=approvals,
    )
    digest = hashlib.sha1(f"{clean_goal}|{'|'.join(allowed)}".encode()).hexdigest()[:12]
    contract.task_id = f"task-{digest}"
    return contract
