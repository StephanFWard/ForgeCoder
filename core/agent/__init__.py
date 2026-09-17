"""Agent layer: task frames, scope contracts, rule checks, and prompt assembly.

The workbench disciplines that make a small local model's coding work safe to
review: what the task actually is (frame), where the change may land (scope),
which rules the reply is scored against (rules), and how the prompt is layered
so all of it fits in a 4K window (prompt).
"""

from core.agent.frame import TaskFrame, build_frame, render_frame
from core.agent.prompt import compose_user_turn, render_task_frame, system_prompt
from core.agent.rules import (
    BEHAVIOR_RULES,
    DEFAULT_RULES,
    RULES_PATH,
    Rule,
    RuleContext,
    RuleFinding,
    RuleReport,
    load_rules,
    parse_rules,
    rules_for_behavior,
    run_rules,
)
from core.agent.scope import (
    DEFAULT_FORBIDDEN,
    NO_ACCEPTANCE,
    ScopeContract,
    acceptance_commands,
    build_contract,
    matches_any,
)

__all__ = [
    "BEHAVIOR_RULES",
    "DEFAULT_FORBIDDEN",
    "DEFAULT_RULES",
    "NO_ACCEPTANCE",
    "RULES_PATH",
    "Rule",
    "RuleContext",
    "RuleFinding",
    "RuleReport",
    "ScopeContract",
    "TaskFrame",
    "acceptance_commands",
    "build_contract",
    "build_frame",
    "compose_user_turn",
    "load_rules",
    "matches_any",
    "parse_rules",
    "render_frame",
    "render_task_frame",
    "rules_for_behavior",
    "run_rules",
    "system_prompt",
]
