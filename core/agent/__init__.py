"""Agent layer: context-aware agents (Jev + per-function agents).

Each Forge capability is its own agent sharing one AgentContext, so the
view/edit/apply branching that used to live in the VS Code sidebar now
lives in the agent return value (Recommendation.actions).
"""
from core.agent.base import FunctionAgent
from core.agent.context import AgentContext, Recommendation
from core.agent.forge import ForgeAgent, DEFAULT
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
from core.agent.tools import ForgeTool, get_tool, list_tools

__all__ = [
    "AgentContext",
    "BEHAVIOR_RULES",
    "DEFAULT",
    "DEFAULT_FORBIDDEN",
    "DEFAULT_RULES",
    "ForgeAgent",
    "ForgeTool",
    "FunctionAgent",
    "NO_ACCEPTANCE",
    "RULES_PATH",
    "Recommendation",
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
    "get_tool",
    "list_tools",
    "load_rules",
    "matches_any",
    "parse_rules",
    "render_frame",
    "render_task_frame",
    "rules_for_behavior",
    "run_rules",
    "system_prompt",
]

