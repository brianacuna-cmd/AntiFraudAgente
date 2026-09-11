"""Single source of truth for the LLM tool allow-list guardrail.

Both the ToolNode interceptor (layer 1) and the application dispatcher
(layer 2) MUST consume ``ALLOWED_TOOLS`` / ``assert_tool_allowed`` from
here — never redeclare the allow-list elsewhere.
"""
from __future__ import annotations

ALLOWED_TOOLS: frozenset[str] = frozenset(
    {
        "get_analysis_pack",
        "put_agent_brief",
        "list_cases",
        "list_aml_alerts",
    }
)

AUTHORING_TOOLS: frozenset[str] = frozenset(
    {
        "create_scoring_rule_via_factor_scoring",
        "update_scoring_rule",
        "activate_scoring_rule",
        "list_scoring_rules",
        "get_scoring_rule",
        "simulate_scoring_rule",
    }
)


class DisallowedToolError(ValueError):
    """Raised when a tool name outside the allow-list in use is invoked."""

    def __init__(self, tool_name: str) -> None:
        super().__init__(f"Tool '{tool_name}' is not in the allow-list")
        self.tool_name = tool_name


def is_tool_allowed(tool_name: str, allowed: frozenset[str] = ALLOWED_TOOLS) -> bool:
    """Return True only if ``tool_name`` is a member of ``allowed``.

    Defaults to the case-analyst ``ALLOWED_TOOLS`` set so every existing
    call-site is unchanged. Pass ``allowed=AUTHORING_TOOLS`` (or any other
    frozenset) to check membership against a different allow-list.
    """
    return tool_name in allowed


def assert_tool_allowed(tool_name: str, allowed: frozenset[str] = ALLOWED_TOOLS) -> None:
    """Raise DisallowedToolError if ``tool_name`` is not in ``allowed``."""
    if not is_tool_allowed(tool_name, allowed=allowed):
        raise DisallowedToolError(tool_name)
