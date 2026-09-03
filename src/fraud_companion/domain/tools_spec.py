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


class DisallowedToolError(ValueError):
    """Raised when a tool name outside ALLOWED_TOOLS is invoked."""

    def __init__(self, tool_name: str) -> None:
        super().__init__(f"Tool '{tool_name}' is not in the allow-list")
        self.tool_name = tool_name


def is_tool_allowed(tool_name: str) -> bool:
    """Return True only if ``tool_name`` is exactly one of the 4 allowed tools."""
    return tool_name in ALLOWED_TOOLS


def assert_tool_allowed(tool_name: str) -> None:
    """Raise DisallowedToolError if ``tool_name`` is not allow-listed."""
    if not is_tool_allowed(tool_name):
        raise DisallowedToolError(tool_name)
