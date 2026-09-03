"""Application-level tool dispatcher guardrail (layer B).

Independent of the agent framework (LangGraph or otherwise): every tool
dispatch MUST pass through :func:`assert_dispatch_allowed` before
execution. This reuses the single source of truth in
``fraud_companion.domain.tools_spec`` — it never redeclares the
allow-list.
"""
from __future__ import annotations

from fraud_companion.domain.tools_spec import ALLOWED_TOOLS, assert_tool_allowed

__all__ = ["ALLOWED_TOOLS", "assert_dispatch_allowed"]


def assert_dispatch_allowed(tool_name: str) -> None:
    """Raise ``DisallowedToolError`` if ``tool_name`` is not allow-listed.

    Call this before dispatching any tool call, regardless of whether the
    call originated from LangGraph's ToolNode or any other path.
    """
    assert_tool_allowed(tool_name)
