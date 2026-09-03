"""Tests for the application-level tool dispatcher guardrail (layer B).

This is independent of LangGraph: it must reject any tool name outside
``domain.tools_spec.ALLOWED_TOOLS`` before dispatch, regardless of the
agent framework in use.
"""
from __future__ import annotations

import pytest

from fraud_companion.application.tool_dispatcher import assert_dispatch_allowed
from fraud_companion.domain.tools_spec import ALLOWED_TOOLS, DisallowedToolError

ALLOWED_NAMES = [
    "get_analysis_pack",
    "put_agent_brief",
    "list_cases",
    "list_aml_alerts",
]

DISALLOWED_NAMES = [
    "resolve",
    "archive",
    "create_case",
    "notes",
    "reassign",
    "enforce",
    "sar",
    "start_review",
    "bulk_action",
    "some_unknown_tool_xyz",
]


@pytest.mark.parametrize("tool_name", ALLOWED_NAMES)
def test_allows_each_allow_listed_tool(tool_name: str) -> None:
    assert_dispatch_allowed(tool_name)  # must not raise


@pytest.mark.parametrize("tool_name", DISALLOWED_NAMES)
def test_rejects_disallowed_tools(tool_name: str) -> None:
    with pytest.raises(DisallowedToolError):
        assert_dispatch_allowed(tool_name)


def test_dispatcher_has_no_duplicated_allow_list_literal() -> None:
    """The dispatcher must reuse tools_spec.ALLOWED_TOOLS, not redeclare it."""
    from fraud_companion.application import tool_dispatcher

    assert tool_dispatcher.ALLOWED_TOOLS is ALLOWED_TOOLS
