"""Tests for the LangGraph ToolNode allow-list guardrail (layer A).

Implemented against the REAL installed langgraph API (1.2.11 /
langgraph-prebuilt 1.1.0): ``ToolNode`` accepts a ``wrap_tool_call``
callable of type
``Callable[[ToolCallRequest, Callable[[ToolCallRequest], ToolMessage | Command]], ToolMessage | Command]``.
This matches the research description of a "ToolCallRequest interceptor",
so no deviation from the plan was required.
"""
from __future__ import annotations

import pytest
from langchain_core.messages import ToolMessage
from langgraph.prebuilt.tool_node import ToolCallRequest

from fraud_companion.adapters.llm.guardrail import build_tool_guardrail
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


def _make_request(tool_name: str) -> ToolCallRequest:
    return ToolCallRequest(
        tool_call={"name": tool_name, "args": {}, "id": "call-1", "type": "tool_call"},
        tool=None,
        state={},
        runtime=None,
    )


@pytest.mark.parametrize("tool_name", ALLOWED_NAMES)
def test_guardrail_delegates_allowed_tools_to_execute(tool_name: str) -> None:
    wrap_tool_call = build_tool_guardrail()
    request = _make_request(tool_name)
    sentinel = ToolMessage(content="ok", tool_call_id="call-1")
    calls: list[ToolCallRequest] = []

    def execute(req: ToolCallRequest) -> ToolMessage:
        calls.append(req)
        return sentinel

    result = wrap_tool_call(request, execute)

    assert result is sentinel
    assert calls == [request]


@pytest.mark.parametrize("tool_name", DISALLOWED_NAMES)
def test_guardrail_rejects_disallowed_tools_without_executing(tool_name: str) -> None:
    wrap_tool_call = build_tool_guardrail()
    request = _make_request(tool_name)

    def execute(req: ToolCallRequest) -> ToolMessage:
        raise AssertionError("execute() must not be called for a disallowed tool")

    with pytest.raises(DisallowedToolError):
        wrap_tool_call(request, execute)


def test_guardrail_module_has_no_duplicated_allow_list_literal() -> None:
    """The interceptor must reuse tools_spec.ALLOWED_TOOLS, not redeclare it."""
    from fraud_companion.adapters.llm import guardrail

    assert guardrail.ALLOWED_TOOLS is ALLOWED_TOOLS
