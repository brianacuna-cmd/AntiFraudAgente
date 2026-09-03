"""LangGraph ToolNode allow-list guardrail (layer A — pre-execution).

Built against the REAL installed API (langgraph 1.2.11 /
langgraph-prebuilt 1.1.0): ``ToolNode`` accepts a ``wrap_tool_call``
keyword argument whose type is
``Callable[[ToolCallRequest, Callable[[ToolCallRequest], ToolMessage | Command]], ToolMessage | Command]``.
This matches the "ToolCallRequest interceptor" described by prior
research (obs 457) — no API deviation was needed.

Both this interceptor and the application-level dispatcher
(``fraud_companion.application.tool_dispatcher``) MUST consume the same
``ALLOWED_TOOLS`` frozenset from ``fraud_companion.domain.tools_spec`` —
this is the single source of truth for the guardrail.
"""
from __future__ import annotations

from typing import Callable

from langchain_core.messages import ToolMessage
from langgraph.prebuilt.tool_node import ToolCallRequest
from langgraph.types import Command

from fraud_companion.domain.tools_spec import ALLOWED_TOOLS, assert_tool_allowed

__all__ = ["ALLOWED_TOOLS", "build_tool_guardrail"]

_Execute = Callable[[ToolCallRequest], "ToolMessage | Command"]
_WrapToolCall = Callable[[ToolCallRequest, _Execute], "ToolMessage | Command"]


def build_tool_guardrail() -> _WrapToolCall:
    """Build a ``wrap_tool_call`` interceptor enforcing the allow-list.

    Pass the returned callable as ``ToolNode(..., wrap_tool_call=...)``.
    Any tool call whose name is not in
    ``fraud_companion.domain.tools_spec.ALLOWED_TOOLS`` is rejected with
    ``DisallowedToolError`` BEFORE ``execute`` runs. Allowed tool calls
    are delegated to ``execute`` unchanged.
    """

    def wrap_tool_call(request: ToolCallRequest, execute: _Execute) -> "ToolMessage | Command":
        tool_name = request.tool_call["name"]
        assert_tool_allowed(tool_name)
        return execute(request)

    return wrap_tool_call
