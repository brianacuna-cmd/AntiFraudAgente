"""B2 integration test: terminal tool errors through the REAL create_agent path.

Proves how a terminal error (CASE_NOT_FOUND) raised inside a tool body behaves
when driven by langchain's create_agent / LangGraph ToolNode as actually
installed — not a hand-mocked agent. handle_case_created must classify it as
SKIPPED_TERMINAL (committable) regardless of whether ToolNode propagates the
exception (default handler) or converts it to an error-status ToolMessage.
"""
from __future__ import annotations

from unittest.mock import MagicMock

from langchain.agents import create_agent
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from fraud_companion.adapters.http.errors import CaseNotFoundError
from fraud_companion.adapters.llm.tools import (
    build_get_analysis_pack_tool,
    build_put_agent_brief_tool,
)
from fraud_companion.application.case_created_handler import (
    HandleResult,
    handle_case_created,
)
from fraud_companion.domain.events import CASE_CREATED_EVENT

CASE_ID = "507f1f77bcf86cd799439011"


class _ToolCallOnceModel(BaseChatModel):
    """Fake chat model that emits exactly one tool call; enough to reach the
    tool node once (the tool raises before any second model turn)."""

    tool_name: str
    tool_args: dict

    def bind_tools(self, tools, **kwargs):  # create_agent binds tools; ignore them
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        ai = AIMessage(
            content="",
            tool_calls=[
                {
                    "name": self.tool_name,
                    "args": self.tool_args,
                    "id": "call-1",
                    "type": "tool_call",
                }
            ],
        )
        return ChatResult(generations=[ChatGeneration(message=ai)])

    @property
    def _llm_type(self) -> str:
        return "tool-call-once"


def _envelope() -> dict:
    return {"eventType": CASE_CREATED_EVENT, "payload": {"caseId": CASE_ID}}


def test_terminal_error_from_tool_is_skipped_terminal_through_create_agent() -> None:
    http_client = MagicMock()
    http_client.get.side_effect = CaseNotFoundError(
        status=404, code="CASE_NOT_FOUND", message="case gone"
    )
    model = _ToolCallOnceModel(
        tool_name="get_analysis_pack", tool_args={"case_id": CASE_ID}
    )
    agent = create_agent(
        model=model,
        tools=[
            build_get_analysis_pack_tool(http_client),
            build_put_agent_brief_tool(http_client),
        ],
    )

    result = handle_case_created(_envelope(), agent)

    assert result is HandleResult.SKIPPED_TERMINAL
    http_client.get.assert_called_once()
