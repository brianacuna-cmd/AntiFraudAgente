"""Tests for the authoring agent factory (Slice 3).

No real network/model call is ever made: ``ChatGoogleGenerativeAI`` and
``create_agent`` are patched at their import site in
``fraud_companion.adapters.llm.authoring_agent``, mirroring
``tests/adapters/llm/test_agent.py``.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from fraud_companion.adapters.http.client import AntiFraudHttpClient
from fraud_companion.adapters.llm.agent import GeminiAgent
from fraud_companion.adapters.llm.authoring_agent import build_authoring_agent
from fraud_companion.config import Settings
from fraud_companion.domain.authoring_policy import AUTHORING_SYSTEM_PROMPT
from fraud_companion.domain.tools_spec import ALLOWED_TOOLS, AUTHORING_TOOLS, DisallowedToolError


@pytest.fixture
def settings() -> Settings:
    return Settings(
        anti_fraud_base_url="https://anti-fraud.internal",
        anti_fraud_agent_api_key="super-secret-key",
        llm_api_key="google-secret-key",
        kafka_bootstrap_servers="kafka-broker:9092",
        kafka_group_id="fraud-companion-consumer",
        kafka_organization_id="org-123",
    )


@pytest.fixture
def http_client() -> MagicMock:
    return MagicMock(spec=AntiFraudHttpClient)


class TestBuildAuthoringAgent:
    def test_constructs_chat_model_with_configured_model_and_key(
        self, settings: Settings, http_client: MagicMock
    ) -> None:
        with patch(
            "fraud_companion.adapters.llm.authoring_agent.ChatGoogleGenerativeAI"
        ) as mock_model_cls:
            build_authoring_agent(settings, http_client)

        mock_model_cls.assert_called_once()
        _, kwargs = mock_model_cls.call_args
        assert kwargs["model"] == settings.llm_model
        assert kwargs["google_api_key"] == settings.llm_api_key
        assert kwargs["temperature"] == settings.llm_temperature
        assert kwargs["max_output_tokens"] == settings.llm_max_output_tokens

    def test_wires_exactly_the_six_authoring_tools(
        self, settings: Settings, http_client: MagicMock
    ) -> None:
        with patch(
            "fraud_companion.adapters.llm.authoring_agent.ChatGoogleGenerativeAI"
        ), patch(
            "fraud_companion.adapters.llm.authoring_agent.create_agent"
        ) as mock_create_agent:
            build_authoring_agent(settings, http_client)

        _, kwargs = mock_create_agent.call_args
        tool_names = {tool.name for tool in kwargs["tools"]}
        assert tool_names == set(AUTHORING_TOOLS)
        assert tool_names.isdisjoint(ALLOWED_TOOLS)

    def test_no_authoring_tool_uses_return_direct(
        self, settings: Settings, http_client: MagicMock
    ) -> None:
        with patch(
            "fraud_companion.adapters.llm.authoring_agent.ChatGoogleGenerativeAI"
        ), patch(
            "fraud_companion.adapters.llm.authoring_agent.create_agent"
        ) as mock_create_agent:
            build_authoring_agent(settings, http_client)

        _, kwargs = mock_create_agent.call_args
        for tool in kwargs["tools"]:
            assert getattr(tool, "return_direct", False) is False

    def test_applies_the_authoring_system_prompt(
        self, settings: Settings, http_client: MagicMock
    ) -> None:
        with patch(
            "fraud_companion.adapters.llm.authoring_agent.ChatGoogleGenerativeAI"
        ), patch(
            "fraud_companion.adapters.llm.authoring_agent.create_agent"
        ) as mock_create_agent:
            build_authoring_agent(settings, http_client)

        _, kwargs = mock_create_agent.call_args
        assert kwargs["system_prompt"] == AUTHORING_SYSTEM_PROMPT

    def test_wires_the_authoring_guardrail_middleware_and_it_rejects_case_tools(
        self, settings: Settings, http_client: MagicMock
    ) -> None:
        with patch(
            "fraud_companion.adapters.llm.authoring_agent.ChatGoogleGenerativeAI"
        ), patch(
            "fraud_companion.adapters.llm.authoring_agent.create_agent"
        ) as mock_create_agent:
            build_authoring_agent(settings, http_client)

        _, kwargs = mock_create_agent.call_args
        assert len(kwargs["middleware"]) == 1

        from langgraph.prebuilt.tool_node import ToolCallRequest

        middleware = kwargs["middleware"][0]
        request = ToolCallRequest(
            tool_call={"name": "put_agent_brief", "args": {}, "id": "call-1", "type": "tool_call"},
            tool=None,
            state={},
            runtime=None,
        )

        def _execute_must_not_be_called(_request: ToolCallRequest) -> None:
            raise AssertionError("execute must never run for a disallowed tool")

        with pytest.raises(DisallowedToolError):
            middleware.wrap_tool_call(request, _execute_must_not_be_called)

    def test_returns_a_gemini_agent_wrapping_the_compiled_agent(
        self, settings: Settings, http_client: MagicMock
    ) -> None:
        compiled = MagicMock()
        compiled.invoke.return_value = "compiled-result"
        with patch(
            "fraud_companion.adapters.llm.authoring_agent.ChatGoogleGenerativeAI"
        ), patch(
            "fraud_companion.adapters.llm.authoring_agent.create_agent"
        ) as mock_create_agent:
            mock_create_agent.return_value = compiled
            result = build_authoring_agent(settings, http_client)

        assert isinstance(result, GeminiAgent)
        assert result.invoke({"messages": []}) == "compiled-result"


class TestCaseAnalystAgentUnaffectedByAuthoring:
    """Regression: importing the authoring module must never pollute the
    case-analyst agent's allow-list, prompt, or tool wiring."""

    def test_case_allowed_tools_still_exactly_four(self) -> None:
        assert len(ALLOWED_TOOLS) == 4
        assert ALLOWED_TOOLS == frozenset(
            {"get_analysis_pack", "put_agent_brief", "list_cases", "list_aml_alerts"}
        )

    def test_case_analyst_builder_never_includes_an_authoring_tool_name(
        self, settings: Settings, http_client: MagicMock
    ) -> None:
        from fraud_companion.adapters.llm.agent import build_agent

        with patch("fraud_companion.adapters.llm.agent.ChatGoogleGenerativeAI"), patch(
            "fraud_companion.adapters.llm.agent.create_agent"
        ) as mock_create_agent:
            build_agent(settings, http_client)

        _, kwargs = mock_create_agent.call_args
        tool_names = {tool.name for tool in kwargs["tools"]}
        assert tool_names.isdisjoint(AUTHORING_TOOLS)
