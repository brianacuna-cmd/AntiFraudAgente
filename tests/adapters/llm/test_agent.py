"""Tests for the Gemini agent factory (Slice 6).

No real network/model call is ever made: ``ChatGoogleGenerativeAI`` and
``create_agent`` are patched at their import site in
``fraud_companion.adapters.llm.agent`` so construction never contacts
Google's API and never needs a real GOOGLE_API_KEY.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from google.genai.errors import APIError
from langchain.agents import create_agent
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from fraud_companion.adapters.http.client import AntiFraudHttpClient
from fraud_companion.adapters.http.errors import CaseClosedError
from fraud_companion.adapters.llm.agent import (
    GeminiAgent,
    UnsupportedProviderError,
    _extract_retry_after,
    build_agent,
)
from fraud_companion.adapters.llm.tools import (
    build_get_analysis_pack_tool,
    build_put_agent_brief_tool,
)
from fraud_companion.application.agent_port import (
    AgentRateLimitedError,
    AgentUnavailableError,
)
from fraud_companion.config import Settings
from fraud_companion.domain.policy import SECURITY_SYSTEM_PROMPT
from fraud_companion.domain.tools_spec import ALLOWED_TOOLS, DisallowedToolError


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


class TestBuildAgent:
    def test_constructs_chat_model_with_configured_model_and_key(
        self, settings: Settings, http_client: MagicMock
    ) -> None:
        with patch("fraud_companion.adapters.llm.agent.ChatGoogleGenerativeAI") as mock_model_cls:
            build_agent(settings, http_client)

        mock_model_cls.assert_called_once()
        _, kwargs = mock_model_cls.call_args
        assert kwargs["model"] == settings.llm_model
        assert kwargs["google_api_key"] == settings.llm_api_key

    def test_passes_temperature_and_max_output_tokens_from_settings(
        self, settings: Settings, http_client: MagicMock
    ) -> None:
        with patch("fraud_companion.adapters.llm.agent.ChatGoogleGenerativeAI") as mock_model_cls:
            build_agent(settings, http_client)

        _, kwargs = mock_model_cls.call_args
        assert kwargs["temperature"] == settings.llm_temperature
        assert kwargs["max_output_tokens"] == settings.llm_max_output_tokens

    def test_api_key_value_never_appears_in_stdout_or_stderr(
        self, settings: Settings, http_client: MagicMock, capsys: pytest.CaptureFixture[str]
    ) -> None:
        with patch("fraud_companion.adapters.llm.agent.ChatGoogleGenerativeAI"):
            build_agent(settings, http_client)

        captured = capsys.readouterr()
        assert settings.llm_api_key not in captured.out
        assert settings.llm_api_key not in captured.err

    def test_wires_exactly_the_four_allowed_tools(
        self, settings: Settings, http_client: MagicMock
    ) -> None:
        with patch("fraud_companion.adapters.llm.agent.ChatGoogleGenerativeAI"), patch(
            "fraud_companion.adapters.llm.agent.create_agent"
        ) as mock_create_agent:
            build_agent(settings, http_client)

        _, kwargs = mock_create_agent.call_args
        tool_names = {tool.name for tool in kwargs["tools"]}
        assert tool_names == set(ALLOWED_TOOLS)

    def test_applies_the_security_system_prompt(
        self, settings: Settings, http_client: MagicMock
    ) -> None:
        with patch("fraud_companion.adapters.llm.agent.ChatGoogleGenerativeAI"), patch(
            "fraud_companion.adapters.llm.agent.create_agent"
        ) as mock_create_agent:
            build_agent(settings, http_client)

        _, kwargs = mock_create_agent.call_args
        assert kwargs["system_prompt"] == SECURITY_SYSTEM_PROMPT

    def test_wires_the_guardrail_middleware_and_it_rejects_disallowed_tools(
        self, settings: Settings, http_client: MagicMock
    ) -> None:
        with patch("fraud_companion.adapters.llm.agent.ChatGoogleGenerativeAI"), patch(
            "fraud_companion.adapters.llm.agent.create_agent"
        ) as mock_create_agent:
            build_agent(settings, http_client)

        _, kwargs = mock_create_agent.call_args
        assert len(kwargs["middleware"]) == 1

        from langgraph.prebuilt.tool_node import ToolCallRequest

        middleware = kwargs["middleware"][0]
        request = ToolCallRequest(
            tool_call={"name": "resolve_case", "args": {}, "id": "call-1", "type": "tool_call"},
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
        with patch("fraud_companion.adapters.llm.agent.ChatGoogleGenerativeAI"), patch(
            "fraud_companion.adapters.llm.agent.create_agent"
        ) as mock_create_agent:
            mock_create_agent.return_value = compiled
            result = build_agent(settings, http_client)

        assert isinstance(result, GeminiAgent)
        # The wrapper delegates invoke to the compiled agent unchanged.
        assert result.invoke({"messages": []}) == "compiled-result"


class TestGeminiAgentErrorTranslation:
    """The adapter is the ONLY place that knows about google.genai errors;
    it maps them onto the provider-agnostic agent-port taxonomy."""

    def test_delegates_invoke_on_success(self) -> None:
        compiled = MagicMock()
        compiled.invoke.return_value = {"messages": ["ok"]}
        assert GeminiAgent(compiled).invoke({"x": 1}) == {"messages": ["ok"]}

    def test_translates_429_to_rate_limited(self) -> None:
        compiled = MagicMock()
        compiled.invoke.side_effect = APIError(429, {})
        with pytest.raises(AgentRateLimitedError):
            GeminiAgent(compiled).invoke({})

    def test_translates_503_to_unavailable(self) -> None:
        compiled = MagicMock()
        compiled.invoke.side_effect = APIError(503, {"error": {"status": "UNAVAILABLE"}})
        with pytest.raises(AgentUnavailableError):
            GeminiAgent(compiled).invoke({})

    def test_extracts_retry_after_from_429_retry_info(self) -> None:
        compiled = MagicMock()
        compiled.invoke.side_effect = APIError(
            429,
            {
                "error": {
                    "status": "RESOURCE_EXHAUSTED",
                    "details": [
                        {
                            "@type": "type.googleapis.com/google.rpc.RetryInfo",
                            "retryDelay": "17s",
                        }
                    ],
                }
            },
        )
        with pytest.raises(AgentRateLimitedError) as exc_info:
            GeminiAgent(compiled).invoke({})
        assert exc_info.value.retry_after == 17.0

    def test_non_backpressure_api_error_propagates_unchanged(self) -> None:
        compiled = MagicMock()
        compiled.invoke.side_effect = APIError(400, {})
        with pytest.raises(APIError):
            GeminiAgent(compiled).invoke({})


class TestProviderSelection:
    def test_default_gemini_provider_builds_agent(
        self, settings: Settings, http_client: MagicMock
    ) -> None:
        with patch("fraud_companion.adapters.llm.agent.ChatGoogleGenerativeAI"), patch(
            "fraud_companion.adapters.llm.agent.create_agent"
        ) as mock_create_agent:
            result = build_agent(settings, http_client)

        mock_create_agent.assert_called_once()
        assert isinstance(result, GeminiAgent)

    def test_explicit_gemini_provider_builds_agent(
        self, settings: Settings, http_client: MagicMock
    ) -> None:
        from dataclasses import replace

        gemini_settings = replace(settings, llm_provider="gemini")
        with patch("fraud_companion.adapters.llm.agent.ChatGoogleGenerativeAI"), patch(
            "fraud_companion.adapters.llm.agent.create_agent"
        ) as mock_create_agent:
            result = build_agent(gemini_settings, http_client)

        mock_create_agent.assert_called_once()
        assert isinstance(result, GeminiAgent)

    def test_unsupported_provider_raises_and_builds_nothing(
        self, settings: Settings, http_client: MagicMock
    ) -> None:
        from dataclasses import replace

        bad_settings = replace(settings, llm_provider="openai")
        with patch("fraud_companion.adapters.llm.agent.ChatGoogleGenerativeAI") as mock_model_cls, patch(
            "fraud_companion.adapters.llm.agent.create_agent"
        ) as mock_create_agent:
            with pytest.raises(UnsupportedProviderError, match="openai"):
                build_agent(bad_settings, http_client)

        mock_model_cls.assert_not_called()
        mock_create_agent.assert_not_called()


class TestRetryAfterClamp:
    def _api_error_with_retry_delay(self, delay: str) -> APIError:
        return APIError(
            429,
            {
                "error": {
                    "status": "RESOURCE_EXHAUSTED",
                    "details": [
                        {
                            "@type": "type.googleapis.com/google.rpc.RetryInfo",
                            "retryDelay": delay,
                        }
                    ],
                }
            },
        )

    def test_normal_retry_after_passes_through(self) -> None:
        exc = self._api_error_with_retry_delay("17s")
        assert _extract_retry_after(exc) == 17.0

    def test_absurdly_large_retry_after_is_clamped(self) -> None:
        exc = self._api_error_with_retry_delay("999999s")
        assert _extract_retry_after(exc) == 60.0

    def test_malformed_retry_after_yields_none(self) -> None:
        exc = self._api_error_with_retry_delay("not-a-delay")
        assert _extract_retry_after(exc) is None


class _ScriptedModel(BaseChatModel):
    """Fake chat model that returns pre-scripted ``AIMessage`` replies in
    order, one per invocation, and counts how many times it was invoked.

    Mirrors the ``_ToolCallOnceModel`` pattern from
    ``tests/application/test_case_created_handler_agent.py`` but supports a
    second scripted turn so tests can prove whether the graph loops back to
    the model after a tool's ToolMessage.
    """

    replies: list[AIMessage]
    call_count: list[int] = [0]

    def bind_tools(self, tools, **kwargs):  # create_agent binds tools; ignore them
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        index = self.call_count[0]
        self.call_count[0] += 1
        ai = self.replies[index]
        return ChatResult(generations=[ChatGeneration(message=ai)])

    @property
    def _llm_type(self) -> str:
        return "scripted-model"


def _put_agent_brief_tool_call() -> AIMessage:
    return AIMessage(
        content="",
        tool_calls=[
            {
                "name": "put_agent_brief",
                "args": {"case_id": "case-1", "brief": "drafted brief"},
                "id": "call-1",
                "type": "tool_call",
            }
        ],
    )


def _get_analysis_pack_tool_call() -> AIMessage:
    return AIMessage(
        content="",
        tool_calls=[
            {
                "name": "get_analysis_pack",
                "args": {"case_id": "case-1"},
                "id": "call-1",
                "type": "tool_call",
            }
        ],
    )


class TestPutAgentBriefTerminatesTheRun:
    """R2/Scenario 2/3: return_direct=True on put_agent_brief must stop the
    ReAct loop right after its ToolMessage — no confirmation model round."""

    def test_run_ends_at_put_agent_brief_tool_message_on_success(self) -> None:
        http_client = MagicMock(spec=AntiFraudHttpClient)
        http_client.put.return_value = {
            "id": "case-1",
            "status": "OPEN",
            "agentBrief": "drafted brief",
        }
        model = _ScriptedModel(
            replies=[
                _put_agent_brief_tool_call(),
                AIMessage(content="brief saved"),
            ],
            call_count=[0],
        )
        agent = create_agent(
            model=model,
            tools=[
                build_get_analysis_pack_tool(http_client),
                build_put_agent_brief_tool(http_client),
            ],
        )

        result = agent.invoke({"messages": [("human", "handle case-1")]})

        messages = result["messages"]
        last_message = messages[-1]
        assert last_message.__class__.__name__ == "ToolMessage"
        assert getattr(last_message, "name", None) == "put_agent_brief"
        assert model.call_count[0] == 1
        assert not any(
            m.__class__.__name__ == "AIMessage" and m.content == "brief saved"
            for m in messages
        )

    def test_run_ends_at_put_agent_brief_tool_message_on_terminal_error(self) -> None:
        http_client = MagicMock(spec=AntiFraudHttpClient)
        http_client.put.side_effect = CaseClosedError(
            status=409, code="CASE_CLOSED", message="closed"
        )
        model = _ScriptedModel(
            replies=[
                _put_agent_brief_tool_call(),
                AIMessage(content="brief saved"),
            ],
            call_count=[0],
        )
        agent = create_agent(
            model=model,
            tools=[
                build_get_analysis_pack_tool(http_client),
                build_put_agent_brief_tool(http_client),
            ],
        )

        # put_agent_brief's HTTP error is not a ToolInvocationError, so it
        # propagates out of the graph rather than becoming an error-status
        # ToolMessage (matches production ToolNode's default error handling,
        # unchanged by this feature). The run still terminates immediately
        # after the single tool call — the model is never invoked a second
        # time to "confirm" the failure, exactly like the success path.
        with pytest.raises(CaseClosedError):
            agent.invoke({"messages": [("human", "handle case-1")]})

        assert model.call_count[0] == 1

    def test_read_only_tool_still_loops_back_to_the_model(self) -> None:
        """Regression (Scenario 4): get_analysis_pack has no return_direct,
        so the graph must still call the model a second time after it."""
        http_client = MagicMock(spec=AntiFraudHttpClient)
        http_client.get.return_value = {
            "case": {"id": "case-1"},
            "timeline": [],
            "snapshot": {"hits": []},
            "amlAlerts": [],
            "relatedCases": [],
            "agentBrief": None,
        }
        model = _ScriptedModel(
            replies=[
                _get_analysis_pack_tool_call(),
                AIMessage(content="final answer"),
            ],
            call_count=[0],
        )
        agent = create_agent(
            model=model,
            tools=[
                build_get_analysis_pack_tool(http_client),
                build_put_agent_brief_tool(http_client),
            ],
        )

        result = agent.invoke({"messages": [("human", "handle case-1")]})

        last_message = result["messages"][-1]
        assert last_message.__class__.__name__ == "AIMessage"
        assert last_message.content == "final answer"
        assert model.call_count[0] == 2
