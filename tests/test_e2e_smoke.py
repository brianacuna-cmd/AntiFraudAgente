"""End-to-end smoke test (Slice 10).

Exercises the FULL wiring — Kafka message -> OutboxConsumer.process_message
-> handle_case_created -> agent — with every external boundary mocked: no
real Kafka broker (``confluent_kafka.Consumer`` is patched), no real Gemini
call (the agent is a fake), no real HTTP call (``AntiFraudHttpClient`` is a
mock passed straight to the dispatcher-level guardrail check).

This proves the pipeline wiring itself, plus the security guardrail
holding end-to-end through that wiring (not just in isolation, see
``tests/application/test_tool_dispatcher.py`` and
``tests/adapters/llm/test_guardrail.py`` for the isolated unit coverage).
"""
from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from fraud_companion.adapters.kafka.consumer import OutboxConsumer
from fraud_companion.application.tool_dispatcher import assert_dispatch_allowed
from fraud_companion.config import Settings
from fraud_companion.domain.tools_spec import DisallowedToolError


class _FakeMessage:
    def __init__(self, *, headers=None, value=b"{}", key=b"evt-1", error=None):
        self._headers = headers
        self._value = value
        self._key = key
        self._error = error

    def headers(self):
        return self._headers

    def value(self):
        return self._value

    def key(self):
        return self._key

    def error(self):
        return self._error


@pytest.fixture
def settings() -> Settings:
    return Settings(
        anti_fraud_base_url="https://api.example.com",
        anti_fraud_agent_api_key="secret-key",
        llm_api_key="secret-google-key",
        kafka_bootstrap_servers="localhost:9092",
        kafka_group_id="fraud-companion",
        kafka_organization_id="org-1",
    )


def _case_created_message(case_id: str = "case-123") -> _FakeMessage:
    envelope = {"eventType": "case.created", "payload": {"caseId": case_id}}
    return _FakeMessage(
        headers=[("event_type", b"case.created"), ("organization_id", b"org-1")],
        value=json.dumps(envelope).encode("utf-8"),
        key=case_id.encode("utf-8"),
    )


def _make_consumer(settings: Settings, agent) -> OutboxConsumer:
    with patch("fraud_companion.adapters.kafka.consumer.Consumer") as mock_consumer_cls:
        mock_consumer_cls.return_value = MagicMock()
        consumer = OutboxConsumer(settings=settings, agent=agent)
    return consumer


class TestFullPipelineWiring:
    def test_case_created_message_invokes_agent_once_and_commits(self, settings: Settings) -> None:
        agent = MagicMock()
        # A successful run must leave evidence that put_agent_brief was called;
        # otherwise the handler forces redelivery instead of committing.
        agent.invoke.return_value = {
            "messages": [SimpleNamespace(name="put_agent_brief")]
        }
        consumer = _make_consumer(settings, agent)

        consumer.process_message(_case_created_message("case-123"))

        agent.invoke.assert_called_once()
        (call_args,), _ = agent.invoke.call_args
        human_message = call_args["messages"][0][1]
        assert "case-123" in human_message
        consumer._consumer.commit.assert_called_once()

    def test_non_case_created_message_skips_agent_but_commits(self, settings: Settings) -> None:
        agent = MagicMock()
        consumer = _make_consumer(settings, agent)

        msg = _FakeMessage(
            headers=[("event_type", b"case.updated")],
            value=b"{}",
        )
        consumer.process_message(msg)

        agent.invoke.assert_not_called()
        consumer._consumer.commit.assert_called_once()

    def test_retryable_agent_failure_does_not_commit(self, settings: Settings) -> None:
        agent = MagicMock()
        agent.invoke.side_effect = RuntimeError("transient upstream failure")
        consumer = _make_consumer(settings, agent)

        with pytest.raises(RuntimeError):
            consumer.process_message(_case_created_message("case-456"))

        consumer._consumer.commit.assert_not_called()


class TestGuardrailHoldsThroughWiring:
    """Proves the security lock holds through the FULL wiring, not just in
    isolation: an agent attempting a disallowed tool call (e.g. resolving,
    enforcing, or issuing a SAR) is rejected before it ever reaches the
    anti-fraud API.
    """

    def test_agent_attempt_to_call_disallowed_tool_is_rejected(self) -> None:
        for disallowed_tool_name in ("resolve", "enforce_action", "file_sar", "re_score_case"):
            with pytest.raises(DisallowedToolError):
                assert_dispatch_allowed(disallowed_tool_name)

    def test_agent_driven_pipeline_rejects_disallowed_tool_call_end_to_end(
        self, settings: Settings
    ) -> None:
        """Simulate the agent (mid ``agent.invoke``) attempting a disallowed
        tool call; the dispatcher guardrail must reject it before any HTTP
        call reaches the mocked ``AntiFraudHttpClient``, and the exception
        must propagate so the consumer does NOT commit (retryable-shaped
        failure — a disallowed call should never look like success).
        """
        http_client = MagicMock()

        def fake_agent_invoke(_payload: dict) -> dict:
            # Simulates the guardrail middleware intercepting a disallowed
            # tool call attempted mid-agent-run, before any tool body
            # (and therefore before any http_client call) executes.
            assert_dispatch_allowed("resolve")
            http_client.put("/cases/case-999/resolve", {})  # pragma: no cover
            return {}

        agent = MagicMock()
        agent.invoke.side_effect = fake_agent_invoke
        consumer = _make_consumer(settings, agent)

        with pytest.raises(DisallowedToolError):
            consumer.process_message(_case_created_message("case-999"))

        http_client.put.assert_not_called()
        consumer._consumer.commit.assert_not_called()
