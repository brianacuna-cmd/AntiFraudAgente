"""Guards against unbounded-cardinality metric labels (design constraint).

Fixed-cardinality label values only: ``outcome``/``kind``/``error_type`` must
stay within the enum sets defined in the design — never a caseId, org, or
other unbounded identifier.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from fraud_companion.adapters.kafka.consumer import OutboxConsumer
from fraud_companion.application.agent_port import AgentRateLimitedError, AgentUnavailableError
from fraud_companion.application.case_created_handler import (
    BriefNotWrittenError,
    handle_case_created,
)
from fraud_companion.config import Settings

_ALLOWED_OUTCOMES = {
    "PROCESSED",
    "SKIPPED_IGNORED",
    "SKIPPED_TERMINAL",
    "SKIPPED_MALFORMED",
    "brief_not_written",
    "error",
}
_ALLOWED_BACKPRESSURE_KINDS = {"rate_limited", "unavailable"}


class _RecordingSink:
    def __init__(self) -> None:
        self.case_outcome_calls: list[str] = []
        self.backpressure_calls: list[dict] = []
        self.provider_error_calls: list[dict] = []

    def observe_token_usage(self, *, input: int, output: int, total: int) -> None:
        pass

    def record_case_outcome(self, outcome: str) -> None:
        self.case_outcome_calls.append(outcome)

    def record_backpressure(self, *, kind: str, attempt: int, escalated: bool) -> None:
        self.backpressure_calls.append({"kind": kind})

    def record_provider_error(self, *, kind: str) -> None:
        self.provider_error_calls.append({"kind": kind})


def test_handle_case_created_outcome_labels_are_fixed_cardinality():
    class _Agent:
        def invoke(self, payload):
            raise BriefNotWrittenError("unused")

    sink = _RecordingSink()

    with pytest.raises(BriefNotWrittenError):
        handle_case_created(
            {"eventType": "case.created", "payload": {"caseId": "case-1"}},
            _Agent(),
            metrics=sink,
        )

    for outcome in sink.case_outcome_calls:
        assert outcome in _ALLOWED_OUTCOMES
        assert "case-1" not in outcome


@patch("fraud_companion.adapters.kafka.consumer.time.sleep")
@patch("fraud_companion.adapters.kafka.consumer.random.uniform", return_value=0.0)
@patch("fraud_companion.adapters.kafka.consumer.Consumer")
def test_consumer_backpressure_labels_are_fixed_cardinality(
    mock_consumer_cls, _mock_uniform, _mock_sleep
):
    import json

    mock_consumer = MagicMock()
    mock_consumer_cls.return_value = mock_consumer

    class _FakeMessage:
        def headers(self):
            return [("event_type", b"case.created"), ("organization_id", b"org-1")]

        def value(self):
            return json.dumps(
                {"eventType": "case.created", "payload": {"caseId": "case-1"}}
            ).encode("utf-8")

        def key(self):
            return b"k"

        def topic(self):
            return "outbox.events"

        def offset(self):
            return 1

        def partition(self):
            return 0

        def error(self):
            return None

    msg = _FakeMessage()
    mock_consumer.poll.return_value = msg

    settings = Settings(
        anti_fraud_base_url="https://api.example.com",
        anti_fraud_agent_api_key="agent-key",
        llm_api_key="google-key",
        kafka_bootstrap_servers="localhost:9092",
        kafka_group_id="fraud-companion",
        kafka_organization_id="org-1",
        kafka_retry_backoff_seconds=0.0,
    )
    sink = _RecordingSink()

    checks = {"n": 0}

    def stop():
        checks["n"] += 1
        return checks["n"] > 1

    with patch(
        "fraud_companion.adapters.kafka.consumer.handle_case_created",
        side_effect=AgentRateLimitedError("429"),
    ):
        consumer = OutboxConsumer(settings=settings, agent=MagicMock(), metrics=sink)
        consumer.run(should_stop=stop, poll_timeout=0)

    for call in sink.backpressure_calls + sink.provider_error_calls:
        assert call["kind"] in _ALLOWED_BACKPRESSURE_KINDS
        assert "case-1" not in call["kind"]
