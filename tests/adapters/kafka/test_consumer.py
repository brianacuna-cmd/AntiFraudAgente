"""Tests for the Kafka consumer wrapper (Slice 7).

confluent_kafka's ``Consumer`` is entirely mocked here — no real broker.
Fake message objects only implement the subset of the confluent-kafka
``Message`` API the consumer actually uses: ``headers()``, ``value()``,
``key()``, ``error()``, ``topic()``, ``offset()``.
"""
from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from fraud_companion.adapters.kafka.consumer import OutboxConsumer
from fraud_companion.application.case_created_handler import HandleResult
from fraud_companion.config import Settings


class _FakeMessage:
    def __init__(
        self,
        *,
        headers=None,
        value=b"{}",
        key=b"some-event-id",
        error=None,
        topic="outbox.events",
        offset=0,
    ):
        self._headers = headers
        self._value = value
        self._key = key
        self._error = error
        self._topic = topic
        self._offset = offset

    def headers(self):
        return self._headers

    def value(self):
        return self._value

    def key(self):
        return self._key

    def error(self):
        return self._error

    def topic(self):
        return self._topic

    def offset(self):
        return self._offset


@pytest.fixture
def settings() -> Settings:
    return Settings(
        anti_fraud_base_url="https://api.example.com",
        anti_fraud_agent_api_key="agent-key",
        google_api_key="google-key",
        kafka_bootstrap_servers="localhost:9092",
        kafka_group_id="fraud-companion",
        kafka_organization_id="org-1",
    )


@pytest.fixture
def fake_agent():
    return MagicMock(name="agent")


def _case_created_envelope(case_id: str = "case-123") -> dict:
    return {"eventType": "case.created", "payload": {"caseId": case_id}}


@patch("fraud_companion.adapters.kafka.consumer.Consumer")
def test_case_created_message_triggers_handler_and_commits(
    mock_consumer_cls, settings, fake_agent
):
    mock_consumer = MagicMock()
    mock_consumer_cls.return_value = mock_consumer

    envelope = _case_created_envelope()
    msg = _FakeMessage(
        headers=[("event_type", b"case.created"), ("organization_id", b"org-1")],
        value=json.dumps(envelope).encode("utf-8"),
    )

    with patch(
        "fraud_companion.adapters.kafka.consumer.handle_case_created",
        return_value=HandleResult.PROCESSED,
    ) as mock_handle:
        consumer = OutboxConsumer(settings=settings, agent=fake_agent)
        consumer.process_message(msg)

    mock_handle.assert_called_once_with(envelope, fake_agent)
    mock_consumer.commit.assert_called_once_with(msg)


@patch("fraud_companion.adapters.kafka.consumer.Consumer")
def test_non_case_created_message_skips_handler_but_commits(
    mock_consumer_cls, settings, fake_agent
):
    mock_consumer = MagicMock()
    mock_consumer_cls.return_value = mock_consumer

    msg = _FakeMessage(
        headers=[("event_type", b"case.updated"), ("organization_id", b"org-1")],
        value=b'{"eventType": "case.updated"}',
    )

    with patch(
        "fraud_companion.adapters.kafka.consumer.handle_case_created"
    ) as mock_handle:
        consumer = OutboxConsumer(settings=settings, agent=fake_agent)
        consumer.process_message(msg)

    mock_handle.assert_not_called()
    mock_consumer.commit.assert_called_once_with(msg)


@patch("fraud_companion.adapters.kafka.consumer.Consumer")
def test_missing_headers_treated_as_non_matching_skipped_and_committed(
    mock_consumer_cls, settings, fake_agent
):
    mock_consumer = MagicMock()
    mock_consumer_cls.return_value = mock_consumer

    msg = _FakeMessage(headers=None, value=b"{}")

    with patch(
        "fraud_companion.adapters.kafka.consumer.handle_case_created"
    ) as mock_handle:
        consumer = OutboxConsumer(settings=settings, agent=fake_agent)
        consumer.process_message(msg)

    mock_handle.assert_not_called()
    mock_consumer.commit.assert_called_once_with(msg)


@patch("fraud_companion.adapters.kafka.consumer.Consumer")
def test_empty_headers_list_treated_as_non_matching(mock_consumer_cls, settings, fake_agent):
    mock_consumer = MagicMock()
    mock_consumer_cls.return_value = mock_consumer

    msg = _FakeMessage(headers=[], value=b"{}")

    with patch(
        "fraud_companion.adapters.kafka.consumer.handle_case_created"
    ) as mock_handle:
        consumer = OutboxConsumer(settings=settings, agent=fake_agent)
        consumer.process_message(msg)

    mock_handle.assert_not_called()
    mock_consumer.commit.assert_called_once_with(msg)


@patch("fraud_companion.adapters.kafka.consumer.Consumer")
def test_retryable_error_from_handler_does_not_commit(mock_consumer_cls, settings, fake_agent):
    mock_consumer = MagicMock()
    mock_consumer_cls.return_value = mock_consumer

    envelope = _case_created_envelope()
    msg = _FakeMessage(
        headers=[("event_type", b"case.created"), ("organization_id", b"org-1")],
        value=json.dumps(envelope).encode("utf-8"),
    )

    with patch(
        "fraud_companion.adapters.kafka.consumer.handle_case_created",
        side_effect=ConnectionError("transient network failure"),
    ):
        consumer = OutboxConsumer(settings=settings, agent=fake_agent)
        with pytest.raises(ConnectionError):
            consumer.process_message(msg)

    mock_consumer.commit.assert_not_called()


@pytest.mark.parametrize("result", [HandleResult.PROCESSED, HandleResult.SKIPPED_TERMINAL])
@patch("fraud_companion.adapters.kafka.consumer.Consumer")
def test_processed_or_skipped_terminal_commits(mock_consumer_cls, settings, fake_agent, result):
    mock_consumer = MagicMock()
    mock_consumer_cls.return_value = mock_consumer

    envelope = _case_created_envelope()
    msg = _FakeMessage(
        headers=[("event_type", b"case.created"), ("organization_id", b"org-1")],
        value=json.dumps(envelope).encode("utf-8"),
    )

    with patch(
        "fraud_companion.adapters.kafka.consumer.handle_case_created",
        return_value=result,
    ):
        consumer = OutboxConsumer(settings=settings, agent=fake_agent)
        consumer.process_message(msg)

    mock_consumer.commit.assert_called_once_with(msg)


@patch("fraud_companion.adapters.kafka.consumer.Consumer")
def test_malformed_json_value_on_case_created_is_logged_and_committed(
    mock_consumer_cls, settings, fake_agent
):
    mock_consumer = MagicMock()
    mock_consumer_cls.return_value = mock_consumer

    msg = _FakeMessage(
        headers=[("event_type", b"case.created"), ("organization_id", b"org-1")],
        value=b"not-valid-json{{{",
    )

    with patch(
        "fraud_companion.adapters.kafka.consumer.handle_case_created"
    ) as mock_handle:
        consumer = OutboxConsumer(settings=settings, agent=fake_agent)
        consumer.process_message(msg)

    mock_handle.assert_not_called()
    mock_consumer.commit.assert_called_once_with(msg)


@patch("fraud_companion.adapters.kafka.consumer.Consumer")
def test_non_matching_organization_id_skips_handler_but_commits(
    mock_consumer_cls, settings, fake_agent
):
    mock_consumer = MagicMock()
    mock_consumer_cls.return_value = mock_consumer

    envelope = _case_created_envelope()
    msg = _FakeMessage(
        headers=[("event_type", b"case.created"), ("organization_id", b"org-2")],
        value=json.dumps(envelope).encode("utf-8"),
    )

    with patch(
        "fraud_companion.adapters.kafka.consumer.handle_case_created"
    ) as mock_handle:
        consumer = OutboxConsumer(settings=settings, agent=fake_agent)
        consumer.process_message(msg)

    mock_handle.assert_not_called()
    mock_consumer.commit.assert_called_once_with(msg)


@patch("fraud_companion.adapters.kafka.consumer.Consumer")
def test_matching_organization_id_processes_as_before(mock_consumer_cls, settings, fake_agent):
    mock_consumer = MagicMock()
    mock_consumer_cls.return_value = mock_consumer

    envelope = _case_created_envelope()
    msg = _FakeMessage(
        headers=[("event_type", b"case.created"), ("organization_id", b"org-1")],
        value=json.dumps(envelope).encode("utf-8"),
    )

    with patch(
        "fraud_companion.adapters.kafka.consumer.handle_case_created",
        return_value=HandleResult.PROCESSED,
    ) as mock_handle:
        consumer = OutboxConsumer(settings=settings, agent=fake_agent)
        consumer.process_message(msg)

    mock_handle.assert_called_once_with(envelope, fake_agent)
    mock_consumer.commit.assert_called_once_with(msg)


@patch("fraud_companion.adapters.kafka.consumer.Consumer")
def test_missing_organization_id_header_on_case_created_skips_and_commits(
    mock_consumer_cls, settings, fake_agent
):
    mock_consumer = MagicMock()
    mock_consumer_cls.return_value = mock_consumer

    envelope = _case_created_envelope()
    msg = _FakeMessage(
        headers=[("event_type", b"case.created")],
        value=json.dumps(envelope).encode("utf-8"),
    )

    with patch(
        "fraud_companion.adapters.kafka.consumer.handle_case_created"
    ) as mock_handle:
        consumer = OutboxConsumer(settings=settings, agent=fake_agent)
        consumer.process_message(msg)

    mock_handle.assert_not_called()
    mock_consumer.commit.assert_called_once_with(msg)


@patch("fraud_companion.adapters.kafka.consumer.Consumer")
def test_unset_kafka_organization_id_config_skips_and_commits_even_on_match(
    mock_consumer_cls, fake_agent
):
    mock_consumer = MagicMock()
    mock_consumer_cls.return_value = mock_consumer

    unconfigured_settings = Settings(
        anti_fraud_base_url="https://api.example.com",
        anti_fraud_agent_api_key="agent-key",
        google_api_key="google-key",
        kafka_bootstrap_servers="localhost:9092",
        kafka_group_id="fraud-companion",
        kafka_organization_id="",
    )

    envelope = _case_created_envelope()
    msg = _FakeMessage(
        headers=[("event_type", b"case.created"), ("organization_id", b"org-1")],
        value=json.dumps(envelope).encode("utf-8"),
    )

    with patch(
        "fraud_companion.adapters.kafka.consumer.handle_case_created"
    ) as mock_handle:
        consumer = OutboxConsumer(settings=unconfigured_settings, agent=fake_agent)
        consumer.process_message(msg)

    mock_handle.assert_not_called()
    mock_consumer.commit.assert_called_once_with(msg)


@patch("fraud_companion.adapters.kafka.consumer.Consumer")
def test_enable_auto_commit_is_configured_false(mock_consumer_cls, settings, fake_agent):
    OutboxConsumer(settings=settings, agent=fake_agent)

    args, kwargs = mock_consumer_cls.call_args
    conf = args[0] if args else kwargs.get("conf")
    assert conf["enable.auto.commit"] is False
    assert conf["bootstrap.servers"] == "localhost:9092"
    assert conf["group.id"] == "fraud-companion"


@patch("fraud_companion.adapters.kafka.consumer.Consumer")
def test_subscribes_to_configured_outbox_topic(mock_consumer_cls, settings, fake_agent):
    mock_consumer = MagicMock()
    mock_consumer_cls.return_value = mock_consumer

    OutboxConsumer(settings=settings, agent=fake_agent)

    mock_consumer.subscribe.assert_called_once_with(["outbox.events"])
