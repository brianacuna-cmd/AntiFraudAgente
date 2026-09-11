"""Tests for the Kafka consumer wrapper (Slice 7).

confluent_kafka's ``Consumer`` is entirely mocked here — no real broker.
Fake message objects only implement the subset of the confluent-kafka
``Message`` API the consumer actually uses: ``headers()``, ``value()``,
``key()``, ``error()``, ``topic()``, ``offset()``.
"""
from __future__ import annotations

import json
import math
import time
from unittest.mock import MagicMock, patch

import pytest

from fraud_companion.adapters.kafka.consumer import OutboxConsumer
from fraud_companion.application.case_created_handler import HandleResult
from fraud_companion.config import Settings


class _FakeClock:
    """A controllable stand-in for ``time.monotonic`` used by tests.

    Never advances on its own — tests call :meth:`advance` explicitly, so
    scenarios are fully deterministic and zero real time elapses.
    """

    def __init__(self, start: float = 0.0) -> None:
        self._t = start

    def __call__(self) -> float:
        return self._t

    def advance(self, seconds: float) -> None:
        self._t += seconds


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
        partition=0,
    ):
        self._headers = headers
        self._value = value
        self._key = key
        self._error = error
        self._topic = topic
        self._offset = offset
        self._partition = partition

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

    def partition(self):
        return self._partition


@pytest.fixture
def settings() -> Settings:
    return Settings(
        anti_fraud_base_url="https://api.example.com",
        anti_fraud_agent_api_key="agent-key",
        llm_api_key="google-key",
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

    mock_handle.assert_called_once_with(envelope, fake_agent, consumer._metrics, consumer._pack_fetcher)
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


def _retry_settings(**overrides) -> Settings:
    base = dict(
        anti_fraud_base_url="https://api.example.com",
        anti_fraud_agent_api_key="agent-key",
        llm_api_key="google-key",
        kafka_bootstrap_servers="localhost:9092",
        kafka_group_id="fraud-companion",
        kafka_organization_id="org-1",
        kafka_retry_backoff_seconds=0.0,  # no real sleeping in tests
    )
    base.update(overrides)
    return Settings(**base)


@patch("fraud_companion.adapters.kafka.consumer.Consumer")
def test_run_seeks_back_and_retries_failed_message_then_commits(
    mock_consumer_cls, fake_agent
):
    # On a retryable failure the loop must NOT advance past the message (which
    # with per-partition commit would silently skip/lose it). It must seek back
    # to the failed offset and retry; a later success then commits.
    mock_consumer = MagicMock()
    mock_consumer_cls.return_value = mock_consumer

    msg = _FakeMessage(
        headers=[("event_type", b"case.created"), ("organization_id", b"org-1")],
        value=json.dumps(_case_created_envelope()).encode("utf-8"),
        topic="outbox.events",
        partition=3,
        offset=42,
    )
    mock_consumer.poll.side_effect = [msg, msg]  # redelivery of the same message

    checks = {"n": 0}

    def stop() -> bool:
        checks["n"] += 1
        return checks["n"] > 2

    with patch(
        "fraud_companion.adapters.kafka.consumer.handle_case_created",
        side_effect=[RuntimeError("transient"), HandleResult.PROCESSED],
    ):
        consumer = OutboxConsumer(settings=_retry_settings(), agent=fake_agent)
        consumer.run(should_stop=stop, poll_timeout=0)  # must not raise

    # Sought back to the exact failed offset, and committed only after success.
    assert mock_consumer.seek.call_count == 1
    tp = mock_consumer.seek.call_args.args[0]
    assert (tp.topic, tp.partition, tp.offset) == ("outbox.events", 3, 42)
    mock_consumer.commit.assert_called_once_with(msg)


@patch("fraud_companion.adapters.kafka.consumer.Consumer")
def test_run_crashes_after_max_delivery_attempts_when_on_exhausted_crash(
    mock_consumer_cls, fake_agent
):
    mock_consumer = MagicMock()
    mock_consumer_cls.return_value = mock_consumer

    msg = _FakeMessage(
        headers=[("event_type", b"case.created"), ("organization_id", b"org-1")],
        value=json.dumps(_case_created_envelope()).encode("utf-8"),
        offset=5,
    )
    mock_consumer.poll.return_value = msg  # always the same failing message

    settings = _retry_settings(kafka_max_delivery_attempts=2, kafka_on_exhausted="crash")

    with patch(
        "fraud_companion.adapters.kafka.consumer.handle_case_created",
        side_effect=RuntimeError("permanent"),
    ):
        consumer = OutboxConsumer(settings=settings, agent=fake_agent)
        with pytest.raises(RuntimeError):
            consumer.run(should_stop=lambda: False, poll_timeout=0)

    # Never committed a message it could not process.
    mock_consumer.commit.assert_not_called()


@patch("fraud_companion.adapters.kafka.consumer.Consumer")
def test_run_skips_and_commits_after_max_attempts_when_on_exhausted_skip(
    mock_consumer_cls, fake_agent
):
    mock_consumer = MagicMock()
    mock_consumer_cls.return_value = mock_consumer

    msg = _FakeMessage(
        headers=[("event_type", b"case.created"), ("organization_id", b"org-1")],
        value=json.dumps(_case_created_envelope()).encode("utf-8"),
        offset=9,
    )
    mock_consumer.poll.return_value = msg

    settings = _retry_settings(kafka_max_delivery_attempts=2, kafka_on_exhausted="skip")

    checks = {"n": 0}

    def stop() -> bool:
        checks["n"] += 1
        return checks["n"] > 3  # give the loop room to exhaust then continue

    with patch(
        "fraud_companion.adapters.kafka.consumer.handle_case_created",
        side_effect=RuntimeError("permanent"),
    ):
        consumer = OutboxConsumer(settings=settings, agent=fake_agent)
        consumer.run(should_stop=stop, poll_timeout=0)  # must not raise

    # After exhausting attempts, the poison message is explicitly committed
    # (skipped) so the partition can progress.
    mock_consumer.commit.assert_called_once_with(msg)


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

    mock_handle.assert_called_once_with(envelope, fake_agent, consumer._metrics, consumer._pack_fetcher)
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
        llm_api_key="google-key",
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


# --- LLM provider backpressure (rate-limit / unavailable) — approach B -------

from fraud_companion.application.agent_port import (  # noqa: E402
    AgentRateLimitedError,
    AgentUnavailableError,
)


@patch("fraud_companion.adapters.kafka.consumer.Consumer")
def test_outbox_consumer_accepts_injectable_clock(mock_consumer_cls, settings, fake_agent):
    fake_clock = _FakeClock()

    consumer = OutboxConsumer(settings=settings, agent=fake_agent, clock=fake_clock)
    assert consumer._clock is fake_clock

    default_consumer = OutboxConsumer(settings=settings, agent=fake_agent)
    assert default_consumer._clock is time.monotonic


@patch("fraud_companion.adapters.kafka.consumer.Consumer")
def test_resume_elapsed_partitions_resumes_only_elapsed(mock_consumer_cls, settings, fake_agent):
    mock_consumer = MagicMock()
    mock_consumer_cls.return_value = mock_consumer

    fake_clock = _FakeClock(start=100.0)
    consumer = OutboxConsumer(settings=settings, agent=fake_agent, clock=fake_clock)

    consumer._paused_until[("outbox.events", 0)] = (100.0, 5)  # elapsed (<=)
    consumer._paused_until[("outbox.events", 1)] = (200.0, 9)  # not elapsed yet

    consumer._resume_elapsed_partitions()

    mock_consumer.resume.assert_called_once()
    resumed_tp = mock_consumer.resume.call_args.args[0][0]
    assert (resumed_tp.topic, resumed_tp.partition) == ("outbox.events", 0)
    seek_tp = mock_consumer.seek.call_args.args[0]
    assert (seek_tp.topic, seek_tp.partition, seek_tp.offset) == ("outbox.events", 0, 5)
    assert ("outbox.events", 0) not in consumer._paused_until
    assert ("outbox.events", 1) in consumer._paused_until


@patch("fraud_companion.adapters.kafka.consumer.time.sleep")
@patch("fraud_companion.adapters.kafka.consumer.random.uniform", return_value=0.0)
@patch("fraud_companion.adapters.kafka.consumer.Consumer")
def test_rate_limited_pauses_seeks_honors_retry_after_and_resumes_without_commit(
    mock_consumer_cls, _mock_uniform, mock_sleep, fake_agent
):
    mock_consumer = MagicMock()
    mock_consumer_cls.return_value = mock_consumer

    msg = _FakeMessage(
        headers=[("event_type", b"case.created"), ("organization_id", b"org-1")],
        value=json.dumps(_case_created_envelope()).encode("utf-8"),
        topic="outbox.events",
        partition=2,
        offset=7,
    )
    mock_consumer.poll.return_value = msg

    fake_clock = _FakeClock()
    checks = {"n": 0}

    def stop() -> bool:
        checks["n"] += 1
        if checks["n"] == 2:
            fake_clock.advance(5.0)  # elapse retry_after before the next iteration
        return checks["n"] > 2

    with patch(
        "fraud_companion.adapters.kafka.consumer.handle_case_created",
        side_effect=AgentRateLimitedError("429", retry_after=5.0),
    ):
        consumer = OutboxConsumer(settings=_retry_settings(), agent=fake_agent, clock=fake_clock)
        consumer.run(should_stop=stop, poll_timeout=0)  # must not raise

    # Paused the affected partition immediately — no blocking wait.
    paused = mock_consumer.pause.call_args.args[0]
    assert (paused[0].topic, paused[0].partition) == ("outbox.events", 2)
    # Sought back to the exact offset.
    tp = mock_consumer.seek.call_args_list[0].args[0]
    assert (tp.topic, tp.partition, tp.offset) == ("outbox.events", 2, 7)
    # Resumed only once the fake clock crossed the retry_after deadline.
    mock_consumer.resume.assert_called()
    # Never blocked on a real/fake sleep.
    mock_sleep.assert_not_called()
    # Backpressure is not the message's fault: never committed => no data loss.
    mock_consumer.commit.assert_not_called()


@patch("fraud_companion.adapters.kafka.consumer.time.sleep")
@patch("fraud_companion.adapters.kafka.consumer.random.uniform", return_value=0.0)
@patch("fraud_companion.adapters.kafka.consumer.Consumer")
def test_backpressure_never_calls_time_sleep(
    mock_consumer_cls, _mock_uniform, mock_sleep, fake_agent
):
    mock_consumer = MagicMock()
    mock_consumer_cls.return_value = mock_consumer

    msg = _FakeMessage(
        headers=[("event_type", b"case.created"), ("organization_id", b"org-1")],
        value=json.dumps(_case_created_envelope()).encode("utf-8"),
        topic="outbox.events",
        partition=0,
        offset=1,
    )
    mock_consumer.poll.return_value = msg

    fake_clock = _FakeClock()
    checks = {"n": 0}

    def stop() -> bool:
        checks["n"] += 1
        if checks["n"] in (2, 3):
            fake_clock.advance(100.0)  # always well past any computed delay
        return checks["n"] > 3

    with patch(
        "fraud_companion.adapters.kafka.consumer.handle_case_created",
        side_effect=AgentRateLimitedError("429", retry_after=5.0),
    ):
        consumer = OutboxConsumer(settings=_retry_settings(), agent=fake_agent, clock=fake_clock)
        consumer.run(should_stop=stop, poll_timeout=0)  # must not raise

    mock_sleep.assert_not_called()


@patch("fraud_companion.adapters.kafka.consumer.random.uniform", return_value=0.0)
@patch("fraud_companion.adapters.kafka.consumer.Consumer")
def test_backpressure_pauses_partition_without_blocking_other_partitions(
    mock_consumer_cls, _mock_uniform, fake_agent
):
    mock_consumer = MagicMock()
    mock_consumer_cls.return_value = mock_consumer

    msg_tp0 = _FakeMessage(
        headers=[("event_type", b"case.created"), ("organization_id", b"org-1")],
        value=json.dumps(_case_created_envelope("case-tp0")).encode("utf-8"),
        topic="outbox.events",
        partition=0,
        offset=1,
    )
    msg_tp1 = _FakeMessage(
        headers=[("event_type", b"case.created"), ("organization_id", b"org-1")],
        value=json.dumps(_case_created_envelope("case-tp1")).encode("utf-8"),
        topic="outbox.events",
        partition=1,
        offset=1,
    )
    mock_consumer.poll.side_effect = [msg_tp0, msg_tp1]

    checks = {"n": 0}

    def stop() -> bool:
        checks["n"] += 1
        return checks["n"] > 2

    def handle_side_effect(envelope, agent, metrics, pack_fetcher=None):
        if envelope["payload"]["caseId"] == "case-tp0":
            raise AgentRateLimitedError("429", retry_after=5.0)
        return HandleResult.PROCESSED

    with patch(
        "fraud_companion.adapters.kafka.consumer.handle_case_created",
        side_effect=handle_side_effect,
    ):
        consumer = OutboxConsumer(
            settings=_retry_settings(), agent=fake_agent, clock=_FakeClock()
        )
        consumer.run(should_stop=stop, poll_timeout=0)

    # tp0 stayed paused and uncommitted; tp1 was processed and committed —
    # proving tp0's backpressure never head-of-line-blocks other partitions.
    mock_consumer.commit.assert_called_once_with(msg_tp1)
    paused = mock_consumer.pause.call_args.args[0]
    assert (paused[0].topic, paused[0].partition) == ("outbox.events", 0)


@patch("fraud_companion.adapters.kafka.consumer.random.uniform", return_value=0.0)
@patch("fraud_companion.adapters.kafka.consumer.Consumer")
def test_backpressure_resumes_and_reprocesses_after_clock_elapses(
    mock_consumer_cls, _mock_uniform, fake_agent
):
    mock_consumer = MagicMock()
    mock_consumer_cls.return_value = mock_consumer

    msg = _FakeMessage(
        headers=[("event_type", b"case.created"), ("organization_id", b"org-1")],
        value=json.dumps(_case_created_envelope()).encode("utf-8"),
        topic="outbox.events",
        partition=0,
        offset=1,
    )
    mock_consumer.poll.return_value = msg

    resumed = {"v": False}

    def resume_side_effect(tps):
        resumed["v"] = True

    mock_consumer.resume.side_effect = resume_side_effect

    def handle_side_effect(envelope, agent, metrics, pack_fetcher=None):
        if not resumed["v"]:
            raise AgentRateLimitedError("429", retry_after=5.0)
        return HandleResult.PROCESSED

    fake_clock = _FakeClock()
    checks = {"n": 0}

    def stop() -> bool:
        checks["n"] += 1
        if checks["n"] == 3:
            fake_clock.advance(10.0)  # past the 5.0s retry_after deadline
        return checks["n"] > 3

    with patch(
        "fraud_companion.adapters.kafka.consumer.handle_case_created",
        side_effect=handle_side_effect,
    ):
        consumer = OutboxConsumer(settings=_retry_settings(), agent=fake_agent, clock=fake_clock)
        consumer.run(should_stop=stop, poll_timeout=0)

    mock_consumer.resume.assert_called_once()
    # Reprocessed at the same (seeked-back) offset once resumed.
    mock_consumer.commit.assert_called_once_with(msg)


@patch("fraud_companion.adapters.kafka.consumer.time.sleep")
@patch("fraud_companion.adapters.kafka.consumer.random.uniform", return_value=0.0)
@patch("fraud_companion.adapters.kafka.consumer.Consumer")
def test_unavailable_uses_exponential_backoff_when_no_retry_after(
    mock_consumer_cls, _mock_uniform, mock_sleep, fake_agent
):
    mock_consumer = MagicMock()
    mock_consumer_cls.return_value = mock_consumer

    msg = _FakeMessage(
        headers=[("event_type", b"case.created"), ("organization_id", b"org-1")],
        value=json.dumps(_case_created_envelope()).encode("utf-8"),
        offset=3,
    )
    mock_consumer.poll.return_value = msg

    fake_clock = _FakeClock()
    checks = {"n": 0}

    def stop() -> bool:
        checks["n"] += 1
        # After the first attempt parks the partition (resume_at = 0 + 2.0),
        # advance the clock past it so `_resume_elapsed_partitions()` resumes
        # the partition and the message is reprocessed for a second attempt.
        if checks["n"] == 2:
            fake_clock.advance(3)  # clock 0 -> 3, past the 2.0 deadline
        return checks["n"] > 3

    with patch(
        "fraud_companion.adapters.kafka.consumer.handle_case_created",
        side_effect=AgentUnavailableError("503"),
    ):
        consumer = OutboxConsumer(
            settings=_retry_settings(kafka_retry_backoff_seconds=2.0),
            agent=fake_agent,
            clock=fake_clock,
        )
        consumer.run(should_stop=stop, poll_timeout=0)

    # Exponential backoff base * 2^(attempt-1): attempt 1 delay = 2.0 (parked at
    # clock 0), attempt 2 delay = 4.0 (parked at clock 3 after resume) -> the
    # final recorded resume_at is 3 + 4.0 = 7.0, proving the delay grew to 4.0.
    key = ("outbox.events", 0)
    resume_at, offset = consumer._paused_until[key]
    assert resume_at == 7.0
    assert offset == 3
    # No blocking sleep anywhere on the backpressure path.
    mock_sleep.assert_not_called()


@patch("fraud_companion.adapters.kafka.consumer.time.sleep")
@patch("fraud_companion.adapters.kafka.consumer.random.uniform", return_value=0.0)
@patch("fraud_companion.adapters.kafka.consumer.Consumer")
def test_provider_backpressure_does_not_crash_or_skip_on_exhausted_budget(
    mock_consumer_cls, _mock_uniform, _mock_sleep, fake_agent
):
    # Even with a tiny permanent-failure budget, provider backpressure must NOT
    # trip crash/skip: it is transient and the case must survive.
    mock_consumer = MagicMock()
    mock_consumer_cls.return_value = mock_consumer

    msg = _FakeMessage(
        headers=[("event_type", b"case.created"), ("organization_id", b"org-1")],
        value=json.dumps(_case_created_envelope()).encode("utf-8"),
        offset=1,
    )
    mock_consumer.poll.return_value = msg

    checks = {"n": 0}

    def stop() -> bool:
        checks["n"] += 1
        return checks["n"] > 5  # many iterations, well past max_delivery_attempts

    settings = _retry_settings(kafka_max_delivery_attempts=2, kafka_on_exhausted="crash")
    with patch(
        "fraud_companion.adapters.kafka.consumer.handle_case_created",
        side_effect=AgentRateLimitedError("429"),
    ):
        consumer = OutboxConsumer(settings=settings, agent=fake_agent)
        consumer.run(should_stop=stop, poll_timeout=0)  # must NOT raise

    mock_consumer.commit.assert_not_called()  # no skip => no data loss


@patch("fraud_companion.adapters.kafka.consumer.time.sleep")
@patch("fraud_companion.adapters.kafka.consumer.random.uniform", return_value=0.0)
@patch("fraud_companion.adapters.kafka.consumer._BACKPRESSURE_MAX_ATTEMPTS", 3)
@patch("fraud_companion.adapters.kafka.consumer.Consumer")
def test_persistent_backpressure_escalates_loudly_after_cap_without_commit(
    mock_consumer_cls, _mock_uniform, _mock_sleep, fake_agent, caplog
):
    # A mislabelled-transient (permanent) 429/5xx must NOT retry forever: once
    # the backpressure cap is exceeded it escalates with a CRITICAL log and
    # PARKS that partition — but it must NOT raise out of run() or crash the
    # process, and it still never commits (no data loss — reprocessed on
    # restart).
    mock_consumer = MagicMock()
    mock_consumer_cls.return_value = mock_consumer

    msg = _FakeMessage(
        headers=[("event_type", b"case.created"), ("organization_id", b"org-1")],
        value=json.dumps(_case_created_envelope()).encode("utf-8"),
        offset=4,
    )
    mock_consumer.poll.return_value = msg

    checks = {"n": 0}

    def stop() -> bool:
        checks["n"] += 1
        # cap patched to 3: stop right after the 4th iteration (attempts=4>3
        # triggers escalation+park); further iterations would re-attempt from
        # a fresh (popped) counter and overwrite the parked state.
        return checks["n"] > 4

    with patch(
        "fraud_companion.adapters.kafka.consumer.handle_case_created",
        side_effect=AgentRateLimitedError("permanent 429"),
    ):
        consumer = OutboxConsumer(
            settings=_retry_settings(), agent=fake_agent, clock=_FakeClock()
        )
        with caplog.at_level("CRITICAL"):
            consumer.run(should_stop=stop, poll_timeout=0)  # must NOT raise

    assert "UNRESOLVED" in caplog.text
    mock_consumer.commit.assert_not_called()
    key = ("outbox.events", 0)
    resume_at, offset = consumer._paused_until[key]
    assert resume_at == math.inf
    assert offset == 4


@patch("fraud_companion.adapters.kafka.consumer.time.sleep")
@patch("fraud_companion.adapters.kafka.consumer.random.uniform", return_value=0.0)
@patch("fraud_companion.adapters.kafka.consumer._BACKPRESSURE_MAX_ATTEMPTS", 1)
@patch("fraud_companion.adapters.kafka.consumer.Consumer")
def test_backpressure_exhaustion_is_partition_scoped_no_process_crash(
    mock_consumer_cls, _mock_uniform, _mock_sleep, fake_agent, caplog
):
    # tp0 exhausts its backpressure budget while tp1 keeps flowing normally
    # in the SAME run() invocation — proving escalation is scoped to the
    # exhausted partition and never crashes/affects other partitions.
    mock_consumer = MagicMock()
    mock_consumer_cls.return_value = mock_consumer

    msg_tp0 = _FakeMessage(
        headers=[("event_type", b"case.created"), ("organization_id", b"org-1")],
        value=json.dumps(_case_created_envelope("case-tp0")).encode("utf-8"),
        topic="outbox.events",
        partition=0,
        offset=5,
    )
    msg_tp1 = _FakeMessage(
        headers=[("event_type", b"case.created"), ("organization_id", b"org-1")],
        value=json.dumps(_case_created_envelope("case-tp1")).encode("utf-8"),
        topic="outbox.events",
        partition=1,
        offset=1,
    )
    # tp0 is polled twice (attempt 1 then escalating attempt 2), tp1 is
    # polled once and interleaved after tp0 has already exceeded the cap.
    mock_consumer.poll.side_effect = [msg_tp0, msg_tp0, msg_tp1]

    checks = {"n": 0}

    def stop() -> bool:
        checks["n"] += 1
        return checks["n"] > 3

    def handle_side_effect(envelope, agent, metrics, pack_fetcher=None):
        if envelope["payload"]["caseId"] == "case-tp0":
            raise AgentRateLimitedError("429")
        return HandleResult.PROCESSED

    with patch(
        "fraud_companion.adapters.kafka.consumer.handle_case_created",
        side_effect=handle_side_effect,
    ):
        consumer = OutboxConsumer(
            settings=_retry_settings(), agent=fake_agent, clock=_FakeClock()
        )
        with caplog.at_level("CRITICAL"):
            consumer.run(should_stop=stop, poll_timeout=0)  # must NOT raise

    assert "UNRESOLVED" in caplog.text
    # tp1 kept processing/committing despite tp0's exhaustion.
    mock_consumer.commit.assert_called_once_with(msg_tp1)
    # tp0's exhausted offset is never committed (default kafka_on_exhausted).
    for call in mock_consumer.commit.call_args_list:
        committed_msg = call.args[0]
        assert committed_msg is not msg_tp0
    # tp0 parked forever (resume_at = inf) — never affects tp1's key.
    tp0_key = ("outbox.events", 0)
    resume_at, offset = consumer._paused_until[tp0_key]
    assert resume_at == math.inf
    assert offset == 5
    assert ("outbox.events", 1) not in consumer._paused_until


@patch("fraud_companion.adapters.kafka.consumer.time.sleep")
@patch("fraud_companion.adapters.kafka.consumer.random.uniform", return_value=0.0)
@patch("fraud_companion.adapters.kafka.consumer.Consumer")
def test_backpressure_respects_attempt_cap_and_never_commits_while_paused(
    mock_consumer_cls, _mock_uniform, _mock_sleep, fake_agent
):
    # Bounded retry + no-data-loss guarantees (R4a, R4b) must survive the
    # partition-scoped escalation rewrite: attempts still increment per
    # (topic,partition,offset) and are enforced before escalation, and the
    # offset is never committed while paused/retrying below the cap.
    mock_consumer = MagicMock()
    mock_consumer_cls.return_value = mock_consumer

    msg = _FakeMessage(
        headers=[("event_type", b"case.created"), ("organization_id", b"org-1")],
        value=json.dumps(_case_created_envelope()).encode("utf-8"),
        topic="outbox.events",
        partition=0,
        offset=2,
    )
    mock_consumer.poll.return_value = msg

    fake_clock = _FakeClock()
    checks = {"n": 0}

    def stop() -> bool:
        checks["n"] += 1
        return checks["n"] > 2  # two below-cap attempts; clock never advances

    with patch(
        "fraud_companion.adapters.kafka.consumer.handle_case_created",
        side_effect=AgentRateLimitedError("429"),
    ):
        consumer = OutboxConsumer(
            settings=_retry_settings(), agent=fake_agent, clock=fake_clock
        )
        consumer.run(should_stop=stop, poll_timeout=0)  # must not raise

    # Attempts tracked per (topic, partition, offset), below the (default 20)
    # cap, so no escalation yet.
    key = ("outbox.events", 0, 2)
    assert consumer._backpressure_attempts[key] == 2
    key_paused = ("outbox.events", 0)
    resume_at, offset = consumer._paused_until[key_paused]
    assert resume_at != math.inf  # still finite: not escalated/parked forever
    assert offset == 2
    # Never committed while paused/retrying below the cap.
    mock_consumer.commit.assert_not_called()


# --- Metrics emission (Slice A) ----------------------------------------------

from fraud_companion.application.metrics_port import NoOpMetricsSink  # noqa: E402


class _FakeMetricsSink:
    def __init__(self, *, raise_on_call: bool = False) -> None:
        self.raise_on_call = raise_on_call
        self.token_usage_calls: list[dict] = []
        self.case_outcome_calls: list[str] = []
        self.backpressure_calls: list[dict] = []
        self.provider_error_calls: list[dict] = []

    def observe_token_usage(self, *, input: int, output: int, total: int) -> None:
        pass

    def record_case_outcome(self, outcome: str) -> None:
        if self.raise_on_call:
            raise RuntimeError("boom")
        self.case_outcome_calls.append(outcome)

    def record_backpressure(self, *, kind: str, attempt: int, escalated: bool) -> None:
        if self.raise_on_call:
            raise RuntimeError("boom")
        self.backpressure_calls.append(
            {"kind": kind, "attempt": attempt, "escalated": escalated}
        )

    def record_provider_error(self, *, kind: str) -> None:
        if self.raise_on_call:
            raise RuntimeError("boom")
        self.provider_error_calls.append({"kind": kind})


@patch("fraud_companion.adapters.kafka.consumer.Consumer")
def test_outbox_consumer_defaults_to_noop_metrics_sink(mock_consumer_cls, settings, fake_agent):
    consumer = OutboxConsumer(settings=settings, agent=fake_agent)

    assert isinstance(consumer._metrics, NoOpMetricsSink)


@patch("fraud_companion.adapters.kafka.consumer.time.sleep")
@patch("fraud_companion.adapters.kafka.consumer.random.uniform", return_value=0.0)
@patch("fraud_companion.adapters.kafka.consumer.Consumer")
def test_rate_limited_retry_records_backpressure_and_provider_error(
    mock_consumer_cls, _mock_uniform, _mock_sleep, fake_agent
):
    mock_consumer = MagicMock()
    mock_consumer_cls.return_value = mock_consumer

    msg = _FakeMessage(
        headers=[("event_type", b"case.created"), ("organization_id", b"org-1")],
        value=json.dumps(_case_created_envelope()).encode("utf-8"),
        offset=7,
    )
    mock_consumer.poll.return_value = msg
    metrics = _FakeMetricsSink()

    checks = {"n": 0}

    def stop() -> bool:
        checks["n"] += 1
        return checks["n"] > 1

    with patch(
        "fraud_companion.adapters.kafka.consumer.handle_case_created",
        side_effect=AgentRateLimitedError("429", retry_after=5.0),
    ):
        consumer = OutboxConsumer(settings=_retry_settings(), agent=fake_agent, metrics=metrics)
        consumer.run(should_stop=stop, poll_timeout=0)

    assert metrics.backpressure_calls == [
        {"kind": "rate_limited", "attempt": 1, "escalated": False}
    ]
    assert metrics.provider_error_calls == [{"kind": "rate_limited"}]


@patch("fraud_companion.adapters.kafka.consumer.time.sleep")
@patch("fraud_companion.adapters.kafka.consumer.random.uniform", return_value=0.0)
@patch("fraud_companion.adapters.kafka.consumer._BACKPRESSURE_MAX_ATTEMPTS", 1)
@patch("fraud_companion.adapters.kafka.consumer.Consumer")
def test_backpressure_escalation_records_escalated_backpressure_and_outcome(
    mock_consumer_cls, _mock_uniform, _mock_sleep, fake_agent, caplog
):
    mock_consumer = MagicMock()
    mock_consumer_cls.return_value = mock_consumer

    msg = _FakeMessage(
        headers=[("event_type", b"case.created"), ("organization_id", b"org-1")],
        value=json.dumps(_case_created_envelope()).encode("utf-8"),
        offset=4,
    )
    mock_consumer.poll.return_value = msg
    metrics = _FakeMetricsSink()

    checks = {"n": 0}

    def stop() -> bool:
        checks["n"] += 1
        # cap patched to 1: stop right after the 2nd iteration (attempt=2>1
        # escalates+parks); further iterations would re-attempt fresh.
        return checks["n"] > 2

    with patch(
        "fraud_companion.adapters.kafka.consumer.handle_case_created",
        side_effect=AgentUnavailableError("503"),
    ):
        consumer = OutboxConsumer(
            settings=_retry_settings(), agent=fake_agent, metrics=metrics, clock=_FakeClock()
        )
        with caplog.at_level("CRITICAL"):
            consumer.run(should_stop=stop, poll_timeout=0)  # must NOT raise

    assert metrics.backpressure_calls == [
        {"kind": "unavailable", "attempt": 1, "escalated": False},
        {"kind": "unavailable", "attempt": 2, "escalated": True},
    ]
    assert metrics.case_outcome_calls == ["error"]


@patch("fraud_companion.adapters.kafka.consumer.Consumer")
def test_process_failure_exhaustion_records_outcome_error_once(
    mock_consumer_cls, fake_agent
):
    mock_consumer = MagicMock()
    mock_consumer_cls.return_value = mock_consumer

    msg = _FakeMessage(
        headers=[("event_type", b"case.created"), ("organization_id", b"org-1")],
        value=json.dumps(_case_created_envelope()).encode("utf-8"),
        offset=9,
    )
    mock_consumer.poll.return_value = msg
    metrics = _FakeMetricsSink()

    settings = _retry_settings(kafka_max_delivery_attempts=2, kafka_on_exhausted="skip")

    checks = {"n": 0}

    def stop() -> bool:
        checks["n"] += 1
        return checks["n"] > 3

    with patch(
        "fraud_companion.adapters.kafka.consumer.handle_case_created",
        side_effect=RuntimeError("permanent"),
    ):
        consumer = OutboxConsumer(settings=settings, agent=fake_agent, metrics=metrics)
        consumer.run(should_stop=stop, poll_timeout=0)

    assert metrics.case_outcome_calls == ["error"]


@patch("fraud_companion.adapters.kafka.consumer.Consumer")
def test_normal_processed_path_does_not_double_emit_outcome(
    mock_consumer_cls, settings, fake_agent
):
    # The handler owns the outcome for its own HandleResult; the consumer must
    # not also emit it for the same success path (avoids double-count).
    mock_consumer = MagicMock()
    mock_consumer_cls.return_value = mock_consumer

    envelope = _case_created_envelope()
    msg = _FakeMessage(
        headers=[("event_type", b"case.created"), ("organization_id", b"org-1")],
        value=json.dumps(envelope).encode("utf-8"),
    )
    metrics = _FakeMetricsSink()

    with patch(
        "fraud_companion.adapters.kafka.consumer.handle_case_created",
        return_value=HandleResult.PROCESSED,
    ):
        consumer = OutboxConsumer(settings=settings, agent=fake_agent, metrics=metrics)
        consumer.process_message(msg)

    assert metrics.case_outcome_calls == []


@patch("fraud_companion.adapters.kafka.consumer.Consumer")
def test_process_message_passes_metrics_into_handler(mock_consumer_cls, settings, fake_agent):
    mock_consumer = MagicMock()
    mock_consumer_cls.return_value = mock_consumer

    envelope = _case_created_envelope()
    msg = _FakeMessage(
        headers=[("event_type", b"case.created"), ("organization_id", b"org-1")],
        value=json.dumps(envelope).encode("utf-8"),
    )
    metrics = _FakeMetricsSink()

    with patch(
        "fraud_companion.adapters.kafka.consumer.handle_case_created",
        return_value=HandleResult.PROCESSED,
    ) as mock_handle:
        consumer = OutboxConsumer(settings=settings, agent=fake_agent, metrics=metrics)
        consumer.process_message(msg)

    mock_handle.assert_called_once_with(envelope, fake_agent, metrics, None)


@patch("fraud_companion.adapters.kafka.consumer.time.sleep")
@patch("fraud_companion.adapters.kafka.consumer.random.uniform", return_value=0.0)
@patch("fraud_companion.adapters.kafka.consumer.Consumer")
def test_raising_metrics_sink_does_not_break_backpressure_handling(
    mock_consumer_cls, _mock_uniform, _mock_sleep, fake_agent
):
    mock_consumer = MagicMock()
    mock_consumer_cls.return_value = mock_consumer

    msg = _FakeMessage(
        headers=[("event_type", b"case.created"), ("organization_id", b"org-1")],
        value=json.dumps(_case_created_envelope()).encode("utf-8"),
        offset=7,
    )
    mock_consumer.poll.return_value = msg
    metrics = _FakeMetricsSink(raise_on_call=True)

    checks = {"n": 0}

    def stop() -> bool:
        checks["n"] += 1
        return checks["n"] > 1

    with patch(
        "fraud_companion.adapters.kafka.consumer.handle_case_created",
        side_effect=AgentRateLimitedError("429", retry_after=5.0),
    ):
        consumer = OutboxConsumer(settings=_retry_settings(), agent=fake_agent, metrics=metrics)
        consumer.run(should_stop=stop, poll_timeout=0)  # must not raise

    mock_consumer.commit.assert_not_called()


@patch("fraud_companion.adapters.kafka.consumer.time.sleep")
@patch("fraud_companion.adapters.kafka.consumer.Consumer")
def test_parked_partition_message_is_skipped_even_if_pause_is_ignored(
    mock_consumer_cls, _mock_sleep, fake_agent
):
    # In-process park guard: even if the broker keeps redelivering a parked
    # partition's message (pause() ineffective), the consumer must NOT
    # re-process it — it seeks back and skips, never committing.
    mock_consumer = MagicMock()
    mock_consumer_cls.return_value = mock_consumer

    msg = _FakeMessage(
        headers=[("event_type", b"case.created"), ("organization_id", b"org-1")],
        value=json.dumps(_case_created_envelope()).encode("utf-8"),
        topic="outbox.events",
        partition=1,
        offset=9,
    )
    mock_consumer.poll.return_value = msg  # broker ignores pause(), keeps redelivering

    consumer = OutboxConsumer(settings=_retry_settings(), agent=fake_agent)
    # Pre-park the partition (as escalation would): resume_at=inf.
    consumer._paused_until[("outbox.events", 1)] = (float("inf"), 9)

    calls = {"n": 0}

    def stop() -> bool:
        calls["n"] += 1
        return calls["n"] > 3

    with patch(
        "fraud_companion.adapters.kafka.consumer.handle_case_created"
    ) as mock_handle:
        consumer.run(should_stop=stop, poll_timeout=0)

    # Never processed and never committed the parked partition's message.
    mock_handle.assert_not_called()
    mock_consumer.commit.assert_not_called()
    # Re-seeks back to the parked offset to hold position.
    assert mock_consumer.seek.called
    tp = mock_consumer.seek.call_args.args[0]
    assert (tp.topic, tp.partition, tp.offset) == ("outbox.events", 1, 9)


@patch("fraud_companion.adapters.kafka.consumer.Consumer")
def test_process_message_threads_pack_fetcher_into_handler(mock_consumer_cls, settings, fake_agent):
    mock_consumer = MagicMock()
    mock_consumer_cls.return_value = mock_consumer

    envelope = _case_created_envelope()
    msg = _FakeMessage(
        headers=[("event_type", b"case.created"), ("organization_id", b"org-1")],
        value=json.dumps(envelope).encode("utf-8"),
    )
    metrics = _FakeMetricsSink()
    pack_fetcher = object()

    with patch(
        "fraud_companion.adapters.kafka.consumer.handle_case_created",
        return_value=HandleResult.PROCESSED,
    ) as mock_handle:
        consumer = OutboxConsumer(
            settings=settings, agent=fake_agent, metrics=metrics, pack_fetcher=pack_fetcher
        )
        consumer.process_message(msg)

    mock_handle.assert_called_once_with(envelope, fake_agent, metrics, pack_fetcher)
