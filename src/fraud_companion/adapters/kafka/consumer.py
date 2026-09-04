"""Kafka outbox consumer (Slice 7).

Wraps ``confluent_kafka.Consumer`` with:

- Manual offset commit only (``enable.auto.commit=false``) — at-least-once
  delivery semantics. The offset is committed only after a message has been
  fully handled (successfully processed, or determined not-for-us / terminal
  / malformed). If ``handle_case_created`` raises a retryable error, the
  offset is *not* committed; the poll loop seeks back to that offset and
  retries with bounded backoff (``kafka_max_delivery_attempts`` /
  ``kafka_retry_backoff_seconds``) so a later success cannot overtake and
  silently drop the failed message. On exhaustion it takes the configured
  ``kafka_on_exhausted`` action (``crash`` — default, surfaces loudly; or
  ``skip`` — commit and drop with a CRITICAL log; ``dead_letter`` is future
  work, see docs/design/consumer-error-handling.md).
- A header-based filter: only messages whose ``event_type`` header equals
  ``CASE_CREATED_EVENT`` are routed to :func:`handle_case_created`. Every
  other message is treated as "not ours" and its offset is committed
  immediately so it is never redelivered forever.
- Poison-message safety: a ``case.created`` message with malformed JSON (or
  otherwise unparseable) is logged and its offset is committed rather than
  retried indefinitely.

The agent instance is injected by the caller (the future entrypoint); this
module never constructs one.
"""
from __future__ import annotations

import json
import logging
import random
import time
from typing import Any

from confluent_kafka import Consumer, KafkaError, Message, TopicPartition

from fraud_companion.application.agent_port import (
    AgentRateLimitedError,
    AgentUnavailableError,
)
from fraud_companion.application.case_created_handler import handle_case_created
from fraud_companion.config import Settings
from fraud_companion.domain.events import CASE_CREATED_EVENT

logger = logging.getLogger(__name__)

_EVENT_TYPE_HEADER = "event_type"
_ORGANIZATION_ID_HEADER = "organization_id"

#: Provider backpressure (rate-limit / unavailable) is transient but can last
#: minutes. These bound the wait so a single retry never sleeps unboundedly.
_BACKPRESSURE_MAX_SECONDS = 60.0
_BACKPRESSURE_JITTER_FRACTION = 0.1
#: Upper bound on consecutive backpressure retries for one offset. Generous
#: (with the 60s cap this spans well over an hour) so genuinely transient
#: provider outages ride through, but bounded so a MISLABELLED-transient
#: condition (revoked key, permanently exhausted quota, sustained outage) is
#: escalated loudly instead of stalling the partition forever.
_BACKPRESSURE_MAX_ATTEMPTS = 20

#: Errors signalling the LLM provider is applying backpressure — retried on a
#: separate, patient path that never counts against the permanent-failure
#: budget (so a transient outage cannot crash the service or drop a case).
_PROVIDER_BACKPRESSURE_ERRORS = (AgentRateLimitedError, AgentUnavailableError)


def _decode_headers(headers: list[tuple[str, bytes]] | None) -> dict[str, str]:
    """Decode confluent-kafka's ``list[(key, bytes)]`` headers into a dict.

    Returns an empty dict for ``None``/empty headers (never raises).
    """
    if not headers:
        return {}

    decoded: dict[str, str] = {}
    for key, raw_value in headers:
        if raw_value is None:
            continue
        try:
            decoded[key] = raw_value.decode("utf-8")
        except (UnicodeDecodeError, AttributeError):
            # Non-decodable header value — ignore it rather than crash.
            continue
    return decoded


class OutboxConsumer:
    """Consumes ``case.created`` events from the outbox topic."""

    def __init__(self, *, settings: Settings, agent: Any) -> None:
        self._settings = settings
        self._agent = agent
        conf = {
            "bootstrap.servers": settings.kafka_bootstrap_servers,
            "group.id": settings.kafka_group_id,
            "enable.auto.commit": False,
        }
        self._consumer = Consumer(conf)
        self._consumer.subscribe([settings.kafka_outbox_topic])
        # Delivery attempt counter keyed by (topic, partition, offset). Bounds
        # retries of a message that keeps failing so a mis-classified permanent
        # error cannot redeliver forever.
        self._delivery_attempts: dict[tuple[str, int, int], int] = {}
        self._backpressure_attempts: dict[tuple[str, int, int], int] = {}

    def process_message(self, msg: Message) -> None:
        """Handle a single already-polled, error-free Kafka message.

        Header filter: only ``event_type == CASE_CREATED_EVENT`` messages
        are routed to :func:`handle_case_created`. Anything else (wrong
        event type, missing/empty headers) is skipped and its offset
        committed immediately.
        """
        headers = _decode_headers(msg.headers())
        event_type = headers.get(_EVENT_TYPE_HEADER)
        organization_id = headers.get(_ORGANIZATION_ID_HEADER)

        if event_type != CASE_CREATED_EVENT:
            logger.info(
                "Skipping message with event_type=%r (key=%r) — not %r",
                event_type,
                msg.key(),
                CASE_CREATED_EVENT,
            )
            self._consumer.commit(msg)
            return

        configured_org_id = self._settings.kafka_organization_id
        if not configured_org_id:
            logger.warning(
                "kafka_organization_id is not configured; refusing to process "
                "case.created message (key=%r) to avoid cross-tenant processing.",
                msg.key(),
            )
            self._consumer.commit(msg)
            return

        if organization_id != configured_org_id:
            logger.info(
                "Skipping case.created message (key=%r) with non-matching "
                "organization_id=%r (expected %r).",
                msg.key(),
                organization_id,
                configured_org_id,
            )
            self._consumer.commit(msg)
            return

        try:
            envelope = json.loads(msg.value())
        except (json.JSONDecodeError, TypeError, ValueError):
            logger.warning(
                "Malformed JSON payload for case.created message (key=%r, "
                "organization_id=%r); committing to avoid infinite redelivery.",
                msg.key(),
                organization_id,
            )
            self._consumer.commit(msg)
            return

        # A raised exception here propagates unchanged (no commit below),
        # so Kafka redelivers this message — at-least-once, and safe
        # because handle_case_created / put_agent_brief is idempotent.
        handle_case_created(envelope, self._agent)

        self._consumer.commit(msg)

    def run(self, *, should_stop: Any = None, poll_timeout: float = 1.0) -> None:
        """Poll loop. ``should_stop`` is an optional zero-arg callable that
        returns ``True`` when the loop should exit — this makes the loop
        testable (run one or a few iterations) without an infinite ``while
        True``.
        """
        def _default_stop() -> bool:
            return False

        stop = should_stop or _default_stop

        while not stop():
            msg = self._consumer.poll(poll_timeout)
            if msg is None:
                continue

            err = msg.error()
            if err is not None:
                if err.code() == KafkaError._PARTITION_EOF:
                    continue
                logger.error("Kafka consumer error: %s", err)
                continue

            key = (msg.topic(), msg.partition(), msg.offset())
            try:
                self.process_message(msg)
            except _PROVIDER_BACKPRESSURE_ERRORS as exc:
                self._on_provider_backpressure(msg, key, exc)
                continue
            except Exception:
                self._on_process_failure(msg, key)
                continue
            else:
                # Success: clear any retry bookkeeping for this offset.
                self._delivery_attempts.pop(key, None)
                self._backpressure_attempts.pop(key, None)

    def _on_provider_backpressure(
        self, msg: Message, key: tuple[str, int, int], exc: Exception
    ) -> None:
        """Handle transient LLM-provider backpressure (rate-limit / unavailable).

        Unlike a per-message processing failure, this is NOT the message's
        fault and can outlast the small per-message retry budget. Counting it
        against ``kafka_max_delivery_attempts`` would crash or drop the case
        within seconds of a provider outage. Instead we pause the partition,
        wait — honouring the provider's ``retry_after`` when supplied, else
        exponential backoff with jitter capped at ``_BACKPRESSURE_MAX_SECONDS``
        — seek back, and resume. The offset is never committed, so a transient
        outage costs latency, never a lost case.
        """
        topic, partition, offset = key
        attempts = self._backpressure_attempts.get(key, 0) + 1
        self._backpressure_attempts[key] = attempts

        if attempts > _BACKPRESSURE_MAX_ATTEMPTS:
            # A "transient" condition that never clears is not transient. Stop
            # stalling this partition silently: drop the bookkeeping and
            # escalate loudly. The offset is still uncommitted, so on restart
            # the case is reprocessed — nothing is lost, but the failure is now
            # operator-visible instead of an endless WARNING loop.
            self._backpressure_attempts.pop(key, None)
            logger.critical(
                "LLM provider backpressure (%s) on key=%r UNRESOLVED after %d "
                "attempts (%s[%d]@%d); escalating as a non-transient failure.",
                type(exc).__name__, msg.key(), _BACKPRESSURE_MAX_ATTEMPTS,
                topic, partition, offset,
            )
            raise exc

        delay = self._backpressure_delay(exc, attempts)

        logger.warning(
            "LLM provider backpressure (%s) on key=%r; pausing %s[%d] and "
            "retrying offset %d in %.1fs (backpressure attempt %d).",
            type(exc).__name__, msg.key(), topic, partition, offset, delay, attempts,
        )

        affected = TopicPartition(topic, partition)
        self._consumer.pause([affected])
        self._consumer.seek(TopicPartition(topic, partition, offset))
        time.sleep(delay)
        self._consumer.resume([affected])

    def _backpressure_delay(self, exc: Exception, attempts: int) -> float:
        """Seconds to wait before retrying under provider backpressure.

        Honours a positive ``retry_after`` hint when present; otherwise uses
        exponential backoff off ``kafka_retry_backoff_seconds``. The result is
        capped, then a small random jitter is added to avoid a thundering herd
        of consumers retrying in lock-step.
        """
        retry_after = getattr(exc, "retry_after", None)
        if isinstance(retry_after, (int, float)) and retry_after > 0:
            base = float(retry_after)
        else:
            base = self._settings.kafka_retry_backoff_seconds * (2 ** (attempts - 1))
        base = min(base, _BACKPRESSURE_MAX_SECONDS)
        return base + random.uniform(0, base * _BACKPRESSURE_JITTER_FRACTION)

    def _on_process_failure(self, msg: Message, key: tuple[str, int, int]) -> None:
        """Handle a per-message processing failure without advancing past it.

        Committing on success advances the per-partition offset, so simply
        continuing past an uncommitted failure would let a later success
        overtake it and silently drop the failed message. Instead we seek back
        to the failed offset and retry with bounded backoff; once the attempts
        are exhausted we take the configured terminal action.
        """
        attempts = self._delivery_attempts.get(key, 0) + 1
        self._delivery_attempts[key] = attempts
        topic, partition, offset = key

        if attempts >= self._settings.kafka_max_delivery_attempts:
            self._delivery_attempts.pop(key, None)
            if self._settings.kafka_on_exhausted == "skip":
                logger.critical(
                    "Message (key=%r, %s[%d]@%d) failed %d attempts; skipping "
                    "(committing) per kafka_on_exhausted=skip. DATA LOSS.",
                    msg.key(), topic, partition, offset, attempts,
                )
                self._consumer.commit(msg)
                return
            # Default: crash — surface a mis-classified permanent failure loudly
            # rather than silently drop or spin forever.
            logger.critical(
                "Message (key=%r, %s[%d]@%d) failed %d attempts; crashing per "
                "kafka_on_exhausted=crash.",
                msg.key(), topic, partition, offset, attempts,
            )
            raise

        logger.warning(
            "Processing failed (key=%r, %s[%d]@%d), attempt %d/%d; seeking back "
            "for redelivery.",
            msg.key(), topic, partition, offset, attempts,
            self._settings.kafka_max_delivery_attempts,
        )
        self._consumer.seek(TopicPartition(topic, partition, offset))
        time.sleep(self._settings.kafka_retry_backoff_seconds * attempts)

    def close(self) -> None:
        self._consumer.close()
