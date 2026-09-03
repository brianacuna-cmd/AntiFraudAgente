"""Kafka outbox consumer (Slice 7).

Wraps ``confluent_kafka.Consumer`` with:

- Manual offset commit only (``enable.auto.commit=false``) — at-least-once
  delivery semantics. The offset is committed only after a message has been
  fully handled (successfully processed, or determined not-for-us / terminal
  / malformed). If ``handle_case_created`` raises a retryable error, the
  offset is *not* committed and Kafka will redeliver the message.
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
from typing import Any

from confluent_kafka import Consumer, KafkaError, Message

from fraud_companion.application.case_created_handler import handle_case_created
from fraud_companion.config import Settings
from fraud_companion.domain.events import CASE_CREATED_EVENT

logger = logging.getLogger(__name__)

_EVENT_TYPE_HEADER = "event_type"
_ORGANIZATION_ID_HEADER = "organization_id"


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

            self.process_message(msg)

    def close(self) -> None:
        self._consumer.close()
