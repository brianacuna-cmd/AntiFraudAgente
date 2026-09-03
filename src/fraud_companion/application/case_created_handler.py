"""``case.created`` application handler (Slice 8).

Pure application-layer orchestration between the future Kafka consumer and
the compiled Gemini agent (``adapters.llm.agent.build_agent``). This module
is intentionally framework-agnostic: it never imports ``confluent_kafka``,
so it can be unit-tested and reused regardless of the transport.

The handler does NOT call the anti-fraud HTTP tools directly. It drives the
agent with an initial instruction naming the ``caseId``; the agent itself
is responsible for calling ``get_analysis_pack`` then ``put_agent_brief``.
The case always stays OPEN — the handler never attempts a status change,
resolve, enforce, or SAR action (no such tools exist in the allow-list).

Idempotency: Kafka delivery is at-least-once, so the same ``caseId`` may be
handled more than once. ``put_agent_brief`` is last-write-wins on the
backend, so replaying this handler for the same case has no harmful side
effect and requires no local dedupe state.

Error contract (see design.md "Error Handling & Retry"):
  - ``CaseNotFoundError`` (404) and ``CaseClosedError`` (409) are terminal:
    the case can never successfully be processed, so we log and return
    ``HandleResult.SKIPPED_TERMINAL`` rather than raising. The future
    consumer commits the offset in this case (poison-message safe, no
    infinite redelivery).
  - Every other error (transient network/5xx, 401/403 auth/config
    defects, or any unrecognized exception) propagates unchanged so the
    future consumer can skip the offset commit and let Kafka redeliver.
"""
from __future__ import annotations

import logging
from enum import Enum, auto
from typing import Any

from fraud_companion.adapters.http.errors import CaseClosedError, CaseNotFoundError
from fraud_companion.domain.events import CASE_CREATED_EVENT

logger = logging.getLogger(__name__)

_TERMINAL_ERROR_TYPES = (CaseNotFoundError, CaseClosedError)


class HandleResult(Enum):
    """Outcome of :func:`handle_case_created`, for the consumer to act on."""

    #: The agent ran successfully for this caseId.
    PROCESSED = auto()
    #: The event envelope was not a ``case.created`` event; nothing was done.
    SKIPPED_IGNORED = auto()
    #: The agent run hit a terminal, non-retryable error (case not found or
    #: already closed). Safe to commit the offset; will never succeed later.
    SKIPPED_TERMINAL = auto()
    #: The envelope was valid JSON but its payload has no usable caseId (or is
    #: not an object). This is a deterministic poison message that can never
    #: succeed, so it is safe (and required) to commit the offset rather than
    #: let it redeliver forever.
    SKIPPED_MALFORMED = auto()


def _extract_case_id(event: dict[str, Any]) -> str:
    payload = event.get("payload")
    if not isinstance(payload, dict):
        raise ValueError("case.created event payload is missing or not an object")
    case_id = payload.get("caseId")
    if not case_id:
        raise ValueError("case.created event payload is missing caseId")
    return str(case_id)


def handle_case_created(event: dict[str, Any], agent: Any) -> HandleResult:
    """Handle one parsed ``case.created`` event envelope.

    ``agent`` is the compiled LangGraph agent returned by
    ``adapters.llm.agent.build_agent`` (or a compatible object exposing
    ``.invoke(dict) -> dict``, as LangGraph's ``create_agent`` result
    does). It is invoked once with an initial human message instructing
    it to process the given caseId; the agent itself calls
    ``get_analysis_pack`` and ``put_agent_brief``.

    Returns a :class:`HandleResult` describing the outcome so the future
    Kafka consumer can decide whether to commit the offset.
    """
    event_type = event.get("eventType")
    if event_type != CASE_CREATED_EVENT:
        logger.info("Ignoring event of type %r (not %r)", event_type, CASE_CREATED_EVENT)
        return HandleResult.SKIPPED_IGNORED

    try:
        case_id = _extract_case_id(event)
    except ValueError as exc:
        # Deterministic poison message: valid JSON, unusable payload. Do NOT
        # raise (that would propagate past the consumer, skip the offset
        # commit, and make Kafka redeliver this message forever). Report it as
        # a committable, non-retryable outcome instead.
        logger.warning(
            "Malformed case.created envelope; skipping to avoid infinite "
            "redelivery: %s",
            exc,
        )
        return HandleResult.SKIPPED_MALFORMED

    try:
        agent.invoke(
            {
                "messages": [
                    (
                        "human",
                        (
                            "A new fraud case has been created with "
                            f"caseId={case_id}. Analyze it and draft the "
                            "agent brief for this case."
                        ),
                    )
                ]
            }
        )
    except _TERMINAL_ERROR_TYPES:
        logger.info(
            "Terminal outcome for caseId=%s: case not found or already closed; "
            "not retrying.",
            case_id,
        )
        return HandleResult.SKIPPED_TERMINAL
    except Exception:
        logger.warning(
            "Retryable failure while processing caseId=%s; propagating for redelivery.",
            case_id,
        )
        raise

    logger.info("Processed case.created for caseId=%s", case_id)
    return HandleResult.PROCESSED
