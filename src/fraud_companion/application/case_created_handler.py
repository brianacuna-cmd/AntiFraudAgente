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
from fraud_companion.application.analysis_pack_port import AnalysisPackFetcher
from fraud_companion.application.metrics_port import MetricsSink, NoOpMetricsSink
from fraud_companion.domain.events import CASE_CREATED_EVENT

logger = logging.getLogger(__name__)

_TERMINAL_ERROR_TYPES = (CaseNotFoundError, CaseClosedError)

#: Name of the sole write tool; a completed run must show evidence it ran.
_PUT_AGENT_BRIEF_TOOL = "put_agent_brief"

#: Stable API error codes that mean the case can never be processed. If
#: ToolNode is configured to convert tool exceptions into error-status
#: ToolMessages (handle_tool_errors=True/str), these surface in message
#: content instead of propagating as CaseNotFoundError/CaseClosedError.
_TERMINAL_ERROR_CODES = ("CASE_NOT_FOUND", "CASE_CLOSED")


class BriefNotWrittenError(RuntimeError):
    """Raised when an agent run completed without error but produced no
    evidence that ``put_agent_brief`` was invoked.

    This is treated as retryable: reporting success would commit the Kafka
    offset and silently drop the case with no brief ever written. Raising
    lets the message redeliver, which is safe because ``put_agent_brief`` is
    idempotent (last-write-wins).
    """


def _tool_name(message: Any) -> Any:
    name = getattr(message, "name", None)
    if name is None and isinstance(message, dict):
        name = message.get("name")
    return name


def _tool_status(message: Any) -> Any:
    status = getattr(message, "status", None)
    if status is None and isinstance(message, dict):
        status = message.get("status")
    return status


def _iter_messages(result: Any) -> list[Any]:
    if isinstance(result, dict):
        return result.get("messages") or []
    return getattr(result, "messages", []) or []


def _message_content(message: Any) -> str:
    content = getattr(message, "content", None)
    if content is None and isinstance(message, dict):
        content = message.get("content")
    return content if isinstance(content, str) else str(content or "")


def _sum_token_usage(result: Any) -> dict[str, int]:
    """Aggregate ``usage_metadata`` across all AIMessages in the agent result.

    LangChain chat models attach ``usage_metadata`` (``input_tokens`` /
    ``output_tokens`` / ``total_tokens``) to each AIMessage. Summing them gives
    the per-case token cost so consumption can be observed and tuned. Messages
    without usage metadata (tool/human messages) contribute nothing.
    """
    totals = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    for message in _iter_messages(result):
        usage = getattr(message, "usage_metadata", None)
        if usage is None and isinstance(message, dict):
            usage = message.get("usage_metadata")
        if not isinstance(usage, dict):
            continue
        for key in totals:
            totals[key] += _coerce_token_count(usage.get(key))
    return totals


def _coerce_token_count(value: Any) -> int:
    """Best-effort int coercion for a token count; never raises.

    Provider/version variance can put a non-int-coercible value in
    ``usage_metadata``. Token accounting is observability, not correctness,
    so a bad value contributes 0 rather than crashing a successful case.
    """
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def _terminal_error_in_result(result: Any) -> bool:
    """True if the run surfaced a terminal error (case not found/closed) as an
    error-status ToolMessage rather than a raised exception.

    This is the defensive path for when ToolNode's ``handle_tool_errors`` is
    set to convert exceptions into error ToolMessages; in the default mode
    those terminal errors propagate and are caught by the ``except`` instead.
    """
    for message in _iter_messages(result):
        if _tool_status(message) != "error":
            continue
        content = _message_content(message)
        if any(code in content for code in _TERMINAL_ERROR_CODES):
            return True
    return False


def _brief_was_written(result: Any) -> bool:
    """Scan the agent's returned messages for proof that put_agent_brief ran
    *successfully*.

    Accepts both a LangGraph result dict (``{"messages": [...]}``) and any
    object exposing ``.messages``. Evidence is a ToolMessage carrying the
    write tool's ``name`` whose status is not ``"error"`` — a tool call that
    the agent requested but that came back with an error status is NOT a
    successful write and does not count.
    """
    for message in _iter_messages(result):
        if _tool_name(message) != _PUT_AGENT_BRIEF_TOOL:
            continue
        if _tool_status(message) == "error":
            continue
        return True
    return False


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


def _safe(fn: Any, *args: Any, **kwargs: Any) -> None:
    """Call a metrics-sink method, swallowing any exception it raises.

    Metrics are observability, not correctness: a broken/raising sink must
    never alter case-processing control flow (see "Emission Is
    Side-Effect-Only"). Mirrors the existing never-fail-the-case convention
    used for token-usage logging above.
    """
    try:
        fn(*args, **kwargs)
    except Exception:  # noqa: BLE001 - metrics emission must never fail the case
        logger.warning("Metrics emission failed via %r; continuing.", fn)


def _extract_case_id(event: dict[str, Any]) -> str:
    payload = event.get("payload")
    if not isinstance(payload, dict):
        raise ValueError("case.created event payload is missing or not an object")
    case_id = payload.get("caseId")
    if not case_id:
        raise ValueError("case.created event payload is missing caseId")
    return str(case_id)


def handle_case_created(
    event: dict[str, Any],
    agent: Any,
    metrics: MetricsSink = NoOpMetricsSink(),
    pack_fetcher: AnalysisPackFetcher | None = None,
) -> HandleResult:
    """Handle one parsed ``case.created`` event envelope.

    ``agent`` is the compiled LangGraph agent returned by
    ``adapters.llm.agent.build_agent`` (or a compatible object exposing
    ``.invoke(dict) -> dict``, as LangGraph's ``create_agent`` result
    does). It is invoked once with an initial human message instructing
    it to process the given caseId.

    When ``pack_fetcher`` is ``None`` (default), the initial message does
    not include the analysis pack and the agent is expected to self-fetch
    via the ``get_analysis_pack`` tool (legacy two-round behaviour).

    When ``pack_fetcher`` is provided, it is called exactly once for the
    caseId BEFORE ``agent.invoke``, and its framed+trimmed pack content is
    injected directly into the initial human message so the agent can go
    straight to ``put_agent_brief`` (single model round). If the fetcher
    raises a terminal error (``CaseNotFoundError``/``CaseClosedError``),
    the case is reported ``SKIPPED_TERMINAL`` without ever invoking the
    agent. Any other fetcher exception propagates unchanged (retryable),
    also without invoking the agent.

    Returns a :class:`HandleResult` describing the outcome so the future
    Kafka consumer can decide whether to commit the offset.
    """
    if not isinstance(event, dict):
        # Valid JSON that parsed to a non-object (array/string/number/null).
        # Do NOT let event.get(...) raise AttributeError past the consumer:
        # that crashes the poll loop and redelivers forever. Commit-safe skip.
        logger.warning(
            "case.created envelope is not a JSON object (%s); skipping to "
            "avoid infinite redelivery.",
            type(event).__name__,
        )
        return HandleResult.SKIPPED_MALFORMED

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

    if pack_fetcher is not None:
        try:
            framed_pack = pack_fetcher(case_id)
        except _TERMINAL_ERROR_TYPES:
            logger.info(
                "Terminal outcome for caseId=%s: case not found or already "
                "closed while pre-fetching the analysis pack; not retrying.",
                case_id,
            )
            _safe(metrics.record_case_outcome, HandleResult.SKIPPED_TERMINAL.name)
            return HandleResult.SKIPPED_TERMINAL
        except Exception:
            logger.warning(
                "Retryable failure while pre-fetching the analysis pack for "
                "caseId=%s; propagating for redelivery.",
                case_id,
            )
            raise
        human_message = (
            "A new fraud case has been created with "
            f"caseId={case_id}. The analysis pack is provided below between "
            "untrusted-data markers; do NOT call get_analysis_pack, go "
            "straight to drafting and writing the agent brief via "
            f"put_agent_brief.\n\n{framed_pack}"
        )
    else:
        human_message = (
            "A new fraud case has been created with "
            f"caseId={case_id}. Analyze it and draft the "
            "agent brief for this case."
        )

    try:
        result = agent.invoke({"messages": [("human", human_message)]})
    except _TERMINAL_ERROR_TYPES:
        logger.info(
            "Terminal outcome for caseId=%s: case not found or already closed; "
            "not retrying.",
            case_id,
        )
        _safe(metrics.record_case_outcome, HandleResult.SKIPPED_TERMINAL.name)
        return HandleResult.SKIPPED_TERMINAL
    except Exception:
        logger.warning(
            "Retryable failure while processing caseId=%s; propagating for redelivery.",
            case_id,
        )
        raise

    try:
        usage = _sum_token_usage(result)
        logger.info(
            "Token usage for caseId=%s: input=%d output=%d total=%d",
            case_id,
            usage["input_tokens"],
            usage["output_tokens"],
            usage["total_tokens"],
        )
        _safe(
            metrics.observe_token_usage,
            input=usage["input_tokens"],
            output=usage["output_tokens"],
            total=usage["total_tokens"],
        )
    except Exception:  # noqa: BLE001 - usage logging must never fail the case
        logger.warning(
            "Failed to compute token usage for caseId=%s; continuing.", case_id
        )

    if _terminal_error_in_result(result):
        # A terminal error (case not found/closed) surfaced as an error-status
        # ToolMessage instead of propagating. The case can never succeed, so
        # commit rather than redeliver.
        logger.info(
            "Terminal error surfaced as ToolMessage for caseId=%s; not retrying.",
            case_id,
        )
        _safe(metrics.record_case_outcome, HandleResult.SKIPPED_TERMINAL.name)
        return HandleResult.SKIPPED_TERMINAL

    if not _brief_was_written(result):
        # The run finished without raising but never wrote the brief. Do NOT
        # report success (that would commit the offset and lose the case).
        logger.warning(
            "Agent run for caseId=%s completed without calling %s; forcing "
            "redelivery.",
            case_id,
            _PUT_AGENT_BRIEF_TOOL,
        )
        _safe(metrics.record_case_outcome, "brief_not_written")
        raise BriefNotWrittenError(
            f"agent did not call {_PUT_AGENT_BRIEF_TOOL} for caseId={case_id}"
        )

    logger.info("Processed case.created for caseId=%s", case_id)
    _safe(metrics.record_case_outcome, HandleResult.PROCESSED.name)
    return HandleResult.PROCESSED
