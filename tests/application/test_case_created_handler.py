"""Tests for the case.created application handler (Slice 8).

The handler is the pure application-layer orchestration between the future
Kafka consumer and the compiled Gemini agent. It never calls the anti-fraud
HTTP tools directly — it drives the agent, which is responsible for calling
``get_analysis_pack`` then ``put_agent_brief`` itself. No network, no real
LLM: the agent object is fully mocked.
"""
from __future__ import annotations

import pytest

from fraud_companion.adapters.http.errors import (
    ApiError,
    CaseClosedError,
    CaseNotFoundError,
    UnauthenticatedError,
)
from fraud_companion.application.case_created_handler import (
    BriefNotWrittenError,
    HandleResult,
    handle_case_created,
)
from fraud_companion.domain.events import CASE_CREATED_EVENT

CASE_ID = "507f1f77bcf86cd799439011"


class _ToolMessage:
    """Minimal stand-in for a LangChain ToolMessage (name + status + content)."""

    def __init__(self, name: str, status: str = "success", content: str = "") -> None:
        self.name = name
        self.status = status
        self.content = content


def _envelope(event_type: str = CASE_CREATED_EVENT, assigned_to: object = None) -> dict:
    return {
        "eventType": event_type,
        "payload": {
            "caseId": CASE_ID,
            "organizationId": "org-1",
            "customerId": "cust-1",
            "riskScore": 87,
            "status": "OPEN",
            "priority": "HIGH",
            "assignedTo": assigned_to,
            "createdAt": "2026-09-03T00:00:00Z",
        },
    }


class _FakeAgent:
    def __init__(
        self,
        raise_on_invoke: Exception | None = None,
        wrote_brief: bool = True,
    ) -> None:
        self.raise_on_invoke = raise_on_invoke
        self.wrote_brief = wrote_brief
        self.invocations: list[dict] = []

    def invoke(self, input_: dict) -> dict:
        self.invocations.append(input_)
        if self.raise_on_invoke is not None:
            raise self.raise_on_invoke
        messages: list[object] = []
        if self.wrote_brief:
            # A successful run leaves a ToolMessage proving put_agent_brief ran.
            messages.append(_ToolMessage("put_agent_brief"))
        return {"messages": messages}


class _UsageMessage:
    """Stand-in for an AIMessage carrying token usage_metadata."""

    def __init__(self, input_tokens: int, output_tokens: int) -> None:
        self.usage_metadata = {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
        }


def test_handle_case_created_logs_token_usage(caplog) -> None:
    class _UsageAgent(_FakeAgent):
        def invoke(self, input_: dict) -> dict:
            self.invocations.append(input_)
            return {
                "messages": [
                    _UsageMessage(1000, 200),
                    _UsageMessage(500, 100),
                    _ToolMessage("put_agent_brief"),
                ]
            }

    agent = _UsageAgent()

    with caplog.at_level("INFO"):
        result = handle_case_created(_envelope(), agent)

    assert result is HandleResult.PROCESSED
    logged = caplog.text
    # Aggregated across both AIMessages: input=1500, output=300, total=1800.
    assert "1500" in logged
    assert "300" in logged
    assert "1800" in logged


def test_handle_case_created_survives_non_numeric_token_usage() -> None:
    """A bad usage_metadata value must never turn a successful case into a
    crash: token accounting is observability, not correctness."""

    class _BadUsageMessage:
        usage_metadata = {"input_tokens": "not-a-number", "output_tokens": None}

    class _BadUsageAgent(_FakeAgent):
        def invoke(self, input_: dict) -> dict:
            self.invocations.append(input_)
            return {
                "messages": [_BadUsageMessage(), _ToolMessage("put_agent_brief")]
            }

    agent = _BadUsageAgent()

    result = handle_case_created(_envelope(), agent)

    assert result is HandleResult.PROCESSED


def test_handle_case_created_invokes_agent_once_with_case_id() -> None:
    agent = _FakeAgent()

    result = handle_case_created(_envelope(), agent)

    assert result is HandleResult.PROCESSED
    assert len(agent.invocations) == 1
    invoked_input = agent.invocations[0]
    # The caseId must appear somewhere in the initial message content sent
    # to the agent.
    serialized = str(invoked_input)
    assert CASE_ID in serialized


def test_handle_case_created_ignores_wrong_event_type() -> None:
    agent = _FakeAgent()

    result = handle_case_created(_envelope(event_type="case.updated"), agent)

    assert result is HandleResult.SKIPPED_IGNORED
    assert agent.invocations == []


def test_handle_case_created_is_idempotent_across_replays() -> None:
    agent = _FakeAgent()

    first = handle_case_created(_envelope(), agent)
    second = handle_case_created(_envelope(), agent)

    assert first is HandleResult.PROCESSED
    assert second is HandleResult.PROCESSED
    assert len(agent.invocations) == 2


@pytest.mark.parametrize("error_cls", [CaseNotFoundError, CaseClosedError])
def test_handle_case_created_treats_terminal_errors_as_processed(error_cls) -> None:
    error = error_cls(status=404, code="CASE_NOT_FOUND", message="not found")
    agent = _FakeAgent(raise_on_invoke=error)

    result = handle_case_created(_envelope(), agent)

    assert result is HandleResult.SKIPPED_TERMINAL


@pytest.mark.parametrize(
    "error",
    [
        ApiError(status=500, code=None, message="boom"),
        UnauthenticatedError(status=401, code="UNAUTHENTICATED", message="no"),
        ConnectionError("network down"),
    ],
)
def test_handle_case_created_reraises_retryable_errors(error) -> None:
    agent = _FakeAgent(raise_on_invoke=error)

    with pytest.raises(type(error)):
        handle_case_created(_envelope(), agent)


@pytest.mark.parametrize(
    "bad_payload",
    [
        {},  # missing caseId
        {"caseId": ""},  # empty caseId
        {"caseId": None},  # null caseId
        "not-a-dict",  # payload is not an object
        None,  # payload absent
        123,  # payload wrong type
    ],
)
def test_handle_case_created_skips_malformed_payload_without_raising(bad_payload) -> None:
    # A valid-JSON envelope whose payload has no usable caseId must NOT raise
    # (raising would propagate past the consumer, block the offset commit, and
    # make Kafka redeliver this poison message forever). It must be reported as
    # a committable, non-retryable outcome.
    agent = _FakeAgent()
    envelope = {"eventType": CASE_CREATED_EVENT, "payload": bad_payload}

    result = handle_case_created(envelope, agent)

    assert result is HandleResult.SKIPPED_MALFORMED
    assert agent.invocations == []


@pytest.mark.parametrize("bad_envelope", [[], "not-a-dict", 5, None, 3.14])
def test_handle_case_created_skips_non_dict_envelope(bad_envelope) -> None:
    # A case.created-headered message whose JSON body parses to a non-object
    # must not raise AttributeError (that would crash the consumer poll loop
    # and redeliver forever). It is a committable malformed outcome.
    agent = _FakeAgent()

    result = handle_case_created(bad_envelope, agent)

    assert result is HandleResult.SKIPPED_MALFORMED
    assert agent.invocations == []


@pytest.mark.parametrize(
    "code",
    ["CASE_NOT_FOUND", "CASE_CLOSED"],
)
def test_handle_case_created_maps_terminal_error_toolmessage_to_skipped_terminal(code) -> None:
    # If ToolNode is configured to convert tool exceptions into error-status
    # ToolMessages (handle_tool_errors=True/str), a terminal error (case not
    # found / closed) surfaces as an error ToolMessage and the run completes
    # without raising. It must be classified SKIPPED_TERMINAL (committable),
    # NOT BriefNotWrittenError (which would redeliver a case that can never
    # succeed). This keeps the pipeline correct regardless of the ToolNode
    # error-handling mode.
    class _TerminalToolMsgAgent(_FakeAgent):
        def invoke(self, input_: dict) -> dict:
            self.invocations.append(input_)
            return {
                "messages": [
                    _ToolMessage(
                        "get_analysis_pack",
                        status="error",
                        content=f"Error: CaseError('[4xx] {code}: nope')",
                    )
                ]
            }

    result = handle_case_created(_envelope(), _TerminalToolMsgAgent())

    assert result is HandleResult.SKIPPED_TERMINAL


def test_handle_case_created_raises_when_brief_tool_errored() -> None:
    # put_agent_brief ran but its ToolMessage came back with an error status
    # (e.g. the tool body raised and LangGraph surfaced it as an error-status
    # ToolMessage still carrying the tool name). This is NOT a successful write
    # and must force redelivery, not a false PROCESSED.
    class _ErrAgent(_FakeAgent):
        def invoke(self, input_: dict) -> dict:
            self.invocations.append(input_)
            return {"messages": [_ToolMessage("put_agent_brief", status="error")]}

    with pytest.raises(BriefNotWrittenError):
        handle_case_created(_envelope(), _ErrAgent())


def test_handle_case_created_raises_when_brief_not_written() -> None:
    # The agent run completed without error but never called put_agent_brief
    # (model reasoned without acting, or a tool failure was swallowed into a
    # ToolMessage). Reporting PROCESSED would commit the offset and silently
    # lose the case with no brief ever written. Must raise so the message is
    # redelivered (safe: put_agent_brief is idempotent / last-write-wins).
    agent = _FakeAgent(wrote_brief=False)

    with pytest.raises(BriefNotWrittenError):
        handle_case_created(_envelope(), agent)


def test_handle_case_created_processed_requires_brief_evidence() -> None:
    agent = _FakeAgent(wrote_brief=True)

    result = handle_case_created(_envelope(), agent)

    assert result is HandleResult.PROCESSED


def test_handle_case_created_accepts_string_or_null_assigned_to() -> None:
    agent = _FakeAgent()

    result_str = handle_case_created(_envelope(assigned_to="user-1"), agent)
    result_null = handle_case_created(_envelope(assigned_to=None), agent)

    assert result_str is HandleResult.PROCESSED
    assert result_null is HandleResult.PROCESSED


class _FakeMetricsSink:
    def __init__(self, *, raise_on_call: bool = False) -> None:
        self.raise_on_call = raise_on_call
        self.token_usage_calls: list[dict] = []
        self.case_outcome_calls: list[str] = []

    def observe_token_usage(self, *, input: int, output: int, total: int) -> None:
        if self.raise_on_call:
            raise RuntimeError("boom")
        self.token_usage_calls.append({"input": input, "output": output, "total": total})

    def record_case_outcome(self, outcome: str) -> None:
        if self.raise_on_call:
            raise RuntimeError("boom")
        self.case_outcome_calls.append(outcome)

    def record_backpressure(self, *, kind: str, attempt: int, escalated: bool) -> None:
        pass

    def record_provider_error(self, *, kind: str) -> None:
        pass


def test_handle_case_created_emits_token_usage_and_outcome() -> None:
    class _UsageAgent(_FakeAgent):
        def invoke(self, input_: dict) -> dict:
            self.invocations.append(input_)
            return {
                "messages": [
                    _UsageMessage(10, 5),
                    _ToolMessage("put_agent_brief"),
                ]
            }

    agent = _UsageAgent()
    metrics = _FakeMetricsSink()

    result = handle_case_created(_envelope(), agent, metrics=metrics)

    assert result is HandleResult.PROCESSED
    assert metrics.token_usage_calls == [{"input": 10, "output": 5, "total": 15}]
    assert metrics.case_outcome_calls == ["PROCESSED"]


def test_handle_case_created_emits_outcome_exactly_once_for_brief_not_written_error() -> None:
    agent = _FakeAgent(wrote_brief=False)
    metrics = _FakeMetricsSink()

    with pytest.raises(BriefNotWrittenError):
        handle_case_created(_envelope(), agent, metrics=metrics)

    assert metrics.case_outcome_calls == ["brief_not_written"]


def test_handle_case_created_works_without_metrics_param() -> None:
    agent = _FakeAgent()

    result = handle_case_created(_envelope(), agent)

    assert result is HandleResult.PROCESSED


def test_handle_case_created_survives_raising_metrics_sink() -> None:
    agent = _FakeAgent()
    metrics = _FakeMetricsSink(raise_on_call=True)

    result = handle_case_created(_envelope(), agent, metrics=metrics)

    assert result is HandleResult.PROCESSED


def test_handle_case_created_identical_outcome_with_noop_and_recording_sink() -> None:
    from fraud_companion.application.metrics_port import NoOpMetricsSink

    agent_a = _FakeAgent()
    agent_b = _FakeAgent()

    result_noop = handle_case_created(_envelope(), agent_a, metrics=NoOpMetricsSink())
    result_fake = handle_case_created(_envelope(), agent_b, metrics=_FakeMetricsSink())

    assert result_noop == result_fake
