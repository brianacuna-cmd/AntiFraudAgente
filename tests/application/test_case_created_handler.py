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
