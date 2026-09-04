"""Tests for the provider-agnostic agent port and its error taxonomy.

These errors are the seam that keeps the rest of the system (handler,
consumer) free of any provider-specific (Gemini/google.genai) knowledge:
adapters translate their SDK exceptions into these, and everything
downstream reacts only to these.
"""
from __future__ import annotations

from fraud_companion.application.agent_port import (
    AgentError,
    AgentRateLimitedError,
    AgentUnavailableError,
    CaseAnalystAgent,
)


def test_rate_limited_is_an_agent_error() -> None:
    assert issubclass(AgentRateLimitedError, AgentError)


def test_unavailable_is_an_agent_error() -> None:
    assert issubclass(AgentUnavailableError, AgentError)


def test_rate_limited_carries_optional_retry_after() -> None:
    err = AgentRateLimitedError("slow down", retry_after=12.5)
    assert err.retry_after == 12.5


def test_rate_limited_retry_after_defaults_to_none() -> None:
    assert AgentRateLimitedError("no hint").retry_after is None


def test_case_analyst_agent_is_satisfied_by_any_invoke_object() -> None:
    class _Duck:
        def invoke(self, payload):  # noqa: ANN001, ANN201
            return payload

    assert isinstance(_Duck(), CaseAnalystAgent)
