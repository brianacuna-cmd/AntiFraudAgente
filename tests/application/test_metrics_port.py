"""Tests for the MetricsSink Protocol port and its no-op implementation."""
from __future__ import annotations

from fraud_companion.application.metrics_port import MetricsSink, NoOpMetricsSink


class FakeMetricsSink:
    """In-memory recording sink used across tests as a MetricsSink fake."""

    def __init__(self) -> None:
        self.token_usage_calls: list[dict] = []
        self.case_outcome_calls: list[str] = []
        self.backpressure_calls: list[dict] = []
        self.provider_error_calls: list[dict] = []

    def observe_token_usage(self, *, input: int, output: int, total: int) -> None:
        self.token_usage_calls.append({"input": input, "output": output, "total": total})

    def record_case_outcome(self, outcome: str) -> None:
        self.case_outcome_calls.append(outcome)

    def record_backpressure(self, *, kind: str, attempt: int, escalated: bool) -> None:
        self.backpressure_calls.append(
            {"kind": kind, "attempt": attempt, "escalated": escalated}
        )

    def record_provider_error(self, *, kind: str) -> None:
        self.provider_error_calls.append({"kind": kind})


def test_fake_sink_satisfies_metrics_sink_protocol():
    fake = FakeMetricsSink()

    assert isinstance(fake, MetricsSink)


def test_noop_sink_satisfies_metrics_sink_protocol():
    noop = NoOpMetricsSink()

    assert isinstance(noop, MetricsSink)


def test_noop_sink_all_methods_are_inert_on_valid_args():
    noop = NoOpMetricsSink()

    assert noop.observe_token_usage(input=1, output=2, total=3) is None
    assert noop.record_case_outcome("PROCESSED") is None
    assert noop.record_backpressure(kind="rate_limited", attempt=1, escalated=False) is None
    assert noop.record_provider_error(kind="unavailable") is None


def test_noop_sink_all_methods_are_inert_on_malformed_args():
    noop = NoOpMetricsSink()

    assert noop.observe_token_usage(input=-1, output=None, total="bad") is None
    assert noop.record_case_outcome(None) is None
    assert noop.record_backpressure(kind=None, attempt=-1, escalated="yes") is None
    assert noop.record_provider_error(kind=123) is None
