"""Unit tests for PrometheusMetricsSink (Slice B, task 12).

Every assertion reads values from the sink's own ``CollectorRegistry`` via
``get_sample_value`` — never the global default registry, and never a real
HTTP server or scrape. Integration-only concerns (a live ``/metrics``
endpoint) are explicitly out of scope for this suite.
"""
from __future__ import annotations

from fraud_companion.adapters.metrics.prometheus_sink import PrometheusMetricsSink


class TestPrometheusMetricsSinkTokenUsage:
    def test_observe_token_usage_increments_kind_labelled_counter(self) -> None:
        sink = PrometheusMetricsSink()

        sink.observe_token_usage(input=10, output=5, total=15)

        registry = sink.registry
        assert (
            registry.get_sample_value("fraud_llm_tokens_total", {"kind": "input"}) == 10
        )
        assert (
            registry.get_sample_value("fraud_llm_tokens_total", {"kind": "output"}) == 5
        )
        assert (
            registry.get_sample_value("fraud_llm_tokens_total", {"kind": "total"}) == 15
        )

    def test_observe_token_usage_accumulates_across_calls(self) -> None:
        sink = PrometheusMetricsSink()

        sink.observe_token_usage(input=10, output=5, total=15)
        sink.observe_token_usage(input=2, output=1, total=3)

        assert (
            sink.registry.get_sample_value("fraud_llm_tokens_total", {"kind": "input"})
            == 12
        )


class TestPrometheusMetricsSinkCaseOutcome:
    def test_record_case_outcome_increments_outcome_labelled_counter(self) -> None:
        sink = PrometheusMetricsSink()

        sink.record_case_outcome("PROCESSED")

        assert (
            sink.registry.get_sample_value(
                "fraud_case_outcome_total", {"outcome": "PROCESSED"}
            )
            == 1
        )

    def test_record_case_outcome_supports_error_outcome(self) -> None:
        sink = PrometheusMetricsSink()

        sink.record_case_outcome("error")

        assert (
            sink.registry.get_sample_value(
                "fraud_case_outcome_total", {"outcome": "error"}
            )
            == 1
        )


class TestPrometheusMetricsSinkBackpressure:
    def test_record_backpressure_maps_escalated_true_to_phase_escalated(self) -> None:
        sink = PrometheusMetricsSink()

        sink.record_backpressure(kind="rate_limited", attempt=3, escalated=True)

        assert (
            sink.registry.get_sample_value(
                "fraud_provider_backpressure_total",
                {"error_type": "rate_limited", "phase": "escalated"},
            )
            == 1
        )

    def test_record_backpressure_maps_escalated_false_to_phase_retry(self) -> None:
        sink = PrometheusMetricsSink()

        sink.record_backpressure(kind="unavailable", attempt=1, escalated=False)

        assert (
            sink.registry.get_sample_value(
                "fraud_provider_backpressure_total",
                {"error_type": "unavailable", "phase": "retry"},
            )
            == 1
        )


class TestPrometheusMetricsSinkProviderError:
    def test_record_provider_error_rate_limited_maps_to_429(self) -> None:
        sink = PrometheusMetricsSink()

        sink.record_provider_error(kind="rate_limited")

        assert (
            sink.registry.get_sample_value(
                "fraud_provider_http_errors_total", {"status": "429"}
            )
            == 1
        )

    def test_record_provider_error_unavailable_maps_to_503(self) -> None:
        sink = PrometheusMetricsSink()

        sink.record_provider_error(kind="unavailable")

        assert (
            sink.registry.get_sample_value(
                "fraud_provider_http_errors_total", {"status": "503"}
            )
            == 1
        )


class TestPrometheusMetricsSinkIsolation:
    def test_each_instance_owns_a_private_registry(self) -> None:
        sink_a = PrometheusMetricsSink()
        sink_b = PrometheusMetricsSink()

        sink_a.record_case_outcome("PROCESSED")

        assert (
            sink_b.registry.get_sample_value(
                "fraud_case_outcome_total", {"outcome": "PROCESSED"}
            )
            is None
        )


# --- Cardinality safety: unknown label values collapse to "other" ----------

from fraud_companion.adapters.metrics.prometheus_sink import PrometheusMetricsSink  # noqa: E402


def test_unknown_case_outcome_collapses_to_other():
    sink = PrometheusMetricsSink()
    sink.record_case_outcome("some-arbitrary-attacker-controlled-value")
    assert sink.registry.get_sample_value(
        "fraud_case_outcome_total", {"outcome": "other"}
    ) == 1.0
    # The raw arbitrary value must NOT create its own series.
    assert sink.registry.get_sample_value(
        "fraud_case_outcome_total",
        {"outcome": "some-arbitrary-attacker-controlled-value"},
    ) is None


def test_known_case_outcome_is_preserved():
    sink = PrometheusMetricsSink()
    sink.record_case_outcome("PROCESSED")
    assert sink.registry.get_sample_value(
        "fraud_case_outcome_total", {"outcome": "PROCESSED"}
    ) == 1.0


def test_unknown_backpressure_kind_collapses_to_other():
    sink = PrometheusMetricsSink()
    sink.record_backpressure(kind="weird-kind", attempt=1, escalated=False)
    assert sink.registry.get_sample_value(
        "fraud_provider_backpressure_total",
        {"error_type": "other", "phase": "retry"},
    ) == 1.0


def test_unknown_provider_error_kind_collapses_to_other():
    sink = PrometheusMetricsSink()
    sink.record_provider_error(kind="mystery")
    assert sink.registry.get_sample_value(
        "fraud_provider_http_errors_total", {"status": "other"}
    ) == 1.0
    # Known kinds still map to their status code.
    sink.record_provider_error(kind="rate_limited")
    assert sink.registry.get_sample_value(
        "fraud_provider_http_errors_total", {"status": "429"}
    ) == 1.0
