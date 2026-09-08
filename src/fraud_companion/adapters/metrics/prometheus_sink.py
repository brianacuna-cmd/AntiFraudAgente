"""Concrete Prometheus-backed :class:`MetricsSink` adapter (Slice B).

This is the only module in the codebase permitted to import
``prometheus_client`` — the "Hexagonal Boundary for Prometheus" requirement.
Every instrument here uses a private, per-instance ``CollectorRegistry``
rather than the global default registry so unit tests stay isolated and a
process may in principle construct more than one sink without cross-talk.

Every label is fixed-cardinality (enum-like values only): no case id or
organization is ever passed as a label value.
"""
from __future__ import annotations

from prometheus_client import CollectorRegistry, Counter

_STATUS_BY_KIND = {
    "rate_limited": "429",
    "unavailable": "503",
}

#: Sentinel label value for any input outside the known fixed set. Collapsing
#: unknown values here keeps Prometheus label cardinality bounded — an
#: arbitrary caller string can never spawn an unbounded number of series.
_OTHER = "other"

#: Allowed, fixed-cardinality label values. Anything else maps to ``_OTHER``.
#: Exactly the values the emission sites produce: ``HandleResult.<name>``
#: (upper-case) from the handler, plus the two literal strings emitted for the
#: brief-not-written path and the consumer's exception-only path.
_ALLOWED_OUTCOMES = frozenset(
    {
        "PROCESSED",
        "SKIPPED_IGNORED",
        "SKIPPED_TERMINAL",
        "SKIPPED_MALFORMED",
        "brief_not_written",
        "error",
    }
)
_ALLOWED_BACKPRESSURE_KINDS = frozenset({"rate_limited", "unavailable"})
_ALLOWED_STATUSES = frozenset(_STATUS_BY_KIND.values())


def _bounded(value: str, allowed: frozenset[str]) -> str:
    """Return ``value`` if it is a known label, else the ``_OTHER`` sentinel."""
    return value if value in allowed else _OTHER


class PrometheusMetricsSink:
    """:class:`MetricsSink` implementation backed by ``prometheus_client``.

    Owns its own :class:`~prometheus_client.CollectorRegistry`; instruments
    are created once at construction time. Methods never raise on valid
    fixed-cardinality label input.
    """

    def __init__(self) -> None:
        self.registry = CollectorRegistry()

        self._tokens_total = Counter(
            "fraud_llm_tokens_total",
            "Total LLM tokens observed, labelled by kind (input/output/total).",
            ["kind"],
            registry=self.registry,
        )
        self._case_outcome_total = Counter(
            "fraud_case_outcome_total",
            "Total cases processed, labelled by outcome category.",
            ["outcome"],
            registry=self.registry,
        )
        self._backpressure_total = Counter(
            "fraud_provider_backpressure_total",
            "Total provider backpressure events, labelled by error type and phase.",
            ["error_type", "phase"],
            registry=self.registry,
        )
        self._http_errors_total = Counter(
            "fraud_provider_http_errors_total",
            "Total provider HTTP error classifications, labelled by status code.",
            ["status"],
            registry=self.registry,
        )

    def observe_token_usage(self, *, input: int, output: int, total: int) -> None:
        self._tokens_total.labels(kind="input").inc(input)
        self._tokens_total.labels(kind="output").inc(output)
        self._tokens_total.labels(kind="total").inc(total)

    def record_case_outcome(self, outcome: str) -> None:
        self._case_outcome_total.labels(
            outcome=_bounded(outcome, _ALLOWED_OUTCOMES)
        ).inc()

    def record_backpressure(self, *, kind: str, attempt: int, escalated: bool) -> None:
        phase = "escalated" if escalated else "retry"
        self._backpressure_total.labels(
            error_type=_bounded(kind, _ALLOWED_BACKPRESSURE_KINDS), phase=phase
        ).inc()

    def record_provider_error(self, *, kind: str) -> None:
        status = _bounded(_STATUS_BY_KIND.get(kind, kind), _ALLOWED_STATUSES)
        self._http_errors_total.labels(status=status).inc()
