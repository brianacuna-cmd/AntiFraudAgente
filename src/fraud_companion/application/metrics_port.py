"""Provider-agnostic port for observability metrics emission.

Mirrors ``agent_port.py``'s seam: the application layer and adapters depend
only on the :class:`MetricsSink` Protocol below, never on a concrete metrics
backend. No ``prometheus_client`` import belongs in this module — a concrete
Prometheus-backed sink lives in ``adapters.metrics.prometheus_sink``.

All four signal points are side-effect-only from the perspective of case
processing: an emission call must never alter control flow, and — per the
"Emission Is Side-Effect-Only" requirement — a raising sink implementation
must never propagate into the processing path. Callers are responsible for
wrapping emission calls (see ``case_created_handler`` and
``adapters.kafka.consumer``'s ``_safe`` helper).
"""
from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class MetricsSink(Protocol):
    """Behavioural contract for recording provider/case observability signals."""

    def observe_token_usage(self, *, input: int, output: int, total: int) -> None: ...

    def record_case_outcome(self, outcome: str) -> None: ...

    def record_backpressure(self, *, kind: str, attempt: int, escalated: bool) -> None: ...

    def record_provider_error(self, *, kind: str) -> None: ...


class NoOpMetricsSink:
    """Inert default :class:`MetricsSink` implementation.

    Used whenever metrics are disabled (``METRICS_ENABLED=false``) and as the
    default parameter value at every call site so existing tests/call sites
    stay green without passing a sink explicitly. Never raises and has no
    observable side effect, regardless of the arguments passed.
    """

    def observe_token_usage(self, *, input: int, output: int, total: int) -> None:
        return None

    def record_case_outcome(self, outcome: str) -> None:
        return None

    def record_backpressure(self, *, kind: str, attempt: int, escalated: bool) -> None:
        return None

    def record_provider_error(self, *, kind: str) -> None:
        return None
