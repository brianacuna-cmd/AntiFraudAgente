"""Provider-agnostic port for the case-analyst LLM agent.

This is the seam that decouples the whole system from any specific LLM
provider. The application (``case_created_handler``) and the Kafka consumer
depend ONLY on:

- :class:`CaseAnalystAgent` — the minimal behavioural contract (``invoke``).
- The error taxonomy below — provider-neutral failure classes.

Concrete adapters (e.g. ``adapters.llm.agent.GeminiAgent``) translate their
SDK-specific exceptions into these classes. Swapping providers therefore
means writing one new adapter that satisfies this port — nothing downstream
changes.
"""
from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class CaseAnalystAgent(Protocol):
    """Anything that can process one case-analysis request.

    The payload/return shapes intentionally stay ``Any``: the handler passes
    the LangGraph-style ``{"messages": [...]}`` dict through unchanged, and
    an adapter for a different framework is free to accept the same shape.
    """

    def invoke(self, payload: dict[str, Any]) -> Any: ...


class AgentError(RuntimeError):
    """Base class for provider-agnostic LLM agent failures."""


class AgentRateLimitedError(AgentError):
    """The provider signalled rate-limit / quota exhaustion (e.g. HTTP 429).

    ``retry_after`` is the server-suggested wait in seconds when the provider
    supplies one (``None`` otherwise), so a backpressure handler can honour it
    instead of guessing.
    """

    def __init__(self, message: str = "", *, retry_after: float | None = None) -> None:
        super().__init__(message or "agent provider rate-limited")
        self.retry_after = retry_after


class AgentUnavailableError(AgentError):
    """The provider is temporarily unavailable (e.g. HTTP 5xx)."""
