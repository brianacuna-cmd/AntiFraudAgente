"""Gemini agent factory (Slice 6).

Built against the REAL installed API — verified via introspection of the
installed packages before writing this module:

- ``langchain.agents.create_agent`` (langchain top-level, 2026 API) is the
  correct import; ``AgentExecutor`` and ``create_react_agent`` are
  deprecated. This matches prior research (obs 457) exactly.
- ``create_agent`` has NO direct parameter to plug a custom ``ToolNode``
  or a raw ``wrap_tool_call`` callable — that was an assumption in the
  design doc that does not hold against the installed API. Instead, the
  installed ``langchain.agents.middleware`` module exposes
  ``AgentMiddleware`` with a ``wrap_tool_call(request, handler)`` hook
  whose signature is IDENTICAL to the ``wrap_tool_call`` closure produced
  by ``guardrail.build_tool_guardrail()``. The
  ``langchain.agents.middleware.wrap_tool_call`` decorator/function turns
  that exact closure into an ``AgentMiddleware`` instance with no
  behavioral change, which is then passed via ``create_agent(...,
  middleware=[...])``. This preserves the guardrail's allow-list
  enforcement (layer 1) unchanged; only the wiring surface differs from
  the original design sketch.
- ``ChatGoogleGenerativeAI`` from ``langchain_google_genai`` accepts
  ``model`` and ``google_api_key`` as constructor kwargs (confirmed via
  ``model_fields`` introspection).
"""
from __future__ import annotations

import re
from typing import Any, Callable

from google.genai.errors import APIError
from langchain.agents import create_agent
from langchain.agents.middleware import wrap_tool_call as _as_wrap_tool_call_middleware
from langchain_google_genai import ChatGoogleGenerativeAI

from fraud_companion.adapters.http.client import AntiFraudHttpClient
from fraud_companion.application.agent_port import (
    AgentRateLimitedError,
    AgentUnavailableError,
)
from fraud_companion.adapters.llm.guardrail import build_tool_guardrail
from fraud_companion.adapters.llm.tools import (
    build_get_analysis_pack_tool,
    build_list_aml_alerts_tool,
    build_list_cases_tool,
    build_put_agent_brief_tool,
)
from fraud_companion.config import Settings
from fraud_companion.domain.policy import SECURITY_SYSTEM_PROMPT


_RETRY_DELAY_RE = re.compile(r"(?P<seconds>\d+(?:\.\d+)?)s")

#: Independent, upstream-facing safeguard applied at parse time so a
#: malformed/absurd provider hint can never propagate an unbounded wait.
#: This does NOT replace the consumer's own backpressure cap
#: (``_backpressure_delay``'s existing 60s ceiling); it is a second,
#: earlier line of defense.
_MAX_RETRY_AFTER_SECONDS = 60.0


class UnsupportedProviderError(ValueError):
    """Raised when ``settings.llm_provider`` has no registered builder."""

    def __init__(self, provider: str, registered: "list[str]") -> None:
        choices = ", ".join(sorted(registered))
        super().__init__(
            f"Unsupported LLM provider {provider!r}; registered providers: {choices}"
        )
        self.provider = provider


def _extract_retry_after(exc: APIError) -> float | None:
    """Best-effort parse of a Gemini 429 ``RetryInfo.retryDelay`` (e.g. "17s").

    Never raises: any unexpected shape yields ``None`` so the caller falls
    back to its own backoff. The parsed value is clamped to
    ``_MAX_RETRY_AFTER_SECONDS`` before being returned.
    """
    details = getattr(exc, "details", None)
    if isinstance(details, dict):
        details = details.get("error", details)
    entries = details.get("details") if isinstance(details, dict) else None
    if not isinstance(entries, list):
        return None
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        delay = entry.get("retryDelay")
        if isinstance(delay, str):
            match = _RETRY_DELAY_RE.fullmatch(delay.strip())
            if match:
                return min(float(match.group("seconds")), _MAX_RETRY_AFTER_SECONDS)
    return None


def _translate_api_error(exc: APIError) -> Exception:
    """Map a google.genai ``APIError`` onto the provider-agnostic taxonomy.

    Only backpressure classes are translated; anything else is returned
    unchanged so it propagates as-is.
    """
    code = getattr(exc, "code", None)
    if code == 429:
        return AgentRateLimitedError(str(exc), retry_after=_extract_retry_after(exc))
    if isinstance(code, int) and 500 <= code < 600:
        return AgentUnavailableError(str(exc))
    return exc


class GeminiAgent:
    """Adapter wrapping the compiled LangGraph agent.

    Its ONLY job beyond delegation is to translate ``google.genai`` API
    errors into the provider-agnostic :mod:`agent_port` taxonomy, so the
    handler and consumer never import a provider SDK. Satisfies
    ``application.agent_port.CaseAnalystAgent``.
    """

    def __init__(self, compiled: Any) -> None:
        self._compiled = compiled

    def invoke(self, payload: dict[str, Any]) -> Any:
        try:
            return self._compiled.invoke(payload)
        except APIError as exc:
            raise _translate_api_error(exc) from exc


def _build_gemini_agent(settings: Settings, http_client: AntiFraudHttpClient) -> GeminiAgent:
    """Build the compiled Gemini agent graph.

    Wires the 4 allow-listed tools bound to ``http_client``, the tool
    allow-list guardrail (layer 1, as middleware), and the security
    system prompt. Returns the compiled LangGraph runnable from
    ``create_agent`` — callers invoke/stream it, they never touch the
    model or the tool node directly. ``ChatGoogleGenerativeAI`` is
    confined to this builder only.
    """
    model = ChatGoogleGenerativeAI(
        model=settings.llm_model,
        google_api_key=settings.llm_api_key,
        temperature=settings.llm_temperature,
        max_output_tokens=settings.llm_max_output_tokens,
    )

    tools = [
        build_get_analysis_pack_tool(http_client),
        build_put_agent_brief_tool(http_client),
        build_list_cases_tool(http_client),
        build_list_aml_alerts_tool(http_client),
    ]

    guardrail_middleware = _as_wrap_tool_call_middleware(build_tool_guardrail())

    compiled = create_agent(
        model=model,
        tools=tools,
        system_prompt=SECURITY_SYSTEM_PROMPT,
        middleware=[guardrail_middleware],
    )
    return GeminiAgent(compiled)


_PROVIDER_BUILDERS: dict[
    str, Callable[[Settings, AntiFraudHttpClient], GeminiAgent]
] = {
    "gemini": _build_gemini_agent,
}


def build_agent(settings: Settings, http_client: AntiFraudHttpClient) -> GeminiAgent:
    """Dispatch to the provider-specific builder keyed by ``settings.llm_provider``.

    Raises ``UnsupportedProviderError`` for an unregistered provider; no
    agent is constructed in that case.
    """
    builder = _PROVIDER_BUILDERS.get(settings.llm_provider)
    if builder is None:
        raise UnsupportedProviderError(
            settings.llm_provider, list(_PROVIDER_BUILDERS)
        )
    return builder(settings, http_client)
