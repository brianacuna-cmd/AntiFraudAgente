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

from langchain.agents import create_agent
from langchain.agents.middleware import wrap_tool_call as _as_wrap_tool_call_middleware
from langchain_google_genai import ChatGoogleGenerativeAI

from fraud_companion.adapters.http.client import AntiFraudHttpClient
from fraud_companion.adapters.llm.guardrail import build_tool_guardrail
from fraud_companion.adapters.llm.tools import (
    build_get_analysis_pack_tool,
    build_list_aml_alerts_tool,
    build_list_cases_tool,
    build_put_agent_brief_tool,
)
from fraud_companion.config import Settings
from fraud_companion.domain.policy import SECURITY_SYSTEM_PROMPT


def build_agent(settings: Settings, http_client: AntiFraudHttpClient):
    """Build the compiled Gemini agent graph.

    Wires the 4 allow-listed tools bound to ``http_client``, the tool
    allow-list guardrail (layer 1, as middleware), and the security
    system prompt. Returns the compiled LangGraph runnable from
    ``create_agent`` — callers invoke/stream it, they never touch the
    model or the tool node directly.
    """
    model = ChatGoogleGenerativeAI(
        model=settings.gemini_model,
        google_api_key=settings.google_api_key,
        temperature=settings.gemini_temperature,
        max_output_tokens=settings.gemini_max_output_tokens,
    )

    tools = [
        build_get_analysis_pack_tool(http_client),
        build_put_agent_brief_tool(http_client),
        build_list_cases_tool(http_client),
        build_list_aml_alerts_tool(http_client),
    ]

    guardrail_middleware = _as_wrap_tool_call_middleware(build_tool_guardrail())

    return create_agent(
        model=model,
        tools=tools,
        system_prompt=SECURITY_SYSTEM_PROMPT,
        middleware=[guardrail_middleware],
    )
