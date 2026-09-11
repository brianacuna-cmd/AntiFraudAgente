"""Authoring agent factory (Slice 3).

Builds a SEPARATE, multi-turn chat agent for scoring-rule authoring. This
module intentionally mirrors ``adapters.llm.agent`` (same
``ChatGoogleGenerativeAI`` construction from ``Settings``, same
``GeminiAgent`` error-translation wrapper) but wires the 6 authoring
tools bound to ``AUTHORING_TOOLS`` and the ``AUTHORING_SYSTEM_PROMPT``
instead. No tool here sets ``return_direct`` — unlike the case-analyst's
``put_agent_brief``, this agent must keep reasoning across multiple human
turns after a tool call completes.

``adapters.llm.agent`` (``build_agent`` / ``_build_gemini_agent`` /
``_PROVIDER_BUILDERS``) is NEVER modified or imported for its internals
here beyond reusing the already-exported ``GeminiAgent`` wrapper.
"""
from __future__ import annotations

from langchain.agents import create_agent
from langchain.agents.middleware import wrap_tool_call as _as_wrap_tool_call_middleware
from langchain_google_genai import ChatGoogleGenerativeAI

from fraud_companion.adapters.http.client import AntiFraudHttpClient
from fraud_companion.adapters.llm.agent import GeminiAgent
from fraud_companion.adapters.llm.authoring_tools import (
    build_activate_scoring_rule_tool,
    build_create_scoring_rule_tool,
    build_get_scoring_rule_tool,
    build_list_scoring_rules_tool,
    build_simulate_scoring_rule_tool,
    build_update_scoring_rule_tool,
)
from fraud_companion.adapters.llm.guardrail import build_tool_guardrail
from fraud_companion.config import Settings
from fraud_companion.domain.authoring_policy import AUTHORING_SYSTEM_PROMPT
from fraud_companion.domain.tools_spec import AUTHORING_TOOLS


def build_authoring_agent(settings: Settings, http_client: AntiFraudHttpClient) -> GeminiAgent:
    """Build the compiled authoring agent graph.

    Wires the 6 authoring tools bound to ``http_client``, the authoring
    tool allow-list guardrail (layer 1, as middleware, bound to
    ``AUTHORING_TOOLS``), and ``AUTHORING_SYSTEM_PROMPT``. Returns a
    ``GeminiAgent`` (the same error-translation wrapper as the
    case-analyst agent) so 429/503 map onto the provider-agnostic
    taxonomy here too. Multi-turn: no tool sets ``return_direct``.
    """
    model = ChatGoogleGenerativeAI(
        model=settings.llm_model,
        google_api_key=settings.llm_api_key,
        temperature=settings.llm_temperature,
        max_output_tokens=settings.llm_max_output_tokens,
    )

    tools = [
        build_create_scoring_rule_tool(http_client),
        build_update_scoring_rule_tool(http_client),
        build_activate_scoring_rule_tool(http_client),
        build_list_scoring_rules_tool(http_client),
        build_get_scoring_rule_tool(http_client),
        build_simulate_scoring_rule_tool(http_client),
    ]

    guardrail_middleware = _as_wrap_tool_call_middleware(
        build_tool_guardrail(allowed=AUTHORING_TOOLS)
    )

    compiled = create_agent(
        model=model,
        tools=tools,
        system_prompt=AUTHORING_SYSTEM_PROMPT,
        middleware=[guardrail_middleware],
    )
    return GeminiAgent(compiled)
