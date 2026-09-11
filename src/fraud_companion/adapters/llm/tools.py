"""LangChain StructuredTools for the four allow-listed agent tools.

``get_analysis_pack`` (read) and ``put_agent_brief`` (the sole write) are
required. ``list_cases`` and ``list_aml_alerts`` are optional read tools
the agent may use for extra chat context. All four are thin wrappers over
``AntiFraudHttpClient``. ``get_analysis_pack`` applies the domain trim
then untrusted-data framing; it does not invent hit explanations.

Tool names MUST match the entries in
``fraud_companion.domain.tools_spec.ALLOWED_TOOLS`` exactly.
"""
from __future__ import annotations

import functools
import json
from typing import Any, Callable

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from fraud_companion.adapters.http.client import AntiFraudHttpClient
from fraud_companion.application.tool_dispatcher import assert_dispatch_allowed
from fraud_companion.domain.analysis_pack import trim_analysis_pack
from fraud_companion.domain.brief import Brief
from fraud_companion.domain.policy import frame_untrusted_pack


def _guard_dispatch(tool_name: str, func: Callable[..., Any]) -> Callable[..., Any]:
    """Wrap ``func`` so that, before doing anything else, it calls
    :func:`assert_dispatch_allowed` for ``tool_name``.

    This is the application-level dispatcher guardrail (layer 2) wired
    directly into the real tool-execution path: even if the LangGraph
    ``wrap_tool_call`` middleware (layer 1) were bypassed or
    misconfigured, the tool body itself refuses to run under a
    disallowed name.
    """

    @functools.wraps(func)
    def _wrapped(*args: Any, **kwargs: Any) -> Any:
        assert_dispatch_allowed(tool_name)
        return func(*args, **kwargs)

    return _wrapped

_GET_ANALYSIS_PACK_DESCRIPTION = """\
Fetch the full analysis pack for a case BEFORE writing an agent brief.

Returns the case, its event timeline, the fraud-detection snapshot, AML
alerts, related cases, and any existing agent brief. Related cases are
reduced to {id, score, hits}. Timeline events that only repeat the
agentBrief or snapshot (AGENT_BRIEFING, SNAPSHOT_REFRESHED) are omitted.
Business facts (provider, amount, wallet age, velocity, hits and their
because text) are left intact. No because text is invented.

The snapshot's `hits` explain WHY the case was flagged; treat their
`points` as already-computed evidence and NEVER re-score or second-guess
them. If a hit has no `because` field, do NOT invent one — leave it
unexplained. The snapshot does NOT include `subjectIdentity` or
`rawPayload`; do not assume they exist. For AML alerts, prefer
`matchedEntry.name` / `matchedEntry.document` when present.
"""

_PUT_AGENT_BRIEF_DESCRIPTION = """\
Write (or overwrite) the analyst-facing brief for a case. This is the
ONLY write tool in the whole system.

The request body is always {"brief": <non-empty, trimmed string>}. Last
write wins and the operation is safe to retry (idempotent). Writing a
brief does NOT close, resolve, or otherwise change the case's status —
it stays OPEN. This tool is forbidden for human users (only the
system:agent role may call it) and fails with a conflict if the case is
already closed.
"""


class GetAnalysisPackArgs(BaseModel):
    """Arguments for ``get_analysis_pack``."""

    case_id: str = Field(..., description="The anti-fraud case identifier.")


class PutAgentBriefArgs(BaseModel):
    """Arguments for ``put_agent_brief``."""

    case_id: str = Field(..., description="The anti-fraud case identifier.")
    brief: str = Field(..., description="The analyst brief text to persist.")


class ListCasesArgs(BaseModel):
    """Arguments for ``list_cases``."""

    customer_id: str = Field(..., description="Customer identifier to filter cases by.")
    limit: int = Field(20, ge=1, le=100, description="Max number of items to return (1-100).")
    offset: int = Field(0, ge=0, description="Pagination offset.")


class ListAmlAlertsArgs(BaseModel):
    """Arguments for ``list_aml_alerts``."""

    customer_id: str = Field(..., description="Customer identifier to filter AML alerts by.")
    limit: int = Field(20, ge=1, le=100, description="Max number of items to return (1-100).")
    offset: int = Field(0, ge=0, description="Pagination offset.")


def build_get_analysis_pack_tool(http_client: AntiFraudHttpClient) -> StructuredTool:
    """Build the read-only ``get_analysis_pack`` StructuredTool."""

    def _get_analysis_pack(case_id: str) -> str:
        pack = http_client.get(f"/cases/{case_id}/analysis-pack")
        return frame_untrusted_pack(json.dumps(trim_analysis_pack(pack)))

    return StructuredTool.from_function(
        func=_guard_dispatch("get_analysis_pack", _get_analysis_pack),
        name="get_analysis_pack",
        description=_GET_ANALYSIS_PACK_DESCRIPTION,
        args_schema=GetAnalysisPackArgs,
    )


def build_put_agent_brief_tool(http_client: AntiFraudHttpClient) -> StructuredTool:
    """Build the write-only ``put_agent_brief`` StructuredTool."""

    def _put_agent_brief(case_id: str, brief: str) -> Any:
        trimmed = Brief.from_raw(brief)
        return http_client.put(
            f"/cases/{case_id}/agent-brief", json_body={"brief": trimmed.value}
        )

    return StructuredTool.from_function(
        func=_guard_dispatch("put_agent_brief", _put_agent_brief),
        name="put_agent_brief",
        description=_PUT_AGENT_BRIEF_DESCRIPTION,
        args_schema=PutAgentBriefArgs,
        return_direct=True,
    )


_LIST_CASES_DESCRIPTION = """\
List cases for a customer (query param customerId, camelCase). Returns
{items, total} exactly as returned by the anti-fraud API — no reshaping.
The tenant is derived from the API key; organizationId/organization_id is
NEVER sent as a query parameter.
"""

_LIST_AML_ALERTS_DESCRIPTION = """\
List AML alerts for a customer (query param customerId, camelCase — even
though the underlying store uses snake_case elsewhere). Returns
{items, total} exactly as returned by the anti-fraud API — no reshaping.
"""


def build_list_cases_tool(http_client: AntiFraudHttpClient) -> StructuredTool:
    """Build the optional read-only ``list_cases`` StructuredTool."""

    def _list_cases(customer_id: str, limit: int = 20, offset: int = 0) -> Any:
        return http_client.get(
            "/cases",
            params={"customerId": customer_id, "limit": limit, "offset": offset},
        )

    return StructuredTool.from_function(
        func=_guard_dispatch("list_cases", _list_cases),
        name="list_cases",
        description=_LIST_CASES_DESCRIPTION,
        args_schema=ListCasesArgs,
    )


def build_list_aml_alerts_tool(http_client: AntiFraudHttpClient) -> StructuredTool:
    """Build the optional read-only ``list_aml_alerts`` StructuredTool."""

    def _list_aml_alerts(customer_id: str, limit: int = 20, offset: int = 0) -> Any:
        return http_client.get(
            "/aml-alerts",
            params={"customerId": customer_id, "limit": limit, "offset": offset},
        )

    return StructuredTool.from_function(
        func=_guard_dispatch("list_aml_alerts", _list_aml_alerts),
        name="list_aml_alerts",
        description=_LIST_AML_ALERTS_DESCRIPTION,
        args_schema=ListAmlAlertsArgs,
    )
