"""LangChain StructuredTools for the two required agent tools.

Only ``get_analysis_pack`` (read) and ``put_agent_brief`` (the sole write)
are implemented here. ``list_cases`` and ``list_aml_alerts`` are a later
slice. Both tools are thin wrappers over ``AntiFraudHttpClient``; they do
not interpret, reshape, or invent data — that is the LLM's job.

Tool names MUST match the entries in
``fraud_companion.domain.tools_spec.ALLOWED_TOOLS`` exactly.
"""
from __future__ import annotations

from typing import Any

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from fraud_companion.adapters.http.client import AntiFraudHttpClient
from fraud_companion.domain.brief import Brief

_GET_ANALYSIS_PACK_DESCRIPTION = """\
Fetch the full analysis pack for a case BEFORE writing an agent brief.

Returns the case, its event timeline, the fraud-detection snapshot, AML
alerts, related cases, and any existing agent brief, exactly as returned
by the anti-fraud API — nothing is reshaped or reinterpreted.

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


def build_get_analysis_pack_tool(http_client: AntiFraudHttpClient) -> StructuredTool:
    """Build the read-only ``get_analysis_pack`` StructuredTool."""

    def _get_analysis_pack(case_id: str) -> Any:
        return http_client.get(f"/cases/{case_id}/analysis-pack")

    return StructuredTool.from_function(
        func=_get_analysis_pack,
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
        func=_put_agent_brief,
        name="put_agent_brief",
        description=_PUT_AGENT_BRIEF_DESCRIPTION,
        args_schema=PutAgentBriefArgs,
    )
