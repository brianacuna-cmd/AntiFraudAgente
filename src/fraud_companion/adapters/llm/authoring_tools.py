"""LangChain StructuredTools for the six scoring-rule authoring tools.

Kept in a SEPARATE module from ``adapters/llm/tools.py`` (the lean
case-analyst tool set) so that authoring schemas and this module's
imports never load into the Kafka/case-brief builder path.

Tool names MUST match the entries in
``fraud_companion.domain.tools_spec.AUTHORING_TOOLS`` exactly. Each tool
is wrapped through ``_guard_dispatch(name, func, allowed=AUTHORING_TOOLS)``
(dispatcher layer-2 guard) mirroring the case-analyst tools' pattern.
"""
from __future__ import annotations

import re
from typing import Any, Literal

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field, field_validator

from fraud_companion.adapters.http.client import AntiFraudHttpClient
from fraud_companion.adapters.llm.tools import _guard_dispatch
from fraud_companion.domain.tools_spec import AUTHORING_TOOLS

Operator = Literal["GT", "GTE", "LT", "LTE", "EQ", "NEQ", "CONTAINS", "IN", "BETWEEN"]

_STATIC_ALLOWED_FIELDS = {
    "provider",
    "providerEventType",
    "caseCustomerId",
    "amountCents",
    "currency",
    "rail",
    "eventId",
    "providerEventId",
    "subjectIdentity.name",
    "subjectIdentity.document",
    "subjectIdentity.walletAddress",
    "subjectIdentity.entryType",
}

_RISK_SIGNALS_FIELD_RE = re.compile(r"^riskSignals\.[A-Za-z0-9]+$")


def _is_allowed_field(field: str) -> bool:
    if field in _STATIC_ALLOWED_FIELDS:
        return True
    return bool(_RISK_SIGNALS_FIELD_RE.match(field))


class Factor(BaseModel):
    """A single scoring factor for factor-scoring rule creation."""

    field: str = Field(..., description="The event/case field to evaluate.")
    operator: Operator = Field(..., description="Comparison operator.")
    value: Any = Field(..., description="The comparison value.")
    points: int = Field(
        ..., ge=-100, le=100, description="Points contributed if the factor matches."
    )
    reason: str = Field(..., description="Human-readable explanation for this factor.")

    @field_validator("field")
    @classmethod
    def _validate_field(cls, value: str) -> str:
        if not _is_allowed_field(value):
            raise ValueError(f"field '{value}' is not in the allowed field list")
        return value


class CreateScoringRuleArgs(BaseModel):
    """Arguments for ``create_scoring_rule_via_factor_scoring``."""

    name: str = Field(..., min_length=1, description="The scoring rule name.")
    factors: list[Factor] = Field(..., description="The list of scoring factors.")


class UpdateScoringRuleArgs(BaseModel):
    """Arguments for ``update_scoring_rule``.

    Only ``rule_id`` is required. ``status`` and ``conditionsVersion`` are
    server-owned and deliberately have no corresponding field here.
    """

    rule_id: str = Field(..., description="The scoring rule identifier.")
    name: str | None = Field(None, description="New name for the rule.")
    conditions: dict[str, Any] | None = Field(
        None, description="New condition graph for the rule."
    )


class ActivateScoringRuleArgs(BaseModel):
    """Arguments for ``activate_scoring_rule``."""

    rule_id: str = Field(..., description="The scoring rule identifier.")


class ListScoringRulesArgs(BaseModel):
    """Arguments for ``list_scoring_rules`` (none required)."""


class GetScoringRuleArgs(BaseModel):
    """Arguments for ``get_scoring_rule``."""

    rule_id: str = Field(..., description="The scoring rule identifier.")


class SimulateScoringRuleArgs(BaseModel):
    """Arguments for ``simulate_scoring_rule``."""

    conditions: dict[str, Any] = Field(..., description="The condition graph to simulate.")
    event: dict[str, Any] = Field(..., description="A sample event to evaluate against.")


_CREATE_SCORING_RULE_DESCRIPTION = """\
Create a new scoring rule via the factor-scoring shortcut. Returns the
created rule DTO verbatim (INACTIVE status) — it must be activated
separately via activate_scoring_rule.
"""

_UPDATE_SCORING_RULE_DESCRIPTION = """\
Partially update a scoring rule's name and/or conditions. Omitted fields
are left unchanged; calling with only rule_id is a safe no-op. Never
touches status or conditionsVersion (server-owned).
"""

_ACTIVATE_SCORING_RULE_DESCRIPTION = """\
Activate a scoring rule by id. Idempotent — safe to call again on an
already-active rule (though the backend may reject a duplicate call with
SCORING_RULE_ACTIVE depending on state).
"""

_LIST_SCORING_RULES_DESCRIPTION = """\
List all scoring rules. Returns {items: [...]} exactly as returned by the
anti-fraud API — no reshaping.
"""

_GET_SCORING_RULE_DESCRIPTION = """\
Fetch a single scoring rule by id. Returns the rule DTO verbatim.
"""

_SIMULATE_SCORING_RULE_DESCRIPTION = """\
Dry-run a condition graph against a sample event. Never mutates any rule.
Returns the response body verbatim, including {"ok": false, ...} for an
invalid/failing graph — that is a normal result, not an error.
"""


def build_create_scoring_rule_tool(http_client: AntiFraudHttpClient) -> StructuredTool:
    """Build the ``create_scoring_rule_via_factor_scoring`` StructuredTool."""

    def _create_scoring_rule(name: str, factors: list[dict[str, Any]]) -> Any:
        body = {
            "name": name,
            "factors": [
                factor if isinstance(factor, dict) else factor.model_dump()
                for factor in factors
            ],
        }
        return http_client.post("/risk-scoring-rules/factor-scoring", json_body=body)

    return StructuredTool.from_function(
        func=_guard_dispatch(
            "create_scoring_rule_via_factor_scoring",
            _create_scoring_rule,
            allowed=AUTHORING_TOOLS,
        ),
        name="create_scoring_rule_via_factor_scoring",
        description=_CREATE_SCORING_RULE_DESCRIPTION,
        args_schema=CreateScoringRuleArgs,
    )


def build_update_scoring_rule_tool(http_client: AntiFraudHttpClient) -> StructuredTool:
    """Build the ``update_scoring_rule`` StructuredTool."""

    def _update_scoring_rule(
        rule_id: str,
        name: str | None = None,
        conditions: dict[str, Any] | None = None,
    ) -> Any:
        body: dict[str, Any] = {}
        if name is not None:
            body["name"] = name
        if conditions is not None:
            body["conditions"] = conditions
        return http_client.patch(f"/risk-scoring-rules/{rule_id}", json_body=body)

    return StructuredTool.from_function(
        func=_guard_dispatch(
            "update_scoring_rule", _update_scoring_rule, allowed=AUTHORING_TOOLS
        ),
        name="update_scoring_rule",
        description=_UPDATE_SCORING_RULE_DESCRIPTION,
        args_schema=UpdateScoringRuleArgs,
    )


def build_activate_scoring_rule_tool(http_client: AntiFraudHttpClient) -> StructuredTool:
    """Build the ``activate_scoring_rule`` StructuredTool."""

    def _activate_scoring_rule(rule_id: str) -> Any:
        return http_client.post(f"/risk-scoring-rules/{rule_id}/activate", json_body=None)

    return StructuredTool.from_function(
        func=_guard_dispatch(
            "activate_scoring_rule", _activate_scoring_rule, allowed=AUTHORING_TOOLS
        ),
        name="activate_scoring_rule",
        description=_ACTIVATE_SCORING_RULE_DESCRIPTION,
        args_schema=ActivateScoringRuleArgs,
    )


def build_list_scoring_rules_tool(http_client: AntiFraudHttpClient) -> StructuredTool:
    """Build the ``list_scoring_rules`` StructuredTool."""

    def _list_scoring_rules() -> Any:
        return http_client.get("/risk-scoring-rules")

    return StructuredTool.from_function(
        func=_guard_dispatch(
            "list_scoring_rules", _list_scoring_rules, allowed=AUTHORING_TOOLS
        ),
        name="list_scoring_rules",
        description=_LIST_SCORING_RULES_DESCRIPTION,
        args_schema=ListScoringRulesArgs,
    )


def build_get_scoring_rule_tool(http_client: AntiFraudHttpClient) -> StructuredTool:
    """Build the ``get_scoring_rule`` StructuredTool."""

    def _get_scoring_rule(rule_id: str) -> Any:
        return http_client.get(f"/risk-scoring-rules/{rule_id}")

    return StructuredTool.from_function(
        func=_guard_dispatch(
            "get_scoring_rule", _get_scoring_rule, allowed=AUTHORING_TOOLS
        ),
        name="get_scoring_rule",
        description=_GET_SCORING_RULE_DESCRIPTION,
        args_schema=GetScoringRuleArgs,
    )


def build_simulate_scoring_rule_tool(http_client: AntiFraudHttpClient) -> StructuredTool:
    """Build the ``simulate_scoring_rule`` StructuredTool."""

    def _simulate_scoring_rule(
        conditions: dict[str, Any], event: dict[str, Any]
    ) -> Any:
        return http_client.post(
            "/risk-scoring-rules/simulate",
            json_body={"conditions": conditions, "event": event},
        )

    return StructuredTool.from_function(
        func=_guard_dispatch(
            "simulate_scoring_rule", _simulate_scoring_rule, allowed=AUTHORING_TOOLS
        ),
        name="simulate_scoring_rule",
        description=_SIMULATE_SCORING_RULE_DESCRIPTION,
        args_schema=SimulateScoringRuleArgs,
    )
