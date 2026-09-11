"""Tests for the six scoring-rule authoring StructuredTools.

All HTTP calls are mocked against ``AntiFraudHttpClient``; no live backend
is exercised (production currently returns 403 FORBIDDEN_ROLE for the
agent's API key until a separate backend repo grants an authoring role).
"""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from pydantic import ValidationError as PydanticValidationError

from fraud_companion.adapters.http.client import AntiFraudHttpClient
from fraud_companion.adapters.http.errors import (
    InvariantViolationError,
    ScoringRuleActiveError,
    ScoringRuleNotFoundError,
)
from fraud_companion.adapters.llm.authoring_tools import (
    CreateScoringRuleArgs,
    build_activate_scoring_rule_tool,
    build_create_scoring_rule_tool,
    build_get_scoring_rule_tool,
    build_list_scoring_rules_tool,
    build_simulate_scoring_rule_tool,
    build_update_scoring_rule_tool,
)
from fraud_companion.domain.tools_spec import AUTHORING_TOOLS, DisallowedToolError


@pytest.fixture
def http_client() -> MagicMock:
    return MagicMock(spec=AntiFraudHttpClient)


class TestCreateScoringRuleTool:
    def test_name_in_authoring_tools(self, http_client: MagicMock) -> None:
        tool = build_create_scoring_rule_tool(http_client)
        assert tool.name == "create_scoring_rule_via_factor_scoring"
        assert tool.name in AUTHORING_TOOLS

    def test_posts_factor_scoring_and_returns_body_verbatim(
        self, http_client: MagicMock
    ) -> None:
        dto = {"id": "r1", "status": "INACTIVE", "name": "New rule"}
        http_client.post.return_value = dto
        tool = build_create_scoring_rule_tool(http_client)

        result = tool.func(
            name="New rule",
            factors=[
                {
                    "field": "amountCents",
                    "operator": "GT",
                    "value": 1000,
                    "points": 10,
                    "reason": "large amount",
                }
            ],
        )

        assert result == dto
        http_client.post.assert_called_once_with(
            "/risk-scoring-rules/factor-scoring",
            json_body={
                "name": "New rule",
                "factors": [
                    {
                        "field": "amountCents",
                        "operator": "GT",
                        "value": 1000,
                        "points": 10,
                        "reason": "large amount",
                    }
                ],
            },
        )

    def test_points_out_of_range_rejected_before_http_call(
        self, http_client: MagicMock
    ) -> None:
        with pytest.raises(PydanticValidationError):
            CreateScoringRuleArgs(
                name="x",
                factors=[
                    {
                        "field": "amountCents",
                        "operator": "GT",
                        "value": 1,
                        "points": 150,
                        "reason": "x",
                    }
                ],
            )
        http_client.post.assert_not_called()

    def test_invalid_operator_rejected_before_http_call(
        self, http_client: MagicMock
    ) -> None:
        with pytest.raises(PydanticValidationError):
            CreateScoringRuleArgs(
                name="x",
                factors=[
                    {
                        "field": "amountCents",
                        "operator": "BOGUS",
                        "value": 1,
                        "points": 1,
                        "reason": "x",
                    }
                ],
            )
        http_client.post.assert_not_called()

    def test_unrecognized_field_rejected_before_http_call(
        self, http_client: MagicMock
    ) -> None:
        with pytest.raises(PydanticValidationError):
            CreateScoringRuleArgs(
                name="x",
                factors=[
                    {
                        "field": "notARealField",
                        "operator": "GT",
                        "value": 1,
                        "points": 1,
                        "reason": "x",
                    }
                ],
            )
        http_client.post.assert_not_called()

    def test_invariant_violation_propagates(self, http_client: MagicMock) -> None:
        http_client.post.side_effect = InvariantViolationError(
            status=400, code="INVARIANT_VIOLATION", message="bad"
        )
        tool = build_create_scoring_rule_tool(http_client)
        with pytest.raises(InvariantViolationError):
            tool.func(
                name="x",
                factors=[
                    {
                        "field": "amountCents",
                        "operator": "GT",
                        "value": 1,
                        "points": 1,
                        "reason": "x",
                    }
                ],
            )

    def test_disallowed_name_rejected_by_dispatcher_guard(self) -> None:
        from fraud_companion.application.tool_dispatcher import assert_dispatch_allowed

        with pytest.raises(DisallowedToolError):
            assert_dispatch_allowed("get_analysis_pack", allowed=AUTHORING_TOOLS)


class TestUpdateScoringRuleTool:
    def test_name_in_authoring_tools(self, http_client: MagicMock) -> None:
        tool = build_update_scoring_rule_tool(http_client)
        assert tool.name == "update_scoring_rule"
        assert tool.name in AUTHORING_TOOLS

    def test_sends_only_non_none_fields(self, http_client: MagicMock) -> None:
        http_client.patch.return_value = {"id": "r1", "name": "New"}
        tool = build_update_scoring_rule_tool(http_client)

        result = tool.func(rule_id="r1", name="New", conditions=None)

        assert result == {"id": "r1", "name": "New"}
        http_client.patch.assert_called_once_with(
            "/risk-scoring-rules/r1", json_body={"name": "New"}
        )

    def test_no_optional_fields_sends_empty_body_noop(
        self, http_client: MagicMock
    ) -> None:
        http_client.patch.return_value = {"id": "r1"}
        tool = build_update_scoring_rule_tool(http_client)

        result = tool.func(rule_id="r1")

        assert result == {"id": "r1"}
        http_client.patch.assert_called_once_with(
            "/risk-scoring-rules/r1", json_body={}
        )

    def test_args_schema_has_no_status_or_conditions_version_field(self) -> None:
        from fraud_companion.adapters.llm.authoring_tools import UpdateScoringRuleArgs

        fields = UpdateScoringRuleArgs.model_fields
        assert "status" not in fields
        assert "conditionsVersion" not in fields
        assert "conditions_version" not in fields

    def test_not_found_propagates(self, http_client: MagicMock) -> None:
        http_client.patch.side_effect = ScoringRuleNotFoundError(
            status=404, code="SCORING_RULE_NOT_FOUND", message="nf"
        )
        tool = build_update_scoring_rule_tool(http_client)
        with pytest.raises(ScoringRuleNotFoundError):
            tool.func(rule_id="r1")


class TestActivateScoringRuleTool:
    def test_name_in_authoring_tools(self, http_client: MagicMock) -> None:
        tool = build_activate_scoring_rule_tool(http_client)
        assert tool.name == "activate_scoring_rule"
        assert tool.name in AUTHORING_TOOLS

    def test_posts_activate_with_no_body(self, http_client: MagicMock) -> None:
        http_client.post.return_value = {"id": "r1", "status": "ACTIVE"}
        tool = build_activate_scoring_rule_tool(http_client)

        result = tool.func(rule_id="r1")

        assert result == {"id": "r1", "status": "ACTIVE"}
        args, kwargs = http_client.post.call_args
        assert args[0] == "/risk-scoring-rules/r1/activate"
        assert kwargs.get("json_body") in (None, {})

    def test_idempotent_two_calls_both_return_200(self, http_client: MagicMock) -> None:
        http_client.post.return_value = {"id": "r1", "status": "ACTIVE"}
        tool = build_activate_scoring_rule_tool(http_client)

        first = tool.func(rule_id="r1")
        second = tool.func(rule_id="r1")

        assert first == {"id": "r1", "status": "ACTIVE"}
        assert second == {"id": "r1", "status": "ACTIVE"}
        assert http_client.post.call_count == 2

    def test_active_conflict_propagates(self, http_client: MagicMock) -> None:
        http_client.post.side_effect = ScoringRuleActiveError(
            status=409, code="SCORING_RULE_ACTIVE", message="already active"
        )
        tool = build_activate_scoring_rule_tool(http_client)
        with pytest.raises(ScoringRuleActiveError):
            tool.func(rule_id="r1")


class TestListScoringRulesTool:
    def test_name_in_authoring_tools(self, http_client: MagicMock) -> None:
        tool = build_list_scoring_rules_tool(http_client)
        assert tool.name == "list_scoring_rules"
        assert tool.name in AUTHORING_TOOLS

    def test_returns_items_verbatim(self, http_client: MagicMock) -> None:
        body = {"items": [{"id": "r1"}, {"id": "r2"}]}
        http_client.get.return_value = body
        tool = build_list_scoring_rules_tool(http_client)

        result = tool.func()

        assert result == body
        http_client.get.assert_called_once_with("/risk-scoring-rules")


class TestGetScoringRuleTool:
    def test_name_in_authoring_tools(self, http_client: MagicMock) -> None:
        tool = build_get_scoring_rule_tool(http_client)
        assert tool.name == "get_scoring_rule"
        assert tool.name in AUTHORING_TOOLS

    def test_returns_dto_verbatim(self, http_client: MagicMock) -> None:
        dto = {"id": "r1", "status": "ACTIVE"}
        http_client.get.return_value = dto
        tool = build_get_scoring_rule_tool(http_client)

        result = tool.func(rule_id="r1")

        assert result == dto
        http_client.get.assert_called_once_with("/risk-scoring-rules/r1")

    def test_not_found_propagates(self, http_client: MagicMock) -> None:
        http_client.get.side_effect = ScoringRuleNotFoundError(
            status=404, code="SCORING_RULE_NOT_FOUND", message="nf"
        )
        tool = build_get_scoring_rule_tool(http_client)
        with pytest.raises(ScoringRuleNotFoundError):
            tool.func(rule_id="r1")


class TestSimulateScoringRuleTool:
    def test_name_in_authoring_tools(self, http_client: MagicMock) -> None:
        tool = build_simulate_scoring_rule_tool(http_client)
        assert tool.name == "simulate_scoring_rule"
        assert tool.name in AUTHORING_TOOLS

    def test_returns_ok_true_body_verbatim(self, http_client: MagicMock) -> None:
        body = {"ok": True, "score": 42}
        http_client.post.return_value = body
        tool = build_simulate_scoring_rule_tool(http_client)

        result = tool.func(conditions={"and": []}, event={"amountCents": 100})

        assert result == body
        http_client.post.assert_called_once_with(
            "/risk-scoring-rules/simulate",
            json_body={"conditions": {"and": []}, "event": {"amountCents": 100}},
        )

    def test_ok_false_body_returned_without_raising(self, http_client: MagicMock) -> None:
        body = {"ok": False, "errors": ["cycle detected"]}
        http_client.post.return_value = body
        tool = build_simulate_scoring_rule_tool(http_client)

        result = tool.func(conditions={"and": []}, event={})

        assert result == body


class TestAllToolsDispatchGuarded:
    @pytest.mark.parametrize(
        "builder_name",
        [
            "build_create_scoring_rule_tool",
            "build_update_scoring_rule_tool",
            "build_activate_scoring_rule_tool",
            "build_list_scoring_rules_tool",
            "build_get_scoring_rule_tool",
            "build_simulate_scoring_rule_tool",
        ],
    )
    def test_every_tool_name_is_in_authoring_tools(
        self, http_client: MagicMock, builder_name: str
    ) -> None:
        import fraud_companion.adapters.llm.authoring_tools as mod

        builder = getattr(mod, builder_name)
        tool = builder(http_client)
        assert tool.name in AUTHORING_TOOLS
