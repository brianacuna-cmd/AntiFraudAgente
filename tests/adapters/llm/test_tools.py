"""Tests for the two required LangChain StructuredTools.

get_analysis_pack is read-only; put_agent_brief is the only write tool.
"""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from fraud_companion.adapters.http.client import AntiFraudHttpClient
from fraud_companion.adapters.http.errors import CaseClosedError, CaseNotFoundError, ForbiddenRoleError
from fraud_companion.adapters.llm.tools import (
    build_get_analysis_pack_tool,
    build_list_aml_alerts_tool,
    build_list_cases_tool,
    build_put_agent_brief_tool,
)
from fraud_companion.domain.brief import BriefValidationError
from fraud_companion.domain.tools_spec import ALLOWED_TOOLS


@pytest.fixture
def http_client() -> MagicMock:
    return MagicMock(spec=AntiFraudHttpClient)


class TestGetAnalysisPackTool:
    def test_name_matches_allow_list(self, http_client: MagicMock) -> None:
        tool = build_get_analysis_pack_tool(http_client)
        assert tool.name == "get_analysis_pack"

    def test_returns_pack_unchanged(self, http_client: MagicMock) -> None:
        pack = {
            "case": {"id": "1"},
            "timeline": [],
            "snapshot": {"hits": [{"points": 10}]},
            "amlAlerts": [],
            "relatedCases": [],
            "agentBrief": None,
        }
        http_client.get.return_value = pack

        tool = build_get_analysis_pack_tool(http_client)
        result = tool.invoke({"case_id": "1"})

        assert result == pack
        # untouched: no "because" invented, no reshaping
        assert result["snapshot"]["hits"][0] == {"points": 10}
        http_client.get.assert_called_once_with("/cases/1/analysis-pack")

    def test_404_propagates_case_not_found_error(self, http_client: MagicMock) -> None:
        http_client.get.side_effect = CaseNotFoundError(
            status=404, code="CASE_NOT_FOUND", message="no case"
        )

        tool = build_get_analysis_pack_tool(http_client)

        with pytest.raises(CaseNotFoundError):
            tool.invoke({"case_id": "does-not-exist"})

    def test_has_pydantic_arg_schema_with_required_case_id(self, http_client: MagicMock) -> None:
        tool = build_get_analysis_pack_tool(http_client)
        schema = tool.args_schema
        assert "case_id" in schema.model_fields
        assert schema.model_fields["case_id"].is_required()


class TestPutAgentBriefTool:
    def test_name_matches_allow_list(self, http_client: MagicMock) -> None:
        tool = build_put_agent_brief_tool(http_client)
        assert tool.name == "put_agent_brief"

    def test_trims_and_sends_body(self, http_client: MagicMock) -> None:
        response_dto = {"id": "1", "status": "OPEN", "agentBrief": "hello world"}
        http_client.put.return_value = response_dto

        tool = build_put_agent_brief_tool(http_client)
        result = tool.invoke({"case_id": "1", "brief": "  hello world  "})

        http_client.put.assert_called_once_with(
            "/cases/1/agent-brief", json_body={"brief": "hello world"}
        )
        assert result == response_dto

    def test_rejects_empty_brief_without_http_call(self, http_client: MagicMock) -> None:
        tool = build_put_agent_brief_tool(http_client)

        with pytest.raises(BriefValidationError):
            tool.invoke({"case_id": "1", "brief": "   "})

        http_client.put.assert_not_called()

    def test_409_propagates_case_closed_error(self, http_client: MagicMock) -> None:
        http_client.put.side_effect = CaseClosedError(status=409, code="CASE_CLOSED", message="closed")

        tool = build_put_agent_brief_tool(http_client)

        with pytest.raises(CaseClosedError):
            tool.invoke({"case_id": "1", "brief": "hi"})

    def test_403_propagates_forbidden_role_error(self, http_client: MagicMock) -> None:
        http_client.put.side_effect = ForbiddenRoleError(
            status=403, code="FORBIDDEN_ROLE", message="humans forbidden"
        )

        tool = build_put_agent_brief_tool(http_client)

        with pytest.raises(ForbiddenRoleError):
            tool.invoke({"case_id": "1", "brief": "hi"})

    def test_has_pydantic_arg_schema_with_required_fields(self, http_client: MagicMock) -> None:
        tool = build_put_agent_brief_tool(http_client)
        schema = tool.args_schema
        assert "case_id" in schema.model_fields
        assert "brief" in schema.model_fields
        assert schema.model_fields["case_id"].is_required()
        assert schema.model_fields["brief"].is_required()


class TestListCasesTool:
    def test_name_matches_allow_list(self, http_client: MagicMock) -> None:
        tool = build_list_cases_tool(http_client)
        assert tool.name == "list_cases"

    def test_sends_customer_id_camel_case_and_never_organization_id(
        self, http_client: MagicMock
    ) -> None:
        http_client.get.return_value = {"items": [], "total": 0}

        tool = build_list_cases_tool(http_client)
        result = tool.invoke({"customer_id": "cust-1"})

        http_client.get.assert_called_once_with(
            "/cases", params={"customerId": "cust-1", "limit": 20, "offset": 0}
        )
        called_params = http_client.get.call_args.kwargs["params"]
        assert "organizationId" not in called_params
        assert "organization_id" not in called_params
        assert result == {"items": [], "total": 0}

    def test_limit_and_offset_passed_through(self, http_client: MagicMock) -> None:
        http_client.get.return_value = {"items": [], "total": 0}

        tool = build_list_cases_tool(http_client)
        tool.invoke({"customer_id": "cust-1", "limit": 5, "offset": 10})

        http_client.get.assert_called_once_with(
            "/cases", params={"customerId": "cust-1", "limit": 5, "offset": 10}
        )

    def test_has_pydantic_arg_schema_with_required_customer_id(
        self, http_client: MagicMock
    ) -> None:
        tool = build_list_cases_tool(http_client)
        schema = tool.args_schema
        assert "customer_id" in schema.model_fields
        assert schema.model_fields["customer_id"].is_required()


class TestListAmlAlertsTool:
    def test_name_matches_allow_list(self, http_client: MagicMock) -> None:
        tool = build_list_aml_alerts_tool(http_client)
        assert tool.name == "list_aml_alerts"

    def test_sends_customer_id_camel_case(self, http_client: MagicMock) -> None:
        http_client.get.return_value = {"items": [], "total": 0}

        tool = build_list_aml_alerts_tool(http_client)
        result = tool.invoke({"customer_id": "cust-1"})

        http_client.get.assert_called_once_with(
            "/aml-alerts", params={"customerId": "cust-1", "limit": 20, "offset": 0}
        )
        assert result == {"items": [], "total": 0}

    def test_limit_and_offset_passed_through(self, http_client: MagicMock) -> None:
        http_client.get.return_value = {"items": [], "total": 0}

        tool = build_list_aml_alerts_tool(http_client)
        tool.invoke({"customer_id": "cust-1", "limit": 5, "offset": 10})

        http_client.get.assert_called_once_with(
            "/aml-alerts", params={"customerId": "cust-1", "limit": 5, "offset": 10}
        )

    def test_has_pydantic_arg_schema_with_required_customer_id(
        self, http_client: MagicMock
    ) -> None:
        tool = build_list_aml_alerts_tool(http_client)
        schema = tool.args_schema
        assert "customer_id" in schema.model_fields
        assert schema.model_fields["customer_id"].is_required()


def test_all_four_tools_names_match_allowed_tools() -> None:
    from unittest.mock import MagicMock as _MagicMock

    http_client = _MagicMock(spec=AntiFraudHttpClient)
    tools = [
        build_get_analysis_pack_tool(http_client),
        build_put_agent_brief_tool(http_client),
        build_list_cases_tool(http_client),
        build_list_aml_alerts_tool(http_client),
    ]
    assert {tool.name for tool in tools} == ALLOWED_TOOLS
