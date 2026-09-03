"""Tests for the two required LangChain StructuredTools.

get_analysis_pack is read-only; put_agent_brief is the only write tool.
"""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from fraud_companion.adapters.http.client import AntiFraudHttpClient
from fraud_companion.adapters.http.errors import CaseClosedError, CaseNotFoundError, ForbiddenRoleError
from fraud_companion.adapters.llm.tools import build_get_analysis_pack_tool, build_put_agent_brief_tool
from fraud_companion.domain.brief import BriefValidationError


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
