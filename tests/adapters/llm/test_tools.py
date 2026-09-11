"""Tests for the two required LangChain StructuredTools.

get_analysis_pack is read-only; put_agent_brief is the only write tool.
"""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from fraud_companion.adapters.http.client import AntiFraudHttpClient
from fraud_companion.adapters.http.errors import CaseClosedError, CaseNotFoundError, ForbiddenRoleError
from fraud_companion.adapters.llm.tools import (
    build_get_analysis_pack_tool,
    build_list_aml_alerts_tool,
    build_list_cases_tool,
    build_put_agent_brief_tool,
    fetch_framed_analysis_pack,
)
from fraud_companion.domain.analysis_pack import trim_analysis_pack
from fraud_companion.domain.brief import BriefValidationError
from fraud_companion.domain.policy import (
    UNTRUSTED_PACK_CLOSE,
    UNTRUSTED_PACK_OPEN,
    frame_untrusted_pack,
)
from fraud_companion.domain.tools_spec import ALLOWED_TOOLS, DisallowedToolError


@pytest.fixture
def http_client() -> MagicMock:
    return MagicMock(spec=AntiFraudHttpClient)


class TestGetAnalysisPackTool:
    def test_name_matches_allow_list(self, http_client: MagicMock) -> None:
        tool = build_get_analysis_pack_tool(http_client)
        assert tool.name == "get_analysis_pack"

    def test_returns_pack_wrapped_in_untrusted_data_markers(
        self, http_client: MagicMock
    ) -> None:
        pack = {
            "case": {"id": "1"},
            "timeline": [],
            "snapshot": {
                "hits": [
                    {"points": 10, "because": "matched a known mule account"},
                ]
            },
            "amlAlerts": [
                {"matchedEntry": {"name": "Jane Doe", "document": "AB123"}},
            ],
            "relatedCases": [],
            "agentBrief": "existing brief text",
        }
        http_client.get.return_value = pack

        tool = build_get_analysis_pack_tool(http_client)
        result = tool.invoke({"case_id": "1"})

        assert isinstance(result, str)
        assert result.startswith(UNTRUSTED_PACK_OPEN)
        assert result.endswith(UNTRUSTED_PACK_CLOSE)

        inner = result[len(UNTRUSTED_PACK_OPEN) : -len(UNTRUSTED_PACK_CLOSE)]
        parsed = json.loads(inner)
        # New contract: domain trim, then untrusted framing. Business facts stay.
        assert parsed == trim_analysis_pack(pack)
        assert "matched a known mule account" in inner
        assert "Jane Doe" in inner
        assert "AB123" in inner
        http_client.get.assert_called_once_with("/cases/1/analysis-pack")

    def test_trims_real_pack_before_untrusted_framing(
        self, http_client: MagicMock
    ) -> None:
        fixture = (
            Path(__file__).resolve().parents[2]
            / "fixtures"
            / "anonymized_analysis_pack.json"
        )
        pack = json.loads(fixture.read_text(encoding="utf-8"))
        http_client.get.return_value = pack

        tool = build_get_analysis_pack_tool(http_client)
        result = tool.invoke({"case_id": pack["case"]["id"]})

        inner = result[len(UNTRUSTED_PACK_OPEN) : -len(UNTRUSTED_PACK_CLOSE)]
        parsed = json.loads(inner)
        assert parsed == trim_analysis_pack(pack)
        assert parsed != pack
        event = parsed["snapshot"]["event"]
        assert event["provider"] == "internal"
        assert event["amountCents"] == 500000
        assert event["riskSignals"]["walletAgeDays"] == 2
        assert event["riskSignals"]["velocity24h"] == 9
        assert all("because" in hit for hit in parsed["snapshot"]["hits"])
        assert all(set(item) <= {"id", "score", "hits"} for item in parsed["relatedCases"])
        assert [event["eventType"] for event in parsed["timeline"]] == ["CASE_CREATED"]

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

    def test_is_a_terminal_tool(self, http_client: MagicMock) -> None:
        tool = build_put_agent_brief_tool(http_client)
        assert tool.return_direct is True


class TestReturnDirectFlagsAcrossTools:
    """Only put_agent_brief is terminal; the three read tools are not."""

    def test_get_analysis_pack_is_not_terminal(self, http_client: MagicMock) -> None:
        tool = build_get_analysis_pack_tool(http_client)
        assert not tool.return_direct

    def test_list_cases_is_not_terminal(self, http_client: MagicMock) -> None:
        tool = build_list_cases_tool(http_client)
        assert not tool.return_direct

    def test_list_aml_alerts_is_not_terminal(self, http_client: MagicMock) -> None:
        tool = build_list_aml_alerts_tool(http_client)
        assert not tool.return_direct


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


class TestDispatcherGuardWiredIntoRealTools:
    """Proves layer 2 (application/tool_dispatcher.assert_dispatch_allowed) is
    actually invoked from the REAL production tool-execution path, not just
    exercised by a test-authored mock. Each StructuredTool's underlying
    function must call assert_dispatch_allowed(<its own name>) before doing
    anything else, so even if the LangGraph middleware layer were bypassed
    or misconfigured, the tool body itself refuses to run.
    """

    def test_get_analysis_pack_tool_calls_real_dispatcher_before_http(
        self, http_client: MagicMock
    ) -> None:
        tool = build_get_analysis_pack_tool(http_client)

        with patch(
            "fraud_companion.adapters.llm.tools.assert_dispatch_allowed",
            side_effect=DisallowedToolError("get_analysis_pack"),
        ) as mock_dispatch:
            with pytest.raises(DisallowedToolError):
                tool.invoke({"case_id": "1"})

        mock_dispatch.assert_called_once_with("get_analysis_pack")
        http_client.get.assert_not_called()

    def test_put_agent_brief_tool_calls_real_dispatcher_before_http(
        self, http_client: MagicMock
    ) -> None:
        tool = build_put_agent_brief_tool(http_client)

        with patch(
            "fraud_companion.adapters.llm.tools.assert_dispatch_allowed",
            side_effect=DisallowedToolError("put_agent_brief"),
        ) as mock_dispatch:
            with pytest.raises(DisallowedToolError):
                tool.invoke({"case_id": "1", "brief": "hi"})

        mock_dispatch.assert_called_once_with("put_agent_brief")
        http_client.put.assert_not_called()

    def test_list_cases_tool_calls_real_dispatcher_before_http(
        self, http_client: MagicMock
    ) -> None:
        tool = build_list_cases_tool(http_client)

        with patch(
            "fraud_companion.adapters.llm.tools.assert_dispatch_allowed",
            side_effect=DisallowedToolError("list_cases"),
        ) as mock_dispatch:
            with pytest.raises(DisallowedToolError):
                tool.invoke({"customer_id": "cust-1"})

        mock_dispatch.assert_called_once_with("list_cases")
        http_client.get.assert_not_called()

    def test_list_aml_alerts_tool_calls_real_dispatcher_before_http(
        self, http_client: MagicMock
    ) -> None:
        tool = build_list_aml_alerts_tool(http_client)

        with patch(
            "fraud_companion.adapters.llm.tools.assert_dispatch_allowed",
            side_effect=DisallowedToolError("list_aml_alerts"),
        ) as mock_dispatch:
            with pytest.raises(DisallowedToolError):
                tool.invoke({"customer_id": "cust-1"})

        mock_dispatch.assert_called_once_with("list_aml_alerts")
        http_client.get.assert_not_called()

    def test_tool_built_under_disallowed_name_is_rejected_by_real_dispatcher(
        self, http_client: MagicMock
    ) -> None:
        """A tool registered under a name outside ALLOWED_TOOLS (e.g. from
        future drift/misconfiguration) must be rejected by the real
        production ``assert_dispatch_allowed`` call baked into the tool
        wrapper itself — not by any test-authored mock side effect.
        """
        from fraud_companion.adapters.llm.tools import _guard_dispatch

        def _fake_get_analysis_pack(case_id: str):  # pragma: no cover - must not run
            return http_client.get(f"/cases/{case_id}/analysis-pack")

        guarded = _guard_dispatch("resolve_case", _fake_get_analysis_pack)

        with pytest.raises(DisallowedToolError):
            guarded(case_id="1")

        http_client.get.assert_not_called()


class TestGuardrailAndPortBoundariesUnchanged:
    """Task 9: guardrail allow-list/layers and provider-agnostic boundaries
    are untouched by the untrusted-data framing change."""

    def test_allowed_tools_is_still_exactly_four_tools(self) -> None:
        assert ALLOWED_TOOLS == {
            "get_analysis_pack",
            "put_agent_brief",
            "list_cases",
            "list_aml_alerts",
        }

    def test_guardrail_module_builds_tool_guardrail_for_same_allow_list(self) -> None:
        from fraud_companion.adapters.llm.guardrail import build_tool_guardrail

        guardrail = build_tool_guardrail()
        assert guardrail is not None

    def test_tool_dispatcher_still_enforces_allow_list(self) -> None:
        from fraud_companion.application.tool_dispatcher import assert_dispatch_allowed
        from fraud_companion.domain.tools_spec import DisallowedToolError

        for name in ALLOWED_TOOLS:
            assert_dispatch_allowed(name)  # must not raise

        with pytest.raises(DisallowedToolError):
            assert_dispatch_allowed("resolve_case")

    def test_agent_port_module_has_no_provider_sdk_imports(self) -> None:
        import ast
        import inspect

        from fraud_companion.application import agent_port

        source = inspect.getsource(agent_port)
        tree = ast.parse(source)
        imported_modules = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported_modules.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported_modules.append(node.module)

        for forbidden in ("genai", "google.generativeai", "langchain"):
            assert not any(forbidden in mod for mod in imported_modules)


class TestFetchFramedAnalysisPack:
    """Task 1: shared fetch+trim+frame fn used by both the pre-fetch path
    and the get_analysis_pack tool path."""

    def test_returns_trimmed_and_framed_pack(self, http_client: MagicMock) -> None:
        pack = {
            "case": {"id": "1"},
            "timeline": [],
            "snapshot": {
                "hits": [
                    {"points": 10, "because": "matched a known mule account"},
                ]
            },
            "amlAlerts": [],
            "relatedCases": [],
            "agentBrief": None,
        }
        http_client.get.return_value = pack

        result = fetch_framed_analysis_pack(http_client, "1")

        assert result == frame_untrusted_pack(json.dumps(trim_analysis_pack(pack)))
        http_client.get.assert_called_once_with("/cases/1/analysis-pack")

    def test_tool_path_and_direct_path_are_byte_identical(
        self, http_client: MagicMock
    ) -> None:
        pack = {
            "case": {"id": "42"},
            "timeline": [],
            "snapshot": {
                "hits": [
                    {"points": 5, "because": "velocity anomaly"},
                ]
            },
            "amlAlerts": [],
            "relatedCases": [],
            "agentBrief": None,
        }
        http_client.get.return_value = pack

        tool = build_get_analysis_pack_tool(http_client)
        tool_result = tool.invoke({"case_id": "42"})

        direct_result = fetch_framed_analysis_pack(http_client, "42")

        assert tool_result == direct_result


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
