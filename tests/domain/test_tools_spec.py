"""Tests for the tool allow-list guardrail (domain security core)."""
import pytest

from fraud_companion.domain.tools_spec import (
    ALLOWED_TOOLS,
    AUTHORING_TOOLS,
    DisallowedToolError,
    assert_tool_allowed,
    is_tool_allowed,
)

AUTHORING_NAMES = [
    "create_scoring_rule_via_factor_scoring",
    "update_scoring_rule",
    "activate_scoring_rule",
    "list_scoring_rules",
    "get_scoring_rule",
    "simulate_scoring_rule",
]


def test_allowed_tools_has_exactly_four_members():
    assert len(ALLOWED_TOOLS) == 4


def test_allowed_tools_is_a_frozenset():
    assert isinstance(ALLOWED_TOOLS, frozenset)


def test_allowed_tools_contains_exact_names():
    assert ALLOWED_TOOLS == frozenset(
        {"get_analysis_pack", "put_agent_brief", "list_cases", "list_aml_alerts"}
    )


@pytest.mark.parametrize(
    "tool_name",
    ["get_analysis_pack", "put_agent_brief", "list_cases", "list_aml_alerts"],
)
def test_is_tool_allowed_returns_true_for_allowed_names(tool_name):
    assert is_tool_allowed(tool_name) is True


@pytest.mark.parametrize(
    "tool_name",
    [
        "resolve",
        "archive",
        "create_case",
        "notes",
        "reassign",
        "enforce",
        "sar",
        "resolve_case",
        "CreateAgentApiKey",
        "",
        "get_analysis_pack ",
    ],
)
def test_is_tool_allowed_returns_false_for_disallowed_names(tool_name):
    assert is_tool_allowed(tool_name) is False


def test_assert_tool_allowed_passes_silently_for_allowed_tool():
    assert_tool_allowed("get_analysis_pack")


def test_assert_tool_allowed_raises_for_disallowed_tool():
    with pytest.raises(DisallowedToolError) as exc_info:
        assert_tool_allowed("resolve_case")

    assert "resolve_case" in str(exc_info.value)


def test_authoring_tools_has_exactly_six_members():
    assert len(AUTHORING_TOOLS) == 6


def test_authoring_tools_is_a_frozenset():
    assert isinstance(AUTHORING_TOOLS, frozenset)


def test_authoring_tools_contains_exact_names():
    assert AUTHORING_TOOLS == frozenset(AUTHORING_NAMES)


def test_authoring_tools_has_zero_overlap_with_allowed_tools():
    assert ALLOWED_TOOLS.isdisjoint(AUTHORING_TOOLS)


@pytest.mark.parametrize("tool_name", AUTHORING_NAMES)
def test_is_tool_allowed_accepts_authoring_names_under_authoring_set(tool_name):
    assert is_tool_allowed(tool_name, allowed=AUTHORING_TOOLS) is True


@pytest.mark.parametrize(
    "tool_name",
    ["get_analysis_pack", "put_agent_brief", "list_cases", "list_aml_alerts"],
)
def test_is_tool_allowed_rejects_case_names_under_authoring_set(tool_name):
    assert is_tool_allowed(tool_name, allowed=AUTHORING_TOOLS) is False


@pytest.mark.parametrize("tool_name", AUTHORING_NAMES)
def test_is_tool_allowed_rejects_authoring_names_under_default_allowed_set(tool_name):
    assert is_tool_allowed(tool_name) is False


@pytest.mark.parametrize("tool_name", AUTHORING_NAMES)
def test_assert_tool_allowed_passes_silently_for_authoring_tool(tool_name):
    assert_tool_allowed(tool_name, allowed=AUTHORING_TOOLS)


def test_assert_tool_allowed_raises_for_case_tool_under_authoring_set():
    with pytest.raises(DisallowedToolError):
        assert_tool_allowed("get_analysis_pack", allowed=AUTHORING_TOOLS)


def test_assert_tool_allowed_default_still_checks_the_four():
    assert len(ALLOWED_TOOLS) == 4
    with pytest.raises(DisallowedToolError):
        assert_tool_allowed("create_scoring_rule_via_factor_scoring")
