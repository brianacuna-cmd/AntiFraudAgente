"""Tests for the tool allow-list guardrail (domain security core)."""
import pytest

from fraud_companion.domain.tools_spec import (
    ALLOWED_TOOLS,
    DisallowedToolError,
    assert_tool_allowed,
    is_tool_allowed,
)


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
