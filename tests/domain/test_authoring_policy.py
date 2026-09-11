"""Tests for the authoring agent's system prompt (Slice 3)."""
from __future__ import annotations

from fraud_companion.domain.authoring_policy import AUTHORING_SYSTEM_PROMPT
from fraud_companion.domain.policy import SECURITY_SYSTEM_PROMPT


class TestAuthoringSystemPrompt:
    def test_is_distinct_from_the_case_analyst_prompt(self) -> None:
        assert AUTHORING_SYSTEM_PROMPT != SECURITY_SYSTEM_PROMPT

    def test_authorizes_the_scoring_rule_authoring_workflow(self) -> None:
        prompt_lower = AUTHORING_SYSTEM_PROMPT.lower()
        for keyword in (
            "create",
            "factor-scoring",
            "update",
            "activate",
            "list",
            "get",
            "simulate",
        ):
            assert keyword in prompt_lower

    def test_forbids_resolving_or_closing_a_case(self) -> None:
        prompt_lower = AUTHORING_SYSTEM_PROMPT.lower()
        assert "resolve" in prompt_lower or "close" in prompt_lower
        assert "forbid" in prompt_lower or "never" in prompt_lower

    def test_forbids_sar_filing(self) -> None:
        assert "SAR" in AUTHORING_SYSTEM_PROMPT

    def test_forbids_setting_a_numeric_risk_score_on_a_case(self) -> None:
        prompt_lower = AUTHORING_SYSTEM_PROMPT.lower()
        assert "riskscore" in prompt_lower

    def test_describes_the_standard_draft_then_activate_flow(self) -> None:
        prompt_lower = AUTHORING_SYSTEM_PROMPT.lower()
        assert "inactive" in prompt_lower
