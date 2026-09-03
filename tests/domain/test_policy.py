"""Tests for the security system prompt policy constant.

These assertions check for the PRESENCE of each hard rule's intent, not
exact wording, so the prompt copy can be refined without breaking tests.
"""
from __future__ import annotations

from fraud_companion.domain.policy import SECURITY_SYSTEM_PROMPT
from fraud_companion.domain.tools_spec import ALLOWED_TOOLS


class TestSecuritySystemPrompt:
    def test_is_a_non_empty_string(self) -> None:
        assert isinstance(SECURITY_SYSTEM_PROMPT, str)
        assert SECURITY_SYSTEM_PROMPT.strip()

    def test_prohibits_rescoring(self) -> None:
        lowered = SECURITY_SYSTEM_PROMPT.lower()
        assert "re-score" in lowered or "rescor" in lowered

    def test_names_exactly_the_four_allowed_tools(self) -> None:
        for name in ALLOWED_TOOLS:
            assert name in SECURITY_SYSTEM_PROMPT

    def test_prohibits_inventing_hit_because(self) -> None:
        assert "because" in SECURITY_SYSTEM_PROMPT.lower()

    def test_prohibits_resolve_and_enforcement_and_sar(self) -> None:
        lowered = SECURITY_SYSTEM_PROMPT.lower()
        assert "resolve" in lowered
        assert "enforce" in lowered
        assert "sar" in lowered

    def test_states_case_stays_open(self) -> None:
        assert "open" in SECURITY_SYSTEM_PROMPT.lower()
