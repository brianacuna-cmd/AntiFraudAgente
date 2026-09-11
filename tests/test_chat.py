"""Tests for the authoring chat REPL entrypoint (Slice 3).

No real network/model/http call is ever made: ``Settings.from_env``,
``AntiFraudHttpClient``, and ``build_authoring_agent`` are patched at
their import site in ``fraud_companion.chat``.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from fraud_companion.chat import chat_main
from fraud_companion.config import Settings


@pytest.fixture
def settings() -> Settings:
    return Settings(
        anti_fraud_base_url="https://anti-fraud.internal",
        anti_fraud_agent_api_key="super-secret-key",
        llm_api_key="google-secret-key",
        kafka_bootstrap_servers="kafka-broker:9092",
        kafka_group_id="fraud-companion-consumer",
        kafka_organization_id="org-123",
    )


class _FakeAgent:
    """Records every invoke() call's message list and returns a canned reply."""

    def __init__(self) -> None:
        self.invocations: list[list] = []

    def invoke(self, payload: dict) -> dict:
        messages = list(payload["messages"])
        self.invocations.append(messages)
        reply = MagicMock()
        reply.content = f"reply-{len(self.invocations)}"
        return {"messages": messages + [reply]}


class TestChatMain:
    def test_builds_agent_from_settings_and_http_client(self, settings: Settings) -> None:
        fake_agent = _FakeAgent()
        with patch("fraud_companion.chat.Settings.from_env", return_value=settings), patch(
            "fraud_companion.chat.AntiFraudHttpClient"
        ) as mock_client_cls, patch(
            "fraud_companion.chat.build_authoring_agent", return_value=fake_agent
        ) as mock_build, patch("builtins.input", side_effect=EOFError):
            chat_main()

        mock_client_cls.assert_called_once()
        mock_build.assert_called_once_with(settings, mock_client_cls.return_value)

    def test_multi_turn_history_accumulates_across_inputs(self, settings: Settings) -> None:
        fake_agent = _FakeAgent()
        with patch("fraud_companion.chat.Settings.from_env", return_value=settings), patch(
            "fraud_companion.chat.AntiFraudHttpClient"
        ), patch(
            "fraud_companion.chat.build_authoring_agent", return_value=fake_agent
        ), patch(
            "builtins.input", side_effect=["create a rule", "now activate it", EOFError]
        ):
            chat_main()

        assert len(fake_agent.invocations) == 2
        # Second turn's message history includes the first turn's human
        # message and the agent's first reply — true multi-turn state.
        assert len(fake_agent.invocations[1]) == 3

    def test_exits_cleanly_on_eof(self, settings: Settings) -> None:
        fake_agent = _FakeAgent()
        with patch("fraud_companion.chat.Settings.from_env", return_value=settings), patch(
            "fraud_companion.chat.AntiFraudHttpClient"
        ), patch(
            "fraud_companion.chat.build_authoring_agent", return_value=fake_agent
        ), patch("builtins.input", side_effect=EOFError):
            chat_main()  # must not raise

    def test_exits_cleanly_on_keyboard_interrupt(self, settings: Settings) -> None:
        fake_agent = _FakeAgent()
        with patch("fraud_companion.chat.Settings.from_env", return_value=settings), patch(
            "fraud_companion.chat.AntiFraudHttpClient"
        ), patch(
            "fraud_companion.chat.build_authoring_agent", return_value=fake_agent
        ), patch("builtins.input", side_effect=KeyboardInterrupt):
            chat_main()  # must not raise
