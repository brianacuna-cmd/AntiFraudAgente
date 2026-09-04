"""Tests for the fraud_companion entrypoint (Slice 9).

Every external boundary (Settings.from_env, AntiFraudHttpClient,
build_agent, OutboxConsumer) is mocked — these tests never touch a real
broker, LLM, or HTTP endpoint, and never spin an infinite loop.
"""
from __future__ import annotations

import logging
from unittest.mock import MagicMock, patch

import pytest

from fraud_companion import __main__ as entrypoint
from fraud_companion.config import MissingSettingError, Settings


def _fake_settings() -> Settings:
    return Settings(
        anti_fraud_base_url="https://anti-fraud.example.com",
        anti_fraud_agent_api_key="super-secret-agent-key",
        google_api_key="super-secret-google-key",
        kafka_bootstrap_servers="localhost:9092",
        kafka_group_id="fraud-companion",
        kafka_organization_id="org-1",
    )


class TestMain:
    def test_main_wires_client_agent_and_consumer_then_runs(self) -> None:
        settings = _fake_settings()
        with (
            patch.object(entrypoint.Settings, "from_env", return_value=settings) as mock_from_env,
            patch.object(entrypoint, "AntiFraudHttpClient") as mock_client_cls,
            patch.object(entrypoint, "build_agent") as mock_build_agent,
            patch.object(entrypoint, "OutboxConsumer") as mock_consumer_cls,
        ):
            mock_consumer = MagicMock()
            mock_consumer_cls.return_value = mock_consumer

            entrypoint.main(should_stop=lambda: True)

            mock_from_env.assert_called_once()
            mock_client_cls.assert_called_once_with(
                settings.anti_fraud_base_url,
                settings.anti_fraud_agent_api_key,
                timeout=settings.http_timeout_seconds,
            )
            mock_build_agent.assert_called_once_with(settings, mock_client_cls.return_value)
            mock_consumer_cls.assert_called_once_with(
                settings=settings, agent=mock_build_agent.return_value
            )
            mock_consumer.run.assert_called_once()
            mock_consumer.close.assert_called_once()

    def test_main_closes_consumer_even_if_run_raises(self) -> None:
        settings = _fake_settings()
        with (
            patch.object(entrypoint.Settings, "from_env", return_value=settings),
            patch.object(entrypoint, "AntiFraudHttpClient"),
            patch.object(entrypoint, "build_agent"),
            patch.object(entrypoint, "OutboxConsumer") as mock_consumer_cls,
        ):
            mock_consumer = MagicMock()
            mock_consumer.run.side_effect = RuntimeError("boom")
            mock_consumer_cls.return_value = mock_consumer

            with pytest.raises(RuntimeError):
                entrypoint.main(should_stop=lambda: True)

            mock_consumer.close.assert_called_once()

    def test_main_fails_fast_with_clear_error_on_missing_env(self) -> None:
        with patch.object(
            entrypoint.Settings,
            "from_env",
            side_effect=MissingSettingError("GOOGLE_API_KEY"),
        ):
            with pytest.raises(MissingSettingError, match="GOOGLE_API_KEY"):
                entrypoint.main(should_stop=lambda: True)

    def test_main_never_logs_secret_values(self, caplog: pytest.LogCaptureFixture) -> None:
        settings = _fake_settings()
        with (
            patch.object(entrypoint.Settings, "from_env", return_value=settings),
            patch.object(entrypoint, "AntiFraudHttpClient"),
            patch.object(entrypoint, "build_agent"),
            patch.object(entrypoint, "OutboxConsumer") as mock_consumer_cls,
        ):
            mock_consumer_cls.return_value = MagicMock()
            with caplog.at_level(logging.DEBUG):
                entrypoint.main(should_stop=lambda: True)

        log_text = caplog.text
        assert settings.anti_fraud_agent_api_key not in log_text
        assert settings.google_api_key not in log_text

    def test_install_signal_handlers_stops_loop_on_signal(self) -> None:
        stop_flag = entrypoint._StopFlag()
        assert stop_flag() is False
        stop_flag.request_stop()
        assert stop_flag() is True


class TestConfigureLogging:
    def test_configure_logging_sets_a_handler(self) -> None:
        entrypoint.configure_logging()
        root = logging.getLogger()
        assert root.handlers
