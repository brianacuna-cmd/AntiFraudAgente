"""Fraud case companion entrypoint (Slice 9).

Wires ``Settings.from_env()`` -> ``AntiFraudHttpClient`` ->
``build_agent`` -> ``OutboxConsumer`` and runs the consumer poll loop
until asked to stop (SIGINT/SIGTERM or an injected ``should_stop``).

Nothing in this module ever logs a secret value (``GOOGLE_API_KEY``,
``ANTI_FRAUD_AGENT_API_KEY`` / the ``X-Agent-Api-Key`` header, or a raw
brief body). ``Settings.__repr__`` already redacts secret fields, and we
only ever log the case id and outcome, never the settings object or the
raw event payload.
"""
from __future__ import annotations

import logging
import signal
from typing import Callable

from fraud_companion.adapters.http.client import AntiFraudHttpClient
from fraud_companion.adapters.kafka.consumer import OutboxConsumer
from fraud_companion.adapters.llm.agent import build_agent
from fraud_companion.config import Settings

logger = logging.getLogger("fraud_companion")


def configure_logging(level: int = logging.INFO) -> None:
    """Configure standard logging with a plain, non-secret-leaking format.

    Only ever logs record messages built by this codebase (caseId,
    outcome, event types) — no code path in this application logs a
    ``Settings`` field flagged secret or a raw ``X-Agent-Api-Key`` header.
    """
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


class _StopFlag:
    """A tiny, testable stop condition toggled by OS signals.

    Passed as ``should_stop`` to ``OutboxConsumer.run`` so a SIGINT/SIGTERM
    cleanly exits the poll loop instead of killing the process mid-commit.
    """

    def __init__(self) -> None:
        self._stop = False

    def __call__(self) -> bool:
        return self._stop

    def request_stop(self, *_args: object) -> None:
        self._stop = True


def _install_signal_handlers(stop_flag: _StopFlag) -> None:
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            signal.signal(sig, stop_flag.request_stop)
        except (ValueError, OSError):
            # signal() only works on the main thread of the main
            # interpreter; skip silently outside that context (e.g. tests
            # running under certain runners).
            pass


def main(
    *,
    should_stop: Callable[[], bool] | None = None,
    poll_timeout: float = 1.0,
) -> None:
    """Load settings, wire the pipeline, and run the consumer poll loop.

    ``should_stop`` is injectable so tests (and this module's own signal
    handling) never depend on an unbounded ``while True`` loop or a real
    broker connection.
    """
    configure_logging()

    settings = Settings.from_env()
    logger.info(
        "Starting fraud_companion (kafka_bootstrap_servers=%s, topic=%s, group=%s)",
        settings.kafka_bootstrap_servers,
        settings.kafka_outbox_topic,
        settings.kafka_group_id,
    )

    http_client = AntiFraudHttpClient(
        settings.anti_fraud_base_url,
        settings.anti_fraud_agent_api_key,
        timeout=settings.http_timeout_seconds,
    )
    agent = build_agent(settings, http_client)
    consumer = OutboxConsumer(settings=settings, agent=agent)

    stop = should_stop
    if stop is None:
        stop_flag = _StopFlag()
        _install_signal_handlers(stop_flag)
        stop = stop_flag

    try:
        consumer.run(should_stop=stop, poll_timeout=poll_timeout)
    finally:
        consumer.close()
        logger.info("fraud_companion stopped")


if __name__ == "__main__":
    main()
