"""Typed application settings loaded from environment variables.

No secret value is ever included in ``repr``/``str`` output.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field

DEFAULT_KAFKA_OUTBOX_TOPIC = "outbox.events"
DEFAULT_GEMINI_MODEL = "gemini-2.5-flash"
#: Low temperature: the brief must be factual and reproducible, never creative.
DEFAULT_GEMINI_TEMPERATURE = 0.2
#: Caps the brief length so output-token cost stays bounded per case.
DEFAULT_GEMINI_MAX_OUTPUT_TOKENS = 800
#: Per-request HTTP timeout (seconds) for the anti-fraud API client. Prevents
#: a slow/hung upstream from blocking the consumer poll thread indefinitely.
DEFAULT_HTTP_TIMEOUT_SECONDS = 30.0
DEFAULT_KAFKA_MAX_DELIVERY_ATTEMPTS = 5
DEFAULT_KAFKA_RETRY_BACKOFF_SECONDS = 1.0
DEFAULT_KAFKA_ON_EXHAUSTED = "crash"
#: Supported terminal actions when a message exhausts its delivery attempts.
#: "dead_letter" is intentionally NOT yet supported (needs a producer + DLQ
#: topic); see docs/design/consumer-error-handling.md.
_ON_EXHAUSTED_CHOICES = frozenset({"crash", "skip"})

_REQUIRED_ENV_VARS = (
    "ANTI_FRAUD_BASE_URL",
    "ANTI_FRAUD_AGENT_API_KEY",
    "GOOGLE_API_KEY",
    "KAFKA_BOOTSTRAP_SERVERS",
    "KAFKA_GROUP_ID",
    "KAFKA_ORGANIZATION_ID",
)

_SECRET_FIELDS = frozenset({"anti_fraud_agent_api_key", "google_api_key"})


def _parse_positive_int(raw: str | None, default: int, var_name: str) -> int:
    if raw is None or raw == "":
        return default
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"{var_name} must be an integer, got {raw!r}") from exc
    if value < 1:
        raise ValueError(f"{var_name} must be >= 1, got {value}")
    return value


def _parse_non_negative_float(raw: str | None, default: float, var_name: str) -> float:
    if raw is None or raw == "":
        return default
    try:
        value = float(raw)
    except ValueError as exc:
        raise ValueError(f"{var_name} must be a number, got {raw!r}") from exc
    if value < 0:
        raise ValueError(f"{var_name} must be >= 0, got {value}")
    return value


def _parse_positive_float(raw: str | None, default: float, var_name: str) -> float:
    if raw is None or raw == "":
        return default
    try:
        value = float(raw)
    except ValueError as exc:
        raise ValueError(f"{var_name} must be a number, got {raw!r}") from exc
    if value <= 0:
        raise ValueError(f"{var_name} must be > 0, got {value}")
    return value


def _parse_temperature(raw: str | None, default: float, var_name: str) -> float:
    if raw is None or raw == "":
        return default
    try:
        value = float(raw)
    except ValueError as exc:
        raise ValueError(f"{var_name} must be a number, got {raw!r}") from exc
    if not 0.0 <= value <= 2.0:
        raise ValueError(f"{var_name} must be between 0 and 2, got {value}")
    return value


def _parse_on_exhausted(raw: str) -> str:
    if raw not in _ON_EXHAUSTED_CHOICES:
        choices = ", ".join(sorted(_ON_EXHAUSTED_CHOICES))
        raise ValueError(
            f"KAFKA_ON_EXHAUSTED must be one of {{{choices}}}, got {raw!r}"
        )
    return raw


class MissingSettingError(RuntimeError):
    """Raised when a required environment variable is not set."""

    def __init__(self, var_name: str) -> None:
        super().__init__(
            f"Missing required environment variable: {var_name}"
        )
        self.var_name = var_name


@dataclass(frozen=True)
class Settings:
    anti_fraud_base_url: str
    anti_fraud_agent_api_key: str
    google_api_key: str
    kafka_bootstrap_servers: str
    kafka_group_id: str
    kafka_organization_id: str
    kafka_outbox_topic: str = field(default=DEFAULT_KAFKA_OUTBOX_TOPIC)
    gemini_model: str = field(default=DEFAULT_GEMINI_MODEL)
    gemini_temperature: float = field(default=DEFAULT_GEMINI_TEMPERATURE)
    gemini_max_output_tokens: int = field(
        default=DEFAULT_GEMINI_MAX_OUTPUT_TOKENS
    )
    http_timeout_seconds: float = field(default=DEFAULT_HTTP_TIMEOUT_SECONDS)
    kafka_max_delivery_attempts: int = field(
        default=DEFAULT_KAFKA_MAX_DELIVERY_ATTEMPTS
    )
    kafka_retry_backoff_seconds: float = field(
        default=DEFAULT_KAFKA_RETRY_BACKOFF_SECONDS
    )
    kafka_on_exhausted: str = field(default=DEFAULT_KAFKA_ON_EXHAUSTED)

    @classmethod
    def from_env(cls, env: dict[str, str] | None = None) -> "Settings":
        source = env if env is not None else os.environ

        values: dict[str, str] = {}
        for var_name in _REQUIRED_ENV_VARS:
            value = source.get(var_name)
            if not value:
                raise MissingSettingError(var_name)
            values[var_name] = value

        return cls(
            anti_fraud_base_url=values["ANTI_FRAUD_BASE_URL"],
            anti_fraud_agent_api_key=values["ANTI_FRAUD_AGENT_API_KEY"],
            google_api_key=values["GOOGLE_API_KEY"],
            kafka_bootstrap_servers=values["KAFKA_BOOTSTRAP_SERVERS"],
            kafka_group_id=values["KAFKA_GROUP_ID"],
            kafka_organization_id=values["KAFKA_ORGANIZATION_ID"],
            kafka_outbox_topic=source.get(
                "KAFKA_OUTBOX_TOPIC", DEFAULT_KAFKA_OUTBOX_TOPIC
            ),
            gemini_model=source.get("GEMINI_MODEL", DEFAULT_GEMINI_MODEL),
            gemini_temperature=_parse_temperature(
                source.get("GEMINI_TEMPERATURE"),
                DEFAULT_GEMINI_TEMPERATURE,
                "GEMINI_TEMPERATURE",
            ),
            gemini_max_output_tokens=_parse_positive_int(
                source.get("GEMINI_MAX_OUTPUT_TOKENS"),
                DEFAULT_GEMINI_MAX_OUTPUT_TOKENS,
                "GEMINI_MAX_OUTPUT_TOKENS",
            ),
            http_timeout_seconds=_parse_positive_float(
                source.get("HTTP_TIMEOUT_SECONDS"),
                DEFAULT_HTTP_TIMEOUT_SECONDS,
                "HTTP_TIMEOUT_SECONDS",
            ),
            kafka_max_delivery_attempts=_parse_positive_int(
                source.get("KAFKA_MAX_DELIVERY_ATTEMPTS"),
                DEFAULT_KAFKA_MAX_DELIVERY_ATTEMPTS,
                "KAFKA_MAX_DELIVERY_ATTEMPTS",
            ),
            kafka_retry_backoff_seconds=_parse_non_negative_float(
                source.get("KAFKA_RETRY_BACKOFF_SECONDS"),
                DEFAULT_KAFKA_RETRY_BACKOFF_SECONDS,
                "KAFKA_RETRY_BACKOFF_SECONDS",
            ),
            kafka_on_exhausted=_parse_on_exhausted(
                source.get("KAFKA_ON_EXHAUSTED", DEFAULT_KAFKA_ON_EXHAUSTED)
            ),
        )

    def __repr__(self) -> str:
        parts = []
        for f in self.__dataclass_fields__:
            if f in _SECRET_FIELDS:
                parts.append(f"{f}='***REDACTED***'")
            else:
                parts.append(f"{f}={getattr(self, f)!r}")
        return f"Settings({', '.join(parts)})"

    __str__ = __repr__
