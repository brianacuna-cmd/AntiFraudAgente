"""Typed application settings loaded from environment variables.

No secret value is ever included in ``repr``/``str`` output.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field

DEFAULT_KAFKA_OUTBOX_TOPIC = "outbox.events"

_REQUIRED_ENV_VARS = (
    "ANTI_FRAUD_BASE_URL",
    "ANTI_FRAUD_AGENT_API_KEY",
    "GOOGLE_API_KEY",
    "KAFKA_BOOTSTRAP_SERVERS",
    "KAFKA_GROUP_ID",
    "KAFKA_ORGANIZATION_ID",
)

_SECRET_FIELDS = frozenset({"anti_fraud_agent_api_key", "google_api_key"})


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
