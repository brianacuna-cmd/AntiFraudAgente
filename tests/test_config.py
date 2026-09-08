"""Tests for fraud_companion.config.Settings."""
import pytest

from fraud_companion.config import Settings, MissingSettingError

REQUIRED_ENV = {
    "ANTI_FRAUD_BASE_URL": "https://anti-fraud.internal",
    "ANTI_FRAUD_AGENT_API_KEY": "super-secret-key",
    "LLM_API_KEY": "llm-secret-key",
    "KAFKA_BOOTSTRAP_SERVERS": "kafka-broker:9092",
    "KAFKA_GROUP_ID": "fraud-companion-consumer",
    "KAFKA_ORGANIZATION_ID": "org-123",
}


def _set_env(monkeypatch, overrides=None, omit=None):
    env = dict(REQUIRED_ENV)
    if overrides:
        env.update(overrides)
    if omit:
        for key in omit:
            env.pop(key, None)
    for key, value in env.items():
        monkeypatch.setenv(key, value)


def test_loads_all_required_settings_from_env(monkeypatch):
    _set_env(monkeypatch)

    settings = Settings.from_env()

    assert settings.anti_fraud_base_url == "https://anti-fraud.internal"
    assert settings.anti_fraud_agent_api_key == "super-secret-key"
    assert settings.llm_api_key == "llm-secret-key"
    assert settings.kafka_bootstrap_servers == "kafka-broker:9092"
    assert settings.kafka_group_id == "fraud-companion-consumer"
    assert settings.kafka_organization_id == "org-123"


def test_http_timeout_defaults(monkeypatch):
    _set_env(monkeypatch)

    settings = Settings.from_env()

    assert settings.http_timeout_seconds == 30.0


def test_http_timeout_can_be_overridden(monkeypatch):
    _set_env(monkeypatch, overrides={"HTTP_TIMEOUT_SECONDS": "12.5"})

    settings = Settings.from_env()

    assert settings.http_timeout_seconds == 12.5


def test_http_timeout_rejects_non_positive(monkeypatch):
    _set_env(monkeypatch, overrides={"HTTP_TIMEOUT_SECONDS": "0"})

    with pytest.raises(ValueError):
        Settings.from_env()


def test_llm_temperature_defaults_to_low_value(monkeypatch):
    _set_env(monkeypatch)

    settings = Settings.from_env()

    assert settings.llm_temperature == 0.2


def test_llm_temperature_can_be_overridden(monkeypatch):
    _set_env(monkeypatch, overrides={"LLM_TEMPERATURE": "0.7"})

    settings = Settings.from_env()

    assert settings.llm_temperature == 0.7


def test_llm_temperature_rejects_out_of_range(monkeypatch):
    _set_env(monkeypatch, overrides={"LLM_TEMPERATURE": "3"})

    with pytest.raises(ValueError):
        Settings.from_env()


def test_llm_max_output_tokens_defaults(monkeypatch):
    _set_env(monkeypatch)

    settings = Settings.from_env()

    assert settings.llm_max_output_tokens == 800


def test_llm_max_output_tokens_can_be_overridden(monkeypatch):
    _set_env(monkeypatch, overrides={"LLM_MAX_OUTPUT_TOKENS": "1200"})

    settings = Settings.from_env()

    assert settings.llm_max_output_tokens == 1200


def test_llm_max_output_tokens_rejects_non_positive(monkeypatch):
    _set_env(monkeypatch, overrides={"LLM_MAX_OUTPUT_TOKENS": "0"})

    with pytest.raises(ValueError):
        Settings.from_env()


def test_kafka_outbox_topic_defaults_to_outbox_events(monkeypatch):
    _set_env(monkeypatch)

    settings = Settings.from_env()

    assert settings.kafka_outbox_topic == "outbox.events"


def test_kafka_outbox_topic_can_be_overridden(monkeypatch):
    _set_env(monkeypatch, overrides={"KAFKA_OUTBOX_TOPIC": "custom.topic"})

    settings = Settings.from_env()

    assert settings.kafka_outbox_topic == "custom.topic"


def test_llm_model_has_a_sensible_default(monkeypatch):
    _set_env(monkeypatch)

    settings = Settings.from_env()

    assert settings.llm_model == "gemini-2.5-flash"
    assert isinstance(settings.llm_model, str)


def test_llm_model_can_be_overridden(monkeypatch):
    _set_env(monkeypatch, overrides={"LLM_MODEL": "gemini-custom-model"})

    settings = Settings.from_env()

    assert settings.llm_model == "gemini-custom-model"


def test_delivery_retry_settings_have_sensible_defaults(monkeypatch):
    _set_env(monkeypatch)

    settings = Settings.from_env()

    assert settings.kafka_max_delivery_attempts >= 1
    assert isinstance(settings.kafka_max_delivery_attempts, int)
    assert settings.kafka_retry_backoff_seconds >= 0
    assert isinstance(settings.kafka_retry_backoff_seconds, float)
    assert settings.kafka_on_exhausted == "crash"


def test_delivery_retry_settings_can_be_overridden(monkeypatch):
    _set_env(
        monkeypatch,
        overrides={
            "KAFKA_MAX_DELIVERY_ATTEMPTS": "7",
            "KAFKA_RETRY_BACKOFF_SECONDS": "2.5",
            "KAFKA_ON_EXHAUSTED": "skip",
        },
    )

    settings = Settings.from_env()

    assert settings.kafka_max_delivery_attempts == 7
    assert settings.kafka_retry_backoff_seconds == 2.5
    assert settings.kafka_on_exhausted == "skip"


@pytest.mark.parametrize("bad_value", ["dead_letter", "retry", "", "CRASH!"])
def test_invalid_on_exhausted_rejected(monkeypatch, bad_value):
    _set_env(monkeypatch, overrides={"KAFKA_ON_EXHAUSTED": bad_value})

    with pytest.raises(ValueError):
        Settings.from_env()


@pytest.mark.parametrize("bad_value", ["0", "-1", "abc"])
def test_invalid_max_delivery_attempts_rejected(monkeypatch, bad_value):
    _set_env(monkeypatch, overrides={"KAFKA_MAX_DELIVERY_ATTEMPTS": bad_value})

    with pytest.raises(ValueError):
        Settings.from_env()


@pytest.mark.parametrize(
    "missing_var",
    [
        "ANTI_FRAUD_BASE_URL",
        "ANTI_FRAUD_AGENT_API_KEY",
        "LLM_API_KEY",
        "KAFKA_BOOTSTRAP_SERVERS",
        "KAFKA_GROUP_ID",
        "KAFKA_ORGANIZATION_ID",
    ],
)
def test_missing_required_var_raises_clear_error(monkeypatch, missing_var):
    _set_env(monkeypatch, omit=[missing_var])

    with pytest.raises(MissingSettingError) as exc_info:
        Settings.from_env()

    assert missing_var in str(exc_info.value)


def test_only_legacy_google_api_key_fails_loud(monkeypatch):
    """Legacy GOOGLE_API_KEY/GEMINI_* env vars are no longer read."""
    _set_env(monkeypatch, omit=["LLM_API_KEY"])
    monkeypatch.setenv("GOOGLE_API_KEY", "legacy-google-key")
    monkeypatch.setenv("GEMINI_MODEL", "legacy-model")

    with pytest.raises(MissingSettingError) as exc_info:
        Settings.from_env()

    assert "LLM_API_KEY" in str(exc_info.value)


def test_repr_never_leaks_secret_values(monkeypatch):
    _set_env(monkeypatch)

    settings = Settings.from_env()
    rendered = repr(settings)

    assert "super-secret-key" not in rendered
    assert "llm-secret-key" not in rendered
