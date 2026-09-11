"""Tests for the shared synchronous anti-fraud HTTP client."""
from __future__ import annotations

from unittest.mock import patch

import httpx
import pytest
import respx

from fraud_companion import config
from fraud_companion.adapters.http.client import AntiFraudHttpClient
from fraud_companion.adapters.http.errors import (
    ApiError,
    CaseClosedError,
    CaseNotFoundError,
    ForbiddenCrossTenantError,
    ForbiddenRoleError,
    ScoringRuleNotFoundError,
    UnauthenticatedError,
    ValidationError,
)

BASE_URL = "https://anti-fraud.example.com"
API_KEY = "super-secret-key"


@pytest.fixture
def client() -> AntiFraudHttpClient:
    return AntiFraudHttpClient(base_url=BASE_URL, api_key=API_KEY)


@respx.mock
def test_get_sends_api_key_header_and_no_authorization(client: AntiFraudHttpClient):
    route = respx.get(f"{BASE_URL}/api/v1/cases/123").mock(
        return_value=httpx.Response(200, json={"id": "123"})
    )

    result = client.get("/cases/123")

    assert result == {"id": "123"}
    sent = route.calls.last.request
    assert sent.headers["X-Agent-Api-Key"] == API_KEY
    assert "Authorization" not in sent.headers


def test_get_forwards_configured_timeout_to_httpx():
    client = AntiFraudHttpClient(base_url=BASE_URL, api_key=API_KEY, timeout=12.5)
    with patch("fraud_companion.adapters.http.client.httpx.get") as mock_get:
        mock_get.return_value = httpx.Response(200, json={})
        client.get("/cases/1")

    _, kwargs = mock_get.call_args
    assert kwargs["timeout"] == 12.5


def test_put_forwards_configured_timeout_to_httpx():
    client = AntiFraudHttpClient(base_url=BASE_URL, api_key=API_KEY, timeout=8.0)
    with patch("fraud_companion.adapters.http.client.httpx.put") as mock_put:
        mock_put.return_value = httpx.Response(200, json={})
        client.put("/cases/1/brief", json_body={"brief": "x"})

    _, kwargs = mock_put.call_args
    assert kwargs["timeout"] == 8.0


def test_get_uses_a_default_timeout_when_unspecified():
    client = AntiFraudHttpClient(base_url=BASE_URL, api_key=API_KEY)
    with patch("fraud_companion.adapters.http.client.httpx.get") as mock_get:
        mock_get.return_value = httpx.Response(200, json={})
        client.get("/cases/1")

    _, kwargs = mock_get.call_args
    assert kwargs["timeout"] is not None
    assert kwargs["timeout"] > 0


def test_default_timeout_tracks_single_source_config_constant():
    client = AntiFraudHttpClient(base_url=BASE_URL, api_key=API_KEY)
    with patch("fraud_companion.adapters.http.client.httpx.get") as mock_get:
        mock_get.return_value = httpx.Response(200, json={})
        client.get("/cases/1")

    _, kwargs = mock_get.call_args
    assert kwargs["timeout"] == config.DEFAULT_HTTP_TIMEOUT_SECONDS


@respx.mock
def test_get_does_not_send_content_type_json(client: AntiFraudHttpClient):
    route = respx.get(f"{BASE_URL}/api/v1/cases/123").mock(
        return_value=httpx.Response(200, json={"id": "123"})
    )

    client.get("/cases/123")

    sent = route.calls.last.request
    content_type = sent.headers.get("content-type")
    assert content_type is None or "application/json" not in content_type


@respx.mock
def test_get_applies_query_params(client: AntiFraudHttpClient):
    route = respx.get(f"{BASE_URL}/api/v1/cases").mock(
        return_value=httpx.Response(200, json={"items": []})
    )

    client.get("/cases", params={"status": "open"})

    sent = route.calls.last.request
    assert sent.url.params["status"] == "open"


@respx.mock
def test_put_sends_api_key_header_content_type_json_and_no_authorization(
    client: AntiFraudHttpClient,
):
    route = respx.put(f"{BASE_URL}/api/v1/cases/123/brief").mock(
        return_value=httpx.Response(200, json={"ok": True})
    )

    result = client.put("/cases/123/brief", json_body={"text": "hello"})

    assert result == {"ok": True}
    sent = route.calls.last.request
    assert sent.headers["X-Agent-Api-Key"] == API_KEY
    assert "Authorization" not in sent.headers
    assert "application/json" in sent.headers["content-type"]
    assert sent.content == b'{"text": "hello"}' or b"hello" in sent.content


@respx.mock
def test_base_url_and_api_v1_prefix_applied(client: AntiFraudHttpClient):
    route = respx.get(f"{BASE_URL}/api/v1/aml-alerts").mock(
        return_value=httpx.Response(200, json=[])
    )

    client.get("/aml-alerts")

    assert route.called


@respx.mock
def test_unwrapped_response_returned_as_is(client: AntiFraudHttpClient):
    # analysis-pack style response: unwrapped JSON (not {items: ...})
    respx.get(f"{BASE_URL}/api/v1/cases/1/analysis-pack").mock(
        return_value=httpx.Response(200, json={"summary": "x", "signals": [1, 2]})
    )

    result = client.get("/cases/1/analysis-pack")

    assert result == {"summary": "x", "signals": [1, 2]}


@respx.mock
def test_401_raises_unauthenticated_error_and_does_not_retry(client: AntiFraudHttpClient):
    route = respx.get(f"{BASE_URL}/api/v1/cases/1").mock(
        return_value=httpx.Response(
            401,
            json={"error": {"code": "UNAUTHENTICATED", "message": "bad key", "metadata": {}}},
        )
    )

    with pytest.raises(UnauthenticatedError) as exc_info:
        client.get("/cases/1")

    assert route.call_count == 1
    assert exc_info.value.status == 401


@respx.mock
def test_403_forbidden_role_raises_forbidden_role_error(client: AntiFraudHttpClient):
    respx.get(f"{BASE_URL}/api/v1/cases/1").mock(
        return_value=httpx.Response(
            403,
            json={"error": {"code": "FORBIDDEN_ROLE", "message": "no role", "metadata": {}}},
        )
    )

    with pytest.raises(ForbiddenRoleError):
        client.get("/cases/1")


@respx.mock
def test_403_forbidden_cross_tenant_raises_forbidden_cross_tenant_error(
    client: AntiFraudHttpClient,
):
    respx.get(f"{BASE_URL}/api/v1/cases/1").mock(
        return_value=httpx.Response(
            403,
            json={
                "error": {
                    "code": "FORBIDDEN_CROSS_TENANT",
                    "message": "wrong tenant",
                    "metadata": {},
                }
            },
        )
    )

    with pytest.raises(ForbiddenCrossTenantError):
        client.get("/cases/1")


@respx.mock
def test_404_raises_case_not_found_error(client: AntiFraudHttpClient):
    respx.get(f"{BASE_URL}/api/v1/cases/999").mock(
        return_value=httpx.Response(
            404,
            json={"error": {"code": "CASE_NOT_FOUND", "message": "no case", "metadata": {}}},
        )
    )

    with pytest.raises(CaseNotFoundError):
        client.get("/cases/999")


@respx.mock
def test_409_raises_case_closed_error(client: AntiFraudHttpClient):
    respx.put(f"{BASE_URL}/api/v1/cases/1/brief").mock(
        return_value=httpx.Response(
            409,
            json={"error": {"code": "CASE_CLOSED", "message": "closed", "metadata": {}}},
        )
    )

    with pytest.raises(CaseClosedError):
        client.put("/cases/1/brief", json_body={"text": "x"})


@respx.mock
def test_400_validation_raises_validation_error_with_metadata(client: AntiFraudHttpClient):
    respx.put(f"{BASE_URL}/api/v1/cases/1/brief").mock(
        return_value=httpx.Response(
            400,
            json={
                "error": {
                    "code": "VALIDATION_ERROR",
                    "message": "bad payload",
                    "metadata": {"field": "text"},
                }
            },
        )
    )

    with pytest.raises(ValidationError) as exc_info:
        client.put("/cases/1/brief", json_body={"text": ""})

    assert exc_info.value.metadata == {"field": "text"}


@respx.mock
def test_unmapped_5xx_raises_generic_api_error(client: AntiFraudHttpClient):
    respx.get(f"{BASE_URL}/api/v1/cases/1").mock(
        return_value=httpx.Response(
            500,
            json={"error": {"code": "INTERNAL", "message": "boom", "metadata": {}}},
        )
    )

    with pytest.raises(ApiError) as exc_info:
        client.get("/cases/1")

    assert exc_info.value.status == 500
    assert exc_info.value.code == "INTERNAL"


@respx.mock
def test_error_body_missing_is_handled_gracefully(client: AntiFraudHttpClient):
    respx.get(f"{BASE_URL}/api/v1/cases/1").mock(return_value=httpx.Response(503))

    with pytest.raises(ApiError) as exc_info:
        client.get("/cases/1")

    assert exc_info.value.status == 503


@respx.mock
def test_error_body_nonstandard_shape_is_handled_gracefully(client: AntiFraudHttpClient):
    respx.get(f"{BASE_URL}/api/v1/cases/1").mock(
        return_value=httpx.Response(400, json={"unexpected": "shape"})
    )

    with pytest.raises(ApiError) as exc_info:
        client.get("/cases/1")

    assert exc_info.value.status == 400
    assert exc_info.value.code is None


# --- post/patch (Slice 1: scoring-rule-authoring-tools) ---


@respx.mock
def test_post_sends_api_key_header_content_type_json_and_no_authorization(
    client: AntiFraudHttpClient,
):
    route = respx.post(f"{BASE_URL}/api/v1/risk-scoring-rules/factor-scoring").mock(
        return_value=httpx.Response(201, json={"id": "r1", "status": "INACTIVE"})
    )

    result = client.post(
        "/risk-scoring-rules/factor-scoring", json_body={"name": "rule-1"}
    )

    assert result == {"id": "r1", "status": "INACTIVE"}
    sent = route.calls.last.request
    assert sent.headers["X-Agent-Api-Key"] == API_KEY
    assert "Authorization" not in sent.headers
    assert "application/json" in sent.headers["content-type"]


@respx.mock
def test_post_url_uses_api_v1_prefix_and_never_sends_organization_id(
    client: AntiFraudHttpClient,
):
    route = respx.post(f"{BASE_URL}/api/v1/risk-scoring-rules/factor-scoring").mock(
        return_value=httpx.Response(201, json={"id": "r1"})
    )

    client.post("/risk-scoring-rules/factor-scoring", json_body={"name": "rule-1"})

    sent = route.calls.last.request
    assert str(sent.url) == f"{BASE_URL}/api/v1/risk-scoring-rules/factor-scoring"
    assert "organizationId" not in (sent.content.decode() if sent.content else "")
    assert "organizationId" not in sent.headers
    assert "organizationId" not in sent.url.params


def test_post_forwards_configured_timeout_to_httpx():
    client = AntiFraudHttpClient(base_url=BASE_URL, api_key=API_KEY, timeout=9.5)
    with patch("fraud_companion.adapters.http.client.httpx.post") as mock_post:
        mock_post.return_value = httpx.Response(201, json={})
        client.post("/risk-scoring-rules/factor-scoring", json_body={"name": "x"})

    _, kwargs = mock_post.call_args
    assert kwargs["timeout"] == 9.5


@respx.mock
def test_patch_empty_body_200_returns_none(client: AntiFraudHttpClient):
    respx.patch(f"{BASE_URL}/api/v1/risk-scoring-rules/r1").mock(
        return_value=httpx.Response(200)
    )

    result = client.patch("/risk-scoring-rules/r1", json_body={"name": "New"})

    assert result is None


@respx.mock
def test_patch_url_uses_api_v1_prefix_and_never_sends_organization_id(
    client: AntiFraudHttpClient,
):
    route = respx.patch(f"{BASE_URL}/api/v1/risk-scoring-rules/r1").mock(
        return_value=httpx.Response(200)
    )

    client.patch("/risk-scoring-rules/r1", json_body={"name": "New"})

    sent = route.calls.last.request
    assert str(sent.url) == f"{BASE_URL}/api/v1/risk-scoring-rules/r1"
    assert "organizationId" not in (sent.content.decode() if sent.content else "")
    assert "organizationId" not in sent.headers


def test_patch_forwards_configured_timeout_to_httpx():
    client = AntiFraudHttpClient(base_url=BASE_URL, api_key=API_KEY, timeout=6.0)
    with patch("fraud_companion.adapters.http.client.httpx.patch") as mock_patch:
        mock_patch.return_value = httpx.Response(200)
        client.patch("/risk-scoring-rules/r1", json_body={"name": "New"})

    _, kwargs = mock_patch.call_args
    assert kwargs["timeout"] == 6.0


@respx.mock
def test_patch_404_scoring_rule_not_found_raises_typed_error(client: AntiFraudHttpClient):
    respx.patch(f"{BASE_URL}/api/v1/risk-scoring-rules/r1").mock(
        return_value=httpx.Response(
            404,
            json={
                "error": {
                    "code": "SCORING_RULE_NOT_FOUND",
                    "message": "no rule",
                    "metadata": {},
                }
            },
        )
    )

    with pytest.raises(ScoringRuleNotFoundError):
        client.patch("/risk-scoring-rules/r1", json_body={"name": "New"})
