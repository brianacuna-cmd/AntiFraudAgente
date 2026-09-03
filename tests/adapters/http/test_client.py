"""Tests for the shared synchronous anti-fraud HTTP client."""
from __future__ import annotations

import httpx
import pytest
import respx

from fraud_companion.adapters.http.client import AntiFraudHttpClient
from fraud_companion.adapters.http.errors import (
    ApiError,
    CaseClosedError,
    CaseNotFoundError,
    ForbiddenCrossTenantError,
    ForbiddenRoleError,
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
