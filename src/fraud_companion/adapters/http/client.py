"""Shared synchronous HTTP client for the anti-fraud API.

Every tool adapter (get_analysis_pack, put_agent_brief, list_cases,
list_aml_alerts) MUST use this client rather than calling httpx directly, so
that authentication, the /api/v1 prefix, and error-envelope mapping stay
centralized in one place.
"""
from __future__ import annotations

from typing import Any

import httpx

from fraud_companion.adapters.http.errors import map_error_response
from fraud_companion.config import DEFAULT_HTTP_TIMEOUT_SECONDS

_API_PREFIX = "/api/v1"


class AntiFraudHttpClient:
    """Thin synchronous wrapper around httpx for the anti-fraud API.

    Sends ``X-Agent-Api-Key`` on every request; NEVER sends an
    ``Authorization`` header. ``Content-Type: application/json`` is set only
    on PUT requests (there is a JSON body to describe).
    """

    def __init__(
        self, base_url: str, api_key: str, timeout: float | None = None
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._timeout = (
            timeout if timeout is not None else DEFAULT_HTTP_TIMEOUT_SECONDS
        )

    def _url(self, path: str) -> str:
        if not path.startswith("/"):
            path = f"/{path}"
        return f"{self._base_url}{_API_PREFIX}{path}"

    def _headers(self) -> dict[str, str]:
        # Never log this dict; it carries the plaintext API key.
        return {"X-Agent-Api-Key": self._api_key}

    def get(self, path: str, params: dict[str, Any] | None = None) -> Any:
        response = httpx.get(
            self._url(path),
            headers=self._headers(),
            params=params,
            timeout=self._timeout,
        )
        return self._handle_response(response)

    def put(self, path: str, json_body: Any) -> Any:
        headers = self._headers()
        headers["Content-Type"] = "application/json"
        response = httpx.put(
            self._url(path),
            headers=headers,
            json=json_body,
            timeout=self._timeout,
        )
        return self._handle_response(response)

    def post(self, path: str, json_body: Any) -> Any:
        headers = self._headers()
        headers["Content-Type"] = "application/json"
        response = httpx.post(
            self._url(path),
            headers=headers,
            json=json_body,
            timeout=self._timeout,
        )
        return self._handle_response(response)

    def patch(self, path: str, json_body: Any) -> Any:
        headers = self._headers()
        headers["Content-Type"] = "application/json"
        response = httpx.patch(
            self._url(path),
            headers=headers,
            json=json_body,
            timeout=self._timeout,
        )
        return self._handle_response(response)

    @staticmethod
    def _handle_response(response: httpx.Response) -> Any:
        if response.status_code // 100 == 2:
            if not response.content:
                return None
            return response.json()

        try:
            body: Any = response.json()
        except ValueError:
            body = None
        raise map_error_response(response.status_code, body)
