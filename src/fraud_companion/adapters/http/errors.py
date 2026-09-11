"""Typed exception hierarchy mapping the anti-fraud API error envelope.

Response envelope shape on any 4xx/5xx:
    {"error": {"code": str, "message": str, "metadata": dict}}

``map_error_response`` maps (status, code) pairs to specific typed
exceptions; anything unrecognized falls back to the generic ``ApiError``.
"""
from __future__ import annotations

from typing import Any


class ApiError(Exception):
    """Base class for all typed anti-fraud API errors.

    Carries the HTTP status, the API error code (if any), the human
    readable message, and any structured metadata from the envelope.
    """

    def __init__(
        self,
        status: int,
        code: str | None,
        message: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self.status = status
        self.code = code
        self.message = message
        self.metadata = metadata if metadata is not None else {}
        super().__init__(f"[{status}] {code}: {message}")


class UnauthenticatedError(ApiError):
    """401 UNAUTHENTICATED — MUST NOT be retried or escalated; no fallback."""


class ForbiddenRoleError(ApiError):
    """403 FORBIDDEN_ROLE."""


class ForbiddenCrossTenantError(ApiError):
    """403 FORBIDDEN_CROSS_TENANT."""


class CaseNotFoundError(ApiError):
    """404 CASE_NOT_FOUND."""


class CaseClosedError(ApiError):
    """409 CASE_CLOSED."""


class ValidationError(ApiError):
    """400 validation failure."""


class InvariantViolationError(ApiError):
    """400 INVARIANT_VIOLATION — scoring-rule domain invariant broken."""


class ScoringRuleNotFoundError(ApiError):
    """404 SCORING_RULE_NOT_FOUND."""


class ScoringRuleActiveError(ApiError):
    """409 SCORING_RULE_ACTIVE — rule is already active."""


_STATUS_CODE_MAP: dict[tuple[int, str], type[ApiError]] = {
    (401, "UNAUTHENTICATED"): UnauthenticatedError,
    (403, "FORBIDDEN_ROLE"): ForbiddenRoleError,
    (403, "FORBIDDEN_CROSS_TENANT"): ForbiddenCrossTenantError,
    (404, "CASE_NOT_FOUND"): CaseNotFoundError,
    (409, "CASE_CLOSED"): CaseClosedError,
    (400, "VALIDATION_ERROR"): ValidationError,
    (400, "INVARIANT_VIOLATION"): InvariantViolationError,
    (404, "SCORING_RULE_NOT_FOUND"): ScoringRuleNotFoundError,
    (409, "SCORING_RULE_ACTIVE"): ScoringRuleActiveError,
}


def map_error_response(status: int, body: Any) -> ApiError:
    """Map an HTTP status + parsed JSON body to a typed ``ApiError``.

    Gracefully handles a missing or nonstandard-shaped body.
    """
    error_obj: dict[str, Any] = {}
    if isinstance(body, dict):
        maybe_error = body.get("error")
        if isinstance(maybe_error, dict):
            error_obj = maybe_error

    code = error_obj.get("code")
    message = error_obj.get("message") or f"HTTP {status} error with no error envelope"
    metadata = error_obj.get("metadata") or {}

    exc_cls = _STATUS_CODE_MAP.get((status, code)) if code else None
    if exc_cls is None:
        return ApiError(status=status, code=code, message=message, metadata=metadata)
    return exc_cls(status=status, code=code, message=message, metadata=metadata)
