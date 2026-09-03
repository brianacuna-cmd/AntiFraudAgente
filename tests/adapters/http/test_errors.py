"""Tests for the typed HTTP error hierarchy and error-envelope mapping."""
from __future__ import annotations

import pytest

from fraud_companion.adapters.http.errors import (
    ApiError,
    CaseClosedError,
    CaseNotFoundError,
    ForbiddenCrossTenantError,
    ForbiddenRoleError,
    UnauthenticatedError,
    ValidationError,
    map_error_response,
)


def test_401_unauthenticated_maps_to_unauthenticated_error():
    exc = map_error_response(
        401, {"error": {"code": "UNAUTHENTICATED", "message": "bad key", "metadata": {}}}
    )
    assert isinstance(exc, UnauthenticatedError)
    assert exc.status == 401
    assert exc.code == "UNAUTHENTICATED"
    assert exc.message == "bad key"
    assert exc.metadata == {}


def test_403_forbidden_role_maps_to_forbidden_role_error():
    exc = map_error_response(
        403, {"error": {"code": "FORBIDDEN_ROLE", "message": "no role", "metadata": {"role": "x"}}}
    )
    assert isinstance(exc, ForbiddenRoleError)
    assert exc.status == 403
    assert exc.metadata == {"role": "x"}


def test_403_forbidden_cross_tenant_maps_to_forbidden_cross_tenant_error():
    exc = map_error_response(
        403,
        {"error": {"code": "FORBIDDEN_CROSS_TENANT", "message": "wrong tenant", "metadata": {}}},
    )
    assert isinstance(exc, ForbiddenCrossTenantError)


def test_404_case_not_found_maps_to_case_not_found_error():
    exc = map_error_response(
        404, {"error": {"code": "CASE_NOT_FOUND", "message": "no case", "metadata": {}}}
    )
    assert isinstance(exc, CaseNotFoundError)


def test_409_case_closed_maps_to_case_closed_error():
    exc = map_error_response(
        409, {"error": {"code": "CASE_CLOSED", "message": "closed", "metadata": {}}}
    )
    assert isinstance(exc, CaseClosedError)


def test_400_validation_maps_to_validation_error():
    exc = map_error_response(
        400,
        {
            "error": {
                "code": "VALIDATION_ERROR",
                "message": "bad payload",
                "metadata": {"field": "text"},
            }
        },
    )
    assert isinstance(exc, ValidationError)
    assert exc.metadata == {"field": "text"}


def test_unknown_status_code_pair_maps_to_generic_api_error():
    exc = map_error_response(
        500, {"error": {"code": "INTERNAL", "message": "boom", "metadata": {}}}
    )
    assert type(exc) is ApiError
    assert exc.status == 500
    assert exc.code == "INTERNAL"
    assert exc.message == "boom"


def test_missing_body_produces_generic_api_error_with_defaults():
    exc = map_error_response(503, None)
    assert isinstance(exc, ApiError)
    assert exc.status == 503
    assert exc.code is None
    assert exc.message


def test_nonstandard_body_shape_produces_generic_api_error_gracefully():
    exc = map_error_response(400, {"unexpected": "shape"})
    assert isinstance(exc, ApiError)
    assert exc.status == 400
    assert exc.code is None


def test_all_domain_errors_are_api_error_subclasses():
    for cls in (
        UnauthenticatedError,
        ForbiddenRoleError,
        ForbiddenCrossTenantError,
        CaseNotFoundError,
        CaseClosedError,
        ValidationError,
    ):
        assert issubclass(cls, ApiError)


def test_api_error_str_includes_status_and_code():
    exc = ApiError(status=500, code="X", message="boom", metadata={})
    assert "500" in str(exc)
    assert "X" in str(exc)
