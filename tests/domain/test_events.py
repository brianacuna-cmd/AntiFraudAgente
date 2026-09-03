"""Tests for domain event name constants."""
from fraud_companion.domain.events import CASE_CREATED_EVENT


def test_case_created_event_name_is_exact():
    assert CASE_CREATED_EVENT == "case.created"
