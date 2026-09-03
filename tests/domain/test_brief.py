"""Tests for the brief value object / validation policy."""
import pytest

from fraud_companion.domain.brief import Brief, BriefValidationError


def test_trims_surrounding_whitespace():
    brief = Brief.from_raw("  Customer shows elevated risk.  ")

    assert brief.value == "Customer shows elevated risk."


def test_valid_brief_passes_through_unchanged_when_already_trimmed():
    brief = Brief.from_raw("Clean brief text.")

    assert brief.value == "Clean brief text."


@pytest.mark.parametrize("raw", ["", "   ", "\t\n  "])
def test_empty_or_whitespace_only_brief_is_rejected(raw):
    with pytest.raises(BriefValidationError):
        Brief.from_raw(raw)


def test_equal_briefs_compare_equal():
    assert Brief.from_raw("same text") == Brief.from_raw("  same text  ")
