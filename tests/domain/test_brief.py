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


def test_brief_at_max_length_is_accepted():
    from fraud_companion.domain.brief import MAX_BRIEF_LENGTH

    brief = Brief.from_raw("a" * MAX_BRIEF_LENGTH)

    assert len(brief.value) == MAX_BRIEF_LENGTH


def test_brief_over_max_length_is_rejected():
    from fraud_companion.domain.brief import MAX_BRIEF_LENGTH

    with pytest.raises(BriefValidationError):
        Brief.from_raw("a" * (MAX_BRIEF_LENGTH + 1))
