"""Brief value object: pure validation/normalization policy.

``put_agent_brief`` is the only mutating tool in the whole system; this
module defines the domain rule for what constitutes a valid brief.
"""
from __future__ import annotations

from dataclasses import dataclass


class BriefValidationError(ValueError):
    """Raised when a brief string fails domain validation."""


@dataclass(frozen=True)
class Brief:
    value: str

    @classmethod
    def from_raw(cls, raw: str) -> "Brief":
        trimmed = raw.strip()
        if not trimmed:
            raise BriefValidationError(
                "Brief must not be empty or whitespace-only"
            )
        return cls(value=trimmed)
