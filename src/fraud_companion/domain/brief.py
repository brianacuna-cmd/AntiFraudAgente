"""Brief value object: pure validation/normalization policy.

``put_agent_brief`` is the only mutating tool in the whole system; this
module defines the domain rule for what constitutes a valid brief.
"""
from __future__ import annotations

from dataclasses import dataclass

#: Upper bound on brief length. Defense-in-depth over the model's
#: ``max_output_tokens`` cap: bounds a runaway or injection-inflated brief
#: before it is persisted. Generous (~2x the typical 800-token brief) so it
#: never trips on legitimate output, only on pathological ones.
MAX_BRIEF_LENGTH = 8000


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
        if len(trimmed) > MAX_BRIEF_LENGTH:
            raise BriefValidationError(
                f"Brief exceeds the maximum length of {MAX_BRIEF_LENGTH} "
                f"characters (got {len(trimmed)})"
            )
        return cls(value=trimmed)
