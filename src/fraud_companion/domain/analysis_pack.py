"""Domain trim of the anti-fraud analysis pack before it reaches the LLM.

Defensive: missing fields/paths are left alone; unknown pack keys are kept.
Never invents hit ``because`` text. The untrusted-data framing is applied
*after* this transform by the tool adapter.
"""
from __future__ import annotations

from typing import Any

#: Timeline event types that only repeat facts already present as
#: ``agentBrief`` or ``snapshot`` on the pack root.
_DUPLICATE_TIMELINE_EVENT_TYPES = frozenset({"AGENT_BRIEFING", "SNAPSHOT_REFRESHED"})


def trim_analysis_pack(pack: Any) -> Any:
    """Return a slimmer copy of ``pack`` without dropping unknown structure.

    Concrete rules on the real GET /analysis-pack shape:

    * ``relatedCases[i]`` → ``{id, score, hits}`` when ``id`` is present
      (``score`` from ``score`` or ``riskScore``; ``hits`` from
      ``finturuCacheSnapshot.hits`` or a top-level ``hits``).
    * Timeline events whose ``eventType`` is ``AGENT_BRIEFING`` or
      ``SNAPSHOT_REFRESHED`` are dropped.
    """
    if not isinstance(pack, dict):
        return pack

    trimmed = dict(pack)

    related = trimmed.get("relatedCases")
    if isinstance(related, list):
        trimmed["relatedCases"] = [_trim_related_case(item) for item in related]

    timeline = trimmed.get("timeline")
    if isinstance(timeline, list):
        trimmed["timeline"] = [
            event for event in timeline if not _is_duplicate_timeline_event(event)
        ]

    return trimmed


def _trim_related_case(item: Any) -> Any:
    if not isinstance(item, dict) or "id" not in item:
        return item

    slim: dict[str, Any] = {"id": item["id"]}
    if "score" in item:
        slim["score"] = item["score"]
    elif "riskScore" in item:
        slim["score"] = item["riskScore"]

    snapshot = item.get("finturuCacheSnapshot")
    if isinstance(snapshot, dict) and "hits" in snapshot:
        slim["hits"] = snapshot["hits"]
    elif "hits" in item:
        slim["hits"] = item["hits"]
    return slim


def _is_duplicate_timeline_event(event: Any) -> bool:
    if not isinstance(event, dict):
        return False
    return event.get("eventType") in _DUPLICATE_TIMELINE_EVENT_TYPES
