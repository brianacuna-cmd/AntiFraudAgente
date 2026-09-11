"""Tests for trim_analysis_pack against the real analysis-pack shape."""
from __future__ import annotations

import copy
import json
from pathlib import Path

from fraud_companion.domain.analysis_pack import trim_analysis_pack

FIXTURE_PATH = (
    Path(__file__).resolve().parent.parent
    / "fixtures"
    / "anonymized_analysis_pack.json"
)


def _approx_tokens(value: object) -> int:
    """Same estimator the live pack report used: compact JSON chars / 4."""
    return len(json.dumps(value, ensure_ascii=False, separators=(",", ":"))) // 4


def _load_fixture() -> dict:
    return json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))


class TestTrimAnalysisPackDefensive:
    def test_non_dict_pack_is_returned_as_is(self) -> None:
        assert trim_analysis_pack("not-a-pack") == "not-a-pack"
        assert trim_analysis_pack([1, 2]) == [1, 2]
        assert trim_analysis_pack(None) is None

    def test_missing_related_cases_and_timeline_are_left_alone(self) -> None:
        pack = {"case": {"id": "1"}, "snapshot": {"hits": []}, "extra": True}
        assert trim_analysis_pack(pack) == pack

    def test_unknown_top_level_keys_are_kept(self) -> None:
        pack = {
            "case": {"id": "1"},
            "relatedCases": [],
            "timeline": [],
            "debugTrace": {"raw": 1},
        }
        trimmed = trim_analysis_pack(pack)
        assert trimmed["debugTrace"] == {"raw": 1}

    def test_related_case_without_id_is_left_intact(self) -> None:
        original = {"customerId": "cust-demo-001", "riskScore": 61, "tags": ["x"]}
        pack = {"relatedCases": [original]}
        assert trim_analysis_pack(pack)["relatedCases"][0] == original

    def test_non_dict_related_item_is_left_intact(self) -> None:
        pack = {"relatedCases": ["sibling-id"]}
        assert trim_analysis_pack(pack)["relatedCases"] == ["sibling-id"]

    def test_does_not_mutate_input(self) -> None:
        pack = {
            "relatedCases": [
                {
                    "id": "sib",
                    "riskScore": 80,
                    "finturuCacheSnapshot": {"hits": [{"points": 1}]},
                    "customerEmail": "keep-on-original@example.test",
                }
            ],
            "timeline": [{"eventType": "AGENT_BRIEFING", "newValue": "brief"}],
        }
        snapshot = copy.deepcopy(pack)
        trim_analysis_pack(pack)
        assert pack == snapshot


class TestTrimRelatedCases:
    def test_projects_related_cases_to_id_score_hits(self) -> None:
        pack = {
            "relatedCases": [
                {
                    "id": "sib-1",
                    "organizationId": "org",
                    "customerId": "cust-demo-001",
                    "customerEmail": "hidden@example.test",
                    "riskScore": 61,
                    "finturuCacheSnapshot": {
                        "hits": [
                            {
                                "id": "r-high-amount",
                                "points": 50,
                                "because": "amountCents >= 300000",
                            }
                        ]
                    },
                    "agentBrief": "a long prior brief that must not survive",
                }
            ]
        }
        trimmed = trim_analysis_pack(pack)
        assert trimmed["relatedCases"] == [
            {
                "id": "sib-1",
                "score": 61,
                "hits": [
                    {
                        "id": "r-high-amount",
                        "points": 50,
                        "because": "amountCents >= 300000",
                    }
                ],
            }
        ]

    def test_already_slim_related_case_is_idempotent(self) -> None:
        slim = {"id": "sib-1", "score": 61, "hits": [{"points": 10}]}
        pack = {"relatedCases": [slim]}
        assert trim_analysis_pack(pack)["relatedCases"] == [slim]


class TestTrimTimelineDuplicates:
    def test_drops_agent_briefing_and_snapshot_refreshed_keeps_case_created(
        self,
    ) -> None:
        pack = {
            "timeline": [
                {"id": "1", "eventType": "CASE_CREATED", "newValue": "OPEN"},
                {"id": "2", "eventType": "AGENT_BRIEFING", "newValue": "brief"},
                {"id": "3", "eventType": "SNAPSHOT_REFRESHED", "newValue": "{}"},
                {"id": "4", "eventType": "PRIORITY_CHANGED", "newValue": "HIGH"},
            ]
        }
        trimmed = trim_analysis_pack(pack)
        assert [event["eventType"] for event in trimmed["timeline"]] == [
            "CASE_CREATED",
            "PRIORITY_CHANGED",
        ]

    def test_non_dict_timeline_item_is_kept(self) -> None:
        pack = {"timeline": ["legacy"]}
        assert trim_analysis_pack(pack)["timeline"] == ["legacy"]


class TestBusinessFactsPreservedOnRealPack:
    def test_fixture_is_the_real_analysis_pack_shape(self) -> None:
        pack = _load_fixture()
        assert set(pack) >= {
            "case",
            "timeline",
            "snapshot",
            "amlAlerts",
            "relatedCases",
            "agentBrief",
        }
        assert pack["snapshot"]["event"]["provider"] == "internal"
        assert pack["snapshot"]["event"]["amountCents"] == 500000
        assert pack["snapshot"]["event"]["riskSignals"]["walletAgeDays"] == 2
        assert pack["snapshot"]["event"]["riskSignals"]["velocity24h"] == 9
        becauses = [hit["because"] for hit in pack["snapshot"]["hits"]]
        assert "amountCents >= 300000" in becauses

    def test_keeps_provider_amount_wallet_age_velocity_and_hit_because(
        self,
    ) -> None:
        pack = _load_fixture()
        trimmed = trim_analysis_pack(pack)
        event = trimmed["snapshot"]["event"]
        assert event["provider"] == "internal"
        assert event["amountCents"] == 500000
        assert event["riskSignals"] == {"walletAgeDays": 2, "velocity24h": 9}
        assert trimmed["snapshot"]["hits"] == pack["snapshot"]["hits"]
        assert all("because" in hit for hit in trimmed["snapshot"]["hits"])
        assert trimmed["agentBrief"] == pack["agentBrief"]
        assert trimmed["case"]["id"] == pack["case"]["id"]
        assert trimmed["amlAlerts"] == []

    def test_does_not_invent_because_on_a_points_only_hit(self) -> None:
        pack = {"snapshot": {"hits": [{"points": 10}]}}
        trimmed = trim_analysis_pack(pack)
        assert trimmed["snapshot"]["hits"] == [{"points": 10}]
        assert "because" not in trimmed["snapshot"]["hits"][0]

    def test_real_pack_token_drop(self) -> None:
        pack = _load_fixture()
        trimmed = trim_analysis_pack(pack)
        before = _approx_tokens(pack)
        after = _approx_tokens(trimmed)
        drop = before - after
        # The live GET of this shape is ~2.5k tokens at chars/4 (~6k at chars/1.5).
        # relatedCases + duplicated AGENT_BRIEFING/SNAPSHOT_REFRESHED are the fat.
        assert before >= 2000
        assert after < before
        assert drop / before >= 0.30, (
            f"expected >=30% drop, got before={before} after={after} "
            f"drop={drop} ratio={drop / before:.2%}"
        )
        related = trimmed["relatedCases"]
        assert related
        assert all(set(item) <= {"id", "score", "hits"} for item in related)
        assert all("because" in hit for item in related for hit in item["hits"])
        assert [event["eventType"] for event in trimmed["timeline"]] == [
            "CASE_CREATED"
        ]
