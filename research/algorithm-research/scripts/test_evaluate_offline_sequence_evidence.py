#!/usr/bin/env python3
import copy
import json
import sys
import unittest
from collections import Counter
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parent))
import evaluate_offline_sequence_evidence as evidence
import run_offline_batch_rankers as runner
import run_offline_sequence_shadow as shadow


def event(kind, session="s", **data):
    return {"kind": kind, "sessionID": session, "data": data}


def action(bundle):
    meta = {"id": "app:" + bundle, "kind": "app", "bundleID": bundle,
            "admission": {"basis": "standalone_application", "policyVersion": "test"}}
    return runner.action_from_meta(meta, catalog_only=False)


def observation(position, current, truth):
    return runner.Observation(
        sample_id="sample-" + str(position), split="test", position=position,
        input_event_index=10 + position, completed_event_index=20 + position,
        input_timestamp_ms=1_000 + position, outcome_timestamp_ms=2_000 + position,
        outcome_event_id="outcome-" + str(position), current=current, truth=truth,
    )


class OfflineSequenceEvidenceTests(unittest.TestCase):
    def test_pre_boundary_counts_only_independent_arrivals_and_honors_gap(self):
        events = [
            event("session.start"),
            event("context.change"),
            event("context.change"),
            event("capture.gap"),
            event("context.change"),
            event("context.change"),
        ]
        prefixes = shadow.PrefixIndex(
            by_event_index={},
            action_by_event_index={1: "app:a", 2: "app:a", 4: "app:a", 5: "app:b"},
            audit={},
        )
        counts, audit = evidence.independent_arrival_counts_before_boundary(events, prefixes, 5)
        self.assertEqual(counts, Counter({"app:a": 2}))
        self.assertEqual(audit["consecutive_duplicate_not_independent"], 1)
        self.assertEqual(audit["post_boundary_arrival_not_used_for_membership"], 1)

    def test_evidence_cohort_uses_only_current_action_and_preboundary_count(self):
        current_seen = action("seen")
        current_unseen = action("unseen")
        target = action("target")
        rows = [observation(0, current_seen, target), observation(1, current_unseen, target)]
        audited = runner.evaluation_rows(rows)
        groups, audit, membership = evidence.evidence_cohorts(rows, audited, Counter({current_seen.key: 2}))
        self.assertEqual([row["position"] for row in groups["current_unit_seen_before_outer_split"]], [0])
        self.assertEqual([row["position"] for row in groups["current_unit_not_seen_before_outer_split"]], [1])
        self.assertEqual([row["position"] for row in groups["seen_current_unit_arrivals_2_3"]], [0])
        self.assertTrue(audit["membership_uses_current_action_only"])
        self.assertFalse(audit["membership_uses_realized_next_target"])
        self.assertFalse(audit["membership_uses_prediction_or_hit"])
        self.assertEqual(membership[0]["pre_boundary_independent_arrivals"], 2)
        self.assertEqual(membership[0]["current_unit_kind"], "app")

    def test_membership_does_not_change_when_future_label_fields_are_changed(self):
        current_seen = action("seen")
        target = action("target")
        rows = [observation(0, current_seen, target)]
        audited = runner.evaluation_rows(rows)
        changed = copy.deepcopy(audited)
        changed[0]["truth_action"] = "future-only-label"
        changed[0]["prediction"] = ["future-only-prediction"]
        before, _, _ = evidence.evidence_cohorts(rows, audited, Counter({current_seen.key: 1}))
        after, _, _ = evidence.evidence_cohorts(rows, changed, Counter({current_seen.key: 1}))
        self.assertEqual([row["position"] for row in before["current_unit_seen_before_outer_split"]],
                         [row["position"] for row in after["current_unit_seen_before_outer_split"]])

    def test_per_unit_report_keeps_raw_current_action_out_of_artifact(self):
        current_seen = action("seen")
        target = action("target")
        rows = [observation(0, current_seen, target)]
        audited = runner.evaluation_rows(rows)
        groups, _, membership = evidence.evidence_cohorts(rows, audited, Counter({current_seen.key: 1}))
        outputs = {
            "no_context_statistical": {0: {"predictions": [target.key], "rankable": [target.key]}},
            "rnn_residual": {0: {"predictions": [], "rankable": [target.key]}},
        }
        report = evidence.per_unit_private_aggregates(
            groups["current_unit_seen_before_outer_split"], membership, outputs)
        self.assertEqual(len(report["units"]), 1)
        self.assertEqual(report["units"][0]["current_unit_token"], runner.token(current_seen.key))
        self.assertNotIn(current_seen.key, json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    unittest.main()
