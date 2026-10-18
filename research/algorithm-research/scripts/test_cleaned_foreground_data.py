#!/usr/bin/env python3
"""Synthetic checks for causal cleaning; no private targets are printed."""

from __future__ import annotations

import copy
import sys
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parent))
import cleaned_foreground_data as cleaning
import run_offline_batch_rankers as offline


SCOPE_ID = "E3E423E3-E735-4665-94C5-6C9A696979EF"


class Fixture:
    def __init__(self):
        self.events = []

    def add(self, kind, time, data=None, session="session-a", app=None):
        event = {"id": f"event-{len(self.events)}", "kind": kind,
                 "sessionID": session, "sequence": len(self.events),
                 "monotonicMs": time, "timestampMs": 100_000 + time,
                 "data": data or {}, "source": "lifecycle" if kind == "observation.scope" else "workspace"}
        if app is not None:
            event["app"] = {"bundleID": app}
        self.events.append(event)
        return event

    def target(self, name, time=0, bundle=None):
        bundle = bundle or name
        meta = {"id": "app:" + name, "kind": "app", "bundleID": bundle,
                "admission": {"basis": "standalone_application",
                              "policyVersion": "target-eligibility-v4"}}
        return self.add("target.update", time, {"target": meta})

    def arrival(self, name, time, *, reason="foreground_change", session="session-a",
                status="foreground_only"):
        return self.add("context.change", time, {
            "targetID": "app:" + name, "reason": reason,
            "inputCoverage": {"version": "foreground-coverage-v2", "status": status,
                              "sources": [{"status": "ready"}]},
        }, session=session, app=name)

    def legacy(self, reason, time, session="session-a"):
        return self.add("capture.gap", time, {"reason": reason}, session=session)

    def scope(self, phase, time, *, scope_id=SCOPE_ID, category="excluded_app",
              session="session-a"):
        return self.add("observation.scope", time, {
            "scopeVersion": "excluded-foreground-v1", "scopeID": scope_id,
            "phase": phase, "category": category,
        }, session=session)

    def sample(self, anchor, outcome, *, status="incomplete"):
        return {"id": "sample-" + anchor["id"], "schemaVersion": 4,
                "labelRuleVersion": "observed-foreground-sequence-30s-v4",
                "sessionID": anchor["sessionID"], "anchorEventID": anchor["id"],
                "inputThroughSequence": anchor["sequence"],
                "inputCutoffMs": anchor["monotonicMs"],
                "inputCoverage": {"version": "foreground-coverage-v2",
                                  "status": "foreground_only"},
                "inventoryEventIDs": [], "status": status,
                "outcomeCoverage": "interrupted",
                "hierarchy": {"destinationTargetID": anchor["data"]["targetID"]},
                "outcomes": [{"eventID": outcome["id"],
                              "targetID": outcome["data"]["targetID"],
                              "hierarchy": {"destinationTargetID": outcome["data"]["targetID"]}}]}

    def derive(self, samples=(), boundary=None, train=()):
        history = offline.TargetHistory(self.events)
        return cleaning.derive_cleaning(
            self.events, list(samples), history,
            test_boundary_index=len(self.events) if boundary is None else boundary,
            existing_train_rows=train,
        )


class CleanedForegroundDataTests(unittest.TestCase):
    def test_first_observed_step_survives_later_incomplete_horizon_without_rewriting_raw(self):
        f = Fixture()
        f.target("a"); f.target("b")
        a = f.arrival("a", 10)
        b = f.arrival("b", 110)
        f.legacy("observation_suspended", 200)
        sample = f.sample(a, b)
        original_events, original_sample = copy.deepcopy(f.events), copy.deepcopy(sample)
        result = f.derive([sample])
        self.assertEqual(len(result.salvaged_train), 1)
        self.assertEqual(result.salvaged_train[0].outcome_event_id, b["id"])
        evidence = result.row_evidence[sample["id"]]
        self.assertTrue(evidence.first_step_confirmed)
        self.assertFalse(evidence.full_horizon_observed)
        self.assertFalse(evidence.inferred_next_eligible)
        self.assertEqual(evidence.input_source_statuses, ("ready",))
        self.assertEqual(f.events, original_events)
        self.assertEqual(sample, original_sample)
        self.assertEqual(len(cleaning.private_ledger_records(result)), 1)

    def test_first_step_cannot_cross_unknown_gap_or_become_test_training(self):
        f = Fixture()
        f.target("a"); f.target("b")
        a = f.arrival("a", 10)
        f.legacy("unknown_failure", 20)
        b = f.arrival("b", 50)
        self.assertFalse(f.derive([f.sample(a, b)]).salvaged_train)
        f2 = Fixture(); f2.target("a"); f2.target("b")
        a2 = f2.arrival("a", 10); b2 = f2.arrival("b", 50)
        result = f2.derive([f2.sample(a2, b2)], boundary=a2["sequence"])
        self.assertFalse(result.salvaged_train)
        self.assertEqual(len(result.salvaged_test_diagnostic), 1)
        guard = f2.derive([f2.sample(a2, b2)], boundary=b2["sequence"])
        self.assertFalse(guard.salvaged_train or guard.salvaged_test_diagnostic)

    def test_first_step_rejects_recovery_and_duplicate_outcome(self):
        f = Fixture(); f.target("a"); f.target("b")
        a = f.arrival("a", 10)
        recovery = f.arrival("b", 20, reason="browser_resync")
        self.assertFalse(f.derive([f.sample(a, recovery)]).salvaged_train)
        b = f.arrival("b", 30)
        row = f.derive([f.sample(a, b)]).salvaged_train[0]
        self.assertFalse(f.derive([f.sample(a, b)], train=[row]).salvaged_train)

    def test_strict_short_legacy_gap_adds_only_observed_next_eligible_target(self):
        f = Fixture(); f.target("a"); f.target("b")
        a = f.arrival("a", 10)
        begin = f.legacy("observation_suspended", 110)
        end = f.legacy("observation_resumed", 410)
        b = f.arrival("b", 710)
        result = f.derive()
        self.assertEqual(len(result.bridges), 1)
        self.assertEqual(len(result.stitched_train), 1)
        bridge = result.bridges[0]
        self.assertEqual((bridge.before_event_id, bridge.after_event_id), (a["id"], b["id"]))
        self.assertEqual((bridge.begin_event_id, bridge.end_event_id), (begin["id"], end["id"]))
        self.assertEqual(bridge.duration_ms, 300)
        self.assertEqual(bridge.observed_pre_gap_ms, 100)
        self.assertEqual(bridge.active_elapsed_ms, 400)
        evidence = result.row_evidence[result.stitched_train[0].sample_id]
        self.assertTrue(evidence.inferred_next_eligible)
        self.assertEqual(evidence.current_hierarchy["destinationTargetID"], "app:a")
        self.assertEqual(len(cleaning.private_ledger_records(result)), 2)

    def test_exact_same_target_bridge_merges_arrival_without_label_or_dwell_bridge(self):
        f = Fixture(); f.target("a")
        f.arrival("a", 10)
        f.legacy("observation_suspended", 20)
        f.legacy("observation_resumed", 120)
        f.arrival("a", 200)
        result = f.derive()
        self.assertEqual(len(result.bridges), 1)
        self.assertTrue(result.bridges[0].same_exact_target)
        self.assertFalse(result.stitched_train)
        self.assertEqual(result.bridges[0].duration_ms, 100)

    def test_bridge_does_not_count_an_outcome_already_in_training_twice(self):
        f = Fixture(); f.target("a"); f.target("b")
        f.arrival("a", 10)
        f.legacy("observation_suspended", 20)
        f.legacy("observation_resumed", 120)
        f.arrival("b", 200)
        first = f.derive()
        self.assertEqual(len(first.stitched_train), 1)
        repeated = f.derive(train=first.stitched_train)
        self.assertEqual(len(repeated.bridges), 1)
        self.assertFalse(repeated.stitched_train)
        self.assertEqual(repeated.audit["legacy_short_gap_duplicate_outcome"], 1)

    def test_different_windows_collapsed_to_one_app_are_not_scored_as_a_switch(self):
        f = Fixture()
        for name in ("one", "two"):
            f.add("target.update", 0, {"target": {
                "id": "window:" + name, "kind": "window", "bundleID": "same.app",
                "admission": {"basis": "standard_work_window",
                              "policyVersion": "target-eligibility-v4"},
            }})
        a = f.arrival("one", 10)
        a["data"]["targetID"] = "window:one"
        a["app"]["bundleID"] = "same.app"
        f.legacy("observation_suspended", 20)
        f.legacy("observation_resumed", 120)
        b = f.arrival("two", 200)
        b["data"]["targetID"] = "window:two"
        b["app"]["bundleID"] = "same.app"
        result = f.derive()
        self.assertEqual(len(result.bridges), 1)
        self.assertFalse(result.bridges[0].same_exact_target)
        self.assertTrue(result.bridges[0].same_action)
        self.assertFalse(result.stitched_train)

    def test_legacy_gap_rejects_nested_unpaired_cross_session_and_slow_cases(self):
        for mode in ("nested", "unpaired", "cross_session", "long", "late", "unknown_before"):
            with self.subTest(mode=mode):
                f = Fixture(); f.target("a"); f.target("b")
                f.arrival("a", 10)
                if mode == "unknown_before":
                    f.legacy("unknown_failure", 15)
                f.legacy("observation_suspended", 20)
                if mode == "nested":
                    f.legacy("observation_suspended", 25)
                if mode != "unpaired":
                    f.legacy("observation_resumed", 1_500 if mode == "long" else 100,
                             session="session-b" if mode == "cross_session" else "session-a")
                f.arrival("b", 2_000 if mode == "late" else 150)
                result = f.derive()
                self.assertFalse(result.bridges)
                self.assertFalse(result.stitched_train)

    def test_paired_exclusion_can_span_long_time_but_never_contains_an_excluded_identity(self):
        f = Fixture(); f.target("a"); f.target("b")
        f.arrival("a", 10)
        f.scope("begin", 110)
        f.add("heartbeat", 40_000)
        f.scope("end", 60_110)
        f.arrival("b", 60_210)
        result = f.derive()
        self.assertEqual(len(result.bridges), 1)
        self.assertEqual(len(result.stitched_train), 1)
        bridge = result.bridges[0]
        self.assertEqual(bridge.duration_ms, 60_000)
        self.assertEqual(bridge.active_elapsed_ms, 200)
        self.assertEqual(bridge.category, "excluded_app")
        ledger = cleaning.private_ledger_records(result)
        self.assertNotIn("excludedObjectID", str(ledger))
        self.assertTrue(result.row_evidence[result.stitched_train[0].sample_id].inferred_next_eligible)

    def test_exclusion_rejects_mismatch_nested_unknown_gap_or_an_unobserved_endpoint(self):
        for mode in ("mismatched", "nested", "unknown_gap", "unobserved_after",
                     "identity_in_scope"):
            with self.subTest(mode=mode):
                f = Fixture(); f.target("a"); f.target("b")
                f.arrival("a", 10)
                begin = f.scope("begin", 20)
                if mode == "identity_in_scope":
                    begin["data"]["excludedBundleID"] = "private.example"
                if mode == "nested": f.scope("begin", 30)
                if mode == "unknown_gap": f.legacy("unknown_failure", 30)
                f.scope("end", 50, scope_id=("3B5849AB-999B-49F5-871D-83BB91D21F87"
                                              if mode == "mismatched" else SCOPE_ID))
                f.arrival("b", 60, status="failed" if mode == "unobserved_after" else "foreground_only")
                result = f.derive()
                self.assertFalse(result.bridges)
                self.assertFalse(result.stitched_train)

    def test_exclusion_active_horizon_is_enforced_after_subtracting_masked_duration(self):
        f = Fixture(); f.target("a"); f.target("b")
        f.arrival("a", 10)
        f.scope("begin", 31_100)
        f.scope("end", 100_000)
        f.arrival("b", 100_100)
        result = f.derive()
        self.assertEqual(len(result.bridges), 1)
        self.assertFalse(result.stitched_train)
        self.assertGreater(result.bridges[0].active_elapsed_ms, 30_000)


if __name__ == "__main__":
    unittest.main()
