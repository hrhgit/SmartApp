#!/usr/bin/env python3
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_offline_batch_rankers as runner


def app(bundle="example.app"):
    return {"id": "app:" + bundle, "kind": "app", "bundleID": bundle,
            "admission": {"basis": "standalone_application", "policyVersion": "target-eligibility-v4"}}


def page(identifier, url):
    return {"id": identifier, "kind": "page", "locator": url,
            "admission": {"basis": "supported_browser_page", "policyVersion": "target-eligibility-v4"}}


def observation(index, completion, current, truth, outcome):
    return runner.Observation(
        sample_id="sample-" + str(index), split="train", position=index,
        input_event_index=index, completed_event_index=completion,
        input_timestamp_ms=index * 10_000, outcome_timestamp_ms=index * 10_000 + 1_000,
        outcome_event_id=outcome, current=current, truth=truth,
    )


class OfflineBatchRankerTests(unittest.TestCase):
    def test_window_normalizes_to_app_without_claiming_window_reopen(self):
        meta = {"id": "window:opaque", "kind": "window", "bundleID": "example.app",
                "admission": {"basis": "standard_work_window", "policyVersion": "target-eligibility-v4"}}
        action = runner.action_from_meta(meta, catalog_only=False)
        self.assertIsNotNone(action)
        self.assertEqual(action.key, "app:example.app")
        self.assertEqual(action.kind, "app")
        self.assertNotIn("locator", action.meta)

    def test_catalog_never_includes_a_future_target_update(self):
        events = [
            {"kind": "target.update", "data": {"target": app("one.app")}},
            {"kind": "context.change", "data": {}},
            {"kind": "target.update", "data": {"target": page("page:later", "https://example.test/later")}},
        ]
        history = runner.TargetHistory(events)
        early = history.catalog_at(1)
        late = history.catalog_at(2)
        self.assertEqual(set(early), {"app:one.app"})
        self.assertEqual(set(late), {"app:one.app", "page:later"})

    def test_parent_fallback_metadata_uses_only_the_same_foreground_event(self):
        current = runner.action_from_meta(page("page:current", "https://example.test/current"), catalog_only=False)
        truth = runner.action_from_meta(page("page:truth", "https://example.test/truth"), catalog_only=False)
        current_parent = runner.foreground_parent_action({
            "kind": "context.change", "app": {"bundleID": "current.browser"},
        })
        truth_parent = runner.foreground_parent_action({
            "kind": "context.change", "app": {"bundleID": "next.browser"},
        })
        self.assertEqual(current_parent.key, "app:current.browser")
        self.assertEqual(truth_parent.key, "app:next.browser")
        self.assertIsNone(runner.foreground_parent_action({"kind": "target.update", "app": {"bundleID": "ignored"}}))
        row = runner.Observation(
            sample_id="sample-parent", split="test", position=9,
            input_event_index=10, completed_event_index=12,
            input_timestamp_ms=10_000, outcome_timestamp_ms=12_000,
            outcome_event_id="outcome-parent", current=current, truth=truth,
            current_parent=current_parent, truth_parent=truth_parent,
            has_prior_exact_action=False, has_prior_parent_action=True,
        )
        scored = runner.evaluation_rows([row])[0]
        self.assertEqual(scored["_scoring"]["current_parent_action"], "app:current.browser")
        self.assertEqual(scored["_scoring"]["truth_parent_action"], "app:next.browser")
        self.assertTrue(scored["_scoring"]["prior_evidence"]["parent_action"])

    def test_history_statistics_deduplicate_one_observed_outcome(self):
        left = runner.action_from_meta(app("left.app"), catalog_only=False)
        right = runner.action_from_meta(app("right.app"), catalog_only=False)
        stats = runner.HistoryStats()
        self.assertTrue(stats.add(observation(1, 2, left, right, "outcome-1")))
        self.assertFalse(stats.add(observation(2, 3, left, right, "outcome-1")))
        self.assertEqual(stats.global_counts[right.key], 1)
        self.assertEqual(stats.conditional_counts[left.key, right.key], 1)

    def test_inner_split_never_uses_mature_labels_from_a_later_segment(self):
        current = runner.action_from_meta(app("left.app"), catalog_only=False)
        truth = runner.action_from_meta(app("right.app"), catalog_only=False)
        rows = [observation(index * 10, index * 10 + 2, current, truth, "outcome-" + str(index))
                for index in range(12)]
        fit, calibration, selection, _ = runner.causal_three_way(rows)
        self.assertLess(max(row.completed_event_index for row in fit),
                        min(row.input_event_index for row in calibration))
        self.assertLess(max(row.completed_event_index for row in calibration),
                        min(row.input_event_index for row in selection))

    def test_conditional_logit_prefers_the_seen_positive_pair(self):
        current = runner.action_from_meta(app("left.app"), catalog_only=False)
        positive = runner.action_from_meta(app("right.app"), catalog_only=False)
        negative = runner.action_from_meta(app("other.app"), catalog_only=False)
        row = observation(10, 12, current, positive, "outcome-1")
        features = []
        for action, signal in ((positive, 1.0), (negative, 0.0)):
            values = {name: "x" for name in runner.CAT_COLUMNS}
            values.update({name: 0.0 for name in runner.NUMERIC_COLUMNS})
            values["candidate_token"] = runner.token(action.key)
            values["origin_candidate_token"] = runner.token(current.key + "\0" + action.key)
            values["conditional_log_count"] = signal
            features.append(values)
        group = runner.QueryGroup(row, [positive, negative], np.array([-1.0, -1.0]), features, 0)
        model = runner.train_linear([group, group], 0.5)
        self.assertGreater(model.correction(group)[0], model.correction(group)[1])


if __name__ == "__main__":
    unittest.main()
