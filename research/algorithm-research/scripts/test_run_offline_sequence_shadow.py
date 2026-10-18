#!/usr/bin/env python3
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np


sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_offline_batch_rankers as runner
import run_offline_sequence_shadow as sequence
import sequence_asof_features as asof


def app(bundle):
    return {
        "id": "app:" + bundle,
        "kind": "app",
        "bundleID": bundle,
        "admission": {"basis": "standalone_application", "policyVersion": "target-eligibility-v4"},
    }


def covered_change(target_id, reason="foreground_change", session="session-a"):
    return {
        "kind": "context.change",
        "source": "foreground:test",
        "sessionID": session,
        "data": {
            "targetID": target_id,
            "reason": reason,
            "inputCoverage": {"version": "foreground-coverage-v2", "status": "foreground_only",
                              "sources": [{"source": "foreground:test", "status": "ready"}]},
        },
    }


def timed(event, number):
    return {**event, "id": "event-" + str(number), "timestampMs": number * 1000,
            "monotonicMs": number * 1000}


def observation(position, event_index, current):
    return runner.Observation(
        sample_id="sample-" + str(position), split="train", position=position,
        input_event_index=event_index, completed_event_index=event_index + 1,
        input_timestamp_ms=event_index * 1_000, outcome_timestamp_ms=event_index * 1_000 + 1,
        outcome_event_id="outcome-" + str(position), current=current, truth=current,
    )


def score(top1, top2, top4, exact_top1=None, exact_top2=None, exact_top4=None):
    exact_top1 = top1 if exact_top1 is None else exact_top1
    exact_top2 = top2 if exact_top2 is None else exact_top2
    exact_top4 = top4 if exact_top4 is None else exact_top4
    return {
        "top1_hits": top1, "top2_hits": top2, "top4_hits": top4,
        "rank_score": {"mean": 0.5},
        "exact": {"top1_hits": exact_top1, "top2_hits": exact_top2, "top4_hits": exact_top4},
    }


class OfflineSequenceShadowTests(unittest.TestCase):
    def test_path_labels_form_one_normalized_training_result(self):
        try:
            torch, _, functional = sequence.require_torch()
        except RuntimeError:
            self.skipTest("private torch runtime unavailable")
        logits = torch.zeros(3, dtype=torch.float32)
        original = sequence.SequenceExample((1, 2), (3, 4, 5), np.zeros(3, np.float32), 0)
        path = sequence.SequenceExample((1, 2), (3, 4, 5), np.zeros(3, np.float32), 0,
                                        positive_indices=(0, 1))
        parent_only = sequence.SequenceExample((1, 2), (3, 4, 5), np.zeros(3, np.float32), None,
                                               positive_indices=(1,))
        self.assertAlmostEqual(float(sequence.sequence_training_loss(logits, original, functional)),
                               np.log(3), places=6)
        # Two useful entries compete for one probability mass.  They do not
        # become two separate training visits or receive an invented reward.
        self.assertAlmostEqual(float(sequence.sequence_training_loss(logits, path, functional)),
                               np.log(3 / 2), places=6)
        self.assertAlmostEqual(float(sequence.sequence_training_loss(logits, parent_only, functional)),
                               np.log(3), places=6)

    def test_path_training_accepts_known_parent_when_exact_is_absent(self):
        try:
            sequence.require_torch()
        except RuntimeError:
            self.skipTest("private torch runtime unavailable")
        config = {**sequence.model_configurations(23, 2, 64)["rnn"][0], "epochs": 1}
        example = sequence.SequenceExample(
            sequence_ids=(1, 2), candidate_ids=(3, 4, 5),
            prior=np.zeros(3, np.float32), truth_index=None,
            positive_indices=(1,),
        )
        model, audit = sequence.train_sequence_model("rnn", config, [example])
        self.assertIsNotNone(model)
        self.assertEqual(audit["examples"], 1)
        self.assertEqual(audit["path_supervised_examples"], 1)
        self.assertEqual(audit["path_examples_without_exact_candidate"], 1)
        self.assertEqual(audit["loss_objective"], "set_likelihood")
        self.assertNotIn("initial_mean_cross_entropy", audit)

    def test_path_training_rejects_duplicate_missing_and_misaligned_labels(self):
        try:
            torch, _, functional = sequence.require_torch()
        except RuntimeError:
            self.skipTest("private torch runtime unavailable")
        logits = torch.zeros(3, dtype=torch.float32)
        def example(truth, positives):
            return sequence.SequenceExample((1, 2), (3, 4, 5), np.zeros(3, np.float32),
                                            truth, positive_indices=positives)
        for bad in (example(None, ()), example(None, (1, 1)), example(None, (3,)),
                    example(0, (1,)), example(None, (True,)), example(None, None)):
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    sequence.sequence_training_loss(logits, bad, functional)

    def test_shared_cutoff_context_is_consumed_only_when_enabled(self):
        try:
            torch, _, _ = sequence.require_torch()
        except RuntimeError:
            self.skipTest("private torch runtime unavailable")
        config = {**sequence.model_configurations(23, 4, 64)["rnn"][0],
                  "use_common_context": True, "common_context_width": 3}
        model = sequence.build_sequence_model("rnn", config)
        sequence_ids = torch.tensor([[1, 2]], dtype=torch.long)
        candidate_ids = torch.tensor([3, 4], dtype=torch.long)
        with self.assertRaisesRegex(ValueError, "Common context"):
            model.raw_scores(sequence_ids, candidate_ids)
        zeros = model.raw_scores(sequence_ids, candidate_ids,
                                 context_numeric=torch.zeros((1, 3)))
        changed = model.raw_scores(sequence_ids, candidate_ids,
                                   context_numeric=torch.tensor([[0.0, 0.0, 10.0]]))
        self.assertFalse(torch.allclose(zeros, changed))

    def test_rich_bridge_is_visible_only_after_resume_and_does_not_invent_dwell(self):
        events = [
            timed({"kind": "target.update", "sessionID": "session-a",
                   "data": {"target": app("one.app")}}, 0),
            timed(covered_change("app:one.app"), 1),
            timed({"kind": "capture.gap", "sessionID": "session-a",
                   "data": {"reason": "observation_suspended"}}, 2),
            timed({"kind": "capture.gap", "sessionID": "session-a",
                   "data": {"reason": "observation_resumed"}}, 3),
            timed({"kind": "target.update", "sessionID": "session-a",
                   "data": {"target": app("two.app")}}, 4),
            timed(covered_change("app:two.app"), 5),
        ]
        bridge = SimpleNamespace(before_index=1, begin_index=2, end_index=3,
                                 after_index=5, kind="legacy_short_gap",
                                 duration_ms=800, same_exact_target=False)
        index = asof.build_rich_index(events, runner.TargetHistory(events), 8, [bridge])
        before = index.by_event_index[1]
        after = index.by_event_index[5]
        self.assertEqual(before.keys, ("app:one.app",))
        self.assertEqual(after.keys, ("app:one.app", "observation:legacy_short_gap", "app:two.app"))
        before_numeric = asof.sequence_numeric(before, events)
        after_numeric = asof.sequence_numeric(after, events)
        self.assertEqual(before_numeric[0, 15], 0.0)
        self.assertEqual(before_numeric[0, 7], 0.0)
        self.assertEqual(after_numeric[1, 15], 1.0)
        self.assertEqual(after_numeric[0, 7], 0.0)
        self.assertEqual(len(index.records), 2)

    def test_same_exact_bridge_does_not_add_arrival_or_frequency(self):
        events = [
            timed({"kind": "target.update", "sessionID": "session-a",
                   "data": {"target": app("one.app")}}, 0),
            timed(covered_change("app:one.app"), 1),
            timed({"kind": "capture.gap", "sessionID": "session-a", "data": {}}, 2),
            timed({"kind": "capture.gap", "sessionID": "session-a", "data": {}}, 3),
            timed(covered_change("app:one.app", reason="resolution_update"), 4),
        ]
        bridge = SimpleNamespace(before_index=1, begin_index=2, end_index=3,
                                 after_index=4, kind="legacy_short_gap",
                                 duration_ms=700, same_exact_target=True,
                                 active_elapsed_ms=2300, observed_pre_gap_ms=1000)
        index = asof.build_rich_index(events, runner.TargetHistory(events), 8, [bridge])
        self.assertEqual(index.by_event_index[4].keys,
                         ("app:one.app", "observation:legacy_short_gap"))
        self.assertEqual(index.by_event_index[4].arrival_count, 1)
        self.assertEqual(len(index.records), 1)
        numeric = asof.sequence_numeric(index.by_event_index[4], events)
        self.assertEqual(numeric[0, 17], 1.0)
        self.assertAlmostEqual(numeric[0, 16], np.log1p(2.3), places=5)
        path, direct = asof.path_frequency(index, 5, 5000)
        self.assertEqual(path["app:one.app"][3], 1)
        self.assertEqual(direct["app:one.app"][3], 1)

    def test_idle_before_same_target_bridge_masks_dwell_but_keeps_relation(self):
        events = [
            timed({"kind": "target.update", "sessionID": "session-a",
                   "data": {"target": app("one.app")}}, 0),
            timed(covered_change("app:one.app"), 1),
            timed({"kind": "idle.change", "sessionID": "session-a", "data": {}}, 2),
            timed({"kind": "capture.gap", "sessionID": "session-a", "data": {}}, 3),
            timed({"kind": "capture.gap", "sessionID": "session-a", "data": {}}, 4),
            timed(covered_change("app:one.app"), 5),
        ]
        bridge = SimpleNamespace(before_index=1, begin_index=3, end_index=4, after_index=5,
                                 kind="legacy_short_gap", duration_ms=700,
                                 same_exact_target=True, active_elapsed_ms=3300,
                                 observed_pre_gap_ms=2000)
        index = asof.build_rich_index(events, runner.TargetHistory(events), 8, [bridge])
        prefix = index.by_event_index[5]
        self.assertEqual(prefix.arrival_count, 1)
        numeric = asof.sequence_numeric(prefix, events)
        self.assertEqual(numeric[0, 7], 0.0)
        self.assertEqual(numeric[0, 17], 0.0)
        self.assertEqual(numeric[1, 15], 1.0)
        later = timed({"kind": "target.update", "sessionID": "session-a",
                       "data": {"target": app("two.app")}}, 6)
        after = timed(covered_change("app:two.app"), 7)
        extended = events + [later, after]
        later_index = asof.build_rich_index(extended, runner.TargetHistory(extended), 8, [bridge])
        later_numeric = asof.sequence_numeric(later_index.by_event_index[7], extended)
        self.assertEqual(later_numeric[0, 17], 0.0)
        self.assertAlmostEqual(later_numeric[0, 6], np.log1p(2.0), places=5)

    def test_unrelated_source_failure_does_not_erase_foreground_dwell(self):
        events = [
            timed({"kind": "target.update", "sessionID": "session-a",
                   "data": {"target": app("one.app")}}, 0),
            timed(covered_change("app:one.app"), 1),
            timed({"kind": "source.status", "source": "unrelated:source",
                   "sessionID": "session-a", "data": {"observationStatus": "unavailable"}}, 2),
            timed({"kind": "target.update", "sessionID": "session-a",
                   "data": {"target": app("two.app")}}, 3),
            timed(covered_change("app:two.app"), 4),
        ]
        index = asof.build_rich_index(events, runner.TargetHistory(events), 8)
        numeric = asof.sequence_numeric(index.by_event_index[4], events)
        self.assertEqual(numeric[0, 7], 1.0)
        self.assertAlmostEqual(numeric[0, 6], np.log1p(3.0), places=5)

    def test_failed_stale_and_unknown_source_quality_are_distinct(self):
        event = timed(covered_change("app:one.app"), 1)
        event["data"]["inputCoverage"]["sources"] = [
            {"status": "ready"}, {"status": "failed"}, {"status": "stale"},
            {"status": "unrecognized"},
        ]
        quality = asof._quality(event)
        self.assertEqual(quality[0], .25)
        self.assertEqual(quality[4], .25)
        self.assertEqual(quality[5], .25)
        self.assertEqual(quality[6], .25)
        source_id, source_status = asof._relevant_source(event)
        step = asof.Step("app:one.app", 1, 1000, 1000, (), quality,
                         source_id=source_id, source_status=source_status)
        self.assertFalse(asof._quality_usable_for_dwell(step))

    def test_unknown_gap_stays_separate_even_if_short(self):
        events = [
            timed({"kind": "target.update", "sessionID": "session-a",
                   "data": {"target": app("one.app")}}, 0),
            timed(covered_change("app:one.app"), 1),
            timed({"kind": "capture.gap", "sessionID": "session-a", "data": {}}, 2),
            timed({"kind": "target.update", "sessionID": "session-a",
                   "data": {"target": app("two.app")}}, 3),
            timed(covered_change("app:two.app"), 4),
        ]
        index = asof.build_rich_index(events, runner.TargetHistory(events), 8)
        self.assertEqual(index.by_event_index[4].keys, ("app:two.app",))

    def test_rich_model_consumes_time_layers_and_frequency(self):
        try:
            sequence.require_torch()
        except RuntimeError:
            self.skipTest("private torch runtime unavailable")
        config = {**sequence.model_configurations(23, 8, 64)["rnn"][0],
                  "input_mode": "full", "epochs": 1}
        model = sequence.build_sequence_model("rnn", config)
        torch, _, _ = sequence.require_torch()
        scores = model.raw_scores(
            torch.tensor([[1, 2]], dtype=torch.long),
            torch.tensor([3, 4], dtype=torch.long),
            sequence_numeric=torch.from_numpy(np.zeros((1, 2, asof.NUMERIC_WIDTH), np.float32)),
            sequence_layers=torch.zeros((1, 2, asof.MAX_LAYERS), dtype=torch.long),
            candidate_numeric=torch.zeros((2, asof.CANDIDATE_WIDTH), dtype=torch.float32),
            candidate_layers=torch.zeros((2, asof.MAX_LAYERS), dtype=torch.long),
        )
        self.assertEqual(tuple(scores.shape), (2,))
        with self.assertRaises(ValueError):
            model.raw_scores(torch.tensor([[1, 2]]), torch.tensor([3, 4]))

    def test_candidate_frequency_and_cutoff_features_ignore_future_arrival(self):
        events = [
            timed({"kind": "target.update", "sessionID": "session-a",
                   "data": {"target": app("one.app")}}, 0),
            timed(covered_change("app:one.app"), 1),
            timed({"kind": "target.update", "sessionID": "session-a",
                   "data": {"target": app("two.app")}}, 2),
            timed(covered_change("app:two.app"), 3),
            timed(covered_change("app:one.app"), 4),
        ]
        history = runner.TargetHistory(events)
        index = asof.build_rich_index(events, history, 8)
        one = runner.action_from_meta(app("one.app"), catalog_only=False)
        two = runner.action_from_meta(app("two.app"), catalog_only=False)
        row = observation(9, 3, two)
        row = runner.Observation(**{**row.__dict__, "input_timestamp_ms": 3000})
        group = runner.QueryGroup(row, [one, two], np.array([0.0, 0.0]), [], 0)
        rows = asof.candidate_feature_rows(group, index, history, events, index.by_event_index[3])
        self.assertEqual(len(rows), 2)
        self.assertEqual(set(rows[0]), set(rows[1]))
        self.assertAlmostEqual(rows[0]["direct_count_all_log"], np.log1p(1))
        self.assertEqual(rows[1]["direct_count_all_log"], 0.0)
        self.assertIn("quality_confirmed", rows[0])
        with self.assertRaises(ValueError):
            asof.candidate_feature_rows(group, index, history, events, index.by_event_index[4])

    def test_path_scoring_catalog_uses_only_actions_known_at_cutoff(self):
        events = [
            timed({"kind": "target.update", "sessionID": "session-a",
                   "data": {"target": app("one.app")}}, 0),
            timed(covered_change("app:one.app"), 1),
            timed({"kind": "target.update", "sessionID": "session-a",
                   "data": {"target": app("future.app")}}, 2),
        ]
        history = runner.TargetHistory(events)
        current = runner.action_from_meta(app("one.app"), catalog_only=False)
        future = runner.action_from_meta(app("future.app"), catalog_only=False)
        row = runner.Observation(
            sample_id="x", split="test", position=0, input_event_index=1,
            completed_event_index=2, input_timestamp_ms=1000, outcome_timestamp_ms=2000,
            outcome_event_id="event-2", current=current, truth=future,
        )
        prepared = sequence.evaluation_rows_with_known_actions([row], history)
        hierarchy = prepared[0]["_scoring"]["path_hierarchy"]
        self.assertIn("app:one.app", hierarchy.executable)
        self.assertNotIn("app:future.app", hierarchy.executable)
        group = runner.QueryGroup(row, [current], np.array([0.0]), [], None)
        scores, outputs, audited = sequence.score_with_known_actions(
            [row], [group], {"baseline": runner.baseline_correction}, history)
        self.assertEqual(len(audited), 1)
        self.assertIn("success", scores["baseline"])
        self.assertEqual(outputs["baseline"][0]["predictions"], ["app:one.app"])

    def test_confirmed_exclusion_masks_prefix_without_inventing_destination(self):
        scope = "E3E423E3-E735-4665-94C5-6C9A696979EF"
        events = [
            {"kind": "target.update", "sessionID": "session-a", "data": {"target": app("one.app")}},
            covered_change("app:one.app"),
            {"kind": "observation.scope", "sessionID": "session-a", "data": {
                "scopeVersion": "excluded-foreground-v1", "scopeID": scope,
                "phase": "begin", "category": "excluded_app"}},
            {"kind": "observation.scope", "sessionID": "session-a", "data": {
                "scopeVersion": "excluded-foreground-v1", "scopeID": scope,
                "phase": "end", "category": "excluded_app"}},
            {"kind": "target.update", "sessionID": "session-a", "data": {"target": app("two.app")}},
            covered_change("app:two.app"),
        ]
        prefixes = sequence.build_prefix_index(events, runner.TargetHistory(events), sequence_length=4)
        self.assertEqual(prefixes.by_event_index[5],
                         ("app:one.app", "observation:excluded", "app:two.app"))
        self.assertEqual(prefixes.audit["masked_excluded_foreground"], 1)

    def test_capture_gap_and_session_change_cut_the_prefix(self):
        events = [
            {"kind": "target.update", "sessionID": "session-a", "data": {"target": app("one.app")}},
            {"kind": "session.start", "sessionID": "session-a", "data": {}},
            covered_change("app:one.app"),
            {"kind": "target.update", "sessionID": "session-a", "data": {"target": app("two.app")}},
            covered_change("app:two.app", reason="app_activation"),
            {"kind": "capture.gap", "sessionID": "session-a", "data": {"reason": "unknown"}},
            {"kind": "target.update", "sessionID": "session-a", "data": {"target": app("three.app")}},
            covered_change("app:three.app"),
            {"kind": "target.update", "sessionID": "session-b", "data": {"target": app("four.app")}},
            covered_change("app:four.app", session="session-b"),
        ]
        prefixes = sequence.build_prefix_index(events, runner.TargetHistory(events), sequence_length=4)
        self.assertEqual(prefixes.by_event_index[4], ("app:one.app", "app:two.app"))
        self.assertEqual(prefixes.by_event_index[7], ("app:three.app",))
        self.assertEqual(prefixes.by_event_index[9], ("app:four.app",))
        self.assertGreaterEqual(prefixes.audit["hard_boundary_capture_gap"], 1)
        self.assertGreaterEqual(prefixes.audit["hard_boundary_session_change"], 1)

    def test_recovery_and_resolution_contexts_cannot_be_sequence_tokens(self):
        events = [
            {"kind": "target.update", "sessionID": "session-a", "data": {"target": app("one.app")}},
            {"kind": "session.start", "sessionID": "session-a", "data": {}},
            covered_change("app:one.app"),
            covered_change("app:one.app", reason="resolution_update"),
            covered_change("app:one.app", reason="browser_resync"),
        ]
        prefixes = sequence.build_prefix_index(events, runner.TargetHistory(events), sequence_length=4)
        self.assertEqual(set(prefixes.by_event_index), {2})
        self.assertEqual(prefixes.audit["context_non_arrival_resolution_update"], 1)
        self.assertEqual(prefixes.audit["context_non_arrival_browser_resync"], 1)

    def test_attachment_never_synthesizes_a_missing_or_mismatched_current_action(self):
        events = [
            {"kind": "target.update", "sessionID": "session-a", "data": {"target": app("one.app")}},
            {"kind": "session.start", "sessionID": "session-a", "data": {}},
            covered_change("app:one.app"),
        ]
        prefixes = sequence.build_prefix_index(events, runner.TargetHistory(events), sequence_length=4)
        one = runner.action_from_meta(app("one.app"), catalog_only=False)
        two = runner.action_from_meta(app("two.app"), catalog_only=False)
        attachments = sequence.attach_sequences([
            observation(1, 2, one), observation(2, 2, two), observation(3, 99, one),
        ], prefixes)
        self.assertEqual(attachments.anchored, {1: ("app:one.app",)})
        self.assertEqual(attachments.eligible, {})
        self.assertEqual(attachments.audit["anchor_action_mismatch"], 1)
        self.assertEqual(attachments.audit["anchor_without_confirmed_prefix"], 1)

    def test_hash_bucket_has_no_zero_and_is_repeatable(self):
        first = sequence.action_bucket("app:one.app", 17)
        self.assertEqual(first, sequence.action_bucket("app:one.app", 17))
        self.assertGreaterEqual(first, 1)
        self.assertLess(first, 17)

    def test_selection_key_keeps_top2_before_rank_score_when_top4_ties(self):
        lower_top2 = score(top1=6, top2=8, top4=12)
        higher_top2 = score(top1=5, top2=9, top4=12)
        self.assertLess(sequence.metric_key(higher_top2, "b"), sequence.metric_key(lower_top2, "a"))


if __name__ == "__main__":
    unittest.main()
