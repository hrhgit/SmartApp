"""Fixed-membership shared-input replay guards, without private user data."""

from __future__ import annotations

from dataclasses import replace
import unittest

import run_offline_batch_rankers as offline
import sequence_asof_features as asof
import run_shared_model_input_study as study
import shared_model_input as shared
from test_shared_model_input import app, event, fixture, observation, observed, page, update


class SharedStudyTests(unittest.TestCase):
    def test_training_and_prediction_are_separate_and_mature_before_boundary(self) -> None:
        _, _, _, row, builder = fixture()
        original_cutoff = builder.at(row)
        batch = study.build_batch([row], [replace(row, position=11, sample_id="test-sample")],
                                  builder, boundary_index=6, weights={row.sample_id: 1.0})
        self.assertEqual(len(batch.train), 1)
        self.assertEqual(len(batch.test), 1)
        self.assertEqual(len(batch.test_feedback), 1)
        self.assertFalse(hasattr(batch.test[0], "outcome"))
        self.assertEqual(batch.train[0].outcome.completed_event_index, 5)
        self.assertEqual(batch.test[0].cutoff.action_ids,
                         batch.test_feedback[0].cutoff.action_ids)
        self.assertEqual(batch.audit["train_ancestor_only"], 1)
        self.assertEqual(batch.audit["test_ancestor_only"], 1)
        self.assertEqual(batch.audit["train_exact_action_available"], 0)
        self.assertEqual(batch.audit["test_exact_action_available"], 0)
        self.assertGreater(batch.audit["first_test_derived_parents"], 0)
        self.assertEqual(batch.audit["first_test_recorded_actions"], 1)
        self.assertEqual(batch.test[0].cutoff.feature_rows(), original_cutoff.feature_rows())
        self.assertEqual(batch.test[0].cutoff.sequence_keys, original_cutoff.sequence_keys)
        self.assertTrue((batch.test[0].cutoff.sequence_numeric ==
                         original_cutoff.sequence_numeric).all())
        self.assertFalse(batch.test[0].cutoff.hierarchy.metadata)
        with self.assertRaises(TypeError):
            batch.test_feedback[0].cutoff.context["time_hour_sin"] = 2.0
        with self.assertRaises(TypeError):
            batch.test_feedback[0].cutoff.candidates[0].metadata["kind"] = "changed"
        with self.assertRaises(ValueError):
            batch.test_feedback[0].cutoff.sequence_numeric[0, 0] = 2.0
        with self.assertRaisesRegex(ValueError, "crossed"):
            study.build_batch([row], [], builder, boundary_index=5,
                              weights={row.sample_id: 1.0})

    def test_model_output_must_use_exact_common_candidates_and_positions(self) -> None:
        _, _, _, row, builder = fixture()
        batch = study.build_batch([], [row], builder, boundary_index=6, weights={})
        case = batch.test[0]
        known = case.cutoff.action_ids[0]
        accepted = study.checked_outputs(batch.test, {
            case.position: {"predictions": list(case.cutoff.action_ids[:4]),
                            "rankable": list(case.cutoff.action_ids)}})
        self.assertEqual(accepted[case.position]["predictions"],
                         list(case.cutoff.action_ids[:4]))
        with self.assertRaisesRegex(ValueError, "omitted"):
            study.checked_outputs(batch.test, {})
        with self.assertRaisesRegex(ValueError, "candidate"):
            study.checked_outputs(batch.test, {case.position: {
                "predictions": ["future_action"], "rankable": ["future_action"]}})
        with self.assertRaisesRegex(ValueError, "candidate"):
            study.checked_outputs(batch.test, {case.position: {
                "predictions": [known, known], "rankable": [known]}})
        with self.assertRaisesRegex(ValueError, "candidate"):
            study.checked_outputs(batch.test, {case.position: {
                "predictions": [known], "rankable": [known]}})

    def test_new_exact_leaf_is_only_counted_after_feedback(self) -> None:
        current = app("other")
        future = page("page:future", "https://new.test/topic/leaf")
        events = [update(current, 0), observed(current["id"], 1),
                  update(future, 2), observed(future["id"], 3)]
        history = offline.TargetHistory(events)
        index = asof.build_rich_index(events, history, 8)
        row = observation(events, current, future, 1, 3)
        builder = shared.SharedInputBuilder(events, history, index)
        batch = study.build_batch([row], [replace(row, position=12, sample_id="test-new")],
                                  builder, boundary_index=4, weights={})
        self.assertEqual(batch.audit["train_no_useful_action"], 1)
        self.assertEqual(batch.audit["test_no_useful_action"], 1)
        self.assertEqual(batch.audit["train_no_useful_then_confirmed_exact"], 1)
        self.assertEqual(batch.audit["test_no_useful_then_confirmed_exact"], 1)
        self.assertEqual(batch.audit["first_test_recorded_actions"], 0)
        self.assertEqual(batch.audit["first_test_derived_parents"], 0)
        self.assertEqual(batch.test[0].cutoff.action_ids, ())
        self.assertEqual(batch.test_feedback[0].outcome.confirmed_destination.action_id,
                         future["id"])

    def test_same_place_with_new_raw_id_is_excluded_from_learning(self) -> None:
        current = page("page:current", "https://example.test/topic/")
        same_place = page("page:new-id", "https://example.test/topic/")
        events = [update(current, 0), observed(current["id"], 1),
                  update(same_place, 2), observed(same_place["id"], 3)]
        history = offline.TargetHistory(events)
        index = asof.build_rich_index(events, history, 8)
        row = observation(events, current, same_place, 1, 3)
        batch = study.build_batch([row], [replace(row, position=12, sample_id="test-same")],
                                  shared.SharedInputBuilder(events, history, index),
                                  boundary_index=4, weights={})
        self.assertEqual(batch.audit["train_matured_navigation"], 1)
        self.assertEqual(batch.audit["train_excluded_same_semantic_not_switch"], 1)
        self.assertEqual(batch.audit["test_feedback_excluded_same_semantic_not_switch"], 1)
        self.assertEqual(len(batch.train), 0)
        self.assertEqual(len(batch.test), 1)
        self.assertEqual(len(batch.test_feedback), 0)

    def test_reused_id_with_new_place_is_excluded_from_learning(self) -> None:
        current = app("other")
        old = page("page:reused", "https://example.test/topic/old")
        changed = page("page:reused", "https://example.test/topic/new")
        events = [update(current, 0), update(old, 1), observed(old["id"], 2),
                  observed(current["id"], 3), update(changed, 4),
                  observed(changed["id"], 5)]
        history = offline.TargetHistory(events)
        index = asof.build_rich_index(events, history, 8)
        row = observation(events, current, changed, 3, 5)
        batch = study.build_batch([row], [replace(row, position=12, sample_id="test-reused")],
                                  shared.SharedInputBuilder(events, history, index),
                                  boundary_index=6, weights={})
        self.assertEqual(batch.audit["train_excluded_reused_id_action_conflict"], 1)
        self.assertEqual(batch.audit["test_feedback_excluded_reused_id_action_conflict"], 1)
        self.assertEqual(len(batch.train), 0)
        self.assertEqual(len(batch.test), 1)
        self.assertEqual(len(batch.test_feedback), 0)


if __name__ == "__main__":
    unittest.main()
