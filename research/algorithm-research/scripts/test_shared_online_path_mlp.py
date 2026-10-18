"""The online neural path update treats one destination as one result."""

from __future__ import annotations

import unittest
from types import SimpleNamespace

import numpy as np

import run_offline_sequence_shadow as sequence
import shared_model_adapters as adapters
import shared_online_path_mlp as online
import run_offline_batch_rankers as offline
from test_shared_model_input import fixture


def example():
    _, _, _, row, builder = fixture()
    cutoff = builder.at(row)
    outcome = builder.matured(cutoff, row, learning_event_index=6)
    prediction = adapters.adapt(cutoff, family="candidate_mlp")
    labels = adapters.attach_matured(prediction, outcome, learning_event_index=6).labels
    return prediction, labels


class SharedOnlinePathMLPTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            sequence.require_torch()
        except RuntimeError:
            raise unittest.SkipTest("private torch runtime unavailable")

    def test_multiple_useful_path_actions_make_one_normalized_update(self):
        prediction, labels = example()
        self.assertEqual(len(labels.useful_indices), 2)
        self.assertIsNone(labels.exact_index)
        model = online.SharedOnlineSetMLP(hidden=8, lr=0.005, negatives=8, seed=7)
        features = model._candidate_matrix(prediction)
        self.assertFalse(np.array_equal(features[0].numpy(), features[1].numpy()))
        before = model.model(features).squeeze(-1)
        expected = (before.logsumexp(dim=0)
                    - before[list(labels.useful_indices)].logsumexp(dim=0))
        result = model.learn(prediction, labels.useful_indices)
        self.assertTrue(result["updated"])
        self.assertAlmostEqual(result["loss"], float(expected), places=6)
        self.assertEqual(result["useful_action_count"], 2)
        self.assertEqual(result["sampled_negative_count"], 1)
        self.assertEqual(result["sampled_action_count"], len(prediction.action_ids))
        audit = model.audit()
        self.assertEqual(audit["optimizer_updates"], 1)
        self.assertEqual(audit["confirmed_arrivals_trained"], 1)
        self.assertEqual(audit["useful_action_labels_seen"], 2)
        scores = model.scores(prediction)
        self.assertEqual(tuple(scores), prediction.action_ids)
        self.assertTrue(all(np.isfinite(value) for value in scores.values()))
        self.assertEqual(model.audit()["prediction_actions_scored"], len(prediction.action_ids))

    def test_seeded_sampling_is_repeatable_and_limited_to_cutoff_candidates(self):
        prediction, labels = example()
        first = online.SharedOnlineSetMLP(hidden=8, lr=0.005, negatives=1, seed=11)
        second = online.SharedOnlineSetMLP(hidden=8, lr=0.005, negatives=1, seed=11)
        one = first.learn(prediction, labels.useful_indices)
        two = second.learn(prediction, labels.useful_indices)
        self.assertEqual(one["candidate_order_sha256"], two["candidate_order_sha256"])
        self.assertEqual(one["sampled_action_count"], two["sampled_action_count"])
        self.assertAlmostEqual(one["loss"], two["loss"], places=6)
        self.assertEqual(first.scores(prediction), second.scores(prediction))
        self.assertLessEqual(one["sampled_action_count"], len(prediction.action_ids))

    def test_feedback_only_leaf_can_train_but_cannot_be_predicted_or_sampled_negative(self):
        prediction, labels = example()
        original_n = len(prediction.action_ids)
        leaf = "page:confirmed-after-feedback"
        template = next(iter(prediction.payload["candidate_features"].values()))
        leaf_row = {**template, "action_id": leaf,
                    "candidate_token": offline.token(leaf)}
        payload = {**prediction.payload,
                   "labels": prediction.action_ids + (leaf,),
                   "candidate_features": {**prediction.payload["candidate_features"],
                                          leaf: leaf_row}}
        learning_view = SimpleNamespace(
            manifest=prediction.manifest, action_ids=prediction.action_ids + (leaf,),
            payload=payload, after_feedback_only=True,
            original_candidate_count=original_n,
        )
        model = online.SharedOnlineSetMLP(hidden=8, lr=0.005, negatives=8, seed=13)
        with self.assertRaisesRegex(ValueError, "cannot enter prediction"):
            model.scores(learning_view)
        useful = labels.useful_indices + (original_n,)
        update = model.learn(learning_view, useful)
        self.assertTrue(update["updated"])
        self.assertTrue(update["after_feedback_only"])
        self.assertEqual(update["original_candidate_count"], original_n)
        self.assertEqual(update["useful_action_count"], len(useful))
        self.assertTrue(all(index < original_n for index in update["sampled_negative_indices"]))
        self.assertEqual(model.audit()["after_feedback_exact_updates"], 1)
        with self.assertRaisesRegex(ValueError, "Appended exact leaf"):
            model.learn(learning_view, labels.useful_indices)

        no_append = SimpleNamespace(
            manifest=prediction.manifest, action_ids=prediction.action_ids,
            payload=prediction.payload, after_feedback_only=True,
            original_candidate_count=original_n,
        )
        second = model.learn(no_append, labels.useful_indices)
        self.assertTrue(second["updated"])
        self.assertEqual(second["original_candidate_count"], original_n)

        first_ever = SimpleNamespace(
            manifest=prediction.manifest, action_ids=(leaf,),
            payload={**payload, "labels": (leaf,),
                     "candidate_features": {leaf: leaf_row}},
            after_feedback_only=True, original_candidate_count=0,
        )
        no_contrast = model.learn(first_ever, (0,))
        self.assertFalse(no_contrast["updated"])
        self.assertEqual(no_contrast["reason"], "no_negative_action")

    def test_real_shared_feedback_adapter_keeps_new_exact_leaf_training_only(self):
        _, _, _, row, builder = fixture()
        cutoff = builder.at(row)
        outcome = builder.matured(cutoff, row, learning_event_index=6)
        prediction = adapters.adapt(cutoff, family="candidate_mlp")
        learning_view, useful = adapters.adapt_online_feedback(
            cutoff, outcome, family="candidate_mlp", learning_event_index=6)
        self.assertFalse(prediction.after_feedback_only)
        self.assertTrue(learning_view.after_feedback_only)
        self.assertEqual(learning_view.original_candidate_count, len(prediction.action_ids))
        self.assertEqual(learning_view.action_ids[:len(prediction.action_ids)],
                         prediction.action_ids)
        self.assertEqual(learning_view.action_ids[-1], outcome.exact_action_id)
        self.assertIn(len(prediction.action_ids), useful)
        self.assertNotIn(outcome.exact_action_id, prediction.action_ids)
        model = online.SharedOnlineSetMLP(hidden=8, lr=0.005, negatives=8, seed=19)
        with self.assertRaisesRegex(ValueError, "cannot enter prediction"):
            model.scores(learning_view)
        update = model.learn(learning_view, useful)
        self.assertTrue(update["updated"])
        self.assertTrue(all(index < len(prediction.action_ids)
                            for index in update["sampled_negative_indices"]))
        self.assertNotIn(outcome.exact_action_id, model.scores(prediction))

    def test_invalid_or_uninformative_labels_do_not_update(self):
        prediction, _ = example()
        model = online.SharedOnlineSetMLP(hidden=8, lr=0.005, negatives=2, seed=3)
        for labels, weight in (((), 1.0), ((0, 0), 1.0), ((len(prediction.action_ids),), 1.0),
                               ((True,), 1.0), ((0,), -0.1)):
            with self.subTest(labels=labels, weight=weight):
                with self.assertRaises(ValueError):
                    model.learn(prediction, labels, weight=weight)
        skipped = model.learn(prediction, tuple(range(len(prediction.action_ids))))
        self.assertFalse(skipped["updated"])
        self.assertEqual(skipped["reason"], "no_negative_action")
        zero = model.learn(prediction, (0,), weight=0.0)
        self.assertFalse(zero["updated"])
        self.assertNotIn("optimizer_updates", model.audit())


if __name__ == "__main__":
    unittest.main()
