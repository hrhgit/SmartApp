"""The common sequence runner keeps action order, labels and time separate."""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace
import unittest

import numpy as np

import evaluation_contract as evaluation
import run_offline_batch_rankers as offline
import run_offline_sequence_shadow as sequence
import sequence_asof_features as asof
import shared_model_input as shared
import shared_sequence_runner as runner
from test_shared_model_input import app, fixture, observed, observation, page, update


def cases():
    events, _, _, train_row, _ = fixture()
    never_seen_at_test = page("page:later", "https://later.test/private")
    events = [*events, observed("page:old", 6), observed("app:other", 7),
              update(never_seen_at_test, 8), observed(never_seen_at_test["id"], 9)]
    history = offline.TargetHistory(events)
    index = asof.build_rich_index(events, history, 8)
    builder = shared.SharedInputBuilder(events, history, index)
    train_cutoff = builder.at(train_row)
    outcome = builder.matured(train_cutoff, train_row, learning_event_index=7)
    test_row = observation(events, app("other"), never_seen_at_test, 7, 9,
                           sample="test-sample")
    test_row = replace(test_row, position=1, split="test")
    test_cutoff = builder.at(test_row)
    return (SimpleNamespace(position=0, cutoff=train_cutoff, outcome=outcome, weight=1.0),
            SimpleNamespace(position=1, cutoff=test_cutoff), test_row, history,
            never_seen_at_test)


class SharedSequenceRunnerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            sequence.require_torch()
        except RuntimeError:
            raise unittest.SkipTest("private torch runtime unavailable")

    def test_all_three_families_train_on_one_parent_supported_arrival(self):
        train, test, test_row, history, later = cases()
        self.assertIsNone(train.outcome.exact_candidate_id)
        self.assertTrue(train.outcome.useful_action_ids)
        self.assertNotIn(later["id"], test.cutoff.action_ids)
        for family in ("rnn", "lstm", "transformer"):
            with self.subTest(family=family):
                config = {**sequence.model_configurations(23, 4, 64)[family][0],
                          "embedding_dim": 8, "hidden_dim": 8,
                          "learning_rate": 0.005, "seed": 9}
                if family == "transformer":
                    config["heads"] = 2
                    config["feedforward_dim"] = 16
                outputs, audit = runner.run_shared_sequence(
                    [train], [test], family=family, mode="full", epochs=1,
                    config=config)
                self.assertEqual(set(outputs), {1})
                self.assertEqual(len(outputs[1]["predictions"]),
                                 min(4, len(test.cutoff.action_ids)))
                self.assertEqual(set(outputs[1]["rankable"]), set(test.cutoff.action_ids))
                self.assertNotIn(later["id"], outputs[1]["rankable"])
                self.assertEqual(audit["counts"]["trained_via_parent_without_exact"], 1)
                self.assertEqual(audit["counts"]["feedback_only_exact_leaf_appended"], 1)
                self.assertEqual(audit["counts"]["trained_without_original_exact"], 1)
                self.assertEqual(audit["training"]["path_examples_without_exact_candidate"], 0)
                self.assertIn("context_numeric", audit["forwarded_fields"])
                self.assertFalse(audit["test_truth_accessed"])
                audited = sequence.evaluation_rows_with_known_actions([test_row], history)
                self.assertEqual(evaluation.measure(audited, outputs)["success"]["n"], 1)

    def test_training_feedback_must_precede_fixed_test_cutoff(self):
        train, test, _, _, _ = cases()
        earlier = SimpleNamespace(position=2, cutoff=train.cutoff)
        with self.assertRaisesRegex(ValueError, "fixed test boundary"):
            runner.run_shared_sequence([train], [earlier], family="rnn", epochs=1)

    def test_new_exact_leaf_is_a_single_positive_when_original_catalog_is_empty(self):
        train, test, _, _, _ = cases()
        empty_cutoff = replace(train.cutoff, candidates=())
        train = SimpleNamespace(position=train.position, cutoff=empty_cutoff,
                                outcome=train.outcome, weight=1.0)
        outputs, audit = runner.run_shared_sequence(
            [train], [test], family="rnn", mode="target_only", epochs=1,
            config={"bucket_count": 64, "sequence_length": 4,
                    "embedding_dim": 8, "hidden_dim": 8})
        self.assertEqual(audit["counts"]["feedback_only_exact_leaf_appended"], 1)
        self.assertEqual(audit["counts"]["no_original_contrast_action"], 1)
        self.assertEqual(audit["training"]["examples"], 1)
        self.assertEqual(set(outputs[1]["rankable"]), set(test.cutoff.action_ids))

    def test_duplicate_arrival_is_rejected_and_short_sequence_falls_back(self):
        train, test, _, _, _ = cases()
        with self.assertRaisesRegex(ValueError, "cannot train.*twice"):
            runner.run_shared_sequence([train, train], [test], family="rnn", epochs=1)
        short = replace(test.cutoff, sequence_steps=test.cutoff.sequence_steps[-1:],
                        sequence_numeric=np.zeros((1, asof.NUMERIC_WIDTH), np.float32))
        outputs, audit = runner.run_shared_sequence(
            [train], [SimpleNamespace(position=1, cutoff=short)],
            family="rnn", mode="target_only", epochs=1,
            config={"bucket_count": 64, "sequence_length": 4,
                    "embedding_dim": 8, "hidden_dim": 8})
        self.assertEqual(outputs[1]["rankable"], list(short.action_ids))
        self.assertEqual(audit["counts"]["prediction_prior_only"], 1)

    def test_action_keyed_prior_uses_same_order_for_fitting_and_fallback(self):
        train, test, _, _, _ = cases()
        short = replace(test.cutoff, sequence_steps=test.cutoff.sequence_steps[-1:],
                        sequence_numeric=np.zeros((1, asof.NUMERIC_WIDTH), np.float32))
        last = short.action_ids[-1]
        def prior(cutoff):
            return {action: float(action == last) for action in reversed(cutoff.action_ids)}
        outputs, audit = runner.run_shared_sequence(
            [train], [SimpleNamespace(position=1, cutoff=short)],
            family="rnn", mode="target_only", epochs=1,
            config={"bucket_count": 64, "sequence_length": 4,
                    "embedding_dim": 8, "hidden_dim": 8},
            prior_for_cutoff=prior)
        self.assertEqual(outputs[1]["predictions"][0], last)
        self.assertEqual(audit["prior_source"], "caller_cutoff_action_scores")
        with self.assertRaisesRegex(ValueError, "cover exactly"):
            runner.run_shared_sequence(
                [train], [test], family="rnn", mode="target_only", epochs=1,
                prior_for_cutoff=lambda cutoff: {})


if __name__ == "__main__":
    unittest.main()
