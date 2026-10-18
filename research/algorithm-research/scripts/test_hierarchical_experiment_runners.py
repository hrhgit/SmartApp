#!/usr/bin/env python3
"""Integration checks shared by the batch and sequence hierarchy runners."""

import sys
import unittest
from pathlib import Path

import numpy as np


sys.path.insert(0, str(Path(__file__).resolve().parent))

import run_hierarchical_batch_experiment as batch_experiment  # noqa: E402
import run_hierarchical_sequence_experiment as sequence_experiment  # noqa: E402
import run_offline_batch_rankers as runner  # noqa: E402
import evaluation_contract as evaluation  # noqa: E402


def action(identifier, kind="directory", *, parent=None, bundle=None):
    meta = {"id": identifier, "kind": kind}
    if parent:
        meta["parentID"] = parent
    if bundle:
        meta["bundleID"] = bundle
    return runner.Action(identifier, meta, kind, bundle)


def synthetic_group():
    parent = action("parent")
    current = action("current", parent="parent", bundle="browser")
    other = action("other")
    row = runner.Observation(
        sample_id="sample", split="test", position=0,
        input_event_index=10, completed_event_index=12,
        input_timestamp_ms=1000, outcome_timestamp_ms=2000,
        outcome_event_id="outcome", current=current, truth=other,
        current_parent=action("app:browser", "app", bundle="browser"),
    )
    # The current target dominates the raw scores.  If it is left in the
    # posterior, its mass incorrectly makes its parent look useful.
    return runner.QueryGroup(
        row=row, actions=[parent, current, other],
        prior=np.asarray([-100.0, 100.0, 0.0]),
        features=[{}, {}, {}], truth_index=2,
    )


def tied_unrelated_group():
    current = action("current")
    truth = action("a")
    row = runner.Observation(
        sample_id="tied", split="test", position=1,
        input_event_index=20, completed_event_index=21,
        input_timestamp_ms=3000, outcome_timestamp_ms=4000,
        outcome_event_id="tied-outcome", current=current, truth=truth,
    )
    return runner.QueryGroup(
        row=row,
        actions=[action("c"), action("b"), action("a")],
        prior=np.zeros(3, dtype=float),
        features=[{}, {}, {}],
        truth_index=2,
    )


def equivalent_current_home_group():
    first = action("home-a", "page", bundle="browser")
    second = action("home-b", "page", bundle="browser")
    first.meta["locator"] = "https://chatgpt.com/"
    second.meta["locator"] = "https://chatgpt.com/"
    other = action("other")
    row = runner.Observation(
        sample_id="current-home", split="test", position=2,
        input_event_index=30, completed_event_index=31,
        input_timestamp_ms=5000, outcome_timestamp_ms=6000,
        outcome_event_id="current-home-outcome", current=first, truth=other,
    )
    return runner.QueryGroup(
        row=row,
        actions=[first, second, other],
        prior=np.asarray([10.0, 100.0, 0.0]),
        features=[{}, {}, {}],
        truth_index=2,
    )


def duplicate_noncurrent_home_group():
    first = action("home-a", "page", bundle="browser")
    second = action("home-b", "page", bundle="browser")
    first.meta["locator"] = "https://chatgpt.com/"
    second.meta["locator"] = "https://chatgpt.com/"
    destinations = [first, second, action("x"), action("y"), action("z")]
    row = runner.Observation(
        sample_id="duplicate-home", split="test", position=3,
        input_event_index=40, completed_event_index=41,
        input_timestamp_ms=7000, outcome_timestamp_ms=8000,
        outcome_event_id="duplicate-home-outcome",
        current=action("current"), truth=destinations[2],
    )
    return runner.QueryGroup(
        row=row,
        actions=destinations,
        prior=np.asarray([100.0, 90.0, 80.0, 70.0, 60.0]),
        features=[{}, {}, {}, {}, {}],
        truth_index=2,
    )


class HierarchicalRunnerTests(unittest.TestCase):
    def test_emitted_home_metadata_reaches_shared_scorer_without_persistence(self):
        current = action("current", bundle="browser")
        truth = action("chat", "page", bundle="browser")
        truth.meta["locator"] = "https://chatgpt.com/c/12345678-1234-1234-1234-123456789abc"
        row = runner.Observation(
            sample_id="chat-sample", split="test", position=0,
            input_event_index=10, completed_event_index=12,
            input_timestamp_ms=1000, outcome_timestamp_ms=2000,
            outcome_event_id="chat-outcome", current=current, truth=truth,
        )
        home = {
            "id": "chat-home", "kind": "page", "bundleID": "browser",
            "locator": "https://chatgpt.com/",
        }
        rows = runner.evaluation_rows([row], [home])
        result = evaluation.measure(rows, {
            0: {"predictions": ["chat-home"], "rankable": ["chat-home"]},
        })
        self.assertEqual(result["top1_hits"], 1)
        self.assertEqual(result["exact"]["top1_hits"], 0)
        self.assertAlmostEqual(result["actionable_topk"]["top1_gain"], 0.6)

    def test_batch_conditions_hierarchy_posterior_on_a_real_change(self):
        group = synthetic_group()
        zero = lambda value: np.zeros(len(value.actions), dtype=float)
        scorers = {name: zero for name in batch_experiment.MODEL_ORDER}
        outputs, _, variants = batch_experiment.build_prediction_variants([group], scorers)

        for name, details in variants.items():
            prediction = outputs[name][0]["predictions"]
            if details["kind"] == "original":
                self.assertEqual(prediction[0], "current")
            else:
                self.assertEqual(prediction[0], "other")
                self.assertNotIn("current", prediction)
                self.assertNotIn("app:browser", prediction)

    def test_sequence_conditions_hierarchy_posterior_on_a_real_change(self):
        group = synthetic_group()
        context = sequence_experiment.build_context(group)
        original = sequence_experiment.original_output(context, group.prior)
        self.assertEqual(original["predictions"][0], "current")
        filtered = sequence_experiment.current_filtered_output(context, group.prior)
        self.assertEqual(filtered["predictions"][0], "other")
        self.assertNotIn("current", filtered["predictions"])

        for policy in sequence_experiment.POLICIES:
            reranked = sequence_experiment.hierarchical_output(context, group.prior, policy)
            self.assertEqual(reranked["predictions"][0], "other")
            self.assertNotIn("current", reranked["predictions"])
            self.assertNotIn("app:browser", reranked["predictions"])

    def test_unrelated_logit_ties_keep_the_original_ranker_order(self):
        group = tied_unrelated_group()
        zero = lambda value: np.zeros(len(value.actions), dtype=float)
        scorers = {name: zero for name in batch_experiment.MODEL_ORDER}
        batch_outputs, _, batch_variants = batch_experiment.build_prediction_variants(
            [group], scorers,
        )
        for name, details in batch_variants.items():
            if details["kind"] == "hierarchical_rerank":
                self.assertEqual(batch_outputs[name][1]["predictions"], ["a", "b", "c"])

        context = sequence_experiment.build_context(group)
        filtered = sequence_experiment.current_filtered_output(context, group.prior)
        self.assertEqual(filtered["predictions"], ["a", "b", "c"])
        for policy in sequence_experiment.POLICIES:
            reranked = sequence_experiment.hierarchical_output(context, group.prior, policy)
            self.assertEqual(reranked["predictions"], filtered["predictions"])

    def test_current_home_filter_removes_every_equivalent_home_id(self):
        group = equivalent_current_home_group()
        zero = lambda value: np.zeros(len(value.actions), dtype=float)
        scorers = {name: zero for name in batch_experiment.MODEL_ORDER}
        outputs, _, variants = batch_experiment.build_prediction_variants([group], scorers)
        for name, details in variants.items():
            predictions = outputs[name][2]["predictions"]
            if details["kind"] == "original":
                self.assertEqual(predictions[0], "home-b")
            else:
                self.assertEqual(predictions, ["other"])

        context = sequence_experiment.build_context(group)
        filtered = sequence_experiment.current_filtered_output(context, group.prior)
        self.assertEqual(filtered["predictions"], ["other"])
        for policy in sequence_experiment.POLICIES:
            reranked = sequence_experiment.hierarchical_output(context, group.prior, policy)
            self.assertEqual(reranked["predictions"], ["other"])

    def test_reference_deduplicates_noncurrent_homes_and_backfills(self):
        group = duplicate_noncurrent_home_group()
        zero = lambda value: np.zeros(len(value.actions), dtype=float)
        scorers = {name: zero for name in batch_experiment.MODEL_ORDER}
        outputs, _, variants = batch_experiment.build_prediction_variants([group], scorers)
        for name, details in variants.items():
            predictions = outputs[name][3]["predictions"]
            if details["kind"] == "original":
                self.assertEqual(predictions, ["home-a", "home-b", "x", "y"])
            elif details["kind"] == "current_filtered_exact":
                self.assertEqual(predictions, ["home-a", "x", "y", "z"])
            else:
                self.assertLessEqual(len(set(predictions) & {"home-a", "home-b"}), 1)

        context = sequence_experiment.build_context(group)
        filtered = sequence_experiment.current_filtered_output(context, group.prior)
        self.assertEqual(filtered["predictions"], ["home-a", "x", "y", "z"])


if __name__ == "__main__":
    unittest.main()
