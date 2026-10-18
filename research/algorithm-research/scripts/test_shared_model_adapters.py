"""All model families consume one ordered, prediction-time SmartApp sample."""

from __future__ import annotations

from dataclasses import replace
from unittest import mock
import unittest

import numpy as np

import evaluation_contract as scoring
import run_offline_batch_rankers as offline
import run_shared_model_input_study as study
import shared_model_adapters as adapters
import shared_model_input as shared
from test_shared_model_input import (app, event, fixture, observation, observed,
                                     page, update, window_fixture)
import sequence_asof_features as asof


def known_fixture() -> tuple[offline.Observation, shared.SharedInputBuilder]:
    current, destination = app("other"), page("page:old", "https://example.test/topic/old")
    events = [update(current, 0), update(destination, 1),
              observed(destination["id"], 2), observed(current["id"], 3),
              observed(destination["id"], 4)]
    history = offline.TargetHistory(events)
    rich = asof.build_rich_index(events, history, 8)
    row = observation(events, current, destination, 3, 4)
    return row, shared.SharedInputBuilder(events, history, rich)


def same_locator_fixture(kind: str, locator: str
                         ) -> tuple[dict, dict, shared.CutoffInput,
                                    shared.MaturedOutcome]:
    current = app("other")
    if kind == "page":
        old, new = page("page:old", locator), page("page:new", locator)
    else:
        old = {"id": "directory:old", "kind": "directory",
               "bundleID": "com.apple.finder", "locator": locator,
               "admission": {"basis": "observed_finder_directory"}}
        new = {**old, "id": "directory:new"}
    events = [update(current, 0), update(old, 1), observed(old["id"], 2),
              observed(current["id"], 3), update(new, 4), observed(new["id"], 5)]
    history = offline.TargetHistory(events)
    rich = asof.build_rich_index(events, history, 8)
    row = observation(events, current, new, 3, 5)
    builder = shared.SharedInputBuilder(events, history, rich)
    cutoff = builder.at(row)
    return old, new, cutoff, builder.matured(cutoff, row, learning_event_index=6)


def score_old_action(old: dict, new: dict) -> float:
    data = {
        "observations": [{"position": 0, "index": 3, "ready_index": 5,
                          "target": app("other"), "known_actions": [old]}],
        "evaluation": [{"position": 0, "sample_id": "same-locator",
                        "truth_action": new["id"], "truth_metadata": new,
                        "evidence_grade": "original", "split": "test"}],
    }
    return scoring.path_action_gain(scoring.prepare_evaluation(data)[0], old["id"])


class SharedAdapterTests(unittest.TestCase):
    def test_every_family_has_same_candidates_and_path_labels(self) -> None:
        _, _, _, row, builder = fixture()
        cutoff = builder.at(row)
        outcome = builder.matured(cutoff, row, learning_event_index=6)
        action_ids = cutoff.action_ids
        labels = None
        for family in ("statistical", "candidate_mlp", "catboost", "rnn", "lstm", "transformer"):
            prediction = adapters.adapt(cutoff, family=family,
                                        bucket=lambda key: len(key) + 1)
            self.assertEqual(prediction.action_ids, action_ids)
            self.assertEqual(prediction.common_context, cutoff.context)
            example = adapters.attach_matured(prediction, outcome, learning_event_index=6)
            self.assertEqual(example.labels.exact_index, None)
            self.assertGreater(len(example.labels.useful_indices), 0)
            if labels is None:
                labels = example.labels
            else:
                self.assertEqual(example.labels, labels)
        self.assertNotIn(outcome.exact_action_id, action_ids)
        self.assertEqual(sum(labels.exact), 0)
        self.assertTrue(all(labels.typed_paths[i] is not None for i in labels.useful_indices))

    def test_exact_and_useful_remain_distinct_when_leaf_is_known(self) -> None:
        row, builder = known_fixture()
        cutoff = builder.at(row)
        outcome = builder.matured(cutoff, row, learning_event_index=5)
        examples = [adapters.attach_matured(
            adapters.adapt(cutoff, family=family, bucket=lambda key: len(key) + 1),
            outcome, learning_event_index=5)
            for family in ("statistical", "candidate_mlp", "catboost", "rnn", "lstm", "transformer")]
        self.assertTrue(all(example.labels == examples[0].labels for example in examples))
        exact_index = examples[0].labels.exact_index
        self.assertIsNotNone(exact_index)
        self.assertEqual(cutoff.action_ids[exact_index], "page:old")
        self.assertTrue(examples[0].labels.exact[exact_index])
        self.assertTrue(examples[0].labels.useful[exact_index])
        self.assertEqual(sum(examples[0].labels.exact), 1)
        self.assertGreaterEqual(sum(examples[0].labels.useful), 1)

    def test_sequence_modes_only_forward_declared_arrays(self) -> None:
        _, _, _, row, builder = fixture()
        cutoff = builder.at(row)
        for family in ("rnn", "lstm", "transformer"):
            target = adapters.adapt(cutoff, family=family, mode="target_only",
                                    bucket=lambda key: len(key) + 1)
            temporal = adapters.adapt(cutoff, family=family, mode="temporal",
                                      bucket=lambda key: len(key) + 1)
            hierarchy = adapters.adapt(cutoff, family=family, mode="hierarchy",
                                       bucket=lambda key: len(key) + 1)
            full = adapters.adapt(cutoff, family=family, mode="full",
                                  bucket=lambda key: len(key) + 1)
            self.assertNotIn("context_numeric", target.payload)
            self.assertNotIn("sequence_numeric", target.payload)
            self.assertIn("context_numeric", temporal.payload)
            self.assertIn("sequence_numeric", temporal.payload)
            self.assertNotIn("candidate_layers", temporal.payload)
            self.assertIn("candidate_layers", hierarchy.payload)
            self.assertNotIn("context_numeric", hierarchy.payload)
            self.assertIn("context_numeric", full.payload)
            self.assertIn("candidate_layers", full.payload)
            self.assertEqual(tuple(full.payload["action_ids"]), cutoff.action_ids)
            self.assertEqual(full.payload["candidate_ids"].shape[0], len(cutoff.action_ids))
            self.assertEqual(full.payload["context_numeric"].shape,
                             (len(shared.COMMON_CONTEXT_COLUMNS),))

    def test_mlp_pairs_and_catboost_rows_include_common_fields(self) -> None:
        _, _, _, row, builder = fixture()
        cutoff = builder.at(row)
        mlp = adapters.adapt(cutoff, family="candidate_mlp")
        pairs = adapters.mlp_pair_inputs(mlp)
        self.assertEqual(tuple(pair["labels"][0] for pair in pairs), cutoff.action_ids)
        self.assertTrue(all("context:time_hour_sin" in pair["x"] for pair in pairs))
        self.assertTrue(all("candidate:path_count_all_log" in pair["x"] for pair in pairs))
        self.assertTrue(any(key.startswith("candidate_layer:") for key in pairs[0]["x"]))
        self.assertTrue(all(any(key.startswith("candidate:candidate_parent_token_1=")
                                for key in pair["x"]) for pair in pairs))
        catboost = adapters.adapt(cutoff, family="catboost")
        self.assertEqual(len(catboost.payload["rows"]), len(cutoff.action_ids))
        self.assertTrue(all("time_hour_sin" in item and "path_count_all_log" in item
                            for item in catboost.payload["rows"]))
        self.assertTrue(all("candidate_parent_token_1" in item
                            and "current_parent_token_1" in item
                            for item in catboost.payload["rows"]))
        self.assertTrue(any(item["candidate_parent_token_1"] != offline.token(None)
                            for item in catboost.payload["rows"]))

    def test_future_outcome_not_in_prediction_and_learning_has_maturity_gate(self) -> None:
        _, _, _, row, builder = fixture()
        cutoff = builder.at(row)
        prediction = adapters.adapt(cutoff, family="catboost")
        self.assertNotIn("page:future", prediction.action_ids)
        self.assertNotIn("outcome", prediction.payload)
        outcome = builder.matured(cutoff, row, learning_event_index=6)
        with self.assertRaises(ValueError):
            adapters.attach_matured(prediction, outcome, learning_event_index=5)
        self.assertEqual(adapters.attach_matured(prediction, outcome,
                                                learning_event_index=6).labels.outcome_event_id, "e5")
        other = replace(outcome, sample_id="different")
        with self.assertRaises(ValueError):
            adapters.attach_matured(prediction, other, learning_event_index=6)

    def test_one_arrival_cannot_be_learned_twice(self) -> None:
        _, _, _, row, builder = fixture()
        cutoff = builder.at(row)
        outcome = builder.matured(cutoff, row, learning_event_index=6)
        prediction = adapters.adapt(cutoff, family="candidate_mlp")
        gate = adapters.MaturedFeedbackGate()
        first = gate.attach_once(prediction, outcome, learning_event_index=6)
        self.assertEqual(first.labels.outcome_event_id, "e5")
        with self.assertRaisesRegex(ValueError, "cannot be learned twice"):
            gate.attach_once(prediction, outcome, learning_event_index=7)

    def test_feedback_training_appends_new_exact_only_after_maturity_for_every_family(self) -> None:
        _, _, _, row, builder = fixture()
        cutoff = builder.at(row)
        outcome = builder.matured(cutoff, row, learning_event_index=6)
        original = cutoff.action_ids
        labels = None
        for family in ("statistical", "candidate_mlp", "catboost", "rnn", "lstm", "transformer"):
            kwargs = {"bucket": lambda key: len(key) + 1} if family in {"rnn", "lstm", "transformer"} else {}
            before = adapters.adapt(cutoff, family=family, **kwargs)
            self.assertFalse(before.after_feedback_only)
            self.assertIsNone(before.original_candidate_count)
            self.assertEqual(before.action_ids, original)
            self.assertNotIn(outcome.exact_action_id, before.action_ids)
            with self.assertRaisesRegex(ValueError, "not-yet-matured"):
                adapters.adapt_feedback_training(cutoff, outcome, family=family,
                                                 learning_event_index=5, **kwargs)
            example = adapters.adapt_feedback_training(cutoff, outcome, family=family,
                                                       learning_event_index=6, **kwargs)
            learned = example.prediction
            self.assertTrue(learned.after_feedback_only)
            self.assertEqual(learned.original_candidate_count, len(original))
            self.assertEqual(learned.action_ids, original + (outcome.exact_action_id,))
            self.assertEqual(example.labels.exact_index, len(original))
            self.assertEqual(example.labels.useful_indices[-1], len(original))
            self.assertEqual(sum(example.labels.exact), 1)
            self.assertGreater(len(example.labels.useful_indices), 1)
            self.assertEqual(learned.cutoff.candidates[-1].source, "recorded_action")
            self.assertEqual(outcome.confirmed_destination.source, "after_feedback_only")
            with self.assertRaisesRegex(ValueError, "original prediction cutoff"):
                adapters.attach_matured(learned, outcome, learning_event_index=6)
            if labels is None:
                labels = example.labels
            else:
                self.assertEqual(example.labels, labels)
            if family == "candidate_mlp":
                pairs = adapters.mlp_pair_inputs(learned)
                self.assertEqual(len(pairs), len(original) + 1)
                self.assertIn("candidate:candidate_source=recorded_action", pairs[-1]["x"])
                self.assertNotIn("candidate:candidate_source=after_feedback_only", pairs[-1]["x"])
            elif family == "catboost":
                self.assertEqual(learned.payload["rows"][-1]["candidate_source"],
                                 "recorded_action")
            elif family == "lstm":
                prior = np.zeros(len(learned.action_ids), dtype=np.float32)
                seq = adapters.to_sequence_example(example, prior)
                self.assertEqual(seq.truth_index, len(original))
                self.assertEqual(seq.positive_indices, example.labels.useful_indices)
                self.assertEqual(len(seq.candidate_ids), len(original) + 1)
        self.assertEqual(cutoff.action_ids, original)
        self.assertNotIn(outcome.exact_action_id, cutoff.action_ids)

    def test_feedback_training_existing_exact_and_empty_original_catalog(self) -> None:
        row, builder = known_fixture()
        cutoff = builder.at(row)
        outcome = builder.matured(cutoff, row, learning_event_index=5)
        example = adapters.adapt_feedback_training(cutoff, outcome,
                                                   family="candidate_mlp",
                                                   learning_event_index=5)
        self.assertEqual(example.prediction.action_ids, cutoff.action_ids)
        self.assertEqual(example.prediction.original_candidate_count, len(cutoff.action_ids))
        self.assertEqual(example.labels.exact_index, cutoff.action_ids.index("page:old"))
        self.assertEqual(sum(example.labels.exact), 1)

        _, _, _, new_row, new_builder = fixture()
        original = new_builder.at(new_row)
        new_outcome = new_builder.matured(original, new_row, learning_event_index=6)
        empty = replace(original, candidates=())
        for family in ("statistical", "candidate_mlp", "catboost", "rnn", "lstm", "transformer"):
            kwargs = {"bucket": lambda key: len(key) + 1} if family in {"rnn", "lstm", "transformer"} else {}
            self.assertEqual(adapters.adapt(empty, family=family, **kwargs).action_ids, ())
            first = adapters.adapt_feedback_training(empty, new_outcome,
                                                     family=family,
                                                     learning_event_index=6, **kwargs)
            self.assertEqual(first.prediction.original_candidate_count, 0)
            self.assertEqual(first.prediction.action_ids, (new_outcome.exact_action_id,))
            self.assertEqual(first.labels.useful_indices, (0,))
            self.assertEqual(first.labels.exact_index, 0)

    def test_verified_window_fallback_is_a_new_feedback_only_app_positive(self) -> None:
        _, row, builder = window_fixture()
        cutoff = builder.at(row)
        outcome = builder.matured(cutoff, row, learning_event_index=4)
        self.assertTrue(adapters.confirmed_destination_is_eligible(outcome))
        example = adapters.adapt_feedback_training(
            cutoff, outcome, family="candidate_mlp", learning_event_index=4)
        self.assertEqual(example.prediction.original_candidate_count, 0)
        self.assertEqual(example.prediction.action_ids, ("app:documents",))
        self.assertEqual(example.labels.useful_indices, (0,))
        self.assertEqual(example.labels.exact_index, 0)
        self.assertEqual(example.prediction.cutoff.candidates[0].source, "recorded_action")
        self.assertEqual(cutoff.action_ids, ())

        metadata = dict(outcome.confirmed_destination.metadata)
        proof = dict(metadata["_verified_window_fallback"])
        for bad in (
            {key: value for key, value in metadata.items()
             if key != "_verified_window_fallback"},
            {**metadata, "_verified_window_fallback": {**proof,
                                                       "outcome_event_id": "forged"}},
            {**metadata, "_verified_window_fallback": {**proof,
                                                       "raw_admission_basis": "authorization_helper"}},
        ):
            unverified = replace(outcome.confirmed_destination, metadata=bad)
            with self.assertRaisesRegex(ValueError, "Unverified after-feedback window"):
                adapters.confirmed_destination_is_eligible(
                    replace(outcome, confirmed_destination=unverified))
            with self.assertRaisesRegex(ValueError, "Unverified after-feedback window"):
                adapters.adapt_feedback_training(
                    cutoff, replace(outcome, confirmed_destination=unverified),
                    family="candidate_mlp", learning_event_index=4)

    def test_feedback_training_rejects_duplicate_arrival_and_changed_locator(self) -> None:
        _, _, _, row, builder = fixture()
        cutoff = builder.at(row)
        outcome = builder.matured(cutoff, row, learning_event_index=6)
        gate = adapters.MaturedFeedbackGate()
        adapters.adapt_feedback_training(cutoff, outcome, family="candidate_mlp",
                                         learning_event_index=6, gate=gate)
        with self.assertRaisesRegex(ValueError, "cannot be learned twice"):
            adapters.adapt_feedback_training(cutoff, outcome, family="candidate_mlp",
                                             learning_event_index=7, gate=gate)
        changed = replace(outcome.confirmed_destination, action_id="page:old",
                          metadata={**outcome.confirmed_destination.metadata,
                                    "id": "page:old"})
        with self.assertRaisesRegex(adapters.FeedbackTrainingUnavailable,
                                    "changed its concrete destination") as error:
            adapters.adapt_online_feedback(cutoff,
                                           replace(outcome, exact_action_id="page:old",
                                                   confirmed_destination=changed),
                                           learning_event_index=6)
        self.assertEqual(error.exception.reason, "reused_id_changed_locator")

        current = replace(outcome.confirmed_destination, action_id="app:other",
                          kind="app", owner_app_id="app:other",
                          semantic_key=cutoff.current_semantic_key,
                          layers=cutoff.current_layers,
                          metadata=app("other"))
        with self.assertRaisesRegex(adapters.FeedbackTrainingUnavailable,
                                    "current destination") as error:
            adapters.adapt_online_feedback(cutoff,
                                           replace(outcome, exact_action_id="app:other",
                                                   confirmed_destination=current),
                                           learning_event_index=6)
        self.assertEqual(error.exception.reason, "same_as_current")

    def test_feedback_training_does_not_append_ineligible_destination(self) -> None:
        _, _, _, row, builder = fixture()
        cutoff = builder.at(row)
        outcome = builder.matured(cutoff, row, learning_event_index=6)
        unavailable = replace(outcome.confirmed_destination,
                              metadata={**outcome.confirmed_destination.metadata,
                                        "admission": {"basis": "unsupported"}})
        example = adapters.adapt_feedback_training(
            cutoff, replace(outcome, confirmed_destination=unavailable),
            family="catboost", learning_event_index=6)
        self.assertEqual(example.prediction.action_ids, cutoff.action_ids)
        self.assertIsNone(example.labels.exact_index)
        self.assertEqual(example.labels.useful_indices,
                         tuple(i for i, action in enumerate(cutoff.action_ids)
                               if action in outcome.candidate_paths))

    def test_nonhome_same_locator_different_id_matches_fixed_score(self) -> None:
        for kind, locator in (("page", "https://example.test/topic/same"),
                              ("directory", "file:///Users/example/Documents/Project")):
            with self.subTest(kind=kind):
                old, new, cutoff, outcome = same_locator_fixture(kind, locator)
                self.assertEqual(score_old_action(old, new), 0.0)
                old_index = cutoff.action_ids.index(old["id"])
                self.assertEqual(cutoff.candidate(old["id"]).semantic_key,
                                 outcome.confirmed_destination.semantic_key)
                cutoff = study._model_cutoff(cutoff)
                self.assertEqual(cutoff.hierarchy.metadata, {})
                example = adapters.adapt_feedback_training(
                    cutoff, outcome, family="candidate_mlp", learning_event_index=6)
                self.assertEqual(example.prediction.action_ids,
                                 cutoff.action_ids + (new["id"],))
                self.assertFalse(example.labels.useful[old_index])
                self.assertIsNone(example.labels.typed_paths[old_index])
                self.assertEqual(example.labels.exact_index, len(cutoff.action_ids))
                self.assertIn(len(cutoff.action_ids), example.labels.useful_indices)

    def test_same_homepage_different_id_preserves_home_acceptance(self) -> None:
        old, new, cutoff, outcome = same_locator_fixture("page", "https://example.test/")
        self.assertEqual(score_old_action(old, new), 1.0)
        self.assertIn(old["id"], outcome.candidate_paths)
        self.assertEqual(outcome.candidate_paths[old["id"]],
                         ("equivalent_service_home",))
        cutoff = study._model_cutoff(cutoff)
        self.assertEqual(cutoff.hierarchy.metadata, {})
        example = adapters.adapt_feedback_training(
            cutoff, outcome, family="candidate_mlp", learning_event_index=6)
        self.assertEqual(example.prediction.action_ids, cutoff.action_ids)
        self.assertIsNone(example.labels.exact_index)
        self.assertIn(cutoff.action_ids.index(old["id"]), example.labels.useful_indices)

    def test_query_group_and_sequence_example_keep_shared_order(self) -> None:
        _, _, _, row, builder = fixture()
        cutoff = builder.at(row)
        outcome = builder.matured(cutoff, row, learning_event_index=6)
        prior = np.linspace(-2.0, -1.0, len(cutoff.action_ids))
        tabular = adapters.attach_matured(adapters.adapt(cutoff, family="catboost"),
                                          outcome, learning_event_index=6)
        group = adapters.to_query_group(tabular, row, prior)
        self.assertEqual(tuple(action.key for action in group.actions), cutoff.action_ids)
        self.assertIsNone(group.truth_index)
        self.assertEqual(group.features, list(cutoff.feature_rows()))
        sequence = adapters.attach_matured(
            adapters.adapt(cutoff, family="lstm", mode="full", bucket=lambda key: len(key) + 1),
            outcome, learning_event_index=6)
        example = adapters.to_sequence_example(sequence, prior)
        self.assertEqual(example.truth_index, None)
        self.assertEqual(example.positive_indices, sequence.labels.useful_indices)
        self.assertEqual(len(example.candidate_ids), len(cutoff.action_ids))
        self.assertEqual(example.context_numeric.shape[0], len(shared.COMMON_CONTEXT_COLUMNS))

    def test_new_field_or_missing_field_fails_instead_of_silently_ignoring(self) -> None:
        _, _, _, row, builder = fixture()
        cutoff = builder.at(row)
        missing = replace(cutoff, context={key: value for key, value in cutoff.context.items()
                                           if key != "quality_failed_fraction"})
        with self.assertRaisesRegex(ValueError, "Common context fields changed"):
            adapters.adapt(missing, family="statistical")
        with mock.patch.object(shared, "COMMON_CONTEXT_COLUMNS",
                               shared.COMMON_CONTEXT_COLUMNS + ("new_feature",)):
            with self.assertRaisesRegex(ValueError, "review every model adapter"):
                adapters.validate_schema()
        broken = replace(adapters.REGISTRY["rnn", "target_only"],
                         intentionally_unused_fields=frozenset())
        with mock.patch.dict(adapters.REGISTRY, {("rnn", "target_only"): broken}):
            with self.assertRaisesRegex(ValueError, "Unaccounted fields"):
                adapters.validate_schema()
        with self.assertRaisesRegex(ValueError, "Undeclared model projection"):
            adapters.manifest_for("lstm", "undocumented")


if __name__ == "__main__":
    unittest.main()
