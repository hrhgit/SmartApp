"""One-arrival and path-label checks for the shared non-sequence replays."""

from __future__ import annotations

from dataclasses import dataclass, replace
import unittest

import run_offline_batch_rankers as offline
import sequence_asof_features as asof
import shared_model_input as shared
import shared_model_adapters as adapters
import shared_tabular_runner as tabular
from test_shared_model_input import app, event, observed, observation, page, update


@dataclass(frozen=True)
class Case:
    position: int
    cutoff: shared.CutoffInput
    outcome: shared.MaturedOutcome | None = None
    weight: float = 1.0


def cases() -> tuple[Case, Case, Case]:
    other = app("other")
    old = page("page:old", "https://example.test/topic/old")
    future = page("page:future", "https://example.test/topic/new")
    events = [update(other, 0), update(old, 1), observed(old["id"], 2),
              observed(other["id"], 3), update(future, 4),
              observed(future["id"], 5), observed(other["id"], 6),
              observed(future["id"], 7)]
    history = offline.TargetHistory(events)
    index = asof.build_rich_index(events, history, 8)
    builder = shared.SharedInputBuilder(events, history, index)
    first = observation(events, other, future, 3, 5, sample="train")
    second = observation(events, other, future, 6, 7, sample="test")
    first_cutoff, second_cutoff = builder.at(first), builder.at(second)
    train = Case(0, first_cutoff, builder.matured(first_cutoff, first,
                                                learning_event_index=6))
    test = Case(1, second_cutoff)
    feedback = Case(1, second_cutoff, builder.matured(second_cutoff, second,
                                                     learning_event_index=len(events)))
    return train, test, feedback


class SharedTabularRunnerTests(unittest.TestCase):
    def test_statistical_and_online_mlp_train_parent_without_extra_arrival(self) -> None:
        train, test, feedback = cases()
        self.assertIsNone(train.outcome.exact_candidate_id)
        self.assertTrue(train.outcome.useful_action_ids)
        statistical, stat_audit = tabular.run_path_statistics((train,), (test,))
        mlp, mlp_audit = tabular.run_shared_online_mlp(
            (train,), (test,), (feedback,), hidden=8)
        for output in (statistical[1], mlp[1]):
            self.assertEqual(set(output["rankable"]), set(test.cutoff.action_ids))
            self.assertEqual(len(output["rankable"]), len(test.cutoff.action_ids))
            self.assertLessEqual(len(output["predictions"]), 4)
        self.assertEqual(stat_audit["train_unique_arrivals"], 1)
        self.assertEqual(stat_audit["train_parent_only_at_input"], 1)
        self.assertEqual(stat_audit["train_new_exact_after_feedback"], 1)
        self.assertEqual(mlp_audit["mature_unique_arrivals"], 1)
        self.assertEqual(mlp_audit["new_exact_after_feedback"], 1)
        self.assertEqual(mlp_audit["parent_only_at_input"], 1)
        self.assertEqual(mlp_audit["optimizer_updates"], 1)
        self.assertEqual(mlp_audit.get("mature_test_feedback", 0), 0)

    def test_catboost_sampling_keeps_all_useful_paths(self) -> None:
        train, _, _ = cases()
        example = adapters.adapt_feedback_training(
            train.cutoff, train.outcome, family="catboost", learning_event_index=6)
        self.assertTrue(example.prediction.after_feedback_only)
        self.assertEqual(example.prediction.original_candidate_count,
                         len(train.cutoff.action_ids))
        self.assertEqual(example.prediction.action_ids[-1], train.outcome.exact_action_id)
        self.assertEqual(example.labels.exact_index, len(example.prediction.action_ids) - 1)
        selected = tabular._sample_training_indices(example, 2)
        self.assertTrue(set(example.labels.useful_indices) <= set(selected))
        self.assertEqual(len(selected), len(set(selected)))

    def test_duplicate_matured_arrival_is_rejected(self) -> None:
        train, test, feedback = cases()
        duplicate = Case(2, train.cutoff, train.outcome)
        with self.assertRaises(ValueError):
            tabular.run_path_statistics((train, duplicate), (test,))

    def test_empty_asof_action_catalog_has_empty_prediction(self) -> None:
        train, test, feedback = cases()
        empty = Case(test.position, replace(test.cutoff, candidates=()))
        statistical, _ = tabular.run_path_statistics((train,), (empty,))
        online, _ = tabular.run_shared_online_mlp((train,), (empty,), (feedback,), hidden=8)
        self.assertEqual(statistical[empty.position], {"predictions": [], "rankable": []})
        self.assertEqual(online[empty.position], {"predictions": [], "rankable": []})

    def test_confirmed_but_not_eligible_destination_is_skipped_consistently(self) -> None:
        train, test, _ = cases()
        ineligible = replace(
            train.outcome,
            candidate_paths={},
            confirmed_destination=replace(
                train.outcome.confirmed_destination,
                metadata={**train.outcome.confirmed_destination.metadata,
                          "admission": {"basis": "excluded"}},
            ),
        )
        case = replace(train, outcome=ineligible)
        _, statistical = tabular.run_path_statistics((case,), (test,))
        _, online = tabular.run_shared_online_mlp((case,), (test,), (), hidden=8)
        self.assertEqual(statistical["train_no_eligible_training_action"], 1)
        self.assertEqual(online["mature_no_eligible_training_action"], 1)
        self.assertEqual(online.get("optimizer_updates", 0), 0)


if __name__ == "__main__":
    unittest.main()
