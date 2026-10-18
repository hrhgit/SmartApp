"""Focused checks for the offline cleaning comparison's causal accounting."""

from __future__ import annotations

import types
import unittest
from unittest import mock

import run_break_cleaning_study as study
import run_offline_batch_rankers as offline
import online_model_suite


def app(name: str) -> offline.Action:
    meta = {"id": "app:" + name, "kind": "app", "bundleID": name,
            "admission": {"basis": "standalone_application"}}
    return offline.Action(meta["id"], meta, "app", name)


def row(sample: str, outcome: str, current: str, truth: str, at: int,
        *, split: str = "train") -> offline.Observation:
    return offline.Observation(
        sample_id=sample, split=split, position=at,
        input_event_index=at, completed_event_index=at + 1,
        input_timestamp_ms=at * 1_000,
        outcome_timestamp_ms=(at + 1) * 1_000,
        outcome_event_id=outcome, current=app(current), truth=app(truth),
    )


class TrainingArmTests(unittest.TestCase):
    def test_original_members_remain_fixed_and_added_arrivals_do_not_double_count(self) -> None:
        original = [row("old-a", "arrival-a", "A", "B", 1),
                    row("old-b", "arrival-b", "B", "C", 3)]
        derived = types.SimpleNamespace(
            salvaged_train=(row("first-a", "arrival-b", "C", "D", 5),
                            row("first-b", "arrival-c", "C", "D", 7)),
            stitched_train=(row("join-a", "arrival-c", "A", "D", 9),
                            row("join-b", "arrival-d", "D", "A", 11)),
        )
        base, base_weights, base_audit = study.training_arm(original, derived, "base", boundary_index=20)
        self.assertEqual([item.sample_id for item in base], ["old-a", "old-b"])
        self.assertEqual(base_weights, {"old-a": 1.0, "old-b": 1.0})
        self.assertEqual(base_audit["original_members_retained"], 2)
        both, weights, audit = study.training_arm(original, derived, "both_half", boundary_index=20)
        self.assertEqual([item.sample_id for item in both], ["old-a", "old-b", "first-b", "join-b"])
        self.assertEqual(weights["first-b"], 1.0)
        self.assertEqual(weights["join-b"], 0.5)
        self.assertEqual(audit["first_step_duplicate_confirmed_arrival"], 1)
        self.assertEqual(audit["qualified_join_duplicate_confirmed_arrival"], 1)

    def test_unmatured_inferred_label_stays_out_of_frozen_training(self) -> None:
        derived = types.SimpleNamespace(salvaged_train=(),
                                        stitched_train=(row("too-late", "late", "A", "B", 10),))
        rows, _, audit = study.training_arm([], derived, "short_join", boundary_index=11)
        self.assertFalse(rows)
        self.assertEqual(audit["qualified_join_not_mature_before_boundary"], 1)

    def test_half_weight_affects_asof_counts_and_time_windows(self) -> None:
        stats = study.WeightedHistoryStats()
        first = row("first", "arrival-a", "A", "B", 1)
        second = row("join", "arrival-b", "A", "B", 3)
        self.assertTrue(stats.add_weighted(first, 1.0))
        self.assertTrue(stats.add_weighted(second, 0.5))
        self.assertFalse(stats.add_weighted(second, 0.5))
        self.assertEqual(stats.global_counts["app:B"], 1.5)
        self.assertEqual(stats.conditional_counts["app:A", "app:B"], 1.5)
        self.assertEqual(stats.summary("app:A", 5_000)["recent_1h"]["app:B"], 1.5)
        with self.assertRaises(ValueError):
            stats.summary("app:A", 2_000)

    def test_same_target_and_current_app_are_not_training_navigation(self) -> None:
        self.assertFalse(study._usable(row("same", "arrival-a", "A", "A", 1)))
        current_page = offline.Action("page:x", {
            "id": "page:x", "kind": "page", "bundleID": "A",
            "locator": "https://example.test/x",
            "admission": {"basis": "supported_browser_page"}}, "page", "A")
        coarse = row("coarse", "arrival-b", "A", "A", 3)
        coarse = offline.dataclasses.replace(coarse, current=current_page)
        self.assertFalse(study._usable(coarse))

    def test_online_negative_labels_are_limited_to_original_cutoff(self) -> None:
        model = online_model_suite.CandidateMLP(8, 0.001, negatives=8)
        model.seen_labels.update({"available_before", "seen_only_later"})
        with mock.patch.object(model.hasher, "transform", wraps=model.hasher.transform) as transform:
            model.learn({"label": "truth", "x": {"current": 1.0},
                         "allowed_negatives": frozenset({"available_before"})})
        pair_rows = transform.call_args.args[0]
        destinations = [next(key for key in pair if key.startswith("destination:"))
                        for pair in pair_rows]
        self.assertEqual(len(destinations), 2)
        self.assertTrue(any(key.endswith("available_before") for key in destinations))
        self.assertFalse(any(key.endswith("seen_only_later") for key in destinations))

    def test_online_path_labels_are_one_update_and_never_negative_examples(self) -> None:
        model = online_model_suite.CandidateMLP(8, 0.001, negatives=8)
        model.seen_labels.update({"page:parent", "page:other"})
        record = {
            "positive_labels": ("page:exact", "page:parent"),
            "allowed_negatives": frozenset({"page:parent", "page:other"}),
            "x": {"current": 1.0},
            "candidate_features": {
                "page:exact": {"path_count_all_log": 2.0, "candidate_kind": "page"},
                "page:parent": {"path_count_all_log": 5.0, "candidate_kind": "page"},
                "page:other": {"path_count_all_log": 1.0, "candidate_kind": "page"},
            },
        }
        with mock.patch.object(model.hasher, "transform", wraps=model.hasher.transform) as transform:
            with mock.patch.object(model.model, "partial_fit", wraps=model.model.partial_fit) as fit:
                model.learn(record)
        self.assertEqual(fit.call_count, 1)
        self.assertEqual(fit.call_args.args[1], [1, 1, 0])
        pair_rows = transform.call_args.args[0]
        self.assertEqual(len(pair_rows), 3)
        self.assertIn("candidate:value:path_count_all_log", pair_rows[0])
        self.assertIn("candidate:category:candidate_kind=page", pair_rows[1])
        self.assertEqual(model.seen_labels, {"page:exact", "page:parent", "page:other"})

    def test_online_model_can_score_new_asof_parent_from_shared_features(self) -> None:
        model = online_model_suite.CandidateMLP(8, 0.001, negatives=2)
        features = {
            "page:known": {"candidate_kind": "page", "path_count_all_log": 1.0},
            "page:new-parent": {"candidate_kind": "page", "path_count_all_log": 1.0},
        }
        model.learn({"positive_labels": ("page:known",),
                     "x": {"known_current": 1.0},
                     "candidate_features": features,
                     "allowed_negatives": ("page:new-parent",),
                     "sample_all_asof_negatives": True})
        scores = model.scores({"labels": ("page:known", "page:new-parent"),
                               "x": {"known_current": 1.0},
                               "candidate_features": features,
                               "allow_unseen_candidates": True})
        self.assertEqual(set(scores), set(features))
        self.assertGreater(scores["page:new-parent"], 0.0)

    def test_path_menu_derives_prior_url_section_without_future_target_leak(self) -> None:
        current = app("other")
        old_pages = [{"id": f"page:{index}", "kind": "page", "bundleID": "browser",
                      "locator": f"https://example.test/topic/{index}",
                      "admission": {"basis": "supported_browser_page"}}
                     for index in range(8)]
        future = {"id": "page:future", "kind": "page", "bundleID": "browser",
                  "locator": "https://example.test/topic/new",
                  "admission": {"basis": "supported_browser_page"}}
        events = [{"kind": "target.update", "data": {"target": current.meta}}]
        events.extend({"kind": "target.update", "data": {"target": item}}
                      for item in old_pages)
        cutoff = len(events)
        events.append({"kind": "session.begin"})
        events.append({"kind": "target.update", "data": {"target": future}})
        history = offline.TargetHistory(events)
        future_action = offline.action_from_meta(future, catalog_only=True)
        query = offline.Observation(
            sample_id="future", split="test", position=cutoff,
            input_event_index=cutoff, completed_event_index=cutoff + 1,
            input_timestamp_ms=cutoff * 1_000,
            outcome_timestamp_ms=(cutoff + 1) * 1_000,
            outcome_event_id="arrival-future", current=current, truth=future_action)
        ranking = [item["id"] for item in old_pages]
        outputs, audit = study.path_menu_variants(
            [query], history,
            {"statistical/base": {cutoff: {"predictions": ranking[:4],
                                           "rankable": ranking}}}, "base")
        menu = outputs["path_menu__statistical/base"][cutoff]["predictions"]
        hierarchy = study.hierarchy_report.build_context(study._prepared(query), history).hierarchy
        locators = {hierarchy.metadata[action].get("locator") for action in menu}
        self.assertIn("https://example.test/topic/", locators)
        self.assertNotIn("page:future", menu)
        self.assertNotIn("app:other", menu)
        self.assertEqual(audit["model_audits"]["statistical/base"]["queries_with_derived_parent"], 1)

        recorded_only, _ = study.path_menu_variants(
            [query], history,
            {"statistical/base": {cutoff: {"predictions": ranking[:4],
                                           "rankable": ranking}}}, "base",
            include_derived=False)
        control_menu = recorded_only["path_menu_recorded_only__statistical/base"][cutoff]["predictions"]
        self.assertTrue(all(action not in hierarchy.derived_parent_ids
                            for action in control_menu))
        control_score = study.evaluation.measure(
            study.evaluation_rows([query], history),
            recorded_only["path_menu_recorded_only__statistical/base"])
        self.assertEqual(control_score["success"]["top4_hits"], 0)

        audited = study.evaluation_rows([query], history)
        score = study.evaluation.measure(audited, outputs["path_menu__statistical/base"])
        self.assertEqual(score["success"]["top4_hits"], 1)
        self.assertEqual(score["exact"]["top4_hits"], 0)
        self.assertLessEqual(score["path_utility"]["product0.6"]["top4_gain"], 1.0)

        old_action = offline.action_from_meta(old_pages[0], catalog_only=True)
        exact_query = offline.dataclasses.replace(
            query, sample_id="known-child", truth=old_action)
        parent = next(action for action in menu
                      if hierarchy.metadata[action].get("locator")
                      == "https://example.test/topic/")
        overlap_output = {cutoff: {"predictions": [parent, "page:0"],
                                   "rankable": [parent, "page:0"]}}
        overlap_score = study.evaluation.measure(
            study.evaluation_rows([exact_query], history), overlap_output)
        self.assertEqual(overlap_score["path_utility"]["product0.6"]["top4_gain"], 1.0)

        current_page_query = offline.dataclasses.replace(query, current=old_action)
        page_outputs, _ = study.path_menu_variants(
            [current_page_query], history,
            {"statistical/base": {cutoff: {"predictions": ranking[:4],
                                           "rankable": ranking}}}, "base")
        page_menu = page_outputs["path_menu__statistical/base"][cutoff]["predictions"]
        self.assertNotIn("page:0", page_menu)
        self.assertNotIn("app:browser", page_menu)

        # A later destination on a different host must not create a prior menu route.
        future_only = dict(future, id="page:unseen-host",
                           locator="https://future-only.test/area/new")
        events.append({"kind": "target.update", "data": {"target": future_only}})
        future_history = offline.TargetHistory(events)
        future_only_query = offline.dataclasses.replace(
            query, truth=offline.action_from_meta(future_only, catalog_only=True))
        future_outputs, _ = study.path_menu_variants(
            [future_only_query], future_history,
            {"statistical/base": {cutoff: {"predictions": ranking[:4],
                                           "rankable": ranking}}}, "base")
        future_tree = study.hierarchy_report.build_context(
            study._prepared(future_only_query), future_history).hierarchy
        emitted = future_outputs["path_menu__statistical/base"][cutoff]["rankable"]
        self.assertFalse(any("future-only.test" in str(future_tree.metadata[action].get("locator", ""))
                             for action in emitted))


if __name__ == "__main__":
    unittest.main()
