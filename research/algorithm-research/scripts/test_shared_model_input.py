"""Causal contract checks for the common, research-only model input layer."""

from __future__ import annotations

from dataclasses import replace
import copy
import unittest
from unittest import mock

import run_offline_batch_rankers as offline
import sequence_asof_features as asof
import shared_model_input as shared


def app(bundle: str) -> dict:
    return {"id": "app:" + bundle, "kind": "app", "bundleID": bundle,
            "admission": {"basis": "standalone_application"}}


def page(identifier: str, url: str, bundle: str = "browser") -> dict:
    return {"id": identifier, "kind": "page", "bundleID": bundle, "locator": url,
            "admission": {"basis": "supported_browser_page"}}


def window(identifier: str, bundle: str = "documents") -> dict:
    return {"id": identifier, "kind": "window", "bundleID": bundle,
            "admission": {"basis": "standard_work_window",
                          "policyVersion": "target-eligibility-v4"}}


def event(kind: str, number: int, *, data: dict | None = None, session: str = "s") -> dict:
    return {"id": "e" + str(number), "kind": kind, "sessionID": session,
            "timestampMs": number * 1_000, "monotonicMs": number * 1_000,
            "data": data or {}}


def observed(target_id: str, number: int, *, reason: str = "foreground_change") -> dict:
    item = event("context.change", number, data={
        "targetID": target_id, "reason": reason,
        "inputCoverage": {"version": "foreground-coverage-v2", "status": "foreground_only",
                          "sources": [{"source": "foreground:test", "status": "ready"}]}})
    item["source"] = "foreground:test"
    return item


def update(meta: dict, number: int) -> dict:
    return event("target.update", number, data={"target": meta})


def observation(events: list[dict], current_meta: dict, truth_meta: dict,
                input_index: int, outcome_index: int, *, sample: str = "sample"
                ) -> offline.Observation:
    current = offline.action_from_meta(current_meta, catalog_only=False)
    truth = offline.action_from_meta(truth_meta, catalog_only=False)
    assert current is not None and truth is not None
    return offline.Observation(
        sample_id=sample, split="train", position=0,
        input_event_index=input_index, completed_event_index=outcome_index,
        input_timestamp_ms=events[input_index]["timestampMs"],
        outcome_timestamp_ms=events[outcome_index]["timestampMs"],
        outcome_event_id=events[outcome_index]["id"], current=current, truth=truth)


def fixture() -> tuple[list[dict], offline.TargetHistory, asof.RichIndex,
                       offline.Observation, shared.SharedInputBuilder]:
    other = app("other")
    old = page("page:old", "https://example.test/topic/old")
    future = page("page:future", "https://example.test/topic/new")
    events = [update(other, 0), update(old, 1), observed(old["id"], 2),
              observed(other["id"], 3), update(future, 4), observed(future["id"], 5)]
    history = offline.TargetHistory(events)
    index = asof.build_rich_index(events, history, 8)
    row = observation(events, other, future, 3, 5)
    return events, history, index, row, shared.SharedInputBuilder(events, history, index)


def window_fixture() -> tuple[list[dict], offline.Observation,
                              shared.SharedInputBuilder]:
    current, destination = app("other"), window("window:first")
    events = [update(current, 0), observed(current["id"], 1),
              update(destination, 2), observed(destination["id"], 3)]
    history = offline.TargetHistory(events)
    index = asof.build_rich_index(events, history, 8)
    row = observation(events, current, destination, 1, 3)
    return events, row, shared.SharedInputBuilder(events, history, index)


class SharedModelInputTests(unittest.TestCase):
    def test_window_feedback_proves_only_the_observed_admitted_window(self) -> None:
        events, row, builder = window_fixture()
        cutoff = builder.at(row)
        self.assertEqual(cutoff.action_ids, ())
        outcome = builder.matured(cutoff, row, learning_event_index=4)
        self.assertEqual(outcome.confirmed_destination.action_id, "app:documents")
        self.assertEqual(outcome.confirmed_destination.metadata["admission"]["basis"],
                         "window_action_fallback")
        self.assertEqual(outcome.confirmed_destination.metadata["_verified_window_fallback"], {
            "outcome_event_id": "e3", "bundleID": "documents",
            "raw_kind": "window", "raw_admission_basis": "standard_work_window",
        })
        self.assertIsNone(offline.action_from_meta(
            dict(outcome.confirmed_destination.metadata), catalog_only=True))

        def rebuilt(changed_events: list[dict]) -> shared.SharedInputBuilder:
            history = offline.TargetHistory(changed_events)
            rich = asof.build_rich_index(changed_events, history, 8)
            return shared.SharedInputBuilder(changed_events, history, rich)

        no_foreground = copy.deepcopy(events)
        no_foreground[3]["data"]["inputCoverage"]["status"] = "unknown"
        with self.assertRaisesRegex(ValueError, "confirmed foreground arrival"):
            rebuilt(no_foreground).matured(cutoff, row, learning_event_index=4)

        wrong_target = copy.deepcopy(events)
        wrong_target[3]["data"]["targetID"] = "app:other"
        with self.assertRaisesRegex(ValueError, "admitted foreground window"):
            rebuilt(wrong_target).matured(cutoff, row, learning_event_index=4)

        unadmitted = copy.deepcopy(events)
        unadmitted[2]["data"]["target"]["admission"]["basis"] = "authorization_helper"
        with self.assertRaisesRegex(ValueError, "admitted foreground window"):
            rebuilt(unadmitted).matured(cutoff, row, learning_event_index=4)

        wrong_window = offline.action_from_meta(window("window:other", "different"),
                                                catalog_only=True)
        with self.assertRaisesRegex(ValueError, "admitted foreground window"):
            builder.matured(cutoff, replace(row, truth=wrong_window),
                            learning_event_index=4)

    def test_asof_candidates_include_parent_of_old_page_but_not_future_leaf(self) -> None:
        events, history, index, row, builder = fixture()
        cutoff = builder.at(row)
        self.assertNotIn("page:future", cutoff.action_ids)
        self.assertIn("page:old", cutoff.action_ids)
        self.assertNotIn("app:other", cutoff.action_ids)
        parent = next(item for item in cutoff.candidates
                      if item.metadata.get("locator") == "https://example.test/topic/")
        self.assertEqual(parent.source, "derived_from_known_locator")
        self.assertEqual(parent.path_counts[-1], 1)
        self.assertEqual(parent.direct_counts[-1], 0)
        self.assertNotEqual(parent.action_id, parent.semantic_key)
        self.assertEqual(cutoff.current_action_id, "app:other")
        self.assertGreaterEqual(len(cutoff.current_layers), 1)
        parent_row = cutoff.feature_rows()[cutoff.action_ids.index(parent.action_id)]
        self.assertEqual(parent_row["candidate_parent_token_1"],
                         offline.token(parent.layers[1].key))
        self.assertEqual(parent_row["current_parent_token_1"], offline.token(None))
        self.assertFalse(any("example.test" in str(value) for key, value in parent_row.items()
                             if key.endswith("_token")))

        with self.assertRaises(ValueError):
            builder.matured(cutoff, row, learning_event_index=5)
        outcome = builder.matured(cutoff, row, learning_event_index=6)
        self.assertIsNone(outcome.exact_candidate_id)
        self.assertEqual(outcome.exact_action_id, "page:future")
        self.assertEqual(outcome.confirmed_destination.action_id, "page:future")
        self.assertEqual(outcome.confirmed_destination.source, "after_feedback_only")
        self.assertEqual(outcome.confirmed_destination.path_counts, (0, 0, 0, 0))
        self.assertTrue(outcome.confirmed_destination.layers)
        self.assertNotIn(outcome.confirmed_destination.action_id, cutoff.action_ids)
        self.assertIn(parent.action_id, outcome.useful_action_ids)
        self.assertIn("url_parent", outcome.candidate_paths[parent.action_id])
        self.assertEqual(outcome.outcome_event_id, "e5")
        self.assertEqual(sum(int(exact) for exact, _, _ in outcome.labels_in_order(cutoff)), 0)

    def test_direct_owner_app_is_useful_for_a_new_foreground_page(self) -> None:
        current, owner = app("other"), app("browser")
        future = page("page:future", "https://example.test/new")
        events = [update(current, 0), update(owner, 1), observed(owner["id"], 2),
                  observed(current["id"], 3), update(future, 4),
                  observed(future["id"], 5)]
        history = offline.TargetHistory(events)
        index = asof.build_rich_index(events, history, 8)
        row = observation(events, current, future, 3, 5)
        builder = shared.SharedInputBuilder(events, history, index)
        cutoff = builder.at(row)
        outcome = builder.matured(cutoff, row, learning_event_index=6)
        self.assertIn(owner["id"], cutoff.action_ids)
        self.assertEqual(outcome.candidate_paths[owner["id"]], ("owner_app",))

    def test_owner_app_through_content_parent_is_not_a_useful_action(self) -> None:
        current, owner = app("other"), app("browser")
        parent = page("page:parent", "https://example.test/topic/")
        future = page("page:future", "https://example.test/topic/leaf", bundle="")
        events = [update(current, 0), update(owner, 1), observed(owner["id"], 2),
                  update(parent, 3), observed(parent["id"], 4),
                  observed(current["id"], 5), update(future, 6),
                  observed(future["id"], 7)]
        history = offline.TargetHistory(events)
        index = asof.build_rich_index(events, history, 8)
        row = observation(events, current, future, 5, 7)
        builder = shared.SharedInputBuilder(events, history, index)
        cutoff = builder.at(row)
        outcome = builder.matured(cutoff, row, learning_event_index=8)
        self.assertIn(owner["id"], cutoff.action_ids)
        self.assertIn(parent["id"], outcome.useful_action_ids)
        self.assertNotIn(owner["id"], outcome.useful_action_ids)
        self.assertIn("url_parent", outcome.candidate_paths[parent["id"]])

    def test_owner_app_through_directory_parent_is_not_a_useful_action(self) -> None:
        current, owner = app("other"), app("finder")
        parent = {"id": "directory:parent", "kind": "directory", "bundleID": "finder",
                  "locator": "file:///tmp/smartapp-study/",
                  "admission": {"basis": "observed_finder_directory"}}
        future = {"id": "directory:future", "kind": "directory", "bundleID": "",
                  "locator": "file:///tmp/smartapp-study/child/",
                  "admission": {"basis": "observed_finder_directory"}}
        events = [update(current, 0), update(owner, 1), observed(owner["id"], 2),
                  update(parent, 3), observed(parent["id"], 4),
                  observed(current["id"], 5), update(future, 6),
                  observed(future["id"], 7)]
        history = offline.TargetHistory(events)
        index = asof.build_rich_index(events, history, 8)
        row = observation(events, current, future, 5, 7)
        builder = shared.SharedInputBuilder(events, history, index)
        cutoff = builder.at(row)
        outcome = builder.matured(cutoff, row, learning_event_index=8)
        self.assertIn(owner["id"], cutoff.action_ids)
        self.assertIn(parent["id"], outcome.useful_action_ids)
        self.assertNotIn(owner["id"], outcome.useful_action_ids)
        self.assertIn("path_parent", outcome.candidate_paths[parent["id"]])

    def test_current_owner_app_is_blocked_even_for_another_page(self) -> None:
        owner = app("browser")
        current = page("page:current", "https://example.test/current")
        future = page("page:future", "https://example.test/next")
        events = [update(owner, 0), observed(owner["id"], 1),
                  update(current, 2), observed(current["id"], 3),
                  update(future, 4), observed(future["id"], 5)]
        history = offline.TargetHistory(events)
        index = asof.build_rich_index(events, history, 8)
        row = observation(events, current, future, 3, 5)
        builder = shared.SharedInputBuilder(events, history, index)
        cutoff = builder.at(row)
        outcome = builder.matured(cutoff, row, learning_event_index=6)
        self.assertNotIn(owner["id"], cutoff.action_ids)
        self.assertNotIn(owner["id"], outcome.useful_action_ids)

    def test_future_target_and_truth_cannot_change_cutoff_input(self) -> None:
        events, _, _, row, builder = fixture()
        first = builder.at(row)
        altered = list(events)
        different = page("page:different", "https://elsewhere.test/private")
        altered[4] = update(different, 4)
        altered[5] = observed(different["id"], 5)
        other_history = offline.TargetHistory(altered)
        other_index = asof.build_rich_index(altered, other_history, 8)
        different_row = observation(altered, row.current.meta, different, 3, 5)
        second = shared.SharedInputBuilder(altered, other_history, other_index).at(different_row)
        self.assertEqual(first.action_ids, second.action_ids)
        self.assertEqual(first.feature_rows(), second.feature_rows())
        self.assertEqual(first.sequence_keys, second.sequence_keys)
        self.assertEqual(first.context, second.context)
        self.assertNotIn("page:different", second.action_ids)

    def test_all_adapters_use_one_action_order_and_feature_schema(self) -> None:
        _, _, _, row, builder = fixture()
        cutoff = builder.at(row)
        action_ids = cutoff.action_ids
        self.assertEqual(cutoff.statistical_view()["action_ids"], action_ids)
        self.assertEqual(cutoff.mlp_view()["labels"], action_ids)
        self.assertEqual(tuple(row["action_id"] for row in cutoff.feature_rows()), action_ids)
        self.assertEqual(len(cutoff.catboost_rows()), len(action_ids))
        self.assertEqual(set(cutoff.catboost_rows()[0]),
                         set(shared.COMMON_CONTEXT_COLUMNS + shared.CANDIDATE_NUMERIC_COLUMNS
                             + shared.CANDIDATE_CATEGORICAL_COLUMNS))
        sequence = cutoff.sequence_view(lambda key: len(key) + 1)
        self.assertEqual(sequence["action_ids"], action_ids)
        self.assertEqual(len(sequence["candidate_ids"]), len(action_ids))
        self.assertEqual(sequence["candidate_numeric"].shape,
                         (len(action_ids), asof.CANDIDATE_WIDTH))
        self.assertEqual(sequence["sequence_numeric"].shape[0], len(cutoff.sequence_steps))
        again = builder.at(row)
        self.assertEqual(again.schema_version, cutoff.schema_version)
        self.assertEqual(again.action_ids, action_ids)

    def test_synthetic_current_owner_is_input_layer_not_candidate(self) -> None:
        current = page("page:current", "https://example.test/current")
        other = page("page:other", "https://example.test/other")
        events = [update(current, 0), update(other, 1), observed(other["id"], 2),
                  observed(current["id"], 3), observed(other["id"], 4)]
        history = offline.TargetHistory(events)
        index = asof.build_rich_index(events, history, 8)
        row = observation(events, current, other, 3, 4)
        cutoff = shared.SharedInputBuilder(events, history, index).at(row)
        self.assertIn("app:browser", tuple(layer.key for layer in cutoff.current_layers))
        self.assertNotIn("app:browser", cutoff.action_ids)
        self.assertNotIn("page:current", cutoff.action_ids)
        self.assertIn("page:other", cutoff.action_ids)

    def test_unpaired_gap_does_not_restore_sequence_or_invent_dwell(self) -> None:
        one, two = app("one"), app("two")
        events = [update(one, 0), observed(one["id"], 1),
                  event("capture.gap", 2, data={"reason": "observation_suspended"}),
                  update(two, 3), observed(two["id"], 4)]
        history = offline.TargetHistory(events)
        index = asof.build_rich_index(events, history, 8)
        row = observation(events, two, one, 4, 4)
        row = replace(row, completed_event_index=5, outcome_event_id="later")
        cutoff = shared.SharedInputBuilder(events, history, index).at(row)
        self.assertEqual(cutoff.sequence_keys, ("app:two",))
        self.assertEqual(cutoff.context["has_confirmed_dwell"], 0.0)
        self.assertEqual(cutoff.context["known_gap_duration"], 0.0)
        self.assertEqual(cutoff.candidate("app:one").direct_counts[-1], 1)

    def test_same_outcome_identity_is_one_training_event(self) -> None:
        _, _, _, row, builder = fixture()
        cutoff = builder.at(row)
        outcome = builder.matured(cutoff, row, learning_event_index=6)
        again = builder.matured(cutoff, row, learning_event_index=7)
        ledger = shared.OutcomeLedger()
        self.assertTrue(ledger.admit(outcome))
        self.assertFalse(ledger.admit(again))
        self.assertEqual(ledger.unique_arrivals, 1)
        with self.assertRaises(ValueError):
            ledger.admit(replace(outcome, exact_action_id="page:conflicting"))
        self.assertEqual(outcome.outcome_event_id, again.outcome_event_id)
        self.assertEqual(outcome.candidate_paths, again.candidate_paths)
        self.assertEqual(len({outcome.outcome_event_id, again.outcome_event_id}), 1)
        self.assertEqual(sum(1 for candidate in cutoff.candidates
                             if candidate.action_id in outcome.useful_action_ids),
                         len(outcome.useful_action_ids))

    def test_reused_id_with_new_locator_is_not_an_exact_match(self) -> None:
        current = app("other")
        old_home = page("page:reused", "https://example.test/topic/")
        changed = page("page:reused", "https://example.test/topic/new")
        events = [update(current, 0), update(old_home, 1), observed(old_home["id"], 2),
                  observed(current["id"], 3), update(changed, 4),
                  observed(changed["id"], 5)]
        history = offline.TargetHistory(events)
        index = asof.build_rich_index(events, history, 8)
        row = observation(events, current, changed, 3, 5)
        builder = shared.SharedInputBuilder(events, history, index)
        cutoff = builder.at(row)
        self.assertIn("page:reused", cutoff.action_ids)
        outcome = builder.matured(cutoff, row, learning_event_index=6)
        self.assertIsNone(outcome.exact_candidate_id)
        self.assertEqual(outcome.confirmed_destination.action_id, "page:reused")
        self.assertNotEqual(outcome.confirmed_destination.semantic_key,
                            cutoff.candidate("page:reused").semantic_key)
        self.assertEqual(outcome.confirmed_destination.metadata["locator"], changed["locator"])
        self.assertIn("page:reused", outcome.useful_action_ids)
        self.assertIn("url_parent", outcome.candidate_paths["page:reused"])

    def test_target_update_alone_cannot_create_action_or_parent(self) -> None:
        current = app("other")
        unvisited = page("page:unvisited", "https://private.test/topic/unvisited")
        events = [update(current, 0), observed(current["id"], 1),
                  update(unvisited, 2), observed(current["id"], 3)]
        history = offline.TargetHistory(events)
        index = asof.build_rich_index(events, history, 8)
        row = observation(events, current, unvisited, 3, 3)
        cutoff = shared.SharedInputBuilder(events, history, index).at(row)
        self.assertNotIn(unvisited["id"], cutoff.action_ids)
        self.assertFalse(any(item.metadata.get("locator", "").startswith("https://private.test")
                             for item in cutoff.candidates))

    def test_foreground_resolution_confirms_action_without_extra_visit_count(self) -> None:
        current = app("other")
        resolved = page("page:resolved", "https://example.test/topic/resolved")
        unvisited = page("page:unvisited", "https://never.test/hidden")
        events = [update(current, 0), observed(current["id"], 1),
                  update(resolved, 2), update(unvisited, 3),
                  observed(resolved["id"], 4, reason="resolution_update"),
                  observed(current["id"], 5)]
        history = offline.TargetHistory(events)
        index = asof.build_rich_index(events, history, 8)
        row = observation(events, current, resolved, 5, 5)
        cutoff = shared.SharedInputBuilder(events, history, index).at(row)
        self.assertIn(resolved["id"], cutoff.action_ids)
        self.assertNotIn(unvisited["id"], cutoff.action_ids)
        self.assertEqual(cutoff.candidate(resolved["id"]).direct_counts[-1], 0)
        self.assertEqual(cutoff.candidate(resolved["id"]).path_counts[-1], 0)
        self.assertTrue(any(item.metadata.get("locator") == "https://example.test/topic/"
                            for item in cutoff.candidates))

    def test_unconfirmed_metadata_update_cannot_change_confirmed_locator(self) -> None:
        old = page("page:stable-id", "https://example.test/old/leaf")
        changed = page("page:stable-id", "https://different.test/new/leaf")
        current = app("other")
        events = [update(old, 0), observed(old["id"], 1), update(current, 2),
                  observed(current["id"], 3), update(changed, 4),
                  observed(current["id"], 5)]
        history = offline.TargetHistory(events)
        index = asof.build_rich_index(events, history, 8)
        row = observation(events, current, changed, 5, 5)
        cutoff = shared.SharedInputBuilder(events, history, index).at(row)
        self.assertEqual(cutoff.candidate(old["id"]).metadata["locator"], old["locator"])
        self.assertFalse(any(item.metadata.get("locator", "").startswith("https://different.test")
                             for item in cutoff.candidates))
        self.assertTrue(any(item.metadata.get("locator") == "https://example.test/old/"
                            for item in cutoff.candidates))

    def test_every_adapter_declares_new_fields_or_fails_closed(self) -> None:
        for name in ("statistical", "candidate_mlp", "catboost", "rnn", "lstm", "transformer"):
            usage = shared.validate_adapter_field_usage(name)
            self.assertFalse(usage["consume"] & usage["intentional_ignore"])
        with mock.patch.object(shared, "COMMON_CONTEXT_COLUMNS",
                               shared.COMMON_CONTEXT_COLUMNS + ("new_context_feature",)):
            with self.assertRaisesRegex(ValueError, "schema changed"):
                shared.validate_adapter_field_usage("catboost")
        incomplete = {"consume": frozenset({"current"}),
                      "intentional_ignore": frozenset({"identity"})}
        with mock.patch.dict(shared.ADAPTER_FIELD_USAGE, {"candidate_mlp": incomplete}):
            with self.assertRaisesRegex(ValueError, "incomplete"):
                shared.validate_adapter_field_usage("candidate_mlp")


if __name__ == "__main__":
    unittest.main()
