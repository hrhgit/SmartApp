import unittest

import hierarchical_fallback_policy as hierarchy
from parallel_destination_data import (FrequencyHistory, asof_hierarchy, layers_for,
                                       make_arrival, semantic_key)
from parallel_destination_model import ParallelDestinationModel


def page(identifier, path):
    return {"id": identifier, "kind": "page",
            "locator": "https://example.test" + path,
            "bundleID": "browser"}


POLICY = hierarchy.FallbackPolicy("product0.6", "product", content_retention=.6)


class ParallelDestinationDataTests(unittest.TestCase):
    def test_recent_frequency_is_asof_and_counts_one_arrival_per_path_layer(self):
        start = page("start", "/start")
        child = page("child", "/docs/a")
        arrival = make_arrival(
            outcome_id="one", input_event_index=1, mature_event_index=2,
            current=start, destination=child,
            input_catalog=[start], mature_catalog=[start, child],
        )
        history = FrequencyHistory()
        history.observe(arrival, 1_000_000)
        recent = history.snapshot(1_100_000)
        parent_key = semantic_key(page("parent", "/docs/"))
        self.assertEqual(recent.total, (1, 1, 1, 1))
        self.assertEqual(recent.count(semantic_key(child), 0), 1)
        self.assertEqual(recent.count(parent_key, 0), 1)
        self.assertEqual(recent.gain(parent_key, 0, POLICY), .6)
        self.assertEqual(history.snapshot(1_400_001).total, (0, 1, 1, 1))
        with self.assertRaises(ValueError):
            history.snapshot(1_000_000)
        with self.assertRaises(ValueError):
            history.observe(arrival, 1_200_000)

    def test_recent_frequency_is_available_to_same_parallel_model(self):
        start = page("start", "/start")
        recent_target = page("recent", "/recent")
        older_target = page("older", "/older")
        tree = asof_hierarchy([start, recent_target, older_target])
        model = ParallelDestinationModel()
        history = FrequencyHistory()
        for index, (target, timestamp) in enumerate(
                ((older_target, 1_000_000), (recent_target, 1_500_000)), start=1):
            arrival = make_arrival(
                outcome_id=str(index), input_event_index=index * 2,
                mature_event_index=index * 2 + 1, current=start,
                destination=target, input_catalog=[start, recent_target, older_target],
                mature_catalog=[start, recent_target, older_target],
            )
            model.learn(arrival)
            history.observe(arrival, timestamp)
        current = layers_for(tree, "start")
        recent = history.snapshot(1_600_000)
        ordinary = model.predict(current, tree, POLICY, current_action="start")
        conditioned = model.predict(current, tree, POLICY, current_action="start",
                                    frequency=recent, recent_window=0)
        self.assertEqual(ordinary.scores["recent"], ordinary.scores["older"])
        self.assertGreater(conditioned.scores["recent"], conditioned.scores["older"])

    def test_one_arrival_has_parallel_input_and_output_without_extra_visits(self):
        old = page("old", "/docs/a")
        future = page("future", "/docs/b")
        arrival = make_arrival(
            outcome_id="arrival-1", input_event_index=10, mature_event_index=20,
            current=old, destination=future,
            input_catalog=[old], mature_catalog=[old, future],
        )
        old_key = semantic_key(old)
        future_key = semantic_key(future)
        docs_key = semantic_key(page("unused", "/docs/"))
        self.assertIn(old_key, {layer.key for layer in arrival.current})
        self.assertIn(docs_key, {layer.key for layer in arrival.current})
        self.assertIn("app:browser", {layer.key for layer in arrival.current})
        self.assertIn(future_key, {layer.key for layer in arrival.destination})
        self.assertIn(docs_key, {layer.key for layer in arrival.destination})
        self.assertIn("app:browser", {layer.key for layer in arrival.destination})
        self.assertEqual(sum(not layer.path for layer in arrival.destination), 1)
        self.assertNotIn(future_key, {layer.key for layer in arrival.current})

        model = ParallelDestinationModel()
        self.assertTrue(model.learn(arrival))
        self.assertFalse(model.learn(arrival))
        self.assertEqual(model.global_n, 1)
        self.assertEqual(model.global_by_key[docs_key][("url_parent",)], 1)

    def test_recorded_and_unvisited_parent_share_one_semantic_object(self):
        child = page("child", "/docs/a")
        derived = asof_hierarchy([child])
        recorded = asof_hierarchy([child, page("docs", "/docs/")])
        derived_docs = next(
            layer for layer in layers_for(derived, "child")
            if layer.key == semantic_key(page("unused", "/docs/"))
        )
        recorded_docs = next(
            layer for layer in layers_for(recorded, "child")
            if layer.key == semantic_key(page("docs", "/docs/"))
        )
        self.assertEqual(derived_docs.key, recorded_docs.key)
        self.assertEqual(derived_docs.evidence, "structural_parent")
        self.assertEqual(recorded_docs.evidence, "confirmed_relation")

    def test_parent_support_can_beat_sparse_child_without_independent_parent_visit(self):
        start = page("start", "/start")
        first = page("first", "/docs/a")
        second = page("second", "/docs/b")
        unrelated = page("unrelated", "/other")
        model = ParallelDestinationModel(shrinkage=5)
        for index, target in enumerate((first, second, unrelated), start=1):
            arrival = make_arrival(
                outcome_id=f"arrival-{index}", input_event_index=index * 10,
                mature_event_index=index * 10 + 1,
                current=start, destination=target,
                input_catalog=[start, first, second, unrelated],
                mature_catalog=[start, first, second, unrelated],
            )
            model.learn(arrival)
        tree = asof_hierarchy([start, first, second, unrelated])
        current = layers_for(tree, "start")
        output = model.predict(current, tree, POLICY, current_action="start")
        docs = next(identifier for identifier in tree.derived_parent_ids
                    if tree.metadata[identifier].get("locator") == "https://example.test/docs/")
        self.assertGreater(output.scores[docs], output.scores["first"])
        self.assertGreater(output.scores[docs], output.scores["second"])
        self.assertLessEqual(max(output.scores.values()), 1)

    def test_future_metadata_cannot_enter_prediction_input(self):
        start = page("start", "/start")
        future = page("future", "/secret/path")
        arrival = make_arrival(
            outcome_id="one", input_event_index=10, mature_event_index=20,
            current=start, destination=future,
            input_catalog=[start], mature_catalog=[start, future],
        )
        self.assertNotIn(semantic_key(future), {layer.key for layer in arrival.current})
        with self.assertRaises(ValueError):
            make_arrival(
                outcome_id="one", input_event_index=20, mature_event_index=20,
                current=start, destination=future,
                input_catalog=[start], mature_catalog=[start, future],
            )

    def test_directory_and_window_class_are_parallel_stable_layers(self):
        current = {"id": "window:one", "kind": "window", "bundleID": "finder"}
        directory = {"id": "folder:one", "kind": "directory", "bundleID": "finder",
                     "locator": "file:///Users/test/Documents/Project"}
        arrival = make_arrival(
            outcome_id="folder-arrival", input_event_index=1, mature_event_index=2,
            current=current, destination=directory,
            input_catalog=[current], mature_catalog=[current, directory],
        )
        self.assertIn("window_class:finder", {layer.key for layer in arrival.current})
        self.assertIn("app:finder", {layer.key for layer in arrival.current})
        self.assertIn("app:finder", {layer.key for layer in arrival.destination})
        parent = {"id": "unvisited", "kind": "directory",
                  "locator": "file:///Users/test/Documents"}
        self.assertIn(semantic_key(parent), {layer.key for layer in arrival.destination})

    def test_current_parent_generalizes_to_an_unseen_sibling_input(self):
        project_a = page("project-a", "/workspace/a")
        project_b = page("project-b", "/workspace/b")
        other = page("other", "/other/current")
        docs = page("docs", "/docs/a")
        distractions = [page(f"d-{i}", f"/other/d-{i}") for i in range(4)]
        catalog = [project_a, project_b, other, docs, *distractions]
        model = ParallelDestinationModel(shrinkage=2)
        for index, (source, destination) in enumerate(
                [(project_a, docs), *[(other, item) for item in distractions]], start=1):
            model.learn(make_arrival(
                outcome_id=f"event-{index}", input_event_index=index * 2,
                mature_event_index=index * 2 + 1, current=source,
                destination=destination, input_catalog=catalog,
                mature_catalog=catalog,
            ))
        tree = asof_hierarchy(catalog)
        unseen_sibling = layers_for(tree, "project-b")
        output = model.predict(unseen_sibling, tree, POLICY,
                               current_action="project-b")
        docs_parent = next(identifier for identifier in tree.derived_parent_ids
                           if tree.metadata[identifier].get("locator") ==
                           "https://example.test/docs/")
        self.assertIn(semantic_key(page("unused", "/workspace/")),
                      output.input_layers)
        self.assertGreater(output.scores[docs_parent],
                           output.scores["d-0"])


if __name__ == "__main__":
    unittest.main()
