"""Synthetic causality and incremental-learning tests; no private data needed."""

import copy
import math
import pickle
import unittest

import online_model_suite as suite


def record(origin="a", label="one"):
    return {"origin": origin, "app": "app", "timestamp": 1000,
            "outcome_timestamp": 1500, "outcome_id": "synthetic-outcome",
            "label": label, "x": {"origin=" + origin: 1.0, "app=app": 1.0}}


def query(origin="a", labels=None):
    item = record(origin=origin)
    for key in ("label", "outcome_timestamp", "outcome_id"):
        item.pop(key)
    item["labels"] = labels or ["one", "two", "never-observed"]
    return item


class OnlineModelSuiteTests(unittest.TestCase):
    def representatives(self):
        seen = set()
        for config in suite.configurations():
            if config["family"] not in seen:
                seen.add(config["family"])
                yield config

    def test_all_configurations_build_and_accept_growing_vocabulary(self):
        configs = suite.configurations()
        self.assertEqual(len(configs), len({x["id"] for x in configs}))
        for config in configs:
            with self.subTest(config=config["id"]):
                model = suite.build(config)
                self.assertEqual(model.scores(query())["never-observed"], 0.0)
                model.learn(record(label="one"))
                first = model.scores(query())
                self.assertGreater(first["one"], 0.0)
                self.assertEqual(first["two"], 0.0)
                model.learn(record(origin="b", label="two"))
                later = model.scores(query(origin="b"))
                self.assertGreater(later["two"], 0.0)
                self.assertEqual(later["never-observed"], 0.0)
                self.assertTrue(all(math.isfinite(value) for value in later.values()))

    def test_repeated_matured_learning_changes_ranking(self):
        for config in self.representatives():
            with self.subTest(family=config["family"]):
                model = suite.build(config)
                for _ in range(10):
                    model.learn(record(origin="a", label="one"))
                    model.learn(record(origin="b", label="two"))
                for _ in range(20):
                    model.learn(record(origin="a", label="one"))
                scores = model.scores(query(origin="a"))
                self.assertGreater(scores["one"], scores["two"])

    def test_prediction_does_not_register_candidate_classes_or_read_truth(self):
        for config in self.representatives():
            with self.subTest(family=config["family"]):
                model = suite.build(config)
                model.learn(record())
                model.learn(record(origin="b", label="two"))
                before = set(model.seen_labels)
                clean = query()
                poison = copy.deepcopy(clean)
                poison.update(label="never-observed", truth="never-observed",
                              outcome_timestamp=99999999)
                self.assertEqual(model.scores(clean), model.scores(poison))
                self.assertEqual(model.seen_labels, before)
                self.assertNotIn("never-observed", model.seen_labels)

    def test_input_only_fallback_and_nonfinite_rejection(self):
        fallback = record()
        fallback.pop("x")
        self.assertIn("origin=a", suite._numeric_x(fallback))
        bad = record()
        bad["x"]["invalid"] = float("nan")
        with self.assertRaises(ValueError):
            suite._numeric_x(bad)

    def test_seeded_replay_is_reproducible(self):
        for config in self.representatives():
            with self.subTest(family=config["family"]):
                first, second = suite.build(config), suite.build(config)
                for index in range(25):
                    item = record(origin=str(index % 3), label=str(index % 5))
                    first.learn(item)
                    second.learn(item)
                q = query(origin="0", labels=[str(n) for n in range(7)])
                self.assertEqual(first.scores(q), second.scores(q))

    def test_mlp_prediction_preserves_weights_and_sampling_state(self):
        config = next(c for c in suite.configurations() if c["family"] == "candidate_mlp")
        model = suite.build(config)
        for index in range(15):
            model.learn(record(label=str(index)))
        state = pickle.dumps(model)
        model.scores(query(labels=["1", "2", "brand-new"]))
        self.assertEqual(state, pickle.dumps(model))

    def test_supplement_is_separate_and_original_grid_stays_first(self):
        all_configs = suite.configurations()
        supplement = suite.supplemental_configurations()
        self.assertEqual(len(all_configs), 91)
        self.assertEqual(len(supplement), 32)
        self.assertEqual(all_configs[59:], supplement)
        self.assertEqual({item["family"] for item in supplement},
                         {"streaming_knn", "online_cluster_frequency", "local_softmax"})

    def test_knn_identical_current_observations_vote_and_ties_use_recent(self):
        model = suite.StreamingKNN(n_neighbors=3, window_size=50)
        for label in ("one", "one", "one", "two", "two"):
            model.learn(record(label=label))
        before = copy.deepcopy(model.model._nn.window)
        scores = model.scores(query())
        self.assertAlmostEqual(scores["one"], 1 / 3)
        self.assertAlmostEqual(scores["two"], 2 / 3)
        self.assertEqual(before, model.model._nn.window)
        a = {"origin": "a", "app": "x"}
        self.assertGreater(suite._current_distance(a, {"origin": "b", "app": "x"}),
                           suite._current_distance(a, {"origin": "a", "app": "y"}))

    def test_knn_fifo_forgets_evicted_outcomes(self):
        model = suite.StreamingKNN(n_neighbors=3, window_size=2)
        for label in ("one", "two", "two"):
            model.learn(record(label=label))
        scores = model.scores(query())
        self.assertEqual(scores["one"], 0)
        self.assertEqual(scores["two"], 1)

    def test_cluster_prediction_readonly_and_counts_not_reassigned(self):
        model = suite.OnlineClusterFrequency(n_clusters=2, center_step=0.5)
        model.learn(record(origin="a", label="one"))
        model.learn(record(origin="b", label="two"))
        old = copy.deepcopy(model.counts)
        state = pickle.dumps(model)
        model.scores(query(origin="future-interface"))
        self.assertEqual(state, pickle.dumps(model))
        before_update_cluster = model.model.predict_one(model._features(record(origin="c")))
        model.learn(record(origin="c", label="three"))
        self.assertEqual(model.last_learned_cluster, before_update_cluster)
        self.assertEqual(model.counts[before_update_cluster]["three"], 1)
        for cluster, counts in old.items():
            for label, value in counts.items():
                self.assertEqual(model.counts[cluster][label], value)

    def test_cluster_centers_do_not_depend_on_destinations(self):
        first = suite.OnlineClusterFrequency(n_clusters=2, center_step=0.2)
        second = suite.OnlineClusterFrequency(n_clusters=2, center_step=0.2)
        for origin in ("a", "b", "c", "a", "d"):
            first.learn(record(origin=origin, label="one"))
            second.learn(record(origin=origin, label="other"))
        self.assertEqual(first.model.centers, second.model.centers)

    def test_local_softmax_isolates_other_origins(self):
        model = suite.LocalSoftmax(learning_rate=0.3, l2=0.001)
        model.learn(record(origin="a", label="one"))
        model.learn(record(origin="a", label="two"))
        before = model.scores(query(origin="a"))
        for _ in range(10):
            model.learn(record(origin="b", label="two"))
        self.assertEqual(before, model.scores(query(origin="a")))
        self.assertEqual(model.scores(query(origin="never-seen")),
                         {label: 0 for label in query()["labels"]})


if __name__ == "__main__":
    unittest.main()
