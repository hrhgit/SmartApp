import math
import unittest

import compare_multistep_statistical as runner
import multistep_statistical_models as models


def transition(sequence, label, outcome_time, outcome_id, input_time=0):
    return {
        "sequence": tuple(sequence),
        "origin": sequence[-1],
        "label": label,
        "input_timestamp": input_time,
        "outcome_timestamp": outcome_time,
        "outcome_id": outcome_id,
        "segment_id": 0,
    }


class SequenceModelTests(unittest.TestCase):
    def test_configuration_grid_is_unique_and_finite(self):
        configs = models.configurations()
        self.assertGreaterEqual(len(configs), 5)
        self.assertEqual(len(configs), len({config["id"] for config in configs}))
        for config in configs:
            model = models.build(config)
            model.learn(transition(("A",), "B", 10, "one"))
            scores = model.scores({"sequence": ("A",), "origin": "A", "timestamp": 20,
                                   "truth": "ignored", "background": {}})
            self.assertTrue(scores)
            self.assertTrue(all(math.isfinite(value) for value in scores.values()))

    def test_future_outcome_is_rejected(self):
        model = models.build(models.configurations()[0])
        model.learn(transition(("A",), "B", 10, "one"))
        with self.assertRaises(ValueError):
            model.scores({"sequence": ("A",), "origin": "A", "timestamp": 10})
        with self.assertRaises(ValueError):
            model.scores({"sequence": ("A",), "origin": "A", "timestamp": 9})

    def test_high_order_suffix_uses_previous_foreground_arrival(self):
        config = next(config for config in models.configurations()
                      if config["id"] == "sequence/markov2-count")
        model = models.build(config)
        model.learn(transition(("A", "B"), "C", 10, "ab-c"))
        model.learn(transition(("X", "B"), "D", 20, "xb-d"))
        self.assertEqual(model.ranked({"sequence": ("A", "B"), "origin": "B", "timestamp": 30})[0], "C")
        fallback = model.ranked({"sequence": ("Y", "B"), "origin": "B", "timestamp": 30})
        self.assertEqual(set(fallback), {"C", "D"})

    def test_duplicate_sequence_outcome_does_not_double_count(self):
        model = models.build(models.configurations()[1])
        record = transition(("A", "B"), "C", 10, "same-outcome")
        self.assertTrue(model.learn(record))
        self.assertFalse(model.learn(record))
        self.assertEqual(model.state_summary()["independent_outcomes"], 1)
        self.assertEqual(model.state_summary()["records_by_order"]["2"], 1)

    def test_query_truth_background_and_candidates_are_ignored(self):
        model = models.build(models.configurations()[1])
        model.learn(transition(("A",), "B", 10, "one"))
        clean = model.scores({"sequence": ("A",), "origin": "A", "timestamp": 20})
        noisy = model.scores({"sequence": ("A",), "origin": "A", "timestamp": 20,
                              "truth": "future-answer", "background": {"other": 999},
                              "candidates": ["future-candidate"]})
        self.assertEqual(clean, noisy)


class ForegroundSequenceBoundaryTests(unittest.TestCase):
    class History:
        @staticmethod
        def metadata_at(target_id, _index):
            return {
                "id": target_id,
                "kind": "app",
                "bundleID": target_id,
                "admission": {"basis": "standalone_application"},
            }

    @staticmethod
    def event(number, kind, target=None, reason=None):
        data = {}
        if target is not None:
            data["targetID"] = target
        if reason is not None:
            data["reason"] = reason
        return {
            "id": f"event-{number}", "sessionID": "session-1", "sequence": number,
            "kind": kind, "timestampMs": number * 1000, "monotonicMs": number * 1000,
            "source": "test", "data": data,
        }

    def test_capture_gap_cuts_sequence_and_resolution_is_not_an_arrival(self):
        events = [
            self.event(0, "session.start"),
            self.event(1, "context.change", "A"),
            self.event(2, "context.change", "A", "resolution_update"),
            self.event(3, "context.change", "B"),
            self.event(4, "capture.gap"),
            self.event(5, "context.change", "C"),
        ]
        contexts, audit = runner.build_foreground_sequences(events, self.History())
        self.assertEqual(contexts[1][0], ("app:A",))
        self.assertEqual(contexts[3][0], ("app:A", "app:B"))
        self.assertEqual(contexts[5][0], ("app:C",))
        self.assertEqual(audit["resolution_updates_skipped"], 1)
        self.assertEqual(audit["observation_boundaries"], 1)


if __name__ == "__main__":
    unittest.main()
