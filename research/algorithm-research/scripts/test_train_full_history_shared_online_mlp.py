"""Final-fit export guards using only constructed foreground observations."""

from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np

import run_offline_batch_rankers as offline
import run_offline_sequence_shadow as sequence
import shared_model_adapters as adapters
import shared_online_path_mlp as online
import train_full_history_shared_online_mlp as fit
from test_shared_model_input import fixture


class FullHistoryExportTests(unittest.TestCase):
    def test_complete_sample_materializes_with_asof_target_metadata(self):
        events, history, _, row, _ = fixture()
        sample = {"id": row.sample_id, "anchorEventID": events[3]["id"],
                  "outcomes": [{"eventID": events[5]["id"],
                                "targetID": row.truth.key}]}
        member = {"sampleID": row.sample_id,
                  "anchorEventID": events[3]["id"],
                  "inputEventIndex": 3, "completedEventIndex": 5}
        with TemporaryDirectory() as directory:
            (Path(directory) / "navigation-samples.jsonl").write_text(
                json.dumps(sample) + "\n", encoding="utf-8")
            observed = fit.materialize_all(Path(directory), events, [member], history)
        self.assertEqual(len(observed), 1)
        self.assertEqual(observed[0].current.key, row.current.key)
        self.assertEqual(observed[0].truth.key, row.truth.key)
        self.assertEqual(observed[0].input_event_index, 3)
        self.assertEqual(observed[0].completed_event_index, 5)
        self.assertFalse(observed[0].has_prior_exact_action)
        self.assertEqual(observed[0].split, "full_history")

    def test_exported_weights_and_adam_state_reproduce_post_training_score(self):
        try:
            sequence.require_torch()
        except RuntimeError:
            self.skipTest("Private research PyTorch runtime unavailable")
        _, _, _, row, builder = fixture()
        cutoff = builder.at(row)
        outcome = builder.matured(cutoff, row, learning_event_index=6)
        example = adapters.adapt_feedback_training(
            cutoff, outcome, family="candidate_mlp", learning_event_index=6,
            gate=adapters.MaturedFeedbackGate())
        model = online.SharedOnlineSetMLP(fit.HIDDEN, fit.LEARNING_RATE,
                                          negatives=fit.NEGATIVES, seed=fit.SEED)
        model.learn(example.prediction, example.labels.useful_indices)
        exported = fit.export_weights(
            model, identity_sha256="a" * 64, trained_through_event_id="e5",
            training_audit={"matured_unique_arrivals": 1})
        self.assertEqual(exported["format"], "online-path-mlp-f32-v1")
        self.assertEqual(len(exported["numericFields"]), 31)
        self.assertEqual(len(exported["layer1Weights"]), 16)
        self.assertEqual(len(exported["layer1Weights"][0]), 543)
        self.assertEqual(exported["identityKeySHA256"], "a" * 64)
        for name in ("layer1Weights", "layer1Bias", "layer2Weights", "layer2Bias"):
            self.assertEqual(exported["optimizerState"][name]["step"], 1)
        prediction = adapters.adapt(cutoff, family="candidate_mlp")
        matrix = model._candidate_matrix(prediction).numpy()
        original = model.model(model.torch.from_numpy(matrix)).detach().numpy().reshape(-1)
        restored = fit.score_exported(exported, matrix).reshape(-1)
        np.testing.assert_allclose(restored, original, atol=1e-4, rtol=1e-5)


if __name__ == "__main__":
    unittest.main()
