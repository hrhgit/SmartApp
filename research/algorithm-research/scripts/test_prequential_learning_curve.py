import unittest
import numpy as np
from prequential_learning_curve import warmup_boundary, cap_training, rolling_intervals
from summarize_prequential_learning import window_curve, curve_members
import evaluation_contract as scoring


class PrequentialTests(unittest.TestCase):
    def test_warmup_counts_outcomes_not_repeated_relations(self):
        rows = [{'position': i, 'ready_index': ready, 'outcome_id': outcome}
                for i, (ready, outcome) in enumerate([(9, 'a'), (10, 'a'), (12, 'b'), (12, 'c')])]
        self.assertEqual(warmup_boundary(rows, 2), 12)
        self.assertEqual(len({r['outcome_id'] for r in rows if r['ready_index'] <= 12}), 3)

    def test_freeze_caps_all_heads_not_just_ranker_parameters(self):
        rows = [{'ready_index': i, 'record': {'observed_step': i}} for i in (9, 10, 11)]
        data = {'streams': {'hard_kind': rows}, 'reference_stream': rows,
                'observations': ['causal evolving input'], 'evaluation': ['evaluator-only truth']}
        frozen = cap_training(data, 10)
        self.assertEqual([r['ready_index'] for r in frozen['streams']['hard_kind']], [9, 10])
        self.assertEqual(len(data['streams']['hard_kind']), 3)
        self.assertIs(frozen['observations'], data['observations'])
        self.assertIs(frozen['evaluation'], data['evaluation'])

    def test_windows_cover_final_endpoint_without_extra_evidence_sampling(self):
        windows = rolling_intervals(1006)
        self.assertEqual(windows[0], (0, 100))
        self.assertEqual(windows[1], (10, 110))
        self.assertEqual(windows[-1], (906, 1006))
        self.assertTrue(all(b-a == 100 for a, b in windows))
        self.assertEqual(len(windows), 92)
        self.assertEqual(rolling_intervals(99), [])
        self.assertEqual(rolling_intervals(100), [(0, 100)])

    def test_insufficient_warmup_fails(self):
        with self.assertRaises(ValueError):
            warmup_boundary([], 100)

    def test_evidence_mask_uses_same_window_and_marks_low_n(self):
        values = np.array([[1., 1., 1.], [0., 0., 0.], [.5, 0., 1.]])
        curve = window_curve(values, [True, False, True], [(0, 2), (1, 3)])
        self.assertEqual([r['n'] for r in curve], [1, 1])
        self.assertEqual([r['score'] for r in curve], [100., 50.])
        self.assertEqual([r['top1'] for r in curve], [100., 0.])
        self.assertTrue(all(r['insufficient_evidence'] for r in curve))

    def test_empty_evidence_window_is_missing_not_zero(self):
        curve = window_curve(np.array([[1., 1., 1.]]), [False], [(0, 1)])
        self.assertEqual(curve[0]['n'], 0)
        self.assertIsNone(curve[0]['score'])

    def test_excluded_labels_do_not_take_up_rolling_positions(self):
        rows = [{'position': i, '_scoring': {'contract': scoring.digest(scoring.RULES), 'eligible': good}}
                for i, good in enumerate([True, False, True])]
        legal, evidence = curve_members({'all': rows, 'evidence_total': rows[1:]})
        self.assertEqual([r['position'] for r in legal], [0, 2])
        self.assertEqual(evidence, {2})


if __name__ == '__main__':
    unittest.main()
