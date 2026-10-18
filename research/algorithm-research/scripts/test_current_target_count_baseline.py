from copy import deepcopy
import unittest
import tempfile
from pathlib import Path
from unittest.mock import patch
from current_target_count_baseline import CurrentTargetCounts, replay
from compare_current_count_baseline import report, DISPLAY


class CountBaselineTests(unittest.TestCase):
    def test_conditional_counts_dominate_any_global_frequency(self):
        m = CurrentTargetCounts()
        m.learn('a', 'rare', 'r')
        for i in range(30):
            m.learn('other', 'popular', str(i))
        self.assertEqual(m.ranked('a'), ['rare', 'popular'])

    def test_global_ties_and_cold_start_without_output_gate(self):
        m = CurrentTargetCounts()
        m.learn('a', 'b', 'b1'); m.learn('a', 'c', 'c1')
        m.learn('other', 'c', 'c2'); m.learn('other', 'new', 'n1')
        self.assertEqual(m.ranked('a'), ['c', 'b', 'new'])
        self.assertEqual(m.ranked('unseen'), ['c', 'b', 'new'])

    def test_independent_global_and_per_origin_dedup(self):
        m = CurrentTargetCounts()
        self.assertTrue(m.learn('a', 'b', 'one'))
        self.assertFalse(m.learn('a', 'b', 'one'))
        self.assertTrue(m.learn('z', 'b', 'one'))
        self.assertEqual(m.global_counts, {'b': 1})
        self.assertEqual(m.conditional['a'], {'b': 1})
        self.assertEqual(m.conditional['z'], {'b': 1})

    def test_stable_ties_not_update_recency(self):
        m = CurrentTargetCounts()
        m.learn('a', 'b', 'one'); m.learn('a', 'c', 'two')
        self.assertEqual(m.ranked('a'), ['b', 'c'])

    def test_conflicting_identity_does_not_corrupt_state(self):
        m = CurrentTargetCounts(); m.learn('a', 'b', 'one')
        before = deepcopy(m.state())
        with self.assertRaises(ValueError):
            m.learn('z', 'different', 'one')
        self.assertEqual(m.state(), before)

    def test_restore_and_prediction_are_read_only(self):
        m = CurrentTargetCounts(); m.learn('a', 'b', 'one')
        before = deepcopy(m.state()); m.ranked('unseen')
        self.assertEqual(m.state(), before)
        restored = CurrentTargetCounts.restore(before)
        self.assertEqual(restored.state(), before)
        self.assertEqual(restored.ranked('a'), m.ranked('a'))

    def test_replay_strict_maturity_and_frozen_feedback_cap(self):
        observations = [{'position': i, 'index': index, 'timestamp': index*100,
                         'origin': 'a', 'annotation': {'identity': {'id': 'a', 'bundleID': 'owner'}}}
                        for i, index in enumerate((10, 20, 30))]
        rows = [{'position': i, 'input_index': q['index']} for i, q in enumerate(observations)]
        stream = [{'ready_index': ready, 'record': {'origin': 'a', 'label': label,
                   'outcome_id': label, 'outcome_timestamp': ready*100-1}}
                  for ready, label in ((9, 'old'), (10, 'equal'), (25, 'late'))]
        with patch('current_target_count_baseline.causal_stream', return_value=(stream, {})):
            outputs, state, _ = replay({'observations': observations}, rows)
            frozen, frozen_state, audit = replay({'observations': observations}, rows, 9)
        self.assertEqual(outputs[0]['predictions'], ['old'])
        self.assertIn('equal', outputs[1]['predictions'])
        self.assertNotIn('late', outputs[1]['predictions'])
        self.assertIn('late', outputs[2]['predictions'])
        self.assertTrue(all(v['predictions'] == ['old'] for v in frozen.values()))
        self.assertEqual(len({r['learned_state_sha256'] for r in audit['cutoffs']}), 1)
        self.assertEqual(frozen_state['global_counts'], {'old': 1})

    def test_report_uses_no_context_as_main_reference(self):
        metrics = {'online': {name: {g: {'n': 1, 'rank_score': {'mean': 1.},
                                        'exact': {'top1_hits': 1, 'top4_hits': 1}}
                                    for g in ('all', 'evidence_total')} for _, name in DISPLAY}}
        deltas = {name: {g: {'composite_points': 0., 'exact_top1_net_hits': 0, 'exact_top4_net_hits': 0,
                             'positive_windows': 0, 'tied_windows': 92, 'negative_windows': 0}
                         for g in ('all', 'evidence_total')} for _, name in DISPLAY}
        with tempfile.TemporaryDirectory() as directory:
            report(Path(directory), metrics, deltas, {'sizes': {'all': 1, 'evidence_total': 1}})
            text = (Path(directory)/'report-private.md').read_text()
        self.assertIn('相对无情景统计', text)
        self.assertIn('无情景统计（主基准）', text)
        self.assertIn('（补充）', text)


if __name__ == '__main__':
    unittest.main()
