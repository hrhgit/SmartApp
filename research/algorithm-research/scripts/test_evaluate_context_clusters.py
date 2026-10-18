import copy
import unittest

import numpy as np

import evaluate_context_knn as core
import evaluate_context_clusters as model
from test_evaluate_context_knn import observation, target


class ClusterTests(unittest.TestCase):
    def test_frequency_ignores_order_but_keeps_repetitions(self):
        a, b = {'app:A': 1.}, {'app:B': 1.}
        self.assertEqual(model.history_tokens([a, a, b], 'frequency'),
            model.history_tokens([a, b, a], 'frequency'))
        self.assertEqual(model.history_tokens([a, a, b], 'frequency')['app:A'], 2.)
        self.assertEqual(model.history_tokens([a, a, b], 'presence')['app:A'], 1.)
        self.assertNotEqual(model.history_tokens([a, a, b], 'ordered'),
            model.history_tokens([a, b, a], 'ordered'))

    def test_same_eight_arrivals_limit_in_all_modes(self):
        history = [{str(i): 1.} for i in range(10)]
        for mode in ['frequency', 'presence', 'ordered']:
            tokens = model.history_tokens(history, mode)
            self.assertEqual(len(tokens), 8)
            self.assertFalse(any(k.endswith(':0') or k == '0' for k in tokens))

    def test_future_features_do_not_change_vocabulary_or_centers(self):
        rows = [{'a': 1.}, {'b': 1.}, {'a': .8}, {'b': .8}, {'a': 1., 'future': 99.}]
        first = model.fit_clusters(rows, [0, 1, 2, 3], 2)
        rows[-1] = {'different_future': 100.}
        second = model.fit_clusters(rows, [0, 1, 2, 3], 2)
        self.assertEqual(first['vocabulary'], ['a', 'b'])
        np.testing.assert_array_equal(first['centers'], second['centers'])
        self.assertLess(first['similarity'][-1], .02)
        self.assertEqual(second['similarity'][-1], 0.)

    def test_missing_source_is_not_observed_empty(self):
        missing = observation()
        empty = observation(bg={'browser': {}}, quality={'browser': 1.})
        a = model.feature_row(missing, [], 'state_only', 0.)
        b = model.feature_row(empty, [], 'state_only', 0.)
        self.assertIn('quality|browser|unobserved', a)
        self.assertIn('quality|browser|observed_empty', b)
        self.assertNotIn('quality|browser|observed_empty', a)
        self.assertNotEqual(a, b)

    def test_only_mature_outcomes_count_once_per_cluster(self):
        items = [observation(i) for i in [1, 2, 9, 10]]
        ledger = [{'target_event_id': k} for k in ['same', 'same', 'not_ready', 'future']]
        counts = model.label_counts(items, {0, 1, 2, 3}, ledger, [0, 0, 0, 0], 3)
        self.assertEqual(counts, {'B': 1})
        self.assertEqual(model.label_counts(items, {0, 1, 2, 3}, ledger, [1, 1, 0, 0], 3), {})

    def test_recent_frequency_uses_actual_outcome_time_and_maturity(self):
        items = [observation(i) for i in [1, 2, 9, 10]]
        items[3].timestamp = 7200000
        ledger = [{'target_event_id': str(i)} for i in range(4)]
        times = {'0': 1000, '1': 7190000, '2': 7191000, '3': 7200100}
        self.assertEqual(model.label_counts(items, {0, 1, 2, 3}, ledger, [0]*4, 3, 1., times), {'B': 1})

    def test_background_idf_uses_only_earlier_inputs(self):
        items = [observation(bg={'browser': {'common': 1., 'rare': 1.}}),
            observation(bg={'browser': {'common': 1.}}),
            observation(bg={'browser': {'future': 1.}})]
        weights = model.background_idf(items, [0, 1])
        self.assertNotIn(('browser', 'future'), weights)
        self.assertGreater(weights['browser', 'rare'], weights['browser', 'common'])
        self.assertEqual(weights['browser', 'common'], 1.)

    def test_candidate_filter_does_not_invent_or_inflate_confidence(self):
        q = observation()
        row = model.rank_frequency(core, q, {'B': 3, 'unavailable': 7})
        self.assertEqual(row['predictions'], ['B'])
        self.assertEqual(row['agreement'], .7)
        self.assertEqual(row['unavailable_outcome_count'], 7)
        self.assertFalse(row['winner_is_displayed_first'])
        self.assertEqual(model.rank_frequency(core, q, {})['predictions'], [])

    def test_gate_and_predictions_do_not_read_query_outcome(self):
        items = [observation(i) for i in [1, 3, 5, 20]]
        ledger = [{'target_event_id': str(i)} for i in range(4)]
        fit = {'assignments': [0]*4, 'similarity': [1.]*4, 'similarity_floor': {0: .5}}
        items[3].source_quality = {'browser': 1.}
        before = model.replay(core, items, {0, 1, 2}, ledger, fit, [3])
        gate = {'min_support': 3, 'min_agreement': .85}
        self.assertTrue(model.accepted(before[3], gate))
        items[3].truth = 'unseen'; items[3].truth_meta = target('unseen')
        after = model.replay(core, items, {0, 1, 2}, ledger, fit, [3])
        self.assertEqual(before, after)

    def test_common_background_does_not_infer_absence(self):
        q = observation(bg={'browser': {'origin:https://example.invalid': .3}}, quality={'browser': .5})
        conditions = model.condition_presence(q, [])
        self.assertIn('background|browser|origin:https://example.invalid', conditions)
        self.assertFalse(any('finder' in condition for condition in conditions))

    def test_window_count_requires_unique_replay_target(self):
        q = observation()
        q.candidates = {'one': {'id': 'one', 'kind': 'window', '_bundle': 'browser'}}
        row = model.rank_frequency(core, q, {'window_class:browser': 5})
        self.assertEqual(row['predictions'], ['one'])
        q.candidates['two'] = {'id': 'two', 'kind': 'window', '_bundle': 'browser'}
        self.assertEqual(model.rank_frequency(core, q, {'window_class:browser': 5})['predictions'], [])


if __name__ == '__main__':
    unittest.main()
