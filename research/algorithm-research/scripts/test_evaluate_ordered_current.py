import unittest

import evaluate_context_knn as core
import evaluate_ordered_current as model
from test_evaluate_context_knn import observation


class OrderedCurrentTests(unittest.TestCase):
    def test_current_is_length_one_of_the_same_ordered_type(self):
        path = model.OrderedContext(('A', 'B', 'C'))
        self.assertEqual(path.suffix(1), model.OrderedContext(('C',)))
        self.assertEqual(path.suffix(2).targets, ('B', 'C'))
        self.assertNotEqual(model.OrderedContext(('A', 'B')), model.OrderedContext(('B', 'A')))
        with self.assertRaises(ValueError):
            path.suffix(4)

    def test_label_must_mature_before_learning(self):
        counts = model.TransitionCounts(); context = model.OrderedContext(('A',))
        with self.assertRaises(ValueError):
            counts.learn(context, 'event', 'B', 5, ready_index=10, asof_index=10)
        self.assertEqual(counts.distribution(context)[0], {})

    def test_outcomes_deduplicate_within_each_ordered_context(self):
        counts = model.TransitionCounts(); a = model.OrderedContext(('A',)); b = model.OrderedContext(('B',))
        self.assertTrue(counts.learn(a, 'event', 'C', 5, 6, 10))
        self.assertFalse(counts.learn(a, 'event', 'C', 5, 6, 11))
        self.assertTrue(counts.learn(b, 'event', 'C', 5, 6, 12))
        self.assertEqual(counts.distribution(a)[0], {'C': 1})

    def test_prediction_uses_current_but_not_path_background_or_truth(self):
        q = observation(); counts = model.TransitionCounts()
        counts.learn(model.current_context(q), 'event', 'B', 5, 6, 10)
        before = counts.predict(core, q)
        q.history = {'different': 99}; q.history_keys = ('X', 'Y'); q.background = {'browser': {'other': 100}}
        q.truth = 'Z'; q.recency = {'C': 999}
        self.assertEqual(counts.predict(core, q), before)
        q.current = 'X'
        self.assertEqual(counts.predict(core, q)['predictions'], [])

    def test_ties_use_latest_outcome_from_this_origin(self):
        q = observation(); counts = model.TransitionCounts(); context = model.current_context(q)
        counts.learn(context, 'old', 'B', 10, 5, 20)
        counts.learn(context, 'recent', 'C', 15, 6, 20)
        self.assertEqual(counts.predict(core, q)['predictions'][:2], ['C', 'B'])
        self.assertEqual(counts.predict(core, q, 'id_control')['predictions'][:2], ['B', 'C'])
        counts.learn(context, 'older2', 'B', 5, 4, 20)
        self.assertEqual(counts.predict(core, q)['predictions'][0], 'B')

    def test_unknown_candidate_does_not_disappear_from_confidence_denominator(self):
        q = observation(); counts = model.TransitionCounts(); context = model.current_context(q)
        counts.learn(context, 'one', 'B', 10, 5, 20)
        counts.learn(context, 'two', 'not_candidate', 15, 6, 20)
        row = counts.predict(core, q)
        self.assertEqual(row['predictions'], ['B'])
        self.assertEqual(row['top1_historical_fraction'], .5)
        self.assertEqual(row['unavailable_outcome_count'], 1)

    def test_window_context_uses_semantics_not_session_instance(self):
        a, b = observation(), observation()
        a.current = 'window@session1:123'; b.current = 'window@session2:999'
        a.current_app = b.current_app = 'browser'
        self.assertEqual(model.current_context(a), model.current_context(b))
        b.current_app = None
        self.assertIsNone(model.current_context(b))

    def test_replay_does_not_read_unmatured_future_samples(self):
        items = [observation(1), observation(3), observation(4)]
        ledger = [{'target_event_id': str(i)} for i in range(3)]
        rows, _, _ = model.replay(core, items, {0, 1, 2}, ledger, {'0': 2, '1': 4, '2': 5}, [1, 2])
        self.assertEqual(rows[1]['evidence_n'], 0)
        self.assertEqual(rows[2]['evidence_n'], 1)


if __name__ == '__main__':
    unittest.main()
