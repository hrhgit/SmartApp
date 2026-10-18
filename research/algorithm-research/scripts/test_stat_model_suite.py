import math
import unittest

import stat_model_suite as suite


def record(origin, label, input_time, outcome_time, event, app='app'):
    return {'origin': origin, 'label': label, 'timestamp': input_time,
            'outcome_timestamp': outcome_time, 'outcome_id': event, 'app': app,
            'x': {f'current:{origin}': 1.0}}


def query(origin='A', now=1000, labels=('B', 'C', 'D'), **extra):
    return {'origin': origin, 'timestamp': now, 'labels': list(labels),
            'app': 'app', 'x': {f'current:{origin}': 1.0}, **extra}


def model(model_id):
    return suite.build(next(c for c in suite.configurations() if c['id'] == model_id))


def ordered(scores):
    return sorted(scores, key=lambda key: (-scores[key], key))


class StatisticalModelTests(unittest.TestCase):
    def test_grid_is_unique_finite_and_every_configuration_is_online(self):
        configs = suite.configurations()
        self.assertTrue(80 <= len(configs) <= 200)
        self.assertEqual(len(configs), len({c['id'] for c in configs}))
        for config in configs:
            with self.subTest(config=config['id']):
                m = suite.build(config)
                self.assertEqual(m.scores(query()), {})
                m.learn(record('A', 'B', 1, 2, 'one'))
                scores = m.scores(query(candidate_last_seen={'B': 2}))
                self.assertIn('B', scores)
                self.assertTrue(all(math.isfinite(v) for v in scores.values()))
                m.learn(record('A', 'NEW', 3, 4, 'two'))
                self.assertIn('NEW', m.scores(query(labels=('B', 'NEW'),
                                                   candidate_last_seen={'B': 2, 'NEW': 4})))

    def test_frequency_count_before_latest_outcome_tie(self):
        m = model('stat_frequency_conditional')
        m.learn(record('A', 'B', 0, 10, 'old'))
        m.learn(record('A', 'C', 0, 20, 'new'))
        self.assertEqual(ordered(m.scores(query())), ['C', 'B'])
        m.learn(record('A', 'B', 0, 5, 'oldest'))
        self.assertEqual(ordered(m.scores(query())), ['B', 'C'])

    def test_global_deduplicates_actual_outcome_across_origins(self):
        m = model('stat_frequency_global')
        self.assertTrue(m.learn(record('A', 'B', 0, 10, 'same')))
        self.assertFalse(m.learn(record('A', 'B', 0, 10, 'same')))
        self.assertTrue(m.learn(record('Z', 'B', 0, 10, 'same')))
        self.assertEqual(len(m.global_records), 1)
        self.assertEqual(len(m.by_app['app']), 1)
        self.assertEqual(m.support_counts('A'), {'B': 1})
        self.assertEqual(m.support_counts('Z'), {'B': 1})

    def test_unseen_origin_cannot_inherit_conditional_evidence(self):
        m = model('stat_frequency_conditional')
        m.learn(record('A', 'B', 0, 10, 'one'))
        self.assertEqual(m.scores(query(origin='Z')), {})
        self.assertEqual(m.support_counts('Z'), {})

    def test_maturity_guard_rejects_equal_or_future_outcomes(self):
        for config in suite.configurations():
            m = suite.build(config)
            m.learn(record('A', 'B', 0, 10, 'one'))
            for now in (9, 10):
                with self.assertRaises(ValueError):
                    m.scores(query(now=now))

    def test_recent_memory_sorts_outcomes_not_learning_order(self):
        m = model('stat_recent_memory_1')
        m.learn(record('A', 'C', 0, 20, 'new'))
        m.learn(record('A', 'B', 0, 10, 'old'))
        self.assertEqual(ordered(m.scores(query())), ['C'])

    def test_time_window_and_decay_use_elapsed_time(self):
        m = model('stat_time_window_1h')
        m.learn(record('A', 'B', 0, 10, 'old'))
        m.learn(record('A', 'C', 0, suite.HOUR_MS, 'new'))
        self.assertEqual(ordered(m.scores(query(now=1.5 * suite.HOUR_MS))), ['C'])
        decay = model('stat_time_decay_1h')
        decay.learn(record('A', 'B', 0, 10, 'old'))
        decay.learn(record('A', 'C', 0, suite.HOUR_MS + 10, 'new'))
        scores = decay.scores(query(now=2 * suite.HOUR_MS + 10))
        self.assertAlmostEqual(scores['C'] / scores['B'], 2, places=7)

    def test_query_labels_limit_output_without_filtering_evidence(self):
        m = model('stat_frequency_conditional')
        m.learn(record('A', 'B', 0, 10, 'one'))
        m.learn(record('A', 'C', 0, 20, 'two'))
        self.assertEqual(set(m.scores(query(labels=('B',)))), {'B'})
        self.assertEqual(m.support_counts('A'), {'B': 1, 'C': 1})

    def test_decay_keeps_relative_evidence_after_a_long_inactive_gap(self):
        m = model('stat_time_decay_0.0333333h')
        m.learn(record('A', 'B', 0, 10, 'old'))
        m.learn(record('A', 'C', 0, 100, 'new'))
        before = m.scores(query(now=1000))
        after = m.scores(query(now=1000 * suite.HOUR_MS))
        self.assertEqual(before, after)
        self.assertEqual(ordered(after), ['C', 'B'])

    def test_candidate_recency_rejects_future_input_and_uses_current_candidates(self):
        m = model('stat_candidate_recency_observed')
        self.assertEqual(ordered(m.scores(query(candidate_last_seen={'C': 900, 'B': 1, 'X': 999}))),
                         ['C', 'B'])
        with self.assertRaises(ValueError):
            m.scores(query(candidate_last_seen={'B': 1000}))

    def test_background_sequence_truth_do_not_affect_scores(self):
        for config in suite.configurations():
            m = suite.build(config)
            m.learn(record('A', 'B', 0, 10, 'one'))
            a = query(candidate_last_seen={'B': 10})
            b = {**a, 'truth': 'C', 'background': {'X': 99}, 'previous_sequence': ('X', 'Y'),
                 'diagnosis_after_outcome': {'support_count': 5000}}
            self.assertEqual(m.scores(a), m.scores(b))


if __name__ == '__main__':
    unittest.main()
