import unittest
from unittest.mock import patch

import review_historical_models as review


def observation(index, timestamp):
    identity = {'id': 'app:current', 'kind': 'app', 'bundleID': 'current'}
    return {'position': 0, 'index': index, 'timestamp': timestamp,
            'origin': 'app:current', 'app': 'current', 'target': identity,
            'annotation': {'identity': identity}}


class ReplayTests(unittest.TestCase):
    def test_maturity_is_strict_and_vocabulary_is_not_future_or_inventory(self):
        q = observation(10, 100)
        def item(label, ready, timestamp):
            return {'ready_index': ready, 'position': 0, 'record': {
                'origin': 'app:current', 'app': 'current', 'timestamp': 1,
                'label': label, 'outcome_id': label, 'outcome_timestamp': timestamp}}
        stream = [item('app:past', 9, 90), item('app:boundary', 10, 100), item('app:future', 11, 110)]
        config = {'id': 'stat_frequency_global', 'family': 'frequency', 'parameters': {'scope': 'global'}}
        with patch.object(review, 'causal_stream', return_value=(stream, {})):
            output, _, audit = review.replay_legacy({'observations': [q]}, [{'position': 0}], config, 'hard_kind')
        self.assertEqual(output[0]['predictions'], ['app:past'])
        self.assertEqual(audit['cutoffs'][0]['mature_updates'], 1)
        self.assertTrue(audit['reload_and_resume_parity'])

    def test_query_allowlist_drops_outcome_and_inventory(self):
        q = observation(10, 100)
        q.update(truth_action='secret', evidence_grade='original', candidates=['secret'],
                 background_state={'tabs': ['secret']}, recent_sequence=['secret'])
        result = review.base_query(q)
        self.assertEqual(set(result), {'origin', 'app', 'timestamp', 'x'})
        self.assertNotIn('secret', str(result))

    def test_same_app_specific_page_is_navigation_but_app_only_is_not(self):
        q = observation(10, 100)
        q['annotation']['identity'].update(id='page:one', kind='page')
        self.assertFalse(review.navigation_label('page:one', q))
        self.assertFalse(review.navigation_label('app:current', q))
        self.assertTrue(review.navigation_label('page:two', q))

    def test_missing_forecasts_not_silently_filled_and_slots_not_deduplicated(self):
        r = review.Review.__new__(review.Review)
        r.by_position = {1: {'sample_id': 'one'}, 2: {'sample_id': 'two'}}
        r.models, r.predictions, r.exceptions = {}, {}, []
        r.window_actions = {'window@one': 'app:a', 'window@two': 'app:a'}
        r.add('test', {1: {'predictions': ['window@one', 'window@two', 'app:b']}},
              'synthetic', 'test', adapt_windows=True)
        self.assertEqual(r.predictions['test'][1]['predictions'], ['app:a', 'app:a', 'app:b'])
        self.assertEqual(r.models['test']['missing_positions'], [2])
        self.assertFalse(r.models['test']['full_common_dataset'])


if __name__ == '__main__':
    unittest.main()
