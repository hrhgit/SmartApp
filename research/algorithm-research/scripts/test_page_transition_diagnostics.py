import unittest

import diagnose_complete_seen_unit_transitions as unit_diagnostic
import page_transition_diagnostics as diagnostics


def frozen_row(sample, current='page', outgoing=0, next_kind='page', hit1=False, hit4=False, candidate=True, supported=False):
    return {
        'sample_id': sample,
        'display_only': {'current_identity_level': current},
        'pre_prediction': {
            'current_key': ('page:'+sample if current == 'page' else 'app:'+sample),
            'historical_outcomes': [
                {'outcome_event_id': f'{sample}:{i}', 'context': [('page:'+sample if current == 'page' else 'app:'+sample)]}
                for i in range(outgoing)
            ],
        },
        'observed_outcome': {'metadata': {'kind': next_kind}},
        'diagnosis_after_outcome': {'candidate_present': candidate, 'eligible': supported},
        'evaluation': {'hit1': hit1, 'hit4': hit4},
    }


class FixedTestCohortTests(unittest.TestCase):
    def test_page_source_membership_is_prospective_and_keeps_fixed_population(self):
        rows = [
            frozen_row('page:new', outgoing=0, hit4=False),
            frozen_row('page:seen', outgoing=1, hit1=True, hit4=True),
            frozen_row('page:many', outgoing=4, next_kind='window', hit4=True, supported=True),
            frozen_row('app:seen', current='app', outgoing=9, hit1=True, hit4=True),
        ]
        groups = diagnostics.fixed_test_cohorts(rows)
        self.assertEqual([row['sample_id'] for row in groups['all_fixed_test']], [row['sample_id'] for row in rows])
        self.assertEqual({row['sample_id'] for row in groups['current_page_with_mature_outgoing']},
                         {'page:seen', 'page:many'})
        self.assertEqual({row['sample_id'] for row in groups['current_page_to_page_with_mature_outgoing']},
                         {'page:seen'})
        self.assertEqual({row['sample_id'] for row in groups['current_page_with_realized_transition_support']},
                         {'page:many'})
        metric = diagnostics.frozen_test_metrics(groups['current_page_with_mature_outgoing'], len(rows))
        self.assertEqual((metric['n'], metric['top1_hits'], metric['top4_hits'], metric['candidate_missing_n']),
                         (2, 1, 2, 0))

    def test_source_history_uses_strict_pre_cutoff_arrivals_and_maturity(self):
        data = {'observations': [
            {'position': 0, 'index': 10, 'origin': 'page:p', 'target': {'id': 'page:p', 'kind': 'page'}},
            {'position': 1, 'index': 20, 'origin': 'page:p', 'target': {'id': 'page:p', 'kind': 'page'}},
            {'position': 2, 'index': 30, 'origin': 'page:p', 'target': {'id': 'page:p', 'kind': 'page'}},
        ]}
        rows = [
            {'position': 1, 'truth_metadata': {'kind': 'page'}},
            {'position': 2, 'truth_metadata': {'kind': 'page'}},
        ]
        training = [
            {'position': 0, 'ready_index': 15, 'head': 0, 'label': 'page:a', 'outcome_id': 'a'},
            # Equal maturity and cutoff must not leak into position 2.
            {'position': 1, 'ready_index': 30, 'head': 0, 'label': 'page:b', 'outcome_id': 'b'},
        ]
        history = diagnostics.source_history(data, rows, training)
        self.assertEqual(history[1]['prior_current_page_arrivals'], 1)
        self.assertEqual(history[1]['mature_outgoing_count'], 1)
        self.assertEqual(history[2]['prior_current_page_arrivals'], 2)
        self.assertEqual(history[2]['mature_outgoing_count'], 1)
        self.assertTrue(history[2]['seen_current_page_before'])
        groups = diagnostics.cohorts(rows, history)
        self.assertEqual(len(groups['seen_current_page_with_mature_outgoing']), 2)


class FixedSplitPageCohortTests(unittest.TestCase):
    def test_seen_page_uses_only_arrivals_strictly_before_fixed_boundary(self):
        data = {'observations': [
            {'position': 0, 'index': 9, 'origin': 'page:p',
             'target': {'id': 'page:p', 'kind': 'page'}},
            {'position': 1, 'index': 10, 'origin': 'page:q',
             'target': {'id': 'page:q', 'kind': 'page'}},
            {'position': 2, 'index': 11, 'origin': 'page:p',
             'target': {'id': 'page:p', 'kind': 'page'}},
            {'position': 3, 'index': 12, 'origin': 'page:q',
             'target': {'id': 'page:q', 'kind': 'page'}},
            # A page-shaped origin without exact target identity is not a
            # concrete current-page source.
            {'position': 4, 'index': 13, 'origin': 'page:r',
             'target': {'id': 'app:example', 'kind': 'app'}},
        ]}
        rows = [
            {'position': 1, 'truth_metadata': {'kind': 'page'}},
            {'position': 2, 'truth_metadata': {'kind': 'app'}},
            {'position': 3, 'truth_metadata': {'kind': 'page'}},
            {'position': 4, 'truth_metadata': {'kind': 'page'}},
        ]
        history = diagnostics.fixed_split_page_history(data, rows, boundary_index=10)
        self.assertFalse(history[1]['seen_current_page_before_split'])
        self.assertEqual(history[2]['pre_split_current_page_arrivals'], 1)
        self.assertTrue(history[2]['seen_current_page_before_split'])
        self.assertFalse(history[3]['seen_current_page_before_split'])
        self.assertFalse(history[4]['current_exact_page'])

        groups = diagnostics.fixed_split_page_cohorts(rows, history)
        self.assertEqual([r['position'] for r in groups['all_complete_validation']], [1, 2, 3, 4])
        self.assertEqual([r['position'] for r in groups['current_exact_page_seen_before_split']], [2])
        self.assertEqual([r['position'] for r in groups['seen_current_page_to_page']], [])
        by_page = diagnostics.source_page_rows(rows, history)
        self.assertEqual(list(by_page), ['page:p'])
        self.assertEqual([r['position'] for r in by_page['page:p']], [2])

    def test_fixed_split_rejects_rows_before_boundary(self):
        data = {'observations': [
            {'position': 0, 'index': 9, 'origin': 'page:p',
             'target': {'id': 'page:p', 'kind': 'page'}},
        ]}
        with self.assertRaisesRegex(ValueError, 'precedes the validation boundary'):
            diagnostics.fixed_split_page_history(data, [{'position': 0}], boundary_index=10)


class FixedSplitUnitCohortTests(unittest.TestCase):
    def test_seen_unit_keeps_app_window_class_and_page_in_one_population(self):
        data = {'observations': [
            {'position': 0, 'index': 8, 'origin': 'app:a',
             'target': {'id': 'app:a', 'kind': 'app'}},
            {'position': 1, 'index': 9, 'origin': 'window_class:app:b',
             'target': {'id': 'window:ephemeral-1', 'kind': 'window'}},
            {'position': 2, 'index': 10, 'origin': 'app:a',
             'target': {'id': 'app:a', 'kind': 'app'}},
            {'position': 3, 'index': 11, 'origin': 'window_class:app:b',
             'target': {'id': 'window:ephemeral-2', 'kind': 'window'}},
            {'position': 4, 'index': 12, 'origin': 'page:p',
             'target': {'id': 'page:p', 'kind': 'page'}},
        ]}
        rows = [
            {'position': 2, 'truth_metadata': {'kind': 'page'}},
            {'position': 3, 'truth_metadata': {'kind': 'app'}},
            {'position': 4, 'truth_metadata': {'kind': 'page'}},
        ]
        history = diagnostics.fixed_split_unit_history(data, rows, boundary_index=10)
        self.assertEqual(history[2]['pre_split_current_unit_arrivals'], 1)
        self.assertEqual(history[3]['current_unit_id'], 'window_class:app:b')
        self.assertTrue(history[3]['seen_current_unit_before_split'])
        self.assertFalse(history[4]['seen_current_unit_before_split'])

        groups = diagnostics.fixed_split_unit_cohorts(rows, history)
        self.assertEqual([r['position'] for r in groups['current_unit_seen_before_split']], [2, 3])
        self.assertEqual([r['position'] for r in groups['seen_current_unit_kind_app']], [2])
        self.assertEqual([r['position'] for r in groups['seen_current_unit_kind_window']], [3])
        by_unit = diagnostics.source_unit_rows(rows, history)
        self.assertEqual(set(by_unit), {'app:a', 'window_class:app:b'})

    def test_same_directory_unit_can_have_multiple_observing_applications(self):
        data = {'observations': [
            {'position': 0, 'index': 1, 'origin': 'directory:d',
             'target': {'id': 'directory:d', 'kind': 'directory', 'locator': 'file:///tmp/d',
                        'bundleID': 'com.apple.finder'}},
            {'position': 1, 'index': 2, 'origin': 'directory:d',
             'target': {'id': 'directory:d', 'kind': 'directory', 'locator': 'file:///tmp/d',
                        'bundleID': 'com.jinghaoshe.qspace.pro'}},
            {'position': 2, 'index': 3, 'origin': 'directory:d',
             'target': {'id': 'directory:d', 'kind': 'directory', 'locator': 'file:///tmp/d/',
                        'bundleID': 'com.apple.finder'}},
        ]}
        catalog = unit_diagnostic.unit_catalog(data)
        self.assertEqual(catalog['directory:d']['bundle_ids'],
                         ['com.apple.finder', 'com.jinghaoshe.qspace.pro'])
        self.assertEqual(catalog['directory:d']['display_variants'],
                         ['file:///tmp/d', 'file:///tmp/d/'])

    def test_empty_unit_stratum_has_no_numeric_score(self):
        self.assertEqual(unit_diagnostic.score({'rank_score': {'mean': None}}), '—')


if __name__ == '__main__':
    unittest.main()
