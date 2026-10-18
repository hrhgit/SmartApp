import unittest

import evaluation_contract as scoring
from summarize_online_origin_evidence import (cohort_rows, origin_history_membership,
                                              paired_help_harm)


def target(key):
    return {'id': key, 'kind': 'app', 'bundleID': key.removeprefix('app:')}


def evaluation_data():
    observations = []
    evaluation = []
    for position, (origin, truth) in enumerate((('app:A', 'app:B'), ('app:C', 'app:D'))):
        observations.append({
            'position': position, 'index': position * 10 + 1, 'ready_index': position * 10 + 5,
            'target': target(origin),
        })
        evaluation.append({
            'position': position, 'sample_id': f'sample:{position}', 'truth_action': truth,
            'truth_metadata': target(truth), 'evidence_grade': 'outcome_conditioned_only',
            'split': 'test',
        })
    return scoring.prepare_evaluation({'observations': observations, 'evaluation': evaluation})


class MembershipTests(unittest.TestCase):
    def test_only_mature_same_origin_outcomes_create_membership(self):
        rows = [{'position': 0}, {'position': 1}, {'position': 2}]
        inputs = {
            0: {'position': 0, 'index': 2, 'origin': 'app:A'},
            1: {'position': 1, 'index': 5, 'origin': 'app:B'},
            2: {'position': 2, 'index': 6, 'origin': 'app:A'},
        }
        stream = [
            {'position': 0, 'input_index': 2, 'ready_index': 5,
             'origin': 'app:A', 'outcome_id': 'first'},
            {'position': 1, 'input_index': 5, 'ready_index': 6,
             'origin': 'app:B', 'outcome_id': 'same_cutoff'},
        ]
        ledger = [
            {'position': 0, 'mature_updates_at_cutoff': 0},
            {'position': 1, 'mature_updates_at_cutoff': 0},
            {'position': 2, 'mature_updates_at_cutoff': 1},
        ]
        membership, audit = origin_history_membership(rows, stream, inputs, ledger)
        self.assertFalse(membership[0]['has_origin_history'])
        self.assertFalse(membership[1]['has_origin_history'])
        self.assertTrue(membership[2]['has_origin_history'])
        self.assertEqual(membership[2]['mature_unique_origin_outcomes'], 1)
        self.assertTrue(audit['causal_ledger_checked'])

    def test_membership_does_not_read_truth_or_model_output(self):
        inputs = {0: {'position': 0, 'index': 3, 'origin': 'app:A'}}
        stream = []
        plain = [{'position': 0}]
        contaminated = [{'position': 0, 'truth_action': 'app:future',
                         'prediction': ['app:future'], 'evidence_grade': 'hit'}]
        self.assertEqual(origin_history_membership(plain, stream, inputs)[0],
                         origin_history_membership(contaminated, stream, inputs)[0])


class CohortAndPairTests(unittest.TestCase):
    def test_cohort_is_membership_based_not_evidence_grade_based(self):
        rows = evaluation_data()
        membership = {
            0: {'has_origin_history': False, 'mature_unique_origin_outcomes': 0},
            1: {'has_origin_history': True, 'mature_unique_origin_outcomes': 4},
        }
        cohorts, audit = cohort_rows(rows, membership, 'fixed_test')
        self.assertEqual(len(cohorts['all_legal']), 2)
        self.assertEqual([row['position'] for row in cohorts['origin_history']], [1])
        self.assertEqual(audit['origin_history_coverage'], .5)
        rows[0]['evidence_grade'] = 'different_truth_conditioned_grade'
        self.assertEqual([row['position'] for row in cohort_rows(rows, membership, 'fixed_test')[0]['origin_history']], [1])

    def test_help_harm_has_top2_and_does_not_persist_targets_or_predictions(self):
        rows = evaluation_data()
        membership = {
            0: {'has_origin_history': True, 'mature_unique_origin_outcomes': 1},
            1: {'has_origin_history': True, 'mature_unique_origin_outcomes': 2},
        }
        baseline = {
            0: {'predictions': ['app:X'], 'rankable': ['app:X']},
            1: {'predictions': ['app:Y'], 'rankable': ['app:Y']},
        }
        rnn = {
            0: {'predictions': ['app:B'], 'rankable': ['app:B']},
            1: {'predictions': ['app:D'], 'rankable': ['app:D']},
        }
        summary, cases = paired_help_harm(rows, baseline, rnn, membership)
        self.assertEqual(summary['accepted']['top2']['helped'], 2)
        self.assertEqual(summary['exact']['top2']['helped'], 2)
        self.assertNotIn('truth_action', cases[0])
        self.assertNotIn('predictions', cases[0])


if __name__ == '__main__':
    unittest.main()
