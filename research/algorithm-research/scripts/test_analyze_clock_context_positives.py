"""Tests for the positive-only clock-context diagnostic helpers."""
from datetime import datetime
import unittest
from zoneinfo import ZoneInfo

import analyze_clock_context_positives as positives


class PositiveClockDiagnosticTests(unittest.TestCase):
    def test_local_context_uses_the_fixed_evaluation_zone(self):
        timestamp = int(datetime(2026, 9, 21, 23, 30,
                                 tzinfo=ZoneInfo('America/Los_Angeles')).timestamp() * 1000)
        self.assertEqual(positives.local_context(timestamp),
                         {'hour3': '21:00–00:00', 'weekday': '星期一'})

    def test_summary_counts_only_strict_positive_effects(self):
        cases = [
            {'position': 1, 'changed_prediction': True, 'positive_any': True,
             'accepted_top1_rescue': False, 'accepted_top4_rescue': True,
             'exact_top1_rescue': False, 'exact_top4_rescue': True,
             'rank_score_improved': True, 'first_action_improved': True,
             'rank_score_delta': .5, 'first_action_loss_delta': 1.},
            {'position': 2, 'changed_prediction': True, 'positive_any': False,
             'accepted_top1_rescue': False, 'accepted_top4_rescue': False,
             'exact_top1_rescue': False, 'exact_top4_rescue': False,
             'rank_score_improved': False, 'first_action_improved': False,
             'rank_score_delta': -.5, 'first_action_loss_delta': -1.},
        ]
        summary = positives.summarize_positive_cases(cases)
        self.assertEqual(summary['eligible_n'], 2)
        self.assertEqual(summary['changed_prediction_n'], 2)
        self.assertEqual(summary['positive_any_n'], 1)
        self.assertEqual(summary['positive_metrics']['accepted_top4_rescue']['n'], 1)
        self.assertEqual(summary['positive_metrics']['accepted_top4_rescue']['positive_rank_score_total'], .5)
        self.assertEqual(summary['positive_metrics']['accepted_top4_rescue']['positive_first_action_loss_total'], 1.)

    def test_time_strata_keeps_the_full_pre_cutoff_denominator(self):
        cases = [
            {'hour3': '09:00–12:00', 'weekday': '星期一', 'accepted_top4_rescue': True},
            {'hour3': '09:00–12:00', 'weekday': '星期一', 'accepted_top4_rescue': False},
            {'hour3': '12:00–15:00', 'weekday': '星期二', 'accepted_top4_rescue': False},
        ]
        strata = positives.positive_time_strata(cases, 'accepted_top4_rescue')
        self.assertEqual(strata['hour3'],
                         [{'value': '09:00–12:00', 'eligible_n': 2, 'positive_n': 1}])
        self.assertEqual(strata['weekday'],
                         [{'value': '星期一', 'eligible_n': 2, 'positive_n': 1}])

    def test_grid_union_deduplicates_an_observed_case_across_configurations(self):
        base = {'position': 3, 'hour3': '12:00–15:00', 'weekday': '星期一',
                'split': 'test', 'evidence_grade': 'original',
                'current_kind': 'page', 'next_kind': 'window'}
        first = {**base, **{metric: False for metric in positives.POSITIVE_METRICS},
                 'accepted_top1_rescue': True}
        second = {**base, **{metric: False for metric in positives.POSITIVE_METRICS},
                  'rank_score_improved': True}
        union = positives.grid_positive_union({'a': [first], 'b': [second]})
        self.assertEqual(union['unique_positive_any_n'], 1)
        self.assertEqual(union['positive_metric_n']['accepted_top1_rescue'], 1)
        self.assertEqual(union['positive_metric_n']['rank_score_improved'], 1)


if __name__ == '__main__':
    unittest.main()
