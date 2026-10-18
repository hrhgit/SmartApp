"""Tests for weekday-only effect-diagnosis aggregation."""
import unittest

import analyze_weekday_context_effects as weekday


def case(position, weekday_name, helped=False, harmed=False):
    result = {
        'position': position, 'weekday': weekday_name, 'hour3': '12:00–15:00',
        'split': 'test', 'evidence_grade': 'original', 'current_kind': 'page', 'next_kind': 'window',
        'helped_any': helped, 'harmed_any': harmed,
    }
    for name in weekday.HELP_METRICS + weekday.HARM_METRICS:
        result[name] = False
    result['accepted_top1_rescue'] = helped
    result['accepted_top4_lost'] = harmed
    return result


class WeekdayEffectDiagnosticTests(unittest.TestCase):
    def test_direction_summary_does_not_cancel_opposite_effects(self):
        cases = [case(1, '星期一', helped=True), case(2, '星期一', harmed=True)]
        helped = weekday.summarize_direction(cases, weekday.HELP_METRICS, 'helped_any')
        harmed = weekday.summarize_direction(cases, weekday.HARM_METRICS, 'harmed_any')
        self.assertEqual(helped['any_n'], 1)
        self.assertEqual(helped['metrics']['accepted_top1_rescue'], 1)
        self.assertEqual(harmed['any_n'], 1)
        self.assertEqual(harmed['metrics']['accepted_top4_lost'], 1)

    def test_weekday_denominator_is_not_limited_to_changed_rows(self):
        cases = [case(1, '星期一', helped=True), case(2, '星期一'), case(3, '星期二')]
        self.assertEqual(weekday.weekday_denominators(cases, 'helped_any'),
                         [{'weekday': '星期一', 'affected_n': 1, 'eligible_n': 2}])

    def test_context_rows_keep_directional_metric_labels(self):
        values = weekday.directional_contexts([case(1, '星期一', helped=True)],
                                              weekday.HELP_METRICS, 'helped_any')
        self.assertEqual(values[0]['n'], 1)
        self.assertEqual(values[0]['metrics'], {'accepted_top1_rescue': 1})


if __name__ == '__main__':
    unittest.main()
