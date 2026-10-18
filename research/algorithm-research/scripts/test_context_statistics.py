"""Synthetic invariants for conditional statistical voting and tuning."""
from copy import deepcopy
from datetime import datetime
import unittest
from zoneinfo import ZoneInfo

import numpy as np

from context_statistics import (BackgroundIndex, StatisticalMemory, channel_values,
    configurations, prepare_queries, select_validation)
from test_multibackground_model import query
from test_temporal_context import record, target
from semantic_annotations import annotate_target


def q(position, current='app:A', background='app:C'):
    value = query(position, current, background)
    value.update(index=position*10, background_through_index=position*10,
        timestamp=position*10000, inventory_through_index=position*10,
        inventory_state={'running_apps': {'target:'+background: 1.}},
        inventory_quality={'running_apps': 1.})
    for profile in value['backgrounds'].values():
        profile['available_span_seconds'] = 300
        profile['requested_span_seconds'] = 300
    return value


class StatisticsTests(unittest.TestCase):
    def example(self):
        inputs = [q(0, background='app:C'), q(1, background='app:D'), q(2, background='app:C')]
        background = BackgroundIndex(inputs)
        memory = StatisticalMemory()
        memory.learn(inputs[0], record(0, inputs[0], 'app:Z'), 1)
        memory.learn(inputs[1], record(1, inputs[1], 'app:Y'), 11)
        prepared = memory.prepare(inputs[2], None, background)
        return inputs, background, memory, prepared

    def test_default_predict_matches_original_statistics(self):
        inputs, _, memory, prepared = self.example()
        config = configurations()[0]
        scores = memory.hierarchy.scores(inputs[2]['target'])
        actual = prepared.scores(config)
        self.assertEqual(set(scores), set(actual))
        for label in scores: self.assertAlmostEqual(scores[label], actual[label], places=14)
        self.assertEqual(memory.hierarchy.ranked(inputs[2]['target']), prepared.predict(config)['predictions'])

    def test_both_context_kinds_favor_matching_historical_background(self):
        _, _, _, prepared = self.example()
        for profile in ('time_combined_rich', 'inventory_apps_windows_rich'):
            config = {**configurations()[0], 'profile': profile, 'floor': .25,
                      'semantic_strength': 0., 'global_strength': 0., 'recency_mix': 0.}
            self.assertEqual(prepared.predict(config)['predictions'][0], 'app:Z')
            self.assertAlmostEqual(prepared.components(config)[0].sum(), 2.)

    def test_disabled_context_is_identical_to_current_only(self):
        _, _, _, prepared = self.example()
        config = configurations()[0]
        expected = prepared.predict(config)
        for profile in ('time_recent_count', 'inventory_all_rich'):
            self.assertEqual(expected, prepared.predict({**config, 'profile': profile, 'floor': 1.}))

    def test_missing_inventory_is_neutral_and_empty_is_observed(self):
        a, b = q(0), q(1)
        a['inventory_state'] = {}; a['inventory_quality'] = {}
        b['inventory_state'] = {'running_apps': {}}
        index = BackgroundIndex([a, b])
        config = {**configurations()[0], 'profile': 'inventory_apps_windows_rich', 'floor': .25}
        self.assertEqual(index.factors(0, np.array([1]), config)[0], 1.)
        values, reliability = channel_values(b, 'inventory/running_apps/rich')
        self.assertEqual(values, {'observed_empty': 1.})
        self.assertEqual(reliability, 1.)

    def test_clock_profiles_use_only_the_cutoff_timestamp_and_can_skip_backgrounds(self):
        zone = ZoneInfo('America/Los_Angeles')
        def timestamp(year, month, day, hour):
            return int(datetime(year, month, day, hour, tzinfo=zone).timestamp() * 1000)
        same_slot, same_hour_other_day, other_hour = q(0), q(1), q(2)
        same_slot['timestamp'] = timestamp(2026, 9, 21, 9)       # Monday, 09:00
        same_hour_other_day['timestamp'] = timestamp(2026, 9, 22, 9)
        other_hour['timestamp'] = timestamp(2026, 9, 28, 18)
        for value in (same_slot, same_hour_other_day, other_hour):
            value.pop('backgrounds')
            value.pop('inventory_state')
            value.pop('inventory_quality')
        index = BackgroundIndex([same_slot, same_hour_other_day, other_hour],
                                profiles=('clock_hour3_weekday',))
        similarity, reliability = index.similarities(0, 'clock_hour3_weekday')
        self.assertAlmostEqual(similarity[0], 1.)
        self.assertAlmostEqual(similarity[1], .5)
        self.assertGreater(similarity[2], .5)  # same Monday, but a different hour
        self.assertLess(similarity[2], 1.)
        np.testing.assert_array_equal(reliability, np.ones(3))
        hour_similarity, _ = BackgroundIndex([same_slot, same_hour_other_day, other_hour],
                                              profiles=('clock_hour3',)).similarities(0, 'clock_hour3')
        self.assertAlmostEqual(hour_similarity[1], 1.)
        self.assertLess(hour_similarity[2], .2)
        altered = deepcopy([same_slot, same_hour_other_day, other_hour])
        altered[1]['truth'] = 'not-a-feature'
        changed, _ = BackgroundIndex(altered, profiles=('clock_hour3_weekday',)).similarities(
            0, 'clock_hour3_weekday')
        np.testing.assert_allclose(similarity, changed, atol=1e-14)

    def test_future_context_and_ordered_sequence_rejected(self):
        value = q(0); value['inventory_through_index'] += 1
        with self.assertRaises(ValueError): BackgroundIndex([value])
        value = q(0); value['current_sequence'].insert(0, 'app:PREVIOUS')
        with self.assertRaises(ValueError): BackgroundIndex([value])

    def test_future_appended_inputs_do_not_change_similarity(self):
        inputs, index, _, _ = self.example()
        expected = index.similarities(2, 'time_combined_rich')
        changed = deepcopy(inputs)
        changed[0]['truth_action'] = 'SECRET'; changed[1]['x'] = {'future': 1000}
        extended = BackgroundIndex(changed+[q(3, background='app:FUTURE')])
        actual = extended.similarities(2, 'time_combined_rich')
        np.testing.assert_allclose(expected[0], actual[0][:3], atol=1e-14)
        np.testing.assert_array_equal(expected[1], actual[1][:3])

    def test_duplicate_behavior_does_not_add_votes(self):
        inputs, _, memory, _ = self.example()
        memory.learn(inputs[0], record(0, inputs[0], 'app:Z'), 1)
        self.assertEqual(len(memory.records), 2)

    def test_service_fold_deduplicates_the_same_actual_outcome(self):
        inputs = [q(0, 'page:a'), q(1, 'page:b')]
        for i, value in enumerate(inputs):
            t = target(value['origin'], 'page')
            t['locator'] = 'https://chatgpt.com/c/00000000-0000-0000-0000-00000000000'+str(i)
            value.update(target=t, annotation=annotate_target(t))
        memory = StatisticalMemory()
        for value in inputs:
            memory.learn(value, record(100, value, 'app:Z'), value['index']+1)
        self.assertEqual(len(memory.training), 2)
        self.assertEqual(len(memory.records), 1)
        self.assertEqual(len(memory.hierarchy.base.records[0]), 1)

    def test_strict_maturity_boundary(self):
        inputs = [q(0), q(1)]
        background = BackgroundIndex(inputs)
        training = [{'position': 0, 'ready_index': 10, 'record': record(0, inputs[0], 'app:Z')}]
        prepared = prepare_queries(inputs, training, [{'position': 1}], {1: None}, background)
        self.assertEqual(prepared[1].predict(configurations()[0])['predictions'], [])
        memory = StatisticalMemory(); memory.learn(inputs[0], training[0]['record'], 10)
        with self.assertRaises(ValueError): memory.prepare(inputs[1], None, background)

    def test_global_popularity_deduplicates_overlapping_outcomes(self):
        inputs = [q(0, 'app:A'), q(1, 'app:B'), q(2, 'app:C'), q(3, 'app:D')]
        memory = StatisticalMemory()
        memory.learn(inputs[0], record(0, inputs[0], 'app:X'), 1)
        memory.learn(inputs[1], record(0, inputs[1], 'app:X'), 11)
        memory.learn(inputs[2], record(2, inputs[2], 'app:Y'), 21)
        prepared = memory.prepare(inputs[3], None, BackgroundIndex(inputs))
        global_counts = prepared.components(configurations()[0])[2]
        self.assertEqual(dict(zip(prepared.labels, global_counts)), {'app:X': .5, 'app:Y': .5})

    def test_grid_equal_budgets_and_selector_ignores_later_results(self):
        configs = configurations()
        self.assertEqual(sum(c['mode'] == 'time' for c in configs), 972)
        self.assertEqual(sum(c['mode'] == 'inventory' for c in configs), 972)
        self.assertEqual(sum(c['mode'] == 'none' for c in configs), 27)
        scores = {g: {'top4_hits': 1, 'top1_hits': 1} for g in ('original', 'expanded', 'all')}
        a = {'config': configs[0], 'validation': scores, 'new_test': 0}
        b = {'config': configs[1], 'validation': scores, 'new_test': 100}
        self.assertIs(select_validation([a, b], 'none'), a)


if __name__ == '__main__':
    unittest.main()
