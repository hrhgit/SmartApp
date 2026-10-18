"""Checks for paired attribution and removal of derived information."""
from copy import deepcopy
import unittest

from diagnose_background_contributions import diagnostic_features, inventory_features, paired
from multibackground_model import features
from test_multibackground_model import query, config
from evaluation_contract import prepare_evaluation


class AttributionTests(unittest.TestCase):
    def q(self):
        q = query()
        q.update(inventory_state={'browser': {'target:private-page': 2.}, 'finder': {}},
                 inventory_quality={'browser': .5, 'finder': 1.}, inventory_through_index=q['index'])
        return q

    def transformed(self, variant):
        return diagnostic_features(self.q(), {**config(), 'diagnostic_variant': variant}, features)

    def test_full_is_exact_and_ignores_inventory(self):
        self.assertEqual(self.transformed('full'), features(self.q(), config()))
        self.assertFalse(any('private-page' in k for k in self.transformed('full')))

    def test_no_frequency_leaves_no_frequency_interactions_or_contrast(self):
        values = self.transformed('without_frequency')
        self.assertFalse(any('/count/' in k or '/observed_count' in k or '/log_arrivals' in k
                             or '/recent_vs_history/' in k for k in values))
        self.assertTrue(any('/dwell/' in k for k in values))

    def test_no_current_removes_interactions_and_semantics(self):
        values = self.transformed('without_current_features')
        self.assertTrue(values)
        self.assertTrue(all(k.startswith('background/') for k in values))

    def test_scale_removal_also_removes_cross_scale_contrast(self):
        values = self.transformed('without_recorded_history')
        self.assertFalse(any('/recorded_history/' in k or '/recent_vs_history/' in k for k in values))
        self.assertTrue(any('/recent5m/' in k for k in values))

    def test_empty_and_missing_sources_are_distinct(self):
        values = inventory_features(self.q())
        self.assertIn('inventory/finder/observed_empty', values)
        self.assertIn('inventory/qspace/unavailable', values)
        self.assertNotIn('inventory/finder/unavailable', values)
        self.assertAlmostEqual(values['inventory/browser/token/target:private-page'], .4)

    def test_future_inventory_is_rejected(self):
        q = self.q(); q['inventory_through_index'] += 1
        with self.assertRaises(ValueError): inventory_features(q)

    def test_inventory_only_excludes_temporal_and_private_arbitrary_fields(self):
        q = self.q(); q['truth_action'] = 'secret-truth'; q['sequence'] = ['secret-previous']
        c = {**config(), 'diagnostic_variant': 'inventory_only'}
        values = diagnostic_features(q, c, features)
        self.assertTrue(any('/inventory/' in k for k in values))
        self.assertFalse(any('/recent5m/' in k or 'secret-' in k for k in values))
        self.assertEqual(q['current_sequence'], [q['origin']])

    def test_equal_accuracy_can_contain_opposite_flips(self):
        data = {'observations': [], 'evaluation': []}
        for position, label in enumerate(('a', 'b')):
            data['observations'].append({'position': position, 'index': 10+position,
                'ready_index': 20+position, 'target': {'id': 'current', 'kind': 'app', 'bundleID': 'current'}})
            data['evaluation'].append({'position': position, 'sample_id': str(position),
                'truth_action': label, 'truth_metadata': {'id': label, 'kind': 'app', 'bundleID': label}})
        rows = prepare_evaluation(data)
        ref = {0: {'predictions': ['a']}, 1: {'predictions': ['x']}}
        alt = {0: {'predictions': ['x']}, 1: {'predictions': ['b']}}
        result = paired(rows, ref, alt)
        self.assertEqual(result['top4']['net_hits'], 0)
        self.assertEqual(result['top4']['helped'], 1)
        self.assertEqual(result['top4']['hurt'], 1)


if __name__ == '__main__':
    unittest.main()
