import copy
from types import SimpleNamespace
import unittest

import compare_action_cleaning as comparison


class ActionComparisonTests(unittest.TestCase):
    def test_old_display_normalization_does_not_fill_from_fifth_place(self):
        windows = {'w1': {'kind': 'window', '_bundle': 'browser'},
                   'w2': {'kind': 'window', '_bundle': 'browser'}}
        original = [SimpleNamespace(candidates=windows)]
        pred = {0: {'predictions': ['w1', 'app:browser', 'page:a', 'w2'],
                    'raw_ranking': ['w1', 'app:browser', 'page:a', 'w2', 'page:correct']}}
        before = copy.deepcopy(pred)
        result = comparison.normalize_old_predictions(pred, original)
        self.assertEqual(result[0]['predictions'], ['app:browser', 'page:a'])
        self.assertEqual(result[0]['raw_ranking'], ['app:browser', 'page:a', 'page:correct'])
        self.assertEqual(pred, before)

    def test_concrete_destinations_and_unidentified_windows_are_not_collapsed(self):
        targets = {'page:a': {'kind': 'page', '_bundle': 'browser'},
                   'page:b': {'kind': 'page', '_bundle': 'browser'},
                   'dir:c': {'kind': 'directory', '_bundle': 'finder'},
                   'w?': {'kind': 'window'}}
        pred = {0: {'predictions': list(targets), 'raw_ranking': list(targets)}}
        self.assertEqual(comparison.normalize_old_predictions(pred,
            [SimpleNamespace(candidates=targets)]), pred)

    def test_all_prior_families_and_refined_parameters_retained_without_duplicates(self):
        configs, signatures = comparison.configurations()
        self.assertEqual(len(configs), 210)
        self.assertEqual(len(signatures), len(configs))
        self.assertEqual(len({c['family'] for c in configs}), 27)
        for c in comparison.stats.configurations() + comparison.online.configurations():
            self.assertIn(comparison.config_signature(c), signatures)
        for c in sum(comparison.refinement_components(), []):
            self.assertIn(comparison.config_signature(c), signatures)

    def test_earlier_selection_tie_does_not_consult_late_score(self):
        def row(ident, late):
            return {'config': {'id': ident}, 'training_cleaning': 'raw', 'scores': {'5': {
                'early_supported': {'top4_hits': 20, 'top1_hits': 10, 'mrr4': .6},
                'late_supported': {'top4_hits': late}}}}
        rows = [row('z', 38), row('a', 1)]
        self.assertEqual(min(rows, key=lambda r: comparison.key(r, 'early_supported'))['config']['id'], 'a')


if __name__ == '__main__':
    unittest.main()
