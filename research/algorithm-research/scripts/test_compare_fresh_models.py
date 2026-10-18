import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from compare_fresh_models import Experiment, save_object, current_only_similarity


class FreshComparisonTests(unittest.TestCase):
    def test_removed_history_is_not_a_perfect_context_match(self):
        q=SimpleNamespace(history={},foreground={'app:a':1.},background={},source_quality={})
        self.assertEqual(current_only_similarity(q,q)[2],0.)
        self.assertEqual(current_only_similarity(q,q)[3],1.)

    def experiment(self, directory):
        exp = Experiment.__new__(Experiment)
        exp.out = Path(directory)
        exp.positions = {10, 11}
        exp.outputs = {}; exp.models = {}
        return exp

    def test_partial_coverage_cannot_enter_common_leaderboard(self):
        with tempfile.TemporaryDirectory() as directory:
            exp = self.experiment(directory)
            with self.assertRaisesRegex(ValueError, 'every common cutoff'):
                exp.add('partial', {10: {'predictions': ['app:a']}}, 'test')
            self.assertEqual(exp.models, {})

    def test_abstention_retains_validation_member(self):
        with tempfile.TemporaryDirectory() as directory:
            exp = self.experiment(directory)
            exp.add('selective', {'10': {'predictions': []}, '11': {'predictions': ['app:a']}}, 'test')
            self.assertEqual(set(exp.outputs['selective']), {10, 11})
            self.assertEqual(exp.outputs['selective'][10]['predictions'], [])

    def test_duplicate_model_or_oversize_menu_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            exp = self.experiment(directory)
            values = {i: {'predictions': []} for i in (10, 11)}
            exp.add('one', values, 'test')
            with self.assertRaises(ValueError): exp.add('one', values, 'test')
            values[10]['predictions'] = list('abcde')
            with self.assertRaises(ValueError): exp.add('too_long', values, 'test')

    def test_persisted_state_reload_parity(self):
        with tempfile.TemporaryDirectory() as directory:
            filename = save_object(Path(directory), 'sample', {'count': 3}, lambda x: x['count'])
            self.assertEqual((Path(directory)/filename).stat().st_mode & 0o777, 0o600)


if __name__ == '__main__': unittest.main()
