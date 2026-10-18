"""Fusion normalization, endpoint and selection invariants."""
import unittest

from compare_temporal_fusion import fuse_scores, select_weight


class FusionTests(unittest.TestCase):
    def test_endpoints_exclude_zero_weight_branch_vocabulary(self):
        self.assertEqual(fuse_scores({'a': 10.}, {'b': 1.}, 0.), {'a': 1.})
        self.assertEqual(fuse_scores({'a': 10.}, {'b': 1.}, 1.), {'b': 1.})

    def test_branches_are_individually_normalized(self):
        result = fuse_scores({'a': 90., 'b': 10.}, {'a': .1, 'c': .9}, .5)
        self.assertAlmostEqual(result['a'], .5)
        self.assertAlmostEqual(result['b'], .05)
        self.assertAlmostEqual(result['c'], .45)

    def test_mixture_does_not_reintroduce_a_third_statistics_branch(self):
        result = fuse_scores({'time-stat': 1.}, {'learned-only': 1.}, .3)
        self.assertEqual(set(result), {'time-stat', 'learned-only'})
        self.assertAlmostEqual(result['time-stat'], .7)
        self.assertAlmostEqual(result['learned-only'], .3)

    def test_invalid_weights_or_scores_rejected(self):
        for weight in (-.1, 1.1, float('nan')):
            with self.assertRaises(ValueError): fuse_scores({'a': 1.}, {'b': 1.}, weight)
        with self.assertRaises(ValueError): fuse_scores({'a': -1.}, {'b': 1.}, .5)

    def test_selector_uses_validation_and_prefers_first_on_ties(self):
        metric = {g: {'top4_hits': 1, 'top1_hits': 1} for g in ('original', 'expanded', 'all')}
        a = {'learning_weight': 0., 'validation': metric, 'new_test': 0}
        b = {'learning_weight': .9, 'validation': metric, 'new_test': 100}
        self.assertIs(select_weight([a, b]), a)


if __name__ == '__main__':
    unittest.main()
