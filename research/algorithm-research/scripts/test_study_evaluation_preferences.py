"""Numerical edge cases for cost sensitivity; target scoring remains shared."""
import unittest

from study_evaluation_preferences import lower_envelope, nondominated


class PreferenceStudyTests(unittest.TestCase):
    def test_crossing_is_exact_not_grid_rounded(self):
        segments = lower_envelope({'a': .2, 'b': .5}, {'a': .6, 'b': .1})
        self.assertEqual([s['winners'] for s in segments], [['a'], ['b']])
        self.assertAlmostEqual(segments[0]['through'], 3/8)
        self.assertEqual(segments[1]['from'], segments[0]['through'])

    def test_crossings_above_winner_do_not_split_its_interval(self):
        segments = lower_envelope({'best': .1, 'a': .3, 'b': .8}, {'best': .1, 'a': .8, 'b': .3})
        self.assertEqual(segments, [{'from': 0., 'through': 1., 'winners': ['best']}])

    def test_identical_best_lines_preserve_tie(self):
        segments = lower_envelope({'a': .1, 'b': .1, 'c': .9}, {'a': .5, 'b': .5, 'c': .9})
        self.assertEqual(segments[0]['winners'], ['a', 'b'])
        self.assertEqual(len(segments), 1)

    def test_pareto_does_not_pick_one_tradeoff_or_drop_duplicates(self):
        self.assertEqual(nondominated({'a': [.1, .5], 'b': [.5, .1], 'copy': [.1, .5], 'worse': [.6, .6]}),
                         ['a', 'b', 'copy'])


if __name__ == '__main__':
    unittest.main()
