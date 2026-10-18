import unittest

import evaluate_clock_context as clock


def metrics(top4, top1=0):
    return {group: {'top4_hits': top4, 'top1_hits': top1}
            for group in ('original', 'expanded', 'all')}


class ClockContextTests(unittest.TestCase):
    def test_grid_is_small_and_preserves_the_current_only_core(self):
        base = {'id': 'none', 'mode': 'none', 'profile': None, 'floor': 1., 'power': 1.,
                'semantic_strength': 64., 'global_strength': 1., 'recency_mix': .5,
                'half_life_hours': 24.}
        configs = clock.configurations(base)
        self.assertEqual(len(configs), 6)
        self.assertEqual({c['profile'] for c in configs}, set(clock.PROFILES))
        self.assertTrue(all(c['clock_time_zone'] == 'America/Los_Angeles' for c in configs))
        self.assertTrue(all(c['semantic_strength'] == base['semantic_strength'] for c in configs))

    def test_selection_uses_the_prescribed_early_validation_order(self):
        configs = [{'id': 'first'}, {'id': 'second'}, {'id': 'third'}]
        grid = [
            {'config': configs[0], 'validation': metrics(5, 3)},
            {'config': configs[1], 'validation': metrics(5, 4)},
            {'config': configs[2], 'validation': metrics(5, 4)},
        ]
        self.assertIs(clock.select(grid), grid[1])

    def test_baseline_can_win_the_joint_selection(self):
        baseline = {'config': {'id': 'baseline'}, 'validation': metrics(6, 4)}
        clock_candidate = {'config': {'id': 'clock'}, 'validation': metrics(6, 3)}
        self.assertIs(clock.select([baseline, clock_candidate]), baseline)


if __name__ == '__main__':
    unittest.main()
