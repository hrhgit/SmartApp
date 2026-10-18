"""Cost hypotheses, unknown mass and menu decision invariants."""
import unittest
import evaluation_contract as s
from bayes_frequency_policy import frequency_costs, expand_service_probabilities, risk_menu, calibration_summary, UNKNOWN, UNKNOWN_CHILD


class FrequencyPolicyTests(unittest.TestCase):
    def test_unknown_bucket_calibration_does_not_grant_an_exact_menu_hit(self):
        import math
        data = {'observations': [{'position': 0, 'index': 1, 'ready_index': 2,
            'target': {'id': 'app:current', 'kind': 'app', 'bundleID': 'current'}}],
            'evaluation': [{'position': 0, 'sample_id': 'sample:0', 'truth_action': 'page:new',
                'truth_metadata': {'id': 'page:new', 'kind': 'page', 'locator':
                    'https://chatgpt.com/c/12345678-1234-1234-1234-123456789abc'},
                'evidence_grade': 'original'}]}
        rows = s.prepare_evaluation(data)
        summary = calibration_summary(rows, {0: {UNKNOWN_CHILD: .7, 'known': .2, UNKNOWN: .1}}, {}, {})
        self.assertEqual(summary['truth_missing_n'], 1)
        self.assertAlmostEqual(summary['nll_floor_1e15'], -math.log(.7))
        self.assertEqual(summary['confidence_bins'][0]['accuracy'], 1.)
        metric = s.measure(rows, {0: {'predictions': [s.SERVICE]}})
        self.assertEqual(metric['exact']['top1_hits'], 0)

    def test_frequency_cost_favors_frequent_misses(self):
        w = frequency_costs(['a', 'b'], {'a': 9, 'b': 1}, {}, 'global_frequency')
        self.assertGreater(w['a'], w['b']); self.assertAlmostEqual(sum(w.values()), 2.)
        decision = s.minimum_risk_action({'a': .4, 'b': .6}, ['a', 'b'], {}, set(), w)
        self.assertEqual(decision['action'], 'a')

    def test_context_without_support_shrinks_to_global(self):
        a = frequency_costs(['a', 'b'], {'a': 9, 'b': 1}, {}, 'global_frequency')
        b = frequency_costs(['a', 'b'], {'a': 9, 'b': 1}, {}, 'context_frequency')
        self.assertEqual(a, b)

    def test_context_can_reverse_global_popularity(self):
        w = frequency_costs(['a', 'b'], {'a': 9, 'b': 1}, {'b': 5}, 'context_frequency')
        self.assertGreater(w['b'], w['a'])

    def test_expansion_preserves_unknown_mass_and_total(self):
        aliases = {'a': s.SERVICE, 'b': s.SERVICE, s.SERVICE: s.SERVICE}
        p, actions = expand_service_probabilities({s.SERVICE: .8, 'x': .2}, aliases, {s.SERVICE},
            {'a': 5, 'b': 2}, {'b': 3}, .1)
        self.assertAlmostEqual(sum(p.values()), 1.)
        self.assertEqual(p[UNKNOWN], .1); self.assertGreater(p[UNKNOWN_CHILD], 0.)
        self.assertGreater(p['b'], p['a']); self.assertNotIn(s.SERVICE, p)
        self.assertIn(s.SERVICE, actions); self.assertNotIn(UNKNOWN_CHILD, actions)

    def test_no_service_children_falls_back_with_unknown_child(self):
        p, actions = expand_service_probabilities({s.SERVICE: 1.}, {s.SERVICE: s.SERVICE}, {s.SERVICE}, {}, {}, .1)
        self.assertEqual(actions, [s.SERVICE]); self.assertAlmostEqual(p[UNKNOWN_CHILD], .9)

    def test_first_menu_action_is_exact_minimum_and_loss_never_increases(self):
        p = {'a': .4, 'b': .3, 'c': .2, 'x': .1}
        aliases = {k: s.SERVICE for k in 'abc'}; aliases[s.SERVICE] = s.SERVICE
        actions = [s.SERVICE, 'a', 'b', 'c', 'x']; costs = {y: 1. for y in p}
        chosen = s.minimum_risk_action(p, actions, aliases, {s.SERVICE}, costs)
        menu, loss = risk_menu(p, actions, aliases, {s.SERVICE}, costs)
        self.assertEqual(menu[0], chosen['action'])
        self.assertEqual(len(set(menu)), 4)
        self.assertTrue(all(a >= b for a, b in zip(loss, loss[1:])))

    def test_invalid_weights_are_rejected(self):
        for weights in ({}, {'a': 0}, {'a': -1}, {'a': float('inf')}):
            with self.assertRaises(ValueError): s.minimum_risk_action({'a': 1.}, ['a'], {}, set(), weights)


if __name__ == '__main__': unittest.main()
