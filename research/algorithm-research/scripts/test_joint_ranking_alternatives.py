import tempfile
import unittest
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix

from statistical_residual import DIMENSIONS, ResidualRanker
from joint_ranking_alternatives import JointRanker, configurations, top2_logistic, past_rows, choose_group


class AlternativesTests(unittest.TestCase):
    def example(self):
        return {'labels': ['a', 'b', 'c'], 'prior': np.array([.6, .3, .1]),
                'X': csr_matrix(([1., 1., 1., .5], ([0, 1, 2, 2], [0, 1, 2, 3])), shape=(3, DIMENSIONS)),
                'gate': np.linspace(.1, 1., 17)}

    def item(self, label='c'):
        return {'position': 0, 'input_index': 10, 'ready_index': 20, 'label': label,
                'weight': 1., 'key': ['origin', label, 'outcome']}

    def config(self, family): return next(c for c in configurations() if c['family'] == family)

    def test_all_families_start_at_statistical_prior(self):
        x = self.example()
        for family in ('linear_top2', 'linear_ce', 'linear_smooth_top2', 'context_gate_ce', 'nonlinear_top2'):
            np.testing.assert_allclose(list(JointRanker(self.config(family)).scores(x).values()), x['prior'])

    def test_smooth_loss_uses_second_incorrect_target(self):
        z = np.array([3., 2., 0.]); _, g = top2_logistic(z, 2)
        self.assertEqual(g[0], 0.)
        for i in range(3):
            d = np.zeros(3); d[i] = 1e-6
            derivative = (top2_logistic(z+d, 2)[0]-top2_logistic(z-d, 2)[0])/2e-6
            self.assertAlmostEqual(derivative, g[i], places=7)

    def test_gate_and_nonlinear_gradients(self):
        x = self.example()
        for family in ('context_gate_ce', 'nonlinear_top2'):
            model = JointRanker(self.config(family)); model.weights[:4] = [.2, -.3, .1, .2]
            if model.V is not None: model.V[:] = .03
            if model.gate is not None: model.gate[:] = .02
            _, gradients = model.gradients(x, 2)
            for name, gradient in gradients.items():
                array = getattr(model, name)
                indices = [(0,), (2,)] if array.ndim == 1 else [(0, 0), (2, 3)]
                for index in indices:
                    original = array[index]
                    array[index] = original+1e-6; plus = model.gradients(x, 2)[0]
                    array[index] = original-1e-6; minus = model.gradients(x, 2)[0]
                    array[index] = original
                    self.assertAlmostEqual((plus-minus)/2e-6, gradient[index], places=6)

    def test_linear_update_matches_previous_model(self):
        config = self.config('linear_top2'); old = ResidualRanker(config); new = JointRanker(config)
        examples = {0: self.example()}
        for label in ('a', 'b', 'c'):
            item = self.item(label); old.learn(item, examples, 21); new.learn(item, examples, 21)
        np.testing.assert_array_equal(old.weights, new.weights)
        np.testing.assert_array_equal(old.accumulator, new.accumulator)

    def test_all_architectures_reload_and_continue(self):
        examples = {0: self.example()}
        for family in ('context_gate_ce', 'nonlinear_top2', 'linear_smooth_top2'):
            model = JointRanker(self.config(family)); model.learn(self.item(), examples, 21)
            with tempfile.TemporaryDirectory() as directory:
                loaded = JointRanker.load(model.save(Path(directory), 'model'))
                model.learn(self.item('a'), examples, 22); loaded.learn(self.item('a'), examples, 22)
                for name, value in model.arrays().items(): np.testing.assert_array_equal(value, loaded.arrays()[name])

    def test_selection_uses_maturity_not_only_prediction_time(self):
        rows = [{'position': 0}, {'position': 1}, {'position': 2}]
        obs = [{'index': 10, 'ready_index': 40}, {'index': 15, 'ready_index': 20}, {'index': 30, 'ready_index': 30}]
        self.assertEqual(past_rows(rows, obs, 30), [{'position': 1}])

    def test_seed_choice_not_optimized(self):
        from evaluation_contract import prepare_evaluation
        rows = prepare_evaluation({'observations': [{'position': 0, 'index': 0, 'ready_index': 1,
            'target': {'id': 'current', 'kind': 'app', 'bundleID': 'current'}}],
            'evaluation': [{'position': 0, 'sample_id': 'sample:0', 'truth_action': 'a',
                'truth_metadata': {'id': 'a', 'kind': 'app', 'bundleID': 'a'},
                'evidence_grade': 'original'}]})
        good = {0: {'predictions': ['a']}}; bad = {0: {'predictions': ['b', 'c']}}
        self.assertEqual(choose_group(rows, {'lucky_seed': [good, bad, bad], 'stable': [good, good, bad]}), 'stable')

    def test_delayed_feedback_and_unseen_truth(self):
        for family in ('context_gate_ce', 'nonlinear_top2'):
            model = JointRanker(self.config(family)); x = self.example()
            with self.assertRaises(ValueError): model.learn(self.item(), {0: x}, 20)
            model.learn(self.item('unseen'), {0: x}, 21)
            self.assertEqual(model.audit['updates'], 0)
            self.assertEqual(x['labels'], ['a', 'b', 'c'])


if __name__ == '__main__': unittest.main()
