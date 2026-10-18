import tempfile
import unittest
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix

from statistical_residual import DIMENSIONS, ResidualRanker, loss_gradient, select_top2


class ResidualTests(unittest.TestCase):
    def example(self):
        X = np.zeros((3, DIMENSIONS)); X[0, 0] = 1; X[1, 1] = 1
        X[2, DIMENSIONS//2] = 1
        return {'labels': ['a', 'b', 'c'], 'prior': np.array([.6, .3, .1]), 'X': csr_matrix(X)}

    def model(self, objective='top2_hinge', context=True):
        return ResidualRanker({'objective': objective, 'context': context, 'lr': .1, 'rehearsal': 7})

    def item(self, label='c'):
        return {'position': 0, 'input_index': 10, 'ready_index': 20, 'label': label,
                'weight': 1., 'key': ['origin', label, 'outcome']}

    def test_zero_residual_reproduces_statistical_prior(self):
        example = self.example()
        np.testing.assert_allclose(list(self.model().scores(example).values()), example['prior'])

    def test_top2_loss_pushes_truth_above_second_incorrect_not_best(self):
        loss, grad = loss_gradient(np.array([3., 2., 0.]), 2, 'top2_hinge')
        self.assertEqual(loss, 3.)
        np.testing.assert_array_equal(grad, [0., 1., -1.])
        self.assertEqual(loss_gradient(np.array([3., 0., 2.]), 2, 'top2_hinge')[0], 0.)
        self.assertEqual(loss_gradient(np.array([3., 0.]), 1, 'top2_hinge')[0], 0.)

    def test_surrogate_gradients_match_finite_difference(self):
        for objective in ('top2_hinge', 'cross_entropy'):
            z = np.array([3., 2., 0.]); _, gradient = loss_gradient(z, 2, objective)
            numeric = np.zeros(3)
            for i in range(3):
                delta = np.zeros(3); delta[i] = 1e-6
                numeric[i] = (loss_gradient(z+delta, 2, objective)[0]-
                              loss_gradient(z-delta, 2, objective)[0])/2e-6
            np.testing.assert_allclose(gradient, numeric, atol=1e-7)

    def test_strict_maturity_and_deduplication(self):
        model = self.model(); examples = {0: self.example()}
        with self.assertRaises(ValueError): model.learn(self.item(), examples, 20)
        model.learn(self.item(), examples, 21); weights = model.weights.copy()
        model.learn(self.item(), examples, 30)
        np.testing.assert_array_equal(weights, model.weights)
        self.assertEqual(model.audit['updates'], 1)

    def test_unseen_outcome_does_not_enter_original_prior(self):
        model = self.model(); x = self.example(); before = x['prior'].copy()
        model.learn(self.item('new'), {0: x}, 21)
        self.assertEqual(model.audit['truth_unavailable_at_original_cutoff'], 1)
        self.assertEqual(model.audit['updates'], 0)
        self.assertEqual(x['labels'], ['a', 'b', 'c'])
        np.testing.assert_array_equal(x['prior'], before)

    def test_context_ablation_and_training_direction(self):
        x = self.example(); model = self.model(context=False)
        old_gap = model.logits(x)[2]-model.logits(x)[1]
        model.learn(self.item(), {0: x}, 21)
        self.assertGreater(model.logits(x)[2]-model.logits(x)[1], old_gap)
        np.testing.assert_array_equal(model.weights[DIMENSIONS//2:], 0.)

    def test_reload_and_next_update_exact(self):
        model = self.model(); examples = {0: self.example()}
        model.learn(self.item(), examples, 21)
        with tempfile.TemporaryDirectory() as directory:
            loaded = ResidualRanker.load(model.save(Path(directory), 'ranker'))
            item = self.item('a'); model.learn(item, examples, 30); loaded.learn(item, examples, 30)
            np.testing.assert_array_equal(model.weights, loaded.weights)
            np.testing.assert_array_equal(model.accumulator, loaded.accumulator)

    def test_top2_selection_does_not_consult_later_results(self):
        def result(top2, top4, future):
            return {'validation': {g: {'top2_hits': top2, 'top1_hits': 1, 'top4_hits': top4}
                                   for g in ('original', 'expanded', 'all')}, 'future': future}
        a, b = result(5, 5, 0), result(4, 10, 100)
        self.assertIs(select_top2([a, b]), a)


if __name__ == '__main__': unittest.main()
