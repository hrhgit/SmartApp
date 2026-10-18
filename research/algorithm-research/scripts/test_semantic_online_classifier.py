import unittest
from copy import deepcopy

from semantic_online_classifier import build, features


def query(origin='page:one', tags=None):
    return {'origin': origin, 'app': 'test.browser', 'timestamp': 1000,
            'annotation': {'tags': tags or []}}


class ClassifierTests(unittest.TestCase):
    CONFIGS = [{'family': 'softmax', 'lr': .3, 'l2': 0., 'semantic': True},
               {'family': 'multinomial_nb', 'alpha': .1, 'semantic': True}]

    def test_dynamic_labels_four_independent_heads_and_immediate_update(self):
        for config in self.CONFIGS:
            model, q = build(config), query()
            self.assertEqual(model.predict(q, 0), {})
            model.learn(q, 'app:a', .5, 0)
            self.assertEqual(model.predict(q, 0), {'app:a': 1.})
            model.learn(q, 'page:b', 1., 0)
            self.assertEqual(set(model.predict(q, 0)), {'app:a', 'page:b'})
            for _ in range(20):
                model.learn(q, 'page:b', .5, 0)
            self.assertGreater(model.predict(q, 0)['page:b'], model.predict(q, 0)['app:a'])
            model.learn(q, 'directory:c', 1., 2)
            self.assertEqual(model.predict(q, 2), {'directory:c': 1.})
            self.assertEqual(model.predict(q, 1), {})
            self.assertEqual(model.predict(q, 3), {})

    def test_no_candidate_or_truth_feature_or_vocabulary_access(self):
        for config in self.CONFIGS:
            q = query(); model = build(config)
            model.learn(q, 'app:learned', 1., 0)
            before = model.predict(q, 0)
            poisoned = {**q, 'candidates': {'app:future': {}}, 'labels': ['app:future'],
                        'truth': 'app:future', 'destination_app': 'app:future',
                        'future': {'labels': ['app:future']}}
            self.assertEqual(model.predict(poisoned, 0), before)
            model.learn(poisoned, 'app:observed', 1., 0)
            self.assertNotIn('app:future', model.predict(q, 0))

    def test_zero_weight_does_not_introduce_class_or_feature(self):
        for config in self.CONFIGS:
            model = build(config)
            model.learn(query('page:future'), 'app:future', 0., 1)
            self.assertEqual(model.predict(query(), 1), {})

    def test_semantic_budget_does_not_scale_with_number_of_url_tags(self):
        tags = [{'key': 'scope:' + str(i), 'strength': .9, 'confidence': .9} for i in range(100)]
        x = features(query(tags=tags), True)
        self.assertAlmostEqual(sum(v for k, v in x.items() if k.startswith('semantic=')), .5)
        self.assertEqual(x['origin=page:one'], 1.)
        self.assertLess(max(v for k, v in x.items() if k.startswith('semantic=')), 1.)
        weak = features(query(tags=[{'key': 'site:generic', 'strength': .1, 'confidence': .8}]), True)
        self.assertAlmostEqual(weak['semantic=site:generic'], .04)

    def test_duplicate_tags_do_not_amplify_and_semantic_switch_ignores_annotation(self):
        tag = {'key': 'course:example', 'strength': .8, 'confidence': .9}
        self.assertEqual(features(query(tags=[tag]), True), features(query(tags=[tag] * 10), True))
        self.assertEqual(features(query(tags=[tag]), False), features(query(), False))

    def test_nb_fractional_weight_is_true_count_weighting(self):
        config = {'family': 'multinomial_nb', 'alpha': .1, 'semantic': True}
        one, halves = build(config), build(config)
        for model in (one, halves):
            model.learn(query('page:other'), 'app:other', 1., 1)
        one.learn(query(), 'app:a', 1., 1)
        halves.learn(query(), 'app:a', .5, 1)
        halves.learn(query(), 'app:a', .5, 1)
        for key, value in one.predict(query(), 1).items():
            self.assertAlmostEqual(value, halves.predict(query(), 1)[key])

    def test_softmax_fractional_weight_multiplies_single_gradient(self):
        full, half = build(self.CONFIGS[0]), build(self.CONFIGS[0])
        for model in (full, half):
            model.learn(query(), 'app:a', 1., 0)
        full.learn(query(), 'app:b', 1., 0)
        half.learn(query(), 'app:b', .5, 0)
        for label, row in full.heads[0].items():
            for feature, value in row.items():
                self.assertAlmostEqual(value / 2, half.heads[0][label][feature])

    def test_input_mutation_does_not_rewrite_past_features(self):
        for config in self.CONFIGS:
            model = build(config)
            q = query(tags=[{'key': 'scope:a', 'strength': 1., 'confidence': 1.}])
            original = deepcopy(q)
            model.learn(q, 'app:a', 1., 1)
            model.learn(query('page:other'), 'app:b', 1., 1)
            before = model.predict(original, 1)
            q['annotation']['tags'][0]['key'] = 'scope:future'
            q['origin'] = 'page:future'
            self.assertEqual(model.predict(original, 1), before)

    def test_invalid_weights_steps_and_configs(self):
        for config in self.CONFIGS:
            model = build(config)
            for weight in (-1., float('nan'), float('inf')):
                with self.assertRaises(ValueError):
                    model.learn(query(), 'app:a', weight, 1)
            for step in (-1, 4, True, 1.5):
                with self.assertRaises(ValueError):
                    model.predict(query(), step)
        with self.assertRaises(ValueError):
            build({'family': 'not-a-model'})


if __name__ == '__main__':
    unittest.main()
