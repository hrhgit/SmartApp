"""Check cached rank-list search against independently streamed fusion scores."""
import copy
import random
import unittest

import compare_online_models as comparison
import evaluate_context_knn as core
from fused_online_models import FusedModel
import search_online_fusions as search
from test_evaluate_context_knn import observation, target


class FixedScores:
    def __init__(self, scores):
        self.values = dict(scores)

    def scores(self, query):
        return dict(self.values)


class FusionSearchTests(unittest.TestCase):
    def fixture(self):
        query = observation(40)
        query.candidates = {
            # Semantic class and concrete-key orders deliberately disagree.
            'window@s:z': {**target('z', 'window'), '_bundle': 'A'},
            'window@s:a': {**target('a', 'window'), '_bundle': 'Z'},
            'app:A': {**target('app:A'), '_bundle': 'A'},
            **{key: {**target(key), '_bundle': key} for key in 'BCDEF'},
        }
        model_input, mapping = comparison.model_input(query)
        model_input['candidate_key_order'] = mapping
        return query, model_input, mapping

    def cached_prediction(self, query, scores, mapping):
        raw, shown = comparison.project_scores(core, scores, mapping, query.candidates)
        return {0: {'sample_id': query.sample_id, 'raw_ranking': raw,
                    'predictions': shown}}

    def test_cached_and_streaming_fusion_match_with_ties_negatives_and_hierarchy(self):
        query, model_input, mapping = self.fixture()
        randomizer = random.Random(42)
        for trial in range(20):
            scores = [{key: randomizer.choice([-2., 0., 1., 1., 2.])
                       for key in model_input['labels'] if randomizer.random() > .2}
                      for _ in range(2)]
            left = self.cached_prediction(query, scores[0], mapping)
            right = self.cached_prediction(query, scores[1], mapping)
            for offset in (0, 10):
                for weight in (0, .1, .25, .5, .75, .9, 1):
                    with self.subTest(trial=trial, offset=offset, weight=weight):
                        cached = search.fuse(core, left, right, weight, offset, [query])[0]
                        live = FusedModel(FixedScores(scores[0]), FixedScores(scores[1]),
                                          weight, offset)
                        raw, shown = comparison.project_scores(
                            core, live.scores(model_input), mapping, query.candidates)
                        self.assertEqual(raw, cached['raw_ranking'])
                        self.assertEqual(shown, cached['predictions'])

    def test_cached_search_never_mutates_original_component_predictions(self):
        query, _, mapping = self.fixture()
        left = self.cached_prediction(query, {'window_class:A': 1., 'B': .5}, mapping)
        right = self.cached_prediction(query, {'app:A': 2., 'C': .5}, mapping)
        before = copy.deepcopy((left, right))
        output = search.fuse(core, left, right, .5, 0, [query])
        output[0]['predictions'].clear()
        output[0]['raw_ranking'].clear()
        self.assertEqual((left, right), before)

    def test_empty_component_is_abstention_and_does_not_add_candidates(self):
        query, _, mapping = self.fixture()
        left = self.cached_prediction(query, {}, mapping)
        right = self.cached_prediction(query, {'B': 1.}, mapping)
        combined = search.fuse(core, left, right, .25, 10, [query])
        self.assertEqual(combined[0]['predictions'], ['B'])
        self.assertEqual(combined[0]['raw_ranking'], ['B'])

    def test_inconsistent_cutoffs_are_rejected(self):
        query, _, mapping = self.fixture()
        left = self.cached_prediction(query, {'B': 1.}, mapping)
        with self.assertRaises(AssertionError):
            search.fuse(core, left, {}, .5, 0, [query])

    def test_auxiliary_feature_scope_cannot_be_reported_as_current_only(self):
        def result(scope):
            return {'config': {'feature_scope': scope}}
        current = result('current_only')
        self.assertEqual(search.combined_scope(current, current), 'current_only')
        self.assertNotEqual(search.combined_scope(current, result('current_plus_clock')),
                            'current_only')
        combined = search.combined_scope(result('current_plus_clock'),
                                         result('current_plus_candidate_recency'))
        self.assertIn('clock', combined)
        self.assertIn('candidate_recency', combined)


if __name__ == '__main__':
    unittest.main()
