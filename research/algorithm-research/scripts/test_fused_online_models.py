import unittest

from fused_online_models import FusedModel


class FixedModel:
    def __init__(self, scores):
        self.values = dict(scores)
        self.records = []

    def learn(self, record):
        self.records.append(record)
        return len(self.records)

    def scores(self, query):
        return dict(self.values)


class CounterModel(FixedModel):
    def learn(self, record):
        super().learn(record)
        label = record['label']
        self.values[label] = self.values.get(label, 0) + 1

    def scores(self, query):
        return {key: value for key, value in self.values.items() if key in query['labels']}


class FusedModelTests(unittest.TestCase):
    def test_weighted_rrf_matches_formula_and_missing_labels_add_no_vote(self):
        m = FusedModel(FixedModel({'A': 10, 'B': 5}),
                       FixedModel({'B': 2, 'C': 1}), .25, rank_offset=3)
        scores = m.scores({})
        self.assertEqual(scores['A'], .75 / 4)
        self.assertEqual(scores['B'], .75 / 5 + .25 / 4)
        self.assertEqual(scores['C'], .25 / 5)

    def test_zero_and_negative_margins_are_ranked_without_filtering(self):
        m = FusedModel(FixedModel({'A': 0, 'B': -10}), FixedModel({}), .5)
        self.assertEqual(m.scores({}), {'A': .5, 'B': .25})

    def test_endpoint_weights_preserve_union_including_zero_final_score(self):
        m = FusedModel(FixedModel({'A': 1}), FixedModel({'B': 1}), 0)
        self.assertEqual(m.scores({}), {'A': 1, 'B': 0})

    def test_cold_start_then_one_transition_updates_both_components(self):
        left, right = CounterModel({}), CounterModel({})
        m = FusedModel(left, right, .5)
        query = {'labels': ['NEW']}
        self.assertEqual(m.scores(query), {})
        row = {'origin': 'current', 'label': 'NEW'}
        m.learn(row)
        self.assertEqual(left.records, [row])
        self.assertEqual(right.records, [row])
        self.assertEqual(m.scores(query), {'NEW': 1})
        self.assertEqual(m.scores({'labels': ['other']}), {})

    def test_tie_mapping_reproduces_current_concrete_candidate_order(self):
        m = FusedModel(FixedModel({'window_class:A': 0, 'window_class:B': 0}),
                       FixedModel({}), .5)
        plain = m.scores({})
        mapped = m.scores({'candidate_key_order': {'window_class:A': 'window@9',
                                                  'window_class:B': 'window@1'}})
        self.assertGreater(plain['window_class:A'], plain['window_class:B'])
        self.assertGreater(mapped['window_class:B'], mapped['window_class:A'])

    def test_score_scale_does_not_change_fusion(self):
        a = FusedModel(FixedModel({'A': 1, 'B': 2}), FixedModel({'C': -1}), .3)
        b = FusedModel(FixedModel({'A': 100, 'B': 200}), FixedModel({'C': 1000}), .3)
        self.assertEqual(a.scores({}), b.scores({}))

    def test_diagnosis_truth_and_future_metadata_are_not_used(self):
        m = FusedModel(FixedModel({'A': 1}), FixedModel({'B': 1}), .5)
        before = m.scores({})
        self.assertEqual(before, m.scores({'truth': 'B', 'diagnosis_after_outcome': {'grade': 'strong'},
                                          'future_candidates': ['C']}))

    def test_invalid_parameters_and_nonfinite_scores_fail(self):
        for weight in (-.1, 1.1, float('nan'), float('inf')):
            with self.assertRaises(ValueError):
                FusedModel(FixedModel({}), FixedModel({}), weight)
        for offset in (-1, float('nan'), float('inf')):
            with self.assertRaises(ValueError):
                FusedModel(FixedModel({}), FixedModel({}), .5, offset)
        m = FusedModel(FixedModel({'A': float('nan')}), FixedModel({}), .5)
        with self.assertRaises(ValueError):
            m.scores({})
        component = FixedModel({})
        with self.assertRaises(ValueError):
            FusedModel(component, component, .5)


if __name__ == '__main__':
    unittest.main()
