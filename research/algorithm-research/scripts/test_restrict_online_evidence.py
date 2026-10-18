import copy
import unittest

import evaluate_context_knn as core
import compare_online_models as comparison
from restrict_online_evidence import EvidenceRestrictedModel, allowed_at_cutoffs, restricted_predictions
from test_evaluate_context_knn import observation, target
from test_compare_online_models import scored_row, stream_row


class FixedModel:
    def __init__(self, scores=None):
        self.values = scores or {'B': .2, 'C': .8}
        self.records = []
        self.query = None

    def learn(self, record):
        self.records.append(copy.deepcopy(record))

    def scores(self, query):
        self.query = copy.deepcopy(query)
        return dict(self.values)


class EvidenceRestrictionTests(unittest.TestCase):
    def test_current_origin_support_is_required_and_global_support_is_insufficient(self):
        base = FixedModel()
        model = EvidenceRestrictedModel(base)
        query = {'origin': 'A', 'labels': ['B', 'C']}
        self.assertEqual(model.scores(query), {})
        model.learn({'origin': 'other', 'label': 'C', 'outcome_id': '1'})
        self.assertEqual(model.scores(query), {})
        model.learn({'origin': 'A', 'label': 'B', 'outcome_id': '2'})
        self.assertEqual(model.scores(query), {'B': .2})
        self.assertEqual(base.query['labels'], ['B', 'C'])

    def test_duplicate_arrival_does_not_update_base_twice(self):
        base = FixedModel()
        model = EvidenceRestrictedModel(base)
        record = {'origin': 'A', 'label': 'B', 'outcome_id': 'arrival'}
        model.learn(record)
        model.learn(record)
        self.assertEqual(len(base.records), 1)
        model.learn({**record, 'origin': 'other'})
        self.assertEqual(len(base.records), 2)

    def test_future_truth_and_cohort_fields_do_not_affect_gate(self):
        model = EvidenceRestrictedModel(FixedModel())
        model.learn({'origin': 'A', 'label': 'B', 'outcome_id': 'arrival'})
        plain = {'origin': 'A', 'labels': ['B', 'C']}
        changed = {**plain, 'truth': 'C', 'grade': 'strong', 'eligible': True,
                   'future_support': {'C': 1000}}
        self.assertEqual(model.scores(plain), model.scores(changed))

    def test_strict_maturity_and_immutable_earlier_gate_snapshots(self):
        items = [observation(1), observation(30), observation(31), observation(40)]
        inputs = {index: {'origin': 'A'} for index in (1, 2, 3)}
        second = stream_row(position=1, ready_index=39, label='C')
        gates = allowed_at_cutoffs(items, [stream_row(), second], inputs)
        self.assertEqual(gates[1], frozenset())
        self.assertEqual(gates[2], frozenset({'B'}))
        self.assertEqual(gates[3], frozenset({'B', 'C'}))
        self.assertIsNot(gates[2], gates[3])

    def test_training_cleaning_streams_never_share_support(self):
        items = [observation(1), observation(31)]
        inputs = {1: {'origin': 'A'}}
        three = allowed_at_cutoffs(items, [stream_row(label='B')], inputs)
        five = allowed_at_cutoffs(items, [stream_row(label='C')], inputs)
        self.assertEqual(three[1], frozenset({'B'}))
        self.assertEqual(five[1], frozenset({'C'}))

    def test_filter_precedes_hierarchy_and_preserves_surviving_raw_order(self):
        query = observation(40)
        query.candidates = {
            'app:X': {**target('app:X'), '_bundle': 'X'},
            'window@s:w': {**target('w', 'window'), '_bundle': 'X'},
            'B': {**target('B'), '_bundle': 'B'},
        }
        source = {0: {'raw_ranking': ['app:X', 'window@s:w', 'B'],
                      'predictions': ['app:X', 'B']}}
        before = copy.deepcopy(source)
        gated = restricted_predictions(core, source, {0: frozenset({'window_class:X', 'B'})}, [query])
        self.assertEqual(gated[0]['raw_ranking'], ['window@s:w', 'B'])
        self.assertEqual(gated[0]['predictions'], ['window@s:w', 'B'])
        self.assertEqual(source, before)

    def test_gate_abstention_does_not_remove_frozen_supported_member(self):
        query = observation(40)
        prediction = restricted_predictions(core, {0: {'raw_ranking': ['B']}},
                                            {0: frozenset()}, [query])
        scores = comparison.score_groups([scored_row(0)], prediction)
        self.assertEqual(scores['supported']['n'], 1)
        self.assertEqual(scores['supported']['top4_hits'], 0)


if __name__ == '__main__':
    unittest.main()
