import unittest
from unrestricted_prediction import UnrestrictedPrediction, remove_evidence_gate


class RecordingModel:
    def learn(self, record):
        pass

    def scores(self, query):
        self.last_query = query
        return {label: 1 for label in query['labels']}


class Tests(unittest.TestCase):
    def test_external_lists_and_future_truth_cannot_restrict_or_inject_labels(self):
        model = RecordingModel()
        predictor = UnrestrictedPrediction(model)
        predictor.learn({'label': 'page:past'})
        q = dict(origin='other', app='browser', timestamp=2, x={},
                 labels=['page:future'], candidates={}, candidate_last_seen={},
                 truth='page:future')
        self.assertEqual(predictor.predict(q)['predictions'], ['page:past'])
        self.assertEqual(set(model.last_query), {'origin', 'app', 'timestamp', 'x', 'labels'})

    def test_cold_start_has_no_future_classes(self):
        p = UnrestrictedPrediction(RecordingModel())
        self.assertEqual(p.predict(dict(origin='x', app='x', timestamp=0, x={}))['predictions'], [])

    def test_remove_same_origin_gate(self):
        base = {'family': 'frequency'}
        self.assertIs(remove_evidence_gate({'family': 'same_origin_evidence_gate',
                                           'parameters': {'base': base}}), base)


if __name__ == '__main__':
    unittest.main()
