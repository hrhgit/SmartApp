import unittest

import extract_current_evidence as audit
from score_current_evidence import score_submission
from test_evaluate_context_knn import observation


class EvidenceAuditTests(unittest.TestCase):
    def fixture(self):
        q = observation(20)
        q.truth = 'B'
        record = {'counts': {'B': 1, 'C': 1}, 'evidence_n': 2,
            'last_conditional_outcome': {'B': q.timestamp - 1000, 'C': q.timestamp - 500},
            'predictions': ['C'], 'top1_support': 1, 'top1_historical_fraction': .5}
        history = [{'destination_key': k, 'outcome_event_id': k,
            'input_event_index': 1, 'outcome_event_index': 2, 'label_ready_event_index': 3}
            for k in ('B', 'C')]
        return q, record, history

    def test_strength_includes_disagreement_and_requires_repetition(self):
        self.assertEqual(audit.grade(1, 1), 'single')
        self.assertEqual(audit.grade(2, 2), 'repeated')
        self.assertEqual(audit.grade(3, 4), 'strong')
        self.assertEqual(audit.grade(20, 100), 'repeated')
        self.assertEqual(audit.grade(0, 0), 'none')

    def test_diagnostic_membership_is_independent_of_prediction_success(self):
        q, record, history = self.fixture()
        before = audit.diagnostic_evidence(q, record, history)
        record['predictions'] = ['B']
        self.assertEqual(audit.diagnostic_evidence(q, record, history), before)
        self.assertTrue(before['eligible'])

    def test_candidate_omission_still_belongs_to_evidence_benchmark(self):
        q, record, history = self.fixture()
        q.candidates.pop('B')
        result = audit.diagnostic_evidence(q, record, history)
        self.assertTrue(result['eligible'])
        self.assertFalse(result['candidate_present'])
        self.assertFalse(result['support_resolves_to_actual_candidate'])

    def test_prospective_gate_never_requires_realized_target(self):
        _, record, _ = self.fixture()
        result = audit.prospective_gate(record)
        record['truth'] = 'future-answer-not-an-input'
        self.assertEqual(audit.prospective_gate(record), result)
        self.assertEqual(result['grade'], 'single')

    def test_future_and_duplicate_witnesses_are_rejected(self):
        q, record, history = self.fixture()
        history[0]['label_ready_event_index'] = q.index
        with self.assertRaises(ValueError):
            audit.diagnostic_evidence(q, record, history)
        history[0]['label_ready_event_index'] = 3
        history[1]['outcome_event_id'] = history[0]['outcome_event_id']
        with self.assertRaises(ValueError):
            audit.diagnostic_evidence(q, record, history)

    def test_window_class_support_is_not_claimed_as_exact_instance(self):
        q, record, history = self.fixture()
        q.truth_meta = {'kind': 'window', '_bundle': 'example.app'}
        record['counts'] = {'window_class:example.app': 1, 'C': 1}
        record['last_conditional_outcome']['window_class:example.app'] = q.timestamp - 1000
        history[0]['destination_key'] = 'window_class:example.app'
        evidence = audit.diagnostic_evidence(q, record, history)
        self.assertTrue(evidence['eligible'])
        self.assertEqual(evidence['identity_level'], 'application_window_class')
        self.assertFalse(evidence['support_resolves_to_actual_candidate'])

    def submission_fixture(self):
        q, record, history = self.fixture()
        return [{'sample_id': 'sample', 'pre_prediction': {'candidates': q.candidates},
            'diagnosis_after_outcome': audit.diagnostic_evidence(q, record, history),
            'prospective_gate': audit.prospective_gate(record),
            'observed_outcome': {'key': q.truth, 'metadata': q.truth_meta},
            'evaluation': {'strict_input_and_label_coverage': True}}]

    def test_abstention_is_an_error_and_missing_submission_is_rejected(self):
        rows = self.submission_fixture()
        with self.assertRaises(ValueError):
            score_submission(rows, [])
        score = score_submission(rows, [{'sample_id': 'sample', 'predictions': []}])
        self.assertEqual(score['evidence_supported']['n'], 1)
        self.assertEqual(score['evidence_supported']['top4'], 0.)
        correct = score_submission(rows, [{'sample_id': 'sample', 'predictions': ['B']}])
        self.assertEqual(correct['evidence_supported']['top4'], 1.)

    def test_submissions_cannot_expand_slots_or_fabricate_candidates(self):
        rows = self.submission_fixture()
        for predictions in (['B'] * 2, ['B', 'C', 'D', 'E', 'F'], ['unobserved']):
            with self.assertRaises(ValueError):
                score_submission(rows, [{'sample_id': 'sample', 'predictions': predictions}])


if __name__ == '__main__':
    unittest.main()
