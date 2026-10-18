import copy
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

from prepare_semantic_trajectory import (anchor_overlap_counts, checked_redacted_snapshot,
    contains_collected_candidates, freeze_evidence, inputs_at_anchors,
    soft_overlap_labels, sorted_stream, supervision_duplicates)


class FakeCore:
    @staticmethod
    def admitted(meta):
        return meta.get('admitted', True)

    @staticmethod
    def synthetic(meta):
        return False

    @staticmethod
    def key_for(meta, session):
        return 'window@' + session + ':' + meta['id'] if meta['kind'] == 'window' else meta['id']


def observed(index, origin='app:a', position=None):
    return {'position': index if position is None else position, 'index': index,
            'origin': origin, 'annotation': {'origin': origin}}


def example_row(position, original=False):
    return {'position': position, 'sample_id': 'sample-' + str(position),
            'truth_action': 'app:z', 'observed_outcome': {'key': 'exact-original-z'},
            'diagnosis_after_outcome': {'eligible': original}}


def training(position, ready, outcome='outcome-z', label='app:z', step=None):
    record = {'label': label, 'outcome_id': outcome}
    if step is not None:
        record.update(action_key=label, arrival_event_id=outcome, observed_step=step)
    return {'position': position, 'input_index': position, 'ready_index': ready,
            'record': record}


def related(a, b):
    return {'level': 'strong', 'score': .8, 'matched_tags': ['shared-scope']}


class EvidenceTests(unittest.TestCase):
    def test_strict_maturity_and_unsorted_stream(self):
        observations = [observed(i) for i in range(11)]
        stream = [training(0, 10, 'not-ready'), training(1, 9, 'ready')]
        result = freeze_evidence([example_row(10)], observations, stream, related)[0]
        self.assertEqual(result['evidence_grade'], 'added_exact')
        self.assertEqual([w['outcome_id'] for w in result['witnesses']], ['ready'])

    def test_same_actual_outcome_is_one_witness_across_duplicate_anchors(self):
        observations = [observed(i, 'other' if i == 0 else 'app:a') for i in range(11)]
        stream = [training(0, 5), training(1, 6)]
        result = freeze_evidence([example_row(10)], observations, stream, related)[0]
        self.assertEqual(result['evidence_count'], 1)
        self.assertEqual(result['witnesses'][0]['source_position'], 1)
        self.assertEqual(result['witnesses'][0]['relation']['level'], 'exact')

    def test_semantic_expansion_preserves_exact_truth_and_original_membership(self):
        observations = [observed(i, 'different' if i == 0 else 'app:a') for i in range(11)]
        rows = [example_row(10)]
        before = copy.deepcopy(rows)
        result = freeze_evidence(rows, observations, [training(0, 5)], related)[0]
        self.assertEqual(result['evidence_grade'], 'semantic_strong')
        self.assertEqual(result['truth_action'], 'app:z')
        self.assertEqual(result['observed_outcome'], before[0]['observed_outcome'])
        self.assertEqual(rows, before)
        original = freeze_evidence([example_row(10, True)], observations, [], related)[0]
        self.assertEqual(original['evidence_grade'], 'original')
        self.assertEqual(original['evidence_count'], 0)

    def test_wrong_actual_outcome_cannot_create_semantic_support(self):
        observations = [observed(i) for i in range(11)]
        result = freeze_evidence([example_row(10)], observations,
                                 [training(0, 5, label='app:other')], related)[0]
        self.assertEqual(result['evidence_grade'], 'unsupported')

    def test_future_reference_mutation_does_not_change_evidence(self):
        observations = [observed(i) for i in range(11)]
        rows = [example_row(10)]
        baseline = freeze_evidence(rows, observations, [training(0, 9)], related)
        changed = freeze_evidence(rows, observations,
            [training(1, 99, 'future-identity', 'app:z'), training(0, 9)], related)
        self.assertEqual(baseline, changed)

    def test_conflicting_labels_are_audited_without_double_counting_identities(self):
        observations = [observed(0), observed(1)]
        stream = [training(0, 5, step=1), training(1, 6, step=1),
                  training(1, 6, label='app:changed', step=1), training(1, 6, step=2)]
        audit = supervision_duplicates(stream, observations)
        self.assertEqual(audit['conflicting_label_groups'], 1)
        self.assertEqual(audit['unique_head_origin_outcomes'], 2)


class InputTests(unittest.TestCase):
    def fixture(self):
        target = {'id': 'app:a', 'kind': 'app', 'bundleID': 'a', 'name': 'before'}
        events = [{'id': 'update', 'sessionID': 's', 'kind': 'target.update',
                   'data': {'target': target}, 'monotonicMs': 0},
                  {'id': 'anchor', 'sessionID': 's', 'kind': 'context.change',
                   'data': {'targetID': 'app:a'}, 'monotonicMs': 1}]
        q = SimpleNamespace(index=1, ready_index=5, timestamp=101, session='s',
            anchor_id='anchor', sample_id='sample', segment=1, current='app:a',
            current_app='a', background={}, source_quality={}, coverage='unknown', strict=False)
        return events, q

    def test_later_metadata_cannot_repair_or_change_current_annotations(self):
        events, q = self.fixture()
        annotate = lambda meta: {'name': meta['name']}
        before = inputs_at_anchors(FakeCore, [q], events, annotate)
        events.append({'id': 'future', 'sessionID': 's', 'kind': 'target.update',
            'data': {'target': {'id': 'app:a', 'kind': 'app', 'bundleID': 'a', 'name': 'future'}},
            'monotonicMs': 2})
        after = inputs_at_anchors(FakeCore, [q], events, annotate)
        self.assertEqual(before, after)
        self.assertEqual(after[0]['annotation']['name'], 'before')

    def test_missing_current_metadata_does_not_use_future_catalog(self):
        events, q = self.fixture()
        events[0]['data']['target']['admitted'] = False
        with self.assertRaisesRegex(ValueError, 'metadata'):
            inputs_at_anchors(FakeCore, [q], events, lambda meta: meta)

    def test_overlap_includes_censored_prior_anchors_but_not_future(self):
        events, q = self.fixture()
        events[1]['monotonicMs'] = 10000
        for ident, time in [('second', 20000), ('third', 40000), ('future', 40001)]:
            events.append({'id': ident, 'sessionID': 's', 'kind': 'context.change',
                'data': {'targetID': 'app:a'}, 'monotonicMs': time})
        samples = [{'anchorEventID': e['id'], 'origin': 'natural_observed',
                    'status': 'censored'} for e in events[1:]]
        observations = [{'position': 0, 'index': 2, 'origin': 'app:a', 'anchor_id': 'second'},
                        {'position': 1, 'index': 3, 'origin': 'app:a', 'anchor_id': 'third'}]
        result = anchor_overlap_counts(FakeCore, events, samples, observations, set())
        self.assertEqual([result[i]['prior_anchor_count'] for i in range(2)], [1, 1])
        samples[0]['status'] = 'observed_horizon'
        self.assertEqual(result, anchor_overlap_counts(FakeCore, events, samples, observations, set()))

    def test_overlap_is_weight_only_and_does_not_mutate_base_supervision(self):
        labels = [{'action_key': 'app:a', 'arrival_event_id': 'real', 'observed_step': 3,
                   'target': {'id': 'app:a'}, 'weight': .4, 'basis': 'soft_kind'}]
        before = copy.deepcopy(labels)
        result = soft_overlap_labels(labels, {'multiplier': .5, 'prior_anchor_count': 3})
        self.assertEqual(labels, before)
        self.assertEqual(result[0]['weight'], .2)
        for key in ('action_key', 'arrival_event_id', 'observed_step', 'target'):
            self.assertEqual(result[0][key], labels[0][key])


class SnapshotTests(unittest.TestCase):
    def snapshot(self, folder):
        names = ('events.jsonl', 'navigation-samples.jsonl', 'target-history.jsonl', 'manifest.json')
        expected = {}
        for name in names:
            data = (name + '\n').encode()
            (folder/name).write_bytes(data)
            expected[name] = hashlib.sha256(data).hexdigest()
        self.sums(folder, expected)
        return expected

    @staticmethod
    def sums(folder, hashes):
        (folder/'SHA256SUMS').write_text(''.join(h + '  ' + name + '\n' for name, h in hashes.items()))

    def test_missing_or_duplicate_checksum_fails_closed(self):
        with TemporaryDirectory() as directory:
            folder = Path(directory); expected = self.snapshot(folder)
            checked_redacted_snapshot(folder, expected)
            self.sums(folder, {k: v for k, v in expected.items() if k != 'manifest.json'})
            with self.assertRaises(ValueError):
                checked_redacted_snapshot(folder, expected)
            self.sums(folder, expected)
            with (folder/'SHA256SUMS').open('a') as f:
                f.write(expected['manifest.json'] + '  manifest.json\n')
            with self.assertRaises(ValueError):
                checked_redacted_snapshot(folder, expected)

    def test_changed_source_requires_exact_authorized_lineage(self):
        with TemporaryDirectory() as directory:
            folder = Path(directory); expected = self.snapshot(folder)
            (folder/'events.jsonl').write_text('redacted\n')
            actual = {**expected, 'events.jsonl': hashlib.sha256(b'redacted\n').hexdigest()}
            marker = {'operation': 'remove-collected-candidate-lists-v1', 'user_authorized': True,
                'old_training_provenance_requires_explicit_redaction_migration': True,
                'changed_file': {'path': str(folder/'events.jsonl'), 'rows': 1, 'changed_rows': 1,
                    'sha256_before': expected['events.jsonl'], 'sha256_after': actual['events.jsonl']}}
            (folder/'candidate-list-redaction.json').write_text(json.dumps(marker))
            actual['candidate-list-redaction.json'] = hashlib.sha256((folder/'candidate-list-redaction.json').read_bytes()).hexdigest()
            self.sums(folder, actual)
            self.assertEqual(checked_redacted_snapshot(folder, expected)['changed_files'], ['events.jsonl'])
            wrong = {**expected, 'events.jsonl': 'different-original'}
            with self.assertRaisesRegex(ValueError, 'lineage'):
                checked_redacted_snapshot(folder, wrong)

    def test_collected_candidates_are_detected_recursively(self):
        self.assertTrue(contains_collected_candidates([{'nested': [{'candidateTargetIDs': []}]}]))
        self.assertFalse(contains_collected_candidates({'inventory': [{'targetID': 'observed'}]}))

    def test_streams_sorted_by_availability_and_reject_backwards_maturity(self):
        stream = sorted_stream([training(1, 9), training(0, 5)])
        self.assertEqual([r['ready_index'] for r in stream], [5, 9])
        with self.assertRaises(ValueError):
            sorted_stream([training(5, 5)])


if __name__ == '__main__':
    unittest.main()
