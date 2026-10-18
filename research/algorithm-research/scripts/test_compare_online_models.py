"""Independent synthetic checks for the causal comparison and fixed benchmark."""
import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import compare_online_models as comparison
import evaluate_context_knn as core
from test_evaluate_context_knn import observation, target


class RecordingLearner:
    """A learner with visible update effects, independent of the model suites."""

    def __init__(self):
        self.learned = []

    def learn(self, record):
        self.learned.append(copy.deepcopy(record))

    def scores(self, query):
        return {row['label']: index + 1 for index, row in enumerate(self.learned)}


def stream_row(position=0, ready_index=30, label='B', outcome_id='arrival'):
    return {'position': position, 'input_index': 1, 'ready_index': ready_index,
            'outcome_index': 20,
            'record': {'origin': 'A', 'app': 'A', 'timestamp': 1000,
                       'x': {'origin=A': 1.}, 'label': label,
                       'outcome_timestamp': 20000, 'outcome_id': outcome_id}}


def scored_row(position, truth='B', eligible=True, present=True, grade='single',
               split='test', strict=False):
    return {'position': position, 'sample_id': f'sample-{position}', 'split': split,
            'observed_outcome': {'key': truth},
            'diagnosis_after_outcome': {'eligible': eligible, 'grade': grade,
                                       'candidate_present': present},
            'evaluation': {'strict_input_and_label_coverage': strict}}


class ComparisonCausalityTests(unittest.TestCase):
    def test_allowlist_ignores_labels_quality_history_and_display_metadata(self):
        query = observation(40)
        query.last_visits = {'B': 20000, 'C': 21000}
        before = comparison.model_input(query)
        changed = copy.deepcopy(query)
        changed.truth = 'secret-future-answer'
        changed.truth_meta = {'locator': 'https://future.invalid/private'}
        changed.strict = not query.strict
        changed.coverage = 'future-coverage'
        changed.source_quality = {'future': 1.}
        changed.foreground = {'future-foreground': 1.}
        changed.history = {'future-history': 1.}
        changed.history_keys = ('future-history',)
        changed.background = {'future-background': {'secret': 1.}}
        changed.next_event_id = 'future-event'
        for metadata in changed.candidates.values():
            metadata['name'] = 'future-display-name'
            metadata['locator'] = 'https://future.invalid/display-only'
        self.assertEqual(comparison.model_input(changed), before)
        self.assertEqual(set(before[0]), {'origin', 'app', 'timestamp', 'x',
                                         'labels', 'candidate_last_seen'})
        self.assertEqual(before[0]['x'],
                         {'origin=A': 1., 'app=A': 1., 'kind=A': 1., 'bias': 1.})

    def test_future_arrival_recency_is_rejected(self):
        query = observation(40)
        query.last_visits['B'] = query.timestamp + 1
        with self.assertRaisesRegex(ValueError, 'Future candidate arrival'):
            comparison.model_input(query)

    def test_query_at_exact_maturity_index_cannot_learn_the_label(self):
        items = [observation(1), observation(30), observation(31)]
        inputs, mappings = {}, {}
        for index in (1, 2):
            inputs[index], mappings[index] = comparison.model_input(items[index])
        model = RecordingLearner()
        predictions, timing = comparison.replay(core, model, items, [stream_row()],
                                                 inputs, mappings)
        self.assertEqual(predictions[1]['predictions'], [])
        self.assertEqual(predictions[1]['learned_unique_outcomes'], 0)
        self.assertEqual(predictions[2]['predictions'], ['B'])
        self.assertEqual(predictions[2]['learned_unique_outcomes'], 1)
        self.assertEqual(len(model.learned), 1)
        self.assertEqual(timing['learn']['n'], 1)

    def test_mutating_unmatured_answer_preserves_all_earlier_predictions(self):
        items = [observation(1), observation(30), observation(31)]
        inputs, mappings = {}, {}
        for index in (1, 2):
            inputs[index], mappings[index] = comparison.model_input(items[index])
        first, _ = comparison.replay(core, RecordingLearner(), items,
                                    [stream_row(label='B')], inputs, mappings)
        second, _ = comparison.replay(core, RecordingLearner(), items,
                                     [stream_row(label='C')], inputs, mappings)
        self.assertEqual(first[1], second[1])
        self.assertEqual(first[2]['predictions'], ['B'])
        self.assertEqual(second[2]['predictions'], ['C'])

    def test_future_outcome_cannot_be_hidden_behind_past_ready_index(self):
        items = [observation(1), observation(30)]
        inp, mapping = comparison.model_input(items[1])
        row = stream_row(ready_index=29)
        row['outcome_index'] = 30
        with self.assertRaises(AssertionError):
            comparison.replay(core, RecordingLearner(), items, [row],
                              {1: inp}, {1: mapping})

    def test_semantic_window_mapping_requires_exactly_one_current_window(self):
        query = observation(40)
        query.candidates = {
            'window@s:w1': {**target('w1', 'window'), '_bundle': 'example.app'},
            'window@s:w2': {**target('w2', 'window'), '_bundle': 'example.app'},
        }
        inp, mapping = comparison.model_input(query)
        self.assertEqual(inp['labels'], [])
        self.assertEqual(mapping, {})
        query.candidates.pop('window@s:w2')
        inp, mapping = comparison.model_input(query)
        self.assertEqual(mapping, {'window_class:example.app': 'window@s:w1'})
        raw, shown = comparison.project_scores(core, {'window_class:example.app': 1.},
                                               mapping, query.candidates)
        self.assertEqual(raw, ['window@s:w1'])
        self.assertEqual(shown, ['window@s:w1'])

    def test_projection_preserves_hierarchy_and_four_slot_limit(self):
        query = observation(40)
        query.candidates = {
            'app:X': {**target('app:X'), '_bundle': 'X'},
            'window@s:w': {**target('w', 'window'), '_bundle': 'X'},
            **{key: {**target(key), '_bundle': key} for key in 'BCDEF'},
        }
        _, mapping = comparison.model_input(query)
        scores = {'app:X': 20, 'window_class:X': 19,
                  **{key: 10-index for index, key in enumerate('BCDEF')},
                  'unobserved': 1000}
        raw, shown = comparison.project_scores(core, scores, mapping, query.candidates)
        self.assertEqual(raw[:2], ['app:X', 'window@s:w'])
        self.assertEqual(shown, ['app:X', 'B', 'C', 'D'])
        self.assertEqual(len(shown), len(set(shown)))
        self.assertNotIn('unobserved', raw)

    def test_projection_rejects_nonfinite_scores(self):
        query = observation(40)
        _, mapping = comparison.model_input(query)
        for value in (float('nan'), float('inf'), '1.0'):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    comparison.project_scores(core, {'B': value}, mapping, query.candidates)

    def test_fixed_evidence_denominator_retains_missing_candidates_and_abstentions(self):
        rows = [scored_row(0, strict=True, grade='strong', split='validation'),
                scored_row(1, truth='missing', present=False, grade='repeated'),
                scored_row(2), scored_row(3, eligible=False, grade='none')]
        predictions = {0: {'predictions': ['B']}, 1: {'predictions': ['C']},
                       2: {'predictions': []}, 3: {'predictions': ['B']}}
        groups = comparison.score_groups(rows, predictions)
        self.assertEqual(groups['supported']['n'], 3)
        self.assertEqual(groups['supported']['top4_hits'], 1)
        self.assertEqual(groups['supported']['top4'], 1/3)
        self.assertEqual(groups['supported']['candidate_missing'], 1)
        self.assertEqual(groups['covered_supported']['n'], 2)
        self.assertEqual(groups['strict_supported']['n'], 1)
        self.assertEqual(groups['late_supported']['n'], 2)
        predictions[2]['predictions'] = ['B']
        changed = comparison.score_groups(rows, predictions)
        self.assertEqual(changed['supported']['n'], 3)
        self.assertEqual(changed['grades']['single']['n'], groups['grades']['single']['n'])
        self.assertEqual(changed['supported']['top4_hits'], 2)

    def test_missing_prediction_is_not_silently_dropped(self):
        with self.assertRaises(KeyError):
            comparison.score_groups([scored_row(0)], {})

    def test_earlier_selection_tie_break_cannot_depend_on_later_score_order(self):
        """An earlier-only selection must be invariant to held-later scores."""
        def result(cleaning, later_hits):
            return {'training_cleaning': cleaning, 'config': {'id': 'same-model'},
                    'scores': {'5': {
                        'early_supported': {'top4_hits': 10, 'top1_hits': 5, 'mrr4': .5},
                        'supported': {'top4_hits': later_hits + 10, 'top1_hits': 5,
                                      'mrr4': .5}}}}
        def choose_early(results):
            full_order = sorted(results, key=comparison.selection_key)
            early_order = sorted(full_order,
                                 key=lambda row: comparison.selection_key(row, 'early_supported'))
            return early_order[0]['training_cleaning']
        first = [result('5', 20), result('raw', 0)]
        changed_later_only = [result('5', 0), result('raw', 20)]
        self.assertEqual(choose_early(first), choose_early(changed_later_only))

    def test_loader_deduplicates_actual_outcome_per_origin_before_any_model(self):
        """Exercise the actual stream constructor, not a copied dedup algorithm."""
        items = [observation(index, truth='C') for index in range(3)]
        for index, query in enumerate(items):
            query.next_event_id = 'event3'
            query.ready_index = 4 + index
            query.current = 'A' if index < 2 else 'B'
            query.current_app = query.current
        events = [{'id': f'event{i}', 'kind': 'context.change', 'timestampMs': i*1000}
                  for i in range(8)]
        ledger = [{'position': index, 'sample_id': query.sample_id,
                   'label_ready_event_index': query.ready_index, 'status': 'stable_target',
                   'target': target('C'), 'target_event_id': 'event3'}
                  for index, query in enumerate(items)]
        rows = []
        for index, query in enumerate(items):
            row = scored_row(index, truth='C')
            row['sample_id'] = query.sample_id
            rows.append(row)
        fake_core = SimpleNamespace(
            verified_snapshot=lambda path: ({}, None),
            load_lines=lambda path: events if path.name == 'events.jsonl' else [],
            make_observations=lambda *args: (items, {}, {}, {}, {}), key_for=core.key_for)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            expansion, audit, base = (root/name for name in ('expansion', 'audit', 'base'))
            expansion.mkdir(); base.mkdir()
            (base/'labels-private.json').write_text(json.dumps({'3': ledger, '5': ledger}))
            membership = {threshold: {'single': [q.sample_id for q in items]}
                          for threshold in ('3', '5')}
            (expansion/'benchmark-membership.json').write_text(
                json.dumps({'sample_ids_by_grade': membership}))
            metas = {
                expansion: {'source_run': str(audit), 'source_artifact_hashes': {}},
                audit: {'cleaning_run': str(base), 'cleaning_artifact_hashes': {},
                        'input_hashes': {}},
                base: {'snapshot': str(root/'snapshot'), 'input_hashes': {},
                       'excluded_sessions': []},
            }
            with patch.object(comparison, 'verify_run', side_effect=lambda p: (metas[p], {})), \
                 patch.object(comparison, 'frozen_core', return_value=fake_core), \
                 patch.object(comparison, 'read_rows',
                              side_effect=lambda p: [] if p.name.startswith('validation') else rows):
                result = comparison.load_dataset(expansion)
        streams, provenance = result[3], result[-1]
        for threshold in ('raw', '3', '5'):
            with self.subTest(threshold=threshold):
                self.assertEqual([r['position'] for r in streams[threshold]], [0, 2])
                self.assertEqual([r['record']['origin'] for r in streams[threshold]], ['A', 'B'])
                self.assertEqual(provenance['cleaning_audits'][threshold]['duplicates_removed'], 1)


if __name__ == '__main__':
    unittest.main()
