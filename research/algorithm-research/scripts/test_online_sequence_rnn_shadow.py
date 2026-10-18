import unittest

import evaluation_contract as scoring
import online_sequence_rnn_shadow as shadow


def target(key, kind='app'):
    bundle = key.removeprefix('app:') if kind == 'app' else 'example.browser'
    value = {'id': key, 'kind': kind, 'bundleID': bundle}
    if kind == 'page':
        value['locator'] = 'https://example.test/' + key
    return value


def observation(position, index, origin='app:A'):
    return {
        'position': position, 'index': index, 'ready_index': index + 3,
        'sample_id': f'sample:{position}', 'anchor_id': f'anchor:{position}',
        'session': 'session', 'origin': origin, 'timestamp': index * 1000,
        'target': target(origin),
    }


def record(position, input_index, ready_index, label='app:B', outcome='out', step=1):
    return {
        'position': position, 'input_index': input_index, 'ready_index': ready_index,
        'record': {
            'observed_step': step, 'action_key': label, 'weight': 1.,
            'event_id': outcome, 'arrival_timestamp_ms': (input_index + 1) * 1000,
            'target': target(label),
        },
    }


def model_record(origin, sequence, label, outcome):
    return {
        'origin': origin, 'sequence': tuple(sequence), 'label': label,
        'outcome_id': outcome, 'weight': 1., 'position': 0,
        'input_index': 1, 'ready_index': 2, 'timestamp': 1000,
        'outcome_timestamp': 2000.,
    }


class ProjectionTests(unittest.TestCase):
    def test_safe_projection_drops_background_candidates_and_evaluator_bundles(self):
        source = {
            'observations': [{**observation(0, 1), 'background_state': {'hidden': 'no'},
                              'candidateTargetIDs': ['app:secret'], 'annotation': {'leak': 1},
                              'future': ['app:later']}],
            'evaluation': [{
                'position': 0, 'sample_id': 'sample:0', 'truth_action': 'app:B',
                'truth_metadata': target('app:B'), 'evidence_grade': 'original', 'split': 'validation',
                'diagnosis_after_outcome': {'would_leak': True},
            }],
            'streams': {'hard_kind': [record(0, 1, 4)]},
        }
        safe = shadow._safe_data(source)
        self.assertEqual(set(safe['observations'][0]), set(shadow.SAFE_OBSERVATION_KEYS))
        self.assertNotIn('background_state', safe['observations'][0])
        self.assertNotIn('candidateTargetIDs', safe['observations'][0])
        self.assertNotIn('diagnosis_after_outcome', safe['evaluation'][0])
        self.assertEqual(set(safe['streams']['hard_kind'][0]['record']),
                         set(shadow.SAFE_STREAM_RECORD_KEYS))

    def test_sequence_input_is_a_whitelist_and_must_end_at_current_origin(self):
        rows = [observation(0, 10)]
        contexts = {0: {'through_index': 10, 'sequence': ['app:X', 'app:A'],
                        'background': {'must_not_flow': True}, 'future': ['app:B']}}
        inputs = shadow.sequence_inputs(rows, contexts)
        self.assertEqual(inputs[0]['sequence'], ('app:X', 'app:A'))
        self.assertEqual(set(inputs[0]), {'position', 'index', 'timestamp', 'origin',
                                          'sequence', 'sequence_length'})
        contexts[0]['sequence'] = ['app:X']
        with self.assertRaisesRegex(ValueError, 'end at the current'):
            shadow.sequence_inputs(rows, contexts)

    def test_head_zero_stream_is_shared_and_never_uses_future_steps_as_extra_votes(self):
        rows = [observation(0, 10)]
        inputs = shadow.sequence_inputs(rows, {0: {'through_index': 10, 'sequence': ['app:A']}})
        data = {'observations': rows, 'evaluation': [], 'streams': {'hard_kind': [
            record(0, 10, 14, 'app:B', 'first', step=1),
            record(0, 10, 14, 'app:C', 'second', step=2),
        ]}}
        stream, audit = shadow.supervision_stream(data, inputs)
        self.assertEqual(len(stream), 1)
        self.assertEqual(stream[0]['label'], 'app:B')
        self.assertEqual(audit['mature_relations'], 1)


class ModelTests(unittest.TestCase):
    def test_current_only_baseline_ignores_sequence(self):
        model = shadow.CurrentOnlyFrequency()
        model.learn(model_record('app:A', ('app:X', 'app:A'), 'app:B', 'one'))
        model.learn(model_record('app:A', ('app:Y', 'app:A'), 'app:C', 'two'))
        self.assertEqual(model.ranked({'origin': 'app:A', 'sequence': ('app:X', 'app:A')}),
                         model.ranked({'origin': 'app:A', 'sequence': ('app:Y', 'app:A')}))

    def test_online_rnn_can_separate_ordered_prefixes_incrementally(self):
        model = shadow.OnlineSequenceRNN({
            'id': 'fixture', 'hidden': 12, 'window': 2, 'input_dim': 64,
            'learning_rate': .08, 'l2': 0., 'seed': 7,
        })
        for index in range(50):
            model.learn(model_record('app:A', ('app:X', 'app:A'), 'app:B', f'x:{index}'))
            model.learn(model_record('app:A', ('app:Y', 'app:A'), 'app:C', f'y:{index}'))
        self.assertEqual(model.ranked({'sequence': ('app:X', 'app:A')})[0], 'app:B')
        self.assertEqual(model.ranked({'sequence': ('app:Y', 'app:A')})[0], 'app:C')

    def test_strict_maturity_means_same_cutoff_feedback_is_unavailable(self):
        inputs = {
            0: {'position': 0, 'index': 2, 'timestamp': 2000, 'origin': 'app:A',
                'sequence': ('app:A',), 'sequence_length': 1},
            1: {'position': 1, 'index': 5, 'timestamp': 5000, 'origin': 'app:A',
                'sequence': ('app:A',), 'sequence_length': 1},
            2: {'position': 2, 'index': 6, 'timestamp': 6000, 'origin': 'app:A',
                'sequence': ('app:A',), 'sequence_length': 1},
        }
        stream = [model_record('app:A', ('app:A',), 'app:B', 'outcome')]
        stream[0].update(position=0, input_index=2, ready_index=5)
        outputs, _, ledger = shadow.causal_replay(inputs, stream, [1, 2], shadow.CurrentOnlyFrequency())
        self.assertEqual(outputs[1]['predictions'], [])
        self.assertEqual(outputs[2]['predictions'], ['app:B'])
        self.assertIsNone(ledger[0]['latest_training_ready_index'])
        self.assertEqual(ledger[1]['latest_training_ready_index'], 5)


class EvaluationTests(unittest.TestCase):
    @staticmethod
    def metric(top1, top2, top4):
        return {
            'n': 5, 'top1_hits': top1, 'top2_hits': top2, 'top4_hits': top4,
            'exact': {'top1_hits': top1, 'top2_hits': top2, 'top4_hits': top4},
            'rank_score': {'total': float(top1 + top2 + top4)},
        }

    def test_top2_breaks_a_top4_selection_tie(self):
        first = {'order': 0, 'config': {'id': 'first'},
                 'validation': {'validation_original': self.metric(1, 2, 4)}}
        second = {'order': 1, 'config': {'id': 'second'},
                  'validation': {'validation_original': self.metric(1, 3, 4)}}
        self.assertEqual(shadow.choose_rnn([first, second])['config']['id'], 'second')

    def test_help_harm_reports_top2_for_accepted_and_exact_scoring(self):
        data = {
            'observations': [observation(0, 1)],
            'evaluation': [{
                'position': 0, 'sample_id': 'sample:0', 'truth_action': 'app:B',
                'truth_metadata': target('app:B'), 'evidence_grade': 'original', 'split': 'test',
            }],
        }
        rows = scoring.prepare_evaluation(data)
        before = {0: {'predictions': ['app:C'], 'rankable': ['app:C']}}
        after = {0: {'predictions': ['app:B'], 'rankable': ['app:B']}}
        summary, cases = shadow.paired_help_harm(rows, before, after)
        self.assertEqual(summary['accepted']['top2']['helped'], 1)
        self.assertEqual(summary['exact']['top2']['helped'], 1)
        self.assertEqual(cases[0]['accepted']['top2']['change'], 'helped')
        self.assertEqual(cases[0]['exact']['top2']['change'], 'helped')

    def test_rolling_intervals_preserve_the_final_endpoint(self):
        self.assertEqual(shadow.rolling_intervals(24), [])
        self.assertEqual(shadow.rolling_intervals(25), [(0, 25)])
        self.assertEqual(shadow.rolling_intervals(51), [(0, 25), (25, 50), (26, 51)])


if __name__ == '__main__':
    unittest.main()
