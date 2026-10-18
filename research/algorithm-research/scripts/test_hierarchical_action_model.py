from copy import deepcopy
import unittest

from hierarchical_action_model import HierarchicalActionModel, augment_training_stream


class RecordingHead:
    def __init__(self, scores):
        self.learned = []
        self.queries = []
        self.values = scores

    def learn(self, record):
        self.learned.append(deepcopy(record))

    def scores(self, query):
        self.queries.append(deepcopy(query))
        return dict(self.values)


def record(label='page:a', owner='app:browser'):
    return {'origin': 'app:editor', 'app': 'editor', 'timestamp': 1000,
            'outcome_timestamp': 1500, 'outcome_id': 'arrival', 'label': label,
            'x': {'origin=app:editor': 1}, 'destination_app': owner}


def query():
    return {'origin': 'app:editor', 'app': 'editor', 'timestamp': 5000,
            'x': {'origin=app:editor': 1}, 'labels': ['page:a', 'app:browser', 'app:editor'],
            'candidate_last_seen': {'page:a': 100, 'app:browser': 200},
            'candidate_key_order': {'page:a': 'page:a', 'app:browser': 'app:browser'}}


def stream_and_ledger():
    item = record(); item.pop('destination_app')
    stream = [{'position': 0, 'input_index': 1, 'outcome_index': 2, 'ready_index': 3,
               'record': item}]
    ledger = [{'position': 0, 'input_through_event_index': 1, 'label_ready_event_index': 3,
               'target_event_id': 'arrival', 'target': {'id': 'page:a', 'kind': 'page',
                   'bundleID': 'browser', 'locator': 'https://example.invalid/a'},
               'training_included_after_dedup': True}]
    return stream, ledger


class HierarchicalActionTests(unittest.TestCase):
    def test_one_arrival_updates_each_head_once_without_owner_feature_leak(self):
        left, right = RecordingHead({}), RecordingHead({})
        model = HierarchicalActionModel(left, right, .25)
        original = record()
        model.learn(original); model.learn(original)
        self.assertEqual(len(left.learned), 1)
        self.assertEqual(len(right.learned), 1)
        self.assertEqual(left.learned[0]['label'], 'page:a')
        self.assertEqual(right.learned[0]['label'], 'app:browser')
        self.assertNotIn('destination_app', left.learned[0])
        self.assertNotIn('destination_app', right.learned[0])
        self.assertEqual(right.learned[0]['x'], original['x'])
        self.assertEqual(original, record())

    def test_missing_owner_skips_auxiliary_and_never_uses_origin_app(self):
        left, right = RecordingHead({}), RecordingHead({})
        model = HierarchicalActionModel(left, right, .25)
        model.learn(record(owner=None))
        self.assertEqual(len(left.learned), 1)
        self.assertEqual(right.learned, [])

    def test_distinct_origins_keep_existing_replay_identity(self):
        left, right = RecordingHead({}), RecordingHead({})
        model = HierarchicalActionModel(left, right, .25)
        model.learn(record())
        other = record(); other['origin'] = 'another-current'
        model.learn(other)
        self.assertEqual((len(left.learned), len(right.learned)), (2, 2))

    def test_conflicting_duplicate_supervision_is_rejected_before_update(self):
        left, right = RecordingHead({}), RecordingHead({})
        model = HierarchicalActionModel(left, right, .25)
        model.learn(record())
        with self.assertRaises(ValueError): model.learn(record(label='page:b'))
        with self.assertRaises(ValueError): model.learn(record(owner='app:other'))
        self.assertEqual((len(left.learned), len(right.learned)), (1, 1))

    def test_auxiliary_query_only_contains_current_app_candidates(self):
        left = RecordingHead({'page:a': 2, 'app:browser': 1})
        right = RecordingHead({'app:browser': 5, 'page:unobserved': 100, 'app:unobserved': 50})
        model = HierarchicalActionModel(left, right, .25)
        q = query(); q['destination_app'] = 'app:future'; q['truth'] = 'future'
        scores = model.scores(q)
        self.assertEqual(right.queries[0]['labels'], ['app:browser', 'app:editor'])
        self.assertNotIn('destination_app', left.queries[0])
        self.assertNotIn('destination_app', right.queries[0])
        self.assertNotIn('truth', right.queries[0])
        self.assertNotIn('page:a', right.queries[0]['candidate_last_seen'])
        self.assertNotIn('app:unobserved', scores)
        self.assertNotIn('page:unobserved', scores)
        self.assertAlmostEqual(scores['page:a'], .75)
        self.assertAlmostEqual(scores['app:browser'], .75 / 2 + .25)

    def test_weight_zero_is_exact_primary_score_control(self):
        left = RecordingHead({'page:a': -.1, 'app:browser': 0.0})
        right = RecordingHead({'app:browser': 1000})
        model = HierarchicalActionModel(left, right, 0)
        self.assertEqual(model.scores(query()), left.values)
        self.assertEqual(right.queries, [])

    def test_ledger_join_preserves_input_and_unknown_owner(self):
        stream, ledger = stream_and_ledger()
        original = deepcopy(stream)
        augmented, audit = augment_training_stream(stream, ledger)
        self.assertEqual(augmented[0]['record']['destination_app'], 'app:browser')
        self.assertEqual(stream, original)
        self.assertEqual(augmented[0]['record']['x'], original[0]['record']['x'])
        ledger[0]['target'].pop('bundleID')
        missing, audit = augment_training_stream(stream, ledger)
        self.assertIsNone(missing[0]['record']['destination_app'])
        self.assertEqual(audit['without_owner'], 1)

    def test_ledger_rejects_wrong_arrival_maturity_and_concrete_label(self):
        for field, value in [('target_event_id', 'different'), ('label_ready_event_index', 99),
                             ('position', 9), ('input_through_event_index', 88)]:
            with self.subTest(field=field):
                stream, ledger = stream_and_ledger(); ledger[0][field] = value
                with self.assertRaises(ValueError): augment_training_stream(stream, ledger)
        stream, ledger = stream_and_ledger(); ledger[0]['target']['id'] = 'page:wrong'
        with self.assertRaises(ValueError): augment_training_stream(stream, ledger)

    def test_conflicting_owner_is_unknown_and_window_label_can_be_app(self):
        stream, ledger = stream_and_ledger()
        ledger[0]['target']['_bundle'] = 'different-owner'
        augmented, audit = augment_training_stream(stream, ledger)
        self.assertIsNone(augmented[0]['record']['destination_app'])
        self.assertEqual(audit['owner_source:conflicting_observed_owners'], 1)
        ledger[0]['target'].update(kind='window', id='window-instance', _bundle='browser')
        stream[0]['record']['label'] = 'app:browser'
        augmented, _ = augment_training_stream(stream, ledger)
        self.assertEqual(augmented[0]['record']['destination_app'], 'app:browser')


if __name__ == '__main__':
    unittest.main()
