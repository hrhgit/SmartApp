import unittest
from copy import deepcopy
from types import SimpleNamespace

from evaluate_evidence_model import DENIED_BUNDLES
from trajectory_supervision import (PastQuickcheckHistory, TrajectoryEvents, action_key, derive_trajectory,
                                    past_quickcheck_history, policies)


def app(name):
    return {'id': 'app:' + name, 'kind': 'app', 'bundleID': name}


def page(name):
    return {'id': 'page:' + name, 'kind': 'page', 'bundleID': 'test.browser',
            'locator': 'https://example.invalid/' + name + '?x=1#section'}


def fixture():
    targets = [app('test.a'), app('test.b'), page('c'), app('test.a'), app('test.d')]
    times = [0, 2000, 4000, 6000, 8000, 31000]
    ids = ['a', 'b', 'c', 'return_a', 'd', 'ready']
    events = {key: {'id': key, 'sessionID': 's', 'monotonicMs': time,
                    'timestampMs': 100000 + time, 'kind': 'context.change', 'data': {}}
              for key, time in zip(ids, times)}
    indices = {key: i for i, key in enumerate(ids)}
    visits = {}
    for i, key in enumerate(ids[:-1]):
        visits[key] = {'session': 's', 'segment': 1, 'arrival_event_id': key,
            'arrival_start_ms': events[key]['timestampMs'], 'arrival_start_monotonic_ms': times[i],
            'initial_target': deepcopy(targets[i]), 'target': deepcopy(targets[i]),
            'previous_target': deepcopy(targets[i - 1]) if i else None,
            'next_target': deepcopy(targets[i + 1]) if i + 1 < len(targets) else None,
            'refinements': [], 'left_censored': False, 'complete': i < 4,
            'end_reason': 'different_observed_destination' if i < 4 else 'export_cutoff',
            'end_event_id': ids[i + 1], 'end_monotonic_ms': times[i + 1],
            'observed_seconds': (times[i + 1] - times[i]) / 1000}
    query = SimpleNamespace(anchor_id='a', index=0, ready_index=5, session='s',
                            current='app:test.a', current_app='test.a', next_event_id='b')
    return query, visits, events, indices


class TrajectoryTests(unittest.TestCase):
    def derive(self, policy='resolved', change=None, history=None):
        query, visits, events, indices = fixture()
        if change:
            change(query, visits, events, indices)
        return derive_trajectory(query, visits, events, indices, policy, history)

    def test_fixed_order_exact_resources_and_real_return(self):
        labels, audit = self.derive()
        self.assertEqual([r['action_key'] for r in labels], ['app:test.b', 'page:c', 'app:test.a'])
        self.assertEqual([r['observed_step'] for r in labels], [1, 2, 3])
        self.assertEqual(audit['step_mask'], [True] * 3)
        self.assertEqual([r['ready_index'] for r in labels], [5] * 3)
        self.assertEqual(labels[1]['target']['locator'], 'https://example.invalid/c?x=1#section')
        self.assertNotIn('app:test.d', [r['action_key'] for r in labels])

    def test_hard_drops_original_steps_without_refilling(self):
        labels, audit = self.derive('hard_kind')
        self.assertEqual([r['observed_step'] for r in labels], [2])
        self.assertEqual([r['observed_step'] for r in audit['dropped']], [1, 3])
        self.assertEqual(audit['step_mask'], [True] * 3)
        self.assertEqual(audit['training_step_mask'], [False, True, False])

    def test_soft_weights_only_completed_short_visits(self):
        labels, audit = self.derive('soft_kind')
        self.assertEqual([r['weight'] for r in labels], [.4, 1., .4])
        self.assertEqual(audit['dropped'], [])

    def test_resolution_is_one_step_at_exact_locator(self):
        def change(q, v, e, i):
            e['resolve'] = {'id': 'resolve', 'sessionID': 's', 'monotonicMs': 3000,
                            'timestampMs': 103000, 'kind': 'context.change', 'data': {}}
            for key in ('c', 'return_a', 'd', 'ready'):
                i[key] += 1
            i['resolve'] = 2
            q.ready_index += 1
            v['b']['refinements'] = [{'event_id': 'resolve', 'monotonic_ms': 3000,
                                     'target': page('resolved')}]
        labels, _ = self.derive(change=change)
        self.assertEqual(len(labels), 3)
        self.assertEqual(labels[0]['action_key'], 'page:resolved')
        self.assertEqual(labels[0]['arrival_event_id'], 'b')
        self.assertEqual(labels[0]['target_evidence_event_id'], 'resolve')
        self.assertEqual(labels[0]['dwell']['seconds'], 1.)

    def test_refinement_after_original_horizon_cannot_change_label(self):
        def change(q, v, e, i):
            e['future'] = {'id': 'future', 'sessionID': 's', 'monotonicMs': 32000,
                           'timestampMs': 132000}
            i['future'] = 6
            v['b']['refinements'] = [{'event_id': 'future', 'monotonic_ms': 32000,
                                     'target': page('future-secret')}]
        labels, _ = self.derive(change=change)
        self.assertEqual(labels[0]['action_key'], 'app:test.b')
        self.assertEqual(labels[0]['refinement_event_ids'], [])

    def test_future_index_refinement_is_ignored_even_with_early_timestamp(self):
        def change(q, v, e, i):
            e['future'] = {'id': 'future', 'sessionID': 's', 'monotonicMs': 2500,
                           'timestampMs': 102500}
            i['future'] = 6
            v['b']['refinements'] = [{'event_id': 'future', 'monotonic_ms': 2500,
                                     'target': page('future-secret')}]
        labels, _ = self.derive(change=change)
        self.assertEqual(labels[0]['action_key'], 'app:test.b')

    def test_censored_exit_masks_followups_without_removing_known_arrival(self):
        for reason in ('capture_gap', 'unadmitted_or_synthetic_target', 'foreground_source_unavailable',
                       'session_boundary', 'same_app_granularity_loss'):
            def change(q, v, e, i):
                v['b']['complete'] = False
                v['b']['end_reason'] = reason
            labels, audit = self.derive('hard_kind', change)
            self.assertEqual(len(labels), 1, reason)
            self.assertEqual(labels[0]['weight'], 1.)
            self.assertIsNone(labels[0]['dwell']['seconds'])
            self.assertEqual(audit['step_mask'], [True, False, False])
            self.assertEqual(audit['masked_steps'], [2, 3])

    def test_input_boundary_prevents_jump_to_later_visit(self):
        def change(q, v, e, i):
            v['a']['end_reason'] = 'capture_gap'
            v['a']['complete'] = False
        labels, audit = self.derive(change=change)
        self.assertEqual(labels, [])
        self.assertEqual(audit['step_mask'], [False] * 3)

    def test_future_input_exit_reason_is_not_disclosed_as_current_quality(self):
        def change(q, v, e, i):
            v['a']['end_reason'] = 'capture_gap'
            v['a']['end_event_id'] = 'ready'
            v['a']['end_monotonic_ms'] = 31000
        labels, audit = self.derive(change=change)
        self.assertEqual(labels, [])
        self.assertEqual(audit['stop_reason'], 'original_30s_horizon')

    def test_late_arrival_is_masked_without_reading_its_label(self):
        def change(q, v, e, i):
            v['c']['arrival_start_monotonic_ms'] = 31001
            v['b']['end_monotonic_ms'] = 31001
            v['c']['initial_target'] = app(next(iter(DENIED_BUNDLES)))
        labels, audit = self.derive(change=change)
        self.assertEqual(len(labels), 1)
        self.assertEqual(audit['stop_reason'], 'original_30s_horizon')
        self.assertIsNone(labels[0]['dwell']['lower_bound_seconds'])

    def test_input_resolution_connects_to_original_episode(self):
        q, visits, events, indices = fixture()
        events['resolve_a'] = {'id': 'resolve_a', 'sessionID': 's', 'monotonicMs': 500,
            'timestampMs': 100500, 'data': {'resolvesContextEventID': 'a'}}
        for key in ('b', 'c', 'return_a', 'd', 'ready'):
            indices[key] += 1
        indices['resolve_a'] = 1
        q.anchor_id, q.index, q.ready_index = 'resolve_a', 1, 6
        labels, _ = derive_trajectory(q, visits, events, indices)
        self.assertEqual([r['action_key'] for r in labels], ['app:test.b', 'page:c', 'app:test.a'])

    def test_unknown_dwell_is_not_short_or_negative(self):
        def change(q, v, e, i):
            v['b']['end_event_id'] = None
            v['b']['end_monotonic_ms'] = None
            v['b']['complete'] = False
        labels, audit = self.derive('hard_kind', change)
        self.assertEqual(len(labels), 1)
        self.assertIsNone(labels[0]['dwell']['lower_bound_seconds'])
        self.assertFalse(audit['absent_steps_are_negative'])

    def test_denied_destination_stops_chain(self):
        def change(q, v, e, i):
            v['c']['initial_target'] = app(next(iter(DENIED_BUNDLES)))
        labels, audit = self.derive(change=change)
        self.assertEqual(len(labels), 1)
        self.assertEqual(audit['step_mask'], [True, False, False])
        self.assertIn('denied', audit['stop_reason'])

    def test_changed_segment_and_left_censor_are_boundaries(self):
        for field, value in [('segment', 2), ('left_censored', True), ('session', 'other')]:
            def change(q, v, e, i):
                v['c'][field] = value
            labels, audit = self.derive(change=change)
            self.assertEqual(len(labels), 1)
            self.assertEqual(audit['masked_steps'], [2, 3])

    def test_app_window_action_fallback_does_not_change_concrete_resources(self):
        self.assertEqual(action_key({'kind': 'window', 'id': 'transient:123',
                                    'bundleID': 'test.editor'}), 'app:test.editor')
        self.assertIsNone(action_key({'kind': 'window', 'id': 'transient:123'}))
        self.assertEqual(action_key({'kind': 'directory', 'id': 'directory:exact',
                                    'locator': 'file:///tmp/example-folder/'}), 'directory:exact')
        self.assertIsNone(action_key({'kind': 'page', 'id': 'page:bad',
                                     'locator': 'javascript:alert(1)'}))

    def test_history_fallback_and_no_post_input_history(self):
        labels, _ = self.derive('soft_quickcheck')
        self.assertEqual(labels[0]['weight'], .4)
        self.assertIn('fallback', labels[0]['basis'])
        with self.assertRaisesRegex(ValueError, 'post-input'):
            self.derive('soft_quickcheck', history={'origin': 'app:test.a', 'n': 5,
                'quick_returns': 5, 'through_index': 1})
        with self.assertRaisesRegex(ValueError, 'different input origin'):
            self.derive('soft_quickcheck', history={'origin': 'app:other', 'n': 0,
                'quick_returns': 0, 'through_index': None})

    def test_past_same_origin_rate_can_soften_short_weight(self):
        labels, audit = self.derive('soft_quickcheck', history={'origin': 'app:test.a',
            'n': 10, 'quick_returns': 5, 'through_index': 0})
        self.assertAlmostEqual(labels[0]['weight'], .7)
        self.assertEqual(audit['history']['n'], 10)

    def test_current_future_return_is_not_used_as_past_habit(self):
        q, visits, events, indices = fixture()
        # Make B a real short A-B-A. Its exit is still AFTER input q.index.
        visits['b']['next_target'] = app('test.a')
        history = PastQuickcheckHistory(visits.values(), indices)
        labels, audit = derive_trajectory(q, visits, events, indices, 'soft_quickcheck', history)
        self.assertEqual(audit['history']['n'], 0)
        self.assertEqual(labels[0]['weight'], .4)
        q.index = 2
        snapshot = history.advance(q)
        self.assertEqual(snapshot['n'], 1)
        self.assertEqual(snapshot['quick_returns'], 1)
        q.index = 1
        with self.assertRaisesRegex(ValueError, 'backward'):
            history.advance(q)

    def test_convenience_history_matches_incremental_snapshot(self):
        q, visits, _, indices = fixture()
        q.index = 4
        self.assertEqual(past_quickcheck_history(visits.values(), indices, q),
                         PastQuickcheckHistory(visits.values(), indices).advance(q))

    def test_event_wrapper_matches_plain_dictionary(self):
        q, v, e, i = fixture()
        self.assertEqual(derive_trajectory(q, v, e, i),
                         derive_trajectory(q, v, TrajectoryEvents(e, i), i))

    def test_explicit_revocation_censors_even_if_episode_metadata_missed_it(self):
        def change(q, v, e, i):
            for key in ('c', 'return_a', 'd', 'ready'):
                i[key] += 1
            i['revoke'] = 2
            q.ready_index += 1
            e['revoke'] = {'id': 'revoke', 'sessionID': 's', 'monotonicMs': 3000,
                'timestampMs': 103000, 'kind': 'target.update', 'data': {'target': app('test.b')}}
        labels, audit = self.derive('hard_kind', change)
        self.assertEqual(len(labels), 1)
        self.assertEqual(labels[0]['weight'], 1.)
        self.assertIsNone(labels[0]['dwell']['seconds'])
        self.assertEqual(audit['stop_reason'], 'explicit_active_target_revocation')

    def test_refinement_after_explicit_gap_is_not_used(self):
        q, v, e, i = fixture()
        for key in ('c', 'return_a', 'd', 'ready'):
            i[key] += 2
        i['gap'], i['refine'] = 2, 3
        q.ready_index += 2
        e['gap'] = {'id': 'gap', 'sessionID': 's', 'monotonicMs': 2500,
                    'timestampMs': 102500, 'kind': 'capture.gap', 'data': {}}
        e['refine'] = {'id': 'refine', 'sessionID': 's', 'monotonicMs': 3000,
                       'timestampMs': 103000, 'kind': 'context.change', 'data': {}}
        v['b']['refinements'] = [{'event_id': 'refine', 'monotonic_ms': 3000, 'target': page('after-gap')}]
        labels, audit = derive_trajectory(q, v, e, i)
        self.assertEqual(len(labels), 1)
        self.assertEqual(labels[0]['action_key'], 'app:test.b')
        self.assertEqual(audit['stop_reason'], 'explicit_observation_boundary')

    def test_inputs_not_mutated(self):
        q, visits, events, indices = fixture()
        before = deepcopy((vars(q), visits, events, indices))
        for policy in policies():
            derive_trajectory(q, visits, events, indices, policy)
        self.assertEqual(before, (vars(q), visits, events, indices))


if __name__ == '__main__':
    unittest.main()
