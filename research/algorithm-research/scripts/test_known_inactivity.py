import unittest
from copy import deepcopy
from types import SimpleNamespace

from known_inactivity import InactivityClock, annotate_events
from describe_dwell import extract_visits
from temporal_context import TemporalHistory
from trajectory_supervision import derive_trajectory, TrajectoryEvents


PAUSE_ID = '5B8FAB08-C213-4105-B28B-48BB173D07B4'


def event(key, time, kind, data=None):
    return {'id': key, 'sessionID': 's', 'timestampMs': time + 100000, 'monotonicMs': time,
            'kind': kind, 'data': data or {}, 'source': 'workspace'}


def pause(phase, time):
    return event(phase, time, 'capture.gap', {'pauseVersion': 'known-inactivity-v1',
        'pauseID': PAUSE_ID, 'pausePhase': phase, 'pauseReasons': ['system_sleep']})


def fixture():
    events = []
    for key in 'abc':
        target = {'id': 'app:' + key, 'kind': 'app', 'name': key, 'bundleID': key,
                  'admission': {'basis': 'standalone_application', 'policyVersion': 'target-eligibility-v4'}}
        events.append(event('target:' + key, 0, 'target.update', {'target': target}))
    events += [event('a', 1000, 'context.change', {'targetID': 'app:a'}),
               event('b', 2000, 'context.change', {'targetID': 'app:b'}),
               pause('begin', 2500), pause('end', 3602500),
               event('resume', 3603000, 'context.change', {'targetID': 'app:b',
                     'reason': 'resolution_update', 'resolvesContextEventID': 'b', 'continuityPauseID': PAUSE_ID}),
               event('c', 3603500, 'context.change', {'targetID': 'app:c'}),
               event('ready', 3631000, 'heartbeat')]
    return events


class KnownInactivityTests(unittest.TestCase):
    def test_excluded_foreground_is_masked_and_does_not_become_observed_dwell(self):
        scope_id = 'E3E423E3-E735-4665-94C5-6C9A696979EF'
        events = [event('a', 0, 'context.change', {'targetID': 'app:a'}),
                  event('begin', 100, 'observation.scope', {'scopeVersion': 'excluded-foreground-v1',
                      'scopeID': scope_id, 'phase': 'begin', 'category': 'excluded_app'}),
                  event('during', 30000, 'heartbeat'),
                  event('end', 60000, 'observation.scope', {'scopeVersion': 'excluded-foreground-v1',
                      'scopeID': scope_id, 'phase': 'end', 'category': 'excluded_app'}),
                  event('b', 61000, 'context.change', {'targetID': 'app:b'})]
        observed = annotate_events(events)
        self.assertEqual([e['continuityDisposition'] for e in observed],
                         ['normal', 'masked', 'suspended', 'masked', 'normal'])
        self.assertEqual(observed[-1]['activeMonotonicMs'], 1100)
        self.assertEqual(events[-1]['monotonicMs'], 61000)

    def test_raw_timestamps_remain_unchanged_and_active_clock_excludes_sleep(self):
        events = fixture()
        original = deepcopy(events)
        observed = annotate_events(events)
        self.assertEqual(events, original)
        self.assertEqual(observed[-1]['activeMonotonicMs'], 31000)
        self.assertEqual(observed[-3]['continuityDisposition'], 'resumed')
        self.assertEqual(observed[-3]['timestampMs'], original[-3]['timestampMs'])

    def test_dwell_and_trajectory_keep_next_step_but_not_sleep_duration(self):
        events = fixture()
        visits, apps, _ = extract_visits(events, [], set())
        self.assertEqual([v['arrival_event_id'] for v in visits], ['a', 'b', 'c'])
        self.assertEqual(visits[1]['observed_seconds'], 1.5)
        self.assertTrue(visits[1]['complete'])
        self.assertEqual(apps[1]['observed_seconds'], 1.5)
        indices = {e['id']: i for i, e in enumerate(events)}
        query = SimpleNamespace(anchor_id='a', index=indices['a'], ready_index=indices['ready'],
                                session='s', current='app:a', current_app='a', next_event_id='b')
        labels, _ = derive_trajectory(query, {v['arrival_event_id']: v for v in visits},
                                      {e['id']: e for e in events}, indices, 'hard_kind')
        self.assertTrue(labels)
        self.assertEqual(labels[0]['action_key'], 'app:c')

    def test_temporal_history_counts_resume_once_and_never_counts_sleep_dwell(self):
        events = fixture()
        h = TemporalHistory()
        for i, e in enumerate(events):
            h.consume(e, i)
        self.assertEqual(h.audit['confirmed_arrivals'], 2)
        self.assertEqual([e['arrival_id'] for e in h.sequence], ['a', 'b', 'c'])
        dwell = sum(span['seconds'] for span in h.spans if span['target']['id'] == 'app:b')
        self.assertLessEqual(dwell, 1.5)
        self.assertGreater(dwell, 0)

    def test_unpaired_or_changed_resume_is_a_boundary(self):
        for mode in ['unpaired', 'changed', 'privacy']:
            events = fixture()
            if mode == 'unpaired':
                events = [e for e in events if e['id'] != 'end']
            if mode == 'changed':
                next(e for e in events if e['id'] == 'resume')['data']['targetID'] = 'app:a'
            if mode == 'privacy':
                events.insert(7, event('private', 3602600, 'capture.gap', {'reason': 'privacy'}))
            visits, _, _ = extract_visits(events, [], set())
            b = next(v for v in visits if v['arrival_event_id'] == 'b')
            self.assertFalse(b['complete'])


if __name__ == '__main__':
    unittest.main()
