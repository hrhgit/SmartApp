from dataclasses import fields
from copy import deepcopy
import unittest

import evaluate_context_knn as core
import explore_training_cleaning as cleaning


def target(name, kind='page'):
    return {'id': name, 'kind': kind, 'locator': 'https://example.invalid/' + name,
            'bundleID': 'browser'}


class CleaningExplorationTests(unittest.TestCase):
    def setUp(self):
        values = {f.name: None for f in fields(core.Observation)}
        values.update(anchor_id='anchor', next_event_id='B', session='s', current='A', truth='old',
                      ready_index=100, index=0, timestamp=0, current_app='browser')
        self.query = core.Observation(**values)
        self.events = {'anchor': {'monotonicMs': 0}}
        self.indices = {'anchor': 0, 'B': 1, 'C': 2, 'A': 3, 'D': 4, 'refine': 5,
                        'late_refine': 101}
        self.configs = {p['id']: p for p in cleaning.policies()}

    def visit(self, name='B', start=1000, end=2000, next_id='C', complete=True,
              category='same_tab_navigation', before='A', after='C', refinements=None):
        return {'session': 's', 'arrival_event_id': name, 'left_censored': False,
                'arrival_start_monotonic_ms': start, 'initial_target': target(name),
                'target': target(name), 'refinements': refinements or [],
                'end_monotonic_ms': end, 'end_event_id': next_id, 'complete': complete,
                'observed_seconds': (end - start) / 1000 if end is not None else None,
                'navigation_class': category, 'previous_target': target(before) if before else None,
                'next_target': target(after) if after else None}

    def derive(self, policy, visits, history=None):
        return cleaning.derive_label(core, self.query, visits, self.events, self.indices,
                                     self.configs[policy], history)

    def test_grid_is_fixed_nineteen_unique_named_policies(self):
        self.assertEqual(len(self.configs), 19)
        self.assertTrue(all(p['name_zh'] for p in self.configs.values()))
        self.assertEqual({self.configs[p]['mode'] for p in ('raw', 'resolved', '3')},
                         {'raw', 'resolved_arrival', 'frozen'})

    def test_navigation_only_merges_confirmed_same_tab_short_visit(self):
        visits = {'B': self.visit(), 'C': self.visit('C', 2000, 20000, 'D', after='D')}
        q, decision = self.derive('nav3', visits)
        self.assertEqual(q.truth, 'C')
        self.assertEqual(q.ready_index, self.query.ready_index)
        self.assertEqual(decision['skipped'][0]['category'], 'navigation_intermediate')

    def test_same_site_or_app_does_not_infer_navigation_when_identity_missing(self):
        visit = self.visit(category='unknown_browser_transition')
        for policy in ('nav3', 'confirmed3'):
            q, decision = self.derive(policy, {'B': visit})
            self.assertEqual(q.truth, 'B')
            self.assertEqual(decision['skipped'], [])

    def test_all_short_can_remove_unknown_browser_transition_but_does_not_call_it_navigation(self):
        visits = {'B': self.visit(category='unknown_browser_transition'),
                  'C': self.visit('C', 2000, 20000, 'D', after='D')}
        q, decision = self.derive('dwell2', visits)
        self.assertEqual(q.truth, 'C')
        self.assertEqual(decision['skipped'][0]['category'], 'suspected_temporary_target')

    def test_return_only_stops_on_A_B_A_without_hunting_for_C(self):
        visits = {'B': self.visit(next_id='A', after='A'),
                  'A': self.visit('A', 2000, 2500, 'C'),
                  'C': self.visit('C', 2500, 20000, 'D', after='D')}
        q, decision = self.derive('return3', visits)
        self.assertIsNone(q)
        self.assertEqual(decision['status'], 'returned_to_current_target')
        self.assertEqual(decision['skipped'][0]['category'], 'quick_return')
        self.assertTrue(decision['skipped'][0]['returns_to_previous'])

    def test_return_only_keeps_fast_non_return_transition(self):
        q, decision = self.derive('return3', {'B': self.visit(after='C')})
        self.assertEqual(q.truth, 'B')
        self.assertEqual(decision['skipped'], [])

    def test_censored_or_unknown_duration_is_not_removed_as_short(self):
        for visit in (self.visit(complete=False), self.visit(end=None, complete=False)):
            for policy in ('nav3', 'return3', 'confirmed3'):
                q, decision = self.derive(policy, {'B': visit})
                self.assertEqual(q.truth, 'B')
                self.assertEqual(decision['skipped'], [])
            q, decision = self.derive('dwell2', {'B': visit})
            self.assertIsNone(q)
            self.assertEqual(decision['skipped'], [])
            self.assertIn('not_treated_as_short', decision['status'])

    def test_long_visit_can_meet_threshold_before_later_censor(self):
        q, decision = self.derive('dwell8', {'B': self.visit(end=100000, complete=False)})
        self.assertEqual(q.truth, 'B')
        self.assertEqual(decision['evidence_available_monotonic_ms'], 9000)

    def test_horizon_prevents_future_exit_from_removing_target(self):
        q, decision = self.derive('nav3', {'B': self.visit(start=29000, end=31000)})
        self.assertEqual(q.truth, 'B')
        self.assertEqual(decision['skipped'], [])
        self.assertEqual(decision['considered']['navigation_class'], 'exit_not_observed_in_horizon')
        q, decision = self.derive('dwell2', {'B': self.visit(start=29000, end=31000)})
        self.assertIsNone(q)
        self.assertEqual(decision['status'], 'insufficient_dwell_before_30s')

    def test_refinement_requires_both_deadline_and_ready_index(self):
        refinements = [
            {'event_id': 'refine', 'monotonic_ms': 35000, 'target': target('C')},
            {'event_id': 'late_refine', 'monotonic_ms': 20000, 'target': target('D')}]
        q, _ = self.derive('resolved', {'B': self.visit(end=50000, refinements=refinements)})
        self.assertEqual(q.truth, 'B')
        refinements = [{'event_id': 'refine', 'monotonic_ms': 1500, 'target': target('C')}]
        q, decision = self.derive('resolved', {'B': self.visit(end=1800, refinements=refinements)})
        self.assertEqual(q.truth, 'C')
        self.assertEqual(q.next_event_id, 'B')
        self.assertEqual(q.ready_index, 100)
        self.assertEqual(decision['evidence_available_monotonic_ms'], 1500)

    def test_post_horizon_end_and_resolution_changes_cannot_change_decision(self):
        original = self.visit(end=50000)
        modified = deepcopy(original)
        modified.update(end_monotonic_ms=100000, end_event_id='late_refine', complete=False,
                        navigation_class='application_switch', target=target('future_final'))
        modified['refinements'] = [{'event_id': 'late_refine', 'monotonic_ms': 40000,
                                   'target': target('future_final')}]
        for policy in ('dwell2', 'nav3', 'return3', 'resolved'):
            left_q, left_decision = self.derive(policy, {'B': original})
            right_q, right_decision = self.derive(policy, {'B': modified})
            self.assertEqual(left_q, right_q)
            self.assertEqual(left_decision, right_decision)

    def test_exact_threshold_is_kept_and_kind_thresholds_are_separate(self):
        q, decision = self.derive('dwell2', {'B': self.visit(end=3000)})
        self.assertEqual(q.truth, 'B')
        self.assertEqual(decision['skipped'], [])
        p = self.configs['kind_app2_resource5']
        self.assertEqual(cleaning.threshold_for(p, 'app', {})[0], 2)
        self.assertEqual(cleaning.threshold_for(p, 'page', {})[0], 5)

    def test_adaptive_quantiles_require_enough_prior_data_and_are_bounded(self):
        p = self.configs['adaptive_kind_q10']
        self.assertEqual(cleaning.threshold_for(p, 'page', {'page': [.1] * 19})[0], 3)
        self.assertEqual(cleaning.threshold_for(p, 'page', {'page': [.1] * 20})[0], 1)
        self.assertEqual(cleaning.threshold_for(p, 'page', {'page': [100] * 20})[0], 8)
        self.assertEqual(cleaning.threshold_for(p, 'app', {'page': [100] * 20})[0], 3)

    def test_adaptive_history_cannot_use_post_input_completed_dwell_or_censor(self):
        visits = [self.visit(end=2000, next_id='C'),
                  self.visit(end=5000, next_id='D'),
                  self.visit(end=6000, next_id='A', complete=False)]
        history = cleaning.PastDwellHistory(visits, self.indices)
        self.assertEqual(dict(history.advance(1)), {})
        self.assertEqual(history.advance(2)['page'], [1.0])
        self.assertEqual(history.through_index, 2)
        # Though this training query matures at event 100, threshold statistics
        # still stop at its input event 2, before the future departure at event 4.
        self.assertEqual(history.advance(2)['page'], [1.0])
        self.assertEqual(history.advance(4)['page'], [1.0, 4.0])
        with self.assertRaises(ValueError):
            history.advance(3)

    def test_denied_helpers_neither_fit_thresholds_nor_bridge_labels(self):
        denied = next(iter(cleaning.DENIED_BUNDLES))
        invalid = self.visit()
        invalid['initial_target']['bundleID'] = denied
        invalid['target']['bundleID'] = denied
        into_denied = self.visit()
        into_denied['next_target']['bundleID'] = denied
        history = cleaning.PastDwellHistory([invalid, into_denied], self.indices)
        self.assertEqual(dict(history.advance(100)), {})
        visits = {'B': invalid, 'C': self.visit('C', 2000, 20000, 'D')}
        q, decision = self.derive('dwell2', visits)
        self.assertIsNone(q)
        self.assertEqual(decision['status'], 'denied_destination_boundary')

    def test_cycle_does_not_create_an_infinite_chain(self):
        visits = {'B': self.visit(next_id='B', after='B')}
        q, decision = self.derive('dwell2', visits)
        self.assertIsNone(q)
        self.assertEqual(decision['status'], 'cyclic_visit_chain')


if __name__ == '__main__':
    unittest.main()
