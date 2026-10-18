"""Synthetic boundary checks; no historical user data is needed."""
from copy import deepcopy
import unittest

import evaluate_context_knn as core
import reopenable_actions as action
from test_evaluate_context_knn import observation


def app(bundle='example.owner'):
    return {'id': 'app:' + bundle, 'kind': 'app', 'bundleID': bundle,
            'admission': {'basis': 'standalone_application', 'policyVersion': 'target-eligibility-v4'}}


def window(bundle='example.owner', ident='window-one'):
    return {'id': ident, 'kind': 'window', 'bundleID': bundle,
            'admission': {'basis': 'standard_work_window', 'policyVersion': 'target-eligibility-v4'}}


def page(ident='page:one', owner=None):
    value = {'id': ident, 'kind': 'page', 'locator': 'https://example.invalid/exact?q=one#part',
             'admission': {'basis': 'supported_browser_page', 'policyVersion': 'target-eligibility-v4'}}
    if owner:
        value['bundleID'] = owner
    return value


def events(n=8):
    return [{'id': 'e' + str(i), 'sessionID': 's', 'kind': 'context.change',
             'timestampMs': i * 1000, 'data': {}} for i in range(n)]


def update(es, index, target):
    es[index] = {**es[index], 'kind': 'target.update', 'data': {'target': target}}


def query(index, candidates=None):
    q = observation(index)
    q.anchor_id = 'e' + str(index)
    q.session = 's'
    q.candidates = candidates or {}
    q.last_visits = {}
    q.recency = {}
    return q


def row(position, meta, key=None, present=True, split='test'):
    return {'sample_id': 's' + str(position), 'position': position, 'split': split,
            'observed_outcome': {'key': key or meta['id'], 'metadata': deepcopy(meta)},
            'diagnosis_after_outcome': {'eligible': True, 'grade': 'single', 'candidate_present': present},
            'evaluation': {'strict_input_and_label_coverage': True}}


def adapt(items, rows, es, streams=None):
    return action.adapt_dataset(core, items, {'5': rows}, streams or {},
                                {'excluded_sessions': []}, events=es)


class ReopenableActionsTests(unittest.TestCase):
    def test_window_collapses_to_admitted_app_without_changing_current_origin(self):
        es = events(); update(es, 0, app())
        w = window(); key = core.key_for(w, 's')
        q = query(2, {key: w})
        q.current = 'window@s:current'; q.current_app = 'example.owner'
        q.last_visits[key] = 1000
        original = deepcopy(q)
        result = adapt([q], [row(0, w, key)], es)
        items, inputs, mappings, _, audit = result
        self.assertEqual(q, original)
        self.assertEqual(items[0].current, q.current)
        self.assertEqual(inputs[0]['origin'], 'window_class:example.owner')
        self.assertEqual(inputs[0]['labels'], ['app:example.owner'])
        self.assertEqual(mappings[0], {'app:example.owner': 'app:example.owner'})
        self.assertEqual(items[0].last_visits['app:example.owner'], 1000)
        self.assertEqual(audit['positions']['0']['window_to_app'], {key: 'app:example.owner'})
        pred = {0: {'predictions': ['app:example.owner']}}
        annotated = action.annotate_predictions(pred, items)
        plan = annotated[0]['action_plans'][0]
        self.assertEqual(plan['type'], 'activate_or_launch_app')
        self.assertFalse(plan['guarantees_specific_window'])
        self.assertTrue(plan['same_foreground_app_may_be_noop'])
        self.assertNotIn('action_plans', pred[0])

    def test_future_app_and_truth_metadata_cannot_backfill_earlier_candidates(self):
        es = events(); update(es, 3, app())
        w = window(); key = core.key_for(w, 's')
        first, later = query(1, {key: w}), query(5, {key: w})
        rows = [row(0, w, key, present=False), row(1, w, key)]
        items, inputs, _, _, audit = adapt([first, later], rows, es)
        self.assertNotIn('app:example.owner', items[0].candidates)
        self.assertIn('app:example.owner', items[1].candidates)
        self.assertEqual(inputs[0]['labels'], [])
        self.assertFalse(audit['cohorts']['5']['old_missing_mappings'][0]['action_candidate_present'])
        poisoned = deepcopy(rows)
        poisoned[0]['observed_outcome']['metadata']['bundleID'] = 'future.other'
        changed = adapt([first, later], poisoned, es)
        self.assertEqual(inputs, changed[1])

    def test_known_app_and_full_locator_survive_session_boundary(self):
        es = events(); update(es, 0, app()); update(es, 1, page())
        for e in es[3:]: e['sessionID'] = 'second'
        first, second = query(2), query(5); second.session = 'second'
        items, _, _, _, audit = adapt([first, second], [row(0, page()), row(1, page())], es)
        self.assertIn('page:one', items[1].candidates)
        self.assertIn('app:example.owner', items[1].candidates)
        self.assertEqual(items[1].candidates['page:one']['locator'], page()['locator'])
        ev = audit['positions']['1']['candidate_evidence']['page:one']
        self.assertEqual(ev['target_update_index'], 1)
        self.assertEqual(ev['known_session'], 's')
        self.assertEqual(ev['source'], 'historical_target_update')
        self.assertIsNone(items[1].candidates['page:one']['_bundle'])

    def test_explicit_rejection_removes_history_and_readmission_restores_only_afterward(self):
        es = events(10); update(es, 0, app()); update(es, 1, page())
        rejected_app = app(); rejected_app.pop('admission')
        rejected_page = page(); rejected_page.pop('admission')
        update(es, 3, rejected_app); update(es, 4, rejected_page); update(es, 6, page())
        queries = [query(2), query(5), query(8)]
        items, *_ = adapt(queries, [row(i, page()) for i in range(3)], es)
        self.assertIn('app:example.owner', items[0].candidates)
        self.assertEqual(items[1].candidates, {})
        self.assertIn('page:one', items[2].candidates)
        self.assertNotIn('app:example.owner', items[2].candidates)

    def test_denied_helpers_missing_locators_and_unapproved_windows_are_not_reopened(self):
        es = events()
        helper = app('com.apple.KeyboardSetupAssistant')
        missing = page(); missing.pop('locator')
        script = page('page:script'); script['locator'] = 'javascript:alert(1)'
        update(es, 0, helper); update(es, 1, missing); update(es, 2, script)
        q = query(4, {'window@s:x': window('unknown.app', 'x')})
        items, *_ = adapt([q], [row(0, helper)], es)
        self.assertEqual(items[0].candidates, {})

    def test_nonempty_unrecognized_admission_reason_does_not_restore_app(self):
        es = events(); update(es, 0, app())
        rejected = app(); rejected['admission']['basis'] = 'work_window_unconfirmed'
        update(es, 2, rejected)
        items, *_ = adapt([query(1), query(4)], [row(0, app()), row(1, app())], es)
        self.assertIn('app:example.owner', items[0].candidates)
        self.assertNotIn('app:example.owner', items[1].candidates)

    def test_training_window_label_requires_app_known_by_label_maturity(self):
        es = events(); update(es, 4, app())
        base = {'position': 0, 'input_index': 1, 'outcome_index': 2,
                'record': {'origin': 'current', 'app': '', 'timestamp': 1000,
                           'label': 'window_class:example.owner', 'outcome_id': 'e2',
                           'outcome_timestamp': 2000, 'x': {'current': 1.0}}}
        stream = [{**deepcopy(base), 'ready_index': 3}, {**deepcopy(base), 'ready_index': 5}]
        original = deepcopy(stream)
        result = adapt([query(6)], [row(0, app())], es, {'test': stream})
        converted = result[3]['test']
        self.assertEqual(stream, original)
        self.assertEqual(len(converted), 1)
        self.assertEqual(converted[0]['record']['label'], 'app:example.owner')
        self.assertEqual(converted[0]['record']['origin'], 'current')
        self.assertEqual(result[4]['streams']['test']['excluded_window_without_admitted_app_at_maturity'], 1)

    def test_app_is_not_a_concrete_page_or_exact_window_hit(self):
        es = events(); update(es, 0, app()); update(es, 1, page(owner='example.owner'))
        w = window(); wkey = core.key_for(w, 's')
        qs = [query(3, {wkey: w}), query(4)]
        rows = [row(0, w, wkey), row(1, page(owner='example.owner'))]
        items, *_ = adapt(qs, rows, es)
        scores = action.score_groups(rows, {0: {'predictions': ['app:example.owner']},
                                            1: {'predictions': ['app:example.owner']}}, items)
        self.assertEqual(scores['supported']['top4_hits'], 1)
        self.assertEqual(scores['exact_original']['supported']['top4_hits'], 0)
        self.assertEqual(scores['app_level']['supported']['top4_hits'], 2)
        self.assertEqual(scores['exact_representable_supported']['n'], 1)
        self.assertEqual(scores['exact_representable_supported']['top4_hits'], 0)

    def test_old_window_prediction_rescores_but_retains_original_exact_metric(self):
        es = events(); update(es, 0, app())
        w = window(); key = core.key_for(w, 's')
        rows = [row(0, w, key)]
        items, *_ = adapt([query(2, {key: w})], rows, es)
        scores = action.score_groups(rows, {0: {'predictions': [key]}}, items)
        self.assertEqual(scores['supported']['top4_hits'], 1)
        self.assertEqual(scores['exact_original']['supported']['top4_hits'], 1)
        self.assertEqual(scores['exact_original']['supported']['candidate_present'], 1)

    def test_missing_page_owner_is_unknown_not_an_app_level_error(self):
        es = events(); update(es, 0, page()); update(es, 1, app())
        rows = [row(0, page()), row(1, app())]
        items, *_ = adapt([query(3), query(4)], rows, es)
        pred = {0: {'predictions': ['page:one']}, 1: {'predictions': ['page:one']}}
        scores = action.score_groups(rows, pred, items)
        self.assertEqual(scores['supported']['top4_hits'], 1)
        app_scores = scores['app_level']['supported']
        self.assertEqual(app_scores['n'], 1)
        self.assertEqual(app_scores['top4_lower'], 0)
        self.assertEqual(app_scores['top4_upper'], 1)
        self.assertEqual(app_scores['unknown_truth_app'], 1)
        self.assertEqual(app_scores['unknown_prediction_app'], 1)
        self.assertEqual(app_scores['n_total'], 2)

    def test_known_app_hit_survives_later_unknown_slot_and_denominator_is_fixed(self):
        es = events(); update(es, 0, page()); update(es, 1, app())
        rows = [row(0, app())]
        items, *_ = adapt([query(3)], rows, es)
        known = {0: {'predictions': ['app:example.owner']}}
        later_unknown = {0: {'predictions': ['app:example.owner', 'page:one']}}
        unknown_first = {0: {'predictions': ['page:one', 'app:example.owner']}}
        a = action.score_groups(rows, known, items)['app_level']['supported']
        b = action.score_groups(rows, later_unknown, items)['app_level']['supported']
        c = action.score_groups(rows, unknown_first, items)['app_level']['supported']
        self.assertEqual((a['n'], b['n'], c['n']), (1, 1, 1))
        self.assertEqual((b['top1_lower'], b['top1_upper']), (1, 1))
        self.assertEqual((c['top1_lower'], c['top1_upper']), (0, 1))
        self.assertEqual((c['top4_lower'], c['top4_upper']), (1, 1))

    def test_same_foreground_app_sensitivity_keeps_primary_members(self):
        es = events(); update(es, 0, app())
        q = query(2); q.current = 'page:current'; q.current_app = 'example.owner'
        rows = [row(0, app())]
        items, *_ = adapt([q], rows, es)
        scores = action.score_groups(rows, {0: {'predictions': ['app:example.owner']}}, items)
        self.assertEqual(scores['supported']['n'], 1)
        self.assertEqual(scores['same_foreground_app_ambiguous']['n'], 1)
        self.assertEqual(scores['without_same_foreground_app_ambiguous']['n'], 0)

    def test_hierarchy_and_original_current_exclusion_remain_intact(self):
        es = events(); update(es, 0, app()); update(es, 1, page(owner='example.owner'))
        q = query(3); q.current = 'page:current'; q.current_app = 'example.owner'
        items, *_ = adapt([q], [row(0, page())], es)
        self.assertEqual(core.select_four(['page:one', 'app:example.owner'], items[0].candidates), ['page:one'])
        q.current = 'app:example.owner'
        changed, *_ = adapt([q], [row(0, page())], es)
        self.assertNotIn('app:example.owner', changed[0].candidates)
        self.assertIn('page:one', changed[0].candidates)

    def test_current_window_arrival_does_not_become_app_visit_history(self):
        es = events(); update(es, 0, app()); update(es, 1, window())
        q = query(3)
        q.current = core.key_for(window(), 's'); q.current_app = 'example.owner'
        q.last_visits = {q.current: q.timestamp, 'app:example.owner': q.timestamp}
        items, inputs, _, _, audit = adapt([q], [row(0, app())], es)
        self.assertIn('app:example.owner', items[0].candidates)
        self.assertNotIn('app:example.owner', inputs[0]['candidate_last_seen'])
        self.assertNotIn(q.current, items[0].last_visits)
        self.assertEqual(set(audit['positions']['0']['recency_filter']['removed_at_cutoff_keys']),
                         {q.current, 'app:example.owner'})

    def test_recency_aggregation_retains_older_window_but_discards_current_and_future(self):
        es = events(); update(es, 0, app()); update(es, 1, window())
        old_window = window(ident='older-window')
        old_key = core.key_for(old_window, 's')
        q = query(3, {old_key: old_window})
        q.current = core.key_for(window(), 's'); q.current_app = 'example.owner'
        q.last_visits = {q.current: q.timestamp, 'app:example.owner': q.timestamp,
                         old_key: q.timestamp - 1, 'unusable-future': q.timestamp + 1}
        items, inputs, _, _, audit = adapt([q], [row(0, app())], es)
        self.assertEqual(inputs[0]['candidate_last_seen']['app:example.owner'], q.timestamp - 1)
        self.assertTrue(all(time < q.timestamp for time in items[0].last_visits.values()))
        self.assertEqual(audit['totals']['visit_entries_at_cutoff_removed'], 2)
        self.assertEqual(audit['totals']['visit_entries_after_cutoff_removed'], 1)
        import stat_model_suite
        config = next(c for c in stat_model_suite.configurations() if c['family'] == 'candidate_recency')
        scores = stat_model_suite.build(config).scores(inputs[0])
        self.assertIn('app:example.owner', scores)


if __name__ == '__main__':
    unittest.main()
