"""Regression tests for stable scoring rules and prediction-independent legality."""
from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import evaluation_contract as s


def target(key, kind='app', bundle='example.next', locator=None):
    value = {'id': key, 'kind': kind, 'bundleID': bundle}
    if locator: value['locator'] = locator
    return value


def data(current=None, truth=None, label=None):
    current = current or target('app:current', bundle='current')
    truth = truth or target('app:next')
    return {'observations': [{'position': 0, 'index': 10, 'ready_index': 20, 'target': current}],
            'evaluation': [{'position': 0, 'sample_id': 'sample:0', 'truth_action': label or truth['id'],
                            'truth_metadata': truth, 'evidence_grade': 'original', 'split': 'test'}]}


def predictions(*labels):
    return {0: {'predictions': list(labels), 'rankable': list(labels)}}


def parent(bundle):
    return target('app:' + bundle, 'app', bundle=bundle)


def page_with_prior_parent(parent_bundle='browser', current_bundle='other'):
    current = target('page:current', 'page', locator='https://example.test/current')
    truth = target('page:truth', 'page', locator='https://example.test/truth')
    value = data(current=current, truth=truth)
    value['observations'][0]['parent_action'] = parent(current_bundle)
    value['evaluation'][0]['truth_parent_metadata'] = parent(parent_bundle)
    value['evaluation'][0]['prior_evidence'] = {'exact_target': False, 'parent_action': True}
    return value


class ScoringTests(unittest.TestCase):
    def test_weighted_loss_retains_same_membership_and_checks_preference(self):
        rows = s.prepare_evaluation(data())
        good = s.measure_weighted_loss(rows, predictions('app:next'), {0: 5.})
        bad = s.measure_weighted_loss(rows, predictions('wrong'), {0: 5.})
        self.assertEqual(good['mean'], 0.); self.assertEqual(bad['mean'], 1.)
        s.require_comparable(good, bad)
        with self.assertRaises(ValueError):
            s.require_comparable(good, s.measure_weighted_loss(rows, predictions('wrong'), {0: 1.}))

    def test_directional_hierarchical_loss(self):
        aliases = {s.SERVICE: s.SERVICE, 'home': s.SERVICE, 'a': s.SERVICE, 'b': s.SERVICE}
        entries = {s.SERVICE, 'home'}
        self.assertEqual(s.decision_loss('a', 'a', aliases, entries), 0.)
        self.assertEqual(s.decision_loss('home', 'a', aliases, entries), .4)
        self.assertEqual(s.decision_loss('b', 'a', aliases, entries), 1.)
        self.assertEqual(s.decision_loss('a', 'home', aliases, entries), 1.)
        self.assertEqual(s.decision_loss(s.SERVICE, 'home', aliases, entries), 0.)
        self.assertEqual(s.decision_loss('home', 'outside', aliases, entries), 1.)

    def test_minimum_risk_prefers_specific_when_concentrated(self):
        aliases = {s.SERVICE: s.SERVICE, 'a': s.SERVICE, 'b': s.SERVICE}
        result = s.minimum_risk_action({'a': .8, 'b': .2}, [s.SERVICE, 'a', 'b'], aliases, {s.SERVICE})
        self.assertEqual(result['action'], 'a')
        self.assertAlmostEqual(result['expected_loss'], .2)

    def test_minimum_risk_falls_back_when_ambiguous(self):
        aliases = {s.SERVICE: s.SERVICE, **{k: s.SERVICE for k in 'abc'}}
        result = s.minimum_risk_action({'a': .4, 'b': .3, 'c': .3}, [s.SERVICE, 'a', 'b', 'c'], aliases, {s.SERVICE})
        self.assertEqual(result['action'], s.SERVICE)
        self.assertEqual(result['expected_loss'], .4)

    def test_minimum_risk_accounts_for_other_service_probability(self):
        aliases = {s.SERVICE: s.SERVICE, 'a': s.SERVICE, 'b': s.SERVICE}
        result = s.minimum_risk_action({'a': .3, 'b': .3, 'other': .4}, [s.SERVICE, 'a', 'b', 'other'], aliases, {s.SERVICE})
        self.assertEqual(result['action'], 'other')
        self.assertAlmostEqual(result['action_expected_losses'][s.SERVICE], .64)

    def test_homepage_intent_is_not_penalized_as_fallback(self):
        aliases = {s.SERVICE: s.SERVICE, 'home': s.SERVICE, 'a': s.SERVICE}
        result = s.minimum_risk_action({'home': .8, 'a': .1, 'other': .1}, ['home', 'a', 'other'], aliases, {s.SERVICE, 'home'})
        self.assertEqual(result['action'], 'home')
        self.assertAlmostEqual(result['expected_loss'], .14)

    def test_unknown_child_mass_is_retained(self):
        aliases = {s.SERVICE: s.SERVICE, 'a': s.SERVICE, 'unknown_child': s.SERVICE}
        result = s.minimum_risk_action({'a': .3, 'unknown_child': .7}, [s.SERVICE, 'a'], aliases, {s.SERVICE})
        self.assertEqual(result['action'], s.SERVICE)

    def test_risk_rejects_raw_scores_and_overlapping_parent_mass(self):
        for posterior in ({'a': 5., 'b': 3.}, {'a': float('nan')}, {'a': -.1, 'b': 1.1}, {s.SERVICE: 1.}):
            with self.assertRaises(ValueError): s.minimum_risk_action(posterior, ['a'], {}, set())

    def test_observed_loss_and_risk_use_same_definition(self):
        d = data(truth=target('page:chat', 'page', locator='https://chatgpt.com/c/12345678-1234-1234-1234-123456789abc'))
        rows = s.prepare_evaluation(d)
        actual = s.measure(rows, predictions(s.SERVICE))['decision_loss']
        forecast = s.minimum_risk_action({'page:chat': 1.}, [s.SERVICE], rows[0]['_scoring']['aliases'], rows[0]['_scoring']['entries'])
        self.assertEqual(actual['mean'], .4)
        self.assertEqual(actual['mean'], forecast['expected_loss'])
        self.assertEqual(actual['definition'], forecast['loss_definition'])

    def test_rank_score_is_linear_for_exact_target_and_counts_misses(self):
        rows = s.prepare_evaluation(data())
        for rank, expected in enumerate((1., .75, .5, .25, 0.), 1):
            p = predictions(*(['wrong']*(rank-1)), 'app:next')
            result = s.measure(rows, p)
            self.assertEqual(result['rank_score']['mean'], expected)
            self.assertEqual(result['rank_score']['n'], 1)
        self.assertEqual(s.measure(rows, predictions())['rank_score']['mean'], 0.)

    def test_rank_score_discounts_homepage_gain_by_position(self):
        rows = s.prepare_evaluation(data(truth=target('chat', 'page',
            locator='https://chatgpt.com/c/12345678-1234-1234-1234-123456789abc')))
        for rank, expected in enumerate((.6, .45, .3, .15, 0.), 1):
            result = s.measure(rows, predictions(*(['wrong']*(rank-1)), s.SERVICE))
            self.assertAlmostEqual(result['rank_score']['mean'], expected)

    def test_known_parent_app_is_partial_help_but_never_an_exact_page_hit(self):
        rows = s.prepare_evaluation(page_with_prior_parent())
        result = s.measure(rows, predictions('app:browser'))
        self.assertEqual(result['exact']['top1_hits'], 0)
        self.assertEqual(result['top1_hits'], 0)
        self.assertAlmostEqual(result['actionable_topk']['top1_gain'], .6)
        self.assertAlmostEqual(result['actionable_topk']['top4_gain'], .6)
        self.assertEqual(result['actionable_topk']['top4_parent_fallback_used_n'], 1)
        self.assertAlmostEqual(result['rank_score']['mean'], .6)
        self.assertAlmostEqual(result['decision_loss']['mean'], .4)
        boards = s.score_leaderboards(rows, {'candidate': predictions('app:browser')})
        self.assertEqual(boards[s.LEADERBOARD_ALL]['candidate']['n'], 1)
        self.assertEqual(boards[s.LEADERBOARD_PRIOR]['candidate']['n'], 1)

    def test_parent_app_needs_prior_confirmation_and_a_different_current_app(self):
        missing = page_with_prior_parent()
        missing['evaluation'][0]['prior_evidence']['parent_action'] = False
        result = s.measure(s.prepare_evaluation(missing), predictions('app:browser'))
        self.assertEqual(result['actionable_topk']['top4_gain'], 0.)
        same = page_with_prior_parent(parent_bundle='browser', current_bundle='browser')
        with self.assertRaises(ValueError):
            s.prepare_evaluation(same)

    def test_exact_page_beats_a_parent_fallback_without_double_counting(self):
        rows = s.prepare_evaluation(page_with_prior_parent())
        result = s.measure(rows, predictions('page:truth', 'app:browser'))
        self.assertEqual(result['exact']['top1_hits'], 1)
        self.assertEqual(result['actionable_topk']['top4_gain'], 1.)
        self.assertEqual(result['actionable_topk']['top4_parent_fallback_used_n'], 0)

    def test_later_exact_match_improves_homepage_without_adding_duplicate_gains(self):
        rows = s.prepare_evaluation(data(truth=target('chat', 'page',
            locator='https://chatgpt.com/c/12345678-1234-1234-1234-123456789abc')))
        homepage = s.measure(rows, predictions(s.SERVICE, s.SERVICE, s.SERVICE, s.SERVICE))
        improved = s.measure(rows, predictions(s.SERVICE, 'wrong', 'wrong', 'chat'))
        self.assertAlmostEqual(homepage['rank_score']['mean'], .6)
        self.assertAlmostEqual(improved['rank_score']['mean'], .7)
        self.assertEqual(improved['rank_score']['prefix_mean_gains'], {'1': .6, '2': .6, '3': .6, '4': 1.})

    def test_wrong_chat_does_not_receive_homepage_gain(self):
        d = data(truth=target('chat', 'page',
            locator='https://chatgpt.com/c/12345678-1234-1234-1234-123456789abc'))
        d['streams'] = {'resolved': [{'record': {'target': target('other-chat', 'page',
            locator='https://chatgpt.com/c/12345678-1234-1234-1234-123456789abd')}}]}
        rows = s.prepare_evaluation(d)
        result = s.measure(rows, predictions('other-chat'))
        self.assertEqual(result['top1_hits'], 1)  # Historical service-acceptance diagnostic.
        self.assertEqual(result['rank_score']['mean'], 0.)

    def test_known_url_intermediate_is_success_with_separate_partial_gain(self):
        current = target('page:elsewhere', 'page', bundle='browser',
                         locator='https://other.test/start')
        truth = target('page:deep', 'page', bundle='browser',
                       locator='https://example.test/topic/subtopic/article')
        parent_page = target('page:parent', 'page', bundle='browser',
                             locator='https://example.test/topic/subtopic/')
        d = data(current=current, truth=truth)
        d['observations'][0]['known_actions'] = [parent_page]
        rows = s.prepare_evaluation(d)
        score = s.measure(rows, predictions('page:parent'))
        self.assertEqual(score['success']['top1_hits'], 1)
        self.assertEqual(score['top1_hits'], 0)  # Legacy service acceptance stays separate.
        self.assertEqual(score['exact']['top1_hits'], 0)
        self.assertAlmostEqual(score['path_utility']['product0.8']['top1_gain'], .8)
        self.assertEqual(score['path_utility']['product0.6']['top1_gain'], .6)
        self.assertEqual(score['rank_score']['mean'], 0.)  # Old fixed loss is unchanged.

    def test_unvisited_parent_can_be_derived_from_an_earlier_known_child(self):
        current = target('page:elsewhere', 'page', bundle='browser',
                         locator='https://other.test/start')
        earlier_child = target('page:old', 'page', bundle='browser',
                               locator='https://example.test/topic/old')
        truth = target('page:new', 'page', bundle='browser',
                       locator='https://example.test/topic/new')
        d = data(current=current, truth=truth)
        d['observations'][0]['known_actions'] = [earlier_child]
        row = s.prepare_evaluation(d)[0]
        hierarchy = row['_scoring']['path_hierarchy']
        parent_id = next(key for key, meta in hierarchy.metadata.items()
                         if meta.get('_derived_parent')
                         and meta.get('locator') == 'https://example.test/topic/')
        self.assertIn(parent_id, hierarchy.executable)
        scored = s.measure([row], predictions(parent_id))
        self.assertEqual(scored['success']['top1_hits'], 1)
        self.assertEqual(scored['exact']['top1_hits'], 0)

    def test_path_success_needs_cutoff_action_and_blocks_current_place(self):
        current = target('page:parent', 'page', bundle='browser',
                         locator='https://example.test/topic/')
        truth = target('page:deep', 'page', bundle='browser',
                       locator='https://example.test/topic/article')
        d = data(current=current, truth=truth)
        d['observations'][0]['known_actions'] = [current]
        rows = s.prepare_evaluation(d)
        self.assertEqual(s.measure(rows, predictions('page:parent'))['success']['top1_hits'], 0)
        # A path discovered only from the future truth cannot become an as-of action.
        d['observations'][0]['known_actions'] = []
        d['observations'][0]['target'] = target('page:elsewhere', 'page', bundle='browser',
                                               locator='https://other.test/')
        rows = s.prepare_evaluation(d)
        self.assertEqual(s.measure(rows, predictions('page:parent'))['success']['top1_hits'], 0)

    def test_same_clean_url_under_another_id_is_still_the_current_place(self):
        current = target('page:current', 'page', bundle='browser',
                         locator='https://example.test/topic/')
        alias = target('page:alias', 'page', bundle='browser',
                       locator='https://example.test/topic')
        truth = target('page:deep', 'page', bundle='browser',
                       locator='https://example.test/topic/article')
        d = data(current=current, truth=truth)
        d['observations'][0]['known_actions'] = [current, alias]
        rows = s.prepare_evaluation(d)
        self.assertEqual(s.measure(rows, predictions('page:alias'))['success']['top1_hits'], 0)

    def test_known_parent_directory_helps_but_wrong_sibling_does_not(self):
        current = target('dir:elsewhere', 'directory', bundle='finder',
                         locator='file:///Users/example/elsewhere/')
        truth = target('file:report', 'file', bundle='finder',
                       locator='file:///Users/example/project/reports/final.pdf')
        parent_dir = target('dir:reports', 'directory', bundle='finder',
                            locator='file:///Users/example/project/reports/')
        sibling = target('dir:other', 'directory', bundle='finder',
                         locator='file:///Users/example/project/other/')
        d = data(current=current, truth=truth)
        d['observations'][0]['known_actions'] = [parent_dir, sibling]
        rows = s.prepare_evaluation(d)
        self.assertEqual(s.measure(rows, predictions('dir:reports'))['success']['top1_hits'], 1)
        self.assertEqual(s.measure(rows, predictions('dir:other'))['success']['top1_hits'], 0)

    def test_future_chatgpt_page_alone_does_not_enable_abstract_home_action(self):
        current = target('page:elsewhere', 'page', bundle='browser',
                         locator='https://other.test/')
        truth = target('page:new-chat', 'page', bundle='browser',
                       locator='https://chatgpt.com/c/12345678-1234-1234-1234-123456789abc')
        d = data(current=current, truth=truth)
        d['observations'][0]['known_actions'] = []
        rows = s.prepare_evaluation(d)
        self.assertEqual(s.measure(rows, predictions(s.SERVICE))['success']['top1_hits'], 0)
        d['observations'][0]['known_actions'] = [target(
            'page:prior-chat', 'page', bundle='browser',
            locator='https://chatgpt.com/c/12345678-1234-1234-1234-123456789abd')]
        rows = s.prepare_evaluation(d)
        # The abstract service label is not itself an openable menu action.
        self.assertEqual(s.measure(rows, predictions(s.SERVICE))['success']['top1_hits'], 0)
        hierarchy = rows[0]['_scoring']['path_hierarchy']
        derived_home = next(key for key, meta in hierarchy.metadata.items()
                            if meta.get('_derived_parent')
                            and meta.get('locator') == 'https://chatgpt.com/')
        self.assertEqual(s.measure(rows, predictions(derived_home))['success']['top1_hits'], 1)

    def test_guessed_future_exact_id_is_diagnostic_but_not_executable_success(self):
        current = target('page:elsewhere', 'page', bundle='browser',
                         locator='https://other.test/')
        truth = target('page:future', 'page', bundle='browser',
                       locator='https://example.test/never-before-seen')
        d = data(current=current, truth=truth)
        d['observations'][0]['known_actions'] = []
        score = s.measure(s.prepare_evaluation(d), predictions('page:future'))
        self.assertEqual(score['exact']['top1_hits'], 1)
        self.assertEqual(score['success']['top1_hits'], 0)

    def test_homepage_as_truth_receives_full_rank_gain(self):
        rows = s.prepare_evaluation(data(truth=target('home', 'page', locator='https://chatgpt.com/')))
        self.assertEqual(s.measure(rows, predictions(s.SERVICE))['rank_score']['mean'], 1.)

    def test_current_target_excluded_even_if_prediction_hits(self):
        d = data(); d['evaluation'][0]['truth_metadata'] = d['observations'][0]['target']
        d['evaluation'][0]['truth_action'] = 'app:current'
        rows = s.prepare_evaluation(d)
        for p in (predictions('app:current'), predictions('wrong')):
            m = s.measure(rows, p)
            self.assertEqual(m['n'], 0); self.assertEqual(m['audit']['excluded_n'], 1)

    def test_current_app_label_excluded_and_window_loss_recorded(self):
        for kind in ('app', 'window'):
            rows = s.prepare_evaluation(data(target('window:1', 'window', 'app'),
                target('window:2' if kind == 'window' else 'app:app', kind, 'app'), 'app:app'))
            self.assertFalse(s.eligible(rows[0]))
            self.assertEqual(rows[0]['_scoring']['distinct_window_lost_in_app_label'], kind == 'window')

    def test_specific_same_app_targets_remain_scoreable(self):
        for kind in ('page', 'directory', 'file', 'window'):
            rows = s.prepare_evaluation(data(target(kind+':1', kind, 'same'), target(kind+':2', kind, 'same')))
            self.assertTrue(s.eligible(rows[0]))
            self.assertEqual(s.measure(rows, predictions(kind+':2'))['top1_hits'], 1)

    def test_chatgpt_service_hit_not_exact_hit(self):
        truth = target('page:chat', 'page', locator='https://chatgpt.com/c/12345678-1234-1234-1234-123456789abc')
        d = data(truth=truth)
        d['streams'] = {'resolved': [{'record': {'target': target('page:home', 'page', locator='https://chatgpt.com/')}}]}
        rows = s.prepare_evaluation(d); m = s.measure(rows, predictions('page:home'))
        self.assertEqual(m['top1_hits'], 1); self.assertEqual(m['exact']['top1_hits'], 0)

    def test_no_broad_domain_folding(self):
        for url in ('https://example.com/c/a', 'https://chatgpt.com/g/abc',
                    'https://chatgpt.com/c/invalid', 'https://chatgpt.com/?q=secret',
                    'https://chatgpt.com:444/', 'https://evil@chatgpt.com/', 'https://chatgpt.com/#/c/a'):
            self.assertIsNone(s.service(target('page:a', 'page', locator=url)))

    def test_model_service_semantics_have_not_drifted(self):
        from navigation_hierarchy import service
        urls = ['https://chatgpt.com/', 'https://chatgpt.com/c/12345678-1234-1234-1234-123456789abc',
                'https://chatgpt.com/?historySearchQuery=a', 'https://chatgpt.com/g/abc', 'https://example.com/']
        for url in urls:
            t = target('page:a', 'page', locator=url)
            self.assertEqual(s.service(t), service(t))

    def test_first_k_emitted_slots_no_dedup_refill_or_raw_ranking(self):
        d = data(); d['streams'] = {'resolved': [{'record': {'target': target(k, 'page', locator=url)}} for k, url in
            [('page:a', 'https://chatgpt.com/'), ('page:b', 'https://chatgpt.com/?src=menu')]]}
        rows = s.prepare_evaluation(d)
        p = predictions('page:a', 'page:b', 'app:next', 'fourth')
        p[0]['raw'] = ['app:next']; m = s.measure(rows, p)
        self.assertEqual(m['top2_hits'], 0); self.assertEqual(m['top4_hits'], 1)
        p = predictions('1', '2', '3', '4', 'app:next')
        self.assertEqual(s.measure(rows, p)['top4_hits'], 0)

    def test_unknown_or_unrankable_truth_is_not_excluded(self):
        rows = s.prepare_evaluation(data())
        m = s.measure(rows, predictions('something-else'))
        self.assertEqual((m['n'], m['top4_hits'], m['rankable_truth_n']), (1, 0, 0))

    def test_empty_prediction_misses_but_missing_prediction_errors(self):
        rows = s.prepare_evaluation(data())
        self.assertEqual(s.measure(rows, predictions())['n'], 1)
        with self.assertRaises(ValueError): s.measure(rows, {})

    def test_unaudited_rows_rejected(self):
        with self.assertRaises(ValueError): s.measure(data()['evaluation'], predictions('app:next'))

    def test_bad_identity_or_chronology_excluded(self):
        d = data(); d['observations'][0]['ready_index'] = 10
        self.assertFalse(s.eligible(s.prepare_evaluation(d)[0]))
        d = data(); d['evaluation'][0]['truth_action'] = 'different'
        self.assertFalse(s.eligible(s.prepare_evaluation(d)[0]))

    def test_audit_does_not_change_original_and_is_archived(self):
        d = data(); before = deepcopy(d)
        with tempfile.TemporaryDirectory() as directory:
            s.prepare_evaluation(d, directory)
            audit = json.loads((Path(directory)/'evaluation-audit-private.json').read_text())
            self.assertEqual(audit['eligible_n'], 1)
            self.assertEqual((Path(directory)/'evaluation_contract.py').read_bytes(), Path(s.__file__).read_bytes())
            with self.assertRaises(ValueError): s.prepare_evaluation(d, directory)
        self.assertEqual(d, before)

    def test_paired_and_selection_share_accepted_hits(self):
        from joint_ranking_alternatives import choose_group
        rows = s.prepare_evaluation(data(truth=target('page:chat', 'page', locator='https://chatgpt.com/')))
        good, bad = predictions(s.SERVICE), predictions('miss')
        m = s.paired_topk(rows, bad, good)
        self.assertEqual(m['top2']['net_hits'], 1)
        self.assertEqual(choose_group(rows, {'bad': [bad], 'good': [good]}), 'good')

    def test_compare_guards_membership_and_contract(self):
        rows = s.prepare_evaluation(data()); a = s.measure(rows, predictions())
        b = s.measure(rows, predictions('app:next')); s.require_comparable(a, b)
        with self.assertRaises(ValueError): s.require_comparable(a, {'top1_hits': 1})
        with self.assertRaises(ValueError): s.require_comparable(a, s.measure([], {}))

    def test_frozen_import_path_cannot_replace_active_scorer(self):
        # Entry points load the scorer before adding immutable model directories.
        for name in ('compare_statistical_residual', 'compare_joint_alternatives',
                     'compare_context_statistics', 'compare_temporal_fusion', 'diagnose_top4_misses'):
            text = Path(__file__).with_name(name+'.py').read_text()
            self.assertLess(text.index('import evaluation_contract'), text.index('sys.path.insert'))
            self.assertNotIn("import_module('compare_statistical_residual')", text)
        with tempfile.TemporaryDirectory() as directory:
            (Path(directory)/'evaluation_contract.py').write_text("raise RuntimeError('legacy scorer imported')")
            script = ('import sys; import evaluation_contract as live; sys.path.insert(0, '+repr(directory)+'); '
                      'import evaluation_contract as frozen; assert frozen is live')
            subprocess.run([sys.executable, '-c', script], cwd=Path(__file__).parent, check=True)


if __name__ == '__main__': unittest.main()
