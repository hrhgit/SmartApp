"""Causal and exact-destination contracts for the metadata-only online suite.

All fixtures are synthetic. No personal run, inventory, or future label catalog
is needed to exercise these boundaries.
"""
from copy import deepcopy
import math
import unittest

import compare_semantic_trajectory as runner
from semantic_annotations import annotate_target, compare_annotations
from semantic_online import SemanticStatistics, SimilarityCache, scores_from_components


def observation(index, origin, url=None):
    return {
        'index': index, 'timestamp': index * 1000, 'origin': origin,
        'app': 'browser',
        'annotation': annotate_target({
            'kind': 'page', 'id': origin, 'bundleID': 'browser',
            'locator': url or f'https://school.example/courses/CS101/{origin}.pdf',
        }),
    }


def pending(position, ready, step, label, event, weight=1.):
    return {
        'position': position, 'ready_index': ready,
        'record': {
            'observed_step': step, 'action_key': label, 'event_id': event,
            'weight': weight, 'arrival_timestamp_ms': 2000,
        },
    }


def evaluation(position, truth='actual:A', grade='original'):
    return {'position': position, 'truth_action': truth,
            'evidence_grade': grade, 'split': 'test'}


class SemanticStatisticsTests(unittest.TestCase):
    def test_semantics_share_only_observed_exact_destination_labels(self):
        model = SemanticStatistics(semantic_strength=4., global_strength=0.)
        model.learn(observation(1, 'resource-old'), 'page:exact?view=1#p=2', outcome_id='a')
        query = observation(10, 'resource-new')
        scores = model.predict(query)
        self.assertEqual(set(scores), {'page:exact?view=1#p=2'})
        self.assertEqual(scores['page:exact?view=1#p=2'], 1.)
        self.assertNotIn(query['origin'], scores)

    def test_destination_query_and_fragment_are_not_semantically_merged(self):
        model = SemanticStatistics(global_strength=0.)
        before = observation(1, 'same')
        labels = ['page:https://dest.example/a?view=1#p=2',
                  'page:https://dest.example/a?view=2#p=2',
                  'page:https://dest.example/a?view=1#p=3']
        for i, label in enumerate(labels):
            model.learn(before, label, outcome_id=str(i))
        self.assertEqual(set(model.predict(observation(10, 'same'))), set(labels))

    def test_weak_relation_retains_less_mass_than_strong_relation(self):
        query = observation(10, 'new')
        strong = observation(1, 'old')
        weak = observation(1, 'weak', 'https://school.example/unrelated.pdf')
        masses = []
        for before in (strong, weak):
            model = SemanticStatistics(semantic_strength=1.)
            model.learn(before, 'actual:A', outcome_id='a')
            semantic = model.components(query)[1]
            expected = compare_annotations(query['annotation'], before['annotation'])['score'] ** 2
            self.assertAlmostEqual(sum(semantic.values()), expected)
            masses.append(sum(semantic.values()))
        self.assertGreater(masses[0], 10 * masses[1])
        self.assertGreater(masses[1], 0.)

    def test_generic_pdf_type_across_hosts_has_no_semantic_vote(self):
        model = SemanticStatistics(semantic_strength=1.)
        model.learn(observation(1, 'old', 'https://alpha.example/a.pdf'),
                    'actual:A', outcome_id='a')
        query = observation(10, 'new', 'https://beta.example/b.pdf')
        self.assertEqual(model.components(query)[1], {})

    def test_training_weight_changes_exact_evidence_proportion(self):
        model = SemanticStatistics(global_strength=0.)
        before = observation(1, 'same')
        model.learn(before, 'actual:A', weight=1., outcome_id='a')
        model.learn(before, 'actual:B', weight=.25, outcome_id='b')
        scores = model.predict(observation(10, 'same'))
        self.assertAlmostEqual(scores['actual:A'], .8)
        self.assertAlmostEqual(scores['actual:B'], .2)

    def test_decay_uses_outcome_time_and_does_not_read_future_fields(self):
        model = SemanticStatistics(global_strength=0., half_life_hours=1.)
        before = observation(1, 'same')
        model.learn(before, 'actual:A', outcome_id='a', outcome_timestamp=1000)
        model.learn(before, 'actual:B', outcome_id='b', outcome_timestamp=3601000)
        query = observation(3602, 'same')
        scores = model.predict(query)
        self.assertAlmostEqual(scores['actual:B'] / scores['actual:A'], 2.)

    def test_full_learned_vocabulary_is_not_restricted_by_inventory(self):
        model = SemanticStatistics()
        for i in range(6):
            model.learn(observation(1, f'old-{i}'), f'actual:{i}', outcome_id=str(i))
        query = observation(10, 'new')
        expected = model.predict(query)
        query.update(labels=['actual:0', 'unseen:future'], candidates=[], inventory=[],
                     background={'unseen:future': {}}, candidate_last_seen={'unseen:future': 999999})
        self.assertEqual(model.predict(query), expected)
        self.assertEqual(set(expected), {f'actual:{i}' for i in range(6)})

    def test_no_training_no_labels_and_truth_metadata_cannot_create_labels(self):
        model = SemanticStatistics(semantic_strength=4.)
        query = observation(10, 'new')
        query.update(truth='unseen:future', truth_action='unseen:future',
                     observed_outcome={'label': 'unseen:future'}, evidence_grade='semantic_strong',
                     future=[{'label': 'unseen:future'}])
        self.assertEqual(model.predict(query), {})
        model.learn(observation(1, 'old'), 'actual:A', outcome_id='a')
        expected = model.predict(observation(10, 'new'))
        self.assertEqual(model.predict(query), expected)
        self.assertEqual(set(expected), {'actual:A'})

    def test_heads_are_independent_and_repeated_learning_is_deduplicated(self):
        model = SemanticStatistics()
        before = observation(1, 'same')
        for head in range(4):
            model.learn(before, f'actual:{head}', step=head, outcome_id='same-event')
            model.learn(before, f'actual:{head}', step=head, outcome_id='same-event')
            exact = model.components(observation(10, 'same'), head)[0]
            self.assertEqual(exact, {f'actual:{head}': 1.})
            self.assertEqual(set(model.predict(observation(10, 'same'), head)), {f'actual:{head}'})

    def test_same_arrival_from_overlapping_origins_has_one_global_vote(self):
        model = SemanticStatistics(semantic_strength=1.)
        model.learn(observation(1, 'old-a'), 'actual:A', outcome_id='shared')
        model.learn(observation(2, 'old-b'), 'actual:A', outcome_id='shared')
        model.learn(observation(3, 'old-c'), 'actual:B', outcome_id='distinct')
        self.assertEqual(model.components(observation(10, 'new'))[2], {'actual:A': .5, 'actual:B': .5})

    def test_overlapping_semantic_origins_use_max_vote_per_arrival(self):
        model = SemanticStatistics(semantic_strength=1.)
        model.learn(observation(1, 'old-a'), 'actual:A', outcome_id='shared')
        model.learn(observation(2, 'old-b'), 'actual:A', outcome_id='shared')
        model.learn(observation(3, 'old-c'), 'actual:B', outcome_id='distinct')
        semantic = model.components(observation(10, 'new'))[1]
        self.assertAlmostEqual(semantic['actual:A'], semantic['actual:B'])

    def test_cached_components_match_direct_prediction_and_do_not_mutate(self):
        records = [(observation(1, 'same'), 'actual:A', .5, 'a'),
                   (observation(2, 'related'), 'actual:B', 1., 'b'),
                   (observation(3, 'other', 'https://other.example/c.pdf'), 'actual:C', .2, 'c')]
        query = observation(10, 'same')
        source = SemanticStatistics(semantic_strength=1., half_life_hours=6.)
        for before, label, weight, event in records:
            source.learn(before, label, weight=weight, outcome_id=event)
        parts = source.components(query)
        saved = deepcopy(parts)
        for semantic_strength in (0., .25, 4.):
            for recency_mix in (0., .25, .5):
                direct = SemanticStatistics(semantic_strength=semantic_strength, global_strength=1.,
                                            half_life_hours=6., recency_mix=recency_mix)
                for before, label, weight, event in records:
                    direct.learn(before, label, weight=weight, outcome_id=event)
                self.assertEqual(direct.predict(query),
                                 scores_from_components(parts, semantic_strength, 1., recency_mix))
        source.learn(observation(20, 'same'), 'unseen:later', outcome_id='later')
        self.assertEqual(parts, saved)

    def test_similarity_cache_keys_metadata_snapshots_not_origin_only(self):
        cache = SimilarityCache()
        query = observation(10, 'query')['annotation']
        older = observation(1, 'same-id')['annotation']
        changed = observation(2, 'same-id', 'https://other.example/new.pdf')['annotation']
        strong = cache.score(query, older)
        self.assertGreater(strong, 0.)
        self.assertEqual(cache.score(query, changed), 0.)
        self.assertEqual(cache.score(older, query), strong)
        self.assertEqual(cache.score(query, older), strong)

    def test_invalid_training_weight_and_step_rejected(self):
        model = SemanticStatistics()
        for weight in (-1., math.nan, math.inf):
            with self.assertRaises(ValueError):
                model.learn(observation(1, 'same'), 'actual:A', weight=weight)
        with self.assertRaises(ValueError):
            model.learn(observation(1, 'same'), 'actual:A', step=4)


class SemanticRunnerTests(unittest.TestCase):
    def test_replay_requires_strict_maturity_and_grades_do_not_affect_predictions(self):
        data = {
            'observations': [observation(1, 'old'), observation(10, 'new'), observation(11, 'new')],
            'evaluation': [evaluation(1), evaluation(2)],
            'streams': {'test': [pending(0, 10, 1, 'actual:A', 'a')]},
        }
        result = runner.replay(data, 'test', SemanticStatistics())
        self.assertEqual(result[1][0], {})
        self.assertEqual(result[2][0], {'actual:A': 1.})
        altered = deepcopy(data)
        for row in altered['evaluation']:
            row.update(evidence_grade='unsupported', truth_action='unseen:future')
        self.assertEqual(runner.replay(altered, 'test', SemanticStatistics()), result)

    def test_replay_rejects_future_training_input_even_with_bad_ready_index(self):
        data = {'observations': [observation(10, 'same')], 'evaluation': [evaluation(0)],
                'streams': {'test': [pending(0, 1, 1, 'actual:A', 'a')]}}
        with self.assertRaisesRegex(ValueError, 'Future input'):
            runner.replay(data, 'test', SemanticStatistics())

    def test_removed_first_step_is_masked_and_retained_second_becomes_head_zero_only(self):
        data = {'observations': [observation(1, 'same')],
                'streams': {'clean': [pending(0, 10, 2, 'actual:B', 'b'),
                                      pending(0, 10, 3, 'actual:C', 'c')]}}
        rows = runner.training_rows(data, 'clean')
        self.assertEqual({r['head']: r['label'] for r in rows},
                         {0: 'actual:B', 2: 'actual:B', 3: 'actual:C'})

    def test_original_cleaning_single_label_stays_separate_from_observed_first_step(self):
        data = {'observations': [observation(1, 'same')],
                'streams': {'resolved': [pending(0, 10, 1, 'actual:A', 'a')]},
                'original_cleaning_stream': [{'position': 0, 'ready_index': 10,
                    'record': {'label': 'actual:B', 'outcome_id': 'b', 'outcome_timestamp': 2000}}]}
        rows = runner.training_rows(data, 'original_cleaning')
        self.assertEqual({r['head']: r['label'] for r in rows}, {0: 'actual:B', 1: 'actual:A'})

    def test_training_dedup_does_not_let_repeated_inputs_multiply_a_transition(self):
        data = {'observations': [observation(1, 'same'), observation(2, 'same')],
                'streams': {'test': [pending(0, 10, 1, 'actual:A', 'a'),
                                     pending(1, 11, 1, 'actual:A', 'a')]}}
        rows = runner.training_rows(data, 'test')
        self.assertEqual(len(rows), 2)  # One vote in head zero, one in observed head one.
        self.assertEqual({r['position'] for r in rows}, {0})

    def test_future_mix_excludes_single_head_and_missing_steps_remain_masked(self):
        data = {'evaluation': [evaluation(0, 'single:only')],
                'future': {'0': {'labels': [{'observed_step': 2, 'action_key': 'actual:B'}]}}}
        distributions = {0: {0: {'single:only': 1.}, 1: {}, 2: {'actual:B': 1.}, 3: {}}}
        metrics, predictions = runner.evaluate(data, distributions, (.6, .3, .1))
        self.assertEqual(predictions[0]['single'], ['single:only'])
        self.assertEqual(predictions[0]['future'], ['actual:B'])
        future = metrics['original']['future']
        self.assertEqual(future['steps'][1]['n'], 0)
        self.assertEqual(future['steps'][2]['n'], 1)
        self.assertEqual(future['steps'][3]['n'], 0)
        self.assertEqual(future['fully_observed_three_n'], 0)

    def test_repeated_future_target_counts_once_but_each_observed_head_is_scored(self):
        rows = [evaluation(0)]
        future = {'0': {'labels': [
            {'observed_step': 1, 'action_key': 'actual:A'},
            {'observed_step': 2, 'action_key': 'actual:B'},
            {'observed_step': 3, 'action_key': 'actual:A'},
        ]}}
        predictions = {0: {'future': ['actual:A'],
                           'heads': {1: ['actual:A'], 2: [], 3: ['actual:A']}}}
        metric = runner.future_metric(rows, predictions, future)
        self.assertEqual(metric['weighted_denominator'], 1.5)
        self.assertEqual(metric['weighted_covered'], 1.)
        self.assertAlmostEqual(metric['weighted_distinct_target_coverage'], 2 / 3)
        self.assertEqual(metric['steps'][3]['top4_hits'], 1)
        self.assertEqual(metric['fully_observed_three_n'], 1)


if __name__ == '__main__':
    unittest.main()
