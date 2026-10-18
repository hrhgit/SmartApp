"""Rebuild historical model families and evaluate ONLY one new shared time block.

Every learner updates from the same mature hard_kind navigation stream. No model
uses an old prediction file, a different validation subset, or new-set tuning.
"""
import argparse
from collections import Counter, defaultdict
from copy import deepcopy
from datetime import datetime, timezone
import gzip
import json
import math
import os
from pathlib import Path
import pickle
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import evaluation_contract as scoring
import evaluate_context_knn as core
import context_statistics as cs
import multibackground_model as multi
import joint_ranking_alternatives as joint
import statistical_residual as residual
import bayes_frequency_policy as risk
import compare_multibackground as comparison
from compare_semantic_trajectory import training_rows
from navigation_hierarchy import SERVICE, learning_record
from semantic_online import SemanticStatistics, normalize, rank
from diagnose_action_granularity import NativeSimilarity
from diagnose_background_contributions import diagnostic_features
from review_historical_models import ROOT, LEGACY, Sources, sha, write, replay_legacy, navigation_label


def log(**message):
    print(json.dumps(message, ensure_ascii=False), flush=True)


def current_only_similarity(left, right):
    # The legacy Jaccard helper treats two empty dictionaries as a perfect
    # match. Here history was deliberately removed, not observed identical.
    parts=list(core.similarity(left,right))
    parts[2]=0.
    return tuple(parts)


def save_object(out, name, model, probe):
    import cloudpickle
    payload = cloudpickle.dumps(model, protocol=5)
    restored = pickle.loads(payload)
    if probe(model) != probe(restored):
        raise ValueError('Reload parity failed: '+name)
    path = out/(name+'-state-private.pkl')
    path.write_bytes(payload); path.chmod(0o600)
    return path.name


class Experiment:
    def __init__(self, dataset, output):
        self.out, self.dataset = output, dataset
        self.sources = Sources()
        self.data = self.sources.read(dataset/'dataset-private.json.gz')
        self.meta = self.sources.read(dataset/'run.json')
        self.scope = self.sources.read(dataset/'summary.json')
        self.rows = scoring.prepare_evaluation(self.data, output)
        self.positions = {r['position'] for r in self.rows}
        self.outputs, self.models = {}, {}
        self.events = core.load_lines(self.sources.check(Path(self.meta['snapshot'])/'events.jsonl'))
        self.inputs, self.homes, audit = multi.prepare_inputs(self.events, self.data['observations'], core.KNOWN_TEST_SESSIONS)
        for q, original in zip(self.inputs, self.data['observations']):
            q.update(inventory_state=original['background_state'], inventory_quality=original['source_quality'],
                     inventory_through_index=q['index'])
        self.stream = [r for r in training_rows(self.data, 'hard_kind') if r['head'] == 0
                       and navigation_label(r['label'], self.data['observations'][r['position']])]
        self.boundary = self.meta['boundary_event_index']
        if not all(self.inputs[pos]['index'] >= self.boundary for pos in self.positions):
            raise ValueError('Old validation cutoff entered the new set')
        selection = self.sources.read(ROOT/LEGACY/'selection.json')
        self.legacy_configs = [deepcopy(v['config']) for v in selection['base_family_winners'].values()]
        for family in ('frequency', 'recency'):
            for scope in ('conditional', 'global'):
                config = {'id': 'stat_'+family+'_'+scope, 'family': family, 'parameters': {'scope': scope}}
                if not any(c['id'] == config['id'] for c in self.legacy_configs):
                    self.legacy_configs.append(config)
        self.legacy_configs.append(deepcopy(selection['best_current_only']['config']['parameters']['base']))
        stat_selection = self.sources.read(ROOT/'context-statistics-v1-20260923-tuning-02/selection.json')
        self.stat_configs = {name: value['config'] for name, value in stat_selection['selected'].items()}
        self.multi_configs = comparison.configurations()
        self.joint_configs = joint.configurations()
        from compare_statistical_residual import configurations
        self.residual_configs = configurations()
        import compare_temporal_context as temporal
        self.temporal_configs = [c for c in temporal.configurations()
                                 if c['mode'] != 'baseline' and c['sequence_weight'] == 0]
        self.early_configs = {}
        for family, run in (
            ('cluster', 'context-clusters-v2-20260923-experiment-01'),
            ('rule', 'context-rules-v1-20260923-experiment-02'),
            ('evidence', 'evidence-model-v1-20260923-experiment-01')):
            values = self.sources.read(ROOT/run/'metrics.json')['5']
            self.early_configs[family] = (values['models']['state_only']['config'] if family == 'cluster'
                                          else values['selected']['config'])
        self.plan = {'scope': self.scope, 'scoring_contract': scoring.contract(),
            'legacy_configs': self.legacy_configs, 'statistics_configs': self.stat_configs,
            'multi_configs': self.multi_configs, 'joint_configs': self.joint_configs,
            'residual_configs': self.residual_configs, 'early_context_configs': self.early_configs,
            'temporal_configs': self.temporal_configs,
            'early_knn_config': {'state_weight': 1., 'k': 5, 'half_life_hours': 1.},
            'labels': 'same hard_kind first retained navigation stream for every family',
            'forecast_label': 'same stable5s observed next destination',
            'input': 'foreground length1; optional unordered temporal statistics or observed background by model',
            'learning': 'prequential; only labels strictly mature before each cutoff; includes earlier mature validation feedback',
            'parameters': 'historical fixed parameters; no fit/selection on validation scores',
            'new_dataset_only': self.meta.get('evaluation_scope') != 'complete_two_cohorts',
            'evaluation_scope': self.meta.get('evaluation_scope','fresh_tail'),
            'report_cohorts': (['all','evidence_total'] if self.meta.get('evaluation_scope') == 'complete_two_cohorts' else 'legacy_diagnostics'),
            'primary_metric': ('rank_score.mean times 100' if self.meta.get('evaluation_scope') == 'complete_two_cohorts' else 'exact_top4'),
            'reused_historical_forecasts': False,
            'inventory_or_same_origin_output_whitelist': False,
            'old_context_adaptation': 'remove ordered history; use mature global action vocabulary; abstention remains explicit miss',
            'training_stream_digest': scoring.digest(self.stream), 'training_relations': len(self.stream),
            'feature_audit': audit}
        write(output/'plan.json', self.plan)
        write(output/'inputs-private.json.gz', self.inputs)
        log(phase='plan_frozen', validation_n=len(self.rows), evidence_n=self.scope['sizes']['evidence_total'],
            training_relations=len(self.stream))

    def add(self, name, outputs, family, config=None, state=None, **extra):
        if set(map(int, outputs)) != self.positions:
            raise ValueError('Every model must predict every common cutoff: '+name)
        outputs = {int(k): v for k, v in outputs.items()}
        if name in self.outputs or any(len(v['predictions']) > 4 for v in outputs.values()):
            raise ValueError('Duplicate model or invalid menu')
        self.outputs[name] = outputs
        self.models[name] = {'family': family, 'config': config, 'state': state,
                            'common_training_cleaning': 'hard_kind', 'full_common_validation': True, **extra}
        write(self.out/'predictions-private.json.gz', self.outputs)
        write(self.out/'models.json', self.models)

    def legacy(self):
        for index, config in enumerate(self.legacy_configs):
            outputs, payload, audit = replay_legacy(self.data, self.rows, config, 'hard_kind')
            state = f'legacy-{index:02d}-state-private.pkl'
            (self.out/state).write_bytes(payload)
            write(self.out/f'legacy-{index:02d}-audit-private.json', audit)
            self.add('legacy/'+config['id'], outputs, config['family'], config, state,
                     adaptation='current-only; unrestricted mature labels; common navigation semantics')
            log(phase='legacy_trained', done=index+1, total=len(self.legacy_configs), family=config['family'])

    def statistics_and_examples(self):
        background = cs.BackgroundIndex(self.inputs)
        memory = cs.StatisticalMemory(); cursor = 0
        outputs = {name: {} for name in self.stat_configs}
        native = {}; examples = {}; training = []; decision_context = {}
        seen_outcomes, global_counts, catalog, root_vocabularies = set(), Counter(), {}, {}
        unseen_count = 0
        time_config = self.stat_configs['time']
        for q in self.inputs:
            while cursor < len(self.stream) and self.stream[cursor]['ready_index'] < q['index']:
                row = self.stream[cursor]; before = self.inputs[row['position']]
                n = len(memory.hierarchy.records)
                memory.learn(before, learning_record(row, before), row['ready_index'])
                if len(memory.hierarchy.records) > n:
                    record = memory.hierarchy.records[-1]
                    if record['navigationDisposition'] != 'navigation':
                        raise ValueError('Non-navigation label reached common training stream')
                    example = examples[row['position']]
                    label = example['mapping'].get(record['navigationLabel'], record['navigationLabel'])
                    training.append({'position': row['position'], 'input_index': before['index'],
                        'ready_index': row['ready_index'], 'label': label, 'weight': record['weight'],
                        'key': [record['origin'], record['label'], record['outcomeID']]})
                    catalog[record['destination']['id']] = record['destination']
                    if record['outcomeID'] not in seen_outcomes:
                        seen_outcomes.add(record['outcomeID']); global_counts[record['label']] += record['weight']
                        root = scoring.service(record['destination']) or record['label']
                        unseen_count += root not in root_vocabularies[row['position']]
                cursor += 1
            prepared = memory.prepare(q, self.homes[q['position']], background)
            prior_scores = prepared.scores(time_config)
            mapping = {prepared.labels[i]: prepared.emitted[j] for i, j in zip(prepared.keep, prepared.output_ids)}
            descriptions = {}
            for record in memory.hierarchy.records:
                emitted = mapping.get(record['navigationLabel'])
                if emitted is None: continue
                target = record['destination']
                description = descriptions.setdefault(emitted, {'aliases': set(),
                    'owner': target.get('bundleID', ''), 'kind': target.get('kind', 'unknown')})
                description['aliases'].add(record['label'])
            descriptions = {k: {**v, 'aliases': sorted(v['aliases'])} for k, v in descriptions.items()}
            example = residual.encode_features(q, prior_scores, descriptions)
            example.update(mapping=mapping, gate=joint.gate_features(q, example['prior']))
            examples[q['position']] = example
            catalog[q['target']['id']] = q['target']
            aliases, entries = scoring.decision_hierarchy(catalog.values())
            aliases[risk.UNKNOWN_CHILD] = scoring.SERVICE
            root_vocabularies[q['position']] = {aliases.get(label, label) for label in example['labels']}
            if q['position'] in self.positions:
                for name, config in self.stat_configs.items():
                    outputs[name][q['position']] = prepared.predict(config)
                native[q['position']] = comparison.bundle(normalize(memory.hierarchy.scores(q['target'])), self.homes[q['position']])
                local = [(i, r) for i, r in enumerate(memory.hierarchy.records) if r['origin'] == q['origin']]
                indices = {tuple(item['key']): item['position'] for item in training}
                local_positions = np.asarray([indices[(r['origin'], r['label'], r['outcomeID'])] for _, r in local], dtype=np.int64)
                factors = background.factors(q['position'], local_positions, time_config) if local else []
                local_counts = Counter()
                for (_, record), factor in zip(local, factors):
                    local_counts[record['label']] += min(1., record['weight'])*float(factor)
                decision_context[q['position']] = {'aliases': aliases, 'entries': entries,
                    'global_counts': dict(global_counts), 'local_counts': dict(local_counts),
                    'unseen_p': (unseen_count+1)/(len(seen_outcomes)+2),
                    'latest_ready_index': self.stream[cursor-1]['ready_index'] if cursor else -1}
            if (q['position']+1) % 250 == 0:
                log(phase='common_causal_features', inputs=q['position']+1, total=len(self.inputs))
        self.examples, self.training, self.decision_context = examples, training, decision_context
        self.stat_scores = {pos: dict(zip(examples[pos]['labels'], map(float, examples[pos]['prior']))) for pos in self.positions}
        self.baseline_scores = {pos: normalize(memory_scores) for pos, memory_scores in self.stat_scores.items()}
        state = save_object(self.out, 'context-statistical-memory', memory,
                            lambda model: model.hierarchy.scores(self.inputs[-1]['target']))
        for name, result in outputs.items():
            self.add('statistics/'+name, result, 'context_statistics', self.stat_configs[name], state)
        self.add('navigation/v2', native, 'navigation_hierarchy_v2', state=state)
        # Each original example retains its own prior; no final-state feature reconstruction.
        import cloudpickle
        with gzip.open(self.out/'causal-examples-private.pkl.gz', 'wb') as f:
            f.write(cloudpickle.dumps({'examples': examples, 'training': training}, protocol=5))
        write(self.out/'decision-context-private.json.gz', {pos: {**ctx, 'entries': sorted(ctx['entries'])}
                                                          for pos, ctx in decision_context.items()})
        log(phase='common_examples_ready', training_relations=len(training), statistical_models=len(outputs)+1)

    def train_rankers(self):
        self.reference_scores = {}
        jobs = [('joint', c, joint.JointRanker) for c in self.joint_configs]
        jobs += [('residual', c, residual.ResidualRanker) for c in self.residual_configs]
        for index, (group, config, cls) in enumerate(jobs):
            model = cls(config); cursor = 0; outputs = {}
            for q in self.inputs:
                while cursor < len(self.training) and self.training[cursor]['ready_index'] < q['index']:
                    model.learn(self.training[cursor], self.examples, q['index']); cursor += 1
                if q['position'] in self.positions:
                    scores = model.scores(self.examples[q['position']])
                    outputs[q['position']] = comparison.bundle(scores, self.homes[q['position']])
                    if config['id'] == 'linear_top2_lr0.1_seed20260923':
                        self.reference_scores[q['position']] = scores
            state = model.save(self.out, group+'-'+config['id'])
            restored = cls.load(state)
            for pos in self.positions:
                if model.scores(self.examples[pos]) != restored.scores(self.examples[pos]):
                    raise ValueError('Ranker reload mismatch')
            self.add(group+'/'+config['id'], outputs, group, config, state.name, audit=dict(model.audit))
            log(phase='ranker_trained', completed=index+1, total=len(jobs), model=config['id'])

    def multi_background(self):
        # Original multi-background experiments mixed with the default native
        # statistics, not the re-tuned time prior. Reconstruct that same branch.
        baseline, _ = comparison.replay_baseline(self.inputs, self.stream, self.rows, self.homes)
        original_features = multi.features
        jobs = [(c, None) for c in self.multi_configs]
        template = next(c for c in self.multi_configs if c['id'] == 'interaction_softmax_combined_rich_lr0.1')
        jobs += [({**template, 'id': template['id']+'_'+v, 'diagnostic_variant': v}, v)
                 for v in ('inventory_only', 'time_and_inventory')]
        for index, (config, variant) in enumerate(jobs):
            if variant:
                with patch.object(multi, 'features', lambda q,c: diagnostic_features(q,c,original_features)):
                    model, outputs, _ = comparison.replay_joint(self.inputs, self.stream, self.rows, self.homes,
                        baseline, config, self.inputs[-1]['index'])
                    state = save_object(self.out, 'background-'+str(index), model, lambda m: m.scores(self.inputs[-1]))
            else:
                model, outputs, _ = comparison.replay_joint(self.inputs, self.stream, self.rows, self.homes,
                    baseline, config, self.inputs[-1]['index'])
                state = save_object(self.out, 'background-'+str(index), model, lambda m: m.scores(self.inputs[-1]))
            for weight, forecasts in outputs.items():
                suffix = 'pure' if weight == 1 else 'mix50'
                self.add('background/'+config['id']+'_'+suffix, forecasts, 'multi_background',
                         {**config, 'learning_weight': weight}, state)
            log(phase='background_trained', completed=index+1, total=len(jobs), model=config['id'])

    def decisions(self):
        outputs = {mode: {} for mode in risk.MODES}
        self.preference_weights = {mode: {} for mode in risk.MODES}
        context_log = {}
        for row in self.rows:
            pos = row['position']; q = self.inputs[pos]; ctx = self.decision_context[pos]
            if ctx['latest_ready_index'] >= q['index']:
                raise ValueError('Unmatured decision evidence')
            posterior, actions = risk.expand_service_probabilities(self.reference_scores[pos], ctx['aliases'],
                ctx['entries'], Counter(ctx['global_counts']), Counter(ctx['local_counts']), ctx['unseen_p'])
            forbidden = {q['target']['id'], 'app:'+q['target'].get('bundleID', '')}
            if q['target'].get('locator') == 'https://chatgpt.com/': forbidden.add(SERVICE)
            actions = [a for a in actions if a not in forbidden]
            context_log[pos] = {'posterior': posterior, 'costs': {}}
            for mode in risk.MODES:
                costs = risk.frequency_costs(posterior, Counter(ctx['global_counts']), Counter(ctx['local_counts']), mode)
                menu, losses = risk.risk_menu(posterior, actions, ctx['aliases'], ctx['entries'], costs)
                home = self.homes[pos]
                outputs[mode][pos] = {'predictions': [home if x == SERVICE and home else x for x in menu]}
                context_log[pos]['costs'][mode] = costs
                truth = row['truth_action']
                unknown = risk.UNKNOWN_CHILD if row['_scoring']['aliases'].get(truth) == SERVICE and risk.UNKNOWN_CHILD in costs else risk.UNKNOWN
                self.preference_weights[mode][pos] = costs.get(truth, costs[unknown])
        for mode, predictions in outputs.items():
            self.add('risk/'+mode, predictions, 'minimum_risk', {'mode': mode, 'fallback_gain': scoring.FALLBACK_GAIN})
        write(self.out/'risk-posterior-and-costs-private.json.gz', context_log)
        write(self.out/'common-frequency-weights-private.json', self.preference_weights)

    def temporal_models(self):
        import compare_temporal_context as temporal
        from temporal_context import build_contexts, TemporalNavigationModel, ContextSimilarity
        contexts, homes, audit = build_contexts(self.events, self.data['observations'], core.KNOWN_TEST_SESSIONS)
        for context in contexts.values():
            context['sequence'] = context['sequence'][-1:]
        inputs = temporal.model_inputs(self.data, contexts)
        similarity = ContextSimilarity()
        for config in self.temporal_configs:
            model = TemporalNavigationModel(config, similarity); cursor = 0; outputs = {}
            for row in self.rows:
                q = inputs[row['position']]
                while cursor < len(self.stream) and self.stream[cursor]['ready_index'] < q['index']:
                    item = self.stream[cursor]; before = inputs[item['position']]
                    model.learn(before, learning_record(item, before)); cursor += 1
                ranked = model.ranked(q, n=10**9)
                resolved = [homes.get(q['position']) if x == SERVICE and homes.get(q['position']) else x for x in ranked]
                outputs[q['position']] = {'predictions': resolved[:4], 'rankable': resolved}
            state = save_object(self.out, 'temporal-'+config['id'], model, lambda m:m.ranked(inputs[-1]))
            self.add('temporal/'+config['id'], outputs, 'temporal_similarity', config, state, feature_audit=audit)
        log(phase='temporal_complete', models=len(self.temporal_configs))

    def early_context(self):
        import evaluate_context_clusters as clusters
        import evaluate_context_rules as rules
        import evaluate_evidence_model as selective
        objects = self.sources.read(self.dataset/'observation-objects-private.json.gz')
        items = [SimpleNamespace(**obj) for obj in objects]
        for q, origin in zip(items, self.data['observations']):
            q.history = {}; q.history_keys = []; q.current = origin['origin']
            q.truth = '__unknown__'; q.truth_meta = {}; q.candidates = {}; q.recency = {}
        keys = [rules.conditions(q) for q in items]; rule_index = defaultdict(list)
        rc = {**self.early_configs['rule'], 'history': 'none', 'representation': 'exact'}
        ec = {**self.early_configs['evidence'], 'history_min': 0., 'representation': 'exact'}
        cc = self.early_configs['cluster']
        fit_positions = sorted({r['position'] for r in self.stream if r['ready_index'] < self.boundary})
        idf = clusters.background_idf(items, fit_positions) if cc.get('background_idf') else None
        features = [clusters.feature_row(q, [], 'state_only', 0., cc.get('foreground_fraction', .3), idf) for q in items]
        cluster = clusters.fit_clusters(features, fit_positions, cc['k'])
        forecasts = {name:{} for name in ('knn_state', 'knn_current_only', 'kmeans_state', 'selective_rule', 'selective_evidence')}
        valid = set(); ledger = {}; times = {}; vocabulary = {}; recency = {}; cursor = 0
        for row in self.rows:
            i = row['position']; q = items[i]
            while cursor < len(self.stream) and self.stream[cursor]['ready_index'] < q.index:
                item = self.stream[cursor]; j = item['position']; target = dict(item['record']['target'])
                target.update(id=item['label'], _bundle=target.get('bundleID'))
                if item['label'].startswith('app:'): target.update(kind='app', parentID=None)
                items[j].truth = item['label']; items[j].truth_meta = target; items[j].ready_index = item['ready_index']
                ledger[j] = {'target_event_id': item['outcome_id']}
                times[item['outcome_id']] = (items[j].session, item['outcome_timestamp'])
                vocabulary[item['label']] = target; recency[item['label']] = item['outcome_timestamp']
                if j not in valid:
                    valid.add(j)
                    for key in keys[j]: rule_index[key].append(j)
                cursor += 1
            q.candidates = {k:v for k,v in vocabulary.items() if navigation_label(k, self.data['observations'][i])}
            q.recency = {k:v for k,v in recency.items() if k in q.candidates}
            pairs = {i:[(j, current_only_similarity(q, items[j])) for j in sorted(valid)]}
            for name, no_background in (('knn_state', False), ('knn_current_only', True)):
                prediction, detail = core.predict_knn(items, i, pairs, weight=1., k=5, half_life_hours=1., no_background=no_background)
                detail.pop('neighbor_label_support', None)
                forecasts[name][i] = {'predictions':prediction, **detail}
            counts = clusters.label_counts(items, valid, ledger, cluster['assignments'], i,
                cc.get('max_age_hours'), {key:t[1] for key,t in times.items()})
            forecasts['kmeans_state'][i] = clusters.rank_frequency(core, q, counts)
            matches = rules.query_rules(items, ledger, times, i, keys, rule_index, rc)
            forecasts['selective_rule'][i] = rules.predict(core,q,matches,rc['min_support'],rc['min_agreement'])
            epairs = {i:[(j,parts) for j,parts in pairs[i] if parts[3]>=.8 and parts[5]>=.25 and parts[4]>=.5]}
            evidence = selective.evidence(core,items,ledger,times,i,epairs,ec)
            if not selective.accept(evidence,ec['min_support'],ec['min_agreement']):
                evidence = {**evidence, 'predictions':[], 'abstain_reason':evidence.get('abstain_reason') or 'support_gate'}
            forecasts['selective_evidence'][i] = evidence
            # Do not persist the legacy API's transient vocabulary as collected inventory.
            q.candidates = {}; q.recency = {}
        state = save_object(self.out,'early-context',{'items':items,'ledger':ledger,'times':times,
            'cluster':cluster,'fit_positions':fit_positions,'keys':keys,'rule_index':dict(rule_index),
            'vocabulary':vocabulary,'recency':recency}, lambda obj:obj['cluster']['assignments'])
        for name, outputs in forecasts.items():
            config = rc if name=='selective_rule' else ec if name=='selective_evidence' else cc if name=='kmeans_state' else self.plan['early_knn_config']
            self.add('early/'+name, outputs, 'early_context', config, state,
                adaptation='foreground length1; unrestricted mature action vocabulary; abstention counted in common denominator')
        log(phase='early_context_complete', models=len(forecasts), cluster_fit_last_position=max(fit_positions))

    def semantics(self):
        model = SemanticStatistics(semantic_strength=16, global_strength=1, recency_mix=.25, similarity=NativeSimilarity())
        stream = [r for r in training_rows(self.data, 'hard_kind')
                  if navigation_label(r['label'], self.data['observations'][r['position']])]
        cursor = 0; outputs = {}; future = {}
        for row in self.rows:
            q = self.data['observations'][row['position']]
            while cursor < len(stream) and stream[cursor]['ready_index'] < q['index']:
                item = stream[cursor]; before = self.data['observations'][item['position']]
                model.learn(before, item['label'], weight=item['weight'], step=item['head'],
                            outcome_id=item['outcome_id'], outcome_timestamp=item['outcome_timestamp'])
                cursor += 1
            scores = {k: v for k, v in model.predict(q, 0).items() if navigation_label(k, q)}
            outputs[row['position']] = comparison.bundle(scores, self.homes[row['position']])
            from semantic_online import blend
            combined = blend([model.predict(q, head) for head in (1,2,3)], [1., .5, .25])
            future[row['position']] = comparison.bundle({k:v for k,v in combined.items() if navigation_label(k,q)}, self.homes[row['position']])
        state = save_object(self.out, 'semantic-statistics', model, lambda m: m.predict(q, 0))
        self.add('semantic/single', outputs, 'semantic_statistics', {'semantic_strength':16, 'global_strength':1,'recency_mix':.25}, state)
        self.add('semantic/multistep', future, 'semantic_multistep', {'future_weights':[1.,.5,.25]}, state,
                 note='same independently deduplicated observed stream; multiple heads, evaluated on same next-step truth')

    def evaluate(self):
        parts = scoring.cohorts(self.rows)
        evidence = parts['evidence_total']
        if self.meta.get('evaluation_scope') == 'complete_two_cohorts':
            parts = {k:parts[k] for k in ('all','evidence_total')}
        else:
            for grade in ('single','repeated','strong'):
                parts['evidence_'+grade] = [r for r in evidence if
                    ('single' if r['evidence_count']==1 else 'repeated' if r['evidence_count']<5 else 'strong') == grade]
        metrics = {name: {g:scoring.measure(rows,p) for g,rows in parts.items()} for name,p in self.outputs.items()}
        for group in parts:
            scoring.require_comparable(*[m[group] for m in metrics.values()])
        common = {name: {mode:{g:scoring.measure_weighted_loss(rows,p,weights) for g,rows in parts.items()}
                   for mode,weights in self.preference_weights.items()} for name,p in self.outputs.items()}
        for mode in risk.MODES:
            for group in parts:
                scoring.require_comparable(*[m[mode][group] for m in common.values()])
        reference = 'joint/linear_top2_lr0.1_seed20260923'
        pairs = {name: scoring.paired_topk(evidence, self.outputs[reference], p) for name,p in self.outputs.items()}
        write(self.out/'metrics-private.json', metrics); write(self.out/'cross-costs-private.json', common)
        write(self.out/'paired-private.json', pairs)
        self.report(metrics, common)
        return metrics

    def report(self, metrics, cross):
        if self.meta.get('evaluation_scope') == 'complete_two_cohorts':
            return self.report_complete(metrics)
        evidence_n = self.scope['sizes']['evidence_total']; all_n = self.scope['sizes']['all']
        def cell(m,k,exact=True):
            value = m['exact'] if exact else m
            return f"{value['top'+str(k)+'_hits']}/{value['n']}"
        def score(m): return f"{100*m['rank_score']['mean']:.2f}" if m['n'] else '—'
        ordered = sorted(metrics, key=lambda n: (-metrics[n]['evidence_total']['exact']['top4_hits'],
                                               -metrics[n]['evidence_total']['rank_score']['mean'],n))
        lines = ['# 全部模型在同一全新验证集上的统一比较', '',
            f'主成绩：全部有历史证据样本 **{evidence_n}条**，覆盖全部合法验证样本 **{evidence_n}/{all_n}={evidence_n/all_n:.1%}**。原始可评分候选43条，统一合法性审查后42条；所有模型均使用完全相同的成员与评分标签。', '',
            f"新验证区间（UTC）：{self.scope['validation_range_utc'][0]} 至 {self.scope['validation_range_utc'][1]}。截点全部晚于此前最后一次评测快照，没有混入原来的84条有证据样本或旧模型的局部样本。", '',
            f'共 **{len(metrics)}组模型／配置**，全部从同一快照重新训练并按时间回放；没有复用旧预测。前台长度1，统一hard_kind训练清洗与导航动作语义，不用库存或同起点支持集硬限制输出。', '',
            '参数来自历史方案或预先声明的适配，不用新验证分数调参。在线模型只学习当前截点之前已成熟的反馈；验证段前面已成熟的行为可以帮助后面的预测，各模型遵守同一时间规则。', '',
            '评分：精确收益1、正确服务首页回退0.6、错误0；综合分是Top1至Top4前缀最佳收益均值×100。决策层也按同一个0/0.4/1损失重新生成菜单。服务接受TopK与精确TopK分开。', '',
            '## 全部有证据样本（所有行同一分母）', '',
            '| 模型 | 精确Top1 | 精确Top2 | 精确Top4 | 接受Top1 | 接受Top4 | 综合分 | 返回覆盖 |',
            '|---|---:|---:|---:|---:|---:|---:|---:|']
        for name in ordered:
            m=metrics[name]['evidence_total']
            lines.append('| '+name+' | '+' | '.join([cell(m,k) for k in (1,2,4)]+
                [cell(m,1,False),cell(m,4,False),score(m),f"{m['prediction_coverage']:.1%}"])+' |')
        lines += ['', '## 全部合法验证样本（含无证据样本，辅助诊断）', '',
            '| 模型 | 精确Top1 | 精确Top4 | 综合分 | 返回覆盖 |', '|---|---:|---:|---:|---:|']
        for name in ordered:
            m=metrics[name]['all']
            lines.append('| '+name+' | '+' | '.join([cell(m,1),cell(m,4),score(m),f"{m['prediction_coverage']:.1%}"])+' |')
        lines += ['', '## 同一频率权重的交叉评价', '',
            'Top1实际平均损失，越低越好。每列权重对所有模型完全相同，只读取截点前成熟行为；权重参考由固定线性Top2策略的结果空间冻结，不随被评分模型改变。', '',
            '| 模型 | 固定损失 | 全局频率损失 | 情景频率损失 |', '|---|---:|---:|---:|']
        for name in ordered:
            lines.append('| '+name+' | '+' | '.join(f"{cross[name][mode]['evidence_total']['mean']:.4f}" for mode in risk.MODES)+' |')
        lines += ['', '## 解释边界', '',
            '- 10条有证据验证样本中一条就是10个百分点。可作为新时段对照，不能据此宣布长期胜负。各随机种子完整列出，不能只挑最好的一次。',
            '- 多步监督与单步监督使用同一原始成熟行为流；多步分支单列，并仍接受相同的下一步评分目标。',
            '- 早期上下文方法统一移除近期有序路径和库存候选限制，明确作为适配模型，不冒称原实现原封不动复现。暂不预测保留为空输出计入共同分母。',
            '- 数据样本、评分代码、规则与成员哈希已冻结。所有模型都预测全部验证截点，合法性排除不依赖模型是否命中。',
            '- 原始采集、历史run、运行模型和安装包均未修改。', '']
        (self.out/'report-private.md').write_text('\n'.join(lines))

    def report_complete(self, metrics):
        sizes=self.scope['sizes']
        lines=['# 完整验证集：统一综合分', '',
            f"所有{len(metrics)}组模型都预测同一完整集合。完整验证集{sizes['all']}条；完整有证据验证集{sizes['evidence_total']}条，覆盖率{sizes['evidence_total']/sizes['all']:.1%}。两个集合固定，不按模型是否命中或是否愿意预测筛选。", '',
            f"原固定集{self.scope['fixed_candidate_n']}条＋新增{self.scope['tail_candidate_n']}条，共{len(self.rows)}条待审样本，统一合法性审查后{sizes['all']}条。保留旧固定成员与评分标签，不把训练专用截点混入验证集。", '',
            '综合分（0–100，越高越好）：精确匹配收益1，正确服务首页回退0.6，错误0；对Top1、Top2、Top3、Top4各前缀的最佳匹配收益取平均，再对全部样本取平均×100。位置越靠前得分越高，同一需求不重复累计。空输出为0分，仍计入分母。', '',
            '以下按完整有证据综合分排序，完整验证集综合分同时列出；不按Top4单项排序，也不混用不同损失定义。包含种子、特征对照与预声明融合比例，配置数不等于独立算法数。', '',
            '| 模型／配置 | 完整验证集综合分 | 完整有证据验证集综合分 |', '|---|---:|---:|']
        ordered=sorted(metrics,key=lambda n:(-metrics[n]['evidence_total']['rank_score']['mean'],
                                            -metrics[n]['all']['rank_score']['mean'],n))
        for name in ordered:
            lines.append(f"| {name} | {100*metrics[name]['all']['rank_score']['mean']:.2f} | {100*metrics[name]['evidence_total']['rank_score']['mean']:.2f} |")
        lines+=['','## 口径与边界','',
            '- 所有模型重新从头按成熟边界回放；相同hard_kind训练清洗、前台长度1、评分标签与评分代码，不使用当前库存或同起点输出硬限制。',
            '- 旧方案必要适配明确记录；选择性模型拒绝预测仍计入完整分母。KMeans中心仅拟合于最早验证截点之前。',
            '- 完整验证集包含此前已用于选参及反复查看的历史开发样本，不称全新独立测试；本轮不再据此调参或部署。',
            '- 有证据集合是基于真实后续目标的诊断集合，证据本身必须在预测截点前已成熟；不是线上可识别的置信保证。',
            '- 精确指既定动作标签；普通窗口所属应用标签不冒充精确窗口命中。评分不改变原始采集、数据库或正在运行的模型。','']
        (self.out/'report-private.md').write_text('\n'.join(lines))

    def finish(self):
        self.sources.verify_unchanged()
        for p in Path(__file__).parent.glob('*.py'):
            (self.out/('source-'+p.name)).write_bytes(p.read_bytes())
        write(self.out/'run.json', {'complete': True, 'created_at':datetime.now(timezone.utc).isoformat(),
            'dataset':str(self.dataset), 'snapshot':self.meta['snapshot'], 'sources':self.sources.accessed,
            'scoring_contract':scoring.contract(), 'models':len(self.models), 'every_model_same_validation':True,
            'evaluation_scope':self.meta.get('evaluation_scope','fresh_tail'),
            'historical_predictions_reused':False, 'validation_hyperparameter_tuning':False,
            'collector_or_runtime_changed':False})
        for p in self.out.iterdir():
            if p.is_file(): p.chmod(0o600)
        write(self.out/'artifact-hashes.json', {p.name:sha(p) for p in self.out.iterdir() if p.is_file()})
        log(phase='complete', output=str(self.out), models=len(self.models))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();out=args.output.resolve()
    if out.exists() or out == ROOT.resolve() or not out.is_relative_to(ROOT.resolve()):
        parser.error('Use a new private run')
    os.umask(0o077);out.mkdir(mode=0o700)
    experiment=Experiment(args.dataset.resolve(),out)
    experiment.legacy()
    experiment.statistics_and_examples()
    experiment.train_rankers()
    experiment.multi_background()
    experiment.temporal_models()
    experiment.early_context()
    experiment.semantics()
    experiment.decisions()
    experiment.evaluate()
    experiment.finish()


if __name__=='__main__':main()
