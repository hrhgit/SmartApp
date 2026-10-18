"""Frozen-cohort causal replay of enrichment, cleaning, and future supervision."""
import argparse
from collections import Counter, defaultdict
import gzip
import itertools
import json
import os
from pathlib import Path

import evaluate_context_knn as io
from semantic_online import SemanticStatistics, SimilarityCache, scores_from_components, blend, rank
from summarize_tuned_history import verify_run


STEP_WEIGHTS = [1., .5, .25]
MIXES = [(1., 0., 0.), (.8, .15, .05), (.6, .3, .1), (1/3, 1/3, 1/3)]


def groups(rows):
    result = {'all': rows, 'original': [r for r in rows if r['evidence_grade'] == 'original'],
              'expanded': [r for r in rows if r['evidence_grade'] != 'unsupported']}
    for grade in ('added_exact', 'semantic_strong', 'semantic_weak', 'unsupported'):
        result[grade] = [r for r in rows if r['evidence_grade'] == grade]
    result['late_original'] = [r for r in result['original'] if r['split'] == 'test']
    return result


def training_rows(data, policy):
    """0: first retained label. 1..3: original observed order, with masks."""
    result, by_position = [], defaultdict(list)
    source = data['streams']['resolved' if policy == 'original_cleaning' else policy]
    for row in source:
        rec = row['record']
        result.append({**row, 'head': rec['observed_step'], 'label': rec['action_key'],
                       'weight': rec['weight'], 'outcome_id': rec['event_id'],
                       'outcome_timestamp': rec['arrival_timestamp_ms']})
        by_position[row['position']].append(row)
    if policy == 'original_cleaning':
        for row in data['original_cleaning_stream']:
            rec = row['record']
            result.append({**row, 'head': 0, 'label': rec['label'], 'weight': 1.,
                           'outcome_id': rec['outcome_id'], 'outcome_timestamp': rec['outcome_timestamp']})
    else:
        for position, rows in by_position.items():
            row = min(rows, key=lambda r:r['record']['observed_step'])
            rec = row['record']
            result.append({**row, 'head': 0, 'label': rec['action_key'], 'weight': rec['weight'],
                           'outcome_id': rec['event_id'], 'outcome_timestamp': rec['arrival_timestamp_ms']})
    # One natural transition must not get extra votes from label refinements.
    seen, dedup, assigned = set(), [], {}
    for row in sorted(result, key=lambda r:(r['ready_index'], r['position'], r['head'])):
        origin = data['observations'][row['position']]['origin']
        identity = (row['head'], origin, row['outcome_id'])
        if identity in assigned and assigned[identity] != row['label']:
            raise ValueError('Conflicting labels for one independent outcome need an explicit correction policy')
        assigned[identity] = row['label']
        key = (row['head'], origin, row['label'], row['outcome_id'])
        if key not in seen:
            seen.add(key); dedup.append(row)
    return dedup


def replay(data, policy, model, components=False):
    stream = training_rows(data, policy)
    cursor, result = 0, {}
    for row in sorted(data['evaluation'], key=lambda r:r['position']):
        q = data['observations'][row['position']]
        while cursor < len(stream) and stream[cursor]['ready_index'] < q['index']:
            before = stream[cursor]
            x = data['observations'][before['position']]
            if x['index'] >= q['index']:
                raise ValueError('Future input entered training')
            kwargs = {'weight': before['weight'], 'step': before['head']}
            if isinstance(model, SemanticStatistics):
                kwargs.update(outcome_id=before['outcome_id'], outcome_timestamp=before['outcome_timestamp'])
            model.learn(x, before['label'], **kwargs)
            cursor += 1
        result[row['position']] = {head: (model.components(q, head) if components else model.predict(q, head))
                                  for head in range(4)}
    return result


def single_metric(rows, predictions):
    hits = [predictions[r['position']]['single'] for r in rows]
    n = len(rows)
    top4 = sum(r['truth_action'] in p for r, p in zip(rows, hits))
    top1 = sum(bool(p) and r['truth_action'] == p[0] for r, p in zip(rows, hits))
    return {'n': n, 'top4_hits': top4, 'top1_hits': top1,
            'top4': top4/n if n else None, 'top1': top1/n if n else None}


def future_metric(rows, predictions, future):
    steps = {i: {'n': 0, 'top1_hits': 0, 'top4_hits': 0} for i in (1, 2, 3)}
    n, all3, any_hit, covered, denominator = 0, 0, 0, 0., 0.
    complete_covered = complete_denominator = 0.
    for row in rows:
        labels = future[str(row['position'])]['labels']
        if not labels:
            continue
        n += 1
        all3 += len(labels) == 3
        prediction = predictions[row['position']]
        distinct = {}
        for label in labels:
            head, actual = label['observed_step'], label['action_key']
            p = prediction['heads'][head]
            steps[head]['n'] += 1
            steps[head]['top1_hits'] += int(bool(p) and p[0] == actual)
            steps[head]['top4_hits'] += int(actual in p)
            distinct.setdefault(actual, STEP_WEIGHTS[head-1])
        value = sum(w for key,w in distinct.items() if key in prediction['future'])
        total = sum(distinct.values())
        covered += value; denominator += total
        any_hit += bool(value)
        if len(labels) == 3:
            complete_covered += value; complete_denominator += total
    for metric in steps.values():
        for k in ('top1', 'top4'):
            metric[k] = metric[k+'_hits']/metric['n'] if metric['n'] else None
    return {'observed_future_n': n, 'fully_observed_three_n': all3, 'steps': steps,
            'any_future_hit_n': any_hit, 'any_future_hit_rate': any_hit/n if n else None,
            'weighted_distinct_target_coverage': covered/denominator if denominator else None,
            'weighted_covered': covered, 'weighted_denominator': denominator,
            'full_three_weighted_coverage': complete_covered/complete_denominator if complete_denominator else None}


def evaluate(data, distributions, mix):
    predictions = {pos: {'single': rank(heads[0]), 'heads': {h:rank(heads[h]) for h in (1,2,3)},
                          'future': rank(blend([heads[h] for h in (1,2,3)], mix))}
                   for pos, heads in distributions.items()}
    return {name: {'single': single_metric(rows, predictions),
                   'future': future_metric(rows, predictions, data['future'])}
            for name, rows in groups(data['evaluation']).items()}, predictions


def select_single(result):
    # Original cohort stays primary; expansion is secondary, fixed before replay.
    scores = result['scores']
    return (scores['original']['single']['top4_hits'], scores['expanded']['single']['top4_hits'],
            scores['original']['single']['top1_hits'], scores['expanded']['single']['top1_hits'])


def select_future(result):
    scores = result['scores']['expanded']['future']
    return (scores['weighted_distinct_target_coverage'] or 0.,
            scores['steps'][1]['top4'] or 0.)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--baseline-run', type=Path, required=True)
    args = parser.parse_args()
    private_root = Path.home()/'Library/Application Support/SmartApp/training'
    if private_root.resolve() not in args.output.resolve().parents:
        raise ValueError('Personal experiment output must stay in private SmartApp training directory')
    _, dataset_hashes = verify_run(args.dataset)
    with gzip.open(args.dataset/'dataset-private.json.gz', 'rt') as f:
        data = json.load(f)
    _, baseline_hashes = verify_run(args.baseline_run)
    baseline = json.loads((args.baseline_run/'predictions-private.json').read_text())
    baseline_inputs = json.loads((args.baseline_run/'model-inputs-private.json').read_text())
    baseline_distributions = {}
    for row in data['evaluation']:
        pos = row['position']; q = data['observations'][pos]
        old, x = baseline[str(pos)], baseline_inputs[str(pos)]
        if old['sample_id'] != row['sample_id'] or any(x[k] != q[k] for k in ('origin','app','timestamp')):
            raise ValueError('Historical control does not match fixed evaluation inputs')
        prior = {label: 1./(20+i) for i,label in enumerate(old['raw_ranking'], 1)}
        baseline_distributions[pos] = {h:prior for h in range(4)}
    source_names = ('compare_semantic_trajectory.py', 'semantic_online.py', 'semantic_online_classifier.py', 'semantic_annotations.py')
    source_bytes = {name: Path(__file__).with_name(name).read_bytes() for name in source_names}
    classifier_configs = ([{'family': 'softmax', 'lr': lr, 'l2': 0., 'semantic': semantic}
                           for lr,semantic in itertools.product((.1, .3), (False, True))]
                          + [{'family': 'multinomial_nb', 'alpha': alpha, 'semantic': semantic}
                             for alpha,semantic in itertools.product((.1, 1.), (False, True))])
    os.umask(0o077)
    args.output.mkdir(mode=0o700, parents=True, exist_ok=False)
    plan = {'single_selection': 'original62 Top4; expanded Top4; original Top1; expanded Top1',
            'future_selection': 'expanded weighted distinct exact-target coverage; observed step1 Top4',
            'evaluation_single_labels': 'frozen original 5s action labels',
            'evaluation_future_labels': 'fixed resolved first3 independent arrivals within original30s; missing steps masked',
            'future_target_weights': STEP_WEIGHTS, 'future_head_mixes': MIXES,
            'semantic_strengths': [0., .25, 1., 4., 16.], 'recency_mixes': [0., .25, .5],
            'half_life_hours': [0., 6.], 'global_strength': 1.,
            'policies': ['original_cleaning'] + list(data['streams']),
            'fresh_holdout': False, 'predictions_use_evidence_grade': False,
            'candidate_or_inventory_output_restriction': False, 'classifier_configs': classifier_configs,
            'historical_control': str(args.baseline_run), 'baseline_rrf_mix': [0., .5], 'rrf_offset': 20}
    io.write_json(args.output/'plan.json', plan)
    results, best_single, best_future = [], None, None
    selected_predictions = {}
    similarity = SimilarityCache()

    def consume(config, distributions, allow_fusion=True):
        nonlocal best_single, best_future
        for base_mix in ([0., .5] if allow_fusion else [0.]):
            dist = distributions
            if base_mix:
                dist = {pos: {h: blend([dict((label, 1./(20+i)) for i,label in enumerate(rank(scores, len(scores)), 1)),
                                       baseline_distributions[pos][h]], [1-base_mix, base_mix])
                              for h,scores in heads.items()} for pos,heads in distributions.items()}
            for mix in MIXES:
                scores, predictions = evaluate(data, dist, mix)
                result = {'id': len(results), 'config': {**config, 'future_mix': mix, 'baseline_rrf_mix': base_mix}, 'scores': scores}
                results.append(result)
                if best_single is None or select_single(result) > select_single(best_single):
                    best_single = result; selected_predictions['single'] = predictions
                if best_future is None or select_future(result) > select_future(best_future):
                    best_future = result; selected_predictions['future'] = predictions

    consume({'family': 'historical_unrestricted_control', 'policy': 'original_cleaning'}, baseline_distributions, allow_fusion=False)

    for policy in plan['policies']:
        for half_life in plan['half_life_hours']:
            model = SemanticStatistics(semantic_strength=1., half_life_hours=half_life, similarity=similarity)
            components = replay(data, policy, model, components=True)
            for alpha, recency in itertools.product(plan['semantic_strengths'], plan['recency_mixes']):
                dist = {pos: {h:scores_from_components(parts, alpha, 1., recency) for h,parts in heads.items()}
                        for pos,heads in components.items()}
                consume({'family': 'semantic_statistics', 'policy': policy, 'semantic_strength': alpha,
                         'half_life_hours': half_life, 'global_strength': 1., 'recency_mix': recency}, dist)
            print(json.dumps({'finished_policy': policy, 'half_life_hours': half_life, 'configs': len(results)}), flush=True)
    from semantic_online_classifier import build
    for policy in plan['policies']:
        for config in classifier_configs:
            consume({**config, 'policy': policy}, replay(data, policy, build(config)))
        print(json.dumps({'finished_classifier_policy': policy, 'configs': len(results)}), flush=True)
    if verify_run(args.dataset)[1] != dataset_hashes:
        raise ValueError('Dataset changed during replay')
    if verify_run(args.baseline_run)[1] != baseline_hashes:
        raise ValueError('Historical control changed during replay')
    if any(Path(__file__).with_name(name).read_bytes() != data_bytes for name,data_bytes in source_bytes.items()):
        raise ValueError('Model source changed during replay')
    io.write_json(args.output/'results.json', results)
    io.write_json(args.output/'selection.json', {'single': best_single, 'future': best_future})
    io.write_json(args.output/'selected-predictions-private.json', selected_predictions)
    io.write_json(args.output/'run.json', {'complete': True, 'dataset': str(args.dataset),
        'dataset_hashes': dataset_hashes, 'test_previously_seen': True, 'config_count_including_future_mixes': len(results),
        'baseline_run': str(args.baseline_run), 'baseline_hashes': baseline_hashes,
        'collection_or_installation_changed': False})
    for name,data_bytes in source_bytes.items():
        (args.output/('source-'+name)).write_bytes(data_bytes)
    io.write_json(args.output/'artifact-hashes.json', {p.name: io.sha(p) for p in args.output.iterdir() if p.is_file()})
    print(json.dumps({'single': best_single, 'future': best_future}, ensure_ascii=False))


if __name__ == '__main__':
    main()
