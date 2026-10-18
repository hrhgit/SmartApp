#!/usr/bin/env python3
"""Train-only context clusters with causal, deduplicated next-arrival counts.

Requires numpy/scipy/scikit-learn. Personal features and artifacts stay in a new
private training run. This experiment does not modify the collector or live model.
"""
import argparse
from collections import Counter, defaultdict
from dataclasses import replace
import json
import math
import os
from pathlib import Path
import sys
from urllib.parse import urlsplit

import numpy as np
import sklearn
from sklearn.cluster import KMeans
from sklearn.feature_extraction import DictVectorizer
from threadpoolctl import threadpool_limits

import evaluate_context_knn as live_core
from diagnose_dwell_errors import frozen_core, label
from evaluate_evidence_model import (DENIED_BUNDLES, clean_input, independent_positions,
    semantic_label, resolve_label, choose, wilson_lower)
from summarize_tuned_history import verify_run

VERSION = 'context-cluster-frequency-v2'
FAMILIES = ('browser', 'finder', 'qspace', 'desktop_windows', 'running_apps')
MODES = ('ordered', 'frequency', 'presence', 'state_only')
MODE_NAMES = {'ordered': '最近访问保留顺序', 'frequency': '最近访问只计频次',
    'presence': '最近访问只计出现', 'state_only': '仅当前前台与后台'}
REFERENCE = 'https://scikit-learn.org/stable/auto_examples/text/plot_document_clustering.html'


def history_tokens(arrivals, mode):
    """Identical last-eight as-of arrivals; only their representation changes."""
    result = Counter()
    if mode == 'state_only':
        return {}
    for position, tokens in enumerate(reversed(arrivals[-8:])):
        for key, value in tokens.items():
            if mode == 'ordered':
                result[f'position:{position}:{key}'] += value
            elif mode == 'frequency':
                result[key] += value
            elif mode == 'presence':
                result[key] = max(result[key], value)
            else:
                raise ValueError('Unknown history mode')
    return dict(result)


def asof_histories(core, events, samples, items, excluded_sessions):
    """Rebuild semantic tokens at observation time, never using future metadata."""
    anchors = {s['anchorEventID']: s for s in samples}
    wanted = {q.anchor_id for q in items}
    catalog, features, result = {}, {}, {}
    session, last_context, segment = None, None, 0
    for e in events:
        data = e.get('data', {})
        if e['sessionID'] != session:
            session, last_context, catalog = e['sessionID'], None, {}
            segment += 1
        if e['kind'] in {'capture.gap', 'session.end'}:
            last_context = None
            segment += 1
        if e['kind'] == 'target.update':
            t = data['target']
            if core.admitted(t) and not core.synthetic(t):
                catalog[t['id']] = t
            else:
                catalog.pop(t['id'], None)
        if session in excluded_sessions or e['kind'] != 'context.change':
            continue
        t = catalog.get(data.get('targetID'))
        if not t:
            last_context = None
            continue
        bundle = e.get('app', {}).get('bundleID') or t.get('bundleID')
        resolution = data.get('reason') == 'resolution_update'
        parent = (data.get('resolvesContextEventID') or last_context) if resolution else e['id']
        if not resolution:
            last_context = e['id']
        tokens = core.target_tokens(t, bundle) if bundle not in DENIED_BUNDLES else {}
        features[e['id']] = (parent, not resolution, tokens, segment, core.key_for(t, session))
        if e['id'] not in wanted:
            continue
        rows = []
        for ref in anchors[e['id']]['recentContextEventIDs']:
            feature = features.get(ref)
            if feature is None or feature[3] != segment:
                continue
            arrival, independent, tok, _, key = feature
            if independent:
                rows.append([ref, tok, key])
            elif arrival:
                for row in reversed(rows):
                    if row[0] == arrival:
                        row[1], row[2] = tok, key
                        break
        result[e['id']] = rows[-8:]
    for q in items:
        assert tuple(r[2] for r in result[q.anchor_id]) == q.history_keys
    return [[row[1] for row in result[q.anchor_id] if row[1]] for q in items]


def add_unit(output, tokens, prefix, scale):
    length = math.sqrt(sum(v*v for v in tokens.values()))
    if length:
        for key, value in tokens.items():
            output[prefix + key] = scale * value / length


def background_idf(items, fit_positions):
    counts = Counter((family, key) for i in fit_positions
        for family, tokens in items[i].background.items() for key in tokens)
    return {pair: 1 + math.log((1 + len(fit_positions))/(1 + count)) for pair, count in counts.items()}


def feature_row(q, arrivals, mode, history_weight, foreground_fraction=.3, idf=None):
    result = {}
    state_weight = 1 - history_weight
    add_unit(result, q.foreground, 'foreground|', math.sqrt(foreground_fraction * state_weight))
    # Fixed source denominator: absent sources cannot turn remaining evidence into
    # a complete background. Status flags distinguish unobserved from observed empty.
    for family in FAMILIES:
        quality = q.source_quality.get(family, 0.)
        scale = math.sqrt((1 - foreground_fraction) * state_weight / len(FAMILIES))
        if family in q.background:
            tokens = {key: value * (idf.get((family, key), 1.) if idf is not None else 1.)
                for key, value in q.background[family].items()}
            add_unit(result, tokens, f'background|{family}|', scale * quality)
            if not q.background[family]:
                result[f'quality|{family}|observed_empty'] = scale * .15 * quality
            status = 'ready' if quality == 1 else 'partial'
        else:
            status = 'unobserved'
        result[f'quality|{family}|{status}'] = scale * .15
    add_unit(result, history_tokens(arrivals, mode), 'history|', math.sqrt(history_weight))
    return result


def fit_clusters(rows, fit_positions, n_clusters, seed=42):
    """Vocabulary and centers are fitted on earlier inputs alone; no labels."""
    vectorizer = DictVectorizer(dtype=np.float64)
    x_train = vectorizer.fit_transform([rows[i] for i in fit_positions])
    model = KMeans(n_clusters=n_clusters, n_init=20, random_state=seed, max_iter=300)
    with threadpool_limits(limits=1):
        model.fit(x_train)
        x = vectorizer.transform(rows)
        assignments = model.predict(x)
    centers = model.cluster_centers_
    # Do not renormalize after dropping test-only vocabulary. Its unexplained
    # feature mass must reduce similarity instead of creating a perfect match.
    lengths = np.sqrt(np.asarray(x.multiply(x).sum(axis=1)).ravel())
    full_lengths = np.array([math.sqrt(sum(v*v for v in row.values())) for row in rows])
    center_lengths = np.linalg.norm(centers, axis=1)
    dots = np.asarray(x.multiply(centers[assignments]).sum(axis=1)).ravel()
    similarity = np.divide(dots, full_lengths * center_lengths[assignments],
        out=np.zeros_like(dots), where=full_lengths * center_lengths[assignments] > 0)
    floors = {c: float(np.quantile([similarity[i] for i in fit_positions if assignments[i] == c], .1))
        for c in range(n_clusters) if any(assignments[i] == c for i in fit_positions)}
    return {'assignments': assignments.tolist(), 'similarity': similarity.tolist(),
        'similarity_floor': floors, 'vocabulary': vectorizer.get_feature_names_out().tolist(),
        'centers': centers.tolist(), 'inertia': float(model.inertia_),
        'known_feature_fraction': np.divide(lengths**2, full_lengths**2,
            out=np.zeros_like(lengths), where=full_lengths > 0).tolist()}


def label_counts(items, valid, ledger, assignments, query_index, max_age_hours=None, outcome_times=None):
    """One actual next arrival per cluster; only after its label has matured."""
    counts = Counter()
    seen = set()
    for j in sorted(valid):
        if j >= query_index:
            break
        if items[j].ready_index >= items[query_index].index or assignments[j] != assignments[query_index]:
            continue
        outcome = ledger[j]['target_event_id']
        if max_age_hours is not None:
            timestamp = outcome_times[outcome]
            if items[query_index].timestamp - timestamp > max_age_hours * 3600000:
                continue
        if outcome not in seen:
            seen.add(outcome)
            counts[semantic_label(items[j], 'window_class')] += 1
    return counts


def rank_frequency(core, q, counts):
    # Counts for unavailable historical answers stay in the denominator; they
    # cannot inflate confidence by disappearing at the candidate-filtering step.
    ranked = sorted(counts, key=lambda key: (-counts[key], key))
    mapped = {}
    for key in ranked:
        destination = resolve_label(key, q)
        if destination:
            mapped[destination] = mapped.get(destination, 0) + counts[key]
    raw = sorted(mapped, key=lambda key: (-mapped[key], key))
    shown = core.select_four(raw, q.candidates)
    total = sum(counts.values())
    winner = ranked[0] if ranked else None
    resolved = resolve_label(winner, q) if winner else None
    return {'predictions': shown, 'counts': dict(counts), 'total_unique_outcomes': total,
        'winner_support': counts[winner] if winner else 0,
        'agreement': counts[winner]/total if total else 0.,
        'winner': winner, 'winner_resolved': resolved,
        'winner_is_displayed_first': bool(shown and shown[0] == resolved),
        'unavailable_outcome_count': total - sum(mapped.values())}


def replay(core, items, valid, ledger, model, positions, max_age_hours=None, outcome_times=None):
    records = {}
    for i in positions:
        counts = label_counts(items, valid, ledger, model['assignments'], i, max_age_hours, outcome_times)
        row = rank_frequency(core, items[i], counts)
        cluster = model['assignments'][i]
        row.update(cluster=cluster, similarity=model['similarity'][i],
            prototype_in_range=model['similarity'][i] + 1e-12 >= model['similarity_floor'].get(cluster, 0.),
            background_observed=any(v >= .25 for v in items[i].source_quality.values()),
            known_feature_fraction=model.get('known_feature_fraction', [1.]*len(items))[i])
        records[i] = row
    return records


def accepted(row, gate):
    return (row['winner_is_displayed_first'] and row['winner_support'] >= gate['min_support']
        and row['agreement'] + 1e-12 >= gate['min_agreement']
        and row['prototype_in_range'] and row['background_observed'])


def summarize(core, items, positions, records, gate=None):
    keep = [i for i in positions if gate is None or accepted(records[i], gate)]
    result = core.metric([core.score_row(items[i], records[i]['predictions']) for i in keep])
    result.update(eligible_n=len(positions), coverage=len(keep)/len(positions) if positions else 0.,
        with_predictions=sum(bool(records[i]['predictions']) for i in keep),
        wilson_lower_descriptive=wilson_lower(result['top1_hits'], result['n']))
    return result


def foreground_class(q):
    # Stable semantic foreground class, without transient window identity.
    keys = [k for k in q.foreground if k.startswith(('target:', 'window-class:'))]
    return tuple(sorted(keys))


def control_model(items, foreground=False):
    classes = {}; assignments = []
    for q in items:
        key = foreground_class(q) if foreground else 'global'
        assignments.append(classes.setdefault(key, len(classes)))
    return {'assignments': assignments, 'similarity': [1.]*len(items),
        'similarity_floor': {c: 0. for c in classes.values()}}


def condition_presence(q, arrivals):
    result = set()
    for key in q.foreground:
        result.add('foreground|' + key)
    for family, tokens in q.background.items():
        for key in tokens:
            result.add(f'background|{family}|' + key)
    for tokens in arrivals:
        for key in tokens:
            result.add('recent|' + key)
    return result


def describe_clusters(core, items, histories, fit_positions, valid, ledger, model, test_start,
                      max_age_hours=None, outcome_times=None):
    conditions = {i: condition_presence(items[i], histories[i]) for i in fit_positions}
    global_counts = Counter(k for tokens in conditions.values() for k in tokens)
    rows = []
    for cluster in sorted(set(model['assignments'][i] for i in fit_positions)):
        members = [i for i in fit_positions if model['assignments'][i] == cluster]
        presence = Counter(k for i in members for k in conditions[i])
        common = [{'condition': k, 'count': n, 'fraction': n/len(members),
            'global_fraction': global_counts[k]/len(fit_positions),
            'lift': n/len(members)/(global_counts[k]/len(fit_positions))}
            for k, n in presence.items() if n/len(members) >= .8]
        common.sort(key=lambda r: (-r['lift'], -r['fraction'], r['condition']))
        # A hypothetical query at the actual test boundary, assigned to this class,
        # exposes only already mature next-arrival counts, never test outcomes.
        assignments = list(model['assignments']); assignments[test_start] = cluster
        counts = label_counts(items, valid, ledger, assignments, test_start, max_age_hours, outcome_times)
        total = sum(counts.values())
        rows.append({'cluster': cluster, 'train_inputs': len(members),
            'strict_train_inputs': sum(items[i].strict for i in members),
            'exact_intersection': sorted(k for k, n in presence.items() if n == len(members)),
            'common_conditions_80pct': common,
            'mature_outcomes_at_test_start': total,
            'next_destinations_at_test_start': [{'target': k, 'count': n, 'fraction': n/total}
                for k, n in counts.most_common()],
            'top1_share_at_test_start': max(counts.values(), default=0)/total if total else None,
            'entropy_bits_at_test_start': -sum(n/total*math.log2(n/total) for n in counts.values()) if total else None,
            'source_quality_distribution': {family: dict(Counter(str(items[i].source_quality.get(family, 0.))
                for i in members)) for family in FAMILIES}})
    return rows


def select_best(grid, mode):
    # Fixed objective before test inspection: next-step Top1, then Top4, then MRR.
    return max((r for r in grid if r['config']['mode'] == mode),
        key=lambda r: (r['validation']['top1'], r['validation']['top4'],
            r['validation']['mrr4'], -r['config']['k'], -r['config']['history_weight']))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-run', required=True, type=Path)
    parser.add_argument('--tuning-run', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--extended', action='store_true',
        help='Train-only background IDF; compare no foreground in clustering and 1h/6h/all-history counts')
    args = parser.parse_args()
    base, out = args.source_run.resolve(), args.output.resolve()
    if not out.is_relative_to(live_core.PRIVATE_ROOT.resolve()) or out == live_core.PRIVATE_ROOT.resolve():
        parser.error('Use a new private training directory')
    source, source_hashes = verify_run(base)
    tuning, tuning_hashes = verify_run(args.tuning_run)
    assert Path(tuning['source_run']) == base
    core = frozen_core(base)
    snapshot = Path(source['snapshot'])
    hashes, _ = core.verified_snapshot(snapshot)
    assert hashes == source['input_hashes']
    events = core.load_lines(snapshot/'events.jsonl')
    outcome_times = {e['id']: e['timestampMs'] for e in events}
    samples = core.load_lines(snapshot/'navigation-samples.jsonl')
    original, audit, arrivals, inventory, display = core.make_observations(events, samples, set(source['excluded_sessions']))
    histories = asof_histories(core, events, samples, original, set(source['excluded_sessions']))
    blocked = {core.key_for(e['data']['target'], e['sessionID']) for e in events if e['kind'] == 'target.update'
        and e['data']['target'].get('bundleID') in DENIED_BUNDLES}
    original = [clean_input(q, blocked) for q in original]
    ledgers = json.loads((base/'labels-private.json').read_text())
    prior_predictions = json.loads((args.tuning_run/'predictions-private.json').read_text())
    n = len(original); train_end = int(n*.6); test_start = int(n*.8)
    fit_positions = independent_positions(original, [i for i in range(train_end)
        if original[i].current_app not in DENIED_BUNDLES])
    cluster_configs = [{'mode': mode, 'history_weight': w, 'k': k, 'foreground_fraction': foreground,
        'background_idf': args.extended}
        for mode in MODES for w in ([0.] if mode == 'state_only' else [.2, .5])
        for k in [4, 8, 12, 20, 32] for foreground in ([.3, 0.] if args.extended else [.3])]
    age_windows = [None, 6., 1.] if args.extended else [None]
    configs = [{'model_id': i, **config, 'max_age_hours': age}
        for i, config in enumerate(cluster_configs) for age in age_windows]
    gates = [{'min_support': support, 'min_agreement': agreement}
        for support in [3, 5] for agreement in [.7, .85, 1.]]
    os.umask(0o077)
    out.mkdir(parents=True, mode=0o700, exist_ok=False)
    core.write_json(out/'search-plan.json', {'version': VERSION, 'configs': configs, 'gates': gates,
        'selection': 'validation Top1, Top4, MRR4, then fewer clusters and lower history weight',
        'selective_target_top1': .8, 'selective_min_validation': 15,
        'train_end_original_position': train_end, 'test_start_original_position': test_start,
        'fit_positions': fit_positions, 'label_counting': 'one actual mature outcome per cluster; uniform frequency',
        'foreground_fraction_within_state': [.3, 0.] if args.extended else [.3],
        'background_idf': args.extended, 'count_age_windows_hours': age_windows,
        'history_length': 8, 'random_state': 42, 'n_init': 20,
        'prediction_spacing_ms': 30000, 'fit_spacing_ms': 30000,
        'window_representation': 'stable app window class; uniquely resolvable current candidates only',
        'test_previously_seen': True, 'no_target_based_acceptance': True, 'sources': [REFERENCE]})
    models = {}; feature_cache = {}
    idf = background_idf(original, fit_positions) if args.extended else None
    for model_id, config in enumerate(cluster_configs):
        feature_key = (config['mode'], config['history_weight'], config['foreground_fraction'])
        if feature_key not in feature_cache:
            feature_cache[feature_key] = [feature_row(q, h, *feature_key, idf=idf) for q, h in zip(original, histories)]
        models[model_id] = fit_clusters(feature_cache[feature_key], fit_positions, config['k'])
        if (model_id+1) % 10 == 0 or model_id+1 == len(cluster_configs):
            print(json.dumps({'stage': 'fit_clusters', 'done': model_id+1, 'total': len(cluster_configs),
                'train_inputs': len(fit_positions), 'features': len(models[model_id]['vocabulary'])}), flush=True)
    results, all_predictions, descriptions = {}, {}, {}
    names = {e['app']['bundleID']: e['app']['name'] for e in events if e.get('app', {}).get('name')}
    metadata = dict(display)
    for threshold, ledger in ledgers.items():
        items = list(original); valid = set()
        for r in ledger:
            i = r['position']; q = items[i]
            assert q.sample_id == r['sample_id'] and q.index == r['input_through_event_index']
            if r['status'] != 'stable_target':
                continue
            t = r['target']
            if q.current_app in DENIED_BUNDLES or t.get('bundleID') in DENIED_BUNDLES:
                continue
            valid.add(i)
            items[i] = replace(q, truth=core.key_for(t, q.session), truth_meta={**t, '_bundle': t.get('bundleID')})
            metadata[items[i].truth] = items[i].truth_meta
        val_full = [i for i in range(train_end, test_start) if i in valid and items[i].ready_index < items[test_start].index]
        test_full = [i for i in range(test_start, n) if i in valid]
        validation = independent_positions(items, val_full)
        test = independent_positions(items, test_full)
        grid, selective_grid = [], []
        val_cache = {}
        for config in configs:
            model_id = config['model_id']
            records = replay(core, items, valid, ledger, models[model_id], validation, config['max_age_hours'], outcome_times)
            val_cache[model_id, config['max_age_hours']] = records
            grid.append({'model_id': model_id, 'config': config,
                'validation': summarize(core, items, validation, records)})
            for gate in gates:
                selective_grid.append({'model_id': model_id, 'config': {**config, **gate},
                    'validation': summarize(core, items, validation, records, gate)})
        selected = {mode: select_best(grid, mode) for mode in MODES}
        selective, status = choose(selective_grid)
        baseline_selected = {}
        for name, foreground in [('global_frequency', False), ('foreground_frequency', True)]:
            baseline_model = control_model(items, foreground)
            candidates = []
            for age in age_windows:
                records = replay(core, items, valid, ledger, baseline_model, validation, age, outcome_times)
                candidates.append({'max_age_hours': age, 'validation': summarize(core, items, validation, records)})
            baseline_selected[name] = max(candidates, key=lambda r:
                (r['validation']['top1'], r['validation']['top4'], r['validation']['mrr4']))
        # Persist all selections before producing test metrics.
        core.write_json(out/f'selection-{threshold}s.json', {'by_mode': selected,
            'selective': selective, 'selective_status': status, 'baselines': baseline_selected})
        core.write_json(out/f'validation-grid-{threshold}s.json', grid)
        core.write_json(out/f'selective-grid-{threshold}s.json', selective_grid)
        comparisons, prediction_rows, cluster_rows = {}, {}, {}
        for mode, selection in {**selected, 'selective': selective}.items():
            model_id = selection['model_id']; model = models[model_id]
            age = selection['config']['max_age_hours']
            records = {**val_cache[model_id, age], **replay(core, items, valid, ledger, model, test_full, age, outcome_times)}
            gate = selection['config'] if mode == 'selective' else None
            comparisons[mode] = {'config': selection['config'], 'validation': selection['validation'],
                'test': summarize(core, items, test, records, gate),
                'strict_test': summarize(core, items, [i for i in test if items[i].strict], records, gate),
                'all_test_cutoffs_secondary': summarize(core, items, test_full, records, gate),
                'test_missing_predictions': sum(not records[i]['predictions'] for i in test)}
            prediction_rows[mode] = [{'position': i, 'subset': 'validation' if i < test_start else 'test',
                'primary_cutoff': i in validation or i in test, 'sample_id': items[i].sample_id,
                'time_utc': core.utc(items[i].timestamp), 'accepted': gate is None or accepted(records[i], gate),
                'truth': items[i].truth, 'truth_meta': items[i].truth_meta,
                'prediction_meta': [items[i].candidates[k] for k in records[i]['predictions']],
                'score': core.score_row(items[i], records[i]['predictions']), **records[i]}
                for i in validation + test_full]
            cluster_rows[mode] = describe_clusters(core, items, histories, fit_positions, valid, ledger, model, test_start,
                age, outcome_times)
        baselines = {}
        for name, foreground in [('global_frequency', False), ('foreground_frequency', True)]:
            age = baseline_selected[name]['max_age_hours']
            records = replay(core, items, valid, ledger, control_model(items, foreground), test_full, age, outcome_times)
            baselines[name] = {**baseline_selected[name], 'test': summarize(core, items, test, records),
                'strict_test': summarize(core, items, [i for i in test if items[i].strict], records),
                'all_test_cutoffs_secondary': summarize(core, items, test_full, records)}
        for name in ['recency', 'selected']:
            saved = {r['position']: r for r in prior_predictions[threshold][name]}
            def baseline_metric(positions):
                return core.metric([core.score_row(items[i], [k for k in saved[i]['predictions'] if k in items[i].candidates])
                    for i in positions])
            baselines['old_knn' if name == 'selected' else name] = {'test': baseline_metric(test),
                'strict_test': baseline_metric([i for i in test if items[i].strict]),
                'all_test_cutoffs_secondary': baseline_metric(test_full)}
        accepted_test = {r['position'] for r in prediction_rows['selective'] if r['subset'] == 'test'
            and r['primary_cutoff'] and r['accepted']}
        recency_saved = {r['position']: r for r in prior_predictions[threshold]['recency']}
        results[threshold] = {'train_inputs': len(fit_positions), 'validation_n': len(validation), 'test_n': len(test),
            'models': comparisons, 'baselines': baselines, 'selective_status': status,
            'selective_same_cutoffs_recency': core.metric([core.score_row(items[i],
                [k for k in recency_saved[i]['predictions'] if k in items[i].candidates]) for i in sorted(accepted_test)])}
        all_predictions[threshold] = prediction_rows; descriptions[threshold] = cluster_rows
        core.write_json(out/f'result-{threshold}s.json', results[threshold])
        print(json.dumps({'stage': 'threshold_complete', 'threshold': threshold,
            'models': {m: {'config': r['config'], 'test': r['test']} for m, r in comparisons.items()},
            'baselines': {m: r['test'] for m, r in baselines.items()}, 'selective_status': status}), flush=True)
    core.write_json(out/'metrics.json', results)
    core.write_json(out/'predictions-private.json', all_predictions)
    core.write_json(out/'cluster-profiles-private.json', descriptions)
    selected_ids = sorted({r['model_id'] for threshold in ledgers for r in
        [*json.loads((out/f'selection-{threshold}s.json').read_text())['by_mode'].values(),
         json.loads((out/f'selection-{threshold}s.json').read_text())['selective']]})
    core.write_json(out/'fitted-models-private.json', {str(i): {'config': cluster_configs[i], **models[i]} for i in selected_ids})
    core.write_json(out/'background-idf-private.json', [{'family': family, 'token': token, 'weight': weight}
        for (family, token), weight in (idf or {}).items()])
    core.write_json(out/'target-catalog-private.json', metadata)
    core.write_json(out/'app-names-private.json', names)
    # Independent arrivals and inventory observations remain distinct, even in a
    # cluster report. Rankings are presentation-only and do not feed the models.
    core.write_json(out/'destination-rankings-private.json', {
        category: {kind: [{'target': key, 'count': count, 'metadata': metadata.get(key)}
            for key, count in counts.most_common() if key not in blocked]
            for kind, counts in collection.items()}
        for category, collection in [('independent_arrivals', arrivals), ('inventory_observations', inventory)]})
    render_report(out, results, descriptions, metadata, names, core, args.extended)
    source_files = [Path(__file__), Path(__file__).with_name('evaluate_evidence_model.py'),
        Path(__file__).with_name('diagnose_dwell_errors.py'), Path(__file__).with_name('summarize_tuned_history.py'),
        Path(__file__).with_name('evaluate_context_knn.py')]
    for p in source_files:
        (out/('source-' + p.name)).write_text(p.read_text())
    core.write_json(out/'run.json', {'version': VERSION, 'source_run': str(base), 'source_artifact_hashes': source_hashes,
        'tuning_run': str(args.tuning_run.resolve()), 'tuning_artifact_hashes': tuning_hashes, 'input_hashes': hashes,
        'raw_data_modified': False, 'live_model_updated': False, 'test_previously_seen': True,
        'python': sys.version, 'numpy': np.__version__, 'sklearn': sklearn.__version__, 'audit': audit})
    core.write_json(out/'artifact-hashes.json', {p.name: core.sha(p) for p in out.iterdir() if p.is_file()})
    print(json.dumps({'output': str(out), 'complete': True}), flush=True)


def readable_condition(condition, metadata, names):
    parts = condition.split('|'); key = parts[-1]
    prefix = {'foreground': '前台', 'background': '后台', 'recent': '近期访问'}[parts[0]]
    if key.startswith('target:'):
        return prefix + ' ' + label(metadata.get(key[7:], {'kind': '未知目标'}), names)
    if key.startswith('app:'):
        return prefix + '应用 ' + names.get(key[4:], key[4:])
    if key.startswith('window-class:'):
        return prefix + '窗口 ' + names.get(key[13:], key[13:])
    if key.startswith('origin:'):
        return prefix + '站点 ' + str(urlsplit(key[7:]).hostname)
    return None


def readable_destination(key, metadata, names):
    if key.startswith('window_class:'):
        bundle = key.removeprefix('window_class:')
        return names.get(bundle, bundle) + ' 窗口类别'
    return label(metadata.get(key, {'kind': '未知目标'}), names)


def render_report(out, results, descriptions, metadata, names, core, extended=False):
    lines = ['# 上下文聚类与类内下一步频率', '',
        '算法：对前台、允许范围内的后台库存、来源质量及最近 8 次到达进行加权稀疏编码，再用 KMeans 聚类。每个特征组按 L2 归一化；后台来源固定分母、保留质量折扣和缺失标记。类别内按已经成熟的独立后续到达频次排序，不用近期访问填补空缺。',
        '顺序、频次与是否出现使用相同的最近 8 次已观察到达。频次版把重复访问相加，不保留访问位置；出现版只保留集合；顺序版保留 8 个位置。第四组不含近期路径，基础版只用前台与后台状态。历史权重搜索 20%/50%。',
        ('扩展版用仅由早期输入拟合的 IDF 降低常驻后台条件的相对影响；比较状态部分前台/后台 30%/70% 与 0%/100%；类内后续频率比较全历史、最近 6 小时和最近 1 小时。频率窗口及类别数等仍只按验证段选择。全局频次和相同前台频次对照也在相同时间窗中选参。'
            if extended else '基础版状态部分前台/后台权重为 30%/70%，使用累计类内频次。'),
        '原截点按时间前 60% 学习、中间 20% 验证、后 20% 测试。词表、中心、类内共有背景条件只由早期输入拟合；验证段选择类别数和历史权重，选参结果落盘后才计算测试指标。标签满原 30 秒观察窗后才允许进入类内计数。每类每个真实后续到达只计一次，不同类的支持数不可相加为全局独立到达数。',
        '主评估每个会话按预测时间间隔至少 30 秒取样。窗口经验按稳定应用窗口类别聚合，只能唯一对应当前候选窗口时才推荐；最终准确率仍按具体目标核对。所有条件均来自预测截点，未用本次真实结果挑选“有证据”样本。',
        '这批后期数据在先前实验中已查看；下列结果属于探索性比较，尚不是新数据上的独立验证。', '',
        '| 清洗阈值 | 模型 | 类别数 | 历史权重 | 状态中前台权重 | 计数时窗/小时 | 验证 Top1 | 后期 Top1 | 后期 Top4 | 后期截点数 |',
        '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for threshold, result in results.items():
        for mode in MODES:
            row = result['models'][mode]; cfg = row['config']; metric = row['test']
            lines.append(f"| {threshold} 秒 | {MODE_NAMES[mode]} | {cfg['k']} | {cfg['history_weight']} | {cfg['foreground_fraction']} | {cfg['max_age_hours'] or '全历史'} | {core.percent(row['validation']['top1'])} | {core.percent(metric['top1'])} | {core.percent(metric['top4'])} | {metric['n']} |")
        for name, title in [('global_frequency', '全局下一步频次'), ('foreground_frequency', '相同前台下一步频次'),
                            ('old_knn', '之前调参近邻'), ('recency', '近期访问排序')]:
            metric = result['baselines'][name]['test']
            baseline = result['baselines'][name]
            age = baseline.get('max_age_hours') or '全历史'
            lines.append(f"| {threshold} 秒 | {title} | — | — | — | {age if 'max_age_hours' in baseline else '—'} | — | {core.percent(metric['top1'])} | {core.percent(metric['top4'])} | {metric['n']} |")
    lines += ['', '## 有证据时才给出预测', '',
        '门槛：第一名可唯一推荐、至少 3/5 个去重支持、历史占比至少 70%/85%/100%、与原型相似度不低于该训练类别第 10 百分位、存在已观察后台来源。验证要求至少接受 15 个截点且 Top1 达到 80%，再尽量提高覆盖。未达标时仅报告探索候选，不声称高概率成立。', '',
        '| 阈值 | 验证状态 | 验证接受/总数 | 测试接受/总数 | 测试 Top1 | 同截点近期访问 Top1 |',
        '|---|---|---:|---:|---:|---:|']
    for threshold, result in results.items():
        row = result['models']['selective']; v = row['validation']; t = row['test']
        lines.append(f"| {threshold} 秒 | {result['selective_status']} | {v['n']}/{v['eligible_n']} | {t['n']}/{t['eligible_n']} | {core.percent(t['top1'])} | {core.percent(result['selective_same_cutoffs_recency']['top1'])} |")
    lines += ['', '## 频次版类别的共有条件', '',
        '下面展示 5 秒清洗、按验证选择的频次版类别。条件比例按早期拟合输入计算；后续行为比例仅使用测试开始之前已经成熟的到达，未使用测试答案解释类别。100% 交集和至少 80% 的共同条件分别保存在私有 JSON；共有后台只表示实际观察到的库存，不能推断意图。条件优先展示相对整体更有区分度的项。', '',
        '| 类别 | 早期输入数 | 有区分度的共同条件 | 测试前后续行为频次 |', '|---|---:|---|---|']
    for row in descriptions.get('5', next(iter(descriptions.values())))['frequency']:
        conditions = []; seen = set()
        for r in row['common_conditions_80pct']:
            text = readable_condition(r['condition'], metadata, names)
            if text and text not in seen:
                conditions.append(f"{text}（{r['fraction']:.0%}）"); seen.add(text)
            if len(conditions) == 4:
                break
        targets = '；'.join(f"{readable_destination(r['target'], metadata, names)} {r['count']}/{row['mature_outcomes_at_test_start']}"
            for r in row['next_destinations_at_test_start'][:3])
        lines.append(f"| {row['cluster']} | {row['train_inputs']} | {'；'.join(conditions)} | {targets} |")
    lines += ['', '## 结果边界与复查', '',
        '严格来源覆盖子集、全部原始截点的次要结果、逐个类别的交集与分歧、逐预测证据、独立到达排行与后台库存排行均在本 run。候选覆盖仍受当时已观察目的地与可用窗口限制；类内占比是历史经验频率，不是经过校准的成功概率。',
        f'实现参考：[scikit-learn 的稀疏向量 KMeans 示例]({REFERENCE})。', '']
    (out/'report-private.md').write_text('\n'.join(lines))


if __name__ == '__main__':
    main()
