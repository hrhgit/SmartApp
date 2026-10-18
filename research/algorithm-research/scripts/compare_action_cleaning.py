#!/usr/bin/env python3
"""Factorial, causal comparison of reopening semantics and training cleaning.

The 62 original supported cutoffs and their five-second observed answers stay
fixed. Generic windows receive application-level scoring explicitly; pages and
paths still require their concrete identities. This is development selection.
"""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
import copy
import gzip
import itertools
import json
import multiprocessing
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace

import compare_online_models as comparison
import evaluate_context_knn as io
from fused_online_models import FusedModel
import online_model_suite as online
import stat_model_suite as stats
from refine_online_winner import components as refinement_components
from search_online_fusions import fuse
from restrict_online_evidence import EvidenceRestrictedModel, allowed_at_cutoffs, restricted_predictions
from summarize_tuned_history import verify_run
from diagnose_dwell_errors import frozen_core

VERSION = 'reopenable-action-cleaning-v1'
PRIOR = io.PRIVATE_ROOT / 'online-comparison-v1-20260923-refinement-01'
DATA = None


def config_signature(config):
    return config['family'], json.dumps(config['parameters'], sort_keys=True)


def configurations():
    result, by_signature = [], {}
    for config in stats.configurations() + online.configurations() + sum(refinement_components(), []):
        signature = config_signature(config)
        if signature in by_signature:
            continue
        config = copy.deepcopy(config)
        config['feature_scope'] = (
            'current_plus_clock' if config['family'] == 'clock_kernel' else
            'current_plus_candidate_recency' if config['family'] in
            ('candidate_recency', 'count_candidate_recency') else 'current_only')
        result.append(config)
        by_signature[signature] = config
    return result, by_signature


def build(config):
    if config['family'] == 'same_origin_evidence_gate':
        return EvidenceRestrictedModel(build(config['parameters']['base']))
    if config['family'] == 'rank_fusion_ensemble':
        p = config['parameters']
        return FusedModel(build(p['left']), build(p['right']), p['right_weight'], p['rank_offset'])
    return (stats if config['id'].startswith('stat_') else online).build(config)


def key(result, group='supported'):
    return comparison.selection_key(result, group)


def dump_gzip(path, value):
    with gzip.open(path, 'wt', encoding='utf-8', compresslevel=6) as f:
        json.dump(value, f, ensure_ascii=False, separators=(',', ':'))


def read_gzip(path):
    with gzip.open(path, 'rt', encoding='utf-8') as f:
        return json.load(f)


def score(cohorts, predictions, items):
    import reopenable_actions as actions
    return {threshold: actions.score_groups(rows, predictions, items)
            for threshold, rows in cohorts.items()}


def fusion_config(left, right, weight, offset, prefix='fusion'):
    return {'id': f"{prefix}-{left['id']}--{right['id']}--{offset}-{weight:g}",
            'family': 'rank_fusion_ensemble', 'feature_scope': 'current_only',
            'parameters': {'left': left, 'right': right,
                           'right_weight': weight, 'rank_offset': offset}}


def gate_config(base):
    return {'id': 'gate-' + base['id'], 'family': 'same_origin_evidence_gate',
            'feature_scope': base['feature_scope'], 'parameters': {'base': base}}


def initialize_worker(path):
    """Fresh processes avoid macOS Objective-C runtime state inherited by fork."""
    global DATA
    payload = read_gzip(Path(path))
    core = frozen_core(Path(payload['cleaning_run']))
    items = [None] * payload['item_count']
    for position, query in payload['query_items'].items():
        items[int(position)] = SimpleNamespace(**query)
    _, signatures = configurations()
    DATA = (core, items, payload['cohorts'], payload['streams'],
            {int(i): q for i, q in payload['inputs'].items()},
            {int(i): q for i, q in payload['mappings'].items()},
            payload['configs'], signatures, Path(payload['output']))


def clean_worker(cleaning):
    core, items, cohorts, streams, inputs, mappings, configs, signatures, out = DATA
    predictions, base_results, fusion_results = {}, [], []
    started = time.monotonic()
    for index, config in enumerate(configs):
        pred, timing = comparison.replay(core, build(config), items, streams[cleaning], inputs, mappings)
        predictions[config['id']] = pred
        base_results.append({'training_cleaning': cleaning, 'config': config,
                             'scores': score(cohorts, pred, items), 'timing': timing})
        if (index + 1) % 50 == 0 or index + 1 == len(configs):
            print(json.dumps({'stage': 'base', 'cleaning': cleaning,
                              'done': index + 1, 'total': len(configs)}), flush=True)
    # Select representatives separately for each training cleaning. All such
    # selection uses this already-seen development cohort, never a fresh test.
    leaders = {}
    for result in sorted(base_results, key=key):
        if result['config']['feature_scope'] == 'current_only':
            leaders.setdefault(result['config']['family'], result)
    stat_leaders = [r for r in leaders.values() if r['config']['id'].startswith('stat_')][:4]
    ml_leaders = [r for r in leaders.values() if not r['config']['id'].startswith('stat_')][:4]
    specs = []
    for a, b in itertools.combinations(stat_leaders + ml_leaders, 2):
        for offset, weight in itertools.product((0, 10), (.1, .25, .5, .75, .9)):
            specs.append(fusion_config(a['config'], b['config'], weight, offset))
    left, right = refinement_components()
    for a, b, weight in itertools.product(left, right, (.25, .4, .5, .6, .75)):
        specs.append(fusion_config(signatures[config_signature(a)],
                                  signatures[config_signature(b)], weight, 0, 'local-fusion'))
    seen = set()
    for config in specs:
        signature = config_signature(config)
        if signature in seen:
            continue
        seen.add(signature)
        p = config['parameters']
        pred = fuse(core, predictions[p['left']['id']], predictions[p['right']['id']],
                    p['right_weight'], p['rank_offset'], items)
        fusion_results.append({'training_cleaning': cleaning, 'config': config,
                               'scores': score(cohorts, pred, items)})
    # Revisit the earlier evidence-gate architecture on every base scorer and
    # the ten best fused scorers for this cleaning; it never changes membership.
    allowed = allowed_at_cutoffs(items, streams[cleaning], inputs)
    gate_results = []
    gate_sources = base_results + sorted(fusion_results, key=key)[:10]
    for result in gate_sources:
        config = result['config']
        if config['family'] == 'rank_fusion_ensemble':
            p = config['parameters']
            pred = fuse(core, predictions[p['left']['id']], predictions[p['right']['id']],
                        p['right_weight'], p['rank_offset'], items)
        else:
            pred = predictions[config['id']]
        gated = restricted_predictions(core, pred, allowed, items)
        gate_results.append({'training_cleaning': cleaning, 'config': gate_config(config),
                             'scores': score(cohorts, gated, items)})
    dump_gzip(out / f'predictions-{cleaning}-private.json.gz', predictions)
    dump_gzip(out / f'metrics-{cleaning}.json.gz', {'base': base_results,
        'fusions': fusion_results, 'gates': gate_results})
    best = min(base_results + fusion_results + gate_results, key=key)
    print(json.dumps({'stage': 'cleaning_complete', 'cleaning': cleaning,
                      'base_replays': len(base_results), 'fusions': len(fusion_results), 'gates': len(gate_results),
                      'best': best['scores']['5']['supported'],
                      'elapsed_seconds': round(time.monotonic()-started, 1)}), flush=True)
    return {'cleaning': cleaning, 'base_replays': len(base_results), 'fusions': len(fusion_results)}


def cached_prediction(result, out, core, items, stream=None, inputs=None):
    predictions = read_gzip(out / f"predictions-{result['training_cleaning']}-private.json.gz")
    def load(config):
        return {int(i): p for i, p in predictions[config['id']].items()}
    def resolve(config):
        if config['family'] == 'same_origin_evidence_gate':
            assert stream is not None and inputs is not None
            return restricted_predictions(core, resolve(config['parameters']['base']),
                allowed_at_cutoffs(items, stream, inputs), items)
        if config['family'] != 'rank_fusion_ensemble':
            return load(config)
        p = config['parameters']
        return fuse(core, load(p['left']), load(p['right']), p['right_weight'], p['rank_offset'], items)
    return resolve(result['config'])


def truth_key(row, items):
    # Future outcome metadata is used only here for evaluation, never to
    # construct candidates or current-interface model inputs.
    observed = row['observed_outcome']
    truth = observed['key']
    if truth.startswith('window@'):
        import reopenable_actions as actions
        return actions.truth_action_key(row, items)
    return truth


def normalize_old_predictions(old_predictions, original):
    """Rescore the old displayed slots, never fill from rank five or later."""
    rescored = {}
    for i, pred in old_predictions.items():
        def normalized(keys):
            result = []
            for destination in keys:
                meta = original[i].candidates.get(destination, {})
                bundle = meta.get('_bundle') or meta.get('bundleID')
                if meta.get('kind') == 'window' and bundle:
                    destination = 'app:' + bundle
                if destination not in result:
                    result.append(destination)
            return result
        rescored[i] = {**pred, 'predictions': normalized(pred['predictions']),
                       'raw_ranking': normalized(pred['raw_ranking'])}
    return rescored


def diagnose(rows, pred, items, streams, inputs):
    errors = []
    for row in rows:
        if not row['diagnosis_after_outcome']['eligible']:
            continue
        position = row['position']
        target = truth_key(row, items)
        ranking = pred[position]['raw_ranking']
        shown = pred[position]['predictions']
        if target in shown:
            continue
        query = items[position]
        origin = inputs[position]['origin']
        mature = [r for r in streams if r['ready_index'] < query.index
                  and r['record']['origin'] == origin]
        support = [r for r in mature if r['record']['label'] == target]
        rank = ranking.index(target)+1 if target in ranking else None
        reason = ('candidate_unavailable' if target not in query.candidates else
                  'cleaning_removed_same_origin_support' if not support else
                  'unscored' if rank is None else
                  'hierarchy_suppression' if rank <= 4 else 'rank_below_four')
        errors.append({'position': position, 'sample_id': row['sample_id'],
                       'origin': row['display_only']['current_description'],
                       'actual': row['observed_outcome']['description'],
                       'truth_action': target, 'raw_rank': rank, 'reason': reason,
                       'grade': row['diagnosis_after_outcome']['grade'],
                       'fixed5_support': row['diagnosis_after_outcome']['support_count'],
                       'training_support': len(support), 'training_origin_total': len(mature),
                       'predictions': shown})
    return errors


def source_snapshot(out):
    names = ('compare_action_cleaning.py', 'reopenable_actions.py', 'explore_training_cleaning.py',
             'compare_online_models.py', 'stat_model_suite.py', 'online_model_suite.py',
             'fused_online_models.py', 'refine_online_winner.py', 'search_online_fusions.py',
             'restrict_online_evidence.py',
             'evaluate_context_knn.py', 'evaluate_evidence_model.py', 'evaluate_ordered_current.py',
             'evaluate_dwell_cleaning.py', 'describe_dwell.py', 'diagnose_dwell_errors.py',
             'summarize_tuned_history.py')
    hashes = {}
    for name in names:
        path = Path(__file__).with_name(name)
        (out / ('source-' + name)).write_bytes(path.read_bytes())
        hashes[name] = io.sha(path)
    io.write_json(out / 'source-hashes.json', hashes)
    return hashes


def main():
    global DATA
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=3, choices=range(1, 5))
    parser.add_argument('--cleanings', nargs='*')
    parser.add_argument('--config-ids', nargs='*')
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    out = args.output.expanduser().resolve()
    if not out.is_relative_to(io.PRIVATE_ROOT.resolve()) or out == io.PRIVATE_ROOT.resolve():
        parser.error('Output must be a new private training run')
    import explore_training_cleaning as cleaning_module
    import reopenable_actions as actions
    configs, signatures = configurations()
    if args.config_ids:
        configs = [c for c in configs if c['id'] in args.config_ids]
    os.umask(0o077)
    out.mkdir(parents=True, mode=0o700, exist_ok=False)
    plan = {'version': VERSION, 'current_context_length': 1,
            'primary_cohort': 'unchanged frozen5s 62 supported IDs and recorded destinations',
            'main_metric': 'generic window or app requires owning application; concrete URL/path requires exact destination',
            'secondary_metrics': ['original exact target', 'application layer', 'native3s sensitivity'],
            'selection': 'development full62 Top4, Top1, MRR4, config ID, cleaning ID',
            'early_control': 'base configurations selected on early24 only, report late38; previously seen data',
            'cleaning': 'independent training factor, fixed evaluation answers',
            'candidate_policy': 'as-of known admitted launchable apps and valid reopenable historical locators',
            'maturity': 'original30s ready_index strictly before prediction cutoff',
            'fusions': {'family_leaders': 'top4 current-only stats and top4 current-only online',
                        'weights': [.1, .25, .5, .75, .9], 'rank_offsets': [0, 10],
                        'local_components': refinement_components(),
                        'local_weights': [.25, .4, .5, .6, .75]},
            'same_origin_evidence_gate': 'all base scorers plus top10 fusions per cleaning; no scoring-membership change',
            'base_configurations': configs, 'smoke': args.smoke,
            'test_previously_seen': True, 'all_possible_algorithms_exhausted': False,
            'raw_data_modified': False, 'live_app_updated': False,
            'runtime': {'python': sys.version, 'executable': sys.executable}}
    io.write_json(out / 'plan.json', plan)
    source_hashes = source_snapshot(out)
    started = time.monotonic()
    core, original, cohorts, old_streams, old_inputs, old_mappings, provenance = comparison.load_dataset(comparison.DEFAULT_EXPANSION)
    streams, cleaning_audit = cleaning_module.build_cleaning_streams(core, original, provenance)
    if args.cleanings:
        streams = {name: streams[name] for name in args.cleanings}
    io.write_json(out / 'cleaning-plan.json', {'cleanings': list(streams),
                  'base_configs': len(configs), 'base_replays': len(configs)*len(streams)})
    dump_gzip(out / 'cleaning-audit-private.json.gz', cleaning_audit)
    io.write_json(out / 'cleaning-summary.json', {key: value for key, value in cleaning_audit.items()
                                               if key != 'ledger_private'})
    items, inputs, mappings, streams, action_audit = actions.adapt_dataset(core, original, cohorts, streams, provenance)
    for i, query in inputs.items():
        query['candidate_key_order'] = mappings[i]
    io.write_json(out / 'action-audit-private.json', action_audit)
    io.write_json(out / 'input-audit.json', provenance)
    io.write_json(out / 'dependencies.json', online.dependency_metadata())
    io.write_json(out / 'benchmark-membership.json', json.loads((comparison.DEFAULT_EXPANSION / 'benchmark-membership.json').read_text()))
    dump_gzip(out / 'training-streams-private.json.gz', streams)
    io.write_json(out / 'query-inputs-private.json', inputs)
    if args.smoke:
        records = []
        for name, stream in streams.items():
            for config in configs:
                pred, timing = comparison.replay(core, build(config), items, stream, inputs, mappings)
                records.append({'config': config, 'training_cleaning': name,
                                'scores': score(cohorts, pred, items), 'timing': timing})
        io.write_json(out / 'metrics.json', records)
        io.write_json(out / 'run.json', {**plan, **provenance, 'complete': True})
        io.write_json(out / 'artifact-hashes.json', {p.name: io.sha(p) for p in out.iterdir() if p.is_file()})
        print(json.dumps({'smoke_complete': True, 'metrics': [
            {'cleaning': r['training_cleaning'], 'model': r['config']['id'], 'main': r['scores']['5']['supported']}
            for r in records]}), flush=True)
        return
    DATA = core, items, cohorts, streams, inputs, mappings, configs, signatures, out
    worker_input = out / 'worker-input-private.json.gz'
    dump_gzip(worker_input, {'cleaning_run': provenance['cleaning_run'],
        'item_count': len(items), 'query_items': {str(i): vars(items[i]) for i in inputs},
        'cohorts': cohorts, 'streams': streams, 'inputs': inputs, 'mappings': mappings,
        'configs': configs, 'output': str(out)})
    print(json.dumps({'stage': 'start_sweep', 'cleanings': len(streams),
                      'configs': len(configs), 'workers': args.workers}), flush=True)
    # Each spawned worker reconstructs only prepared cutoff observations from
    # private JSON. It receives no pickled code or process-global macOS state.
    with ProcessPoolExecutor(max_workers=args.workers,
                             mp_context=multiprocessing.get_context('spawn'),
                             initializer=initialize_worker, initargs=(str(worker_input),)) as pool:
        jobs = {pool.submit(clean_worker, cleaning): cleaning for cleaning in streams}
        for job in as_completed(jobs):
            try:
                job.result()
            except BaseException:
                for pending in jobs:
                    pending.cancel()
                raise
    base_results, fusion_results, gate_results = [], [], []
    for cleaning in streams:
        rows = read_gzip(out / f'metrics-{cleaning}.json.gz')
        base_results.extend(rows['base'])
        fusion_results.extend(rows['fusions'])
        gate_results.extend(rows['gates'])
    all_results = base_results + fusion_results + gate_results
    ordered = sorted(all_results, key=key)
    best = next(r for r in ordered if r['config']['feature_scope'] == 'current_only')
    overall = ordered[0]
    by_family = {}
    for r in sorted(base_results, key=key):
        by_family.setdefault(r['config']['family'], r)
    early = min([r for r in base_results if r['config']['feature_scope'] == 'current_only'],
                key=lambda r: key(r, 'early_supported'))
    _, prior_hashes = verify_run(PRIOR)
    prior_best = json.loads((PRIOR / 'selection.json').read_text())['best_current_only']
    ablations = {}
    prior_ablations = []
    selected_predictions = None
    parity = []
    # Fixed architecture under every cleaning is an explicitly separate
    # comparison from taking the winner of each architecture+cleaning grid.
    for config in (best['config'], prior_best['config']):
        values = []
        for cleaning in streams:
            pred, timing = comparison.replay(core, build(config), items, streams[cleaning], inputs, mappings)
            result = {'training_cleaning': cleaning, 'config': config,
                      'scores': score(cohorts, pred, items), 'timing': timing}
            values.append(result)
            if config['id'] == best['config']['id']:
                same = next((r for r in all_results if r['training_cleaning'] == cleaning
                             and r['config']['id'] == config['id']), None)
                if same:
                    cached = cached_prediction(same, out, core, items, streams[cleaning], inputs)
                    assert all(pred[i]['raw_ranking'] == cached[i]['raw_ranking'] and
                               pred[i]['predictions'] == cached[i]['predictions'] for i in pred)
                    parity.append({'cleaning': cleaning, 'queries': len(pred), 'full_ranking_equal': True})
                if cleaning == best['training_cleaning']:
                    selected_predictions = pred
                io.write_json(out / f'best-predictions-{cleaning}-private.json', pred)
            else:
                if cleaning in ('3', '5', 'raw'):
                    io.write_json(out / f'prior-architecture-{cleaning}-private.json', pred)
        ablations[config['id']] = values
        if config['id'] == prior_best['config']['id']:
            prior_ablations = values
    # No evaluated ablation may quietly outperform a stale reported winner.
    evaluated = all_results + [r for values in ablations.values() for r in values]
    final_best = min([r for r in evaluated if r['config']['feature_scope'] == 'current_only'], key=key)
    grid_best = best
    best = next((r for values in ablations.values() for r in values
                 if r['config']['id'] == final_best['config']['id']
                 and r['training_cleaning'] == final_best['training_cleaning']), final_best)
    if best['config']['id'] != grid_best['config']['id']:
        for cleaning in streams:
            pred, _ = comparison.replay(core, build(best['config']), items,
                streams[cleaning], inputs, mappings)
            io.write_json(out / f'best-predictions-{cleaning}-private.json', pred)
    selected_predictions = {int(i): p for i, p in json.loads(
        (out / f"best-predictions-{best['training_cleaning']}-private.json").read_text()).items()}
    io.write_json(out / 'selected-predictions-private.json', selected_predictions)
    annotated = actions.annotate_predictions(selected_predictions, items)
    io.write_json(out / 'selected-actions-private.json', annotated)
    errors = diagnose(cohorts['5'], selected_predictions, items,
                      streams[best['training_cleaning']], inputs)
    io.write_json(out / 'selected-errors-private.json', errors)
    # The old shown recommendations are rescaled only for scoring; this does
    # not fill empty slots from fifth place or retrospectively add candidates.
    old_pred = {int(i): p for i, p in json.loads((PRIOR / 'best-predictions-3-private.json').read_text()).items()}
    rescored = normalize_old_predictions(old_pred, original)
    rescore_metrics = score(cohorts, rescored, items)
    io.write_json(out / 'old-recommendations-rescored-private.json', rescored)
    best_per_cleaning = {name: min([r for r in evaluated if r['training_cleaning'] == name
        and r['config']['feature_scope'] == 'current_only'], key=key) for name in streams}
    overall = min(evaluated, key=key)
    selection = {'best_current_only': best, 'best_overall': overall, 'original_grid_best': grid_best,
                 'base_family_winners': by_family, 'base_earlier_selected': early,
                 'best_per_cleaning': best_per_cleaning,
                 'old_exact_result': prior_best['scores'],
                 'old_recommendations_new_metric': rescore_metrics,
                 'prior_architecture_new_targets': prior_ablations,
                 'base_config_count': len(configs), 'base_replay_count': len(base_results),
                 'fusion_count': len(fusion_results), 'cleaning_count': len(streams),
                 'evidence_gate_count': len(gate_results),
                 'primary_cohort_size': 62, 'test_previously_seen': True,
                 'errors_by_reason': dict(Counter(e['reason'] for e in errors))}
    io.write_json(out / 'selection.json', selection)
    io.write_json(out / 'cleaning-ablation.json', ablations)
    io.write_json(out / 'streaming-parity.json', parity)
    report(out, selection, ablations, errors)
    assert all(io.sha(Path(__file__).with_name(name)) == digest for name, digest in source_hashes.items()), 'Sources changed during sweep'
    io.write_json(out / 'run.json', {'version': VERSION, **provenance,
        'prior_run': str(PRIOR), 'prior_hashes': prior_hashes,
        'elapsed_seconds': time.monotonic()-started, 'complete': True,
        'raw_data_modified': False, 'live_app_updated': False, 'test_previously_seen': True})
    io.write_json(out / 'artifact-hashes.json', {p.name: io.sha(p) for p in out.iterdir() if p.is_file()})
    print(json.dumps({'complete': True, 'output': str(out),
                      'cleaning': best['training_cleaning'], 'config': best['config'],
                      'scores': best['scores']['5'], 'error_reasons': selection['errors_by_reason']}), flush=True)


def report(out, selection, ablations, errors):
    best = selection['best_current_only']
    def hits(metric):
        return f"{metric['top4_hits']}/{metric['n']} ({metric['top4']:.1%})" if metric['n'] else '0/0'
    lines = ['# 可启动/重开目标与独立清洗实验', '',
        '固定原5秒定义的62条有证据样本和原始观测答案。窗口没有跨会话定位符时按所属应用动作评分；网页、目录、文件仍要求具体目标。应用层正确不等于恢复了原窗口或页面。', '',
        f"比较{selection['cleaning_count']}种训练清洗、{selection['base_config_count']}组基础参数，共{selection['base_replay_count']}次基础因果回放、{selection['fusion_count']}组融合及{selection['evidence_gate_count']}组同起点证据限制。所有选参使用已经查看过的开发数据，晚38条只是时间分段报告。", '',
        '## 得分变化的来源', '',
        f"旧推荐原精确目标Top4：{hits(selection['old_exact_result']['5']['supported'])}。",
        f"原推荐完全不变，仅改为本轮动作评分：{hits(selection['old_recommendations_new_metric']['5']['supported'])}。",
        f"本轮当前界面模型最优：{hits(best['scores']['5']['supported'])}；Top1 {best['scores']['5']['supported']['top1_hits']}/62。", '',
        f"训练清洗：`{best['training_cleaning']}`。模型：", '```json',
        json.dumps(best['config'], ensure_ascii=False, indent=2), '```', '',
        '## 清洗与模型分别比较', '',
        '| 训练清洗 | 固定最优架构 Top4 | 固定架构 Top1 | 该清洗下重新选模型 Top4 |',
        '|---|---:|---:|---:|']
    chosen_ablations = ablations.get(best['config']['id'], [])
    for row in chosen_ablations:
        name = row['training_cleaning']
        m = row['scores']['5']['supported']
        lines.append(f"| {name} | {hits(m)} | {m['top1_hits']}/{m['n']} | {hits(selection['best_per_cleaning'][name]['scores']['5']['supported'])} |")
    lines += ['', '所有行仍在同一62条评分。清洗只改变历史训练标签；新清洗改变了多少到达、是否丢失证据、不同规则的决策依据均在cleaning-audit-private.json.gz。未从主评分中删掉难例。', '',
              '## 模型家族比较', '', '| 家族 | 最优训练清洗 | Top4 | Top1 |', '|---|---|---:|---:|']
    for family, row in selection['base_family_winners'].items():
        m = row['scores']['5']['supported']
        lines.append(f"| {family} | {row['training_cleaning']} | {hits(m)} | {m['top1_hits']}/{m['n']} |")
    lines += ['', '## 分组', '', '```json', json.dumps(best['scores']['5'], ensure_ascii=False, indent=2), '```', '',
              '## 剩余错误', '', f"错因计数：{selection['errors_by_reason']}。", '',
              '| 起点 | 实际目标 | 该清洗历史支持/起点样本 | 原始排名 | 原因 |', '|---|---|---:|---:|---|']
    for e in errors:
        lines.append(f"| {e['origin']} | {e['actual']} | {e['training_support']}/{e['training_origin_total']} | {e['raw_rank']} | {e['reason']} |")
    early = selection['base_earlier_selected']
    lines += ['', '## 时间分段与限制', '',
              f"只按较早24条选基础模型：{early['config']['id']}，清洗{early['training_cleaning']}；早段{hits(early['scores']['5']['early_supported'])}，晚段{hits(early['scores']['5']['late_supported'])}。该选择未使用晚段分数，但整批历史数据此前已被查看。", '',
              '每个模型先预测、仅学习已过原30秒观察窗的历史标签；自适应清洗的阈值只用截点前完整停留，不使用整份导出的未来分布。缺失、失败和过期不被当成确认关闭或零停留。跨会话目标仅由此前的准入记录与有效定位符构建。', '',
              '全部为离线回放。没有真实启动或重开应用、没有修改原始采集、旧run或安装模型；动作计划只验证了记录中的可执行依据，未验证每一次操作系统实际执行成功。', '']
    (out / 'report-private.md').write_text('\n'.join(lines), encoding='utf-8')


if __name__ == '__main__':
    main()
