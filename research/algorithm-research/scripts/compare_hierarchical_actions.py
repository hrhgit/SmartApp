#!/usr/bin/env python3
"""Small two-head action experiment using SHA-pinned prepared replay inputs.

The source sweep can still be running: only its already-complete prepared data,
three-second metrics and predictions are referenced, and their individual hashes
are verified before and after this independent run. No source files are changed.
"""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
import gzip
import hashlib
import json
import os
from pathlib import Path
import time
from types import SimpleNamespace

import compare_action_cleaning as driver
import compare_online_models as comparison
from diagnose_dwell_errors import frozen_core
import evaluate_context_knn as io
import hierarchical_action_model as hierarchical
import online_model_suite as online
import reopenable_actions as actions
import stat_model_suite as stats
from summarize_tuned_history import verify_run


DEFAULT_SOURCE = io.PRIVATE_ROOT / 'reopenable-action-cleaning-v1-20260923-sweep-03'


def sha_stream(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def read_gzip(path):
    with gzip.open(path, 'rt', encoding='utf-8') as handle:
        return json.load(handle)


def dump_gzip(path, value):
    with gzip.open(path, 'wt', encoding='utf-8', compresslevel=6) as handle:
        json.dump(value, handle, ensure_ascii=False, separators=(',', ':'))


def build(config):
    if config['family'] == 'hierarchical_action_heads':
        p = config['parameters']
        return hierarchical.HierarchicalActionModel(driver.build(p['primary']),
                driver.build(p['application']), p['app_weight'], p['rank_offset'])
    return driver.build(config)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-run', type=Path, default=DEFAULT_SOURCE)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    source, out = args.source_run.resolve(), args.output.resolve()
    if not out.is_relative_to(io.PRIVATE_ROOT.resolve()) or out == io.PRIVATE_ROOT.resolve():
        parser.error('Output must be a new private training directory')
    pinned_names = ('worker-input-private.json.gz', 'cleaning-audit-private.json.gz',
                    'metrics-3.json.gz', 'predictions-3-private.json.gz',
                    'source-hashes.json', 'input-audit.json', 'benchmark-membership.json')
    file_hashes = {name: sha_stream(source / name) for name in pinned_names}
    parent_source_hashes = json.loads((source / 'source-hashes.json').read_text())
    for name, digest in parent_source_hashes.items():
        if sha_stream(Path(__file__).with_name(name)) != digest:
            raise ValueError('Live replay source differs from frozen main sweep: ' + name)
    prior_metrics = read_gzip(source / 'metrics-3.json.gz')
    prior = min((row for group in ('base', 'fusions', 'gates') for row in prior_metrics[group]
                 if row['config']['feature_scope'] == 'current_only'), key=comparison.selection_key)
    primary = deepcopy(prior['config'])
    del prior_metrics
    frequency = next(deepcopy(c) for c in stats.configurations() if c['id'] == 'stat_frequency_conditional')
    softmax = next(deepcopy(c) for c in online.configurations() if c['id'] == 'online-softmax-0.0-1.0')
    application_heads = [('frequency', frequency), ('softmax', softmax)]
    weights = (0.0, 0.1, 0.25, 0.5)
    configs = []
    for head_name, head in application_heads:
        for weight in weights:
            configs.append({'id': f'hierarchical-fixed-primary-{head_name}-{weight:g}',
                'family': 'hierarchical_action_heads', 'feature_scope': 'current_only',
                'parameters': {'primary': primary, 'application': head,
                               'app_weight': weight, 'rank_offset': 0}})
    os.umask(0o077)
    out.mkdir(parents=True, mode=0o700, exist_ok=False)
    plan = {'version': 'hierarchical-action-heads-v1', 'source_run': str(source),
        'source_parent_run_may_be_in_progress': True, 'pinned_source_files': file_hashes,
        'primary_selection': 'fixed current-only winner from completed metrics-3.json.gz; no reselect across cleanings',
        'primary_selected_result': prior, 'application_head_choices': application_heads,
        'weights': weights, 'configurations': configs,
        'cleaning_count_expected': 19, 'replays_expected': 152,
        'application_supervision': 'matching cleaning ledger target bundleID/_bundle only; missing/conflicting owners skipped',
        'nonexclusive_targets': 'each head learns same origin/outcome once; page/app are not extra independent arrivals',
        'owner_visibility': 'training only at original mature label time; never input query or x features',
        'primary_scoring': 'same frozen62 actionable targets; page/directory/file still exact',
        'selection': 'full62 development Top4, Top1, MRR4, ID,cleaning; zero-weight controls remain eligible',
        'late38_independent': False, 'raw_data_modified': False, 'live_app_updated': False}
    io.write_json(out / 'plan.json', plan)
    source_hashes = dict(parent_source_hashes)
    for name in ('hierarchical_action_model.py', 'compare_hierarchical_actions.py',
                 'test_hierarchical_action_model.py'):
        source_hashes[name] = sha_stream(Path(__file__).with_name(name))
    for name in source_hashes:
        (out / ('source-' + name)).write_bytes(Path(__file__).with_name(name).read_bytes())
    io.write_json(out / 'source-hashes.json', source_hashes)
    io.write_json(out / 'dependencies.json', online.dependency_metadata())
    print(json.dumps({'stage': 'load_prepared', 'planned_replays': 152,
                      'primary_prior_top4': prior['scores']['5']['supported']['top4_hits']}), flush=True)
    started = time.monotonic()
    payload = read_gzip(source / 'worker-input-private.json.gz')
    cleaning_audit = read_gzip(source / 'cleaning-audit-private.json.gz')
    core = frozen_core(Path(payload['cleaning_run']))
    items = [None] * payload['item_count']
    for position, query in payload['query_items'].items():
        items[int(position)] = SimpleNamespace(**query)
    inputs = {int(index): query for index, query in payload['inputs'].items()}
    mappings = {int(index): mapping for index, mapping in payload['mappings'].items()}
    cohorts = payload['cohorts']
    original_streams = payload['streams']
    assert len(original_streams) == 19
    streams, augmentation_audits = {}, {}
    for cleaning, stream in original_streams.items():
        streams[cleaning], augmentation_audits[cleaning] = hierarchical.augment_training_stream(
            stream, cleaning_audit['ledger_private'][cleaning])
    io.write_json(out / 'augmentation-audit.json', augmentation_audits)
    dump_gzip(out / 'augmented-training-streams-private.json.gz', streams)
    (out / 'benchmark-membership.json').write_bytes((source / 'benchmark-membership.json').read_bytes())
    del payload, cleaning_audit
    results, baselines, parity = [], [], []
    best, best_prediction = None, None
    for cleaning, stream in streams.items():
        baseline_pred, baseline_timing = comparison.replay(core, driver.build(primary), items,
                    original_streams[cleaning], inputs, mappings)
        baseline = {'training_cleaning': cleaning, 'config': primary, 'timing': baseline_timing,
                    'scores': driver.score(cohorts, baseline_pred, items)}
        if cleaning == '3':
            assert baseline['scores'] == prior['scores'], 'Fixed primary replay differs from source 3s metrics'
        baselines.append(baseline)
        predictions = {'unmodified_primary': baseline_pred}
        per_cleaning = []
        for config in configs:
            model = build(config)
            pred, timing = comparison.replay(core, model, items, stream, inputs, mappings)
            result = {'training_cleaning': cleaning, 'config': config, 'timing': timing,
                      'scores': driver.score(cohorts, pred, items)}
            if config['parameters']['app_weight'] == 0:
                assert pred.keys() == baseline_pred.keys()
                assert all(pred[index]['raw_ranking'] == baseline_pred[index]['raw_ranking'] and
                           pred[index]['predictions'] == baseline_pred[index]['predictions']
                           for index in pred), 'Zero application weight must preserve full primary ranks'
                parity.append({'training_cleaning': cleaning, 'config_id': config['id'],
                               'queries': len(pred), 'full_raw_ranking_equal': True, 'top4_equal': True})
            predictions[config['id']] = pred
            per_cleaning.append(result)
            results.append(result)
            if best is None or comparison.selection_key(result) < comparison.selection_key(best):
                best, best_prediction = result, deepcopy(pred)
        dump_gzip(out / f'predictions-{cleaning}-private.json.gz', predictions)
        io.write_json(out / f'metrics-{cleaning}.json', per_cleaning)
        print(json.dumps({'stage': 'cleaning_complete', 'cleaning': cleaning,
                          'completed': len(results), 'total': 152,
                          'best': best['scores']['5']['supported']}), flush=True)
    assert len(results) == 152 and len(parity) == 38
    results.sort(key=comparison.selection_key)
    selected_config = best['config']['id']
    ablation = [result for result in results if result['config']['id'] == selected_config]
    best_per_cleaning = {name: min((result for result in results if result['training_cleaning'] == name),
                                   key=comparison.selection_key) for name in streams}
    best_per_weight = {str(weight): min((result for result in results
                       if result['config']['parameters']['app_weight'] == weight), key=comparison.selection_key)
                       for weight in weights}
    primary_best = min(baselines, key=comparison.selection_key)
    errors = driver.diagnose(cohorts['5'], best_prediction, items,
                             streams[best['training_cleaning']], inputs)
    for error in errors:
        index = error['position']; origin = inputs[index]['origin']
        error['auxiliary_same_origin_support'] = sum(
            row['ready_index'] < items[index].index and row['record']['origin'] == origin
            and row['record'].get('destination_app') == error['truth_action']
            for row in streams[best['training_cleaning']])
    selection = {'best_development': best, 'best_current_only': best,
                 'current_only_recommended': best, 'source_primary_selected': prior,
                 'best_unmodified_primary_across_cleanings': primary_best,
                 'best_per_cleaning': best_per_cleaning, 'best_per_app_weight': best_per_weight,
                 'number_replays': len(results), 'baseline_replays': len(baselines),
                 'zero_weight_rank_parity_checks': len(parity),
                 'errors_by_reason': dict(Counter(row['reason'] for row in errors)),
                 'primary_cohort_size': 62, 'full62_development_selection': True}
    io.write_json(out / 'selection.json', selection)
    io.write_json(out / 'metrics.json', results)
    io.write_json(out / 'primary-baselines.json', baselines)
    io.write_json(out / 'cleaning-ablation.json', {selected_config: ablation})
    io.write_json(out / 'zero-weight-parity.json', parity)
    io.write_json(out / 'selected-errors-private.json',
                  {best['training_cleaning'] + ':' + selected_config: errors})
    dump_gzip(out / 'best-predictions-private.json.gz', actions.annotate_predictions(best_prediction, items))
    main_scores = best['scores']['5']
    lines = ['# 同次到达的双层增量监督实验', '',
        '主head固定为来源3秒清洗下最优的当前界面模型。辅助head从同一成熟到达显式保存的owner学习所属应用；不从以后catalog填补owner，不把应用命中算作具体网页/目录命中。', '',
        '19种训练清洗 × 2种应用head × 4种应用权重，共152次回放。0权重保留为正式候选，38组0权重回放均逐条复现主head完整排名和Top4。', '',
        f"最优Top4：{main_scores['supported']['top4_hits']}/62；Top1：{main_scores['supported']['top1_hits']}/62；原晚段：{main_scores['late_supported']['top4_hits']}/38。", '',
        f"同一固定主head不增加应用监督、只改变清洗，最高Top4为{primary_best['scores']['5']['supported']['top4_hits']}/62。", '',
        '```json', json.dumps(best['config'], ensure_ascii=False, indent=2), '```', '',
        '| 应用head权重 | 全部清洗下最优Top4 | Top1 |', '|---|---:|---:|']
    for weight, result in best_per_weight.items():
        scores = result['scores']['5']['supported']
        lines.append(f"| {weight} | {scores['top4_hits']}/62 | {scores['top1_hits']}/62 |")
    lines += ['', '62条主评分成员及5秒答案保持固定。本次仍在已查看的开发集上选参；晚38条不构成新的独立测试。没有执行真实启动/重开，没有修改原始采集或父run。', '']
    (out / 'report-private.md').write_text('\n'.join(lines))
    assert all(sha_stream(source / name) == digest for name, digest in file_hashes.items()), 'Pinned source data changed'
    assert all(sha_stream(Path(__file__).with_name(name)) == digest
               for name, digest in source_hashes.items()), 'Replay source code changed during run'
    io.write_json(out / 'run.json', {'version': plan['version'], 'source_run': str(source),
        'source_parent_run_may_be_in_progress': True, 'source_file_hashes': file_hashes,
        'source_hashes': source_hashes, 'zero_weight_rank_parity_verified': True,
        'raw_data_modified': False, 'live_app_updated': False, 'test_previously_seen': True,
        'elapsed_seconds': time.monotonic() - started, 'complete': True})
    io.write_json(out / 'artifact-hashes.json',
                  {path.name: sha_stream(path) for path in out.iterdir() if path.is_file()})
    verify_run(out)
    assert out.stat().st_mode & 0o777 == 0o700
    assert all(path.stat().st_mode & 0o777 == 0o600 for path in out.iterdir() if path.is_file())
    print(json.dumps({'complete': True, 'output': str(out), 'best': best,
                      'unmodified_primary_best': primary_best['scores']['5']['supported']}), flush=True)


if __name__ == '__main__':
    main()
