#!/usr/bin/env python3
"""Bounded, explicitly retrospective refinement of a fixed online architecture.

The frozen 62-example five-second cohort selects parameters; the late 38 subset
is reported separately and is not an independent holdout. Each prediction still
uses strictly earlier matured training outcomes. Cleaning is a separate factor.
"""
from __future__ import annotations

import argparse
from collections import Counter
import itertools
import json
import os
from pathlib import Path
import sys
import time

import compare_online_models as comparison
import evaluate_context_knn as io
from fused_online_models import FusedModel
import online_model_suite as online
from search_online_fusions import fuse
from summarize_tuned_history import verify_run


PRIOR = io.PRIVATE_ROOT / 'online-comparison-v1-20260923-final-01'
CLEANINGS = ('3', '5', 'raw')


def components():
    left = []
    for rate, l2 in itertools.product((0.5, 1.0, 2.0), (0.0, 0.001)):
        left.append({'id': f'refine-softmax-{l2}-{rate}', 'family': 'softmax',
                     'feature_scope': 'current_only',
                     'parameters': {'learning_rate': rate, 'l2': l2}})
    right = []
    for neighbors, capacity in itertools.product((10, 20, 40), (500, 1000)):
        right.append({'id': f'refine-streaming_knn-{neighbors}-{capacity}',
                      'family': 'streaming_knn', 'feature_scope': 'current_only',
                      'parameters': {'n_neighbors': neighbors, 'window_size': capacity}})
    return left, right


def build(config):
    if config['family'] == 'rank_fusion_ensemble':
        p = config['parameters']
        return FusedModel(build(p['left']), build(p['right']),
                          p['right_weight'], p['rank_offset'])
    return online.build(config)


def fusion_config(left, right, weight):
    lp, rp = left['parameters'], right['parameters']
    return {'id': f"refine-fusion-{lp['l2']}-{lp['learning_rate']}-{rp['n_neighbors']}-{rp['window_size']}-{weight}",
            'family': 'rank_fusion_ensemble', 'feature_scope': 'current_only',
            'parameters': {'left': left, 'right': right,
                           'right_weight': weight, 'rank_offset': 0}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prior-run', type=Path, default=PRIOR)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    out, prior_path = args.output.resolve(), args.prior_run.resolve()
    if not out.is_relative_to(io.PRIVATE_ROOT.resolve()) or out == io.PRIVATE_ROOT.resolve():
        parser.error('Output must be a new child of the private training directory')
    prior_provenance, prior_hashes = verify_run(prior_path)
    expansion = Path(prior_provenance['expansion_run'])
    _, expansion_hashes = verify_run(expansion)
    if prior_provenance['expansion_hashes'] != expansion_hashes:
        raise ValueError('Prior final and expansion hashes differ')
    old = json.loads((prior_path / 'selection.json').read_text())['best_current_only']
    assert old['config']['feature_scope'] == 'current_only'
    left, right = components()
    weights = (0.25, 0.4, 0.5, 0.6, 0.75)
    plan = {
        'version': 'current-online-winner-refinement-v1',
        'component_grid': {'left': left, 'right': right},
        'fusion_right_weights': weights, 'fusion_rank_offset': 0,
        'training_cleanings': CLEANINGS, 'base_replays': 36, 'fusion_evaluations': 540,
        'primary_scoring': 'same frozen five-second answers and 62 supported sample IDs',
        'selection': 'full62 Top4, then Top1, MRR4, config ID and cleaning; old winner retained',
        'post_selection_late_subset': 'original late38 is reported, not an independent holdout',
        'features': 'current interface only; no new state, timestamp, background or recency inputs',
        'candidates': 'same prediction-time mapping and hierarchy suppression, no candidate changes',
        'label_maturity': 'original ready_index strictly before prediction input index',
        'deduplication': '(semantic current origin, actual outcome event ID)',
        'prior_final': str(prior_path), 'prior_final_hashes': prior_hashes,
        'expansion_run': str(expansion), 'expansion_hashes': expansion_hashes,
        'raw_data_modified': False, 'live_model_updated': False, 'test_previously_seen': True,
        'python': sys.version, 'executable': sys.executable,
    }
    os.umask(0o077)
    out.mkdir(parents=True, mode=0o700, exist_ok=False)
    io.write_json(out / 'plan.json', plan)
    source_files = ('refine_online_winner.py', 'compare_online_models.py',
                    'search_online_fusions.py', 'fused_online_models.py',
                    'online_model_suite.py', 'evaluate_context_knn.py',
                    'evaluate_evidence_model.py', 'evaluate_ordered_current.py',
                    'diagnose_dwell_errors.py', 'summarize_tuned_history.py')
    source_hashes = {}
    for name in source_files:
        source = Path(__file__).with_name(name)
        target = out / ('source-' + name)
        target.write_bytes(source.read_bytes())
        source_hashes[name] = io.sha(source)
    io.write_json(out / 'source-hashes.json', source_hashes)
    io.write_json(out / 'dependencies.json', online.dependency_metadata())
    started = time.monotonic()
    print(json.dumps({'stage': 'load', 'planned_component_replays': 36,
                      'planned_fusion_evaluations': 540}), flush=True)
    core, items, cohorts, streams, inputs, mappings, provenance = comparison.load_dataset(expansion)
    assert len([r for r in cohorts['5'] if r['diagnosis_after_outcome']['eligible']]) == 62
    for position, query in inputs.items():
        query['candidate_key_order'] = mappings[position]
    io.write_json(out / 'input-audit.json', provenance)
    (out / 'benchmark-membership.json').write_bytes((expansion / 'benchmark-membership.json').read_bytes())
    cache, base_metrics, fusion_metrics = {}, [], []
    for cleaning in CLEANINGS:
        for config in left + right:
            predictions, timing = comparison.replay(
                core, build(config), items, streams[cleaning], inputs, mappings)
            cache[(cleaning, config['id'])] = predictions
            result = {'training_cleaning': cleaning, 'config': config, 'timing': timing,
                      'scores': {threshold: comparison.score_groups(rows, predictions)
                                 for threshold, rows in cohorts.items()}}
            base_metrics.append(result)
            io.write_json(out / f"base-predictions-{cleaning}-{config['id']}-private.json",
                          {'config': config, 'training_cleaning': cleaning,
                           'predictions': predictions})
        for a, b, weight in itertools.product(left, right, weights):
            config = fusion_config(a, b, weight)
            predictions = fuse(core, cache[(cleaning, a['id'])],
                               cache[(cleaning, b['id'])], weight, 0, items)
            fusion_metrics.append({'training_cleaning': cleaning, 'config': config,
                                   'scores': {threshold: comparison.score_groups(rows, predictions)
                                              for threshold, rows in cohorts.items()}})
        current = min([old] + base_metrics + fusion_metrics, key=comparison.selection_key)
        print(json.dumps({'stage': 'grid', 'cleaning': cleaning,
                          'completed_base_replays': len(base_metrics),
                          'completed_fusions': len(fusion_metrics),
                          'best': current['scores']['5']['supported']}), flush=True)
    assert len(base_metrics) == 36 and len(fusion_metrics) == 540
    selected = min([old] + base_metrics + fusion_metrics, key=comparison.selection_key)
    best_grid = min(base_metrics + fusion_metrics, key=comparison.selection_key)

    def cached_prediction(result):
        cleaning, config = result['training_cleaning'], result['config']
        if config['id'] == old['config']['id']:
            path = prior_path / f"final-predictions-{cleaning}-{config['id']}-private.json"
            return {int(k): value for k, value in json.loads(path.read_text()).items()}
        if config['family'] == 'rank_fusion_ensemble':
            p = config['parameters']
            return fuse(core, cache[(cleaning, p['left']['id'])],
                        cache[(cleaning, p['right']['id'])], p['right_weight'], p['rank_offset'], items)
        return cache[(cleaning, config['id'])]

    # Stream the selected architecture afresh under every cleaning factor. These
    # replays also verify that cached rank fusion equals the deployable wrapper.
    ablations = []
    verified = []
    for cleaning in CLEANINGS:
        predictions, timing = comparison.replay(core, build(selected['config']),
                                               items, streams[cleaning], inputs, mappings)
        comparison_result = {'training_cleaning': cleaning, 'config': selected['config']}
        expected = cached_prediction(comparison_result)
        assert predictions.keys() == expected.keys()
        assert all(predictions[i]['raw_ranking'] == expected[i]['raw_ranking'] and
                   predictions[i]['predictions'] == expected[i]['predictions']
                   for i in predictions), 'Full streaming/cached rank or Top4 parity failed'
        verified.append({'training_cleaning': cleaning, 'queries': len(predictions),
                         'all_raw_rankings_equal': True, 'all_top4_equal': True})
        result = {**comparison_result, 'timing': timing,
                  'scores': {threshold: comparison.score_groups(rows, predictions)
                             for threshold, rows in cohorts.items()},
                  'prediction_file': f'best-predictions-{cleaning}-private.json'}
        ablations.append(result)
        io.write_json(out / result['prediction_file'], predictions)
        io.write_json(out / f'best-errors-{cleaning}-private.json',
                      comparison.diagnose(cohorts['5'], predictions))
    best = min(ablations, key=comparison.selection_key)
    # Scoring/parity ensures a fresh rerun cannot silently improve the old grid.
    assert comparison.selection_key(best)[:3] <= comparison.selection_key(old)[:3]
    assert comparison.selection_key(best)[:3] == comparison.selection_key(selected)[:3]
    robustness = {}
    for cleaning in CLEANINGS:
        rows = [r for r in fusion_metrics if r['training_cleaning'] == cleaning]
        distribution = Counter(r['scores']['5']['supported']['top4_hits'] for r in rows)
        robustness[cleaning] = {'evaluated_fusions': len(rows),
                               'top4_hit_distribution': dict(sorted(distribution.items())),
                               'at_least_old54': sum(n for hits, n in distribution.items() if hits >= 54),
                               'best': min(rows, key=comparison.selection_key)}
    selection = {'best_development': best, 'best_current_only': best, 'current_only_recommended': best,
                 'prior_best': old, 'best_refinement_grid': best_grid,
                 'primary_cohort_size': 62, 'candidate_ceiling': 56,
                 'component_replays': 36, 'fusion_evaluations': 540,
                 'selected_architecture_replays': 3,
                 'improvement_top4_hits': best['scores']['5']['supported']['top4_hits'] - old['scores']['5']['supported']['top4_hits'],
                 'improvement_top1_hits': best['scores']['5']['supported']['top1_hits'] - old['scores']['5']['supported']['top1_hits'],
                 'full62_development_selection': True, 'late38_independent': False}
    io.write_json(out / 'selection.json', selection)
    io.write_json(out / 'component-grid.json', base_metrics)
    io.write_json(out / 'fusion-grid.json', fusion_metrics)
    io.write_json(out / 'cleaning-ablation.json', {best['config']['id']: ablations})
    io.write_json(out / 'local-parameter-robustness.json', robustness)
    io.write_json(out / 'streaming-parity.json', verified)
    selected_errors = json.loads((out / f"best-errors-{best['training_cleaning']}-private.json").read_text())
    io.write_json(out / 'selected-errors-private.json',
                  {best['training_cleaning'] + ':' + best['config']['id']: selected_errors})
    main_scores = best['scores']['5']
    lines = ['# 固定当前界面融合模型的局部参数搜索', '',
             '本次在已查看的62条有证据开发样本上选参；原晚38条是分段报告，不是独立测试集。成员、答案、候选、层级去重和成熟标签时序均保持固定。', '',
             '计划在运行前写入：Softmax学习率0.5/1/2，L2为0/0.001；流式kNN邻居10/20/40、容量500/1000；右侧kNN融合权重0.25/0.4/0.5/0.6/0.75，倒数排名偏移0；训练标签分别为3秒、5秒和未做短停留清洗。', '',
             f"36次基础回放、540组融合评分及3次最终真实流式复核。最优Top4为{main_scores['supported']['top4_hits']}/62，Top1为{main_scores['supported']['top1_hits']}/62；候选存在样本Top4为{main_scores['covered_supported']['top4_hits']}/56，原晚段为{main_scores['late_supported']['top4_hits']}/38。", '',
             f"相对上轮54/62，Top4增加{selection['improvement_top4_hits']}条。", '',
             '```json', json.dumps(best['config'], ensure_ascii=False, indent=2), '```', '',
             '同一模型参数下，训练清洗与评分答案分开：评分始终固定5秒答案及62条成员。', '',
             '| 训练清洗 | Top4/62 | Top1/62 | 原晚38 Top4 |', '|---|---:|---:|---:|']
    for result in ablations:
        scores = result['scores']['5']
        lines.append(f"| {result['training_cleaning']} | {scores['supported']['top4_hits']}/62 | {scores['supported']['top1_hits']}/62 | {scores['late_supported']['top4_hits']}/38 |")
    lines += ['', '证据分组：', '', '```json',
              json.dumps(main_scores['grades'], ensure_ascii=False, indent=2), '```', '',
              '所有最终回放的完整候选排名与Top4均逐条等于缓存融合结果。局部参数表现分布、实际学习/预测耗时、失败记录、依赖及源代码快照保存在本私有run。原始数据和线上模型均未修改。', '']
    (out / 'report-private.md').write_text('\n'.join(lines))
    # Frozen source snapshots, not source paths, define the completed run.
    assert all(io.sha(Path(__file__).with_name(name)) == digest
               for name, digest in source_hashes.items()), 'Source changed during run'
    io.write_json(out / 'run.json', {'version': plan['version'], **provenance,
                                   'prior_final': str(prior_path), 'prior_final_hashes': prior_hashes,
                                   'source_hashes': source_hashes, 'raw_data_modified': False,
                                   'live_model_updated': False, 'test_previously_seen': True,
                                   'streaming_cached_predictions_verified': True,
                                   'elapsed_seconds': time.monotonic() - started, 'complete': True})
    io.write_json(out / 'artifact-hashes.json',
                  {path.name: io.sha(path) for path in out.iterdir() if path.is_file()})
    verify_run(out)
    assert (out.stat().st_mode & 0o777) == 0o700
    assert all((path.stat().st_mode & 0o777) == 0o600 for path in out.iterdir() if path.is_file())
    print(json.dumps({'complete': True, 'output': str(out), 'best': best,
                      'delta_top4': selection['improvement_top4_hits'],
                      'delta_top1': selection['improvement_top1_hits']}), flush=True)


if __name__ == '__main__':
    main()
