"""Measure a strictly narrower current-transition statistical baseline.

It keeps current-origin counts, semantic-similarity votes, and global mature
counts. It removes both rank-based recency mixing and age decay. The same
causal replay and current composite scorer are used for the two fixed cohorts.
"""
import argparse
from datetime import datetime, timezone
import os
from pathlib import Path

import evaluation_contract as scoring
import evaluate_context_knn as core
import context_statistics as cs
import multibackground_model as multi
import compare_multibackground as comparison
from compare_semantic_trajectory import training_rows
from navigation_hierarchy import learning_record
from review_historical_models import ROOT, Sources, sha, write, navigation_label


CONFIG = {
    'id': 'narrow_current_semantic_global_no_recency_no_decay',
    'mode': 'none', 'profile': None, 'floor': 1.0, 'power': 1.0,
    'semantic_strength': 64.0, 'global_strength': 1.0,
    'recency_mix': 0.0, 'half_life_hours': 0.0,
}


def score_parts(rows, predictions):
    parts = scoring.cohorts(rows)
    parts = {key: parts[key] for key in ('all', 'evidence_total')}
    return {key: scoring.measure(value, predictions) for key, value in parts.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--reference-run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--global-strength', type=float, default=CONFIG['global_strength'])
    args = parser.parse_args()
    config = dict(CONFIG)
    config['global_strength'] = args.global_strength
    if args.global_strength == 0:
        config['id'] = 'narrow_current_semantic_no_global_no_recency_no_decay'
        model_label = '前两项窄化模型'
        component_count_text = '两类'
        component_text = '当前来源的历史转移、语义相似来源的转移'
        model_explanation = '- 这是只保留前两类统计的版本；全局成熟目标频次、近期排序和远近时间衰减均未使用。'
    else:
        model_label = '三项窄化模型'
        component_count_text = '三类'
        component_text = '当前来源的历史转移、语义相似来源的转移、全局成熟目标频次'
        model_explanation = '- 这不是“只按当前来源做条件概率”的最窄版本，因为仍保留语义相似票和全局频次；本轮只去掉近期排序与远近时间衰减。'
    dataset, reference, out = (p.resolve() for p in (args.dataset, args.reference_run, args.output))
    if out.exists() or out == ROOT.resolve() or not out.is_relative_to(ROOT.resolve()):
        parser.error('Use a new private training directory')
    os.umask(0o077); out.mkdir(mode=0o700)
    sources = Sources()
    data = sources.read(dataset/'dataset-private.json.gz')
    dataset_run = sources.read(dataset/'run.json')
    if dataset_run.get('evaluation_scope') != 'complete_two_cohorts':
        raise ValueError('Expected the complete two-cohort validation dataset')
    rows = scoring.prepare_evaluation(data, out)
    positions = {r['position'] for r in rows}
    events = core.load_lines(sources.check(Path(dataset_run['snapshot'])/'events.jsonl'))
    inputs, homes, input_audit = multi.prepare_inputs(events, data['observations'], core.KNOWN_TEST_SESSIONS)
    # The complete validation dataset is built from the foreground/time-input
    # contract and does not carry the legacy inventory fields.  The statistical
    # model below does not use inventory, but its shared query validator expects
    # an explicitly frozen empty inventory source rather than a missing field.
    for q in inputs:
        q.update(inventory_state={}, inventory_quality={}, inventory_through_index=q['index'])
    background = cs.BackgroundIndex(inputs)
    stream = [r for r in training_rows(data, 'hard_kind') if r['head'] == 0
              and navigation_label(r['label'], data['observations'][r['position']])]
    memory = cs.StatisticalMemory(); cursor = 0; predictions = {}
    for row in sorted(rows, key=lambda r: r['position']):
        pos = row['position']; q = inputs[pos]
        while cursor < len(stream) and stream[cursor]['ready_index'] < q['index']:
            item = stream[cursor]; before = inputs[item['position']]
            memory.learn(before, learning_record(item, before), item['ready_index'])
            cursor += 1
        predictions[pos] = memory.prepare(q, homes[pos], background).predict(config)
    if set(predictions) != positions:
        raise ValueError('The narrow model did not predict every common cutoff')
    metrics = score_parts(rows, predictions)
    reference_hashes = sources.read(reference/'artifact-hashes.json')
    for name in reference_hashes:
        sources.check(reference/name)
    reference_meta = sources.read(reference/'run.json')
    if reference_meta.get('evaluation_scope') != 'complete_two_cohorts':
        raise ValueError('Reference run is not the same complete validation scope')
    reference_predictions = sources.read(reference/'predictions-private.json.gz')['statistics/none']
    reference_predictions = {int(k): v for k, v in reference_predictions.items()}
    if set(reference_predictions) != positions:
        raise ValueError('Reference baseline has different validation members')
    baseline_metrics = score_parts(rows, reference_predictions)
    for group in metrics:
        scoring.require_comparable(metrics[group], baseline_metrics[group])
    payload = {'predictions': predictions, 'metrics': metrics, 'baseline_statistics_none': baseline_metrics,
               'config': config, 'training_relations': len(stream), 'mature_relations_at_last_cutoff': cursor}
    write(out/'predictions-private.json', predictions)
    write(out/'metrics-private.json', metrics)
    write(out/'baseline-metrics-private.json', baseline_metrics)
    write(out/'plan.json', {'scope': 'complete_two_cohorts', 'cohorts': ['all', 'evidence_total'],
        'config': config, 'interpretation': 'current-origin + semantic only' if args.global_strength == 0 else 'current-origin + semantic + global only',
        'causal_replay': 'learn only hard_kind records with ready_index strictly before each cutoff',
        'same_scorer': scoring.contract(), 'dataset': str(dataset), 'reference_run': str(reference),
        'input_audit': input_audit, 'training_relations': len(stream), 'rows': len(rows)})
    lines = ['# 窄化纯统计模型复测', '',
        f'本模型只保留{component_count_text}统计：{component_text}。',
        '删除两项：历史记录的时间衰减（所有成熟记录等权）和最近访问排序混合（recency_mix=0）。不读取时间窗口背景或后台库存。', '',
        '综合分沿用统一口径：精确收益1、正确服务首页回退0.6、错误0；Top1至Top4前缀最佳收益平均后换算为100分。', '',
        '| 模型 | 完整验证集 | 完整有证据验证集 |', '|---|---:|---:|']
    for label, value in [('原“纯统计：仅当前”', baseline_metrics), (model_label, metrics)]:
        lines.append(f"| {label} | {100*value['all']['rank_score']['mean']:.2f} | {100*value['evidence_total']['rank_score']['mean']:.2f} |")
    lines += ['', '## 解释', '',
        model_explanation,
        '- 所有模型仍按相同成熟边界在线学习；该模型没有使用预测之后才成熟的标签。',
        '- 两个分母固定为完整合法验证集和完整有证据验证集，未按模型命中情况筛选。', '']
    (out/'report-private.md').write_text('\n'.join(lines))
    sources.verify_unchanged()
    write(out/'run.json', {'complete': True, 'created_at': datetime.now(timezone.utc).isoformat(),
        'dataset': str(dataset), 'reference_run': str(reference), 'evaluation_scope': 'complete_two_cohorts',
        'model': config, 'same_scorer': scoring.contract(), 'historical_predictions_reused_for_training': False,
        'source_files': sources.accessed})
    for path in out.iterdir():
        if path.is_file(): path.chmod(0o600)
    write(out/'artifact-hashes.json', {path.name: sha(path) for path in out.iterdir() if path.is_file()})
    print({'output': str(out), 'metrics': metrics, 'baseline': baseline_metrics}, flush=True)


if __name__ == '__main__':
    main()
