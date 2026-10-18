"""Apply the fixed smooth-Top2 online correction to two narrow statistics.

The statistical priors differ only by the global mature-frequency component:
the two-component prior keeps current-origin and semantic votes, while the
three-component prior also keeps global votes.  Both use no recency mixture or
age decay.  The learned correction is the previously fixed
linear_smooth_top2_lr0.1_seed20260923 model.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import os
from pathlib import Path

import numpy as np

import context_statistics as cs
import evaluation_contract as scoring
import evaluate_context_knn as core
import joint_ranking_alternatives as smooth
import multibackground_model as multi
import compare_multibackground as comparison
import statistical_residual as residual
from compare_semantic_trajectory import training_rows
from navigation_hierarchy import learning_record
from review_historical_models import ROOT, Sources, navigation_label, sha, write


BASE_CONFIGS = {
    'two': {
        'id': 'narrow_current_semantic_no_global_no_recency_no_decay',
        'mode': 'none', 'profile': None, 'floor': 1.0, 'power': 1.0,
        'semantic_strength': 64.0, 'global_strength': 0.0,
        'recency_mix': 0.0, 'half_life_hours': 0.0,
    },
    'three': {
        'id': 'narrow_current_semantic_global_no_recency_no_decay',
        'mode': 'none', 'profile': None, 'floor': 1.0, 'power': 1.0,
        'semantic_strength': 64.0, 'global_strength': 1.0,
        'recency_mix': 0.0, 'half_life_hours': 0.0,
    },
}

SMOOTH_CONFIG = {
    'id': 'linear_smooth_top2_lr0.1_seed20260923',
    'group': 'linear_smooth_top2_lr0.1',
    'family': 'linear_smooth_top2', 'objective': 'top2_logistic',
    'lr': 0.1, 'seed': smooth.REFERENCE_SEED, 'rehearsal': 7,
    'context': True,
}


def checked_source(path, sources):
    path = path.resolve()
    if not path.is_relative_to(ROOT.resolve()):
        raise ValueError('Source run must be under the private training root')
    meta = sources.read(path/'run.json')
    hashes = sources.read(path/'artifact-hashes.json')
    for name in hashes:
        sources.check(path/name)
    return meta, hashes


def score_parts(rows, predictions):
    groups = scoring.cohorts(rows)
    return {name: scoring.measure(groups[name], predictions)
            for name in ('all', 'evidence_total')}


def descriptions_for(memory, mapping):
    descriptions = {}
    for record in memory.hierarchy.records:
        emitted = mapping.get(record['navigationLabel'])
        if emitted is None:
            continue
        target = record.get('destination') or {}
        desc = descriptions.setdefault(emitted, {
            'aliases': set(), 'owner': target.get('bundleID', ''),
            'kind': target.get('kind', 'unknown'),
        })
        desc['aliases'].add(record['label'])
    return {key: {**value, 'aliases': sorted(value['aliases'])}
            for key, value in descriptions.items()}


def positive_prior(scores):
    """Keep zero-vote known labels without changing their rank order."""
    floor = float(np.finfo(float).tiny)
    return {label: max(float(value), floor) for label, value in scores.items()}


def build_base(inputs, homes, background, stream, config):
    """Build frozen per-cutoff priors and all mature training relations."""
    memory = cs.StatisticalMemory()
    examples, training, baseline = {}, [], {}
    cursor = 0

    def mature_until(cutoff):
        nonlocal cursor
        while cursor < len(stream) and stream[cursor]['ready_index'] < cutoff:
            item = stream[cursor]
            before = inputs[item['position']]
            before_count = len(memory.hierarchy.records)
            memory.learn(before, learning_record(item, before), item['ready_index'])
            if len(memory.hierarchy.records) > before_count:
                record = memory.hierarchy.records[-1]
                if record['navigationDisposition'] == 'navigation':
                    example = examples[item['position']]
                    label = example['mapping'].get(record['navigationLabel'], record['navigationLabel'])
                    training.append({
                        'position': item['position'],
                        'input_index': before['index'],
                        'ready_index': item['ready_index'],
                        'label': label,
                        'weight': record['weight'],
                        'key': [record['origin'], record['label'], record['outcomeID']],
                    })
            cursor += 1

    for q in inputs:
        mature_until(q['index'])
        prepared = memory.prepare(q, homes[q['position']], background)
        scores = prepared.scores(config)
        mapping = {
            prepared.labels[i]: prepared.emitted[j]
            for i, j in zip(prepared.keep, prepared.output_ids)
        }
        # The two-component prior can assign exact zero to known labels.  The
        # learned ranker operates in log-probability space, so preserve those
        # labels in the frozen vocabulary with the smallest positive float.
        example = residual.encode_features(
            q, positive_prior(scores), descriptions_for(memory, mapping))
        example['mapping'] = mapping
        examples[q['position']] = example
        baseline[q['position']] = prepared.predict(config)

    # Relations whose outcomes mature after the final prediction cutoff still
    # belong to the online learner's eventual memory, but not to any prediction
    # before their ready_index.
    mature_until(float('inf'))
    if cursor != len(stream):
        raise ValueError('Not all mature relations were replayed')
    return examples, training, baseline


def smooth_predictions(inputs, homes, examples, training, eval_positions):
    model = smooth.JointRanker(SMOOTH_CONFIG)
    cursor = 0
    predictions = {}
    for q in inputs:
        while cursor < len(training) and training[cursor]['ready_index'] < q['index']:
            model.learn(training[cursor], examples, q['index'])
            cursor += 1
        if q['position'] in eval_positions:
            predictions[q['position']] = comparison.bundle(
                model.scores(examples[q['position']]), homes[q['position']])
    while cursor < len(training):
        model.learn(training[cursor], examples, float('inf'))
        cursor += 1
    return model, predictions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--two-run', type=Path, required=True)
    parser.add_argument('--three-run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    dataset, two_run, three_run, out = (p.resolve() for p in (
        args.dataset, args.two_run, args.three_run, args.output))
    if out.exists() or out == ROOT.resolve() or not out.is_relative_to(ROOT.resolve()):
        parser.error('Use a new private training directory')
    os.umask(0o077)
    out.mkdir(mode=0o700)

    sources = Sources()
    data = sources.read(dataset/'dataset-private.json.gz')
    dataset_meta = sources.read(dataset/'run.json')
    if dataset_meta.get('evaluation_scope') != 'complete_two_cohorts':
        raise ValueError('Expected complete two-cohort validation data')
    two_meta, _ = checked_source(two_run, sources)
    three_meta, _ = checked_source(three_run, sources)
    for meta in (two_meta, three_meta):
        if meta.get('evaluation_scope') != 'complete_two_cohorts':
            raise ValueError('Narrow prior run is not the complete validation scope')
        if Path(meta['dataset']).resolve() != dataset:
            raise ValueError('Narrow prior uses a different dataset')

    rows = scoring.prepare_evaluation(data, out)
    positions = {row['position'] for row in rows}
    events = core.load_lines(sources.check(Path(dataset_meta['snapshot'])/'events.jsonl'))
    inputs, homes, input_audit = multi.prepare_inputs(
        events, data['observations'], core.KNOWN_TEST_SESSIONS)
    for q in inputs:
        q.update(inventory_state={}, inventory_quality={}, inventory_through_index=q['index'])
    background = cs.BackgroundIndex(inputs)
    stream = sorted(
        (row for row in training_rows(data, 'hard_kind')
         if row['head'] == 0 and
         navigation_label(row['label'], data['observations'][row['position']])),
        key=lambda row: (row['ready_index'], row['position']))

    predictions = {}
    metrics = {}
    audits = {}
    saved = {}
    source_predictions = {}
    run_paths = {'two': two_run, 'three': three_run}
    for name in ('two', 'three'):
        source_payload = sources.read(run_paths[name]/'predictions-private.json')
        source_predictions[name] = {
            int(key): value for key, value in source_payload.items()
        }
        examples, training, base = build_base(
            inputs, homes, background, stream, BASE_CONFIGS[name])
        if {key: base[key] for key in positions} != {
                key: source_predictions[name][key] for key in positions}:
            raise ValueError(f'{name} prior parity failed against the frozen narrow run')
        model, learned = smooth_predictions(inputs, homes, examples, training, positions)
        predictions[f'{name}_base'] = {key: base[key] for key in positions}
        predictions[f'{name}_smooth_top2'] = learned
        metrics[f'{name}_base'] = score_parts(rows, predictions[f'{name}_base'])
        metrics[f'{name}_smooth_top2'] = score_parts(rows, learned)
        audits[name] = {
            'training_relations': len(training),
            'model': dict(model.audit),
            'base_config': BASE_CONFIGS[name],
        }
        state = model.save(out, f'model-{name}-smooth-top2')
        loaded = smooth.JointRanker.load(state)
        for position in positions:
            if model.scores(examples[position]) != loaded.scores(examples[position]):
                raise ValueError(f'{name} model reload parity failed')
        saved[name] = state.name

    write(out/'predictions-private.json', predictions)
    write(out/'metrics-private.json', metrics)
    write(out/'plan.json', {
        'scope': 'complete_two_cohorts',
        'base_models': BASE_CONFIGS,
        'learned_model': SMOOTH_CONFIG,
        'base_prior_parity_checked_against': {k: str(v) for k, v in run_paths.items()},
        'causal_replay': 'learn only hard_kind navigation relations with ready_index strictly before each cutoff',
        'same_scorer': scoring.contract(),
        'learned_features': 'the established residual feature encoder, including frozen temporal background descriptors',
        'zero_prior_floor': float(np.finfo(float).tiny),
        'zero_prior_policy': 'replace only zero scores by the smallest positive float; preserve all labels and their rank order',
        'input_audit': input_audit,
        'audits': audits,
        'saved_models': saved,
        'dataset': str(dataset),
    })

    def cell(value, key):
        return f"{value[f'top{key}_hits']}/{value['n']} ({value[f'top{key}']:.1%})"

    lines = [
        '# 前两项／前三项窄统计上的平滑 Top2 学习', '',
        '基础统计先验都关闭近期排序和历史时间衰减；前两项只保留当前来源转移与语义相似来源转移，前三项再加入全局成熟目标频次。',
        '学习层固定使用此前的 `linear_smooth_top2_lr0.1_seed20260923`：平滑 Top2 logistic 损失、学习率 0.1、最多复习 7 条成熟关系、固定参考种子。',
        '学习层沿用原平滑 Top2 的上下文特征编码，因此虽然统计先验不含近期／衰减，学习修正仍可从冻结的时间背景描述中学习；这是“窄统计先验＋原平滑学习层”的对照。', '',
        '| 模型 | 完整验证集综合分 | 有证据验证集综合分 | 完整 Top1 / Top2 / Top4 | 有证据 Top1 / Top2 / Top4 |',
        '|---|---:|---:|---|---|',
    ]
    for name, label in (
        ('two_base', '前两项基础'), ('two_smooth_top2', '前两项＋平滑Top2'),
        ('three_base', '前三项基础'), ('three_smooth_top2', '前三项＋平滑Top2')):
        all_metrics = metrics[name]['all']
        evidence_metrics = metrics[name]['evidence_total']
        lines.append(
            f"| {label} | {100*all_metrics['rank_score']['mean']:.2f} | "
            f"{100*evidence_metrics['rank_score']['mean']:.2f} | "
            f"{cell(all_metrics, 1)} / {cell(all_metrics, 2)} / {cell(all_metrics, 4)} | "
            f"{cell(evidence_metrics, 1)} / {cell(evidence_metrics, 2)} / {cell(evidence_metrics, 4)} |"
        )
    lines += ['', '## 学习层增益', '']
    for name, label in (('two', '前两项'), ('three', '前三项')):
        base = metrics[f'{name}_base']
        learned = metrics[f'{name}_smooth_top2']
        lines.append(
            f"- {label}：完整集综合分 "
            f"{100*base['all']['rank_score']['mean']:.2f} → "
            f"{100*learned['all']['rank_score']['mean']:.2f}；"
            f"有证据集 {100*base['evidence_total']['rank_score']['mean']:.2f} → "
            f"{100*learned['evidence_total']['rank_score']['mean']:.2f}。"
        )
    lines += ['', '## 边界', '',
              '- 训练标签仍按真实成熟时间进入；没有使用预测截点之后才成熟的标签。',
              '- 两个分母固定为完整合法验证集和完整有证据验证集；没有按模型命中情况筛样本。',
              '- 这是固定平滑 Top2 配置的迁移对照，不是针对两个窄先验重新搜索学习率或模型结构。', '']
    (out/'report-private.md').write_text('\n'.join(lines))
    (out/'report-private.md').chmod(0o600)
    sources.verify_unchanged()
    write(out/'run.json', {
        'complete': True,
        'created_at': datetime.now(timezone.utc).isoformat(),
        'dataset': str(dataset),
        'evaluation_scope': 'complete_two_cohorts',
        'base_runs': {k: str(v) for k, v in run_paths.items()},
        'learned_model': SMOOTH_CONFIG,
        'same_scorer': scoring.contract(),
        'historical_predictions_reused_for_training': False,
        'source_files': sources.accessed,
    })
    for path in out.iterdir():
        if path.is_file():
            path.chmod(0o600)
    write(out/'artifact-hashes.json', {
        path.name: sha(path) for path in out.iterdir() if path.is_file()
    })
    print({'output': str(out), 'metrics': metrics, 'audits': audits}, flush=True)


if __name__ == '__main__':
    main()
