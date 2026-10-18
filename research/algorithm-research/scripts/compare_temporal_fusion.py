"""Fuse frozen time-context statistics and time-context learned scores.

Branch configurations are fixed from earlier early-validation selections. Only
the scalar mixture weight is selected here, again on early validation alone.
Historical learned scores are replayed causally, never obtained from final weights.
"""
import argparse
from copy import deepcopy
from datetime import datetime, timezone
import gzip
import importlib
import json
import math
import os
from pathlib import Path
import sys
import evaluation_contract as scoring

from diagnose_background_contributions import PRIVATE_ROOT, asof_homes, checked_run, paired, sha, write_json


LEARNING_WEIGHTS = tuple(i/10 for i in range(11))


def fuse_scores(statistical, learned, learning_weight):
    if not math.isfinite(learning_weight) or not 0 <= learning_weight <= 1:
        raise ValueError('Learning weight must be between zero and one')
    def unit(values):
        if any(not math.isfinite(v) or v < 0 for v in values.values()):
            raise ValueError('Branch scores must be finite and nonnegative')
        total = sum(values.values())
        return {k: v/total for k, v in values.items()} if total else {}
    a, b = unit(statistical), unit(learned)
    if learning_weight == 0: return a
    if learning_weight == 1: return b
    result = {k: (1-learning_weight)*a.get(k, 0.)+learning_weight*b.get(k, 0.)
              for k in sorted(a.keys() | b.keys())}
    return unit(result)


def select_weight(results):
    def key(result):
        m = result['validation']
        return tuple(m[g][k] for g, k in (
            ('original', 'top4_hits'), ('expanded', 'top4_hits'),
            ('original', 'top1_hits'), ('expanded', 'top1_hits'),
            ('all', 'top4_hits'), ('all', 'top1_hits')))
    # The fixed grid starts with statistics-only; exact ties prefer less learning weight.
    return max(results, key=key)


def weight_name(value):
    return f'learning_{value:.1f}'


def report(summary):
    selected = summary['selected_weight']
    def label(weight):
        return f"统计 {1-weight:.0%}＋学习 {weight:.0%}"
    def cell(m, key):
        return f"{m[key+'_hits']}/{m['n']} ({m[key]:.1%})" if m['n'] else '—'
    display = list(dict.fromkeys([0., 1., .5, selected]))
    lines = ['# 时间背景统计模型与时间背景学习模型融合', '',
        summary['score_scope'],
        f"仅按早期验证选中的权重：{label(selected)}。两个分支分别归一化，再加权排序；比例是分数权重，不是正确预测的贡献率。", '',
        '| 模型 | 有证据合计 Top1 | 有证据合计 Top4 | 返回覆盖 |', '|---|---:|---:|---:|']
    for w in display:
        m = summary['results'][weight_name(w)]['accepted']['evidence_total']['all']
        name = label(w)+('（验证选中）' if w == selected else '')
        lines.append(f"| {name} | {cell(m,'top1')} | {cell(m,'top4')} | {m['prediction_coverage']:.1%} |")
    previous = summary['previous_fusion']['evidence_total']['all']
    lines += [f"| 之前：无背景旧统计 50%＋时间学习 50% | {cell(previous,'top1')} | {cell(previous,'top4')} | {previous['prediction_coverage']:.1%} |", '',
        '## 固定分支与因果回放', '',
        '- 统计分支固定为上轮选出的 5/60 分钟频次背景模型：floor=0.25、相似度平方、语义强度 4、近期目的地回退 0.25、历史票数半衰期 168 小时。',
        '- 学习分支固定为显式交互 softmax，学习率 0.1，当前目标＋5/60 分钟、24 小时、累计背景，使用频次、最近访问衰减、非空闲停留及覆盖信息。读取它的纯学习分数，避免把旧统计分支再次嵌套进融合。',
        '- 两个分支都不读取后台库存，前台序列固定为 1，没有近期有序路径。统计接口需要的库存字段仅设置为空占位，不从原数据读取。',
        '- 使用相同冻结输入、清洗、标签、成熟边界和导航输出语义。每次预测只学习此前已经成熟的标签；后面的新标签不会影响前面的预测。',
        '- 分支参数不变；只比较学习权重 0、0.1、…、1，共 11 组。旧早期验证 74 条中原有证据 24 条的 Top4 优先，再按既定扩展证据和 Top1 顺序破同分，完全同分取更小学习权重。',
        '- selection.json 先保存，随后才计算较晚、新增和合计指标；合计分数不反过来选权重。数据此前已多轮查看，不能称为新独立测试。', '',
        '## 权重验证成绩', '', '| 学习分数权重 | 验证有证据 Top1 | 验证有证据 Top4 |', '|---|---:|---:|']
    for w in LEARNING_WEIGHTS:
        m = summary['results'][weight_name(w)]['validation']['original']
        lines.append(f"| {w:.0%} | {cell(m,'top1')} | {cell(m,'top4')} |")
    lines += ['', '## 时段及证据强弱细分', '', '| 范围 | 模型 | Top1 | Top4 |', '|---|---|---:|---:|']
    for cohort, title in [('old_evidence', '旧固定证据'), ('late_old_evidence', '旧较晚证据'),
                          ('new_evidence', '新增证据'), ('expanded_total', '含较弱语义支持的全部证据')]:
        for w in list(dict.fromkeys([0., selected])):
            m = summary['results'][weight_name(w)]['accepted'][cohort]['all']
            lines.append(f"| {title} | {label(w)} | {cell(m,'top1')} | {cell(m,'top4')} |")
    pair = summary['selected_vs_statistics']['evidence_total']
    lines += ['', '## 与时间统计单模型逐样本比较', '',
        f"- Top1：修正 {pair['top1']['helped']} 次、损失 {pair['top1']['hurt']} 次，净变化 {pair['top1']['net_hits']:+d} 次。",
        f"- Top4：修正 {pair['top4']['helped']} 次、损失 {pair['top4']['hurt']} 次，净变化 {pair['top4']['net_hits']:+d} 次。", '',
        '## 复核与保存', '',
        '- 两个端点分别逐样本复现原时间统计与纯学习模型的预测及可输出集合；学习分支最终权重和优化器数组与原已保存模型完全一致。',
        '- 归一化后的分支分数、各权重预测、选择、精确及服务级指标分别保存。融合模型清单引用经过哈希验证的两个不可变分支，保存所选权重与调用方式。',
        '- 清单重载后，利用已保存的因果分支分数重新计算全部所选预测，并检查一致。两分支的原始模型重载检查沿用各自完成的 run。',
        '- 原始快照、旧模型、源码和结果均校验哈希；目录 0700，文件 0600。没有更改采集、部署或运行中的模型。', '']
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--statistics-run', type=Path, required=True)
    parser.add_argument('--learning-run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    out = args.output.expanduser().resolve()
    if out.exists() or out == PRIVATE_ROOT.resolve() or not out.is_relative_to(PRIVATE_ROOT.resolve()):
        parser.error('Use a new private training run')
    statistics_run, learning_run = args.statistics_run.resolve(), args.learning_run.resolve()
    stat_meta, stat_hashes = checked_run(statistics_run)
    learn_meta, learn_hashes = checked_run(learning_run)
    if stat_meta['dataset'] != learn_meta['dataset'] or stat_meta['snapshot'] != learn_meta['snapshot']:
        raise ValueError('Branches must use the identical frozen dataset and snapshot')
    dataset = Path(learn_meta['dataset']); _, data_hashes = checked_run(dataset)
    snapshot = Path(learn_meta['snapshot'])
    if sha(snapshot/'events.jsonl') != learn_meta['events_sha256']:
        raise ValueError('Raw snapshot changed')
    os.umask(0o077); sys.dont_write_bytecode = True; out.mkdir(mode=0o700)
    for path in statistics_run.glob('*.py'):
        (out/path.name).write_bytes(path.read_bytes())
    for name, digest in learn_meta['source_hashes'].items():
        if not (out/name).exists():
            (out/name).write_bytes((learning_run/('source-'+name)).read_bytes())
        if sha(out/name) != digest:
            raise ValueError('Branch dependency mismatch: '+name)
    local_sources = {}
    for name in ('compare_temporal_fusion.py', 'test_compare_temporal_fusion.py'):
        path = Path(__file__).with_name(name)
        local_sources[name] = sha(path); (out/name).write_bytes(path.read_bytes())
    sys.path.insert(0, str(out))
    comparison = importlib.import_module('compare_multibackground')
    cs = importlib.import_module('context_statistics')
    multi = importlib.import_module('multibackground_model')
    cohort_fn = scoring.cohorts
    import numpy as np
    with gzip.open(dataset/'dataset-private.json.gz', 'rt') as f: data = json.load(f)
    with gzip.open(learning_run/'inputs-private.json.gz', 'rt') as f: inputs = json.load(f)
    for q in inputs:
        q.update(inventory_state={}, inventory_quality={}, inventory_through_index=q['index'])
    homes, asof_index = asof_homes(snapshot, data['observations'], comparison.io, multi.ENTRY)
    rows = scoring.prepare_evaluation(data, out); parts = cohort_fn(rows)
    test_index = min(inputs[r['position']]['index'] for r in rows if r['split'] == 'test')
    validation = [r for r in rows if r['split'] == 'validation' and
                  data['observations'][r['position']]['ready_index'] < test_index]
    stat_config = json.loads((statistics_run/'model-time.json').read_text())['config']
    learning_config = deepcopy(json.loads((learning_run/'selection.json').read_text())['selected_combined']['config'])
    # model.scores is the learned branch only; serving_scores would reintroduce old statistics.
    learning_config['joint_weight'] = 1.
    plan = {'statistical_config': stat_config, 'learning_config': learning_config,
        'learning_weights': LEARNING_WEIGHTS, 'only_mixture_weight_is_tuned': True,
        'validation_positions': [r['position'] for r in validation],
        'statistical_branch': 'normalized PreparedQuery.scores', 'learning_branch': 'JointNavigationModel.scores',
        'nested_old_statistics_in_learning_branch': False, 'background_inventory_features': False,
        'current_sequence_length': 1, 'statistics_hashes': stat_hashes, 'learning_hashes': learn_hashes,
        'dataset_hashes': data_hashes, 'new_source_hashes': local_sources}
    write_json(out/'plan.json', plan)
    stream = [r for r in comparison.training_rows(data, 'hard_kind') if r['head'] == 0]
    training = [{'position': r['position'], 'ready_index': r['ready_index'],
                 'record': comparison.learning_record(r, inputs[r['position']])} for r in stream]
    statistical_queries = cs.prepare_queries(inputs, training, rows, homes, cs.BackgroundIndex(inputs))
    statistical_scores = {pos: comparison.normalize(q.scores(stat_config)) for pos, q in statistical_queries.items()}
    model = multi.JointNavigationModel(learning_config); cursor = 0; learned_scores = {}
    for row in sorted(rows, key=lambda r: r['position']):
        pos = row['position']; q = inputs[pos]
        cursor = comparison.advance(model, stream, cursor, inputs, q['index'])
        learned_scores[pos] = model.scores(q)
    comparison.advance(model, stream, cursor, inputs, asof_index+1)
    with np.load(learning_run/'model-combined-weights-private.npz', allow_pickle=False) as saved:
        if set(saved.files) != set(model.classifier.state_arrays()):
            raise ValueError('Learned state arrays changed')
        for name, values in model.classifier.state_arrays().items():
            if not np.array_equal(values, saved[name]):
                raise ValueError('Replayed learned weights differ: '+name)
    print(json.dumps({'phase': 'causal_scores_ready', 'queries': len(rows),
                      'learning_updates': model.classifier.update_count, 'weight_parity': True}), flush=True)
    expected_stat = {int(k): v for k, v in json.loads((statistics_run/'predictions-private.json').read_text())['time'].items()}
    previous = json.loads((learning_run/'predictions-private.json').read_text())
    pure_name = learning_config['id'].rsplit('_', 1)[0]+'_pure'
    expected_learning = {int(k): v for k, v in previous[pure_name].items()}
    def predictions(weight, positions):
        return {pos: comparison.bundle(fuse_scores(statistical_scores[pos], learned_scores[pos], weight), homes[pos])
                for pos in positions}
    if predictions(0., statistical_scores) != expected_stat or predictions(1., learned_scores) != expected_learning:
        raise ValueError('Pure endpoint prediction parity failed')
    validation_grid = []
    for weight in LEARNING_WEIGHTS:
        values = predictions(weight, [r['position'] for r in validation])
        validation_grid.append({'learning_weight': weight, 'statistics_weight': 1-weight,
                                'validation': scoring.grouped_metrics(validation, values)})
    chosen = deepcopy(select_weight(validation_grid)); selected_weight = chosen['learning_weight']
    write_json(out/'validation-grid.json', validation_grid)
    write_json(out/'selection.json', {'selected': chosen, 'late_or_new_scores_used': False})
    print(json.dumps({'phase': 'weight_selected', 'statistics_weight': 1-selected_weight,
                      'learning_weight': selected_weight}), flush=True)
    outputs, results = {}, {}
    for entry in validation_grid:
        weight = entry['learning_weight']; name = weight_name(weight)
        outputs[name] = predictions(weight, statistical_scores)
        results[name] = {**entry,
            'accepted': {group: scoring.grouped_metrics(members, outputs[name]) for group, members in parts.items()},
            'exact': {group: {g: m['exact'] for g, m in scoring.grouped_metrics(members, outputs[name]).items()} for group, members in parts.items()}}
    selected = outputs[weight_name(selected_weight)]
    pair = {group: scoring.paired_topk(members, outputs[weight_name(0.)], selected) for group, members in parts.items()}
    previous_fusion = {int(k): v for k, v in previous[learning_config['id']].items()}
    branch_paths = {
        'statistics_config': statistics_run/'model-time.json',
        'statistics_memory': statistics_run/'model-memory-private.json.gz',
        'learning_state': learning_run/'model-combined-state-private.json.gz',
        'learning_weights': learning_run/'model-combined-weights-private.npz'}
    manifest = {'version': 'temporal-statistics-learning-fusion-v1', 'learning_weight': selected_weight,
        'statistics_weight': 1-selected_weight, 'learning_call': 'scores', 'statistics_call': 'PreparedQuery.scores',
        'normalize_each_branch_before_mixture': True, 'current_sequence_length': 1,
        'background_inventory_features': False, 'deployment': False,
        'branches': {name: {'path': str(path), 'sha256': sha(path)} for name, path in branch_paths.items()}}
    write_json(out/'fusion-model.json', manifest)
    score_payload = {'statistical': statistical_scores, 'learned': learned_scores, 'homes': homes}
    with gzip.open(out/'branch-scores-private.json.gz', 'wt') as f: json.dump(score_payload, f)
    loaded_manifest = json.loads((out/'fusion-model.json').read_text())
    for item in loaded_manifest['branches'].values():
        if sha(Path(item['path'])) != item['sha256']: raise ValueError('Branch model checksum mismatch')
    with gzip.open(out/'branch-scores-private.json.gz', 'rt') as f: loaded_scores = json.load(f)
    for pos, expected in selected.items():
        scores = fuse_scores(loaded_scores['statistical'][str(pos)], loaded_scores['learned'][str(pos)],
                             loaded_manifest['learning_weight'])
        if comparison.bundle(scores, loaded_scores['homes'][str(pos)]) != expected:
            raise ValueError('Reloaded fusion differs')
    summary = {'selected_weight': selected_weight, 'results': results, 'selected_vs_statistics': pair,
        'previous_fusion': {group: scoring.grouped_metrics(members, previous_fusion) for group, members in parts.items()},
        'cohort_sizes': scoring.cohort_sizes(parts), 'scoring_contract': scoring.contract(), 'score_scope': scoring.report_scope(rows),
        'pure_endpoint_parity': True, 'learned_final_weight_parity': True, 'fusion_manifest_reload_parity': True,
        'collection_or_runtime_changed': False}
    write_json(out/'summary.json', summary)
    write_json(out/'predictions-private.json', outputs)
    (out/'report-private.md').write_text(report(summary))
    for path, expected in [(statistics_run, stat_hashes), (learning_run, learn_hashes), (dataset, data_hashes)]:
        if checked_run(path)[1] != expected: raise ValueError('A source run changed')
    if sha(snapshot/'events.jsonl') != learn_meta['events_sha256']: raise ValueError('Raw snapshot changed')
    for name, digest in local_sources.items():
        if sha(Path(__file__).with_name(name)) != digest: raise ValueError('Source changed during replay')
    write_json(out/'run.json', {'complete': True, 'version': manifest['version'],
        'created_at': datetime.now(timezone.utc).isoformat(), 'statistics_run': str(statistics_run),
        'learning_run': str(learning_run), 'dataset': str(dataset), 'snapshot': str(snapshot),
        'statistics_hashes': stat_hashes, 'learning_hashes': learn_hashes, 'dataset_hashes': data_hashes,
        'raw_sha256': learn_meta['events_sha256'], 'source_hashes_verified_before_after': True,
        'source_hashes': local_sources, 'pure_endpoint_parity': True, 'learned_final_weight_parity': True,
        'fusion_manifest_reload_parity': True, 'collection_or_runtime_changed': False, 'raw_modified': False})
    write_json(out/'artifact-hashes.json', {p.name: sha(p) for p in sorted(out.iterdir()) if p.is_file()})
    if any(p.stat().st_mode & 0o777 != 0o600 for p in out.iterdir() if p.is_file()):
        raise ValueError('Private artifact permissions changed')
    print(json.dumps({'complete': True, 'output': str(out), 'selected_weight': selected_weight,
        'evidence': results[weight_name(selected_weight)]['accepted']['evidence_total']['all']}), flush=True)


if __name__ == '__main__':
    main()
