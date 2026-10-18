"""Bounded architecture comparison, seed sensitivity and forward selection."""
import argparse
from copy import deepcopy
from datetime import datetime, timezone
import gzip
import importlib
import json
import os
from pathlib import Path
import sys
import evaluation_contract as metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--residual-run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    import hashlib
    def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
    def verify(path):
        hashes = json.loads((path/'artifact-hashes.json').read_text())
        for name, digest in hashes.items():
            if Path(name).name != name or sha(path/name) != digest: raise ValueError('Changed artifact: '+name)
        return hashes
    def write(path, value): path.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n')
    root = Path.home()/'Library/Application Support/SmartApp/training'
    source, out = args.residual_run.resolve(), args.output.resolve()
    if out.exists() or out == root or not out.is_relative_to(root): parser.error('Use a new private run')
    source_hashes = verify(source); source_meta = json.loads((source/'run.json').read_text())
    parent = Path(source_meta['source']); parent_hashes = verify(parent)
    parent_meta = json.loads((parent/'run.json').read_text())
    dataset, learning = Path(source_meta['dataset']), Path(parent_meta['learning_run'])
    dataset_hashes = verify(dataset); learning_hashes = verify(learning)
    snapshot = Path(source_meta['snapshot']); raw_hash = sha(snapshot/'events.jsonl')
    if raw_hash != parent_meta['raw_sha256']: raise ValueError('Changed raw snapshot')
    os.umask(0o077); sys.dont_write_bytecode = True; out.mkdir(mode=0o700)
    for path in source.glob('*.py'): (out/path.name).write_bytes(path.read_bytes())
    for name in ('joint_ranking_alternatives.py', 'compare_joint_alternatives.py', 'test_joint_ranking_alternatives.py'):
        (out/name).write_bytes(Path(__file__).with_name(name).read_bytes())
    sys.path.insert(0, str(out))
    models = importlib.import_module('joint_ranking_alternatives')
    comp = importlib.import_module('compare_multibackground')
    cohort_fn = metrics.cohorts
    import numpy as np
    from scipy.sparse import load_npz
    with gzip.open(dataset/'dataset-private.json.gz', 'rt') as f: data = json.load(f)
    with gzip.open(learning/'inputs-private.json.gz', 'rt') as f: inputs = json.load(f)
    with gzip.open(source/'causal-examples-private.json.gz', 'rt') as f: cache = json.load(f)
    matrix = load_npz(source/'causal-features.npz'); examples = {}
    for i, pos in enumerate(cache['positions']):
        stored = cache['examples'][str(pos)]; prior = np.array(stored['prior'])
        examples[pos] = {'labels': stored['labels'], 'prior': prior,
                         'X': matrix[cache['offsets'][i]:cache['offsets'][i+1]],
                         'gate': models.gate_features(inputs[pos], prior)}
    rows = sorted(metrics.prepare_evaluation(data, out), key=lambda r: r['position']); cohorts = cohort_fn(rows)
    late = [r for r in rows if r['split'] != 'validation']
    folds = [list(part) for part in np.array_split(np.array(late, dtype=object), 4)]
    first_cutoff = inputs[late[0]['position']]['index']
    validation = models.past_rows(rows, data['observations'], first_cutoff)
    training = cache['training']; end = max(r['ready_index'] for r in training)+1
    plan = {'configurations': models.configurations(), 'reference_seed': models.REFERENCE_SEED,
            'seeds': models.SEEDS, 'selection': 'sum Top2/Top1/Top4 across all three fixed seeds; evidence, expanded, all',
            'validation_positions': [r['position'] for r in validation],
            'forward_folds': [{'positions': [r['position'] for r in part], 'cutoff': inputs[part[0]['position']]['index']}
                              for part in folds],
            'fold_selection': 'only earlier evaluation rows whose outcomes matured strictly before fold start',
            'same_frozen_statistical_prior': True, 'base_prior_parameters_were_selected_on_old_early_validation': True,
            'fresh_holdout': False, 'manual_mixture_grid': False, 'current_sequence_length': 1, 'inventory_features': False,
            'source': str(source), 'source_hashes': source_hashes, 'raw_sha256': raw_hash,
            'bootstrap': '10000 paired resamples of entire observed sessions; descriptive, not post-selection-valid significance'}
    write(out/'plan.json', plan)
    previous = json.loads((source/'predictions-private.json').read_text())
    previous_summary = json.loads((source/'summary.json').read_text())
    outputs = {name: {int(k): v for k, v in previous[name].items()} for name in ('statistics', 'current_only_statistics')}
    grouped = {name: [outputs[name]]*len(models.SEEDS) for name in outputs}
    configs = models.configurations(); audit = {}; gates = {}; saved = {}
    eval_positions = {r['position'] for r in rows}
    for config in configs:
        model = models.JointRanker(config); cursor = 0; predictions = {}; gate_values = {}
        for q in inputs:
            while cursor < len(training) and training[cursor]['ready_index'] < q['index']:
                model.learn(training[cursor], examples, q['index']); cursor += 1
            pos = q['position']
            if pos in eval_positions:
                predictions[pos] = comp.bundle(model.scores(examples[pos]), cache['homes'][str(pos)])
                if model.gate is not None and examples[pos]['labels']:
                    gate_values[pos] = model.forward(examples[pos])[1]['a']
        while cursor < len(training): model.learn(training[cursor], examples, end); cursor += 1
        if config['seed'] == models.REFERENCE_SEED and config['family'] in ('linear_top2', 'linear_ce'):
            old = ('top2_hinge' if config['family'] == 'linear_top2' else 'cross_entropy')+'_context_lr'+str(config['lr'])
            assert predictions == {int(k): v for k, v in previous[old].items()}
            from statistical_residual import ResidualRanker
            reference = ResidualRanker.load(source/('model-'+old+'-state.json.gz'))
            assert np.array_equal(model.weights, reference.weights)
        name = config['id']; outputs[name] = predictions
        grouped.setdefault(config['group'], []).append(predictions)
        audit[name] = dict(model.audit)
        state = model.save(out, 'model-'+name); saved[name] = state.name
        loaded = models.JointRanker.load(state)
        for pos in eval_positions: assert model.scores(examples[pos]) == loaded.scores(examples[pos])
        item = deepcopy(next(r for r in reversed(training) if r['label'] in examples[r['position']]['labels']))
        item['key'] = ['synthetic-reload-check']+item['key']
        model.learn(item, examples, end); loaded.learn(item, examples, end)
        for key, value in model.arrays().items(): assert np.array_equal(value, loaded.arrays()[key])
        if gate_values: gates[name] = gate_values
        print(json.dumps({'phase': 'trained', 'id': name, 'validation_evidence':
                          metrics.measure([r for r in validation if models.evidence(r)], predictions),
                          'reload_resume_parity': True}), flush=True)
    # All selection uses only the declared earlier, mature predictions.
    selected = models.choose_group(validation, grouped)
    by_family = {family: models.choose_group(validation, {name: values for name, values in grouped.items()
                                                         if name.startswith(family+'_lr')}) for family in models.FAMILIES}
    forward_selections = []
    for part in folds:
        cutoff = inputs[part[0]['position']]['index']
        past = models.past_rows(rows, data['observations'], cutoff)
        chosen = models.choose_group(past, grouped)
        forward_selections.append({'selected': chosen, 'cutoff': cutoff,
                                   'past_positions': [r['position'] for r in past],
                                   'future_positions': [r['position'] for r in part]})
    selection = {'selected': selected, 'by_family': by_family, 'forward': forward_selections,
                 'never_select_best_seed': True, 'full_aggregate_used_for_selection': False}
    write(out/'selection.json', selection)
    # Reference seed is fixed before any metrics; ranges describe the other seeds.
    summaries = {}
    for group, predictions in grouped.items():
        per_seed = [{name: metrics.measure(part, p) for name, part in cohorts.items()} for p in predictions]
        summaries[group] = {'reference': per_seed[0], 'per_seed': per_seed,
                            'validation_key': models.selection_key(validation, predictions)}
    def reference_prediction(group): return grouped[group][0]
    adaptive = {}; fold_report = []
    for part, choice in zip(folds, forward_selections):
        for r in part: adaptive[r['position']] = reference_prediction(choice['selected'])[r['position']]
        subset = [r for r in part if models.evidence(r)]
        fold_report.append({**choice, 'evidence_n': sum(metrics.eligible(r) for r in subset),
                            'models': {name: metrics.measure(subset, reference_prediction(name))
                                       for name in dict.fromkeys(['statistics', 'current_only_statistics',
                                                                  'linear_top2_lr0.1', selected, choice['selected']])}})
    late_groups = {'late_evidence': [r for r in late if models.evidence(r)], 'late_all': late}
    forward_scores = {g: metrics.measure(part, adaptive) for g, part in late_groups.items()}
    for name in dict.fromkeys(['statistics', 'current_only_statistics', 'linear_top2_lr0.1', selected]):
        forward_scores[name] = {g: metrics.measure(part, reference_prediction(name)) for g, part in late_groups.items()}
    def session_bootstrap(before, after):
        totals = {}
        for r in cohorts['evidence_total']:
            if not metrics.eligible(r): continue
            session = data['observations'][r['position']]['session']
            delta = (int(metrics.hits(r, after[r['position']])['accepted'][2])-
                     int(metrics.hits(r, before[r['position']])['accepted'][2]))
            value = totals.setdefault(session, [0, 0]); value[0] += delta; value[1] += 1
        values = np.array(list(totals.values())); rng = np.random.default_rng(20260923)
        draws = rng.integers(0, len(values), size=(10000, len(values)))
        sums = values[draws].sum(axis=1); gains = sums[:, 0]/sums[:, 1]
        return {'session_clusters': len(values), 'point_gain': float(values[:, 0].sum()/values[:, 1].sum()),
                'percentile_95': np.quantile(gains, [.025, .975]).tolist(), 'exploratory_only': True}
    pairs = {name: metrics.paired_topk(cohorts['evidence_total'], outputs['statistics'], reference_prediction(name))
             for name in summaries}
    bootstrap = {name: session_bootstrap(outputs['statistics'], reference_prediction(name))
                 for name in dict.fromkeys(['linear_top2_lr0.1', selected, *by_family.values()])}
    summary = {'selected': selected, 'family_selected': by_family, 'results': summaries,
               'cohort_sizes': metrics.cohort_sizes(cohorts), 'scoring_contract': metrics.contract(),
               'forward_scores': forward_scores,
               'folds': fold_report, 'paired_vs_statistics': pairs, 'session_bootstrap': bootstrap,
               'audit': audit, 'saved_models': saved, 'old_reference_prediction_and_weight_parity': True,
               'all_reload_resume_parity': True, 'data_days': 5.350695, 'fresh_holdout': False,
               'collector_or_runtime_changed': False}
    write(out/'summary.json', summary)
    with gzip.open(out/'predictions-private.json.gz', 'wt') as f: json.dump(outputs, f)
    with gzip.open(out/'learned-gates-private.json.gz', 'wt') as f: json.dump(gates, f)
    labels = {'statistics': '时间背景统计', 'current_only_statistics': '仅当前目标统计',
              'linear_top2': '线性前二修正', 'linear_ce': '线性整体排序修正',
              'linear_smooth_top2': '平滑前二修正', 'context_gate_ce': '情景门控修正', 'nonlinear_top2': '非线性前二修正'}
    def label(name):
        return labels.get(name, next((v for k, v in labels.items() if name.startswith(k+'_lr')), name))
    def cell(m, k): return f"{m[f'top{k}_hits']}/{m['n']} ({m[f'top{k}']:.1%})" if m['n'] else '—'
    display = list(dict.fromkeys(['statistics', 'current_only_statistics', 'linear_top2_lr0.1', *by_family.values()]))
    lines = ['# 统计与情景联合学习：架构、随机性与时间稳定性', '',
             metrics.report_scope(rows), '',
             '五类机制 × 两个学习率 × 三个预先固定随机种子，共 30 次训练。线性两类复现原实验，另补平滑前二、情景门控与 16 单元非线性修正。所有模型基于同一统计先验，初始预测与统计相同，不手调融合比例。',
             f'固定较早验证段按三种子合计成绩选择：{selected}。不挑表现最好的随机种子。下表采用预先固定参考种子，范围来自全部三个种子。', '',
             '| 方案 | Top1 | Top2 | Top4 | 三种子 Top2 命中范围 |', '|---|---:|---:|---:|---:|']
    for name in display:
        r = summaries[name]; m = r['reference']['evidence_total']
        hits = [v['evidence_total']['top2_hits'] for v in r['per_seed']]
        lines.append('| '+label(name)+' (`'+name+'`) | '+' | '.join(cell(m, k) for k in (1, 2, 4))+
                     f" | {min(hits)}–{max(hits)}/{m['n']} |")
    lines += ['', '## 按时间前推选择', '',
              '将早期验证之后的 160 条评测按时间等分成四段，每段 40 条；边界不依据正确率或证据强弱选择。每段只用截点前已成熟的历史评测选择架构和学习率，随后在该段评分。模型自身始终从全部有效成熟训练关系在线更新。统计基础超参数来自原早期验证，故仅在其后做此检查。', '',
              '| 段 | 有证据数 | 选择的方案 | Top1 | Top2 | Top4 |', '|---|---:|---|---:|---:|---:|']
    for i, r in enumerate(fold_report, 1):
        m = r['models'][r['selected']]
        lines.append(f"| {i} | {m['n']} | {r['selected']} | "+' | '.join(cell(m, k) for k in (1, 2, 4))+' |')
    lines += ['', f"后续合法有证据 {forward_scores['late_evidence']['n']} 条合计（不含早期验证）：", '',
              '| 策略 | Top1 | Top2 | Top4 |', '|---|---:|---:|---:|']
    for name, value in [('每段依据此前成绩选择', forward_scores),
                        *[(label(name), forward_scores[name]) for name in dict.fromkeys(['statistics', 'current_only_statistics', 'linear_top2_lr0.1', selected])]]:
        lines.append('| '+name+' | '+' | '.join(cell(value['late_evidence'], k) for k in (1, 2, 4))+' |')
    lines += ['', '## 分时段与证据强弱', '', '| 方案 | 范围 | Top1 | Top2 | Top4 |', '|---|---|---:|---:|---:|']
    for name in dict.fromkeys(['statistics', 'linear_top2_lr0.1', selected]):
        for group in ('old_evidence', 'late_old_evidence', 'new_evidence', 'expanded_total'):
            m = summaries[name]['reference'][group]
            lines.append('| '+name+' | '+group+' | '+' | '.join(cell(m, k) for k in (1, 2, 4))+' |')
    lines += ['', '## 解释边界与复核', '',
              '- 情景门控在训练中同时学习情景修正和其使用强度；非线性模型在同一输入上加入 16 单元 tanh。增加复杂度不保证增益。',
              '- 每次训练使用相同的 1128 个成熟导航关系；原截点输出词表不可用的 334 条只进入统计记忆，残差训练 794 次。所有模型共用审查后的合法评分分母。',
              '- 统计先验、输出词表和情景特征都保留原截点状态，复习旧样本不重算含自身答案的统计分数；不输入后台库存或近期有序路径。',
              '- 参考线性模型的全部历史预测和最终权重与上轮完全一致；30 个模型均验证真实权重、优化器、去重和随机状态重载及继续训练一致。',
              '- 按采集会话成组重采样的 Top2 差值区间保存在 summary.json。只有 8 个会话且已多轮选模，该区间只能用于探索稳定性，不作为显著性证明。',
              '- 所有分段均来自已查看的同一批历史数据；重复种子和时间前推不能替代全新自然行为。结果不能证明跨周或长期稳定，也没有部署或更改采集。', '']
    (out/'report-private.md').write_text('\n'.join(lines))
    for path, hashes in ((source, source_hashes), (parent, parent_hashes), (dataset, dataset_hashes), (learning, learning_hashes)):
        assert verify(path) == hashes
    assert sha(snapshot/'events.jsonl') == raw_hash
    write(out/'run.json', {'complete': True, 'version': 'joint-ranking-alternatives-v1',
                          'created_at': datetime.now(timezone.utc).isoformat(), 'source': str(source),
                          'selected': selected, 'raw_sha256': raw_hash, 'training_runs': len(configs),
                          'source_hashes_verified': True, 'runtime_changed': False})
    write(out/'artifact-hashes.json', {p.name: sha(p) for p in sorted(out.iterdir()) if p.is_file()})
    assert all(p.stat().st_mode & 0o777 == 0o600 for p in out.iterdir() if p.is_file())
    print(json.dumps({'complete': True, 'output': str(out), 'selected': selected,
                      'family_selected': by_family,
                      'evidence_results': {name: summaries[name]['reference']['evidence_total'] for name in display},
                      'forward': forward_scores}), flush=True)


if __name__ == '__main__': main()
