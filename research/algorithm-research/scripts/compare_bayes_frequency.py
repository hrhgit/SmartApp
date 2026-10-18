"""Causal offline comparison of fixed, global-frequency and contextual miss costs."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import gzip
import importlib
import json
import os
from pathlib import Path
import sys

import evaluation_contract as scoring
import bayes_frequency_policy as policy
from rescore_predictions import sha, verify


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--residual-run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(); source, out = args.residual_run.resolve(), args.output.resolve()
    root = (Path.home()/'Library/Application Support/SmartApp/training').resolve()
    if out.exists() or out == root or not out.is_relative_to(root): parser.error('Use a new private run')
    meta = json.loads((source/'run.json').read_text()); fusion = Path(meta['source'])
    parent = json.loads((fusion/'run.json').read_text())
    dataset, learning, statistics = (Path(parent[k]) for k in ('dataset', 'learning_run', 'statistics_run'))
    sources = [source, fusion, dataset, learning, statistics]
    hashes = {str(p): verify(p) for p in sources}
    os.umask(0o077); sys.dont_write_bytecode = True; out.mkdir(mode=0o700)
    def write(name, value): (out/name).write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n')
    # Frozen predictor/cleaning only. The scorer is already loaded from current code.
    for p in source.glob('*.py'): (out/p.name).write_bytes(p.read_bytes())
    for name in ('compare_bayes_frequency.py', 'bayes_frequency_policy.py', 'rescore_predictions.py'):
        (out/name).write_bytes(Path(__file__).with_name(name).read_bytes())
    sys.path.insert(0, str(out))
    comp = importlib.import_module('compare_multibackground')
    cs = importlib.import_module('context_statistics')
    residual = importlib.import_module('statistical_residual')
    import numpy as np
    from scipy.sparse import load_npz
    with gzip.open(dataset/'dataset-private.json.gz', 'rt') as f: data = json.load(f)
    with gzip.open(learning/'inputs-private.json.gz', 'rt') as f: inputs = json.load(f)
    with gzip.open(source/'causal-examples-private.json.gz', 'rt') as f: cache = json.load(f)
    for q in inputs: q.update(inventory_state={}, inventory_quality={}, inventory_through_index=q['index'])
    matrix = load_npz(source/'causal-features.npz'); examples = {}
    for i, pos in enumerate(cache['positions']):
        x = cache['examples'][str(pos)]
        examples[pos] = {'labels': x['labels'], 'prior': np.array(x['prior']),
                         'X': matrix[cache['offsets'][i]:cache['offsets'][i+1]], 'mapping': x['mapping']}
    rows = scoring.prepare_evaluation(data, out); parts = scoring.cohorts(rows)
    by_position = {r['position']: r for r in rows}
    chosen = json.loads((source/'selection.json').read_text())['selected']
    old = json.loads((source/'predictions-private.json').read_text())[chosen['id']]
    config = json.loads((statistics/'model-time.json').read_text())['config']
    write('plan.json', {'predictor': chosen['config'], 'cost_modes': policy.MODES,
        'frequency_direction': 'larger frequency means larger miss cost',
        'context': 'same current origin, weighted by frozen time-background similarity',
        'context_background_config': config, 'child_dirichlet_prior_strength': policy.PRIOR_STRENGTH,
        'unseen_root_mass': 'Beta(1,1) predictive rate of prequential root-vocabulary misses',
        'child_prior': 'global service outcomes plus one unknown-child pseudo-arrival',
        'cost_smoothing': 'global add-one; context adds one global pseudo-arrival',
        'frequency_unit': 'unique mature cleaned navigation outcomes, not inventory or heartbeats',
        'selection_or_cost_tuning': False, 'frozen_predictor_causal_replay': True,
        'top1': 'exact minimum expected weighted loss under estimated probabilities',
        'top2_top4': 'greedy minimum marginal set loss, ideal menu selection assumption',
        'scoring': f'all policies evaluated by identical unweighted 0/{scoring.FALLBACK_LOSS:g}/1 loss, graded rank score and exact/accepted TopK',
        'scoring_contract': scoring.contract(), 'source_hashes': hashes,
        'probabilities_calibrated': False, 'fresh_holdout': False})
    stream = [r for r in comp.training_rows(data, 'hard_kind') if r['head'] == 0]
    memory, background = cs.StatisticalMemory(), cs.BackgroundIndex(inputs)
    model = residual.ResidualRanker(chosen['config']); cursor = learner_cursor = 0
    global_counts = Counter(); seen_outcomes = set(); records = []; catalog = {}
    root_vocabularies = {}; unseen_count = 0
    outputs = {'existing_order': {int(k): v for k, v in old.items()}, **{k: {} for k in policy.MODES}}
    posterior_log = {}; decision_log = {}; causal_audit = []
    aliases_log = {}; entries_log = {}
    for q in inputs:
        while cursor < len(stream) and stream[cursor]['ready_index'] < q['index']:
            item = stream[cursor]; before = inputs[item['position']]
            assert before['index'] <= item['ready_index'] < q['index']
            assert item['record']['target_evidence_index'] <= item['ready_index']
            n = len(memory.hierarchy.records)
            memory.learn(before, comp.learning_record(item, before), item['ready_index'])
            if len(memory.hierarchy.records) > n:
                r = memory.hierarchy.records[-1]
                if r['navigationDisposition'] == 'navigation':
                    target = r['destination']; catalog[target['id']] = target
                    records.append({**r, 'position': item['position'], 'ready_index': item['ready_index']})
                    if r['outcomeID'] not in seen_outcomes:
                        seen_outcomes.add(r['outcomeID']); global_counts[r['label']] += min(1., r['weight'])
                        actual_root = scoring.service(target) or r['label']
                        unseen_count += actual_root not in root_vocabularies[item['position']]
            cursor += 1
        while learner_cursor < len(cache['training']) and cache['training'][learner_cursor]['ready_index'] < q['index']:
            model.learn(cache['training'][learner_cursor], examples, q['index']); learner_cursor += 1
        pos = q['position']; catalog[q['target']['id']] = q['target']
        aliases, entries = scoring.decision_hierarchy(catalog.values())
        aliases[policy.UNKNOWN_CHILD] = scoring.SERVICE
        root_vocabularies[pos] = {aliases.get(y, y) for y in examples[pos]['labels']}
        if pos not in by_position: continue
        scores = model.scores(examples[pos]); home = cache['homes'][str(pos)]
        assert comp.bundle(scores, home) == old[str(pos)]
        # Conditional frequencies: same current target, similar unordered time backgrounds.
        local = [r for r in records if r['origin'] == q['origin']]
        positions = np.asarray([r['position'] for r in local], dtype=np.int64)
        factors = background.factors(pos, positions, config) if local else []
        local_counts = Counter()
        for r, factor in zip(local, factors): local_counts[r['label']] += min(1., r['weight'])*float(factor)
        unseen_p = (unseen_count+1)/(len(seen_outcomes)+2)
        posterior, actions = policy.expand_service_probabilities(scores, aliases, entries, global_counts, local_counts, unseen_p)
        current = q['target']; forbidden = {current['id'], 'app:'+current.get('bundleID', '')}
        if current.get('locator') == 'https://chatgpt.com/': forbidden.add(scoring.SERVICE)
        actions = [a for a in actions if a not in forbidden]
        posterior_log[pos] = posterior; decision_log[pos] = {}
        aliases_log[pos] = aliases; entries_log[pos] = sorted(entries)
        for mode in policy.MODES:
            costs = policy.frequency_costs(posterior, global_counts, local_counts, mode)
            menu, risks = policy.risk_menu(posterior, actions, aliases, entries, costs)
            emitted = [home if a == scoring.SERVICE and home else a for a in menu]
            all_emitted = list(dict.fromkeys(home if a == scoring.SERVICE and home else a for a in actions))
            outputs[mode][pos] = {'predictions': emitted, 'raw': menu, 'rankable': all_emitted}
            decision_log[pos][mode] = {'outcome_costs': costs, 'expected_set_losses': risks,
                                      'global_counts': dict(global_counts), 'context_counts': dict(local_counts)}
        causal_audit.append({'position': pos, 'cutoff': q['index'], 'mature_outcomes': len(seen_outcomes),
                             'latest_used_ready_index': max((r['ready_index'] for r in records), default=-1),
                             'unseen_root_probability': unseen_p, 'evaluation_metadata_used_for_prediction': False})
        if len(posterior_log) % 50 == 0: print(json.dumps({'phase': 'replay', 'predictions': len(posterior_log)}), flush=True)
    results = {name: {group: scoring.measure(members, pred) for group, members in parts.items()}
               for name, pred in outputs.items()}
    for group in parts: scoring.require_comparable(*[m[group] for m in results.values()])
    # Cross-evaluate every policy under each SAME predeclared preference metric.
    # Truth only looks up a frozen cost here, after prediction is complete.
    preference_weights = {}
    for mode in policy.MODES:
        weights = {}
        for r in rows:
            costs = decision_log[r['position']][mode]['outcome_costs']; truth = r['truth_action']
            unknown = (policy.UNKNOWN_CHILD if r['_scoring']['aliases'].get(truth) == scoring.SERVICE
                       and policy.UNKNOWN_CHILD in costs else policy.UNKNOWN)
            weights[r['position']] = costs.get(truth, costs[unknown])
        preference_weights[mode] = weights
    preference_scores = {metric: {name: {g: scoring.measure_weighted_loss(part, pred, weights)
        for g, part in parts.items()} for name, pred in outputs.items()}
        for metric, weights in preference_weights.items()}
    for metric in preference_scores.values():
        for group in parts: scoring.require_comparable(*[m[group] for m in metric.values()])
    calibration = {group: policy.calibration_summary(members, posterior_log, aliases_log, entries_log)
                   for group, members in parts.items()}
    for group, members in parts.items():
        valid = [r for r in members if scoring.eligible(r)]
        calibration[group]['estimated_fixed_top1_loss'] = sum(decision_log[r['position']]['fixed']['expected_set_losses'][0]
            if decision_log[r['position']]['fixed']['expected_set_losses'] else 1. for r in valid)/len(valid) if valid else None
        calibration[group]['observed_fixed_top1_loss'] = results['fixed'][group]['decision_loss']['mean']
    changes = {name: {group: scoring.paired_topk(members, outputs['fixed'], pred)
                      for group, members in parts.items()} for name, pred in outputs.items()}
    changed = {name: [r['position'] for r in rows if scoring.eligible(r) and
                     outputs['fixed'][r['position']]['predictions'][:1] != pred[r['position']]['predictions'][:1]]
               for name, pred in outputs.items()}
    summary = {'scope': scoring.report_scope(rows), 'cohort_sizes': scoring.cohort_sizes(parts),
               'results': results, 'paired_vs_fixed': changes, 'top1_changed_positions': changed,
               'calibration': calibration, 'common_preference_scores': preference_scores,
               'existing_prediction_replay_parity': True,
               'model_reselected': False, 'runtime_changed': False, 'fresh_holdout': False,
               'scoring_contract': scoring.contract()}
    write('summary.json', summary); write('causal-audit-private.json', causal_audit)
    for name, payload in [('predictions-private.json.gz', outputs), ('posterior-private.json.gz', posterior_log),
                          ('decision-costs-private.json.gz', decision_log)]:
        with gzip.open(out/name, 'wt') as f: json.dump(payload, f)
    lines = ['# 最小期望损失与频率代价：离线对照', '', summary['scope'], '',
             '固定原学习器与清洗规则。只回放原成熟反馈，逐截点复现原预测后比较三种决策代价。频率是偏好代理，并非测得的找回时间。', '',
             '| 决策 | 接受 Top1 | 接受 Top2 | 接受 Top4 | 精确 Top1 | 精确 Top2 | 精确 Top4 | 固定实际平均损失↓ | 综合排名分↑ |',
             '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for name, scores in results.items():
        m = scores['evidence_total']
        values = [f"{v[f'top{k}_hits']}/{v['n']}" for v in (m, m['exact']) for k in (1, 2, 4)]
        values.append(f"{m['decision_loss']['total']:g}/{m['n']} = {m['decision_loss']['mean']:.4f}")
        values.append(f"{100*m['rank_score']['mean']:.2f}")
        lines.append('| '+name+' | '+' | '.join(values)+' |')
    lines += ['', f'## 分组实际损失（全部方案同一 0/{scoring.FALLBACK_LOSS:g}/1 口径）', '',
              '| 决策 | 旧有证据 | 旧较晚有证据 | 新有证据 | 全部合法 |', '|---|---:|---:|---:|---:|']
    for name, scores in results.items():
        lines.append('| '+name+' | '+' | '.join(f"{scores[g]['decision_loss']['mean']:.4f} ({scores[g]['n']})"
            for g in ('old_evidence', 'late_old_evidence', 'new_evidence', 'all'))+' |')
    lines += ['', '## 相同偏好权重下的交叉评分（84 条合法有证据）', '',
              '每列对所有方案使用完全相同的预测前频率权重，分子为加权实际损失、分母为权重总和。不同列代表不同偏好，不跨列宣称提升。', '',
              '| 决策 | 固定代价↓ | 全局频率偏好↓ | 情景频率偏好↓ |', '|---|---:|---:|---:|']
    for name in results:
        lines.append('| '+name+' | '+' | '.join(f"{preference_scores[m][name]['evidence_total']['mean']:.4f}"
                                                 for m in policy.MODES)+' |')
    lines += ['', '## 概率与结论边界', '',
              '- 服务内分布用成熟具体到达估计；当前起点与相似时间背景的计数以全局服务分布为 Dirichlet 先验收缩，浓度固定 2。未知子页面保留概率。',
              '- 总服务概率仍来自原 hinge 排序学习器，并未校准；另用成熟独立结果的 Beta(1,1) 预测估计新类别质量。这是近似概率模型上的最小风险，不是真实分布下的贝叶斯误差下界。',
              '- 所有频率仅来自成熟、独立的派生导航结果。全局加一平滑；情景频率使用一个全局伪样本，稀疏时收缩。没有根据已看过的评分结果搜索成本系数。',
              '- Top1 为给定概率和代价下的最优单动作；后续菜单项按边际期望损失贪心加入，假定用户可选最合适的一项。它不是 TopK 全局最优保证。',
              '- 原排序到固定风险方案同时改变了服务内分布和菜单策略；比较频率代价的影响，应比较 fixed、global_frequency、context_frequency 三行。',
              '- 全部方案由唯一评分入口按相同合法样本、相同固定损失评分；频率模型不能用自己较便宜的计分器宣称提升。具体未知目标仍保留为错误，不删除。',
              '- 未重新选择学习模型、未部署、未变更采集。既有数据已多轮查看，不能作为全新独立测试。', '']
    (out/'report-private.md').write_text('\n'.join(lines))
    for p in sources: assert verify(p) == hashes[str(p)]
    write('run.json', {'complete': True, 'created_at': datetime.now(timezone.utc).isoformat(),
        'source': str(source), 'dataset': str(dataset), 'source_hashes': hashes,
        'raw_modified': False, 'runtime_changed': False, 'scoring_contract': scoring.contract()})
    write('artifact-hashes.json', {p.name: sha(p) for p in out.iterdir() if p.is_file()})
    assert all(p.stat().st_mode & 0o777 == 0o600 for p in out.iterdir() if p.is_file())
    print(json.dumps({'output': str(out), 'evidence': {name: m['evidence_total'] for name, m in results.items()}}), flush=True)


if __name__ == '__main__': main()
