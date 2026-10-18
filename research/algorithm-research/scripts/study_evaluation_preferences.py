"""Re-score frozen policies together and describe sensitivity to shared preferences.

This is an evaluation study, not a policy/hyperparameter selection experiment.
All label, membership, TopK and realized-loss semantics live in evaluation_contract.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import gzip
from itertools import combinations
import json
import math
import os
from pathlib import Path

import evaluation_contract as scoring
from bayes_frequency_policy import MODES, UNKNOWN, UNKNOWN_CHILD
from rescore_predictions import sha, verify


def lower_envelope(start, end):
    """Exact winner intervals for convex mixtures of two fixed metric columns."""
    if start.keys() != end.keys() or not start:
        raise ValueError('Metric columns need the same nonempty policy set')
    cuts = {0., 1.}
    for a, b in combinations(start, 2):
        difference = start[a] - start[b]
        slope = end[a] - end[b] - difference
        if slope:
            crossing = -difference / slope
            if 0 < crossing < 1:
                cuts.add(crossing)
    segments = []
    cuts = sorted(cuts)
    for left, right in zip(cuts, cuts[1:]):
        midpoint = (left + right) / 2
        values = {name: (1-midpoint)*start[name] + midpoint*end[name] for name in start}
        best = min(values.values())
        winners = sorted(name for name, value in values.items() if math.isclose(value, best, abs_tol=1e-12))
        if segments and segments[-1]['winners'] == winners:
            segments[-1]['through'] = right
        else:
            segments.append({'from': left, 'through': right, 'winners': winners})
    return segments


def nondominated(matrix):
    """Pareto set over these declared cost columns only, not over TopK metrics."""
    return [name for name, values in matrix.items() if not any(
        other != name and all(a <= b for a, b in zip(alternative, values))
        and any(a < b for a, b in zip(alternative, values))
        for other, alternative in matrix.items())]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    source, out = args.source.resolve(), args.output.resolve()
    root = (Path.home()/'Library/Application Support/SmartApp/training').resolve()
    if out.exists() or out == root or not out.is_relative_to(root):
        parser.error('Use a new private training run')
    source_hashes = verify(source)
    meta = json.loads((source/'run.json').read_text())
    dataset = Path(meta['dataset']); dataset_hashes = verify(dataset)
    previous = json.loads((source/'summary.json').read_text())
    def read_gzip(path):
        with gzip.open(path, 'rt') as stream:
            return json.load(stream)
    data = read_gzip(dataset/'dataset-private.json.gz')
    predictions = read_gzip(source/'predictions-private.json.gz')
    costs = read_gzip(source/'decision-costs-private.json.gz')
    os.umask(0o077); out.mkdir(mode=0o700)
    def write(name, value):
        (out/name).write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n')
    write('plan.json', {
        'source_contract_matches_current': previous['scoring_contract'] == scoring.contract(),
        'prediction_generating_contract': previous['scoring_contract'], 'primary_cohort': 'evidence_total',
        'secondary_scopes': 'all canonical cohorts, including all legal samples',
        'cost_columns': list(MODES),
        'sensitivity': '(1-alpha)*fixed_metric + alpha*frequency_metric; frozen predictions',
        'alpha_meaning': 'evaluation preference only; neither training nor policy blend',
        'normalization': 'each endpoint metric has its own shared weight-sum denominator',
        'robustness': 'paired leave-one-session-out; descriptive, not confidence intervals',
        'no_model_selection': True, 'fresh_holdout': False,
        'source': str(source), 'source_hashes': source_hashes,
        'dataset': str(dataset), 'dataset_hashes': dataset_hashes,
        'scoring_contract': scoring.contract()})
    rows = scoring.prepare_evaluation(data, out); cohorts = scoring.cohorts(rows)
    observations = {r['position']: r for r in data['observations']}
    weights = {mode: {} for mode in MODES}
    for r in rows:
        for mode in MODES:
            frozen = costs[str(r['position'])][mode]['outcome_costs']
            truth = r['truth_action']
            unknown = (UNKNOWN_CHILD if r['_scoring']['aliases'].get(truth) == scoring.SERVICE
                       and UNKNOWN_CHILD in frozen else UNKNOWN)
            weights[mode][r['position']] = frozen.get(truth, frozen[unknown])
    results = {name: {group: scoring.measure(part, output) for group, part in cohorts.items()}
               for name, output in predictions.items()}
    common = {mode: {name: {group: scoring.measure_weighted_loss(part, output, weights[mode])
                           for group, part in cohorts.items()} for name, output in predictions.items()}
              for mode in MODES}
    for group in cohorts:
        scoring.require_comparable(*[result[group] for result in results.values()])
        for mode in MODES:
            scoring.require_comparable(*[result[group] for result in common[mode].values()])
    # Recompute through the current scorer; the previous report is only a parity check.
    parity = (previous['results'] == results and previous['common_preference_scores'] == common)
    if previous['scoring_contract'] == scoring.contract() and not parity:
        raise AssertionError('Unexplained difference from source scores')
    studies = {}
    for group, part in cohorts.items():
        valid = [r for r in part if scoring.eligible(r)]
        positions = [r['position'] for r in valid]
        sessions = Counter(observations[pos]['session'] for pos in positions)
        matrix = {name: {mode: common[mode][name][group]['mean'] for mode in MODES}
                  for name in predictions}
        concentrations = {}; leave_out = {}
        for mode in MODES:
            mass = sum(weights[mode][p] for p in positions)
            normalized = sorted((weights[mode][p]/mass for p in positions), reverse=True)
            concentrations[mode] = {
                'weight_concentration_equivalent_n': 1/sum(q*q for q in normalized),
                'not_independent_sample_size': True,
                'largest_row_share': normalized[0], 'ten_largest_share': sum(normalized[:10])}
            paired = {name: [] for name in predictions if name != 'fixed'}
            for session in sessions:
                remaining = [r for r in valid if observations[r['position']]['session'] != session]
                if not remaining:
                    continue
                scores = {name: scoring.measure_weighted_loss(remaining, pred, weights[mode])
                          for name, pred in predictions.items()}
                scoring.require_comparable(*scores.values())
                for name in paired:
                    paired[name].append(scores[name]['mean']-scores['fixed']['mean'])
            leave_out[mode] = {name: {'deltas_vs_fixed': deltas,
                'minimum': min(deltas) if deltas else None,
                'maximum': max(deltas) if deltas else None} for name, deltas in paired.items()}
        curves = {}
        for mode in ('global_frequency', 'context_frequency'):
            start = {name: matrix[name]['fixed'] for name in predictions}
            end = {name: matrix[name][mode] for name in predictions}
            curves[mode] = {'best_intervals': lower_envelope(start, end)}
            # Validate curve arithmetic against the SAME evaluator with combined weights.
            mass = sum(weights[mode][p] for p in positions)
            for alpha in (.25, .5, .75):
                mixed = {p: (1-alpha)/len(positions)+alpha*weights[mode][p]/mass for p in positions}
                scores = [scoring.measure_weighted_loss(valid, pred, mixed) for pred in predictions.values()]
                scoring.require_comparable(*scores)
                for name, score in zip(predictions, scores):
                    assert math.isclose(score['mean'], (1-alpha)*start[name]+alpha*end[name], abs_tol=1e-12)
        studies[group] = {'n': len(valid), 'session_n': len(sessions),
            'session_sizes': sorted(sessions.values(), reverse=True), 'matrix': matrix,
            'nondominated_on_three_costs_only': nondominated({name: list(m.values()) for name, m in matrix.items()}),
            'concentrations': concentrations, 'paired_leave_one_session_out': leave_out,
            'preference_sensitivity': curves}
    summary = {'scope': scoring.report_scope(rows), 'cohort_sizes': scoring.cohort_sizes(cohorts),
        'results': results, 'common_preference_scores': common, 'studies': studies,
        'source_score_parity': parity, 'scoring_contract': scoring.contract(),
        'trained': False, 'predictions_changed': False, 'model_selected': False, 'runtime_changed': False}
    write('summary.json', summary)
    lines = ['# 统一评价方案：固定预测的偏好敏感性研究', '', summary['scope'], '',
        f'全部历史预测按当前唯一评分入口重算，固定实际损失为 0/{scoring.FALLBACK_LOSS:g}/1。未训练、更换模型或重新运行决策，原预测生成时的代价见 plan.json。', '',
        f'综合排名分为 Top1 至 Top{scoring.RANK_CUTOFF} 最佳匹配收益的平均乘 100；精确收益 1，正确首页回退 {scoring.FALLBACK_GAIN:g}，错误 0，每个前缀只取最佳收益。', '',
        '## 已约定有证据成绩', '',
        '| 策略 | 接受 Top1/2/4 | 精确 Top1/2/4 | 固定实际损失↓ | 综合排名分↑ |', '|---|---|---|---:|---:|']
    for name, scores in results.items():
        m = scores['evidence_total']
        cells = [' / '.join(str(level[f'top{k}_hits']) for k in (1, 2, 4))+'（分母 '+str(m['n'])+'）'
                 for level in (m, m['exact'])]
        lines.append(f"| {name} | {' | '.join(cells)} | {m['decision_loss']['mean']:.4f} | {m['rank_score']['total']:g}/{m['n']} × 100 = {100*m['rank_score']['mean']:.2f} |")
    for group in ('evidence_total', 'all'):
        study = studies[group]
        lines += ['', f"## {group}：{study['n']} 条，{study['session_n']} 个会话", '',
            '| 策略 | 固定评价↓ | 全局频率评价↓ | 情景频率评价↓ |', '|---|---:|---:|---:|']
        for name, m in study['matrix'].items():
            lines.append('| '+name+' | '+' | '.join(f'{m[mode]:.4f}' for mode in MODES)+' |')
        lines += ['', '每列对所有策略采用完全相同的权重及分母。不同列回答不同偏好问题。', '',
            '在这三个实际损失指标上非劣的策略：'+', '.join(study['nondominated_on_three_costs_only'])+'。这不包含 Top2/Top4 的取舍。', '',
            '敏感性只混合两列评价分数：Rα=(1−α)R固定+αR频率；不混合模型、概率或重新生成菜单。α 不是推荐值，也不按区间长度选赢家。']
        for mode, curve in study['preference_sensitivity'].items():
            lines.append('- '+mode+'：'+'；'.join(f"α ∈ ({v['from']:.6f}, {v['through']:.6f}) 时 {', '.join(v['winners'])} 最低"
                for v in curve['best_intervals'])+'；交界处并列。')
        lines += ['', '权重集中度等效条数（不是独立样本量、也不能作为置信区间分母）：'+
            '，'.join(f"{mode}={v['weight_concentration_equivalent_n']:.1f}" for mode, v in study['concentrations'].items())+'。', '',
            '| 同一评价下相对 fixed 的损失差 | 全样本 | 逐一去掉一个会话后的范围 |', '|---|---:|---:|']
        for mode in ('global_frequency', 'context_frequency'):
            v = study['paired_leave_one_session_out'][mode][mode]
            delta = study['matrix'][mode][mode]-study['matrix']['fixed'][mode]
            lines.append(f"| {mode} | {delta:+.4f} | {v['minimum']:+.4f} 至 {v['maximum']:+.4f} |")
    lines += ['', '## 解释限制', '',
        '- 胜负翻转表明目标偏好不同，不能由这些分数倒推出哪种偏好才正确。',
        '- 原输出到 fixed 同时改变了层级决策和菜单；三种风险策略才隔离了频率代价的影响。',
        '- 固定有证据集依赖真实下一步作事后诊断，不等于线上可识别接受集；全部合法集仅作补充，不能替换主分母。',
        '- 全部合法样本与有证据样本的排序不同；不得挑有利样本范围或权重规则宣称胜出。',
        '- 仅 8 个会话且既有样本已被多轮查看。去一会话分析是稳定性诊断，不是独立测试或统计显著性证明。',
        '- 频率权重来自原 run 的公共历史统计和共同结果空间；对不同词表的未来模型需先冻结模型无关的公共频率参考，不能各算各的权重。',
        '- 决策时将同一截点的全部代价乘同一正数不改变选择，但会改变跨截点评价的权重；评价必须固定跨情景单位。',
        '- 三种风险策略共享同一未校准概率分布，Brier/log loss 无法区分其决策优劣。',
        '- TopK 命中与菜单内最小损失不能直接当作实际节省时间；自然到达记录不提供未采取推荐的反事实使用结果。', '',
        '方法与后续评价协议建议见源码区 UNIFIED_EVALUATION.md。']
    (out/'report-private.md').write_text('\n'.join(lines)+'\n')
    for name in ('study_evaluation_preferences.py', 'bayes_frequency_policy.py', 'rescore_predictions.py'):
        (out/name).write_bytes(Path(__file__).with_name(name).read_bytes())
    (out/'UNIFIED_EVALUATION.md').write_bytes(Path(__file__).parents[1].joinpath('UNIFIED_EVALUATION.md').read_bytes())
    assert verify(source) == source_hashes and verify(dataset) == dataset_hashes
    write('run.json', {'complete': True, 'created_at': datetime.now(timezone.utc).isoformat(),
        'source': str(source), 'dataset': str(dataset), 'source_hashes': source_hashes,
        'dataset_hashes': dataset_hashes, 'scoring_contract': scoring.contract(),
        'raw_modified': False, 'runtime_changed': False})
    write('artifact-hashes.json', {p.name: sha(p) for p in out.iterdir() if p.is_file()})
    assert all(p.stat().st_mode & 0o777 == 0o600 for p in out.iterdir() if p.is_file())
    print(json.dumps({'complete': True, 'output': str(out), 'parity': parity,
        'evidence_sensitivity': studies['evidence_total']['preference_sensitivity']}, ensure_ascii=False))


if __name__ == '__main__':
    main()
