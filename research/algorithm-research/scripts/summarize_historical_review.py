"""Readable comparison, fixed-seed ranges and shared-cost cross scoring."""
import argparse
from collections import defaultdict
from datetime import datetime, timezone
import importlib.metadata
import json
import os
from pathlib import Path

import evaluation_contract as scoring
import bayes_frequency_policy as policy
from review_historical_models import ROOT, DATASET, REFERENCE, Sources, sha, write


JOINT = 'joint-ranking-alternatives-v1-20260923-experiment-01/'
STAT = 'context-statistics-v1-20260923-tuning-02/'
MODERN = 'modern_replay/hard_kind/'
FUSION = 'fusion-stat_recency_global--online-adaptive_hoeffding_tree-0.01-100-mc-20260923--0-0.5'
DISPLAY = [
    ('固定风险决策', REFERENCE+'/fixed'),
    ('既定线性Top2修正／原排序', REFERENCE+'/existing_order'),
    ('平滑Top2修正（lr0.1，固定参考种子）', JOINT+'linear_smooth_top2_lr0.1_seed20260923'),
    ('交叉熵整体排序（固定参考种子）', JOINT+'linear_ce_lr0.1_seed20260923'),
    ('情景门控（历史所选，固定参考种子）', JOINT+'context_gate_ce_lr0.03_seed20260923'),
    ('非线性Top2（历史所选，固定参考种子）', JOINT+'nonlinear_top2_lr0.03_seed20260923'),
    ('仅当前目标统计', STAT+'none'),
    ('时间背景统计', STAT+'time'),
    ('后台状态统计', STAT+'inventory'),
    ('条件频次＋历史最近访问（共同清洗）', MODERN+'stat_count_candidate_recency_rank_0.5'),
    ('条件频次＋历史最近访问（resolved清洗）', 'modern_replay/resolved/stat_count_candidate_recency_rank_0.5'),
    ('旧近期访问＋自适应树融合（取消硬限制）', MODERN+FUSION),
    ('贝叶斯平滑统计', MODERN+'stat_bayesian_smoothing_global_decay_6h_30'),
    ('纯近期访问', MODERN+'stat_recency_global'),
    ('流式KNN（共同输入与标签）', MODERN+'online-streaming_knn-20-200'),
    ('在线聚类（共同输入与标签）', MODERN+'online-online_cluster_frequency-0.5-16-20260923'),
    ('独立在线Softmax', MODERN+'online-softmax-0.0-0.1'),
    ('独立MLP', MODERN+'online-candidate_mlp-32-0.01-2048-8-20260923'),
    ('情景频率代价决策', REFERENCE+'/context_frequency'),
    ('全局频率代价决策', REFERENCE+'/global_frequency'),
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    source, out = args.source.resolve(), args.output.resolve()
    if out.exists() or out == ROOT.resolve() or not out.is_relative_to(ROOT.resolve()):
        parser.error('Use a new private training run')
    sources = Sources()
    run = sources.read(source/'run.json')
    if not run['complete'] or run['scoring_contract'] != scoring.contract():
        raise ValueError('Review is incomplete or scoring changed')
    metrics = sources.read(source/'metrics-private.json')
    models = sources.read(source/'models.json')
    predictions = sources.read(source/'predictions-private.json.gz')
    paired = sources.read(source/'paired-private.json')
    data = sources.read(ROOT/DATASET/'dataset-private.json.gz')
    costs = sources.read(ROOT/REFERENCE/'decision-costs-private.json.gz')
    old_summary = sources.read(ROOT/REFERENCE/'summary.json')
    os.umask(0o077); out.mkdir(mode=0o700)
    rows = scoring.prepare_evaluation(data, out)
    weights = {}
    for mode in policy.MODES:
        weights[mode] = {}
        for row in rows:
            values = costs[str(row['position'])][mode]['outcome_costs']
            truth = row['truth_action']
            unknown = (policy.UNKNOWN_CHILD if row['_scoring']['aliases'].get(truth) == scoring.SERVICE
                       and policy.UNKNOWN_CHILD in values else policy.UNKNOWN)
            weights[mode][row['position']] = values.get(truth, values[unknown])
    cross = {}
    for name, pred in predictions.items():
        common = [r for r in rows if str(r['position']) in pred]
        parts = scoring.cohorts(common)
        cross[name] = {mode: {g: scoring.measure_weighted_loss(part, pred, w)
                             for g, part in parts.items()} for mode, w in weights.items()}
    for mode in policy.MODES:
        for name in ('existing_order', 'fixed', 'global_frequency', 'context_frequency'):
            for group, current in cross[REFERENCE+'/'+name][mode].items():
                expected = old_summary['common_preference_scores'][mode][name][group]
                if current['mean'] != expected['mean'] or current['n'] != expected['n']:
                    raise ValueError('Shared preference reference parity failed')
        full = [n for n, m in models.items() if m['full_common_dataset']]
        for group in scoring.cohorts(rows):
            scoring.require_comparable(*[cross[n][mode][group] for n in full])
    write(out/'common-preference-cross-scores-private.json', cross)
    write(out/'common-preference-weights-private.json', weights)

    seeds = defaultdict(list)
    for name in models:
        if name.startswith(JOINT) and '_seed' in name:
            group, seed = name.removeprefix(JOINT).rsplit('_seed', 1)
            m = metrics[name]['evidence_total']
            seeds[group].append({'seed': int(seed), 'top1': m['exact']['top1_hits'],
                                'top2': m['exact']['top2_hits'], 'top4': m['exact']['top4_hits'],
                                'rank_score': 100*m['rank_score']['mean']})
    for values in seeds.values():
        if sorted(x['seed'] for x in values) != [20260923, 20260924, 20260925]:
            raise ValueError('Seed group is incomplete')
    write(out/'seed-ranges.json', seeds)

    def cell(m, k):
        return f"{m['exact']['top'+str(k)+'_hits']}/{m['n']}"
    def score(m):
        return f"{100*m['rank_score']['mean']:.2f}"
    def report_table(display, group='evidence_total'):
        result = ['| 方案 | 精确Top1 | 精确Top2 | 精确Top4 | 接受Top4 | 综合分 | 返回覆盖 |',
                  '|---|---:|---:|---:|---:|---:|---:|']
        for title, name in display:
            m = metrics[name][group]
            result.append('| '+title+' | '+' | '.join([cell(m,k) for k in (1,2,4)]+
                [f"{m['top4_hits']}/{m['n']}",score(m),f"{m['prediction_coverage']:.1%}"])+' |')
        return result
    display = list(DISPLAY)
    multi = 'multibackground-v1-20260923-experiment-01'
    selection = sources.read(ROOT/multi/'selection.json')
    for group, result in selection['selected_by_background'].items():
        display.append(('多背景学习／'+group+'（历史所选）', multi+'/'+result['config']['id']))
    lines = ['# 历史方案按当前标准重新比较', '', scoring.report_scope(rows), '',
        f"本轮实际重评 **{len(models)}组模型／配置输出**：其中 **37次从头因果回放**（27个旧基础家族、额外基线／融合与清洗对照），其余237组复用冻结预测重新评分。**{run['full_dataset_models']}组覆盖全部234个原评测截点**，经统一审查后在228条合法样本、其中84条主证据样本上比较；88组更早记录在旧样本／共有子集单列。配置数不是独立算法数，也不是独立验证次数。", '',
        '评分统一为精确命中收益1、正确首页回退0.6、错误0。综合分＝Top1至Top4每个前缀最佳收益的平均×100。先比较精确Top4，同时看Top1与位置收益。服务接受TopK仍另列，不能当成精确会话命中。', '',
        '## 代表方案：全部84条有证据样本', '',
        '联合排序采用预先固定参考种子20260923，其他种子在后表完整报告。代表表用于覆盖研究路线，不是按本轮最好随机种子拼接的赢家榜。resolved属于独立训练清洗对照，评分成员和答案不变。', '']
    lines += report_table(display)
    lines += ['', '## 当前比较得出的判断', '',
        '- 固定风险决策的精确Top4为80/84，在本次186组完整覆盖输出中最高；它在主证据集领先，不能据此宣称所有时刻或未来数据全面领先。',
        '- 简单的“条件频次＋历史最近访问”值得保留：共同清洗下精确Top1/Top4为59/84、77/84；resolved清洗为56/84、78/84。相对既定线性Top2的57/84、76/84，覆盖有竞争力，但综合位置收益较低。',
        '- 旧96.8%方案仍有历史强项：原冻结输出在合法旧证据集为58/60；共同条件下取消库存／同起点硬限制、统一导航标签的适配融合为55/60，全部84条为76/84。这里同时发生多处适配，不能把差异全部归因于去掉某一个限制。',
        '- 平滑Top2修正仍值得对照：参考种子精确58/84、77/84，综合87.08；既定线性Top2为57/84、76/84，综合86.49。改善很小，不能宣布已证明优胜。',
        '- KNN、聚类、独立Softmax和独立MLP在共同输入／标签回放下仍未超过强统计与情景修正。当前证据不支持直接恢复这些旧模型。',
        '- 时间背景统计与仅当前统计各有取舍；后台状态统计此配置的结果与仅当前相同。特征价值仍依赖表示与学习方式，本轮不能推出停止后台采集。', '',
        '## 新旧时段及较晚样本', '',
        '| 方案 | 旧证据Top1/Top4（60） | 新增证据Top1/Top4（24） | 去除早期验证Top1/Top4（60） | 较晚综合分 | 全部合法综合分（228） |',
        '|---|---:|---:|---:|---:|---:|']
    for title, name in display:
        r = metrics[name]
        pairs = [f"{r[g]['exact']['top1_hits']} / {r[g]['exact']['top4_hits']}" for g in ('old_evidence','new_evidence','test_evidence')]
        lines.append('| '+title+' | '+' | '.join(pairs+[score(r['test_evidence']),score(r['all'])])+' |')
    lines += ['', '## 随机种子稳定性：全部三次均列入', '',
        '| 配置 | 精确Top1范围/84 | 精确Top2范围/84 | 精确Top4范围/84 | 综合分范围 |',
        '|---|---:|---:|---:|---:|']
    for name, values in seeds.items():
        ranges = [f"{min(v[k] for v in values)}–{max(v[k] for v in values)}" for k in ('top1','top2','top4')]
        ranges += [f"{min(v['rank_score'] for v in values):.2f}–{max(v['rank_score'] for v in values):.2f}"]
        lines.append('| '+name+' | '+' | '.join(ranges)+' |')
    lines += ['', '## 相同频率偏好下交叉计分', '',
        '下表均为Top1实际平均损失，越低越好；每列所有模型使用同一截点前历史权重。固定损失、全局频率偏好、情景频率偏好是三种不同评价目标，不跨列比较。频率不是测得的找回时间。', '',
        '| 方案 | 固定损失 | 全局频率加权损失 | 情景频率加权损失 |', '|---|---:|---:|---:|']
    for title, name in display:
        lines.append('| '+title+' | '+' | '.join(f"{cross[name][mode]['evidence_total']['mean']:.4f}" for mode in policy.MODES)+' |')
    lines += ['', '## 与既定线性Top2的逐样本比较', '',
        '下面是精确Top4修正/损失；净多一条不代表只改变了一条。', '',
        '| 方案 | 新增命中 | 丢失命中 | 净命中变化 |', '|---|---:|---:|---:|']
    for title, name in display:
        p = paired[name]['exact']['4']
        lines.append(f"| {title} | {p['helped']} | {p['hurt']} | {p['net']:+d} |")
    lines += ['', '## 完整记录与限制', '',
        f'- [274组完整分数与早期模型共有子集对照](<{source}/report-private.md>)',
        f'- [每模型新旧时段与证据强弱明细](<{source}/metrics-private.json>)',
        f'- [37次回放预先冻结的配置与适配规则](<{source}/replay-plan.json>)',
        '- 部分旧模型使用不同训练标签、库存硬限制或较长历史路径，保留为历史诊断；现代适配回放统一长度1当前输入、成熟反馈、导航动作和无限制历史输出词表。两者分开解释。',
        '- 历史窗口预测仅按已知所属应用映射为共同动作标签，保留原槽位，不去重、不补位。没有把普通窗口应用标签命中宣称为具体窗口实例命中。',
        '- 所有旧数据都已用于开发。较晚时段、多个种子以及重评分不能替代冻结方案后的全新自然行为验证。',
        '- 未重新调参、未部署、未修改原始数据或旧run。37次回放模型均核验重载和继续学习一致；评分代码、样本成员、标签与源文件校验值均留档。', '']
    (out/'report-private.md').write_text('\n'.join(lines))
    sources.verify_unchanged()
    for name in ('summarize_historical_review.py', 'review_historical_models.py', 'bayes_frequency_policy.py'):
        (out/name).write_bytes(Path(__file__).with_name(name).read_bytes())
    write(out/'run.json', {'complete': True, 'created_at': datetime.now(timezone.utc).isoformat(),
        'source': str(source), 'source_files': sources.accessed, 'shared_preference_reference_parity': True,
        'scoring_contract': scoring.contract(), 'models': len(models), 'runtime_changed': False})
    write(out/'artifact-hashes.json', {p.name: sha(p) for p in out.iterdir() if p.is_file()})
    assert all(p.stat().st_mode & 0o777 == 0o600 for p in out.iterdir() if p.is_file())
    print(json.dumps({'complete': True, 'output': str(out), 'models': len(models)}))


if __name__ == '__main__':
    main()
