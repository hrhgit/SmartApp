"""Independently re-score the fresh comparison and write a readable summary."""
import argparse
from collections import Counter
from datetime import datetime
import os
from pathlib import Path
from zoneinfo import ZoneInfo

import evaluation_contract as scoring
from review_historical_models import ROOT, Sources, sha, write


REPRESENTATIVES = [
    ('当前线性Top2（固定参考种子）','joint/linear_top2_lr0.1_seed20260923'),
    ('非线性Top2（同一参考种子）','joint/nonlinear_top2_lr0.1_seed20260923'),
    ('平滑Top2（同一参考种子）','joint/linear_smooth_top2_lr0.1_seed20260923'),
    ('早期统计排名融合','legacy/stat_rank_fusion_0_0.5'),
    ('早期条件频次＋最近访问','legacy/stat_count_recency_rank_0.5'),
    ('早期自适应树','legacy/online-adaptive_hoeffding_tree-0.01-10-mc-20260923'),
    ('上下文近邻（当前约束适配）','early/knn_state'),
    ('KMeans后台状态聚类','early/kmeans_state'),
    ('选择性规则','early/selective_rule'),
    ('选择性相似证据','early/selective_evidence'),
    ('纯统计：仅当前','statistics/none'),
    ('纯统计：时间背景','statistics/time'),
    ('纯统计：后台状态','statistics/inventory'),
    ('多背景联合softmax：纯学习','background/interaction_softmax_combined_rich_lr0.1_pure'),
    ('后台状态softmax：纯学习','background/interaction_softmax_combined_rich_lr0.1_inventory_only_pure'),
    ('时间统计＋时间学习固定50/50','fusion/learning_0.5'),
    ('语义softmax：lr0.3','semantic_classifier/03_single'),
    ('分层应用辅助头：频次、权重0.25','hierarchical/frequency_0.25'),
    ('风险决策：统一固定损失','risk/fixed'),
    ('风险决策：全局频率成本','risk/global_frequency'),
    ('风险决策：情景频率成本','risk/context_frequency'),
]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();run=args.run.resolve();out=args.output.resolve()
    if out.exists() or out==ROOT.resolve() or not out.is_relative_to(ROOT.resolve()): parser.error('Use a new private run')
    os.umask(0o077);out.mkdir(mode=0o700);sources=Sources()
    meta=sources.read(run/'run.json')
    if not meta['complete'] or meta['scoring_contract']!=scoring.contract(): raise ValueError('Scorer changed or incomplete run')
    for name in sources.read(run/'artifact-hashes.json'): sources.check(run/name)
    data=sources.read(Path(meta['dataset'])/'dataset-private.json.gz')
    scope=sources.read(Path(meta['dataset'])/'summary.json')
    rows=scoring.prepare_evaluation(data,out);parts=scoring.cohorts(rows)
    for grade in ('single','repeated','strong'):
        parts['evidence_'+grade]=[r for r in parts['evidence_total'] if
            ('single' if r['evidence_count']==1 else 'repeated' if r['evidence_count']<5 else 'strong')==grade]
    outputs={name:{int(p):v for p,v in values.items()} for name,values in sources.read(run/'predictions-private.json.gz').items()}
    metrics=sources.read(run/'metrics-private.json');models=sources.read(run/'models.json')
    weights={mode:{int(p):w for p,w in values.items()} for mode,values in sources.read(run/'common-frequency-weights-private.json').items()}
    cross=sources.read(run/'cross-costs-private.json')
    for name, predictions in outputs.items():
        if set(predictions)!={r['position'] for r in rows}: raise ValueError('Partial model')
        for group, members in parts.items():
            if scoring.measure(members,predictions)!=metrics[name][group]: raise ValueError('Score mismatch '+name)
            for mode,value in weights.items():
                if scoring.measure_weighted_loss(members,predictions,value)!=cross[name][mode][group]: raise ValueError('Cost mismatch '+name)
    for group in parts: scoring.require_comparable(*[value[group] for value in metrics.values()])
    en=metrics[next(iter(metrics))]['evidence_total']['n'];an=metrics[next(iter(metrics))]['all']['n']
    boundary=scope['boundary_event_index']
    if not all(data['observations'][r['position']]['index']>=boundary for r in rows): raise ValueError('Old member')
    outcomes={r['position']:r['record']['outcome_id'] for r in data['reference_stream']}
    distinct=len({outcomes[r['position']] for r in parts['evidence_total'] if scoring.eligible(r)})
    dates=[datetime.fromisoformat(value).astimezone(ZoneInfo('America/Los_Angeles')).strftime('%Y-%m-%d %H:%M:%S %Z') for value in scope['validation_range_utc']]
    def fraction(m,k):return f"{m['exact']['top'+str(k)+'_hits']}/{m['n']}"
    def accepted(m,k):return f"{m['top'+str(k)+'_hits']}/{m['n']}"
    lines=['# 历史模型统一新验证集复测', '',
        f'**有证据主评测：{en}条，覆盖{en}/{an}＝{en/an:.1%}；对应{distinct}个不同后续结果事件。** 全部模型另接受同一{an}条合法样本的全量诊断。', '',
        f'新验证区间：{dates[0]} 至 {dates[1]}。所有预测截点位于上一次评测快照之后；旧84条有证据样本没有混进本次分母。', '',
        f'共 **{len(metrics)}组模型／配置**。这是配置数，不是独立算法数量；包含预声明种子、消融和融合比例。全部先在新数据上重训、按时间回放，再由同一个评分入口计分；没有拿旧预测直接重评分冒充新验证。', '',
        '## 代表方案：先看同一有证据集', '',
        '| 方案 | 精确Top1 | 精确Top4 | 服务接受Top4 | 综合分/100 | 全42条精确Top1 | 全42条精确Top4 |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for label,name in REPRESENTATIVES:
        m=metrics[name]['evidence_total'];a=metrics[name]['all']
        lines.append(f'| {label} | {fraction(m,1)} | {fraction(m,4)} | {accepted(m,4)} | {100*m["rank_score"]["mean"]:.1f} | {fraction(a,1)} | {fraction(a,4)} |')
    linear=metrics['joint/linear_top2_lr0.1_seed20260923']
    nonlinear=metrics['joint/nonlinear_top2_lr0.1_seed20260923']
    old=metrics['legacy/stat_rank_fusion_0_0.5']
    lines+=['','## 哪些值得重新考虑','',
        f'- **非线性Top2值得继续验证。** 固定参考种子下，有证据精确Top4为{fraction(nonlinear["evidence_total"],4)}，线性为{fraction(linear["evidence_total"],4)}；但Top1为{fraction(nonlinear["evidence_total"],1)}，低于线性的{fraction(linear["evidence_total"],1)}。这是Top4覆盖与第一位排序的取舍，不能只报最好的一列。',
        f'- **旧统计排名融合有局部价值，但不足以整体替代。** 有证据精确Top1为{fraction(old["evidence_total"],1)}；全量精确Top4却只有{fraction(old["all"],4)}，线性Top2为{fraction(linear["all"],4)}。只盯有证据小集会漏掉泛化差异。',
        '- **统计基础仍值得保留。** 当前／时间／后台纯统计在这10条主评测上成绩相同；纯多背景学习没有展示全面替代统计基础的优势。这不等于证明后台采集无用。',
        '- **选择性规则和相似证据本轮没有得到复活依据。** 空输出按未命中计入统一分母；不能把拒绝预测的样本移除后称其高准确率。',
        '- 这些是本轮结果的探索性解读，不是新一轮选参或上线决策。',
        '','## 随机种子完整对照','','同一学习率0.1，以下三个种子事先固定，不用新验证集挑选种子。', '',
        '| 架构 | 种子 | 有证据精确Top1 | 有证据精确Top4 | 全量精确Top1 | 全量精确Top4 |','|---|---:|---:|---:|---:|---:|']
    for family in ('linear_top2','linear_smooth_top2','nonlinear_top2'):
        for seed in (20260923,20260924,20260925):
            v=metrics[f'joint/{family}_lr0.1_seed{seed}'];m=v['evidence_total'];a=v['all']
            lines.append(f'| {family} | {seed} | {fraction(m,1)} | {fraction(m,4)} | {fraction(a,1)} | {fraction(a,4)} |')
    lines+=['','## 证据强弱：固定参考种子','','| 证据范围 | 线性Top2精确Top1/Top4 | 非线性Top2精确Top1/Top4 |','|---|---:|---:|']
    for group,label in [('evidence_single','一次历史支持'),('evidence_repeated','2–4次支持'),('evidence_strong','至少5次支持'),('expanded_total','含较弱语义支持')]:
        a=metrics['joint/linear_top2_lr0.1_seed20260923'][group];b=metrics['joint/nonlinear_top2_lr0.1_seed20260923'][group]
        lines.append(f'| {label} | {fraction(a,1)}、{fraction(a,4)} | {fraction(b,1)}、{fraction(b,4)} |')
    lines+=['','## 配对变化：相对当前线性Top2','','每行都比较同一条样本，不用不同分母相减。这里是统一服务接受口径；精确成绩见上表。','','| 对照 | Top1修正/损失 | Top4修正/损失 |','|---|---:|---:|']
    for label,name in REPRESENTATIVES:
        pair=scoring.paired_topk(parts['evidence_total'],outputs['joint/linear_top2_lr0.1_seed20260923'],outputs[name])
        lines.append(f'| {label} | {pair["top1"]["helped"]}/{pair["top1"]["hurt"]} | {pair["top4"]["helped"]}/{pair["top4"]["hurt"]} |')
    lines+=['','## 统一原则与解释边界','',
        '- 使用同一hard_kind训练清洗、同一稳定停留评分标签、同一预测截点和同一合法性审查；43条待审样本统一排除1条无可评分目标变化的样本。不是靠换清洗或分母赢分。',
        '- 前台序列长度1；时间背景是无序统计，后台状态按方案可选。不以当前打开库存或同起点支持集硬性限定输出。早期方案因此属于明确适配复测。',
        '- 所有在线更新只读当前截点前已经成熟的标签；允许验证期较早的成熟行为更新较晚预测。不是整段完全禁止更新的静态holdout，也没有用新验证成绩选参。KMeans只在验证期之前的输入上拟合中心。',
        '- 多步头是显式监督因素，和其他模型接受同一个下一步评分。应用辅助头与目的地头是同一次行为的非互斥标签，不当成两次独立到达。',
        '- 有证据分组使用真实后续目标定义，但历史证据必须在预测前已成熟；它是事后诊断集，不是线上可识别的90%置信保证。',
        '- 精确收益1、正确服务首页回退0.6、错误0。综合分为Top1到Top4前缀最佳收益的平均，不重复累计同一需求。普通窗口输出所属应用时，精确指约定动作标签，不代表精确窗口命中。',
        '- 风险策略分数不是经过校准的可靠后验。固定、全局频率、情景频率损失对所有模型交叉评分，不能只挑对某模型有利的计分标准。',
        '- 选择性规则和证据模型的空输出计为未命中，仍保留在共同分母；不是只在它们愿意推荐的子集上比较。',
        f'- 当前仅{en}条有证据样本，一条就是{100/en:.0f}个百分点。最多支持“值得继续验证”，不能据此更换线上模型或宣布长期冠军。已查看结果后的分析也不能继续称独立选模证据。',
        '- 清洗变体、有序路径、候选硬限制不作为额外模型混排；统一改成当前约束后复测相应模型家族。不是声称逐个复刻全部历史超参数与旧数据协议。',
        '- 未部署或修改运行中的SmartApp；未更改采集设置及原始数据库。', '',
        f'完整{len(metrics)}组成绩、统一频率成本交叉评分：[完整表](<{run}/report-private.md>)。',
        f'新训练模型及复测产物：[运行清单](<{run}/run.json>)。', '']
    (out/'summary-private.md').write_text('\n'.join(lines))
    sources.verify_unchanged()
    audit={'complete':True,'same_new_validation_for_all':True,'models':len(metrics),'legal_n':an,'evidence_n':en,
        'evidence_distinct_outcomes':distinct,'independent_score_recomputation':True,'all_artifact_hashes_verified':True,
        'boundary_event_index':boundary,'family_configuration_counts':dict(Counter(v['family'] for v in models.values())),
        'scoring_contract':scoring.contract(),'source_run':str(run),'sources':sources.accessed}
    write(out/'verification.json',audit)
    (out/('source-'+Path(__file__).name)).write_bytes(Path(__file__).read_bytes())
    for path in out.iterdir():
        if path.is_file():path.chmod(0o600)
    write(out/'artifact-hashes.json',{p.name:sha(p) for p in out.iterdir() if p.is_file()})
    print({k:audit[k] for k in ('models','legal_n','evidence_n','evidence_distinct_outcomes','independent_score_recomputation')},flush=True)


if __name__=='__main__':main()
