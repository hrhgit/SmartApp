"""Verify and summarize exactly two full cohorts using the existing ranked gain."""
import argparse
from collections import Counter
import os
from pathlib import Path

import evaluation_contract as scoring
from review_historical_models import ROOT, Sources, sha, write
from summarize_fresh_comparison import REPRESENTATIVES


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();run=args.run.resolve();out=args.output.resolve()
    if out.exists() or out==ROOT.resolve() or not out.is_relative_to(ROOT.resolve()):parser.error('Use a new private run')
    os.umask(0o077);out.mkdir(mode=0o700);sources=Sources()
    meta=sources.read(run/'run.json')
    if not meta['complete'] or meta['scoring_contract']!=scoring.contract() or meta.get('evaluation_scope')!='complete_two_cohorts':
        raise ValueError('Not a completed two-cohort run using the current scorer')
    for name in sources.read(run/'artifact-hashes.json'):sources.check(run/name)
    dataset=Path(meta['dataset']);data=sources.read(dataset/'dataset-private.json.gz')
    scope=sources.read(dataset/'summary.json');rows=scoring.prepare_evaluation(data,out)
    parts={k:v for k,v in scoring.cohorts(rows).items() if k in ('all','evidence_total')}
    values={name:{int(p):v for p,v in predictions.items()} for name,predictions in sources.read(run/'predictions-private.json.gz').items()}
    metrics=sources.read(run/'metrics-private.json');models=sources.read(run/'models.json')
    expected_positions={r['position'] for r in rows};originals=sources.read(Path(sources.read(dataset/'run.json')['fixed_dataset'])/'dataset-private.json.gz')
    old_ids={r['sample_id'] for r in originals['evaluation']}
    if not old_ids<={r['sample_id'] for r in rows}:raise ValueError('Incomplete fixed validation')
    for name,predictions in values.items():
        if set(predictions)!=expected_positions or set(metrics[name])!=set(parts):raise ValueError('Not exactly two complete cohorts')
        for group,members in parts.items():
            if scoring.measure(members,predictions)!=metrics[name][group]:raise ValueError('Score mismatch: '+name)
    for group in parts:scoring.require_comparable(*[scores[group] for scores in metrics.values()])
    sizes=scoring.cohort_sizes(parts);all_n=sizes['all'];evidence_n=sizes['evidence_total']
    if sizes!=scope['sizes']:raise ValueError('Scope mismatch')
    # Keep every seed and historical setting, with descriptive ordering only.
    order=sorted(metrics,key=lambda n:(-metrics[n]['evidence_total']['rank_score']['mean'],
                                      -metrics[n]['all']['rank_score']['mean'],n))
    def entry(name,label=None):
        return f"| {label or name} | {100*metrics[name]['all']['rank_score']['mean']:.2f} | {100*metrics[name]['evidence_total']['rank_score']['mean']:.2f} |"
    reference='joint/linear_top2_lr0.1_seed20260923'
    lines=['# 完整验证集与完整有证据验证集：统一综合分','',
        f'完整有证据验证集 **{evidence_n}条**，占完整合法验证集 **{all_n}条** 的 **{evidence_n/all_n:.1%}**。本报告只比较这两个集合，不另设新增小集、强弱子集或模型自选接受集。', '',
        f'原固定验证成员{scope["fixed_candidate_n"]}条＋新增{scope["tail_candidate_n"]}条＝{len(rows)}条；统一合法性审查排除{len(rows)-all_n}条，最终{all_n}条。旧固定成员和评分标签保持不变，训练专用截点不充当验证样本。', '',
        f'**{len(metrics)}组模型／配置均从头按时间回放，统一评分后又独立重算校验。** 不比较旧报告分数，不根据模型是否命中改变分母。', '',
        '## 综合分定义','',
        '综合分＝100 × 所有样本上（Top1、Top2、Top3、Top4各前缀最佳匹配收益的平均）。精确匹配收益1，正确服务首页回退0.6，错误0。精确命中在第1/2/3/4位时，单条分别贡献100/75/50/25分；首页回退与后续精确目标只取当时最佳收益，不重复累加。空输出计0分。', '',
        '## 代表方案','',
        '| 方案 | 完整验证集综合分 | 完整有证据验证集综合分 |','|---|---:|---:|']
    for label,name in REPRESENTATIVES:lines.append(entry(name,label))
    lines+=['','## 全部配置','',
        '以下仅作描述性排序：先按完整有证据综合分，再按完整验证集综合分。没有在本轮重新调参、选择随机种子或部署。', '',
        '| 模型／配置 | 完整验证集综合分 | 完整有证据验证集综合分 |','|---|---:|---:|']
    for name in order:lines.append(entry(name))
    lines+=['','## 解释边界','',
        '- 完整集合包含之前用于选参和反复查看的历史开发验证段，不应称为全新独立测试。',
        '- 前台长度1、统一hard_kind训练清洗及导航语义。早期模型移除有序路径／库存候选／同起点硬限制后明确标为适配复测。',
        '- 每次预测只学习此前成熟的标签；统计先验保留原截点信息，不用后来学到答案后的先验重算早期训练特征。KMeans中心只使用最早验证截点之前的输入。',
        '- 完整有证据集仍是根据真实后续目标定义的诊断集合；支持证据必须在预测前已经存在。不是线上置信保证。',
        '- 所有空输出保留在两个固定集合相应分母中。种子与消融都完整保留，配置数不是算法种类数。',
        '- 没有修改运行模型、原始数据和采集设置。完整URL与路径仅保存在私有run。', '',
        f'[全部底层指标](<{run}/metrics-private.json>) · [训练模型清单](<{run}/models.json>)', '']
    (out/'report-private.md').write_text('\n'.join(lines))
    audit={'complete':True,'models':len(metrics),'sizes':sizes,'exactly_two_cohorts':True,
        'primary_metric':'rank_score.mean times 100','independent_recomputation':True,
        'all_fixed_members_retained':True,'scoring_contract':scoring.contract(),
        'sources':sources.accessed,'source_run':str(run),
        'reference_scores':{k:100*metrics[reference][k]['rank_score']['mean'] for k in parts},
        'family_configuration_counts':dict(Counter(v['family'] for v in models.values()))}
    sources.verify_unchanged();write(out/'verification.json',audit)
    (out/('source-'+Path(__file__).name)).write_bytes(Path(__file__).read_bytes())
    for p in out.iterdir():
        if p.is_file():p.chmod(0o600)
    write(out/'artifact-hashes.json',{p.name:sha(p) for p in out.iterdir() if p.is_file()})
    print({k:audit[k] for k in ('models','sizes','exactly_two_cohorts','independent_recomputation','reference_scores')},flush=True)


if __name__=='__main__':main()
