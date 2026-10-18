#!/usr/bin/env python3
"""Generate the private human-readable comparison from verified completed runs."""
from __future__ import annotations
import argparse
from collections import Counter
import json
import os
from pathlib import Path
import compare_online_models as comp
from summarize_tuned_history import verify_run
import evaluate_context_knn as core


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--runs',nargs='+',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True)
    args=ap.parse_args();out=args.output.resolve()
    if not out.is_relative_to(core.PRIVATE_ROOT.resolve()) or out==core.PRIVATE_ROOT.resolve():
        ap.error('Use a new private training directory')
    chosen=[];hashes={};all_selections={}
    for run in args.runs:
        _,h=verify_run(run);hashes[str(run)]=h
        s=json.loads((run/'selection.json').read_text());all_selections[str(run)]=s
        for kind in ['best_current_only','best_development']:
            r=s[kind]
            if r['config']['feature_scope']=='current_only':
                chosen.append((r,run))
    selected,source=min(chosen,key=lambda x:comp.selection_key(x[0]))
    ident=selected['config']['id'];cleaning=selected['training_cleaning'];scores=selected['scores']['5']
    ablations=json.loads((source/'cleaning-ablation.json').read_text())[ident]
    errors=json.loads((source/'selected-errors-private.json').read_text())[cleaning+':'+ident]
    reason_counts=dict(Counter(r['reason'] for r in errors))
    assert scores['supported']['n']==62 and len(errors)==62-scores['supported']['top4_hits']
    os.umask(0o077);out.mkdir(mode=0o700,parents=True,exist_ok=False)
    summary={'selected_source_run':str(source),'selected':selected,'cleaning_ablation':ablations,
        'error_counts':reason_counts,'scope':'best current-only development model among completed comparisons',
        'raw_data_modified':False,'live_model_updated':False,'test_previously_seen':True}
    core.write_json(out/'summary.json',summary)
    core.write_json(out/'errors-private.json',errors)
    core.write_json(out/'compared-selections.json',all_selections)
    def cell(m,metric='top4'):
        return f"{m[metric+'_hits']}/{m['n']}（{100*m[metric]:.1f}%）"
    lines=['# 快速模型比较：清洗与架构分层','',
        f"本次搜索的当前界面模型最高 Top4：**{cell(scores['supported'])}**；Top1：{cell(scores['supported'],'top1')}。",'',
        '评分始终使用此前冻结的62条有证据样本及5秒稳定目标答案。没有因错误、弃权或候选缺失删除样本。原始较晚38条单列。所有结果属于反复分析、调参后的开发回放，不能解释为新数据泛化保证。', '',
        f"原条件频次基线46/62（74.2%）。本次主模型较晚子集 {cell(scores['late_supported'])}；实际候选存在时 {cell(scores['covered_supported'])}；严格完整来源 {cell(scores['strict_supported'])}。",'',
        '## 数据清洗层','',
        '先按目的地准入、真实观察边界和同一次到达的补全关系处理数据，再单独配置停留阈值。完整观察到短于阈值且随后离开的目标可作为中间步骤跳过；依据已知窗口和标签页入口区分同标签页正常导航，未知归属不猜测，未知时长或观测中断不当作短暂停留。返回当前目标时不继续追找更晚的答案。原始事件与定位符保留。', '',
        f"入选模型训练使用 **{cleaning} 秒**停留阈值。下面保持模型结构、参数以及评分62条正确答案全部不变，只改变历史训练监督：", '',
        '| 历史训练标签 | 固定62条 Top4 | 固定62条 Top1 | 原晚38条 Top4 |', '|---|---:|---:|---:|']
    for r in ablations:
        t=r['training_cleaning'];s=r['scores']['5']
        title='不做停留清洗' if t=='raw' else t+'秒'
        lines.append(f"| {title} | {cell(s['supported'])} | {cell(s['supported'],'top1')} | {cell(s['late_supported'])} |")
    lines+=['', '这与改变评分标签是两个问题：原生3秒64条敏感性评分独立保存在指标里。所有学习标签仍等原30秒观察窗口结束后才允许进入模型，避免尚未确认的结果提前影响预测。', '',
        '## 模型架构层','',
        '模型输入仅当前目的地、所属应用与类型，上下文长度1。应用及应用窗口类别采用已确认粗粒度，网页/目录仍按具体目标。没有前序页面序列、后台相似度或未来答案。', '',
        f"```json\n{json.dumps(selected['config'],ensure_ascii=False,indent=2)}\n```", '',
        '基础比较覆盖频次、近期记忆、衰减、贝叶斯平滑、在线Softmax/逻辑回归、感知机/大间隔更新、朴素贝叶斯、Hoeffding树/自适应森林、流式近邻、在线聚类、小型MLP；另检查排序融合、预测前历史支持限制，以及时刻/候选最近访问的独立消融。它是有限且记录范围的搜索，不声称穷尽全部可能模型。', '',
        '窗口类别分数只有能唯一对应到当前具体候选时才展示；统一进行候选有效性及父子层级去重，最多返回四条。即刻更新指确认新标签后能逐条学习，不代表标签确认无需等待。', '',
        '## 证据强弱','', '| 分组 | Top4 | Top1 |', '|---|---:|---:|']
    for g,title in [('strong','反复且集中支持'),('repeated','重复支持'),('single','仅一次支持')]:
        m=scores['grades'][g];lines.append(f"| {title} | {cell(m)} | {cell(m,'top1')} |")
    lines+=['','## 未命中情况','',f"未命中原因：`{json.dumps(reason_counts,ensure_ascii=False)}`。",'',
        '62条中6条真实窗口未进入原候选，固定候选的理论上限是56/62（90.3%）。正确应用不能冒充正确窗口。支持集合是事后依据实际答案定义的诊断集合；预测时的历史支持限制如有，只读取过去已成熟转移，不读取集合成员或本次真实答案。', '',
        '| 当前界面 | 实际下一步 | 原因 | 原始排序名次 |', '|---|---|---|---:|']
    for e in errors:
        lines.append(f"| {e['origin']} | {e['actual']} | {e['reason']} | {e['raw_rank'] or '—'} |")
    lines+=['','## 复核与产物','',
        '每次预测严格只学习 ready_index 小于当前截点的历史，同一当前类别与实际后续到达去重。原频次基线已逐条复现；最终结构完整回放与缓存融合排序核对一致。来源完整子集样本少，其全对不能外推。', '',
        '实际模型/清洗参数、每条预测、延迟、完整搜索网格与不可变源代码保存在各来源run；本汇总保存来源哈希。原数据、此前run、实际菜单未改。', '',
        f"最终模型详细run：[报告](<{source/'report-private.md'}>)。", '']
    family_winners={}
    for s in all_selections.values():
        family_winners.update(s.get('base_family_winners',{}))
    for run in args.runs:
        grid=run/'component-grid.json'
        if grid.is_file():
            assert grid.name in hashes[str(run)]
            for r in json.loads(grid.read_text()):
                family=r['config']['family']
                if family not in family_winners or comp.selection_key(r)<comp.selection_key(family_winners[family]):
                    family_winners[family]=r
    if family_winners:
        lines+=['## 各基础方法的已搜索最优配置','',
            '以下包含基础全面比较与局部参数检查的家族最优，均对同一62条评分；训练清洗可不同，表中明确列出。额外特征实验不与仅当前界面的结构改动混为一谈。', '',
            '| 家族 | 训练清洗 | Top4 | Top1 | 输入范围 |', '|---|---|---:|---:|---|']
        for family,r in family_winners.items():
            m=r['scores']['5']['supported']
            lines.append(f"| {family} | {r['training_cleaning']} | {cell(m)} | {cell(m,'top1')} | {r['config']['feature_scope']} |")
        lines+=['']
    (out/'report-private.md').write_text('\n'.join(lines))
    (out/'source-finalize_online_comparison.py').write_bytes(Path(__file__).read_bytes())
    core.write_json(out/'run.json',{'version':'current-online-comparison-summary-v1','source_runs':hashes,
        'raw_data_modified':False,'live_model_updated':False,'test_previously_seen':True})
    core.write_json(out/'artifact-hashes.json',{p.name:core.sha(p) for p in out.iterdir() if p.is_file()})
    print(json.dumps({'output':str(out),'source':str(source),'selected_config':selected['config'],
        'cleaning':cleaning,'scores':scores,'error_counts':reason_counts}),flush=True)

if __name__=='__main__':main()
