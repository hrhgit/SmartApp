#!/usr/bin/env python3
"""Validation-only hyperparameter search on immutable 3s/5s cleaned labels.

The model, features, candidate policy and original chronological cutoffs stay fixed.
Test data has been inspected before, so results are exploratory, not a fresh holdout.
"""
import argparse
from collections import Counter
from dataclasses import replace
import json
import os
from pathlib import Path

import evaluate_context_knn as live_core
from diagnose_dwell_errors import frozen_core

WEIGHTS=[0.,.05,.1,.2,.3,.35,.4,.5,.55,.6,.7,.75,.8,.9,1.]
KS=[1,3,5,10,15,30,50,100,200]
HALF_LIVES=[.1,.25,.5,1.,3.,6.,12.,24.,72.,168.]
VERSION='dwell-knn-tuning-v1'


def selection_key(row):
    c=row['config'];m=row['validation']
    return (m['top4'],m['mrr4'],-abs(c['weight']-.55),-abs(c['k']-15),
        -abs(c['half_life_hours']-6),-c['weight'],-c['k'],-c['half_life_hours'])


def grid_search(core,items,pairs,positions,weights=WEIGHTS,ks=KS,half_lives=HALF_LIVES):
    """Reuse neighbor order; accumulate identical votes at increasing k cutoffs."""
    grid=[]
    for weight in weights:
        cache={}
        for i in positions:
            q=items[i]
            ranked=[(core.combined_similarity(parts,weight),j) for j,parts in pairs[i]
                if items[j].truth in q.candidates]
            ranked=sorted((x for x in ranked if x[0]>0),key=lambda x:(x[0],items[x[1]].index),reverse=True)[:max(ks)]
            cache[i]=[(sim,j,max(0.,(q.timestamp-items[j].timestamp)/3600000)) for sim,j in ranked]
        for half_life in half_lives:
            scores={k:[] for k in ks}
            for i in positions:
                q=items[i];neighbors=cache[i];votes=Counter();cursor=0
                for k in ks:
                    while cursor<min(k,len(neighbors)):
                        sim,j,age=neighbors[cursor]
                        votes[items[j].truth]+=sim**2*2**(-age/half_life)
                        cursor+=1
                    _,shown=core.rank_votes(votes,q)
                    scores[k].append(core.score_row(q,shown))
            for k in ks:
                grid.append({'config':{'weight':weight,'k':k,'half_life_hours':half_life},
                    'validation':core.metric(scores[k])})
        print(json.dumps({'stage':'grid_weight_complete','weight':weight,'configs_completed':len(grid)}),flush=True)
    return grid


def evaluate(core,items,pairs,positions,config=None,baseline=None):
    m,rows,_=core.evaluate(items,pairs,positions,config,baseline)
    return {'all':m,'strict':core.metric([r for r in rows if items[r['position']].strict]),
        'unknown_coverage':core.metric([r for r in rows if not items[r['position']].strict])},rows


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',required=True,type=Path)
    parser.add_argument('--output',required=True,type=Path)
    args=parser.parse_args();run=args.run.expanduser().resolve();out=args.output.expanduser().resolve()
    if not out.is_relative_to(live_core.PRIVATE_ROOT.resolve()) or out==live_core.PRIVATE_ROOT.resolve():
        parser.error('Use a new private training directory')
    source=json.loads((run/'run.json').read_text());checks=json.loads((run/'artifact-hashes.json').read_text())
    for name,digest in checks.items():
        if Path(name).name!=name or live_core.sha(run/name)!=digest:raise ValueError('Source run changed')
    core=frozen_core(run);snapshot=Path(source['snapshot'])
    hashes,_=core.verified_snapshot(snapshot)
    if hashes!=source['input_hashes']:raise ValueError('Snapshot changed')
    events=core.load_lines(snapshot/'events.jsonl');samples=core.load_lines(snapshot/'navigation-samples.jsonl')
    original,audit,*_=core.make_observations(events,samples,set(source['excluded_sessions']))
    ledgers=json.loads((run/'labels-private.json').read_text())
    old_predictions=json.loads((run/'predictions-private.json').read_text())
    n=len(original);train_end=int(n*.6);test_start=int(n*.8)
    assert test_start==source['test_boundary_original_position']
    os.umask(0o077);out.mkdir(parents=True,mode=0o700,exist_ok=False)
    # Declare the entire search and tie-breaking before evaluating any configuration.
    core.write_json(out/'search-plan.json',{'version':VERSION,'source_run':str(run),
        'weights':WEIGHTS,'neighbor_counts':KS,'half_life_hours':HALF_LIVES,
        'configs_per_threshold':len(WEIGHTS)*len(KS)*len(HALF_LIVES),
        'objective':'validation Top4, then MRR4; then closest to original defaults; then lower values',
        'weight_zero':'history-only control, excluded from contextual model selection',
        'train_end_original_position':train_end,'test_start_original_position':test_start,
        'validation_maturity_required_before_test':True,'test_previously_seen':True,
        'no_algorithm_or_label_changes':True})
    pairs=[[] for _ in original]
    for i in range(train_end,n):
        q=original[i]
        pairs[i]=[(j,core.similarity(q,original[j])) for j in core.eligible_neighbors(original,i)]
        if (i-train_end)%100==0:print(json.dumps({'stage':'causal_pairs','done':i-train_end,'total':n-train_end}),flush=True)
    results={};all_grids={};all_predictions={}
    for t,ledger in ledgers.items():
        items=list(original);valid=set()
        for r in ledger:
            i=r['position'];q=items[i]
            assert q.sample_id==r['sample_id'] and q.ready_index==r['label_ready_event_index']
            if r['status']=='stable_target':
                target=r['target'];valid.add(i)
                items[i]=replace(q,truth=core.key_for(target,q.session),truth_meta={**target,'_bundle':target.get('bundleID')})
        ps=[[(j,s) for j,s in p if j in valid] for p in pairs]
        validation=[i for i in range(train_end,test_start) if i in valid and items[i].ready_index<items[test_start].index]
        test=[i for i in range(test_start,n) if i in valid]
        if not validation or not test:raise ValueError('Empty validation/test')
        # Verify reconstruction against the saved, previously evaluated configuration.
        previous=source['configs']['knn_previous_validation_selected']
        old_metrics,old_rows=evaluate(core,items,ps,test,previous)
        saved=old_predictions[t]['knn_previous_validation_selected']
        assert [(r['position'],r['predictions'],r['hit4']) for r in old_rows]==[(r['position'],r['predictions'],r['hit4']) for r in saved]
        print(json.dumps({'stage':'search_start','threshold':t,'validation_n':len(validation),'test_n':len(test)}),flush=True)
        grid=grid_search(core,items,ps,validation)
        best=max((r for r in grid if r['config']['weight']>0),key=selection_key)
        history=max((r for r in grid if r['config']['weight']==0),key=selection_key)
        # Check optimized search scoring against the frozen reference implementation.
        for chosen in [best,history]:
            reference,_,_=core.evaluate(items,ps,validation,chosen['config'])
            assert reference==chosen['validation'],'Optimized scoring differs from original'
        selected_test,rows=evaluate(core,items,ps,test,best['config'])
        history_test,history_rows=evaluate(core,items,ps,test,history['config'])
        recency,recency_rows=evaluate(core,items,ps,test,baseline='recency')
        markov,_=evaluate(core,items,ps,test,baseline='markov')
        old_validation,_=evaluate(core,items,ps,validation,previous)
        new_bypos={r['position']:r for r in rows};old_bypos={r['position']:r for r in old_rows}
        paired=Counter(('old_hit' if old_bypos[i]['hit4'] else 'old_miss')+'_'+
            ('new_hit' if new_bypos[i]['hit4'] else 'new_miss') for i in test)
        result={'validation_n':len(validation),'validation_strict_n':sum(items[i].strict for i in validation),
            'test_n':len(test),'best_config':best['config'],'best_validation':best['validation'],
            'selected_test':selected_test,'previous_config':previous,'previous_validation':old_validation,
            'previous_test':old_metrics,'recency_test':recency,'markov_test':markov,
            'history_only_control':{'best_config':history['config'],'validation':history['validation'],'test':history_test},
            'paired_test_change':dict(paired),'validation_best_by_weight':[
                max((r for r in grid if r['config']['weight']==w),key=selection_key) for w in WEIGHTS],
            'top10_validation_configs':sorted((r for r in grid if r['config']['weight']>0),key=selection_key,reverse=True)[:10]}
        results[t]=result;all_grids[t]=grid
        all_predictions[t]={'selected':rows,'history_only_control':history_rows,'recency':recency_rows}
        core.write_json(out/f'validation-grid-{t}s.json',grid)
        core.write_json(out/f'result-{t}s.json',result)
        print(json.dumps({'stage':'threshold_complete','threshold':t,'config':best['config'],
            'validation':best['validation'],'test':selected_test}),flush=True)
    metrics={'version':VERSION,'audit':audit,'results':results}
    core.write_json(out/'metrics.json',metrics)
    core.write_json(out/'predictions-private.json',all_predictions)
    render_report(out,results,core)
    (out/'analysis-source.py').write_text(Path(__file__).read_text())
    (out/'core-source.py').write_text((run/'core-source.py').read_text())
    core.write_json(out/'run.json',{'version':VERSION,'source_run':str(run),'source_artifact_hashes':checks,
        'input_hashes':hashes,'snapshot':str(snapshot),'search_plan_sha256':core.sha(out/'search-plan.json'),
        'predictions_reconstruction_verified':True,'optimized_scoring_verified':True,
        'algorithm_and_labels_unchanged':True,'test_previously_seen':True,'live_model_updated':False})
    core.write_json(out/'artifact-hashes.json',{p.name:core.sha(p) for p in out.iterdir() if p.is_file()})
    print(json.dumps({'output':str(out),'results':{t:{k:r[k] for k in ['best_config','best_validation','selected_test','recency_test']} for t,r in results.items()}},ensure_ascii=False))


def render_report(out,results,core):
    lines=['# 清洗后近邻模型重新调参','',
        f'每个阈值搜索 {len(WEIGHTS)*len(KS)*len(HALF_LIVES)} 个组合，共两档。算法、特征、候选、清洗标签与原时间分割不变。',
        '只根据测试开始前已成熟的验证标签选择参数：首先最大化 Top4，其次 MRR4，再按预先声明的默认值距离打破平局。状态权重 0 只作历史单组对照，不能充当上下文模型。',
        '测试期间按原规则在线学习此前已满 30 秒的样本；不利用当前或未来标签。测试段此前已查看，因此这里是探索性历史回放，不是新的独立留出验证。','',
        '| 清洗规则 | 验证数 | 状态权重 | 历史权重 | k | 半衰期（小时） | 验证 Top4 | 测试 Top1 | 测试 Top4 | 测试数 |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for t,r in results.items():
        c=r['best_config'];v=r['best_validation'];test=r['selected_test']['all']
        lines.append(f"| <{t} 秒 | {r['validation_n']} | {c['weight']} | {1-c['weight']:.2f} | {c['k']} | {c['half_life_hours']} | {core.percent(v['top4'])} | {core.percent(test['top1'])} | {core.percent(test['top4'])} | {test['n']} |")
    lines += ['', '## 相同测试样本与标签的对照','',
        '| 清洗规则 | 上次参数 Top4 | 重新选参 Top4 | 近期访问 Top4 | 一阶转移 Top4 |','|---|---:|---:|---:|---:|']
    for t,r in results.items():
        lines.append(f"| {t} 秒 | {core.percent(r['previous_test']['all']['top4'])} | {core.percent(r['selected_test']['all']['top4'])} | {core.percent(r['recency_test']['all']['top4'])} | {core.percent(r['markov_test']['all']['top4'])} |")
    for t,r in results.items():
        lines += ['',f'## {t} 秒档细节','',
            f"验证段严格完整覆盖样本 {r['validation_strict_n']} 个，旧覆盖未知样本单列；测试严格覆盖 {r['selected_test']['strict']['n']} 个，Top4 为 {core.percent(r['selected_test']['strict']['top4'])}。",
            f"纯历史对照的验证最优参数 {r['history_only_control']['best_config']}，测试 Top4 为 {core.percent(r['history_only_control']['test']['all']['top4'])}。",'',
            '| 状态权重 | 该权重的最佳 k | 最佳半衰期 | 验证 Top4 |','|---|---:|---:|---:|']
        for x in r['validation_best_by_weight']:
            c=x['config'];lines.append(f"| {c['weight']} | {c['k']} | {c['half_life_hours']} | {core.percent(x['validation']['top4'])} |")
    lines += ['', '## 解释范围','',
        '- “最优”仅指此次离散搜索范围内验证 Top4 最优，不是全局最优，也不保证测试最优。',
        '- 两种清洗规则的目标与样本不同，不能直接用二者百分比判断哪种规则更正确。',
        '- 数据量有限且行为相关；大网格可能拟合验证段，需要新数据进一步验证。',
        '- 本轮未修复已发现的缺失背景相似度高估，也未改变具体窗口实例/URL 标签粒度。',
        '- 参数搜索产物保存在私有 run，原始记录、既有 run 与运行中的菜单没有改写。']
    (out/'report-private.md').write_text('\n'.join(lines)+'\n')


if __name__=='__main__':main()
