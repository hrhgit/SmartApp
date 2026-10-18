#!/usr/bin/env python3
"""Compare a bounded rank-fusion grid and consolidate immutable model sweeps.

This is exploratory development selection. Base models were replayed causally;
model selection on the 62 already examined answers is not a fresh holdout.
"""
from __future__ import annotations
import argparse
from collections import Counter
import itertools
import json
import os
from pathlib import Path
import time

import compare_online_models as comparison
from summarize_tuned_history import verify_run
import evaluate_context_knn as core_io


def key(result, group='supported'):
    return comparison.selection_key(result, group)


def combined_scope(left, right):
    scopes = {left['config'].get('feature_scope','current_only'), right['config'].get('feature_scope','current_only')}
    return 'current_only' if scopes == {'current_only'} else '+'.join(sorted(scopes-{'current_only'}))


def fuse(core, left, right, weight, offset, items):
    output = {}
    assert left.keys() == right.keys()
    for i in left:
        a, b = left[i]['raw_ranking'], right[i]['raw_ranking']
        scores = {k: (1-weight)/(offset+rank) for rank,k in enumerate(a,1)}
        for rank,k in enumerate(b,1):
            scores[k] = scores.get(k,0) + weight/(offset+rank)
        ranked = sorted(scores, key=lambda k:(-scores[k],k))
        output[i] = {'sample_id': items[i].sample_id, 'raw_ranking': ranked,
            'predictions': core.select_four(ranked, items[i].candidates)}
    return output


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--runs',type=Path,nargs='+',required=True)
    ap.add_argument('--output',type=Path,required=True)
    args=ap.parse_args();out=args.output.resolve()
    if not out.is_relative_to(core_io.PRIVATE_ROOT.resolve()) or out == core_io.PRIVATE_ROOT.resolve():
        ap.error('Use a new private run')
    os.umask(0o077);out.mkdir(parents=True,mode=0o700,exist_ok=False)
    records=[];source_hashes={};expected=None
    for run in args.runs:
        prov, hashes=verify_run(run)
        if expected is None:expected=prov['expansion_run']
        assert prov['expansion_run']==expected
        source_hashes[str(run)]=hashes
        for r in json.loads((run/'metrics.json').read_text()):
            records.append({**r,'source_run':str(run)})
    identities=[(r['training_cleaning'],r['config']['id']) for r in records]
    assert len(identities)==len(set(identities))
    core,items,cohorts,streams,inputs,mappings,provenance=comparison.load_dataset(Path(expected))
    prediction_cache={}
    def prediction(r):
        ident=(r['training_cleaning'],r['config']['id'])
        if ident not in prediction_cache:
            p=Path(r['source_run'])/f"predictions-{r['training_cleaning']}-{r['config']['id']}-private.json"
            prediction_cache[ident]={int(i):v for i,v in json.loads(p.read_text())['predictions'].items()}
        return prediction_cache[ident]
    # A modest, specified list: five strongest current-only statistical families,
    # five current-only ML families, plus at most two explicitly auxiliary inputs.
    plan={'weights':[.1,.25,.5,.75,.9],'rank_offsets':[0,10],
        'component_selection':'per cleaning: best per family, top5 statistical current-only + top5 ML current-only + top2 auxiliary families',
        'component_selection_metric':'full62 development Top4, then Top1,MRR,config ID,cleaning',
        'primary_scoring':'fixed5s labels and frozen62; three training cleanings independent',
        'no_claim_of_exhaustive_all_models':True,'test_previously_seen':True,
        'temporal_control':'recomputed base-only earlier24 selection; explicit cleaning tie resolves without late results',
        'source_runs':source_hashes}
    core_io.write_json(out/'plan.json',plan)
    fusion_results=[];fusion_specs={}
    for cleaning in ('5','3','raw'):
        ordered=sorted([r for r in records if r['training_cleaning']==cleaning],key=key)
        leaders={}
        for r in ordered:leaders.setdefault(r['config']['family'],r)
        leaders=list(leaders.values())
        stat=[r for r in leaders if r['config']['id'].startswith('stat_') and r['config']['feature_scope']=='current_only'][:5]
        ml=[r for r in leaders if r['config']['id'].startswith('online-') and r['config']['feature_scope']=='current_only'][:5]
        aux=[r for r in leaders if r['config']['feature_scope']!='current_only'][:2]
        selected=stat+ml+aux
        for pair_index,(a,b) in enumerate(itertools.combinations(selected,2)):
            for offset,weight in itertools.product(plan['rank_offsets'],plan['weights']):
                fusion_id=f"fusion-{cleaning}-{pair_index}-{offset}-{weight:g}"
                params={'left':a['config'],'right':b['config'],'right_weight':weight,'rank_offset':offset}
                config={'id':fusion_id,'family':'rank_fusion_ensemble','feature_scope':combined_scope(a,b),'parameters':params}
                pred=fuse(core,prediction(a),prediction(b),weight,offset,items)
                result={'training_cleaning':cleaning,'config':config,
                    'scores':{t:comparison.score_groups(rows,pred) for t,rows in cohorts.items()}}
                fusion_results.append(result);fusion_specs[fusion_id]=(a,b,weight,offset)
        print(json.dumps({'stage':'fusions','cleaning':cleaning,'components':len(selected),
            'completed_total':len(fusion_results),'best':sorted(fusion_results,key=key)[0]['scores']['5']['supported']}),flush=True)
    all_results=sorted(records+fusion_results,key=key)
    best=all_results[0]
    best_pure=next(r for r in all_results if r['config']['feature_scope']=='current_only')
    best5=next(r for r in all_results if r['training_cleaning']=='5')
    pure5=next(r for r in all_results if r['training_cleaning']=='5' and r['config']['feature_scope']=='current_only')
    by_family={}
    for r in sorted(records,key=key):by_family.setdefault(r['config']['family'],r)
    early=min(records,key=lambda r:key(r,'early_supported'))
    selections={'best_development':best,'best_current_only':best_pure,'best_training5':best5,
        'best_current_only_training5':pure5,'base_family_winners':by_family,
        'base_earlier_selected':early,'base_config_count':len(set(r['config']['id'] for r in records)),
        'base_replay_count':len(records),'fusion_config_count':len(fusion_results),
        'primary_cohort_size':62,'candidate_ceiling':56,'current_only_recommended':best_pure,
        'early_selection_note':'recomputed using an explicit total tie key; do not reuse sweep-01 early-selection artifact'}
    def get_prediction(r):
        if r['config']['family']!='rank_fusion_ensemble':return prediction(r)
        a,b,w,o=fusion_specs[r['config']['id']]
        return fuse(core,prediction(a),prediction(b),w,o,items)
    audits={}
    # Rerun selected architecture unchanged under all three cleaning factors.
    # This isolates training cleaning from component/hyperparameter selection.
    from fused_online_models import FusedModel
    import stat_model_suite, online_model_suite
    def build(config):
        if config['family']=='rank_fusion_ensemble':
            p=config['parameters']
            return FusedModel(build(p['left']),build(p['right']),p['right_weight'],p['rank_offset'])
        return (stat_model_suite if config['id'].startswith('stat_') else online_model_suite).build(config)
    for i,q in inputs.items():q['candidate_key_order']=mappings[i]
    original_selections=(best,best_pure,best5,pure5)
    selected_unique={r['config']['id']:r for r in original_selections}
    ablation_results=[]
    for ident,r in selected_unique.items():
        fixed=[]
        for cleaning in ('5','3','raw'):
            pred,timing=comparison.replay(core,build(r['config']),items,streams[cleaning],inputs,mappings)
            for chosen in original_selections:
                if chosen['config']['id']==ident and cleaning==chosen['training_cleaning']:
                    cached=get_prediction(chosen)
                    assert all(pred[i]['predictions']==cached[i]['predictions'] and
                        pred[i]['raw_ranking']==cached[i]['raw_ranking'] for i in pred), 'Streaming/cached fusion parity failed'
            m={t:comparison.score_groups(rows,pred) for t,rows in cohorts.items()}
            fixed.append({'training_cleaning':cleaning,'scores':m,'timing':timing})
            ablation_results.append({'training_cleaning':cleaning,'config':r['config'],
                'scores':m,'timing':timing,'prediction_file':f'final-predictions-{cleaning}-{ident}-private.json'})
            core_io.write_json(out/f'final-predictions-{cleaning}-{ident}-private.json',pred)
            if cleaning==r['training_cleaning']:
                core_io.write_json(out/f'errors-{ident}-private.json',comparison.diagnose(cohorts['5'],pred))
        audits[ident]=fixed
    # A fixed-architecture cleaning ablation is itself an evaluated candidate.
    # Promote any improvement instead of retaining a stale grid-best headline.
    final_order=sorted(all_results+ablation_results,key=key)
    best=final_order[0]
    best_pure=next(r for r in final_order if r['config']['feature_scope']=='current_only')
    best5=next(r for r in final_order if r['training_cleaning']=='5')
    pure5=next(r for r in final_order if r['training_cleaning']=='5' and r['config']['feature_scope']=='current_only')
    selections.update(best_development=best,best_current_only=best_pure,best_training5=best5,
        best_current_only_training5=pure5,current_only_recommended=best_pure)
    selected_errors={}
    for r in (best,best_pure,best5,pure5):
        ident=r['config']['id'];cleaning=r['training_cleaning']
        p=out/f'final-predictions-{cleaning}-{ident}-private.json'
        pred={int(i):v for i,v in json.loads(p.read_text()).items()}
        selected_errors[f'{cleaning}:{ident}']=comparison.diagnose(cohorts['5'],pred)
    core_io.write_json(out/'selected-errors-private.json',selected_errors)
    core_io.write_json(out/'selection.json',selections)
    core_io.write_json(out/'cleaning-ablation.json',audits)
    core_io.write_json(out/'ablation-results.json',ablation_results)
    core_io.write_json(out/'base-metrics.json',records)
    core_io.write_json(out/'fusion-grid.json',fusion_results)
    core_io.write_json(out/'benchmark-membership.json',json.loads((Path(expected)/'benchmark-membership.json').read_text()))
    for filename in ('search_online_fusions.py','compare_online_models.py','stat_model_suite.py','online_model_suite.py','fused_online_models.py'):
        (out/('source-'+filename)).write_bytes(Path(__file__).with_name(filename).read_bytes())
    core_io.write_json(out/'dependencies.json',online_model_suite.dependency_metadata())
    lines=['# 固定有证据集合上的在线模型比较','',
        f"比较{selections['base_config_count']}组基础配置、三种训练清洗共{len(records)}次因果回放；再做{len(fusion_results)}组两模型排序融合。全部为开发集搜索，不是独立新测试。",'',
        '主集合62条使用固定5秒清洗答案；候选缺失6条仍为失败，固定候选上限56/62=90.3%。支持强度分组及成员未改。', '',
        '| 类别 | 模型 | 训练清洗 | Top4/62 | Top1/62 | 晚38 Top4 |', '|---|---|---|---:|---:|---:|']
    for name,r in [('仅当前界面最优',best_pure),('允许额外时刻/近期访问最优',best),*by_family.items()]:
        m=r['scores']['5']['supported'];late=r['scores']['5']['late_supported']
        lines.append(f"| {name} | {r['config']['id']} | {r['training_cleaning']} | {m['top4_hits']}/62 ({m['top4']:.1%}) | {m['top1_hits']}/62 | {late['top4_hits']}/38 |")
    architecture=('当前界面 → 各在线模型独立打分 → 各自排序 → 加权倒数排名融合 → 当前候选唯一映射及层级去重 → 前4条。'
        if best_pure['config']['family']=='rank_fusion_ensemble' else
        '当前界面 → 已选在线模型打分 → 当前候选唯一映射及层级去重 → 前4条。')
    lines+=['','## 当前界面最优配置','',f"```json\n{json.dumps(best_pure['config'],ensure_ascii=False,indent=2)}\n```",'',
        architecture+'新行为需先确认派生稳定标签，再调用增量学习。', '',
        '## 清洗对照：固定最优模型结构与参数','', '| 训练标签 | 固定62 Top4 | 原晚38 Top4 | 原生3秒64条 Top4（不同标签/集合） |', '|---|---:|---:|---:|']
    for r in audits[best_pure['config']['id']]:
        s=r['scores'];lines.append(f"| {r['training_cleaning']} | {s['5']['supported']['top4_hits']}/62 | {s['5']['late_supported']['top4_hits']}/38 | {s['3']['supported']['top4_hits']}/64 |")
    lines+=['', '清洗只改变历史监督：短于阈值且已完整观察的中间到达可跳过；同标签页导航按已观察窗口/入口归属单列，未知归属不猜测，未知停留或中断不当短暂停留。A→短B→A不继续找C。原始事件保留。未做停留清洗对照仍经过准入与30秒完整观察窗过滤。', '',
        '## 分组与选参边界','',f"当前界面最优证据分组：`{json.dumps(best_pure['scores']['5']['grades'],ensure_ascii=False)}`",'',
        f"候选存在：{best_pure['scores']['5']['covered_supported']['top4_hits']}/56；严格完整来源：{best_pure['scores']['5']['strict_supported']['top4_hits']}/13。",'',
        f"仅使用较早24条选择基础模型为{early['config']['id']}（训练{early['training_cleaning']}），较早{early['scores']['5']['early_supported']['top4_hits']}/24，较晚{early['scores']['5']['late_supported']['top4_hits']}/38。该控制不用于选择开发最高分融合；数据此前仍已被查看。",'',
        '模型选择使用开发答案，但每次预测/更新仅使用截点前成熟信息。按实际下一步筛出的有证据集合不能当作线上可提前识别的置信保证。当前上下文仍为长度1；时刻与候选最近访问配置明确单列。', '',
        '逐条预测、错误、分组、实际延迟、完整网格与源代码哈希在本私有run。原始数据、此前run和线上菜单均未修改。', '']
    (out/'report-private.md').write_text('\n'.join(lines))
    core_io.write_json(out/'run.json',{'version':'current-online-fusion-comparison-v1','source_runs':source_hashes,
        **provenance,'raw_data_modified':False,'live_model_updated':False,'test_previously_seen':True,
        'streaming_cached_predictions_verified':True})
    core_io.write_json(out/'artifact-hashes.json',{p.name:core_io.sha(p) for p in out.iterdir() if p.is_file()})
    print(json.dumps({'complete':True,'output':str(out),'current_only':{
        'config':best_pure['config'],'training_cleaning':best_pure['training_cleaning'],
        'main':best_pure['scores']['5']['supported']},'overall':{
        'config':best['config'],'training_cleaning':best['training_cleaning'],
        'main':best['scores']['5']['supported']}}),flush=True)

if __name__=='__main__':main()
