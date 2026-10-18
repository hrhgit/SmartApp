#!/usr/bin/env python3
"""Selective contextual memory: radius gates, clustered support and abstention.

Selection uses validation only. Evidence gates never read the query's future label.
Unseen/ambiguous outcomes contribute disagreement rather than being silently removed.
"""
import argparse
from collections import Counter,defaultdict
from dataclasses import replace
import itertools
import json
import math
import os
from pathlib import Path

import evaluate_context_knn as live_core
from diagnose_dwell_errors import frozen_core
from summarize_tuned_history import verify_run

VERSION='selective-context-memory-v1'
DENIED_BUNDLES={'com.apple.KeyboardSetupAssistant'}
MIN_VALIDATION_ACCEPTED=15
TARGET_PRECISION=.8
REFERENCE_URLS=[
    'https://arxiv.org/abs/1705.08500',
    'https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.RadiusNeighborsClassifier.html']


def semantic_label(q,representation):
    t=q.truth_meta
    if representation=='window_class' and t['kind']=='window' and t.get('_bundle'):
        return 'window_class:'+t['_bundle']
    return q.truth


def resolve_label(key,q):
    if key.startswith('window_class:'):
        bundle=key.removeprefix('window_class:')
        matches=[k for k,t in q.candidates.items() if t['kind']=='window' and t.get('_bundle')==bundle]
        return matches[0] if len(matches)==1 else None
    return key if key in q.candidates else None


def wilson_lower(hits,n):
    if not n:return 0.
    z=1.96;p=hits/n
    return (p+z*z/(2*n)-z*math.sqrt(p*(1-p)/n+z*z/(4*n*n)))/(1+z*z/n)


def independent_positions(items,positions,spacing_ms=30000):
    """Thin using only prediction times, never true labels or acceptance results."""
    previous={};selected=[]
    for i in positions:
        q=items[i]
        if q.timestamp-previous.get(q.session,-math.inf)>=spacing_ms:
            selected.append(i);previous[q.session]=q.timestamp
    return selected


def clean_input(q,blocked_keys):
    def keep(tokens):
        return {k:v for k,v in tokens.items() if not any(b in k for b in DENIED_BUNDLES)
            and not any('target:'+b in k for b in blocked_keys)}
    return replace(q,foreground=keep(q.foreground),history=keep(q.history),
        background={k:keep(v) for k,v in q.background.items()},
        candidates={k:t for k,t in q.candidates.items() if t.get('_bundle') not in DENIED_BUNDLES and k not in blocked_keys},
        recency={k:v for k,v in q.recency.items() if k not in blocked_keys})


def prepare_pairs(core,items,valid,positions):
    pairs={}
    for counter,i in enumerate(positions):
        q=items[i];rows=[]
        for j in core.eligible_neighbors(items,i):
            if j not in valid:continue
            r=items[j]
            # A common foreground semantic class is required before more expensive context comparison.
            fg=core.weighted_jaccard(q.foreground,r.foreground)
            if fg<.8:continue
            parts=core.similarity(q,r)
            if parts[5]<.25 or parts[4]<.5:continue
            rows.append((j,parts))
        pairs[i]=rows
        if counter and counter%100==0:print(json.dumps({'stage':'pair_evidence','done':counter,'total':len(positions)}),flush=True)
    return pairs


def evidence(core,items,ledger,outcome_times,i,pairs,config):
    q=items[i];by_outcome={}
    for j,parts in pairs[i]:
        r=items[j]
        if config['foreground']=='exact' and q.current!=r.current:continue
        if parts[4]<config['background_min'] or parts[2]<config['history_min']:continue
        # Fixed denominator: missing background never inflates similarity by renormalization.
        similarity=.25*parts[3]+.30*parts[2]+.45*parts[4]*parts[5]
        age_hours=max(0.,(q.timestamp-r.timestamp)/3600000)
        weight=similarity**2*2**(-age_hours/config['half_life_hours'])
        outcome=ledger[j]['target_event_id']
        if outcome not in by_outcome or weight>by_outcome[outcome]['weight']:
            by_outcome[outcome]={'position':j,'outcome_event_id':outcome,
                'label':semantic_label(r,config['representation']),'weight':weight,
                'similarity':similarity,'foreground_similarity':parts[3],'history_similarity':parts[2],
                'background_similarity':parts[4],'background_reliability':parts[5],
                'strict_source_pair':q.strict and r.strict}
    if not by_outcome:return {'predictions':[],'abstain_reason':'no_similar_evidence','evidence_count':0}
    # A 30s cluster can contribute at most its strongest event's weight. Other labels in
    # that cluster retain their share, preventing correlated repeats from erasing disagreement.
    groups=[];last_session=None;group_start=None
    ordered=sorted(by_outcome.values(),key=lambda x:outcome_times[x['outcome_event_id']])
    for r in ordered:
        session,timestamp=outcome_times[r['outcome_event_id']]
        if session!=last_session or group_start is None or timestamp-group_start>=30000:
            groups.append([]);group_start=timestamp;last_session=session
        groups[-1].append(r)
    votes=Counter();support=Counter();group_weights=[]
    for group in groups:
        total=sum(r['weight'] for r in group);maximum=max(r['weight'] for r in group)
        if not total:continue
        group_weights.append(maximum)
        for label in {r['label'] for r in group}:support[label]+=1
        for r in group:votes[r['label']]+=r['weight']*maximum/total
    total=sum(votes.values())
    if not total:return {'predictions':[],'abstain_reason':'no_effective_evidence','evidence_count':len(ordered)}
    ranked=sorted(votes,key=lambda k:(-votes[k],k));winner=ranked[0]
    mapped=resolve_label(winner,q)
    confidence=votes[winner]/total
    neff=sum(group_weights)**2/sum(w*w for w in group_weights)
    # No recency fill: every returned target must have direct contextual evidence.
    raw=[]
    for key in ranked:
        resolved=resolve_label(key,q)
        if resolved and resolved not in raw:raw.append(resolved)
    shown=core.select_four(raw,q.candidates) if mapped else []
    return {'predictions':shown,'winner':winner,'winner_resolved':mapped,
        'agreement':confidence,'winner_support_groups':support[winner],
        'evidence_count':len(ordered),'evidence_groups':len(group_weights),'effective_groups':neff,
        'strict_evidence_groups':sum(any(r['strict_source_pair'] for r in g) for g in groups),
        'abstain_reason':None if mapped else 'winner_unavailable_or_ambiguous',
        'evidence':ordered,'votes':dict(votes)}


def accept(e,min_support,min_agreement):
    return bool(e['predictions'] and e.get('winner_resolved')==e['predictions'][0]
        and e.get('winner_support_groups',0)>=min_support and e.get('effective_groups',0)>=1.5
        and e.get('agreement',0)+1e-12>=min_agreement)


def summarize(core,items,positions,records,min_support,min_agreement):
    accepted=[i for i in positions if accept(records[i],min_support,min_agreement)]
    rows=[core.score_row(items[i],records[i]['predictions']) for i in accepted]
    m=core.metric(rows)
    return {**m,'eligible_n':len(positions),'coverage':len(accepted)/len(positions) if positions else 0.,
        'wilson_lower_descriptive':wilson_lower(m['top1_hits'],m['n'])},accepted


def choose(grid):
    enough=[r for r in grid if r['validation']['n']>=MIN_VALIDATION_ACCEPTED]
    qualified=[r for r in enough if r['validation']['top1']>=TARGET_PRECISION]
    if qualified:
        return max(qualified,key=lambda r:(r['validation']['coverage'],r['validation']['wilson_lower_descriptive'],
            r['validation']['top1'],r['validation']['top4'])),'validation_target_met'
    if enough:
        return max(enough,key=lambda r:(r['validation']['wilson_lower_descriptive'],r['validation']['top1'],
            r['validation']['coverage'])),'validation_target_not_met'
    return max(grid,key=lambda r:(r['validation']['wilson_lower_descriptive'],r['validation']['n'])),'too_few_accepted_validation_samples'


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-run',required=True,type=Path)
    parser.add_argument('--tuning-run',required=True,type=Path)
    parser.add_argument('--output',required=True,type=Path)
    args=parser.parse_args();base=args.source_run.resolve();out=args.output.resolve()
    if not out.is_relative_to(live_core.PRIVATE_ROOT.resolve()) or out==live_core.PRIVATE_ROOT.resolve():
        parser.error('Use a new private training directory')
    source,source_hashes=verify_run(base);tuning,tuning_hashes=verify_run(args.tuning_run)
    assert Path(tuning['source_run'])==base
    core=frozen_core(base);snapshot=Path(source['snapshot']);hashes,_=core.verified_snapshot(snapshot)
    assert hashes==source['input_hashes']
    events=core.load_lines(snapshot/'events.jsonl');samples=core.load_lines(snapshot/'navigation-samples.jsonl')
    original,*_=core.make_observations(events,samples,set(source['excluded_sessions']))
    ledger=json.loads((base/'labels-private.json').read_text())
    byid={e['id']:e for e in events}
    outcome_times={e['id']:(e['sessionID'],e['timestampMs']) for e in events}
    blocked_keys={core.key_for(e['data']['target'],e['sessionID']) for e in events if e['kind']=='target.update'
        and e['data']['target'].get('bundleID') in DENIED_BUNDLES}
    original=[clean_input(q,blocked_keys) for q in original]
    predictions=json.loads((args.tuning_run/'predictions-private.json').read_text())
    n=len(original);train_end=int(n*.6);test_start=int(n*.8)
    shapes=[dict(zip(['foreground','background_min','history_min','half_life_hours','representation'],v))
        for v in itertools.product(['exact','semantic'],[.5,.75,.9],[0.,.3,.5],[.25,1.,6.],['exact','window_class'])]
    gates=list(itertools.product([2,3],[.7,.85,1.]))
    os.umask(0o077);out.mkdir(parents=True,mode=0o700,exist_ok=False)
    core.write_json(out/'search-plan.json',{'version':VERSION,'algorithm_configs':shapes,'gates':gates,
        'validation_target_top1':TARGET_PRECISION,'min_accepted_validation':MIN_VALIDATION_ACCEPTED,
        'selection':'maximize coverage after precision target and count requirement; otherwise best descriptive Wilson lower score',
        'prediction_spacing_ms':30000,'support_cluster_ms':30000,'min_background_reliability':.25,
        'minimum_effective_groups':1.5,'train_end_original_position':train_end,'test_start_original_position':test_start,
        'denied_bundles':sorted(DENIED_BUNDLES),'sources':REFERENCE_URLS,
        'test_previously_seen':True,'no_target_based_acceptance':True})
    results={};all_records={}
    for threshold,ls in ledger.items():
        items=list(original);valid=set();excluded=Counter()
        for r in ls:
            i=r['position'];q=items[i]
            if r['status']!='stable_target':continue
            t=r['target']
            if q.current_app in DENIED_BUNDLES or t.get('bundleID') in DENIED_BUNDLES:
                excluded['out_of_scope_helper']+=1;continue
            valid.add(i);items[i]=replace(q,truth=core.key_for(t,q.session),truth_meta={**t,'_bundle':t.get('bundleID')})
        val_full=[i for i in range(train_end,test_start) if i in valid and items[i].ready_index<items[test_start].index]
        test_full=[i for i in range(test_start,n) if i in valid]
        validation=independent_positions(items,val_full);test=independent_positions(items,test_full)
        print(json.dumps({'stage':'evidence_start','threshold':threshold,'validation_n':len(validation),'test_n':len(test)}),flush=True)
        pairs=prepare_pairs(core,items,valid,validation)
        grid=[]
        for number,config in enumerate(shapes):
            records={i:evidence(core,items,ls,outcome_times,i,pairs,config) for i in validation}
            for support,agreement in gates:
                m,_=summarize(core,items,validation,records,support,agreement)
                grid.append({'config':{**config,'min_support':support,'min_agreement':agreement},'validation':m})
            if number%30==0:print(json.dumps({'stage':'validation_shapes','threshold':threshold,'done':number,'total':len(shapes)}),flush=True)
        best,status=choose(grid);config=best['config']
        # Lock selection before computing any evidence or score on the late test set.
        core.write_json(out/f'selection-{threshold}s.json',{'status':status,**best})
        pairs.update(prepare_pairs(core,items,valid,test_full))
        records={i:evidence(core,items,ls,outcome_times,i,pairs,config) for i in validation+test_full}
        val_m,_=summarize(core,items,validation,records,config['min_support'],config['min_agreement'])
        assert val_m==best['validation']
        test_m,accepted=summarize(core,items,test,records,config['min_support'],config['min_agreement'])
        full_m,full_accepted=summarize(core,items,test_full,records,config['min_support'],config['min_agreement'])
        strict_m,_=summarize(core,items,[i for i in test if items[i].strict],records,config['min_support'],config['min_agreement'])
        baseline={}
        for model in ['selected','recency']:
            saved={r['position']:r for r in predictions[threshold][model]}
            baseline[model]=core.metric([core.score_row(items[i],[k for k in saved[i]['predictions'] if k in items[i].candidates]) for i in accepted])
        reasons=Counter()
        for i in test:
            e=records[i]
            if i in accepted:reasons['accepted']+=1
            elif e['abstain_reason']:reasons[e['abstain_reason']]+=1
            elif e['winner_support_groups']<config['min_support']:reasons['too_few_distinct_support_groups']+=1
            elif e['effective_groups']<1.5:reasons['evidence_weight_too_concentrated']+=1
            else:reasons['historical_outcomes_disagree']+=1
        results[threshold]={'selection_status':status,'selected':best,'test':test_m,'test_strict':strict_m,
            'all_test_cutoffs_secondary':full_m,'same_accepted_test_baselines':baseline,'test_decisions':dict(reasons),
            'original_validation_n':len(val_full),'original_test_n':len(test_full),'excluded':dict(excluded)}
        all_records[threshold]=[{'position':i,'sample_id':items[i].sample_id,'time_utc':core.utc(items[i].timestamp),
            'subset':'validation' if i<test_start else 'test','primary_cutoff':i in validation or i in test,
            'accepted':accept(records[i],config['min_support'],config['min_agreement']),
            'truth':items[i].truth,'truth_meta':items[i].truth_meta,
            'predicted_meta':[items[i].candidates[k] for k in records[i]['predictions']],
            'score':core.score_row(items[i],records[i]['predictions']),**records[i]} for i in validation+test_full]
        core.write_json(out/f'validation-grid-{threshold}s.json',grid)
        core.write_json(out/f'result-{threshold}s.json',results[threshold])
        print(json.dumps({'stage':'threshold_complete','threshold':threshold,**results[threshold]}),flush=True)
    core.write_json(out/'metrics.json',results);core.write_json(out/'predictions-private.json',all_records)
    render_report(out,results,core)
    (out/'analysis-source.py').write_text(Path(__file__).read_text())
    core.write_json(out/'run.json',{'version':VERSION,'source_run':str(base),'source_artifact_hashes':source_hashes,
        'tuning_run':str(args.tuning_run.resolve()),'tuning_artifact_hashes':tuning_hashes,'input_hashes':hashes,
        'raw_data_modified':False,'live_model_updated':False,'test_previously_seen':True})
    core.write_json(out/'artifact-hashes.json',{p.name:core.sha(p) for p in out.iterdir() if p.is_file()})
    print(json.dumps({'output':str(out),'results':results},ensure_ascii=False))


def render_report(out,results,core):
    lines=['# 优先预测有相似情景证据的行为','',
        '使用带暂不预测机制的情景记忆：前台匹配、背景相似度和共同已观察来源门槛先筛选历史，再按路径相似度及时间衰减投票。拒绝强行凑 k 个近邻，拒绝对缺失背景重新归一化成满分。',
        '预测是否有证据只看当前输入及此前已成熟的历史，不读取本次真实后续目标。历史后续标签按到达事件去重，并按 30 秒片段限制总权重；主评估也按预测时间至少间隔 30 秒取样，避免连续重叠样本夸大结果。',
        '窗口类别经验仅在当前候选中能唯一映射到一个具体窗口时使用；多个同类窗口不能猜成一个。不可达或模糊的历史答案仍参与分歧计算，不静默删除来抬高一致率。',
        '验证目标预先设为 Top1 至少 80%、接受不少于 15 个截点，再最大化覆盖率。不达标时明确报告，不能将小样本高命中说成可靠保证。Wilson 分数只作排序参考，相关行为及参数搜索不支持独立二项置信保证。','',
        '| 清洗规则 | 验证是否达标 | 验证接受/总数 | 验证 Top1 | 测试接受/总数 | 测试 Top1 | 测试 Top4 | 测试覆盖率 |',
        '|---|---|---:|---:|---:|---:|---:|---:|']
    for t,r in results.items():
        v=r['selected']['validation'];m=r['test']
        lines.append(f"| {t} 秒 | {r['selection_status']} | {v['n']}/{v['eligible_n']} | {core.percent(v['top1'])} | {m['n']}/{m['eligible_n']} | {core.percent(m['top1'])} | {core.percent(m['top4'])} | {core.percent(m['coverage'])} |")
        lines += ['',f"所选参数（{t} 秒）：`{json.dumps(r['selected']['config'],ensure_ascii=False)}`",'',
            f"相同被接受测试截点：旧近邻 Top1 {core.percent(r['same_accepted_test_baselines']['selected']['top1'])}，近期访问 Top1 {core.percent(r['same_accepted_test_baselines']['recency']['top1'])}。",
            f"严格完整来源覆盖测试：接受 {r['test_strict']['n']}/{r['test_strict']['eligible_n']}，Top1 {core.percent(r['test_strict']['top1'])}。",
            f"暂不预测原因：{r['test_decisions']}。",'']
    lines += ['## 研究依据与限制','',
        '- [Selective Classification](https://arxiv.org/abs/1705.08500)：同时报告准确率与覆盖率，证据不足允许不预测。本实现没有套用论文的神经网络或风险保证。',
        '- [Radius Neighbors](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.RadiusNeighborsClassifier.html)：按相似度范围收集证据，而非固定必须取若干邻居。此处使用项目自己的混合上下文特征。',
        '- 旧数据覆盖未知仍只能算部分观测证据；严格来源结果单列。原后期测试此前已经查看，属于探索性历史回放。',
        '- 发现历史辅助组件 KeyboardSetupAssistant 后，按已确认范围在派生研究输入、候选和学习标签中排除；原始数据不删除。',
        '- 本轮缺少证据的场景明确暂不预测，不研究兜底策略，也不改变实际菜单。']
    (out/'report-private.md').write_text('\n'.join(lines)+'\n')


if __name__=='__main__':main()
