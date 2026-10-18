#!/usr/bin/env python3
"""Online conditional context rules with clustered support and selective prediction."""
import argparse
from collections import Counter,defaultdict
from dataclasses import replace
import itertools
import json
import os
from pathlib import Path

import evaluate_context_knn as live_core
import evaluate_evidence_model as selective
from diagnose_dwell_errors import frozen_core
from summarize_tuned_history import verify_run

VERSION='selective-context-rules-v1'


def conditions(q):
    foreground={
        'exact':q.current,
        'semantic':'|'.join(sorted(k for k in q.foreground if k.startswith(('kind:','app:','origin:'))))}
    prior=tuple(sorted(k for k in q.history if k.startswith('position:1:app:')))
    histories={'none':()}
    if prior:histories['previous_app']=prior
    background=[]
    for family,tokens in q.background.items():
        if q.source_quality.get(family,0)<.25:continue
        for token in tokens:
            if token.startswith(('app:','origin:','selected:app:','selected:origin:')):
                background.append((family,token))
    return [(mode,fg,history,h,bg) for mode,fg in foreground.items()
        for history,h in histories.items() for bg in background]


def query_rules(items,ledger,times,index,keys,rule_index,config):
    q=items[index];records=[]
    for key in keys[index]:
        if key[0]!=config['foreground'] or key[2]!=config['history']:continue
        outcomes={}
        for j in rule_index[key]:
            r=items[j]
            if j>=index or r.ready_index>=q.index:continue
            if q.timestamp-r.timestamp>config['lookback_hours']*3600000:continue
            # Multiple prediction anchors preceding one arrival contribute one observation.
            event=ledger[j]['target_event_id']
            outcomes[event]={'event_id':event,'position':j,'label':selective.semantic_label(r,config['representation'])}
        if len(outcomes)<2:continue
        groups=[];last_session=None;start=None
        for event,r in sorted(outcomes.items(),key=lambda kv:times[kv[0]]):
            session,timestamp=times[event]
            if session!=last_session or start is None or timestamp-start>=30000:
                groups.append([]);last_session=session;start=timestamp
            groups[-1].append(r)
        votes=Counter();support=Counter()
        for group in groups:
            for label in {r['label'] for r in group}:support[label]+=1
            for r in group:votes[r['label']]+=1/len(group)
        ranked=sorted(votes,key=lambda k:(-votes[k],k));winner=ranked[0]
        total=sum(votes.values());agreement=votes[winner]/total
        raw=[]
        for label in ranked:
            resolved=selective.resolve_label(label,q)
            if resolved and resolved not in raw:raw.append(resolved)
        mapped=selective.resolve_label(winner,q)
        records.append({'rule':key,'winner':winner,'winner_resolved':mapped,'raw':raw,
            'agreement':agreement,'winner_support_groups':support[winner],'effective_groups':len(groups),
            'evidence_groups':len(groups),'evidence_count':len(outcomes),'evidence':list(outcomes.values()),
            'rule_ranking_score':selective.wilson_lower(votes[winner],total),'votes':dict(votes)})
    return records


def predict(core,q,rules,support,agreement):
    eligible=[r for r in rules if r['winner_support_groups']>=support and r['agreement']+1e-12>=agreement and r['effective_groups']>=1.5]
    if not eligible:return {'predictions':[],'abstain_reason':'no_repeated_consistent_rule','evidence_count':0}
    best=max(eligible,key=lambda r:(r['rule_ranking_score'],r['agreement'],r['evidence_groups'],str(r['rule'])))
    # Do not hide an unreachable winner in favor of an available lower-confidence rule.
    predictions=core.select_four(best['raw'],q.candidates) if best['winner_resolved'] else []
    return {**best,'predictions':predictions,'abstain_reason':None if predictions else 'winner_unavailable_or_ambiguous'}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-run',required=True,type=Path)
    parser.add_argument('--tuning-run',required=True,type=Path)
    parser.add_argument('--output',required=True,type=Path)
    args=parser.parse_args();base=args.source_run.resolve();out=args.output.resolve()
    if not out.is_relative_to(live_core.PRIVATE_ROOT.resolve()) or out==live_core.PRIVATE_ROOT.resolve():
        parser.error('Use a new private training directory')
    source,source_hashes=verify_run(base);tuning,tune_hashes=verify_run(args.tuning_run)
    assert Path(tuning['source_run'])==base
    core=frozen_core(base);snapshot=Path(source['snapshot']);hashes,_=core.verified_snapshot(snapshot)
    assert hashes==source['input_hashes']
    events=core.load_lines(snapshot/'events.jsonl');samples=core.load_lines(snapshot/'navigation-samples.jsonl')
    original,*_=core.make_observations(events,samples,set(source['excluded_sessions']))
    times={e['id']:(e['sessionID'],e['timestampMs']) for e in events}
    blocked={core.key_for(e['data']['target'],e['sessionID']) for e in events if e['kind']=='target.update'
        and e['data']['target'].get('bundleID') in selective.DENIED_BUNDLES}
    original=[selective.clean_input(q,blocked) for q in original]
    ledgers=json.loads((base/'labels-private.json').read_text())
    baselines=json.loads((args.tuning_run/'predictions-private.json').read_text())
    n=len(original);train_end=int(n*.6);test_start=int(n*.8)
    shapes=[dict(zip(['foreground','history','lookback_hours','representation'],v))
        for v in itertools.product(['exact','semantic'],['none','previous_app'],[1.,6.,24.],['exact','window_class'])]
    gates=list(itertools.product([2,3,5],[.7,.85,1.]))
    os.umask(0o077);out.mkdir(parents=True,mode=0o700,exist_ok=False)
    core.write_json(out/'search-plan.json',{'version':VERSION,'shapes':shapes,'gates':gates,
        'min_validation_accepted':selective.MIN_VALIDATION_ACCEPTED,'target_top1':selective.TARGET_PRECISION,
        'select_before_test':True,'prediction_spacing_ms':30000,'support_cluster_ms':30000,
        'references':['https://www.cs.uic.edu/~liub/publications/papers_topics.html','https://arxiv.org/abs/1705.08500'],
        'test_previously_seen':True})
    results={};details={}
    for t,ledger in ledgers.items():
        items=list(original);valid=set()
        for r in ledger:
            i=r['position'];q=items[i]
            if r['status']!='stable_target':continue
            target=r['target']
            if q.current_app in selective.DENIED_BUNDLES or target.get('bundleID') in selective.DENIED_BUNDLES:continue
            valid.add(i);items[i]=replace(q,truth=core.key_for(target,q.session),truth_meta={**target,'_bundle':target.get('bundleID')})
        keys=[conditions(q) for q in items];rule_index=defaultdict(list)
        for i in sorted(valid):
            for key in keys[i]:rule_index[key].append(i)
        val_full=[i for i in range(train_end,test_start) if i in valid and items[i].ready_index<items[test_start].index]
        test_full=[i for i in range(test_start,n) if i in valid]
        validation=selective.independent_positions(items,val_full);test=selective.independent_positions(items,test_full)
        grid=[]
        for k,config in enumerate(shapes):
            cached={i:query_rules(items,ledger,times,i,keys,rule_index,config) for i in validation}
            for support,agreement in gates:
                records={i:predict(core,items[i],cached[i],support,agreement) for i in validation}
                m,_=selective.summarize(core,items,validation,records,support,agreement)
                grid.append({'config':{**config,'min_support':support,'min_agreement':agreement},'validation':m})
            if k%6==0:print(json.dumps({'stage':'rule_search','threshold':t,'done':k,'total':len(shapes)}),flush=True)
        best,status=selective.choose(grid);cfg=best['config']
        core.write_json(out/f'selection-{t}s.json',{'status':status,**best})
        positions=validation+test_full
        records={i:predict(core,items[i],query_rules(items,ledger,times,i,keys,rule_index,cfg),cfg['min_support'],cfg['min_agreement']) for i in positions}
        vm,_=selective.summarize(core,items,validation,records,cfg['min_support'],cfg['min_agreement']);assert vm==best['validation']
        tm,accepted=selective.summarize(core,items,test,records,cfg['min_support'],cfg['min_agreement'])
        fm,_=selective.summarize(core,items,test_full,records,cfg['min_support'],cfg['min_agreement'])
        sm,_=selective.summarize(core,items,[i for i in test if items[i].strict],records,cfg['min_support'],cfg['min_agreement'])
        comparison={}
        for model in ['selected','recency']:
            saved={r['position']:r for r in baselines[t][model]}
            comparison[model]=core.metric([core.score_row(items[i],[k for k in saved[i]['predictions'] if k in items[i].candidates]) for i in accepted])
        results[t]={'selection_status':status,'selected':best,'test':tm,'test_strict':sm,
            'all_test_cutoffs_secondary':fm,'same_accepted_test_baselines':comparison,
            'test_decisions':dict(Counter('accepted' if i in accepted else records[i]['abstain_reason'] or 'gate_failed' for i in test))}
        details[t]=[{'position':i,'time_utc':core.utc(items[i].timestamp),'sample_id':items[i].sample_id,
            'subset':'validation' if i<test_start else 'test','primary_cutoff':i in validation or i in test,
            'accepted':selective.accept(records[i],cfg['min_support'],cfg['min_agreement']),
            'truth':items[i].truth,'truth_meta':items[i].truth_meta,'predicted_meta':[items[i].candidates[k] for k in records[i]['predictions']],
            'score':core.score_row(items[i],records[i]['predictions']),**records[i]} for i in positions]
        core.write_json(out/f'validation-grid-{t}s.json',grid)
        print(json.dumps({'stage':'rule_threshold_complete','threshold':t,**results[t]}),flush=True)
    core.write_json(out/'metrics.json',results);core.write_json(out/'predictions-private.json',details)
    render_report(out,results,core)
    (out/'analysis-source.py').write_text(Path(__file__).read_text())
    (out/'evidence-source.py').write_text(Path(selective.__file__).read_text())
    core.write_json(out/'run.json',{'source_run':str(base),'source_artifact_hashes':source_hashes,
        'tuning_run':str(args.tuning_run.resolve()),'tuning_artifact_hashes':tune_hashes,'input_hashes':hashes,
        'version':VERSION,'test_previously_seen':True,'live_model_updated':False})
    core.write_json(out/'artifact-hashes.json',{p.name:core.sha(p) for p in out.iterdir() if p.is_file()})
    print(json.dumps({'output':str(out),'results':results},ensure_ascii=False))


def render_report(out,results,core):
    lines=['# 有证据情景：条件规则实验','',
        '规则包含前台具体目标或稳定语义类别、一个已观察到的后台应用/站点条件，并可要求前一步应用相同。规则从预测时已经成熟的历史标签学习；未来标签不能更新当前规则。',
        '同一次后续到达的重复锚点合并，30 秒内行为片段总权重封顶；主测试截点也相隔至少 30 秒。候选缺失的历史仍计入规则分歧。窗口类别只有唯一可达实例时才解析成具体目标。',
        '验证段预先要求至少 15 个被接受截点、Top1 至少 80%，达标后优先覆盖率。未达标会明确报告。不同规则不能当成相互独立的重复证据。','',
        '| 清洗 | 验证状态 | 验证接受/总数 | 验证 Top1 | 后期接受/总数 | 后期 Top1 | 后期 Top4 | 覆盖率 |','|---|---|---:|---:|---:|---:|---:|---:|']
    for t,r in results.items():
        v=r['selected']['validation'];m=r['test']
        lines.append(f"| {t} 秒 | {r['selection_status']} | {v['n']}/{v['eligible_n']} | {core.percent(v['top1'])} | {m['n']}/{m['eligible_n']} | {core.percent(m['top1'])} | {core.percent(m['top4'])} | {core.percent(m['coverage'])} |")
    for t,r in results.items():
        lines += ['',f'## {t} 秒档', '',f"参数：`{json.dumps(r['selected']['config'])}`",'',
            f"相同被接受截点：旧近邻 Top1 {core.percent(r['same_accepted_test_baselines']['selected']['top1'])}；近期访问 Top1 {core.percent(r['same_accepted_test_baselines']['recency']['top1'])}。",
            f"严格完整来源测试接受 {r['test_strict']['n']}/{r['test_strict']['eligible_n']}，Top1 {core.percent(r['test_strict']['top1'])}。",
            f"次要的全部原始测试截点：接受 {r['all_test_cutoffs_secondary']['n']}/{r['all_test_cutoffs_secondary']['eligible_n']}，Top1 {core.percent(r['all_test_cutoffs_secondary']['top1'])}。",
            f"决策统计：{r['test_decisions']}。"]
    lines += ['', '## 限制与研究依据','',
        '这是受[关联规则分类](https://www.cs.uic.edu/~liub/publications/papers_topics.html)和[选择性预测](https://arxiv.org/abs/1705.08500)启发的轻量在线实现，不等同于完整 CBA 算法或经过校准的风险保证。',
        '旧数据大量来源覆盖未知；规则的高历史一致率不等于校准后的成功概率。后期测试此前已查看，且多种算法尝试也增加过拟合风险；样本不足时不能宣称可靠的高准确率。原始数据及实际菜单均不改变。']
    (out/'report-private.md').write_text('\n'.join(lines)+'\n')


if __name__=='__main__':main()
