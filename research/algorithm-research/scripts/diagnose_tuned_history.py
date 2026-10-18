#!/usr/bin/env python3
"""Trace failures with mature target history under the validation-selected model."""
import argparse
from collections import Counter
from dataclasses import replace
import json
import os
from pathlib import Path

import evaluate_context_knn as live_core
from diagnose_dwell_errors import frozen_core,label
from summarize_tuned_history import verify_run


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tuning-run',required=True,type=Path)
    parser.add_argument('--output',required=True,type=Path)
    args=parser.parse_args();run=args.tuning_run.resolve();out=args.output.resolve()
    if not out.is_relative_to(live_core.PRIVATE_ROOT.resolve()) or out==live_core.PRIVATE_ROOT.resolve():
        parser.error('Use a new private training directory')
    tuning,tune_hashes=verify_run(run);base=Path(tuning['source_run']);source,base_hashes=verify_run(base)
    core=frozen_core(base);snapshot=Path(tuning['snapshot']);hashes,_=core.verified_snapshot(snapshot)
    assert hashes==tuning['input_hashes']
    events=core.load_lines(snapshot/'events.jsonl');samples=core.load_lines(snapshot/'navigation-samples.jsonl')
    original,*_=core.make_observations(events,samples,set(source['excluded_sessions']))
    ledger=json.loads((base/'labels-private.json').read_text())
    results=json.loads((run/'metrics.json').read_text())['results']
    predictions=json.loads((run/'predictions-private.json').read_text())
    names={e['app']['bundleID']:e['app']['name'] for e in events if e.get('app',{}).get('name')}
    metadata={core.key_for(e['data']['target'],e['sessionID']):e['data']['target'] for e in events if e['kind']=='target.update'}
    metrics={};cases={}
    for t,ls in ledger.items():
        items=list(original);valid=set()
        for r in ls:
            i=r['position'];q=items[i]
            assert q.sample_id==r['sample_id']
            if r['status']=='stable_target':
                target=r['target'];valid.add(i)
                items[i]=replace(q,truth=core.key_for(target,q.session),truth_meta={**target,'_bundle':target.get('bundleID')})
        config=results[t]['best_config'];records=[];warm_n=0
        for row in predictions[t]['selected']:
            i=row['position'];q=items[i]
            mature=[j for j in core.eligible_neighbors(items,i) if j in valid]
            support=[j for j in mature if items[j].truth==q.truth]
            if not support:continue
            warm_n+=1
            if row['hit4']:continue
            ps=[[] for _ in items];ps[i]=[(j,core.similarity(q,items[j])) for j in mature]
            shown,extra=core.predict_knn(items,i,ps,**config)
            assert shown==row['predictions'] and extra['neighbor_examples']==row['neighbor_examples']
            assert all(core.score_row(q,shown)[key]==row[key] for key in core.score_row(q,shown))
            ranked=sorted([(core.combined_similarity(parts,config['weight']),j,parts) for j,parts in ps[i]
                if items[j].truth in q.candidates],key=lambda x:(x[0],items[x[1]].index),reverse=True)
            candidates=[x for x in ranked if x[0]>0]
            correct_ranks=[rank+1 for rank,x in enumerate(candidates) if items[x[1]].truth==q.truth]
            chosen=candidates[:config['k']]
            truth=next((x for x in candidates if items[x[1]].truth==q.truth),None)
            def describe(x):
                sim,j,parts=x
                return {'sample_id':items[j].sample_id,'current':label(metadata.get(items[j].current,{}),names),
                    'next':label(items[j].truth_meta,names),'next_meta':items[j].truth_meta,
                    'similarity':sim,'foreground_similarity':parts[3],'background_similarity':parts[4],
                    'background_reliability':parts[5],'history_similarity':parts[2],
                    'age_minutes':max(0,(q.timestamp-items[j].timestamp)/60000)}
            reason='not_candidate' if not row['covered'] else (
                'hierarchy_suppressed' if q.truth in row['raw'][:4] and q.truth not in shown else
                'not_in_top_k' if not row['neighbor_label_support'] else 'outvoted')
            same_current=[j for j in support if items[j].current==q.current]
            records.append({'position':i,'sample_id':q.sample_id,'time_utc':core.utc(q.timestamp),
                'current':label(metadata.get(q.current,{}),names),'actual':label(q.truth_meta,names),
                'actual_meta':q.truth_meta,'predicted':[label(q.candidates[k],names) for k in shown],
                'predicted_meta':[q.candidates[k] for k in shown],'reason':reason,'kind':q.truth_meta['kind'],
                'strict':q.strict,'mature_support':len(support),'same_current_support':len(same_current),
                'best_correct_neighbor_rank':correct_ranks[0] if correct_ranks else None,
                'best_correct_neighbor':describe(truth) if truth else None,
                'chosen_neighbors':[describe(x) for x in chosen],
                'query_background_count':q.background_count,'query_source_quality':q.source_quality})
        retrieval=[r for r in records if r['reason']=='not_in_top_k']
        metrics[t]={'warm_n':warm_n,'warm_misses':len(records),'reasons':dict(Counter(r['reason'] for r in records)),
            'retrieval_misses_with_same_current_history':sum(r['same_current_support']>0 for r in retrieval),
            'retrieval_misses_without_same_current_history':sum(r['same_current_support']==0 for r in retrieval),
            'best_correct_neighbor_ranks':dict(Counter('4-10' if r['best_correct_neighbor_rank']<=10 else
                '11-30' if r['best_correct_neighbor_rank']<=30 else '>30' for r in retrieval)),
            'candidate_miss_kinds':dict(Counter(r['kind'] for r in records if r['reason']=='not_candidate'))}
        cases[t]=records
    os.umask(0o077);out.mkdir(parents=True,mode=0o700,exist_ok=False)
    core.write_json(out/'metrics.json',metrics);core.write_json(out/'cases-private.json',cases)
    lines=['# 新参数下：有历史却未命中的样本','',
        '“有历史”仅表示预测时有至少一个成熟清洗标签指向正确目标，不保证该历史的前台、后台和此前路径与当前相同。这里重算新参数预测并核对已保存结果，参数与标签保持不变。','',
        '| 规则 | 有历史测试数 | 错误数 | 正确历史未进入 k 个近邻 | 不在候选 | 展示去重 | 投票落后 |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for t,m in metrics.items():
        c=m['reasons'];lines.append(f"| {t} 秒 | {m['warm_n']} | {m['warm_misses']} | {c.get('not_in_top_k',0)} | {c.get('not_candidate',0)} | {c.get('hierarchy_suppressed',0)} | {c.get('outvoted',0)} |")
    for t,m in metrics.items():
        lines += ['',f'## {t} 秒档','',
            f"未检索成功的错误里，{m['retrieval_misses_with_same_current_history']} 个已有相同当前具体目标的历史，{m['retrieval_misses_without_same_current_history']} 个没有；相同前台仍不等于相同完整情景。",
            f"正确目标历史的最佳相似度名次分布：{m['best_correct_neighbor_ranks']}。",'']
        selected=sorted((r for r in cases[t] if r['reason']=='not_in_top_k'),key=lambda r:(-r['same_current_support'],-r['mature_support']))[:3]
        for r in selected:
            lines += [f"- {r['time_utc']}：{r['current']} → 实际 {r['actual']}；预测：{'、'.join(r['predicted'])}。",
                f"  - 正确目标成熟样本 {r['mature_support']} 个，其中当前目标相同 {r['same_current_support']} 个；最佳正确近邻排第 {r['best_correct_neighbor_rank']}。"]
            for title,x in [('选中的最近邻',r['chosen_neighbors'][0]),('最接近的正确目标历史',r['best_correct_neighbor'])]:
                lines.append(f"  - {title}：{x['current']} → {x['next']}；总相似度 {x['similarity']:.3f}，前台 {x['foreground_similarity']:.3f}，后台 {x['background_similarity']:.3f}（共同证据权重 {x['background_reliability']:.3f}），历史路径 {x['history_similarity']:.3f}。")
    lines += ['', '当前实现先按相似度选 k 个样本，再做时间衰减投票；增加近邻数未必提升结果，不能仅凭被截断就断言扩大 k 一定有效。参数只能调节既有特征，无法补回未观察到的任务意图。']
    (out/'report-private.md').write_text('\n'.join(lines)+'\n')
    (out/'analysis-source.py').write_text(Path(__file__).read_text())
    core.write_json(out/'run.json',{'tuning_run':str(run),'tuning_artifact_hashes':tune_hashes,
        'source_artifact_hashes':base_hashes,'input_hashes':hashes,'predictions_verified':True,'predictions_changed':False})
    core.write_json(out/'artifact-hashes.json',{p.name:core.sha(p) for p in out.iterdir() if p.is_file()})
    print(json.dumps({'output':str(out),'metrics':metrics},ensure_ascii=False))


if __name__=='__main__':main()
