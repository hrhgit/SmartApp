#!/usr/bin/env python3
"""Audit an immutable dwell-cleaning replay without changing predictions or labels."""
import argparse
from collections import Counter, defaultdict
from dataclasses import replace
import importlib.util
import json
import os
from pathlib import Path
import sys
from urllib.parse import urlsplit

import evaluate_context_knn as current_core

MODEL='knn_previous_validation_selected'
REASONS={
    'not_observed_before_cutoff':'预测时尚未观察过这个具体目标',
    'known_but_not_candidate':'目标曾被观察过，但当时不在候选中',
    'no_mature_clean_label':'目标可推荐，但没有已成熟的清洗后学习样本',
    'truth_not_retrieved':'有可用的正确目标历史样本，但没有选进近邻',
    'hierarchy_suppressed':'原排序靠前，但被层级去重移出展示',
    'supported_but_outvoted':'近邻支持正确目标，但投票排名落后',
}


def classify(covered,known,support,neighbor_support,suppressed):
    if not covered:return 'known_but_not_candidate' if known else 'not_observed_before_cutoff'
    if not support:return 'no_mature_clean_label'
    if suppressed:return 'hierarchy_suppressed'
    if not neighbor_support:return 'truth_not_retrieved'
    return 'supported_but_outvoted'


def frozen_core(run):
    spec=importlib.util.spec_from_file_location('dwell_diagnosis_frozen_core',run/'core-source.py')
    module=importlib.util.module_from_spec(spec)
    sys.modules[spec.name]=module;spec.loader.exec_module(module)
    return module


def label(meta,names):
    app=names.get(meta.get('_bundle') or meta.get('bundleID'),meta.get('_bundle') or meta.get('bundleID') or '')
    kind=meta.get('kind')
    if kind=='page':return '网页 '+str(urlsplit(meta.get('locator','')).hostname or '(地址未知)')
    if kind=='directory':return '目录 '+Path(urlsplit(meta.get('locator','')).path).name
    if kind=='file':return '文件 '+Path(urlsplit(meta.get('locator','')).path).name
    return (app or meta.get('name') or kind or '未知')+(' 窗口' if kind=='window' else '')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',required=True,type=Path)
    parser.add_argument('--output',required=True,type=Path)
    args=parser.parse_args();run=args.run.expanduser().resolve();output=args.output.expanduser().resolve()
    if not output.is_relative_to(current_core.PRIVATE_ROOT.resolve()) or output==current_core.PRIVATE_ROOT.resolve():
        parser.error('Output must be a new private training directory')
    provenance=json.loads((run/'run.json').read_text())
    checks=json.loads((run/'artifact-hashes.json').read_text())
    for name,digest in checks.items():
        if Path(name).name!=name or current_core.sha(run/name)!=digest:raise ValueError('Changed source run artifact')
    core=frozen_core(run);snapshot=Path(provenance['snapshot'])
    hashes,_=core.verified_snapshot(snapshot)
    if hashes!=provenance['input_hashes']:raise ValueError('Changed snapshot')
    events=core.load_lines(snapshot/'events.jsonl');samples=core.load_lines(snapshot/'navigation-samples.jsonl')
    items,*_=core.make_observations(events,samples,set(provenance['excluded_sessions']))
    predictions=json.loads((run/'predictions-private.json').read_text())
    ledgers=json.loads((run/'labels-private.json').read_text())
    config=provenance['configs'][MODEL]
    names={e['app']['bundleID']:e['app']['name'] for e in events if e.get('app',{}).get('name')}
    first_seen={};metadata={}
    for i,e in enumerate(events):
        if e['sessionID'] in provenance['excluded_sessions']:continue
        if e['kind']=='target.update':
            t=e['data']['target']
            if core.admitted(t) and not core.synthetic(t):
                key=core.key_for(t,e['sessionID']);first_seen.setdefault(key,i);metadata[key]=t
    os.umask(0o077);output.mkdir(parents=True,mode=0o700,exist_ok=False)
    metrics={};details={};examples={}
    for threshold,ledger in ledgers.items():
        updated=list(items);valid=set()
        for r in ledger:
            i=r['position'];q=items[i]
            assert q.sample_id==r['sample_id'] and q.index==r['input_through_event_index']
            if r['status']=='stable_target':
                valid.add(i);t=r['target']
                updated[i]=replace(q,truth=core.key_for(t,q.session),truth_meta={**t,'_bundle':t.get('bundleID')})
        index_by_sample={q.sample_id:i for i,q in enumerate(updated)}
        records=[]
        for row in predictions[threshold][MODEL]:
            i=row['position'];q=updated[i]
            assert i in valid
            scored=core.score_row(q,row['predictions'])
            assert all(scored[k]==row[k] for k in scored),'Prediction/label mismatch'
            mature=[j for j in range(i) if j in valid and updated[j].ready_index<q.index]
            support=[j for j in mature if updated[j].truth==q.truth]
            neighbor_indices=[index_by_sample[s] for s in row['neighbor_examples']]
            assert len(neighbor_indices)==row['neighbors'] and len(neighbor_indices)<=config['k']
            assert all(j in mature for j in neighbor_indices)
            assert sum(updated[j].truth==q.truth for j in neighbor_indices)==row['neighbor_label_support']
            ns=[]
            for j in neighbor_indices:
                parts=core.similarity(q,updated[j]);sim=core.combined_similarity(parts,config['weight'])
                ns.append({'sample_id':updated[j].sample_id,'current':label(metadata.get(updated[j].current,{}),names),
                    'actual':label(updated[j].truth_meta,names),'actual_meta':updated[j].truth_meta,
                    'similarity':sim,'foreground_similarity':parts[3],'background_similarity':parts[4],
                    'background_reliability':parts[5],'history_similarity':parts[2],
                    'background_count':updated[j].background_count})
            if ns:assert abs(max(x['similarity'] for x in ns)-row['best_similarity'])<1e-10
            top_only_fg=any(x['similarity']>=.999 and x['background_reliability']==0 for x in ns)
            raw_support=sum(items[j].truth==q.truth for j in range(i) if items[j].ready_index<q.index)
            reason='hit' if row['hit4'] else classify(row['covered'],first_seen.get(q.truth,float('inf'))<=q.index,
                len(support),row['neighbor_label_support'],q.truth in row['raw'][:4] and q.truth not in row['predictions'])
            recency_hit=next(r['hit4'] for r in predictions[threshold]['recency'] if r['position']==i)
            available_sims=[(core.combined_similarity(core.similarity(q,updated[j]),config['weight']),j) for j in support]
            best_true=max(available_sims,default=(None,None))
            records.append({'position':i,'time_utc':core.utc(q.timestamp),'sample_id':q.sample_id,'reason':reason,
                'strict':q.strict,'coverage':q.coverage,'current':label(metadata.get(q.current,{}),names),
                'actual':label(q.truth_meta,names),'actual_meta':q.truth_meta,
                'predicted':[label(q.candidates[k],names) for k in row['predictions']],
                'predicted_meta':[q.candidates[k] for k in row['predictions']],
                'kind':q.truth_meta['kind'],'query_background_count':q.background_count,
                'source_quality':q.source_quality,'mature_clean_support':len(support),'mature_raw_support':raw_support,
                'neighbor_label_support':row['neighbor_label_support'],'best_similarity':row['best_similarity'],
                'best_available_truth_similarity':best_true[0],
                'perfect_similarity_without_background':top_only_fg,'neighbors':ns,
                'same_application_in_predictions':row['app_hit4'],'recency_hit':recency_hit,
                'changed_label':ledger[i].get('changed',False),'skipped':ledger[i]['skipped']})
        misses=[r for r in records if r['reason']!='hit'];hits=[r for r in records if r['reason']=='hit']
        summary={'evaluated':len(records),'hits':len(hits),'misses':len(misses),
            'primary_reasons':dict(Counter(r['reason'] for r in misses)),
            'by_kind':{k:{'all':sum(r['kind']==k for r in records),'misses':sum(r['kind']==k for r in misses)}
                for k in sorted({r['kind'] for r in records})},
            'strict_primary_reasons':dict(Counter(r['reason'] for r in misses if r['strict'])),
            'strict_test_n':sum(r['strict'] for r in records),
            'overlapping_flags':{
                'no_correct_target_in_selected_neighbors':sum(r['neighbor_label_support']==0 for r in misses),
                'perfect_similarity_without_background':sum(r['perfect_similarity_without_background'] for r in misses),
                'right_app_wrong_target':sum(r['same_application_in_predictions'] for r in misses),
                'raw_examples_exist_but_no_clean_examples':sum(r['mature_raw_support']>0 and not r['mature_clean_support'] for r in misses),
                'recency_hits_knn_misses':sum(r['recency_hit'] for r in misses),
                'knn_hits_recency_misses':sum(not r['recency_hit'] for r in hits),
                'changed_label':sum(r['changed_label'] for r in misses)},
            'primary_reason_by_kind':{reason:dict(Counter(r['kind'] for r in misses if r['reason']==reason)) for reason in REASONS},
            'most_common_missed_transitions':[
                {'current':a,'actual':b,'count':n} for (a,b),n in Counter((r['current'],r['actual']) for r in misses).most_common(10)]}
        metrics[threshold]=summary;details[threshold]=records
        chosen=[]
        for reason in REASONS:
            candidates=[r for r in misses if r['reason']==reason]
            candidates.sort(key=lambda r:(-r['mature_clean_support'] if reason=='truth_not_retrieved' else 0,
                -int(r['perfect_similarity_without_background']),r['position']))
            chosen.extend(candidates[:2])
        examples[threshold]=chosen
    core.write_json(output/'metrics.json',metrics)
    core.write_json(output/'cases-private.json',details)
    render_report(output,metrics,examples,config)
    (output/'analysis-source.py').write_text(Path(__file__).read_text())
    core.write_json(output/'run.json',{'source_run':str(run),'source_artifact_hashes':checks,'input_hashes':hashes,
        'model':MODEL,'config':config,'predictions_or_labels_changed':False})
    core.write_json(output/'artifact-hashes.json',{p.name:core.sha(p) for p in output.iterdir() if p.is_file()})
    print(json.dumps({'output':str(output),'metrics':metrics},ensure_ascii=False))


def render_report(output,metrics,examples,config):
    lines=['# 清洗后的错误诊断','',
        '复用上一轮固定预测、标签与时间分割，不训练、不改阈值；重新核对每个预测分数、成熟学习样本和所选近邻。错误定义为正确具体目标不在 Top4。',
        f"使用上次验证段配置：状态权重 {config['weight']}、近邻数 {config['k']}、衰减半衰期 {config['half_life_hours']} 小时。状态权重 1 表示本配置不使用历史序列相似度。",'',
        '下表为按固定优先级互斥分类的失败位置，不等同于已经验证的行为因果。后面的附加标记可以重叠。','',
        '| 原因 | 3 秒 | 5 秒 |','|---|---:|---:|']
    for reason,title in REASONS.items():
        lines.append(f"| {title} | {metrics['3']['primary_reasons'].get(reason,0)} | {metrics['5']['primary_reasons'].get(reason,0)} |")
    for t,m in metrics.items():
        lines += ['',f'## {t} 秒规则','',f"测试 {m['evaluated']} 个，命中 {m['hits']}，错误 {m['misses']}。",'',
            '### 错误目标类型','', '| 类型 | 测试数 | 错误数 |','|---|---:|---:|']
        lines += [f"| {k} | {v['all']} | {v['misses']} |" for k,v in m['by_kind'].items()]
        lines += ['', '### 可重叠现象', '', *[f'- {k}: {v}' for k,v in m['overlapping_flags'].items()], '',
            '### 具体案例','']
        for r in examples[t]:
            lines += [f"- {r['time_utc']}：{r['current']} → 实际 {r['actual']}；预测：{'、'.join(r['predicted'])}。",
                f"  - {REASONS[r['reason']]}；已成熟正确目标样本 {r['mature_clean_support']}，选中近邻支持 {r['neighbor_label_support']}，当前后台入口 {r['query_background_count']}。"]
            for n in r['neighbors'][:2]:
                lines.append(f"  - 近邻：{n['current']} → {n['actual']}；相似度 {n['similarity']:.3f}，后台共同证据权重 {n['background_reliability']:.3f}，历史相似度 {n['history_similarity']:.3f}。")
    lines += ['', '## 解释边界','',
        '- 具体目标首次出现可能是新 URL、目录，也可能是新会话的窗口实例；不等于首次使用该应用。',
        '- 后台共同证据为零时仍可能因前台相同得到 1.0 相似度。这证明评分的证据不足，不能仅据此认定它造成了全部错误。',
        '- 清洗可以把同一次到达细化为网页或目录，也可以跳过短暂目标；因此原始标签样本数与清洗标签样本数并非同一含义。',
        '- 近期访问命中而近邻未命中的对照说明当前排序存在改进空间，不证明用户具有单一稳定意图。',
        '- 旧覆盖未知数据和少量严格完整样本分开统计；当前结果不能替代新数据上的前瞻验证。',
        '- 完整目标定位符只在 cases-private.json 中保存，报告展示应用及域名等简要描述。']
    (output/'report-private.md').write_text('\n'.join(lines)+'\n')


if __name__=='__main__':main()
