#!/usr/bin/env python3
"""Compare 3s/5s derived stable-target labels with fixed causal replay settings.

Same prediction cutoffs, features, candidates and original test boundary. Retains
the raw sequence; only matured supervision is changed. No test-based tuning.
"""
import argparse
from collections import Counter
from dataclasses import replace
import json
import os
from pathlib import Path

import describe_dwell as dwell
import evaluate_context_knn as core

VERSION='dwell-cleaning-v1'
CONFIGS={
    'knn_fixed':{'weight':.55,'k':15,'half_life_hours':6.},
    'knn_previous_validation_selected':{'weight':1.,'k':5,'half_life_hours':1.},
}


def derive_label(query, visit_by_id, events_by_id, threshold):
    """Only use target refinements/departures visible within the original 30s horizon."""
    anchor=events_by_id[query.anchor_id]
    deadline=anchor['monotonicMs']+30000
    event_id=query.next_event_id
    skipped=[]
    while True:
        v=visit_by_id.get(event_id)
        if not v or v['session']!=query.session or v['left_censored']:
            return None,{'status':'missing_or_unlinked_visit','skipped':skipped}
        if v['arrival_start_monotonic_ms']>deadline:
            return None,{'status':'no_stable_target_within_30s','skipped':skipped}
        target=v['initial_target']
        start=v['arrival_start_monotonic_ms']
        # Resolution updates after the prediction horizon must not alter this label.
        for r in v['refinements']:
            if r['monotonic_ms']<=deadline:
                target=r['target'];start=r['monotonic_ms']
        key=core.key_for(target,query.session)
        if key==query.current:
            # A -> short B -> A means returning, not permission to hunt for a later C.
            return None,{'status':'returned_to_current_target','skipped':skipped}
        observed_until=min(v['end_monotonic_ms'],deadline)
        if observed_until-start>=threshold*1000:
            meta={**target,'_bundle':target.get('bundleID')}
            return replace(query,truth=key,truth_meta=meta,
                offset_ms=v['arrival_start_monotonic_ms']-anchor['monotonicMs'],
                next_event_id=event_id),{'status':'stable_target','skipped':skipped,
                    'target_event_id':event_id,'target':target,
                    'evidence_available_monotonic_ms':start+threshold*1000,
                    'changed':key!=query.truth}
        if v['end_monotonic_ms']>deadline:
            return None,{'status':'insufficient_dwell_before_30s','skipped':skipped}
        if not v['complete']:
            return None,{'status':'observation_boundary_before_threshold','skipped':skipped}
        skipped.append({'event_id':event_id,'seconds':v['observed_seconds'],
            'category':'navigation_intermediate' if v['navigation_class']=='same_tab_navigation' else 'temporary_target',
            'navigation_class':v['navigation_class']})
        event_id=v['end_event_id']


def run_predictions(items,pairs,positions):
    output={}
    for name in ['recency','markov',*CONFIGS]:
        m,rows,_=core.evaluate(items,pairs,positions,CONFIGS.get(name),
            name if name in {'recency','markov'} else None)
        output[name]={'all':m,'strict':core.metric([r for r in rows if items[r['position']].strict]),
            'unknown_coverage':core.metric([r for r in rows if not items[r['position']].strict]),'rows':rows}
    return output


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--snapshot',required=True,type=Path)
    parser.add_argument('--output',required=True,type=Path)
    args=parser.parse_args();output=args.output.expanduser().resolve()
    if not output.is_relative_to(core.PRIVATE_ROOT.resolve()) or output==core.PRIVATE_ROOT.resolve():
        parser.error('Use a new private SmartApp/training directory')
    os.umask(0o077);output.mkdir(parents=True,mode=0o700,exist_ok=False)
    hashes,_=core.verified_snapshot(args.snapshot)
    events=core.load_lines(args.snapshot/'events.jsonl')
    samples=core.load_lines(args.snapshot/'navigation-samples.jsonl')
    visits,apps,dwell_audit=dwell.extract_visits(events,samples,core.KNOWN_TEST_SESSIONS)
    items,audit,_,_,_=core.make_observations(events,samples,core.KNOWN_TEST_SESSIONS)
    byevent={e['id']:e for e in events};byvisit={v['arrival_event_id']:v for v in visits}
    n=len(items);test_start=int(n*.8);test=list(range(test_start,n))
    pairs=[]
    for i,q in enumerate(items):
        pairs.append([(j,core.similarity(q,items[j])) for j in core.eligible_neighbors(items,i)])
        if i and i%300==0:print(json.dumps({'stage':'causal_pairs','done':i,'total':n}),flush=True)
    baseline=run_predictions(items,pairs,test)
    page_complete=[v for v in visits if v['complete'] and v['target']['kind']=='page']
    page_switch=[v for v in page_complete if v['navigation_class'] in
        {'application_switch','browser_window_switch','same_window_tab_switch'}]
    metrics={'version':VERSION,'audit':audit,'dwell_audit':dwell_audit,
        'baseline':{k:{x:y for x,y in v.items() if x!='rows'} for k,v in baseline.items()},
        'page_transition_classes':dict(Counter(v['navigation_class'] for v in page_complete)),
        'pages_all':dwell.summarize(page_complete),'pages_confirmed_switch':dwell.summarize(page_switch),
        'pages_excluding_same_tab_navigation':dwell.summarize([v for v in page_complete if v['navigation_class']!='same_tab_navigation']),
        'page_thresholds':dwell.threshold_diagnostics(page_complete),'variants':{}}
    ledgers={};predictions={}
    for threshold in (3,5):
        modified=[];valid=set();ledger=[]
        for i,q in enumerate(items):
            new,decision=derive_label(q,byvisit,byevent,threshold)
            if new:
                valid.add(i)
                assert decision['evidence_available_monotonic_ms']<=byevent[q.anchor_id]['monotonicMs']+30000
            modified.append(new or q)
            ledger.append({'position':i,'sample_id':q.sample_id,'anchor_id':q.anchor_id,
                'input_through_event_index':q.index,'label_ready_event_index':q.ready_index,**decision})
        filtered_pairs=[[(j,s) for j,s in row if j in valid] for row in pairs]
        positions=[i for i in test if i in valid]
        trained=run_predictions(modified,filtered_pairs,positions)
        # Frozen original-training predictions scored on exactly the same revised truths.
        raw_on_revised={}
        for name,r in baseline.items():
            scored=[{'position':row['position'],**core.score_row(modified[row['position']],row['predictions'])}
                for row in r['rows'] if row['position'] in valid]
            raw_on_revised[name]=core.metric(scored)
        case_results={}
        for name,r in trained.items():
            misses=[row for row in r['rows'] if not row['hit4']]
            case_results[name]=dict(Counter('outside_candidate_set' if not x['covered'] else
                'right_app_wrong_destination' if x['app_hit4'] else 'ranking_or_sparse_history' for x in misses))
        metrics['variants'][str(threshold)]={'eligible_all':len(valid),'eligible_test':len(positions),
            'status_all':dict(Counter(r['status'] for r in ledger)),
            'status_test':dict(Counter(ledger[i]['status'] for i in test)),
            'changed_labels_test':sum(ledger[i].get('changed',False) for i in positions),
            'training_and_labels_cleaned':{k:{x:y for x,y in v.items() if x!='rows'} for k,v in trained.items()},
            'original_training_scored_on_same_clean_labels':raw_on_revised,'miss_categories':case_results}
        predictions[str(threshold)]={k:v['rows'] for k,v in trained.items()}
        ledgers[str(threshold)]=ledger
    # A common evaluation set with identical targets isolates the effect of choosing 3s vs 5s supervision.
    common=[i for i in test if all(ledgers[t][i]['status']=='stable_target' for t in ['3','5'])
        and core.key_for(ledgers['3'][i]['target'],items[i].session)==core.key_for(ledgers['5'][i]['target'],items[i].session)]
    metrics['common_identical_test_targets']={'n':len(common),'variants':{}}
    for t in ['3','5']:
        metrics['common_identical_test_targets']['variants'][t]={name:core.metric([r for r in rows if r['position'] in common])
            for name,rows in predictions[t].items()}
    core.write_json(output/'metrics.json',metrics)
    core.write_json(output/'labels-private.json',ledgers)
    core.write_json(output/'predictions-private.json',predictions)
    with (output/'visits-private.jsonl').open('w') as f:
        for v in visits:f.write(json.dumps(v,ensure_ascii=False)+'\n')
    render_report(output,metrics)
    for module,name in [(core,'core-source.py'),(dwell,'dwell-source.py')]:
        (output/name).write_text(Path(module.__file__).read_text())
    (output/'analysis-source.py').write_text(Path(__file__).read_text())
    core.write_json(output/'run.json',{'version':VERSION,'snapshot':str(args.snapshot.resolve()),
        'input_hashes':hashes,'configs':CONFIGS,'thresholds_seconds':[3,5],
        'test_boundary_original_position':test_start,'no_retuning':True,
        'raw_data_modified':False,'existing_runs_modified':False,
        'learning_inputs':'unchanged_as_observed','label_ready_rule':'original_complete_30_second_horizon',
        'label_rule':'skip completed visits shorter than threshold, stop on return or observation boundary; require stable target within 30 seconds',
        'excluded_sessions':sorted(core.KNOWN_TEST_SESSIONS)})
    core.write_json(output/'artifact-hashes.json',{p.name:core.sha(p) for p in output.iterdir() if p.is_file()})
    print(json.dumps({'output':str(output),'page_transition_classes':metrics['page_transition_classes'],
        'variants':metrics['variants'],'common':metrics['common_identical_test_targets']},ensure_ascii=False),flush=True)


def render_report(output,m):
    lines=['# 区分连续导航后的 3 秒 / 5 秒清洗实验','',
        '连续导航依据同会话中、截至各次观察的库存入口确认：同应用、同窗口 containerID、同标签页入口且 URL 改变。不同标签页单独计为切换，缺失标识计为未知。它不能区分自动跳转与主动点击，也不证明操作正确。','',
        '仅清洗派生学习标签：短暂连续导航记为导航中间步骤，其他短停留记为疑似临时目标；二者都可跳过，保留原因和原始顺序。未知时长不跳过。A→短 B→A 单列返回当前目标，不继续寻找 C。','',
        '预测仍在原始截点进行，输入历史、状态和候选不变；同次到达的目标细化只在已观察到后用于标签。30 秒内必须观察到目标稳定达到阈值，标签仍满 30 秒后才能进入近邻记忆。','',
        '## 完整网页区间的后续行为','', '| 类别 | 数量 |','|---|---:|']
    lines += [f'| {k} | {v} |' for k,v in m['page_transition_classes'].items()]
    lines += ['','## 阈值范围','', '| 阈值 | 全部短网页 | 同标签页连续导航 | 其余短网页 |','|---|---:|---:|---:|']
    for t in ['3','5']:
        r=m['page_thresholds'][t];nav=r['navigation_classes'].get('same_tab_navigation',0)
        lines.append(f"| <{t} 秒 | {r['n']} | {nav} | {r['n']-nav} |")
    lines += ['','## 固定时间分割的历史回放','',
        '原始评估样本和测试起点保持不变；配置沿用初始固定参数与上次验证段选出的参数，本轮不重调参。原测试集已在此前查看，本轮是探索性比较，不能当作新的独立留出验证。不同阈值的样本及标签有变化，不能仅比较各自命中率宣称模型提升。','',
        '| 口径 | 测试数 | 近期访问 Top4 | 固定近邻 Top4 | 上次选参近邻 Top4 |','|---|---:|---:|---:|---:|']
    b=m['baseline'];lines.append(f"| 原始标签 | {b['recency']['all']['n']} | {core.percent(b['recency']['all']['top4'])} | {core.percent(b['knn_fixed']['all']['top4'])} | {core.percent(b['knn_previous_validation_selected']['all']['top4'])} |")
    for t,v in m['variants'].items():
        r=v['training_and_labels_cleaned']
        lines.append(f"| <{t} 秒跳过 | {v['eligible_test']} | {core.percent(r['recency']['all']['top4'])} | {core.percent(r['knn_fixed']['all']['top4'])} | {core.percent(r['knn_previous_validation_selected']['all']['top4'])} |")
    lines += ['','## 相同清洗标签：原训练与清洗训练对照','',
        '| 阈值 | 模型 | 原训练 Top4 | 清洗训练 Top4 |','|---|---|---:|---:|']
    for t,v in m['variants'].items():
        for name in CONFIGS:
            lines.append(f"| {t} 秒 | {name} | {core.percent(v['original_training_scored_on_same_clean_labels'][name]['top4'])} | {core.percent(v['training_and_labels_cleaned'][name]['all']['top4'])} |")
    lines += ['','## 3 秒与 5 秒的共同测试子集','',
        f"截点和清洗后正确目标均相同：{m['common_identical_test_targets']['n']} 个。",'',
        '| 阈值 | 固定近邻 Top4 | 上次选参近邻 Top4 |','|---|---:|---:|']
    for t,r in m['common_identical_test_targets']['variants'].items():
        lines.append(f"| {t} 秒 | {core.percent(r['knn_fixed']['top4'])} | {core.percent(r['knn_previous_validation_selected']['top4'])} |")
    lines += ['','## 边界与审计','',
        '- 大部分旧记录来源覆盖未知，严格完整覆盖结果另存 metrics.json；无观测边界保证的记录不伪造标签。',
        '- 同目标重复观察只有入口相同才合并；旧版停留脚本曾合并不同入口的相同 URL，因此本次计数可能略有变化。',
        '- 模型原型在后台缺失时仍可能高估状态相似度；本实验保持评分实现不变，以隔离标签处理变化。',
        '- labels-private.json 保存每个截点的跳过步骤、原因、标签证据时间及不纳入原因；predictions-private.json 保存预测。',
        '- 原始数据与此前 run 均未改写，本轮未更新正在运行的推荐菜单。']
    (output/'report-private.md').write_text('\n'.join(lines)+'\n')


if __name__=='__main__':main()
