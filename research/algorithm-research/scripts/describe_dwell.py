#!/usr/bin/env python3
"""Observed foreground dwell, censoring and reversible transient-label candidates.

Uses the full immutable event stream, not the 30-second prediction sample subset.
No raw event, target, or training label is deleted. Standard library analysis;
optional matplotlib for the standalone aggregate plot.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from urllib.parse import urlsplit

import evaluate_context_knn as core
from known_inactivity import annotate_events, active_time, is_boundary

VERSION = 'observed-dwell-v3'
BINS = [0, .5, 1, 2, 3, 5, 10, 30, 60, 180, 600, math.inf]
BIN_NAMES = ['0–0.5 s', '0.5–1 s', '1–2 s', '2–3 s', '3–5 s', '5–10 s',
             '10–30 s', '30–60 s', '1–3 min', '3–10 min', '≥10 min']
THRESHOLDS = [.5, 1, 2, 3, 5]


def compact_target(t, bundle=None):
    return {k:v for k,v in {**t, 'bundleID':bundle or t.get('bundleID')}.items()
            if k in {'id','name','kind','locator','bundleID','origin'}}


def identity(t, session):
    return core.key_for(t, session)


def overlap(start, end, idle_intervals):
    return sum(max(0., min(end,b)-max(start,a)) for a,b in idle_intervals)


def navigation_class(target, entrance, next_target, next_entrance):
    """Classify observed transitions, not user intent or whether a click occurred."""
    if not next_target:
        return 'unobserved_exit'
    a,b=target.get('bundleID'),next_target.get('bundleID')
    if a and b and a!=b:
        return 'application_switch'
    if target['kind']!='page' or next_target['kind']!='page':
        return 'non_page_transition'
    if not a or a!=b or not entrance or not next_entrance:
        return 'unknown_browser_transition'
    window_a,window_b=entrance.get('containerID'),next_entrance.get('containerID')
    if not window_a or not window_b:
        return 'unknown_browser_transition'
    if window_a!=window_b:
        return 'browser_window_switch'
    tab_a,tab_b=entrance.get('id'),next_entrance.get('id')
    if not entrance.get('tabID') or not next_entrance.get('tabID') or not tab_a or not tab_b:
        return 'unknown_browser_transition'
    if tab_a!=tab_b:
        return 'same_window_tab_switch'
    return 'same_tab_navigation' if target.get('locator')!=next_target.get('locator') else 'same_tab_same_page'


def extract_visits(events, samples, excluded_sessions):
    """Arrival episodes with as-observed refinements and explicit censor boundaries."""
    events = annotate_events(events)
    sample_by_anchor = {s['anchorEventID']:s for s in samples}
    catalog, current, current_app = {}, None, None
    inventories = {}
    visits, apps = [], []
    session, segment, idle_start = None, 0, None
    idle_intervals = defaultdict(list)
    previous_target = None
    audit = Counter()

    def close_visit(e, reason, next_target=None, next_entrance=None):
        nonlocal current
        if current is None:
            return
        current['end_ms'] = e['timestampMs']
        current['end_monotonic_ms'] = e['monotonicMs']
        current['end_active_ms'] = active_time(e)
        current['end_event_id'] = e['id']
        current['end_reason'] = reason
        current['next_target'] = compact_target(next_target) if next_target else None
        current['next_entrance'] = next_entrance
        current['navigation_class'] = navigation_class(current['target'],current['entrance'],next_target,next_entrance)
        current['observed_seconds'] = max(0., (active_time(e)-current['target_start_active_ms'])/1000)
        current['arrival_episode_seconds'] = max(0., (active_time(e)-current['arrival_start_active_ms'])/1000)
        current['complete'] = reason == 'different_observed_destination' and not current['left_censored']
        current['short_status'] = 'duration_observed' if current['complete'] else 'censored_not_short_by_default'
        visits.append(current)
        current = None

    def close_app(e, reason):
        nonlocal current_app
        if current_app is None:
            return
        current_app['end_ms'] = e['timestampMs']
        current_app['end_active_ms'] = active_time(e)
        current_app['end_reason'] = reason
        current_app['observed_seconds'] = max(0., (active_time(e)-current_app['start_active_ms'])/1000)
        current_app['complete'] = reason == 'different_foreground_application' and not current_app['left_censored']
        apps.append(current_app)
        current_app = None

    for i,e in enumerate(events):
        d=e.get('data', {})
        if e['sessionID'] != session:
            if i:
                # Never count off-session wall time as foreground dwell.
                close_visit(events[i-1], 'session_boundary')
                close_app(events[i-1], 'session_boundary')
                if idle_start is not None:
                    idle_intervals[session].append((idle_start,active_time(events[i-1])))
            session=e['sessionID'];catalog={};inventories={};segment+=1;idle_start=None;previous_target=None
        if e['kind']=='target.update':
            t=d['target']
            if core.admitted(t) and not core.synthetic(t):catalog[t['id']]=t
            else:catalog.pop(t['id'],None)
        if session in excluded_sessions:
            audit['excluded_validation_events']+=1
            continue
        if e['kind']=='inventory.change':
            inventories[e['source']] = (e['id'],d.get('entrances',[]))
        if e.get('continuityDisposition') == 'suspended':
            audit['known_inactivity_events'] += 1
            continue
        if e['kind']=='idle.change':
            if d.get('idle') and idle_start is None:idle_start=active_time(e)
            if not d.get('idle') and idle_start is not None:
                idle_intervals[session].append((idle_start,active_time(e)));idle_start=None
        if is_boundary(e) or e.get('continuityDisposition') == 'boundary':
            close_visit(e, 'capture_gap' if e['kind']=='capture.gap' else 'session_end')
            close_app(e, 'capture_gap' if e['kind']=='capture.gap' else 'session_end')
            if idle_start is not None:
                idle_intervals[session].append((idle_start,active_time(e)));idle_start=None
            segment+=1;previous_target=None;inventories={}
            continue
        if e['kind']=='source.status' and d.get('observationStatus') in {'unavailable','stale','initializing'}:
            inventories.pop(e['source'],None)
            if current:
                # Failure of the current observation family censors the interval; unrelated
                # background failures do not pretend that the foreground target was left.
                f=core.family(e['source'])
                cf=current['source_family']
                if f!='other' and (e['source']==current['source'] or f==cf):
                    close_visit(e,'foreground_source_unavailable')
                    previous_target=None
            continue
        if e['kind']!='context.change':continue
        t=catalog.get(d.get('targetID'))
        if not t:
            close_visit(e,'unadmitted_or_synthetic_target')
            close_app(e,'unadmitted_or_synthetic_target')
            previous_target=None;segment+=1;audit['excluded_contexts']+=1
            continue
        bundle=e.get('app',{}).get('bundleID') or t.get('bundleID')
        t=compact_target(t,bundle)
        matches=[{**entry,'inventory_event_id':ref,'inventory_source':src}
            for src,(ref,entries) in inventories.items() for entry in entries
            if d.get('entranceID') and entry['id']==d['entranceID'] and entry.get('targetID')==t['id']]
        entrance=matches[0] if len(matches)==1 else None
        resolution=d.get('reason')=='resolution_update'
        audit['resolution_events' if resolution else 'independent_context_events']+=1
        if current_app and current_app['bundleID']!=bundle:
            close_app(e,'different_foreground_application')
        if current_app is None and bundle:
            current_app={'session':session,'segment':segment,'bundleID':bundle,
                'name':e.get('app',{}).get('name') or bundle,'start_ms':e['timestampMs'],
                'start_monotonic_ms':e['monotonicMs'],'start_active_ms':active_time(e),'left_censored':resolution,'start_event_id':e['id']}
        linked_resolution = resolution and current is not None and (
            d.get('resolvesContextEventID') in current['linked_ids'] or
            (not d.get('resolvesContextEventID') and bundle==current['target'].get('bundleID')))
        if linked_resolution:
            current['linked_ids'].append(e['id'])
            # Refinement belongs to the same arrival. Its specific locator is only known now.
            if current['target']['id']!=t['id']:
                current['refinements'].append({'event_id':e['id'],'timestamp_ms':e['timestampMs'],
                    'monotonic_ms':e['monotonicMs'],'active_ms':active_time(e),'target':t,
                    'previous_kind':current['target']['kind'],'kind':t['kind']})
                current['target']=t
                current['target_start_ms']=e['timestampMs']
                current['target_start_monotonic_ms']=e['monotonicMs']
                current['target_start_active_ms']=active_time(e)
                current['source']=e['source'];current['source_family']=core.family(e['source'])
            current['entrance']=entrance
            continue
        if resolution and current:
            close_visit(e,'unlinked_resolution')
            previous_target=None
        same_entrance = current is not None and (current['entrance'] or {}).get('id')==(entrance or {}).get('id')
        if current and same_entrance and identity(current['target'],session)==identity(t,session):
            current['repeat_observations']+=1
            current['linked_ids'].append(e['id'])
            continue
        if current:
            prev=current['target']
            reason='different_observed_destination'
            if prev['kind'] in {'page','directory','file'} and t['kind'] in {'app','window'} and bundle==prev.get('bundleID'):
                # A coarse observation does not prove the specific resource changed.
                reason='same_app_granularity_loss'
            close_visit(e,reason,t,entrance)
            previous_target=prev if reason=='different_observed_destination' else None
        cov=sample_by_anchor.get(e['id'],{}).get('inputCoverage',{}).get('status','unknown')
        current={'session':session,'segment':segment,'arrival_event_id':e['id'],
            'arrival_start_ms':e['timestampMs'],'arrival_start_monotonic_ms':e['monotonicMs'],
            'arrival_start_active_ms':active_time(e),'target_start_active_ms':active_time(e),'linked_ids':[e['id']],
            'target_start_ms':e['timestampMs'],'target_start_monotonic_ms':e['monotonicMs'],
            'initial_target':t,'target':t,'previous_target':previous_target,
            'entrance':entrance,
            'source':e['source'],'source_family':core.family(e['source']),
            'arrival_reason':d.get('reason'),'input_coverage':cov,'left_censored':resolution,
            'refinements':[],'repeat_observations':0}
    if events:
        close_visit(events[-1],'export_cutoff');close_app(events[-1],'export_cutoff')
        if idle_start is not None:idle_intervals[session].append((idle_start,active_time(events[-1])))
    for v in visits:
        v['idle_seconds']=overlap(v['target_start_active_ms'],v['end_active_ms'],idle_intervals[v['session']])/1000
        before,after=v['previous_target'],v['next_target']
        v['returns_to_previous'] = bool(before and after and identity(before,v['session'])==identity(after,v['session']))
        v['next_same_app'] = bool(after and after.get('bundleID')==v['target'].get('bundleID'))
        v['same_app_granularity_transition'] = bool(after and v['next_same_app'] and after['kind']!=v['target']['kind'])
        v['same_origin_navigation'] = bool(after and after.get('locator') and v['target'].get('locator') and
            v['target']['kind']=='page' and after['kind']=='page' and
            urlsplit(after['locator']).netloc==urlsplit(v['target']['locator']).netloc)
        v['transient_candidate_thresholds_seconds'] = [t for t in THRESHOLDS if v['complete'] and v['observed_seconds']<t]
    for a in apps:
        a['idle_seconds']=overlap(a['start_active_ms'],a['end_active_ms'],idle_intervals[a['session']])/1000
    return visits,apps,dict(audit)


def quantile(values,p):
    if not values:return None
    values=sorted(values);x=(len(values)-1)*p;i=int(x);j=math.ceil(x)
    return values[i]+(values[j]-values[i])*(x-i)


def summarize(rows):
    complete=[v for v in rows if v['complete']]
    values=[v['observed_seconds'] for v in complete]
    histogram=[]
    for lo,hi,label in zip(BINS,BINS[1:],BIN_NAMES):
        n=sum(lo<=x<hi for x in values)
        histogram.append({'bin':label,'count':n,'share':n/len(values) if values else None})
    return {'observed_intervals':len(rows),'complete_n':len(values),
        'censored_n':len(rows)-len(values),
        'end_reasons':dict(Counter(v['end_reason'] for v in rows)),
        'mean_seconds':statistics.mean(values) if values else None,
        'median_seconds':quantile(values,.5),'p25_seconds':quantile(values,.25),
        'p75_seconds':quantile(values,.75),'p90_seconds':quantile(values,.9),
        'p95_seconds':quantile(values,.95),'max_seconds':max(values) if values else None,
        'histogram':histogram,
        'thresholds':{str(t):{'n':sum(x<t for x in values),
            'share':sum(x<t for x in values)/len(values) if values else None} for t in THRESHOLDS},
        'complete_with_idle_n':sum(v.get('idle_seconds',0)>0 for v in complete)}


def kaplan_meier(rows):
    """Observed-duration survival with known right censoring; descriptive only.

    Source/gap censoring need not be independent of dwell, so this is sensitivity
    analysis, not an unbiased estimate of unobserved dwell or actual intent.
    """
    rows=[r for r in rows if not r.get('left_censored')]
    times=defaultdict(lambda:[0,0])
    for r in rows:times[r['observed_seconds']][0 if r['complete'] else 1]+=1
    n=len(rows);survival=1.;curve=[];median=None
    for t,(deaths,censored) in sorted(times.items()):
        survival*=1-deaths/n
        if median is None and survival<=.5:median=t
        curve.append({'seconds':t,'survival':survival,'at_risk':n,'observed_departures':deaths,'censored':censored})
        n-=deaths+censored
    return {'n':len(rows),'median_seconds':median,'curve':curve}


def threshold_diagnostics(rows):
    output={}
    for threshold in THRESHOLDS:
        short=[v for v in rows if v['complete'] and v['observed_seconds']<threshold]
        output[str(threshold)]={'n':len(short),'kind_counts':dict(Counter(v['target']['kind'] for v in short)),
            'navigation_classes':dict(Counter(v['navigation_class'] for v in short)),
            'return_to_previous_n':sum(v['returns_to_previous'] for v in short),
            'same_app_granularity_transition_n':sum(v['same_app_granularity_transition'] for v in short),
            'same_origin_navigation_n':sum(v['same_origin_navigation'] for v in short),
            'arrival_reasons':dict(Counter(v['arrival_reason'] for v in short))}
    return output


def write_plot(output,bykind):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    page=bykind.get('page',{})
    if not page or not page['complete_n']:return
    fig,axs=plt.subplots(1,2,figsize=(12,4.6),layout='constrained')
    hs=page['histogram'];xs=list(range(len(hs)))
    axs[0].bar(xs,[100*x['share'] for x in hs],color='#3876ad')
    axs[0].set_xticks(xs,[x['bin'] for x in hs],rotation=50,ha='right')
    axs[0].set_ylabel('Share of completed page intervals (%)')
    axs[0].set_title(f"Observed page dwell (n={page['complete_n']})")
    for i,x in enumerate(hs):
        axs[0].text(i,100*x['share']+.6,str(x['count']),ha='center',fontsize=9)
    axs[0].set_ylim(0,max(100*x['share'] for x in hs)*1.2+2)
    kinds=[k for k in ['page','window','app','directory','file'] if bykind[k]['complete_n']]
    medians=[bykind[k]['median_seconds'] for k in kinds]
    axs[1].barh(kinds,medians,color='#3876ad')
    axs[1].set_xlabel('Median observed dwell (seconds)')
    axs[1].set_title('By most specific observed destination')
    for i,(k,value) in enumerate(zip(kinds,medians)):
        axs[1].text(value+.2,i,f"{value:.1f}s · n={bykind[k]['complete_n']}",va='center',fontsize=9)
    axs[1].set_xlim(0,max(medians)*1.6+1)
    for ax in axs:
        ax.spines[['top','right']].set_visible(False)
    fig.suptitle('SmartApp historical observations · known validation sessions excluded',fontsize=12)
    fig.savefig(output/'dwell-distribution.png',dpi=170)
    fig.savefig(output/'dwell-distribution.svg')
    plt.close(fig)


def nfmt(x):return '—' if x is None else f'{x:.2f}'


def report(output,m,visits):
    page=m['by_kind']['page'];thresholds=m['page_threshold_diagnostics']
    lines=['# 前台观测停留时长：描述性分析与临时目标候选', '',
        '## 统计口径', '',
        '- 使用完整原始事件流，未筛选“30 秒内必须有下一步”的预测样本。',
        '- 从目标被前台观察到开始，到另一个目标被观察到结束；这是观测时长，不是精确操作时间或有效阅读时长。',
        '- 同次到达的 resolution_update 不增加访问；细化为网页/目录后，从具体目标可确认时计该目标的时长。相同目标重复通知不重置计时。',
        '- 会话结束、采集缺口、前台来源失败、导出尾部或退化为同应用的粗粒度观察属于删失，不能当作已经离开或短暂停留。',
        '- 系统 60 秒 idle 标志单独记录；可能在阅读，也可能离开电脑，因此不直接扣掉或判错。',
        '- complete 表示这段区间有观察到的首尾，和数据源覆盖 complete 不同；旧覆盖未知数据单列。',
        '- 下列短停留仅标记为疑似临时；不代表误选，也不向模型提供负反馈。原始数据和既有预测标签未改写。', '',
        '## 各目标类型', '',
        '| 最具体目标类型 | 有首尾的区间 | 删失区间 | 中位数（秒） | P25 | P75 | P90 |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for kind,r in m['by_kind'].items():
        lines.append(f"| {kind} | {r['complete_n']} | {r['censored_n']} | {nfmt(r['median_seconds'])} | {nfmt(r['p25_seconds'])} | {nfmt(r['p75_seconds'])} | {nfmt(r['p90_seconds'])} |")
    lines += ['', 'app 行仅指尚未解析出更具体目标的应用级区间；下面另列跨同应用内部导航合并的应用前台区间。不同层级不能相加为新的独立到达。', '',
        f"应用前台区间中位数：{nfmt(m['foreground_application']['median_seconds'])} 秒；有首尾 {m['foreground_application']['complete_n']} 个。", '',
        '## 网页停留分布', '', '| 区间 | 次数 | 占有首尾网页区间 |', '|---|---:|---:|']
    for h in page['histogram']:
        lines.append(f"| {h['bin']} | {h['count']} | {core.percent(h['share'])} |")
    lines += ['', '## 短停留阈值敏感性', '', '| 小于阈值 | 网页次数 | 占比 | 返回前一目标 | 同站点后续导航 |', '|---|---:|---:|---:|---:|']
    for t in THRESHOLDS:
        a=page['thresholds'][str(t)];d=thresholds[str(t)]
        lines.append(f"| {t:g} 秒 | {a['n']} | {core.percent(a['share'])} | {d['return_to_previous_n']} | {d['same_origin_navigation_n']} |")
    lines += ['', '返回前一目标与同站点导航可重叠；一次完成的短操作、网页自动导航与误选不能仅靠上述信息区分。', '',
        '## 观测质量与长停留', '',
        f"- 网页有首尾区间 {page['complete_n']}；删失 {page['censored_n']}。删失区间不参与上表短停留比例。",
        f"- 网页起点覆盖 complete 子集：{m['pages_start_coverage_complete']['complete_n']} 个，时长中位数 {nfmt(m['pages_start_coverage_complete']['median_seconds'])} 秒。这不证明整个停留区间的所有后台来源完整。",
        f"- 网页旧覆盖 unknown 子集：{m['pages_start_coverage_unknown']['complete_n']} 个，中位数 {nfmt(m['pages_start_coverage_unknown']['median_seconds'])} 秒。",
        f"- 有首尾网页区间中，出现 idle 标志 {page['complete_with_idle_n']} 个；无 idle 标志子集中位数 {nfmt(m['pages_without_idle']['median_seconds'])} 秒。这是不同样本子集，不能解释为扣除空闲后的时长。",
        f"- 仅完整区间均值 {nfmt(page['mean_seconds'])} 秒，最长 {nfmt(page['max_seconds'])} 秒，长尾会拉高均值。",
        f"- Kaplan–Meier 右删失敏感性中位数 {nfmt(m['page_survival']['median_seconds'])} 秒；来源中断可能与行为相关，不满足独立删失保证，仅作描述性对照。", '',
        '## 同标签页连续导航与目标切换', '',
        '仅根据截至事件时已观察的库存入口判断，不能由同站点推断同标签页，也不能由同窗口推断未切换标签页。相同窗口和标签页中 URL 变化只证明连续导航，无法区分主动导航与自动跳转。', '',
        '| 网页后续行为 | 有首尾区间数 |', '|---|---:|']
    lines += [f'| {k} | {v} |' for k,v in m['page_navigation_classes'].items()]
    lines += ['', f"排除同标签页连续导航后：{m['pages_excluding_same_tab_navigation']['complete_n']} 个区间，中位数 {nfmt(m['pages_excluding_same_tab_navigation']['median_seconds'])} 秒。", '',
        '## 清洗建议的适用范围', '',
        '- 先比较 0.5、1、2、3、5 秒这几种标记的规模；不能从单一分位数直接推出“误选阈值”。',
        '- 短留并返回原目标、同应用的层级变化、同站点连续导航应分别分析；后两者尤其可能是目标解析或正常导航。',
        '- 临时目标的判断依赖后续离开事件，只能在该事件已经观察到后用于派生标签，不能泄漏到先前输入。',
        '- A→B（短留）→A 应保留“返回当前目标”的单独类别；不能为了生成推荐标签继续强找另一个目标。',
        '- 后续训练实验应同时保留未清洗结果；改变标签口径后的命中率不能直接当作模型能力提升。', '',
        '## 私有具体目标统计', '',
        '以下目标只按有观察首尾的区间计数；不把后台库存列作访问。完整统计与每条区间可在本 run 的 CSV/JSONL 中追溯。', '']
    for kind in ['app','window','page','directory','file']:
        groups=defaultdict(list)
        for v in visits:
            if v['target']['kind']==kind and v['complete']:groups[identity(v['target'],v['session'])].append(v)
        lines += [f'### {kind}', '', '| 目标 | 区间数 | 中位秒数 | <1 秒 | <2 秒 |', '|---|---:|---:|---:|---:|']
        for _,rows in sorted(groups.items(),key=lambda kv:-len(kv[1]))[:10]:
            durations=[r['observed_seconds'] for r in rows]
            lines.append(f"| {core.describe_target(rows[-1]['target'])} | {len(rows)} | {statistics.median(durations):.2f} | {sum(x<1 for x in durations)} | {sum(x<2 for x in durations)} |")
        if not groups:lines.append('| 无数据 | 0 | — | — | — |')
        lines.append('')
    (output/'report-private.md').write_text('\n'.join(lines))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--snapshot',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--plot',action='store_true')
    args=parser.parse_args();output=args.output.expanduser().resolve()
    if not output.is_relative_to(core.PRIVATE_ROOT.resolve()) or output==core.PRIVATE_ROOT.resolve():
        parser.error('Use a new private SmartApp/training run directory')
    os.umask(0o077);output.mkdir(mode=0o700,parents=True,exist_ok=False)
    hashes,_=core.verified_snapshot(args.snapshot)
    events=core.load_lines(args.snapshot/'events.jsonl');samples=core.load_lines(args.snapshot/'navigation-samples.jsonl')
    visits,apps,audit=extract_visits(events,samples,core.KNOWN_TEST_SESSIONS)
    pages=[v for v in visits if v['target']['kind']=='page']
    metrics={'version':VERSION,'range_utc':[core.utc(events[0]['timestampMs']),core.utc(events[-1]['timestampMs'])],
        'audit':audit,'by_kind':{k:summarize([v for v in visits if v['target']['kind']==k]) for k in sorted(core.KINDS)},
        'foreground_application':summarize(apps),
        'pages_start_coverage_complete':summarize([v for v in pages if v['input_coverage']=='complete']),
        'pages_start_coverage_unknown':summarize([v for v in pages if v['input_coverage']=='unknown']),
        'pages_without_idle':summarize([v for v in pages if not v['idle_seconds']]),
        'page_threshold_diagnostics':threshold_diagnostics(pages),
        'page_navigation_classes':dict(Counter(v['navigation_class'] for v in pages if v['complete'])),
        'pages_excluding_same_tab_navigation':summarize([v for v in pages if v['navigation_class']!='same_tab_navigation']),
        'all_threshold_diagnostics':threshold_diagnostics(visits),
        'page_survival':kaplan_meier(pages)}
    core.write_json(output/'metrics.json',metrics)
    with (output/'visits-private.jsonl').open('w') as f:
        for v in visits:f.write(json.dumps(v,ensure_ascii=False)+'\n')
    with (output/'foreground-applications-private.jsonl').open('w') as f:
        for a in apps:f.write(json.dumps(a,ensure_ascii=False)+'\n')
    with (output/'transient-candidates-private.csv').open('w',newline='') as f:
        writer=csv.writer(f);writer.writerow(['arrival_event_id','kind','target','start_utc','observed_seconds',
            'complete','end_reason','input_coverage','idle_seconds','return_to_previous','under_0.5s','under_1s','under_2s','under_3s','under_5s'])
        for v in visits:
            writer.writerow([v['arrival_event_id'],v['target']['kind'],core.describe_target(v['target']),
                core.utc(v['target_start_ms']),v['observed_seconds'],v['complete'],v['end_reason'],
                v['input_coverage'],v['idle_seconds'],v['returns_to_previous'],
                *[v['complete'] and v['observed_seconds']<t for t in THRESHOLDS]])
    report(output,metrics,visits)
    if args.plot:write_plot(output,metrics['by_kind'])
    (output/'analysis-source.py').write_text(Path(__file__).read_text())
    (output/'known_inactivity.py').write_text(Path(__file__).with_name('known_inactivity.py').read_text())
    (output/'core-source.py').write_text(Path(core.__file__).read_text())
    core.write_json(output/'run.json',{'version':VERSION,'snapshot':str(args.snapshot.resolve()),
        'input_hashes':hashes,'code_sha256':core.sha(Path(__file__)),'core_sha256':core.sha(Path(core.__file__)),
        'excluded_sessions':sorted(core.KNOWN_TEST_SESSIONS),'threshold_candidates':THRESHOLDS,
        'raw_data_modified':False,'training_labels_modified':False})
    core.write_json(output/'artifact-hashes.json',{p.name:core.sha(p) for p in output.iterdir() if p.is_file()})
    print(json.dumps({'output':str(output),'by_kind':metrics['by_kind'],
        'page_threshold_diagnostics':metrics['page_threshold_diagnostics'],
        'page_survival_median':metrics['page_survival']['median_seconds']},ensure_ascii=False))


if __name__=='__main__':main()
