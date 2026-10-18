#!/usr/bin/env python3
"""Ordered destination contexts, with only the length-one model enabled.

Standard library only. Labels mature causally, one actual outcome per context.
The current experiment uses no preceding visits or background similarity.
"""
import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass, replace
import json
import os
from pathlib import Path

import evaluate_context_knn as live_core
from diagnose_dwell_errors import frozen_core, label
from evaluate_evidence_model import DENIED_BUNDLES, clean_input, independent_positions, semantic_label, resolve_label
from summarize_tuned_history import verify_run

VERSION = 'ordered-current-transition-v1'


@dataclass(frozen=True)
class OrderedContext:
    """Oldest to newest; [current] is the same type as [previous, current]."""
    targets: tuple

    def __post_init__(self):
        if not isinstance(self.targets, tuple) or not self.targets or not all(isinstance(k, str) and k for k in self.targets):
            raise ValueError('A context requires an ordered, nonempty tuple of observed targets')

    @property
    def length(self):
        return len(self.targets)

    def suffix(self, length):
        if not 1 <= length <= self.length:
            raise ValueError('Cannot fabricate missing earlier context')
        return OrderedContext(self.targets[-length:])


def current_context(q):
    # Window instances are not stable cross-session learning identities. Use only
    # the available semantic class, and never call it a known specific document.
    if q.current.startswith('window@'):
        if not q.current_app:
            return None
        key = 'window_class:' + q.current_app
    else:
        key = q.current
    return OrderedContext((key,))


class TransitionCounts:
    def __init__(self):
        self.evidence = defaultdict(dict)

    def learn(self, context, outcome_id, destination, timestamp, ready_index, asof_index):
        if ready_index >= asof_index:
            raise ValueError('A label is not mature at this prediction cutoff')
        if context is None:
            return False
        rows = self.evidence[context]
        if outcome_id in rows:
            return False
        rows[outcome_id] = {'destination': destination, 'timestamp': timestamp, 'ready_index': ready_index}
        return True

    def distribution(self, context):
        counts, last = Counter(), {}
        for evidence in self.evidence.get(context, {}).values():
            key = evidence['destination']; counts[key] += 1
            last[key] = max(last.get(key, float('-inf')), evidence['timestamp'])
        return counts, last

    def predict(self, core, q, tie_break='latest_conditional_outcome'):
        context = current_context(q)
        counts, last = self.distribution(context)
        mapped, mapped_last = Counter(), {}
        for key, count in counts.items():
            destination = resolve_label(key, q)
            if destination:
                mapped[destination] += count
                mapped_last[destination] = max(mapped_last.get(destination, float('-inf')), last[key])
        if tie_break not in {'latest_conditional_outcome', 'id_control'}:
            raise ValueError('Unknown tie break')
        raw = sorted(mapped, key=lambda k: (-mapped[k],
            -mapped_last[k] if tie_break == 'latest_conditional_outcome' else 0, k))
        shown = core.select_four(raw, q.candidates)
        total = sum(counts.values())
        support = mapped[shown[0]] if shown else 0
        return {'context': list(context.targets) if context else None, 'context_length': 1,
            'predictions': shown, 'raw_ranking': raw, 'counts': dict(counts),
            'last_conditional_outcome': last, 'available_counts': dict(mapped),
            'evidence_n': total, 'top1_support': support,
            'top1_historical_fraction': support/total if total else 0.,
            'unavailable_outcome_count': total - sum(mapped.values()),
            'abstain_reason': None if shown else ('unknown_current' if context is None else
                'no_current_history' if not total else 'no_uniquely_available_outcome')}


def replay(core, items, valid, ledger, outcome_times, positions):
    model = TransitionCounts()
    pending = sorted(valid, key=lambda j: (items[j].ready_index, j))
    cursor = 0; records = {}; controls = {}; audit = {'matured_samples': 0, 'learned_outcomes': 0, 'duplicate_outcomes': 0}
    for i in sorted(set(positions)):
        q = items[i]
        while cursor < len(pending) and items[pending[cursor]].ready_index < q.index:
            j = pending[cursor]; r = items[j]; cursor += 1
            assert j < i
            outcome = ledger[j]['target_event_id']
            learned = model.learn(current_context(r), outcome, semantic_label(r, 'window_class'),
                outcome_times[outcome], r.ready_index, q.index)
            audit['matured_samples'] += 1
            audit['learned_outcomes' if learned else 'duplicate_outcomes'] += 1
        records[i] = model.predict(core, q)
        controls[i] = model.predict(core, q, 'id_control')
    return records, controls, audit


def metrics(core, items, positions, records):
    def score(indices):
        m = core.metric([core.score_row(items[i], records[i]['predictions']) for i in indices])
        return {**m, 'eligible_n': len(positions), 'coverage': len(indices)/len(positions) if positions else 0.}
    return {'all': score(positions),
        'with_current_history': score([i for i in positions if records[i]['evidence_n'] > 0]),
        'at_least_3_current_outcomes': score([i for i in positions if records[i]['evidence_n'] >= 3]),
        'with_predictions': score([i for i in positions if records[i]['predictions']]),
        'high_agreement_exploratory': score([i for i in positions if records[i]['top1_support'] >= 3
            and records[i]['top1_historical_fraction'] >= .7])}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-run', required=True, type=Path)
    parser.add_argument('--tuning-run', required=True, type=Path)
    parser.add_argument('--cluster-run', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args(); out = args.output.resolve(); base = args.source_run.resolve()
    if not out.is_relative_to(live_core.PRIVATE_ROOT.resolve()) or out == live_core.PRIVATE_ROOT.resolve():
        parser.error('Use a new private training directory')
    source, source_hashes = verify_run(base)
    tuning, tuning_hashes = verify_run(args.tuning_run)
    clustering, cluster_hashes = verify_run(args.cluster_run)
    assert Path(tuning['source_run']) == base == Path(clustering['source_run'])
    core = frozen_core(base); snapshot = Path(source['snapshot'])
    hashes, _ = core.verified_snapshot(snapshot); assert hashes == source['input_hashes']
    events = core.load_lines(snapshot/'events.jsonl'); samples = core.load_lines(snapshot/'navigation-samples.jsonl')
    original, audit, arrivals, inventory, display = core.make_observations(events, samples, set(source['excluded_sessions']))
    blocked = {core.key_for(e['data']['target'], e['sessionID']) for e in events if e['kind'] == 'target.update'
        and e['data']['target'].get('bundleID') in DENIED_BUNDLES}
    original = [clean_input(q, blocked) for q in original]
    ledgers = json.loads((base/'labels-private.json').read_text())
    previous = json.loads((args.tuning_run/'predictions-private.json').read_text())
    cluster_metrics = json.loads((args.cluster_run/'metrics.json').read_text())
    cluster_predictions = json.loads((args.cluster_run/'predictions-private.json').read_text())
    names = {e['app']['bundleID']: e['app']['name'] for e in events if e.get('app', {}).get('name')}
    outcome_times = {e['id']: e['timestampMs'] for e in events}
    n = len(original); train_end = int(n*.6); test_start = int(n*.8)
    os.umask(0o077); out.mkdir(mode=0o700, parents=True, exist_ok=False)
    core.write_json(out/'plan.json', {'version': VERSION, 'enabled_context_lengths': [1],
        'context_order': 'oldest to newest, ending at current', 'counts': 'all mature historical outcomes',
        'tie_break': 'latest conditional actual outcome, then stable ID only if still tied',
        'count_deduplication': 'one actual outcome per ordered context',
        'background': 'candidate availability only; not an input to transition scores',
        'preceding_visits': 'not read by this length-one implementation',
        'window_identity': 'stable application window class; unique concrete replay mapping required',
        'train_end_original_position': train_end, 'test_start_original_position': test_start,
        'prediction_spacing_ms': 30000, 'tuning': False, 'test_previously_seen': True})
    results, predictions = {}, {}
    for threshold, ledger in ledgers.items():
        items = list(original); valid = set()
        for row in ledger:
            i = row['position']; q = items[i]
            assert q.sample_id == row['sample_id'] and q.index == row['input_through_event_index']
            if row['status'] != 'stable_target':
                continue
            target = row['target']
            if q.current_app in DENIED_BUNDLES or target.get('bundleID') in DENIED_BUNDLES:
                continue
            items[i] = replace(q, truth=core.key_for(target, q.session), truth_meta={**target, '_bundle': target.get('bundleID')})
            valid.add(i)
        val_full = [i for i in range(train_end, test_start) if i in valid and items[i].ready_index < items[test_start].index]
        test_full = [i for i in range(test_start, n) if i in valid]
        validation = independent_positions(items, val_full); test = independent_positions(items, test_full)
        records, controls, replay_audit = replay(core, items, valid, ledger, outcome_times, val_full + test_full)
        control = core.metric([core.score_row(items[i], controls[i]['predictions']) for i in test])
        # This is the existing same-foreground frequency baseline, now exposed as
        # an ordered length-one model. Only the declared tie-breaking rule changes.
        saved_control = cluster_metrics[threshold]['baselines']['foreground_frequency']
        assert saved_control['max_age_hours'] is None
        assert all(control[k] == saved_control['test'][k] for k in control), 'Current-state baseline parity failed'
        baselines = {'id_tie_control': control}
        for name in ['recency', 'selected']:
            saved = {r['position']: r for r in previous[threshold][name]}
            baselines['old_knn' if name == 'selected' else name] = core.metric([core.score_row(items[i],
                [k for k in saved[i]['predictions'] if k in items[i].candidates]) for i in test])
        cluster_saved = {r['position']: r for r in cluster_predictions[threshold]['frequency']}
        baselines['previous_cluster_frequency'] = core.metric([core.score_row(items[i], cluster_saved[i]['predictions']) for i in test])
        reasons = Counter()
        for i in test:
            row = records[i]; q = items[i]
            if core.score_row(q, row['predictions'])['hit4']:
                reasons['hit'] += 1
            elif q.truth not in q.candidates:
                reasons['not_in_candidates'] += 1
            elif not row['evidence_n']:
                reasons['current_has_no_mature_history'] += 1
            elif semantic_label(q, 'window_class') not in row['counts']:
                reasons['this_next_target_not_seen_from_current'] += 1
            elif q.truth not in row['raw_ranking']:
                reasons['window_class_not_uniquely_resolvable'] += 1
            elif q.truth in row['raw_ranking'][:4]:
                reasons['hierarchy_suppression'] += 1
            else:
                reasons['frequency_or_tie_rank'] += 1
        results[threshold] = {'validation': metrics(core, items, validation, records),
            'test': metrics(core, items, test, records),
            'strict_test': metrics(core, items, [i for i in test if items[i].strict], records),
            'all_test_cutoffs_secondary': metrics(core, items, test_full, records),
            'same_test_baselines': baselines, 'miss_reasons': dict(reasons), 'replay_audit': replay_audit,
            'tie_changes': {'different_top1': sum(records[i]['predictions'][:1] != controls[i]['predictions'][:1] for i in test),
                'different_top4': sum(set(records[i]['predictions']) != set(controls[i]['predictions']) for i in test)}}
        predictions[threshold] = [{'position': i, 'subset': 'validation' if i < test_start else 'test',
            'primary_cutoff': i in validation or i in test, 'sample_id': items[i].sample_id,
            'time_utc': core.utc(items[i].timestamp), 'current_meta': {**display.get(items[i].current, {}), '_bundle': items[i].current_app},
            'truth': items[i].truth, 'truth_meta': items[i].truth_meta,
            'prediction_meta': [items[i].candidates[k] for k in records[i]['predictions']],
            'score': core.score_row(items[i], records[i]['predictions']),
            'id_control_score': core.score_row(items[i], controls[i]['predictions']), **records[i]}
            for i in val_full + test_full]
        print(json.dumps({'threshold': threshold, 'test': results[threshold]['test'], 'reasons': dict(reasons)}), flush=True)
    core.write_json(out/'metrics.json', results); core.write_json(out/'predictions-private.json', predictions)
    core.write_json(out/'destination-rankings-private.json', {
        source_name: {kind: [{'target': k, 'count': count, 'metadata': display.get(k)}
            for k, count in counts.most_common() if k not in blocked] for kind, counts in collection.items()}
        for source_name, collection in [('independent_arrivals', arrivals), ('inventory_observations', inventory)]})
    render_report(out, results, core)
    for name in [Path(__file__).name, 'evaluate_context_knn.py', 'evaluate_evidence_model.py',
                 'diagnose_dwell_errors.py', 'summarize_tuned_history.py']:
        (out/('source-' + name)).write_text(Path(__file__).with_name(name).read_text())
    core.write_json(out/'run.json', {'version': VERSION, 'source_run': str(base), 'source_artifact_hashes': source_hashes,
        'tuning_run': str(args.tuning_run.resolve()), 'tuning_hashes': tuning_hashes,
        'cluster_run': str(args.cluster_run.resolve()), 'cluster_hashes': cluster_hashes,
        'input_hashes': hashes, 'raw_data_modified': False, 'live_model_updated': False,
        'test_previously_seen': True, 'audit': audit})
    core.write_json(out/'artifact-hashes.json', {p.name: core.sha(p) for p in out.iterdir() if p.is_file()})
    print(json.dumps({'output': str(out), 'complete': True}), flush=True)


def render_report(out, results, core):
    lines = ['# 有序界面序列：仅启用当前界面', '',
        '统一表示为以当前界面结尾的有序目标序列。本次只启用长度 1：[当前界面] → 下一步稳定目标。长度 2 的 [前一个界面, 当前界面] 与更长序列属于同一数据类型，目前不训练、不选参。',
        '预测分数只来自相同当前目标的已成熟后续频次。不读取更早访问序列，不根据后台相似度聚类；后台观测只影响实际入口候选。每个起点下同一次实际后续到达只计一次，标签满原 30 秒观察窗后才进入计数。',
        '网页和目录以具体已知目标区分；无法确认稳定具体定位符的窗口使用已知应用窗口类别，推荐时必须能唯一对应当前具体候选。应用层观测仍是应用层，不能补称具体网页。',
        '首要排序依据为条件频次；同次数优先取这个起点下最近一次观察到的后续目标，仍同时时才按 ID 固定排序。没有历史或没有可唯一对应的候选时返回空结果，不用全局近期访问填满。',
        '无参数搜索。沿用前 60% 学习、中间 20% 验证、最后 20% 后期测试；主评估同会话截点至少间隔 30 秒。历史计数在线更新，但只读预测时已经成熟的标签。原测试段此前已多次查看，本次仍属探索性回放。', '',
        '| 清洗阈值 | 后期截点 | Top1 | Top4 | 可给出推荐的截点 | 推荐覆盖率 |', '|---|---:|---:|---:|---:|---:|']
    for threshold, result in results.items():
        m = result['test']['all']; shown = result['test']['with_predictions']
        lines.append(f"| {threshold} 秒 | {m['n']} | {m['top1_hits']}/{m['n']}（{core.percent(m['top1'])}） | {m['top4_hits']}/{m['n']}（{core.percent(m['top4'])}） | {shown['n']} | {core.percent(shown['coverage'])} |")
    lines += ['', '## 起点历史证据分组', '',
        '分组只使用当前起点在预测前已有的学习证据，不根据本次真实下一步筛选。历史高一致率是探索性统计，不是校准成功概率。', '',
        '| 阈值 | 分组 | 截点数 | 占全部测试 | Top1 | Top4 |', '|---|---|---:|---:|---:|---:|']
    for threshold, result in results.items():
        for group, title in [('with_current_history', '起点至少有一次成熟后续'), ('at_least_3_current_outcomes', '起点至少有三次成熟后续'),
                             ('with_predictions', '实际可给出推荐'), ('high_agreement_exploratory', '首选至少三次支持且历史占比 ≥70%')]:
            m = result['test'][group]
            lines.append(f"| {threshold} 秒 | {title} | {m['n']} | {core.percent(m['coverage'])} | {core.percent(m['top1'])} | {core.percent(m['top4'])} |")
    lines += ['', '## 同一批后期截点对照', '',
        'ID 平局对照已逐指标复现此前“相同前台下一步频次”结果。本次将其整理为独立模型，并明确统一的有序序列接口；不能把已有基线重新命名说成新的准确率发现。', '',
        '| 阈值 | 方法 | Top1 | Top4 |', '|---|---|---:|---:|']
    for threshold, result in results.items():
        for name, title in [('id_tie_control', '旧的相同前台频次（ID 平局）'), ('previous_cluster_frequency', '此前背景聚类＋频次'),
                            ('old_knn', '此前调参近邻'), ('recency', '按最后访问时间排序')]:
            m = result['same_test_baselines'][name]
            lines.append(f"| {threshold} 秒 | {title} | {core.percent(m['top1'])} | {core.percent(m['top4'])} |")
    lines += ['', '当前实现是离线可复现原型，未替换线上菜单。逐预测条件、实际目标、候选、频次、成熟时间与原始到达/库存排行保存在同一私有 run。', '']
    (out/'report-private.md').write_text('\n'.join(lines))


if __name__ == '__main__':
    main()
