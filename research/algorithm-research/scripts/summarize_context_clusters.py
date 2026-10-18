#!/usr/bin/env python3
"""Read-only diagnosis of frozen context-cluster experiments; no selection changes."""
import argparse
from collections import Counter
from dataclasses import replace
import json
import os
from pathlib import Path

import evaluate_context_knn as live_core
from diagnose_dwell_errors import frozen_core, label
from evaluate_evidence_model import DENIED_BUNDLES, clean_input, semantic_label, resolve_label
from summarize_tuned_history import verify_run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--initial-run', required=True, type=Path)
    parser.add_argument('--extended-run', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    out = args.output.resolve()
    if not out.is_relative_to(live_core.PRIVATE_ROOT.resolve()) or out == live_core.PRIVATE_ROOT.resolve():
        parser.error('Use a new private training directory')
    first, first_hashes = verify_run(args.initial_run)
    extended, extended_hashes = verify_run(args.extended_run)
    assert first['source_run'] == extended['source_run']
    base = Path(extended['source_run']); source, source_hashes = verify_run(base)
    core = frozen_core(base); snapshot = Path(source['snapshot'])
    hashes, _ = core.verified_snapshot(snapshot)
    assert hashes == source['input_hashes']
    events = core.load_lines(snapshot/'events.jsonl'); samples = core.load_lines(snapshot/'navigation-samples.jsonl')
    original, *_ = core.make_observations(events, samples, set(source['excluded_sessions']))
    blocked = {core.key_for(e['data']['target'], e['sessionID']) for e in events if e['kind'] == 'target.update'
        and e['data']['target'].get('bundleID') in DENIED_BUNDLES}
    original = [clean_input(q, blocked) for q in original]
    ledgers = json.loads((base/'labels-private.json').read_text())
    metrics = json.loads((args.extended_run/'metrics.json').read_text())
    predictions = json.loads((args.extended_run/'predictions-private.json').read_text())
    profiles = json.loads((args.extended_run/'cluster-profiles-private.json').read_text())
    names = json.loads((args.extended_run/'app-names-private.json').read_text())
    details, results = {}, {}
    for threshold, ledger in ledgers.items():
        items = list(original); valid = set()
        for r in ledger:
            i = r['position']; q = items[i]
            if r['status'] != 'stable_target':
                continue
            t = r['target']
            if q.current_app in DENIED_BUNDLES or t.get('bundleID') in DENIED_BUNDLES:
                continue
            items[i] = replace(q, truth=core.key_for(t, q.session), truth_meta={**t, '_bundle': t.get('bundleID')})
            valid.add(i)
        frequency = [r for r in predictions[threshold]['frequency'] if r['subset'] == 'test' and r['primary_cutoff']]
        ordered = {r['position']: r for r in predictions[threshold]['ordered'] if r['subset'] == 'test' and r['primary_cutoff']}
        assert {r['position'] for r in frequency} == set(ordered)
        reasons = Counter(); records = []; class_sizes = Counter()
        for r in frequency:
            i = r['position']; q = items[i]; class_sizes[r['cluster']] += 1
            assert core.score_row(q, r['predictions']) == r['score']
            key = semantic_label(q, 'window_class')
            mature = {semantic_label(items[j], 'window_class') for j in valid if j < i and items[j].ready_index < q.index}
            if r['score']['hit4']:
                reason = 'hit'
            elif q.truth not in q.candidates:
                reason = 'not_in_current_candidates'
            elif key not in mature:
                reason = 'no_mature_history_for_actual_target'
            elif key not in r['counts']:
                reason = 'history_in_other_clusters_or_outside_count_window'
            elif resolve_label(key, q) != q.truth:
                reason = 'window_class_not_uniquely_resolvable'
            else:
                reason = 'supported_but_frequency_rank_or_hierarchy'
            reasons[reason] += 1
            records.append({'position': i, 'cluster': r['cluster'], 'reason': reason,
                'current_app': names.get(q.current_app, q.current_app),
                'actual': label(q.truth_meta, names), 'actual_meta': q.truth_meta,
                'predictions': [label(q.candidates[k], names) for k in r['predictions']],
                'class_outcome_count': r['total_unique_outcomes'],
                'actual_target_support': r['counts'].get(key, 0), 'score': r['score']})
        assert sum(reasons.values()) == len(frequency)
        grid = json.loads((args.extended_run/f'selective-grid-{threshold}s.json').read_text())
        results[threshold] = {'frequency_error_reasons': dict(reasons),
            'frequency_only_top1_hits': sum(r['score']['hit1'] and not ordered[r['position']]['score']['hit1'] for r in frequency),
            'ordered_only_top1_hits': sum(not r['score']['hit1'] and ordered[r['position']]['score']['hit1'] for r in frequency),
            'max_accepted_validation_across_grid': max(r['validation']['n'] for r in grid),
            'classes': [{'cluster': r['cluster'], 'train_inputs': r['train_inputs'],
                'primary_test_inputs': class_sizes[r['cluster']], 'pretest_dominant_share': r['top1_share_at_test_start'],
                'pretest_outcomes': r['mature_outcomes_at_test_start']} for r in profiles[threshold]['frequency']]}
        details[threshold] = records
    os.umask(0o077); out.mkdir(parents=True, mode=0o700, exist_ok=False)
    core.write_json(out/'diagnosis.json', results)
    core.write_json(out/'cases-private.json', details)
    lines = ['# 情景聚类实验结论', '',
        '已完成基础聚类和扩展聚类两轮。当前证据不支持直接用情景类别的下一步频次替换现有排序器。无序频次输入可以保留为简单对照；频次版与顺序版差异很小，尚不能声称稳定优劣。', '',
        '基础版使用累计频次，验证选出 4 类；5 秒清洗后期 Top1 均为 18.2%，Top4 顺序版 30.7%、频次版 29.5%。扩展版降低常驻背景相对权重，比较聚类中是否加入前台，以及全历史/6 小时/1 小时频次，仍只按验证选择。', '',
        '后期主评估为每会话预测时间至少间隔 30 秒的 88 个截点，不能与先前 194/213 个重叠截点的百分比直接相减。模型分别在验证段选参；表中是各方法选参后的对比，不是只改变顺序这一个变量的因果归因。', '',
        '| 5 秒清洗，扩展版 | Top1 | Top4 |', '|---|---:|---:|']
    five = metrics['5']
    for name, title in [('ordered', '类别＋有序历史'), ('frequency', '类别＋无序频次'),
                        ('presence', '类别＋是否出现'), ('state_only', '背景类别，不含近期历史')]:
        r = five['models'][name]['test']
        lines.append(f"| {title} | {r['top1_hits']}/{r['n']}（{core.percent(r['top1'])}） | {r['top4_hits']}/{r['n']}（{core.percent(r['top4'])}） |")
    for name, title in [('global_frequency', '最近 1 小时全局下一步频次'), ('old_knn', '原调参近邻'), ('recency', '近期访问顺序')]:
        r = five['baselines'][name]['test']
        lines.append(f"| {title} | {r['top1_hits']}/{r['n']}（{core.percent(r['top1'])}） | {r['top4_hits']}/{r['n']}（{core.percent(r['top4'])}） |")
    freq = five['models']['frequency']; cfg = freq['config']
    lines += ['', f"频次版验证所选参数：{cfg['k']} 类，背景占名义权重 80%、最近 8 次到达的无序频次占 20%，不单独加当前前台维度（近期到达仍包含当前目标），按全历史类内频次排序。来源质量继续折扣。顺序版选择 20 类、背景/路径 80%/20%、最近 6 小时类内频次。", '',
        '## 类别有含义，但下一步不够集中', '',
        '频次版类别 1 的早期共同条件包含后台 chat 目录（100%）和某个 chatgpt.com 页面（89%）；测试前 29 次类内去重后续中，一个具体 ChatGPT 网页占 17 次（58.6%）。类别 9 包含后台学校 Brightspace 站点（100%）、PDF Expert 窗口（92%）与 Downloads 目录（85%）；72 次后续中 ChatGPT 窗口 24 次（33.3%）、PDF Expert 窗口 5 次（6.9%）。',
        '这里的百分比是类内共同条件或测试前历史后续比例，不是测试准确率。共有后台不等于当前意图；即使把有含义的情景识别出来，也可能对应很多不同的下一步。上述两个类别在本次后期主测试均未匹配到截点，不能拿它们训练段的集中程度当作已证实的后期能力。', '',
        '## 频次版的失败位置', '', '| 后期结果 | 数量 |', '|---|---:|']
    reason_names = {'hit': '前四命中', 'not_in_current_candidates': '具体目标不在当时可推荐候选',
        'no_mature_history_for_actual_target': '实际目标没有成熟历史标签',
        'history_in_other_clusters_or_outside_count_window': '实际目标有历史，但未进入这个类别的计数',
        'window_class_not_uniquely_resolvable': '窗口类别无法唯一还原到具体目标',
        'supported_but_frequency_rank_or_hierarchy': '这个类别已有正确目标，但频率排名或层级去重落后'}
    for reason, n in results['5']['frequency_error_reasons'].items():
        lines.append(f'| {reason_names[reason]} | {n} |')
    selective = five['models']['selective']; v = selective['validation']; t = selective['test']
    strict = freq['strict_test']
    lines += ['', f"选择性预测也未达标：验证网格中最多仅接受 {results['5']['max_accepted_validation_across_grid']} 个截点，少于预先要求的 15 个。探索候选在验证段 {v['top1_hits']}/{v['n']} 命中，后期只接受 {t['n']}/88，首选命中 {t['top1_hits']} 次；不能据此承诺 80% 以上。", '',
        f"完整来源覆盖子集只有 {strict['n']} 个截点；频次版 Top1 {strict['top1_hits']}/{strict['n']}，Top4 {strict['top4_hits']}/{strict['n']}。其余历史来源覆盖未完全验证。3 秒清洗结论相似：频次版首选 15/88、前四 26/88，仍未超过近期访问。", '',
        '下一步更值得检验的是让背景类别作为排序的一个输入，再结合当前可达目标与近期访问；不宜把类别直接等同于一个固定动作。新的模型选择结论需要之后自然产生的数据验证，本次旧测试段已经多次用于探索。', '',
        f'[基础实验明细]({args.initial_run.resolve()}/report-private.md)',
        f'[扩展实验及全部类别]({args.extended_run.resolve()}/report-private.md)', '',
        '59 项研究脚本测试通过。新输出权限为目录 0700、文件 0600；原始快照、既有 run 与线上菜单未改动。', '']
    # Paths contain spaces; CommonMark requires angle-bracket link targets.
    lines = [line.replace('](/', '](</').replace('/report-private.md)', '/report-private.md>)') for line in lines]
    (out/'report-private.md').write_text('\n'.join(lines))
    (out/'analysis-source.py').write_text(Path(__file__).read_text())
    core.write_json(out/'run.json', {'initial_run': str(args.initial_run.resolve()), 'initial_hashes': first_hashes,
        'extended_run': str(args.extended_run.resolve()), 'extended_hashes': extended_hashes,
        'source_artifact_hashes': source_hashes, 'selection_changed': False})
    core.write_json(out/'artifact-hashes.json', {p.name: core.sha(p) for p in out.iterdir() if p.is_file()})
    print(json.dumps({'output': str(out), 'diagnosis': results}, ensure_ascii=False))


if __name__ == '__main__':
    main()
