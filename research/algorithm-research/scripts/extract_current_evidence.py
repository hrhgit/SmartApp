#!/usr/bin/env python3
"""Freeze a graded, outcome-conditioned diagnostic benchmark; do not fit a model.

The prospective gate and the retrospective cohort are explicitly separate.
Complete target locators and supporting event references stay in a private run.
"""
import argparse
from collections import Counter, defaultdict
from dataclasses import replace
import json
import os
from pathlib import Path

import evaluate_context_knn as live_core
import evaluate_ordered_current as model
from diagnose_dwell_errors import frozen_core, label
from evaluate_evidence_model import DENIED_BUNDLES, clean_input, semantic_label, resolve_label
from summarize_tuned_history import verify_run

VERSION = 'ordered-current-evidence-audit-v1'
GRADES = ('strong', 'repeated', 'single', 'none')
TITLES = {'strong': '反复且集中支持', 'repeated': '重复支持但未达到集中门槛',
          'single': '仅一次支持', 'none': '没有这条转移的支持'}
GRADE_DEFINITIONS = {'strong': 'support >= 3 and support / all outcomes from current >= 0.70',
    'repeated': 'support >= 2 and not strong', 'single': 'support == 1', 'none': 'support == 0'}


def make_plan(run):
    return {'version': VERSION, 'source_run': str(run), 'grades': GRADE_DEFINITIONS,
        'grade_order': list(GRADES), 'grading_fixed_before_extraction': True,
        'definition': 'At least one mature unique historical outcome from the same current context to the realized next semantic destination; never select by prediction correctness.',
        'primary_threshold_seconds': 5, 'sensitivity_threshold_seconds': 3,
        'population': 'existing late 20% test, original primary cutoffs at least 30 seconds apart within session',
        'primary_metric': 'exact destination Top4', 'secondary_metric': 'exact destination Top1',
        'objective': 'Approach 100% on the fixed evidence-supported benchmark; report coverage and all errors, without a claim of guarantee.',
        'candidate_omissions': 'remain failures in primary evidence-supported score; covered-only score is separate',
        'window_support': 'application window class only, not proof of the exact prior window instance',
        'predictive_gate': 'grade the displayed first candidate using the same support/share rules; only uses pre-prediction values',
        'evidence_age': 'reported in hours from last supporting outcome, not used to select cases or optimize thresholds',
        'leakage_boundary': 'true-label-conditioned cohort fields and witness selection are for scoring after prediction only',
        'test_previously_seen': True, 'parameter_search': False,
        'historical_labels': 'only ready_index < query.index; deduplicate by (ordered context, actual outcome event ID)'}


def grade(support, total):
    if support < 0 or total < support:
        raise ValueError('Invalid evidence counts')
    if support >= 3 and support / total >= .7:
        return 'strong'
    if support >= 2:
        return 'repeated'
    return 'single' if support else 'none'


def prospective_gate(record):
    """No realized outcome is an input to this gate."""
    return {'grade': grade(record['top1_support'], record['evidence_n']),
            'has_origin_history': record['evidence_n'] > 0,
            'has_prediction': bool(record['predictions']),
            'first_support': record['top1_support'],
            'first_share': record['top1_historical_fraction']}


def diagnostic_evidence(q, record, witnesses):
    key = semantic_label(q, 'window_class')
    support = record['counts'].get(key, 0)
    selected = [w for w in witnesses if w['destination_key'] == key]
    if len(selected) != support:
        raise ValueError('Historical witness count does not match frozen prediction')
    if len({w['outcome_event_id'] for w in witnesses}) != len(witnesses):
        raise ValueError('Duplicate historical outcome')
    if len(witnesses) != record['evidence_n']:
        raise ValueError('Missing conditional history')
    if any(w['label_ready_event_index'] >= q.index or w['input_event_index'] >= q.index
           or w['outcome_event_index'] >= q.index for w in witnesses):
        raise ValueError('Future or unmatured historical evidence')
    total = record['evidence_n']
    age = (q.timestamp - record['last_conditional_outcome'][key]) / 3600000 if support else None
    if age is not None and age < 0:
        raise ValueError('Future supporting outcome time')
    return {'eligible': support > 0, 'grade': grade(support, total),
            'destination_key': key, 'support_count': support, 'origin_outcome_count': total,
            'share': support / total if total else None,
            'competing_destination_count': len(record['counts']) - bool(support),
            'hours_since_last_support': age,
            'candidate_present': q.truth in q.candidates,
            'support_resolves_to_actual_candidate': resolve_label(key, q) == q.truth,
            'identity_level': 'application_window_class' if key.startswith('window_class:')
                else q.truth_meta['kind'],
            'supporting_outcome_ids': [w['outcome_event_id'] for w in selected]}


def summarize(rows, population_n):
    n = len(rows)
    hit1 = sum(r['evaluation']['hit1'] for r in rows)
    hit4 = sum(r['evaluation']['hit4'] for r in rows)
    return {'n': n, 'population_n': population_n, 'coverage': n / population_n if population_n else 0.,
            'top1_hits': hit1, 'top4_hits': hit4, 'top1': hit1 / n if n else None,
            'top4': hit4 / n if n else None, 'top4_misses': n - hit4,
            'candidate_missing': sum(not r['diagnosis_after_outcome']['candidate_present'] for r in rows),
            'window_class_support': sum(r['diagnosis_after_outcome']['eligible'] and r['diagnosis_after_outcome']['identity_level'] == 'application_window_class' for r in rows),
            'strict_source_n': sum(r['evaluation']['strict_input_and_label_coverage'] for r in rows)}


def summarize_groups(rows):
    n = len(rows)
    supported = [r for r in rows if r['diagnosis_after_outcome']['eligible']]
    covered = [r for r in supported if r['diagnosis_after_outcome']['candidate_present']]
    return {'all': summarize(rows, n), 'evidence_supported': summarize(supported, n),
            'evidence_supported_candidate_present': summarize(covered, n),
            'grades': {g: summarize([r for r in rows if r['diagnosis_after_outcome']['grade'] == g], n) for g in GRADES},
            'prospective_first_candidate_grades': {g: summarize([r for r in rows if r['prospective_gate']['grade'] == g], n) for g in GRADES},
            'strict_evidence_supported': summarize([r for r in supported if r['evaluation']['strict_input_and_label_coverage']], n),
            'identity_levels': {k: summarize([r for r in supported if r['diagnosis_after_outcome']['identity_level'] == k], n)
                                for k in sorted({r['diagnosis_after_outcome']['identity_level'] for r in supported})},
            'miss_reasons': dict(Counter(r['evaluation']['miss_reason'] for r in supported if not r['evaluation']['hit4']))}


def miss_reason(q, record, evidence):
    if q.truth in record['predictions']:
        return 'hit'
    if not evidence['candidate_present']:
        return 'not_in_candidates'
    if not evidence['eligible']:
        return 'no_same_current_transition'
    if not evidence['support_resolves_to_actual_candidate']:
        return 'semantic_class_not_resolvable'
    return 'hierarchy_suppression' if q.truth in record['raw_ranking'][:4] else 'frequency_or_tie_rank'


def reconstruct(core, items, valid, ledger, saved, events, names):
    counts = model.TransitionCounts()
    history = defaultdict(dict)
    event_indices = {e['id']: i for i, e in enumerate(events)}
    pending = sorted(valid, key=lambda j: (items[j].ready_index, j))
    cursor = 0
    rows = []
    for frozen in sorted(saved, key=lambda r: r['position']):
        i = frozen['position']; q = items[i]
        if q.sample_id != frozen['sample_id']:
            raise ValueError('Sample identity changed')
        while cursor < len(pending) and items[pending[cursor]].ready_index < q.index:
            j = pending[cursor]; cursor += 1
            r = items[j]; context = model.current_context(r)
            if j >= i:
                raise ValueError('Future training sample')
            outcome_id = ledger[j]['target_event_id']; event_index = event_indices[outcome_id]
            destination = semantic_label(r, 'window_class')
            if counts.learn(context, outcome_id, destination, events[event_index]['timestampMs'], r.ready_index, q.index):
                history[context][outcome_id] = {
                    'source_sample_id': r.sample_id, 'source_position': j, 'session': r.session,
                    'context': list(context.targets), 'input_time_utc': core.utc(r.timestamp),
                    'input_event_index': r.index, 'label_ready_event_index': r.ready_index,
                    'outcome_event_id': outcome_id, 'outcome_event_index': event_index,
                    'outcome_time_utc': core.utc(events[event_index]['timestampMs']),
                    'destination_key': destination, 'destination_meta': r.truth_meta}
        record = counts.predict(core, q)
        if any(record[k] != frozen[k] for k in record) or core.score_row(q, record['predictions']) != frozen['score']:
            raise ValueError('Replay differs from frozen predictions')
        witnesses = list(history.get(model.current_context(q), {}).values())
        evidence = diagnostic_evidence(q, record, witnesses)
        rows.append({'sample_id': q.sample_id, 'position': i, 'split': frozen['subset'],
            'primary_cutoff': frozen['primary_cutoff'],
            'display_only': {'current_description': label(frozen['current_meta'], names),
                'current_identity_level': 'application_window_class' if q.current.startswith('window@') else frozen['current_meta'].get('kind', 'unknown')},
            'pre_prediction': {'time_utc': core.utc(q.timestamp), 'session': q.session,
                'input_event_index': q.index, 'reference_max_event_index': q.reference_max_index,
                'context': record['context'], 'current_key': q.current,
                'source_quality': q.source_quality, 'coverage': q.coverage,
                'candidates': q.candidates,
                'counts': record['counts'], 'historical_outcomes': witnesses,
                'candidate_historical_supports': record['available_counts']},
            'prospective_gate': prospective_gate(record),
            'baseline_prediction': {'keys': record['predictions'], 'raw_ranking': record['raw_ranking'],
                'descriptions': [label(q.candidates[k], names) for k in record['predictions']]},
            'observed_outcome': {'key': q.truth, 'metadata': q.truth_meta,
                'description': label(q.truth_meta, names), 'event_id': ledger[i]['target_event_id']},
            'diagnosis_after_outcome': evidence,
            'evaluation': {**frozen['score'], 'miss_reason': miss_reason(q, record, evidence),
                'strict_input_and_label_coverage': q.strict,
                'actual_raw_rank': record['raw_ranking'].index(q.truth) + 1 if q.truth in record['raw_ranking'] else None}})
    return rows


def write_jsonl(path, rows):
    with path.open('x') as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + '\n')


def fmt_rate(hits, n):
    return f'{hits}/{n}（{hits/n:.1%}）' if n else '—（无样本）'


def render_report(out, metrics, primary_rows):
    total_n = metrics['5']['test']['all']['n']
    supported_n = metrics['5']['test']['evidence_supported']['n']
    covered_n = metrics['5']['test']['evidence_supported_candidate_present']['n']
    ceiling = covered_n / supported_n if supported_n else 0.
    lines = ['# 当前界面模型：有证据测试样本', '',
        '目标是在有历史行为证据的固定样本上逼近全对。本轮抽取样本、核对证据并固定评分口径，不改变预测或根据命中情况筛样。',
        '主口径沿用精确目标 Top4，同时报告 Top1。5 秒清洗为主，3 秒为敏感性对照。', '',
        '证据定义：预测前，同一个当前界面（窗口仅按应用窗口类别）至少发生过一次通向实际下一步语义目标的成熟、去重历史行为。不是仅出现过该目标，也不是后台库存中存在该目标。',
        '强支持：至少 3 次且占该起点全部历史后续的 70% 以上；重复支持：至少 2 次但不满足强支持；一次支持：恰好 1 次。阈值在本轮提取前写入 plan.json，没有按准确率调节。不可推荐的历史后续仍计入占比分母。',
        '证据年龄只作解释，不用来删样。窗口类别证据不等于同一个具体窗口的重复行为。', '',
        '## 固定诊断集的评分', '',
        '| 清洗 | 证据组 | 样本数 | 占全部后期测试 | Top1 | Top4 | 候选遗漏 |',
        '|---|---|---:|---:|---:|---:|---:|']
    for threshold, result in metrics.items():
        m = result['test']
        for title, s in [('全部有证据', m['evidence_supported'])] + [(TITLES[g], m['grades'][g]) for g in GRADES[:-1]]:
            lines.append(f"| {threshold} 秒 | {title} | {s['n']} | {s['coverage']:.1%} | {fmt_rate(s['top1_hits'], s['n'])} | {fmt_rate(s['top4_hits'], s['n'])} | {s['candidate_missing']} |")
    lines += ['', '候选遗漏仍计入上表的失败。下面的候选内结果只用于分离候选构造与排序问题，不替代主分母。', '']
    for threshold, result in metrics.items():
        s = result['test']['evidence_supported_candidate_present']
        lines.append(f"- {threshold} 秒：有证据且目标在候选中 {s['n']} 次，Top4 {fmt_rate(s['top4_hits'], s['n'])}；全部有证据错误分类 {result['test']['miss_reasons']}。")
    lines += ['', '## 预测前能识别的支持强弱', '',
        '上表按实际下一步有无历史支持进行事后归类。模型预测时不知道实际下一步，不能把这些样本的正确答案或入选标记作为输入。下表用相同门槛判断首选候选的支持，仅读取预测前已有历史。', '',
        '| 清洗 | 首选证据组 | 样本数 | 占全部测试 | Top1 | Top4 |', '|---|---|---:|---:|---:|---:|']
    for threshold, result in metrics.items():
        for g in GRADES:
            s = result['test']['prospective_first_candidate_grades'][g]
            lines.append(f"| {threshold} 秒 | {TITLES[g]} | {s['n']} | {s['coverage']:.1%} | {fmt_rate(s['top1_hits'], s['n'])} | {fmt_rate(s['top4_hits'], s['n'])} |")
    lines += ['', '## 评分边界与后续使用', '',
        '- 使用固定 evidence-test-5s-private.jsonl 中的全部样本评价；按样本 ID 对齐，精确具体目标计分，空推荐或漏候选仍算错。Top4 是前四个合法、互异且符合现有层级规则的候选中包含实际目标；同应用但不同具体目标不算命中。',
        f'- 先预测再使用 observed_outcome、diagnosis_after_outcome 评分；这些字段以及按真实答案挑出的 supporting_outcome_ids 禁止用于预测或门槛选择。输入限定在 pre_prediction；展示名称来自汇总目录，只用于阅读。inputs-test-5s-private.jsonl 提供全部 {total_n} 个截点的无答案输入，不能先按正确答案支持情况决定是否预测。',
        f'- score_current_evidence.py 要求对全部后期测试截点提交预测，再按固定证据组评分；拒绝漏交样本、超过四项、重复目标、未在冻结候选中的目标或不符合层级规则的推荐。若修复候选生成，需要保留本次样本成员并另行审计新候选的时间依据；当前候选冻结时，有证据主集的 Top4 上限至多 {covered_n}/{supported_n}（{ceiling:.1%}）。',
        '- 这一诊断集旨在检查已有证据是否被利用，不代表全部场景或可部署的高置信接受集。未来门槛只在独立验证段选择，后续新数据确认；测试段早已多次查看，不能当作新保留集。',
        '- 只有一次支持是可追溯线索，不是行为必然重复的证明；支持强也不是校准后的成功概率。同起点出现不同下一步会产生真实歧义。',
        '- 输入来源完整的子集、窗口类别支持数量、验证段分组见 metrics.json。原始独立到达与后台库存的各类型排行保留在 destination-rankings-private.json，两者不混计。',
        '- 全部原始事件与已有 run 保持不变，线上菜单未更新。', '',
        '## 5 秒主诊断集逐条清单', '',
        '每项包含预测截点、当前界面、实际下一步、支持数/全部后续数、证据年龄、推荐与命中。完整定位符及所有历史事件指针见同目录 JSONL。', '']
    for g in GRADES[:-1]:
        lines += [f'### {TITLES[g]}', '']
        group = [r for r in primary_rows if r['diagnosis_after_outcome']['grade'] == g]
        if not group:
            lines += ['无样本。', '']
        for r in group:
            e = r['diagnosis_after_outcome']; p = r['pre_prediction']; b = r['baseline_prediction']
            lines += [f"- `{r['sample_id']}` · {p['time_utc']} · {r['display_only']['current_description']} → {r['observed_outcome']['description']}",
                f"  - 支持 {e['support_count']}/{e['origin_outcome_count']}（{e['share']:.1%}）；最近支持距截点 {e['hours_since_last_support']:.2f} 小时；证据层级 {e['identity_level']}。",
                f"  - 推荐：{'、'.join(b['descriptions']) or '空'}；Top1 {'✓' if r['evaluation']['hit1'] else '×'}；Top4 {'✓' if r['evaluation']['hit4'] else '×'}；{r['evaluation']['miss_reason']}。"]
    (out / 'report-private.md').write_text('\n'.join(lines) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True, help='New private directory containing only the prewritten plan.json')
    parser.add_argument('--prepare', action='store_true', help='Only create the new private directory and fixed grading plan')
    args = parser.parse_args(); run = args.run.expanduser().resolve(); out = args.output.expanduser().resolve()
    if not out.is_relative_to(live_core.PRIVATE_ROOT.resolve()) or out == live_core.PRIVATE_ROOT.resolve():
        parser.error('Use a private training directory')
    if args.prepare:
        verify_run(run)
        os.umask(0o077); out.mkdir(mode=0o700, parents=True, exist_ok=False)
        live_core.write_json(out / 'plan.json', make_plan(run))
        print(json.dumps({'prepared': str(out)}))
        return
    if not out.is_dir() or {p.name for p in out.iterdir()} != {'plan.json'}:
        parser.error('Output must contain only the fixed plan.json; never overwrite a completed run')
    plan = json.loads((out / 'plan.json').read_text())
    if plan['version'] != VERSION or plan['grades'] != GRADE_DEFINITIONS or Path(plan['source_run']) != run:
        raise ValueError('Plan and implementation disagree')
    os.umask(0o077)
    provenance, hashes = verify_run(run)
    base = Path(provenance['source_run']); source, source_hashes = verify_run(base)
    if source_hashes != provenance['source_artifact_hashes']:
        raise ValueError('Source run provenance changed')
    for name in ('evaluate_ordered_current.py', 'evaluate_evidence_model.py', 'evaluate_context_knn.py', 'diagnose_dwell_errors.py', 'summarize_tuned_history.py'):
        if live_core.sha(Path(__file__).with_name(name)) != live_core.sha(run / ('source-' + name)):
            raise ValueError('Replayer dependency differs from frozen source: ' + name)
    core = frozen_core(base); snapshot = Path(source['snapshot'])
    snapshot_hashes, _ = core.verified_snapshot(snapshot)
    if snapshot_hashes != source['input_hashes']:
        raise ValueError('Source snapshot changed')
    events = core.load_lines(snapshot / 'events.jsonl')
    original, *_ = core.make_observations(events, core.load_lines(snapshot / 'navigation-samples.jsonl'), set(source['excluded_sessions']))
    blocked = {core.key_for(e['data']['target'], e['sessionID']) for e in events if e['kind'] == 'target.update'
        and e['data']['target'].get('bundleID') in DENIED_BUNDLES}
    original = [clean_input(q, blocked) for q in original]
    ledgers = json.loads((base / 'labels-private.json').read_text())
    saved = json.loads((run / 'predictions-private.json').read_text())
    names = {e['app']['bundleID']: e['app']['name'] for e in events if e.get('app', {}).get('name')}
    results = {}; membership = {}; submissions = {}; five_second_test = []
    for threshold in ('3', '5'):
        ledger = ledgers[threshold]; items = list(original); valid = set()
        for r in ledger:
            i = r['position']; q = items[i]
            if q.sample_id != r['sample_id'] or q.index != r['input_through_event_index']:
                raise ValueError('Cleaning ledger differs from source inputs')
            if r['status'] != 'stable_target' or q.current_app in DENIED_BUNDLES or r['target'].get('bundleID') in DENIED_BUNDLES:
                continue
            target = r['target']; valid.add(i)
            items[i] = replace(q, truth=core.key_for(target, q.session), truth_meta={**target, '_bundle': target.get('bundleID')})
        rows = reconstruct(core, items, valid, ledger, saved[threshold], events, names)
        test = [r for r in rows if r['split'] == 'test' and r['primary_cutoff']]
        validation = [r for r in rows if r['split'] == 'validation' and r['primary_cutoff']]
        supported = [r for r in test if r['diagnosis_after_outcome']['eligible']]
        results[threshold] = {'test': summarize_groups(test), 'validation': summarize_groups(validation),
            'secondary_all_test_cutoffs': summarize_groups([r for r in rows if r['split'] == 'test']),
            'reproduced_saved_prediction_count': len(rows)}
        write_jsonl(out / f'all-test-{threshold}s-private.jsonl', test)
        write_jsonl(out / f'inputs-test-{threshold}s-private.jsonl', [
            {'sample_id': r['sample_id'], 'position': r['position'], 'pre_prediction': r['pre_prediction']} for r in test])
        write_jsonl(out / f'evidence-test-{threshold}s-private.jsonl', supported)
        write_jsonl(out / f'validation-{threshold}s-private.jsonl', validation)
        write_jsonl(out / f'evidence-misses-{threshold}s-private.jsonl', [r for r in supported if not r['evaluation']['hit4']])
        membership[threshold] = {g: [r['sample_id'] for r in supported if r['diagnosis_after_outcome']['grade'] == g] for g in GRADES[:-1]}
        submissions[threshold] = [{'sample_id': r['sample_id'], 'predictions': r['baseline_prediction']['keys']} for r in test]
        if threshold == '5':
            five_second_test = test
    core.write_json(out / 'metrics.json', results)
    core.write_json(out / 'benchmark-membership.json', {'version': VERSION, 'definition': plan['definition'], 'sample_ids_by_grade': membership})
    core.write_json(out / 'baseline-submission-private.json', submissions)
    (out / 'destination-rankings-private.json').write_bytes((run / 'destination-rankings-private.json').read_bytes())
    render_report(out, results, five_second_test)
    (out / 'analysis-source.py').write_text(Path(__file__).read_text())
    (out / 'core-source.py').write_bytes((base / 'core-source.py').read_bytes())
    core.write_json(out / 'run.json', {'version': VERSION, 'source_run': str(run), 'source_artifact_hashes': hashes,
        'cleaning_run': str(base), 'cleaning_artifact_hashes': source_hashes, 'snapshot': str(snapshot),
        'input_hashes': snapshot_hashes, 'test_previously_seen': True, 'model_updated': False,
        'predictions_or_labels_changed': False, 'parameter_search': False})
    core.write_json(out / 'artifact-hashes.json', {p.name: core.sha(p) for p in out.iterdir() if p.is_file()})
    print(json.dumps({'output': str(out), 'results': {t: r['test'] for t, r in results.items()}}, ensure_ascii=False))


if __name__ == '__main__':
    main()
