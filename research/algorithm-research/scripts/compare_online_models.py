#!/usr/bin/env python3
"""Causal model sweep on frozen outcome-conditioned diagnostic cohorts.

Only current-interface features reach learners. Cleaning is a separate training
factor; every primary score uses the same frozen five-second truth and 62 IDs.
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
from dataclasses import replace
import importlib
import json
import math
import os
from pathlib import Path
import statistics
import sys
import time

import evaluate_context_knn as live_core
from diagnose_dwell_errors import frozen_core, label
from evaluate_evidence_model import DENIED_BUNDLES, clean_input, semantic_label, resolve_label
from evaluate_ordered_current import current_context
from summarize_tuned_history import verify_run

VERSION = 'current-online-comparison-v1'
ROOT = live_core.PRIVATE_ROOT
DEFAULT_EXPANSION = ROOT/'ordered-current-evidence-expansion-v1-20260923-assessment-01'


def semantic_candidate(key, meta):
    if meta['kind'] == 'window' and meta.get('_bundle'):
        return 'window_class:' + meta['_bundle']
    return key


def model_input(q):
    """Explicit allowlist: no truth, strict, evidence grade, future witness, history."""
    context = current_context(q)
    origin = context.targets[0] if context else '__unknown_current__'
    kind = 'window_class' if origin.startswith('window_class:') else origin.split(':', 1)[0]
    app = q.current_app or ''
    x = {'origin=' + origin: 1., 'app=' + app: 1., 'kind=' + kind: 1., 'bias': 1.}
    mapping = {}
    for key, meta in q.candidates.items():
        sem = semantic_candidate(key, meta)
        resolved = resolve_label(sem, q)
        if resolved:
            mapping[sem] = resolved
    recency = {}
    for sem, key in mapping.items():
        if key in q.last_visits:
            t = q.last_visits[key]
            if t > q.timestamp:
                raise ValueError('Future candidate arrival timestamp')
            recency[sem] = t
    return {'origin': origin, 'app': app, 'timestamp': q.timestamp, 'x': x,
            'labels': sorted(mapping), 'candidate_last_seen': recency}, mapping


def project_scores(core, scores, mapping, candidates):
    if any(not isinstance(v, (int, float)) or not math.isfinite(v) for v in scores.values()):
        raise ValueError('Nonfinite/non-numeric model score')
    raw = sorted((mapping[k] for k in scores if k in mapping),
                 key=lambda k: (-scores[semantic_candidate(k, candidates[k])], k))
    if len(raw) != len(set(raw)):
        raise ValueError('Semantic mapping duplicated a candidate')
    shown = core.select_four(raw, candidates)
    assert len(shown) <= 4 and len(shown) == len(set(shown))
    return raw, shown


def read_rows(path):
    result = []
    with path.open() as f:
        for line in f:
            r = json.loads(line)
            # Keep private full locators only in the established parent run.
            result.append({k: r[k] for k in ('sample_id', 'position', 'split', 'observed_outcome',
                'diagnosis_after_outcome', 'evaluation', 'baseline_prediction', 'display_only')})
    return result


def load_dataset(expansion):
    exp, exp_hashes = verify_run(expansion)
    audit_path = Path(exp['source_run']); audit, audit_hashes = verify_run(audit_path)
    assert audit_hashes == exp['source_artifact_hashes']
    base = Path(audit['cleaning_run']); source, source_hashes = verify_run(base)
    assert source_hashes == audit['cleaning_artifact_hashes']
    core = frozen_core(base); snapshot = Path(source['snapshot'])
    hashes, _ = core.verified_snapshot(snapshot)
    assert hashes == source['input_hashes'] == audit['input_hashes']
    events = core.load_lines(snapshot/'events.jsonl')
    original, observation_audit, *_ = core.make_observations(events,
        core.load_lines(snapshot/'navigation-samples.jsonl'), set(source['excluded_sessions']))
    blocked = {core.key_for(e['data']['target'], e['sessionID']) for e in events
        if e['kind'] == 'target.update' and e['data']['target'].get('bundleID') in DENIED_BUNDLES}
    original = [clean_input(q, blocked) for q in original]
    ledger = json.loads((base/'labels-private.json').read_text())
    membership = json.loads((expansion/'benchmark-membership.json').read_text())['sample_ids_by_grade']
    cohorts = {}
    for threshold in ('5', '3'):
        rows = read_rows(audit_path/f'validation-{threshold}s-private.jsonl') + read_rows(audit_path/f'all-test-{threshold}s-private.jsonl')
        expected = {sid: g for g, ids in membership[threshold].items() for sid in ids}
        assert len(expected) == sum(map(len, membership[threshold].values()))
        assert {r['sample_id']: r['diagnosis_after_outcome']['grade'] for r in rows if r['diagnosis_after_outcome']['eligible']} == expected
        assert len({r['sample_id'] for r in rows}) == len(rows)
        for r in rows:
            q = original[r['position']]
            assert q.sample_id == r['sample_id'] and q.reference_max_index <= q.index
            assert r['observed_outcome']['key'] == core.key_for(ledger[threshold][r['position']]['target'], q.session)
            assert (r['observed_outcome']['key'] in q.candidates) == r['diagnosis_after_outcome']['candidate_present']
        cohorts[threshold] = sorted(rows, key=lambda r: r['position'])
    event_index = {e['id']: i for i, e in enumerate(events)}
    outcome_times = {e['id']: e['timestampMs'] for e in events}
    inputs, mappings = {}, {}
    positions = sorted({r['position'] for rows in cohorts.values() for r in rows})
    for i in positions:
        inputs[i], mappings[i] = model_input(original[i])
    streams, cleaning_audits = {}, {}
    for threshold in ('raw', '3', '5'):
        pending = []; excluded = Counter()
        for i, q in enumerate(original):
            if q.current_app in DENIED_BUNDLES:
                excluded['denied_origin'] += 1; continue
            if threshold == 'raw':
                t, outcome_id = q.truth_meta, q.next_event_id
                destination = semantic_label(q, 'window_class')
            else:
                row = ledger[threshold][i]
                assert row['position'] == i and row['sample_id'] == q.sample_id
                assert row['label_ready_event_index'] == q.ready_index
                if row['status'] != 'stable_target':
                    excluded[row['status']] += 1; continue
                t = {**row['target'], '_bundle': row['target'].get('bundleID')}
                outcome_id = row['target_event_id']
                destination = semantic_label(replace(q, truth=core.key_for(t, q.session), truth_meta=t), 'window_class')
            if t.get('_bundle') in DENIED_BUNDLES or t.get('bundleID') in DENIED_BUNDLES:
                excluded['denied_destination'] += 1; continue
            context = current_context(q)
            if context is None:
                excluded['unknown_origin'] += 1; continue
            inp, _ = model_input(q)
            assert q.index < event_index[outcome_id] <= q.ready_index
            record = {k: inp[k] for k in ('origin', 'app', 'timestamp', 'x')}
            record.update(label=destination, outcome_timestamp=outcome_times[outcome_id], outcome_id=outcome_id)
            pending.append({'position': i, 'input_index': q.index, 'ready_index': q.ready_index,
                'outcome_index': event_index[outcome_id], 'record': record})
        pending.sort(key=lambda r: (r['ready_index'], r['position']))
        seen = set(); dedup = []
        for row in pending:
            key = (row['record']['origin'], row['record']['outcome_id'])
            if key not in seen:
                seen.add(key); dedup.append(row)
        streams[threshold] = dedup
        cleaning_audits[threshold] = {'eligible_training_observations': len(pending),
            'unique_origin_outcomes': len(dedup), 'duplicates_removed': len(pending)-len(dedup), 'excluded': dict(excluded)}
    provenance = {'expansion_run': str(expansion), 'expansion_hashes': exp_hashes,
        'audit_run': str(audit_path), 'audit_hashes': audit_hashes, 'cleaning_run': str(base),
        'cleaning_hashes': source_hashes, 'snapshot': str(snapshot), 'input_hashes': hashes,
        'observation_audit': observation_audit, 'cleaning_audits': cleaning_audits}
    return core, original, cohorts, streams, inputs, mappings, provenance


def metric(rows, predictions):
    n = len(rows)
    ranks = [predictions[r['position']]['predictions'].index(r['observed_outcome']['key']) + 1
        if r['observed_outcome']['key'] in predictions[r['position']]['predictions'] else 0 for r in rows]
    covered = sum(r['diagnosis_after_outcome']['candidate_present'] for r in rows)
    return {'n': n, 'top4_hits': sum(0 < r <= 4 for r in ranks), 'top1_hits': sum(r == 1 for r in ranks),
        'top4': sum(0 < r <= 4 for r in ranks)/n if n else None,
        'top1': sum(r == 1 for r in ranks)/n if n else None,
        'mrr4': sum(1/r if r else 0 for r in ranks)/n if n else None,
        'candidate_present': covered, 'candidate_missing': n-covered}


def score_groups(rows, predictions):
    evidence = [r for r in rows if r['diagnosis_after_outcome']['eligible']]
    return {'all': metric(rows, predictions), 'supported': metric(evidence, predictions),
        'late_supported': metric([r for r in evidence if r['split'] == 'test'], predictions),
        'early_supported': metric([r for r in evidence if r['split'] == 'validation'], predictions),
        'covered_supported': metric([r for r in evidence if r['diagnosis_after_outcome']['candidate_present']], predictions),
        'strict_supported': metric([r for r in evidence if r['evaluation']['strict_input_and_label_coverage']], predictions),
        'grades': {g: metric([r for r in evidence if r['diagnosis_after_outcome']['grade'] == g], predictions)
            for g in ('strong', 'repeated', 'single')}}


def replay(core, model, items, stream, inputs, mappings):
    cursor = 0; predictions = {}; learn_ms = []; predict_ms = []
    for i in sorted(inputs):
        q = items[i]
        while cursor < len(stream) and stream[cursor]['ready_index'] < q.index:
            row = stream[cursor]; cursor += 1
            assert row['position'] < i and row['input_index'] < q.index and row['outcome_index'] < q.index
            start = time.perf_counter(); model.learn(row['record']); learn_ms.append((time.perf_counter()-start)*1000)
        start = time.perf_counter()
        scores = model.scores(inputs[i])
        raw, shown = project_scores(core, scores, mappings[i], q.candidates)
        predict_ms.append((time.perf_counter()-start)*1000)
        predictions[i] = {'sample_id': q.sample_id, 'predictions': shown, 'raw_ranking': raw,
            'learned_unique_outcomes': cursor}
    def timing(values):
        return {'n': len(values), 'median_ms': statistics.median(values) if values else 0,
            'p95_ms': sorted(values)[int(.95*(len(values)-1))] if values else 0,
            'total_ms': sum(values)}
    return predictions, {'learn': timing(learn_ms), 'predict': timing(predict_ms)}


def selection_key(r, group='supported'):
    m = r['scores']['5'][group]
    return (-m['top4_hits'], -m['top1_hits'], -m['mrr4'], r['config']['id'], r['training_cleaning'])


def diagnose(rows, predictions):
    errors = []
    for r in rows:
        if not r['diagnosis_after_outcome']['eligible']:
            continue
        p = predictions[r['position']]; truth = r['observed_outcome']['key']
        if truth in p['predictions']:
            continue
        rank = p['raw_ranking'].index(truth)+1 if truth in p['raw_ranking'] else None
        reason = ('candidate_missing' if not r['diagnosis_after_outcome']['candidate_present'] else
            'window_class_ambiguous' if not r['diagnosis_after_outcome']['support_resolves_to_actual_candidate'] else
            'hierarchy_suppression' if rank and rank <= 4 else
            'unscored_candidate' if rank is None else 'rank_below_four')
        errors.append({'sample_id': r['sample_id'], 'position': r['position'], 'split': r['split'],
            'origin': r['display_only']['current_description'], 'actual': r['observed_outcome']['description'],
            'grade': r['diagnosis_after_outcome']['grade'], 'support_count': r['diagnosis_after_outcome']['support_count'],
            'origin_outcome_count': r['diagnosis_after_outcome']['origin_outcome_count'],
            'reason': reason, 'raw_rank': rank, 'predictions': p['predictions']})
    return errors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--expansion-run', type=Path, default=DEFAULT_EXPANSION)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--suite', choices=('stat', 'online', 'supplemental', 'all'), default='all')
    parser.add_argument('--training-cleaning', nargs='+', default=['5', '3', 'raw'], choices=('5','3','raw'))
    parser.add_argument('--config-ids', nargs='*')
    args = parser.parse_args(); out = args.output.resolve()
    if not out.is_relative_to(ROOT.resolve()) or out == ROOT.resolve():
        parser.error('Use a new private training directory')
    modules = []
    if args.suite in ('stat', 'all'):
        modules.append(importlib.import_module('stat_model_suite'))
    if args.suite in ('online', 'supplemental', 'all'):
        modules.append(importlib.import_module('online_model_suite'))
    configs = [(module, c) for module in modules for c in
        (module.supplemental_configurations() if args.suite == 'supplemental' else module.configurations())
        if not args.config_ids or c['id'] in args.config_ids]
    for _, config in configs:
        config['feature_scope'] = ('current_plus_clock' if config['family'] == 'clock_kernel' else
            'current_plus_candidate_recency' if config['family'] in ('candidate_recency', 'count_candidate_recency')
            else 'current_only')
    assert configs and len({c['id'] for _, c in configs}) == len(configs)
    os.umask(0o077); out.mkdir(mode=0o700, parents=True, exist_ok=False)
    plan = {'version': VERSION, 'current_context_length': 1, 'features': ['current_origin', 'current_app', 'current_kind'],
        'no_previous_sequence_or_background_similarity': True,
        'candidate_recency': 'separate identified statistical ablation; only already observed arrivals',
        'clock_kernel': 'separate identified ablation using observed time of day in addition to current interface',
        'primary_cohort': 'frozen expanded 5s evidence IDs and answers, 62 rows; never selected by correctness',
        'sensitivity_cohort': 'frozen native 3s evidence IDs and answers, 64 rows; separate result',
        'training_cleaning_factors': args.training_cleaning, 'configurations': [c for _, c in configs],
        'label_maturity': 'original complete30s horizon ready_index strictly before prediction input index',
        'deduplication': '(current semantic origin, actual outcome event ID)',
        'candidate_policy': 'frozen existing candidates; window class must resolve uniquely; unchanged hierarchy suppression',
        'selection': 'maximize full62 Top4, then Top1, MRR4, stable configuration ID; development best only',
        'additional_temporal_selection': 'select on earlier24 evidence rows, report original late38 without using it for that choice',
        'raw_data_modified': False, 'live_model_updated': False, 'test_previously_seen': True,
        'exhaustive_all_possible_models': False, 'runtime': {'python': sys.version, 'executable': sys.executable}}
    core_write = live_core.write_json
    core_write(out/'plan.json', plan)
    for file in [Path(__file__), *(Path(m.__file__) for m in modules),
        *[Path(__file__).with_name(n) for n in ('evaluate_context_knn.py','evaluate_evidence_model.py',
          'evaluate_ordered_current.py','diagnose_dwell_errors.py','summarize_tuned_history.py')]]:
        (out/('source-'+file.name)).write_bytes(file.read_bytes())
    core, items, cohorts, streams, inputs, mappings, provenance = load_dataset(args.expansion_run.resolve())
    core_write(out/'input-audit.json', provenance)
    if any(m.__name__ == 'online_model_suite' for m in modules):
        module = next(m for m in modules if m.__name__ == 'online_model_suite')
        if hasattr(module, 'dependency_metadata'):
            core_write(out/'dependencies.json', module.dependency_metadata())
    all_results = []; best_predictions = {}
    for training in args.training_cleaning:
        for index, (module, config) in enumerate(configs):
            model = module.build(config)
            pred, timings = replay(core, model, items, streams[training], inputs, mappings)
            result = {'training_cleaning': training, 'config': config, 'timing': timings,
                'scores': {t: score_groups(rows, pred) for t, rows in cohorts.items()}}
            if config['id'] == 'stat_frequency_conditional' and training in ('3','5'):
                assert all(pred[r['position']]['predictions'] == r['baseline_prediction']['keys']
                    for r in cohorts[training]), 'Frozen transition baseline parity failed'
                result['frozen_baseline_parity'] = True
            all_results.append(result)
            path = out/f"predictions-{training}-{config['id']}-private.json"
            core_write(path, {'config': config, 'training_cleaning': training,
                'predictions': {str(k): v for k, v in pred.items()}})
            if index % 10 == 0 or index+1 == len(configs):
                print(json.dumps({'stage': 'model', 'training_cleaning': training, 'done': index+1,
                    'total': len(configs), 'model': config['id'], 'main': result['scores']['5']['supported']}), flush=True)
        core_write(out/f'metrics-{training}.json', [r for r in all_results if r['training_cleaning'] == training])
    all_results.sort(key=selection_key)
    best = all_results[0]
    def read_pred(result):
        path = out/f"predictions-{result['training_cleaning']}-{result['config']['id']}-private.json"
        return {int(k): v for k, v in json.loads(path.read_text())['predictions'].items()}
    by_family = {}
    for result in all_results:
        by_family.setdefault(result['config']['family'], result)
    early_best = sorted(all_results, key=lambda r: selection_key(r, 'early_supported'))[0]
    main5 = [r for r in all_results if r['training_cleaning'] == '5']
    best_fixed5 = main5[0] if main5 else None
    selection = {'best_development': best, 'best_fixed5_training': best_fixed5,
        'best_per_family': by_family, 'earlier_segment_selected': early_best,
        'number_configurations': len(configs), 'number_replays': len(all_results)}
    core_write(out/'selection.json', selection)
    core_write(out/'metrics.json', all_results)
    core_write(out/'best-errors-private.json', diagnose(cohorts['5'], read_pred(best)))
    report(out, selection, all_results, provenance, diagnose(cohorts['5'], read_pred(best)))
    core_write(out/'run.json', {'version': VERSION, **provenance, 'raw_data_modified': False,
        'live_model_updated': False, 'test_previously_seen': True, 'complete': True})
    core_write(out/'artifact-hashes.json', {p.name: live_core.sha(p) for p in out.iterdir() if p.is_file()})
    print(json.dumps({'complete': True, 'output': str(out), 'best': {'config': best['config'],
        'training_cleaning': best['training_cleaning'], 'main': best['scores']['5']['supported']},
        'early_selected': {'config': early_best['config'], 'late': early_best['scores']['5']['late_supported']},
        'number_replays': len(all_results)}), flush=True)


def report(out, selection, results, provenance, errors):
    def cell(m):
        return f"{m['top4_hits']}/{m['n']} ({100*m['top4']:.1f}%)" if m['n'] else '0/0'
    b = selection['best_development']
    lines = ['# 当前界面在线模型比较', '',
        '主评分始终使用冻结的 5 秒清洗答案与 62 条有证据样本；模型只接收当前界面、应用、类别，统计近期访问是单独注明的消融项。标签满原 30 秒观察窗口后才可学习，同一起点的同一后续到达去重。',
        '这是已反复查看的开发回放，搜索最高分不是新数据泛化准确率。清洗是独立训练因子；原始训练、3 秒训练、5 秒训练都在同一主评分集比较。另报原生 3 秒口径，不能混为同一批样本。', '',
        f"实际比较 {selection['number_configurations']} 组模型参数，{selection['number_replays']} 次顺序回放；不声称穷尽所有可能算法。", '',
        '| 模型家族最优配置 | 训练清洗 | 主62 Top4 | 原晚38 Top4 | Top1 |', '|---|---|---:|---:|---:|']
    for family, r in selection['best_per_family'].items():
        m = r['scores']['5']['supported']; late = r['scores']['5']['late_supported']
        lines.append(f"| {family} / {r['config']['id']} | {r['training_cleaning']} | {cell(m)} | {cell(late)} | {m['top1_hits']}/{m['n']} |")
    lines += ['', '## 最优结果与分组', '', f"配置：`{json.dumps(b['config'], ensure_ascii=False)}`；训练清洗 `{b['training_cleaning']}`。", '',
        '| 证据分组 | Top4 | Top1 |', '|---|---:|---:|']
    for g,m in b['scores']['5']['grades'].items():
        lines.append(f"| {g} | {cell(m)} | {m['top1_hits']}/{m['n']} |")
    lines += ['', f"原候选缺失6条，固定候选上限56/62=90.3%；这些失败保留。最优模型错因：{dict(Counter(e['reason'] for e in errors))}。", '',
        '## 固定模型的清洗对照', '', '| 训练标签处理 | 同一主62 Top4 | 晚38 Top4 |', '|---|---:|---:|']
    for r in results:
        if r['config']['id'] == b['config']['id']:
            lines.append(f"| {r['training_cleaning']} | {cell(r['scores']['5']['supported'])} | {cell(r['scores']['5']['late_supported'])} |")
    e = selection['earlier_segment_selected']
    lines += ['', '## 时间分段选参核对', '',
        f"仅按较早24条选参得到 `{e['config']['id']}`（训练清洗 `{e['training_cleaning']}`），较早段 {cell(e['scores']['5']['early_supported'])}，较晚38条 {cell(e['scores']['5']['late_supported'])}。参数选择和后期评分分开，但历史数据此前已看过，仍不是全新的独立验证。", '',
        '## 数据处理', '',
        '- 5 秒作为主标签，3 秒敏感性；仅跳过已完整观察且停留短于阈值的中间目标。正常同标签页导航依据实际窗口/入口关系标记，归属未知不猜测，未知时长不当短停留。',
        '- A→短B→A 不继续寻找更晚C；无法在30秒内确认稳定目标不作为清洗训练标签。未清洗对照仍遵守准入、来源和成熟时间。',
        '- 原始采集、此前 run、实际推荐菜单均未修改。完整目标与预测仅保存私有 run，父 run 的完整定位符仍是可追溯依据。', '',
        '## 可行性与边界', '',
        '- 各模型逐条更新，所有类词表只来自已成熟训练标签；评测对象不会作为训练前置类词表。',
        '- 当前上下文长度1；不偷加前序页面或后台相似度。窗口学习为应用窗口类别，预测必须唯一对应当前具体候选。',
        '- 在固定有证据集合优化不等于线上能提前知道这些样本有证据，未承诺自动置信门控。',
        '- timing 为本机离线单次更新/预测开销，不包含采集或等待标签稳定，完整数值在 metrics.json。', '',
        '详细配置、清洗审计、每条预测、错误与哈希均在此 run。', '']
    (out/'report-private.md').write_text('\n'.join(lines))

if __name__ == '__main__':
    main()
