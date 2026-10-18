"""Audit causal evidence and representation failures without changing evaluation."""
import argparse
from collections import Counter, defaultdict
from copy import deepcopy
from datetime import datetime, timezone
import gzip
import hashlib
import importlib
import json
import os
from pathlib import Path
import sys
import evaluation_contract as scoring
from urllib.parse import urlsplit


REASONS = {
    'same_application_suppressed': '评分要求同应用标签，但输出语义过滤该标签',
    'current_destination_suppressed': '评分要求当前目标，但输出语义过滤该标签',
    'service_folded_specific_target': '具体会话被归并为服务，未通过具体目标恢复门槛',
    'unobserved_target': '具体目标在截点前尚未观察到',
    'known_without_output_class': '目标已知但尚无可输出的成熟类别',
    'residual_demoted_hit': '统计本已命中，学习修正将其移出前四',
    'reference_training_mismatch': '参考证据存在，但同起点支持未进入当前训练标签',
    'supported_ranking_miss': '已有同起点训练支持，但排序落后',
    'weak_transfer_ranking_miss': '可输出目标，但缺少同起点训练支持',
}


def rank(label, values): return values.index(label)+1 if label in values else None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--alternatives-run', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args(); alt, out = args.alternatives_run.resolve(), args.output.resolve()
    root = Path.home()/'Library/Application Support/SmartApp/training'
    if out.exists() or out == root or not out.is_relative_to(root): parser.error('Use a new private run')
    def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
    def verify(path):
        checks = json.loads((path/'artifact-hashes.json').read_text())
        for name, digest in checks.items():
            if Path(name).name != name or sha(path/name) != digest: raise ValueError('Changed artifact: '+name)
        return checks
    def write(path, value): path.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n')
    alt_hashes = verify(alt); alt_meta = json.loads((alt/'run.json').read_text())
    residual_run = Path(alt_meta['source']); residual_hashes = verify(residual_run)
    residual_meta = json.loads((residual_run/'run.json').read_text())
    fusion = Path(residual_meta['source']); fusion_hashes = verify(fusion)
    fusion_meta = json.loads((fusion/'run.json').read_text())
    dataset, learning, statistics = (Path(fusion_meta[k]) for k in ('dataset', 'learning_run', 'statistics_run'))
    dependencies = {p: verify(p) for p in (dataset, learning, statistics)}
    original_dataset = Path(json.loads((dataset/'run.json').read_text())['base_dataset'])
    dependencies[original_dataset] = verify(original_dataset)
    snapshot = Path(fusion_meta['snapshot']); raw_hash = sha(snapshot/'events.jsonl')
    if raw_hash != fusion_meta['raw_sha256']: raise ValueError('Changed raw snapshot')
    os.umask(0o077); sys.dont_write_bytecode = True; out.mkdir(mode=0o700)
    # Reuse frozen model and cleaning semantics. No model, label or threshold changes.
    sys.path.insert(0, str(residual_run))
    comp = importlib.import_module('compare_multibackground')
    cs = importlib.import_module('context_statistics')
    nav = importlib.import_module('navigation_hierarchy')
    residual = importlib.import_module('statistical_residual')
    metric = scoring.measure
    cohorts = scoring.cohorts
    import numpy as np
    from scipy.sparse import load_npz
    with gzip.open(dataset/'dataset-private.json.gz', 'rt') as f: data = json.load(f)
    with gzip.open(original_dataset/'dataset-private.json.gz', 'rt') as f: original_data = json.load(f)
    ledgers = {r['position']: r for r in original_data['trajectory_ledgers']['hard_kind']}
    with gzip.open(learning/'inputs-private.json.gz', 'rt') as f: inputs = json.load(f)
    for q in inputs: q.update(inventory_state={}, inventory_quality={}, inventory_through_index=q['index'])
    with gzip.open(residual_run/'causal-examples-private.json.gz', 'rt') as f: cache = json.load(f)
    with gzip.open(alt/'predictions-private.json.gz', 'rt') as f: alternatives = json.load(f)
    previous = json.loads((residual_run/'predictions-private.json').read_text())
    selected = json.loads((residual_run/'selection.json').read_text())['selected']
    pred = {int(k): v for k, v in previous[selected['id']].items()}
    statpred = previous['statistics']
    rows = scoring.prepare_evaluation(data, out); by_position = {r['position']: r for r in rows}; parts = cohorts(rows)
    misses = {r['position'] for r in rows if scoring.eligible(r) and
              not scoring.hits(r, pred[r['position']])['accepted'][4]}
    matrix = load_npz(residual_run/'causal-features.npz'); examples = {}
    for i, pos in enumerate(cache['positions']):
        item = cache['examples'][str(pos)]
        examples[pos] = {'labels': item['labels'], 'prior': np.array(item['prior']),
                         'X': matrix[cache['offsets'][i]:cache['offsets'][i+1]], 'mapping': item['mapping']}
    # Catalog knowledge and raw observation traces are diagnostic; never feed outcome metadata back into prediction.
    first_seen, catalog, contexts, event_positions = {}, {}, [], {}
    events_by_id = {}
    with (snapshot/'events.jsonl').open() as f:
        for index, line in enumerate(f):
            event = json.loads(line); event_positions[event['id']] = index
            if event['kind'] == 'target.update':
                target = event['data']['target']
                if comp.io.admitted(target) and not comp.io.synthetic(target):
                    catalog[target['id']] = target; first_seen.setdefault(target['id'], index)
                    if target.get('bundleID'): first_seen.setdefault('app:'+target['bundleID'], index)
            elif event['kind'] == 'context.change':
                target_id = event['data'].get('targetID')
                contexts.append({'index': index, 'id': event['id'], 'timestamp': event['timestampMs'],
                                 'source': event.get('source'), 'reason': event['data'].get('reason'),
                                 'target_id': target_id, 'target': deepcopy(catalog.get(target_id, {}))})
                events_by_id[event['id']] = contexts[-1]
    reference = data['reference_stream']
    references_by_pos = defaultdict(list)
    for r in reference: references_by_pos[r['position']].append(r)
    stream = [r for r in comp.training_rows(data, 'hard_kind') if r['head'] == 0]
    train_by_pos = {r['position']: r for r in stream}
    memory, background = cs.StatisticalMemory(), cs.BackgroundIndex(inputs)
    model = residual.ResidualRanker(selected['config']); cursor = residual_cursor = 0
    statconfig = json.loads((statistics/'model-time.json').read_text())['config']
    details = []
    for q in inputs:
        while cursor < len(stream) and stream[cursor]['ready_index'] < q['index']:
            item = stream[cursor]
            memory.learn(inputs[item['position']], comp.learning_record(item, inputs[item['position']]), item['ready_index'])
            cursor += 1
        while residual_cursor < len(cache['training']) and cache['training'][residual_cursor]['ready_index'] < q['index']:
            model.learn(cache['training'][residual_cursor], examples, q['index']); residual_cursor += 1
        pos = q['position']
        if pos in by_position:
            assert comp.bundle(model.scores(examples[pos]), cache['homes'][str(pos)]) == pred[pos]
        if pos not in misses: continue
        row = by_position[pos]; truth = row['truth_action']; identity = q['annotation']['identity']
        x = examples[pos]; output = pred[pos]
        prepared = memory.prepare(q, cache['homes'][str(pos)], background)
        assert prepared.predict(statconfig) == statpred[str(pos)]
        exact = [r for r in memory.hierarchy.records if r['origin'] == q['origin'] and r['label'] == truth]
        global_records = [r for r in memory.hierarchy.records if r['label'] == truth]
        eligible = [r for r in exact if r['navigationDisposition'] == 'navigation']
        raw_reference = [r for r in reference if r['ready_index'] < q['index'] and
                         r['record']['label'] == truth and inputs[r['position']]['origin'] == q['origin']]
        witness_details = []
        for r in raw_reference:
            item = train_by_pos.get(r['position'])
            witness_details.append({'position': r['position'], 'ready_index': r['ready_index'],
                                    'outcome_id': r['record']['outcome_id'],
                                    'selected_training_label': item['label'] if item else None,
                                    'selected_training_outcome': item['outcome_id'] if item else None,
                                    'matching_training_relation': bool(item and item['label'] == truth and
                                                                       item['outcome_id'] == r['record']['outcome_id']),
                                    'trajectory_stop_reason': ledgers.get(r['position'], {}).get('stop_reason'),
                                    'trajectory_observed_count': ledgers.get(r['position'], {}).get('observed_count'),
                                    'trajectory_retained_count': ledgers.get(r['position'], {}).get('retained_count')})
        original_witnesses = row.get('witnesses') or []
        for witness in original_witnesses: assert witness['ready_index'] < q['index']
        disposition = nav.disposition(truth, identity)
        service = nav.service(row['truth_metadata'])
        model_rank = rank(truth, output['rankable']); stat_rank = rank(truth, statpred[str(pos)]['rankable'])
        known = first_seen.get(truth, float('inf')) <= q['index']
        if disposition == 'same_application_unspecified': reason = 'same_application_suppressed'
        elif disposition == 'current_destination': reason = 'current_destination_suppressed'
        elif model_rank is None and service and service in prepared.labels: reason = 'service_folded_specific_target'
        elif model_rank is None: reason = 'known_without_output_class' if known else 'unobserved_target'
        elif stat_rank is not None and stat_rank <= 4: reason = 'residual_demoted_hit'
        elif raw_reference and not eligible: reason = 'reference_training_mismatch'
        elif eligible: reason = 'supported_ranking_miss'
        else: reason = 'weak_transfer_ranking_miss'
        times = {}
        for scale, profile in q['backgrounds'].items():
            exact_key = 'target:'+truth
            window_key = 'target:window_class:'+truth[4:] if truth.startswith('app:') else ''
            owner = row['truth_metadata'].get('bundleID', '')
            times[scale] = {channel: {'target': profile[channel].get(exact_key, 0.),
                                     'window_class_same_app': profile[channel].get(window_key, 0.) if window_key else 0.,
                                     'owner': profile[channel].get('owner:'+owner, 0.)}
                            for channel in ('count', 'recent', 'dwell')}
            times[scale]['coverage'] = {k: profile[k] for k in ('confirmed_arrivals', 'observed_nonidle_seconds', 'available_span_seconds')}
        corrections = None
        if truth in x['labels']:
            base = np.asarray(x['X'][:, :residual.DIMENSIONS//2] @ model.weights[:residual.DIMENSIONS//2]).ravel()
            context = np.asarray(x['X'][:, residual.DIMENSIONS//2:] @ model.weights[residual.DIMENSIONS//2:]).ravel()
            logs = np.log(x['prior']); i = x['labels'].index(truth)
            no_context = dict(zip(x['labels'], map(float, np.exp(logs+base-(logs+base).max()))))
            diagnostic = comp.bundle(no_context, cache['homes'][str(pos)])
            corrections = {'truth_prior': float(x['prior'][i]), 'truth_base_delta': float(base[i]),
                           'truth_context_delta': float(context[i]),
                           'rank_without_context_correction': rank(truth, diagnostic['rankable']),
                           'top4_and_truth': [{'label': label, 'prior': float(x['prior'][j]),
                                               'base_delta': float(base[j]), 'context_delta': float(context[j])}
                                             for label in dict.fromkeys([*output['raw'], truth])
                                             if label in x['labels'] for j in [x['labels'].index(label)]]}
        service_details = None
        if service:
            related = [r for r in memory.hierarchy.records if r['origin'] == q['origin'] and r['navigationLabel'] == service]
            service_key = cache['homes'][str(pos)] or service
            service_details = {'same_origin_service_outcomes': len({r['outcomeID'] for r in related}),
                               'same_origin_concrete_outcomes': dict(Counter(r['label'] for r in {r['outcomeID']: r for r in related}.values())),
                               'chosen_specific': memory.hierarchy.specific(service, q['origin']),
                               'service_entry_rank': rank(service_key, output['rankable'])}
        ref = next((r for r in references_by_pos[pos] if r['record']['label'] == truth), None)
        outcome_id = ref['record']['outcome_id'] if ref else row.get('observed_outcome')
        outcome_index = event_positions.get(outcome_id) if isinstance(outcome_id, str) else None
        trace = [r for r in contexts if q['index'] <= r['index'] <= outcome_index] if outcome_index is not None else []
        trace_owners = {r['target'].get('bundleID') for r in trace if r['target'].get('bundleID')}
        trace_windows = {r['target_id'] for r in trace if r['target'].get('kind') == 'window'}
        trace_pattern = None
        if reason == 'same_application_suppressed':
            trace_pattern = ('intervening_other_app_then_return' if len(trace_owners) > 1 else
                             'different_windows_same_app' if len(trace_windows) > 1 else
                             'window_then_same_app_cross_source_observation')
        training_here = train_by_pos.get(pos)
        exact_counts = Counter(r['label'] for r in {r['outcomeID']: r for r in memory.hierarchy.records
                                                  if r['origin'] == q['origin'] and r['navigationDisposition'] == 'navigation'}.values())
        alternatives_hit = [name for name, predictions in alternatives.items()
                            if scoring.hits(row, predictions[str(pos)])['accepted'][4]]
        details.append({'position': pos, 'split': row['split'], 'primary_evidence': row['evidence_grade'] == 'original' or bool(row.get('new_exact_history')),
                        'expanded_evidence': row['evidence_grade'] != 'unsupported', 'evidence_grade': row['evidence_grade'],
                        'cutoff_index': q['index'], 'timestamp': q['timestamp'], 'reason': reason,
                        'current': q['target'], 'truth': row['truth_metadata'], 'truth_action': truth,
                        'known_before_cutoff': known, 'first_seen_index': first_seen.get(truth),
                        'model_rank': model_rank, 'statistical_rank': stat_rank, 'predictions': output['predictions'],
                        'mature_exact_origin_outcomes': len({r['outcomeID'] for r in exact}),
                        'mature_exact_origin_navigation_outcomes': len({r['outcomeID'] for r in eligible}),
                        'mature_global_truth_outcomes': len({r['outcomeID'] for r in global_records}),
                        'reference_exact_origin_outcomes': len({r['record']['outcome_id'] for r in raw_reference}),
                        'reference_witnesses': witness_details, 'original_witnesses': original_witnesses,
                        'same_origin_leading_labels': exact_counts.most_common(8), 'time_background': times,
                        'window_history_not_directly_aligned_to_app_target_feature': bool(truth.startswith('app:') and truth in x['labels'] and
                            any(v['count']['window_class_same_app'] > 0 for v in times.values())),
                        'corrections': corrections, 'service': service_details, 'alternative_models_hit': alternatives_hit,
                        'scored_outcome_id': outcome_id, 'scored_outcome_index': outcome_index,
                        'post_cutoff_trace_diagnostic_only': trace,
                        'post_cutoff_trace_pattern_diagnostic_only': trace_pattern,
                        'current_training_trajectory_stop_diagnostic_only': ledgers.get(pos, {}).get('stop_reason'),
                        'current_training_label_diagnostic_only': training_here['label'] if training_here else None})
    primary = [r for r in details if r['primary_evidence']]
    summary = {'model': selected['id'], 'scoring_contract': scoring.contract(),
               'metrics': {name: metric(part, pred) for name, part in parts.items()},
               'misses': {name: {'n': len(part), 'reasons': dict(Counter(r['reason'] for r in part)),
                                'distinct_scored_outcomes': len({r['scored_outcome_id'] for r in part}),
                                'other_model_top4_hit': sum(bool(r['alternative_models_hit']) for r in part)}
                          for name, part in [('primary', primary), ('expanded', [r for r in details if r['expanded_evidence']]), ('all', details)]},
               'primary_rankable_misses': sum(r['model_rank'] is not None for r in primary),
               'primary_output_blocked': sum(r['model_rank'] is None for r in primary),
               'primary_window_background_alignment_gap': sum(r['window_history_not_directly_aligned_to_app_target_feature'] for r in primary),
               'same_app_trace_patterns': dict(Counter(r['post_cutoff_trace_pattern_diagnostic_only'] for r in primary
                                                       if r['reason'] == 'same_application_suppressed')),
               'all_prediction_replay_parity': True, 'prediction_or_label_changed': False, 'trained_or_deployed': False}
    assert len(details) == summary['metrics']['all']['n']-summary['metrics']['all']['top4_hits']
    assert len(primary) == summary['metrics']['evidence_total']['n']-summary['metrics']['evidence_total']['top4_hits']
    write(out/'summary.json', summary); write(out/'cases-private.json', details)
    def short(target):
        kind = target.get('kind'); owner = target.get('bundleID', '')
        if kind == 'page': return '网页 '+str(urlsplit(target.get('locator', '')).hostname)
        if kind in ('directory', 'file'): return kind+' '+Path(urlsplit(target.get('locator', '')).path).name
        return (target.get('name') or owner or kind or '未知')
    m = summary['metrics']['evidence_total']
    lines = ['# 当前选中模型的 Top4 漏选诊断', '',
             scoring.report_scope(rows),
             f"接受成绩：Top1 {m['top1_hits']}/{m['n']}，Top2 {m['top2_hits']}/{m['n']}，Top4 {m['top4_hits']}/{m['n']}（{m['top4']:.1%}）；以下审计剩余 {len(primary)} 条。", '',
             '| 失败位置 | 条数 |', '|---|---:|']
    for reason, count in summary['misses']['primary']['reasons'].items(): lines.append(f'| {REASONS[reason]} | {count} |')
    lines += ['', '其中 9 条受输出定义限制，4 条是可输出目标的排序漏选。13 个漏选截点对应 12 个不同评分结果事件，不等于 13 次独立操作。', '',
              '四个具体 ChatGPT 会话漏选的服务入口均已排第一，但精确会话仍无法输出。五个同应用标签失败中，三个呈现窗口后跟同应用跨来源观察，一个是同应用不同窗口，一个含离开应用后返回；这些情况不能用同一条规则随意删除。', '',
              '额外发现背景表示未完全对齐：窗口访问以 window_class 记录，但应用输出标签及其 aliases 使用 app，窗口计数没有直接进入该应用的目标计数通道，只可能通过 owner 通道保留部分信号。这是可验证的特征缺口，其修复收益仍需另行重训评测。']
    lines += ['', '## 全部有证据漏选逐条检查', '']
    for r in primary:
        lines += [f"### 截点 {r['position']}：{short(r['current'])} → {short(r['truth'])}", '',
                  f"- {REASONS[r['reason']]}。统计排名 {r['statistical_rank']}，当前排名 {r['model_rank']}（None 表示无法输出）。",
                  f"- 参考口径同起点支持 {r['reference_exact_origin_outcomes']} 个独立结果；实际训练同起点支持 {r['mature_exact_origin_navigation_outcomes']} 个；全局该目标历史 {r['mature_global_truth_outcomes']} 个。",
                  f"- 近期 5/60 分钟具体目标到达计数：{r['time_background']['recent5m']['count']['target']} / {r['time_background']['recent60m']['count']['target']}。实际训练标签与本例评分标签{'相同' if r['current_training_label_diagnostic_only'] == r['truth_action'] else '不同或未保留'}（只用于事后诊断）。",
                  f"- 已有对照模型命中数 {len(r['alternative_models_hit'])}；不能将逐例挑最佳模型当成实际系统成绩。"]
        if r['service']:
            s = r['service']; lines += [f"- 同起点服务支持 {s['same_origin_service_outcomes']} 个，具体目标分布 {s['same_origin_concrete_outcomes']}；当时未恢复具体会话，服务入口排名 {s['service_entry_rank']}。"]
        if r['corrections']:
            c = r['corrections']; lines += [f"- 真目标基础修正 {c['truth_base_delta']:.3f}，情景修正 {c['truth_context_delta']:.3f}；仅关闭已学情景修正时的诊断排名 {c['rank_without_context_correction']}。此为固定权重诊断，不是重训消融。"]
        if r['window_history_not_directly_aligned_to_app_target_feature']:
            lines += ['- 存在该应用的窗口类别背景记录，但没有直接对齐到应用目标的背景计数通道；owner 通道仍保留部分线索。']
        for witness in r['reference_witnesses']:
            if not witness['matching_training_relation']:
                lines += [f"- 参考见证截点 {witness['position']} 未以同一目标/结果进入当前训练；轨迹终止原因 {witness['trajectory_stop_reason']}，保留 {witness['trajectory_retained_count']} 项。"]
        if r['current_training_label_diagnostic_only'] != r['truth_action']:
            lines += [f"- 本截点训练标签与评分标签不一致；轨迹终止原因 {r['current_training_trajectory_stop_diagnostic_only']}。保留原评分，不据此更换成较容易的答案。"]
        lines += ['']
    lines += ['## 全部评测的辅助诊断', '',
              f"全部 {summary['metrics']['all']['n']} 条合法评测中漏选 {len(details)} 条；主结论仍以上述 {m['n']} 条合法有证据集合为准。", '',
              '| 原因 | 全部漏选条数 |', '|---|---:|']
    for reason, count in summary['misses']['all']['reasons'].items(): lines.append(f'| {REASONS[reason]} | {count} |')
    lines += ['', '## 解释边界', '',
              '- 原始目录/URL、证据索引、训练见证、背景统计和预测后观察轨迹仅保存在这个私有 run。预测后的轨迹只用于核对评分标签与观测语义，未作为预测线索。',
              '- 当前应用粗标签已由统一合法性审查排除；服务归并接受命中与精确目标诊断分开。参考证据与实际清洗训练支持仍可能不同。',
              '- 近期频次、过往转移只能说明存在理由，不保证能唯一确定真实下一步。已知但第一次形成训练标签与首次观察目标分开统计。',
              '- 漏选按评测截点计数，不等于独立用户错误次数；部分截点可能共享后续结果。合法性审查不依据预测是否命中。',
              '- 此次只修正评分规则与合法分母，没有重新训练或部署；原预测、统计基线、输入与源码通过哈希和因果回放复核。', '']
    (out/'report-private.md').write_text('\n'.join(lines))
    (out/'analysis-source.py').write_bytes(Path(__file__).read_bytes())
    for path, checks in {alt: alt_hashes, residual_run: residual_hashes, fusion: fusion_hashes, **dependencies}.items(): assert verify(path) == checks
    assert sha(snapshot/'events.jsonl') == raw_hash
    write(out/'run.json', {'complete': True, 'created_at': datetime.now(timezone.utc).isoformat(),
                          'alternatives_run': str(alt), 'residual_run': str(residual_run), 'dataset': str(dataset),
                          'source_hashes': {str(p): h for p, h in {alt: alt_hashes, residual_run: residual_hashes, **dependencies}.items()},
                          'raw_sha256': raw_hash, 'prediction_or_label_changed': False})
    write(out/'artifact-hashes.json', {p.name: sha(p) for p in out.iterdir() if p.is_file()})
    assert all(p.stat().st_mode & 0o777 == 0o600 for p in out.iterdir() if p.is_file())
    print(json.dumps({'output': str(out), **summary}, ensure_ascii=False), flush=True)


if __name__ == '__main__': main()
