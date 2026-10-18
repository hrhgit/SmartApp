"""Diagnose where weekday-only clock reweighting helps or harms in replay.

This evaluates only the two predeclared ``clock_weekday`` configurations
against the frozen current-target statistical baseline.  It is an offline,
post-hoc diagnostic: outcome metadata is used only after prediction to score
effects and is never passed to the predictor.
"""
import argparse
from collections import Counter, defaultdict
from copy import deepcopy
from datetime import datetime, timezone
import gzip
import hashlib
import json
import os
from pathlib import Path
from zoneinfo import ZoneInfo

import context_statistics as statistics
import evaluation_contract as scoring
from evaluate_clock_context import (PROFILES, asof_homes, build_training,
                                    configurations, foreground_inputs, predictions)
from review_historical_models import ROOT, Sources, prediction_file, sha, write


VERSION = 'weekday-context-effects-v1'
SOURCE_NAMES = (
    'analyze_weekday_context_effects.py', 'context_statistics.py',
    'evaluate_clock_context.py', 'evaluation_contract.py',
    'compare_semantic_trajectory.py', 'navigation_hierarchy.py',
    'evaluate_context_knn.py', 'review_historical_models.py',
)
HELP_METRICS = (
    'accepted_top1_rescue', 'accepted_top4_rescue',
    'exact_top1_rescue', 'exact_top4_rescue',
    'rank_score_improved', 'first_action_improved',
)
HARM_METRICS = (
    'accepted_top1_lost', 'accepted_top4_lost',
    'exact_top1_lost', 'exact_top4_lost',
    'rank_score_decreased', 'first_action_worsened',
)
METRIC_LABELS = {
    'accepted_top1_rescue': '接受 Top1 救回',
    'accepted_top4_rescue': '接受 Top4 救回',
    'exact_top1_rescue': '精确 Top1 救回',
    'exact_top4_rescue': '精确 Top4 救回',
    'rank_score_improved': '综合位置分提高',
    'first_action_improved': '首项实际损失降低',
    'accepted_top1_lost': '接受 Top1 丢失',
    'accepted_top4_lost': '接受 Top4 丢失',
    'exact_top1_lost': '精确 Top1 丢失',
    'exact_top4_lost': '精确 Top4 丢失',
    'rank_score_decreased': '综合位置分降低',
    'first_action_worsened': '首项实际损失升高',
}
WEEKDAYS = ('星期一', '星期二', '星期三', '星期四', '星期五', '星期六', '星期日')
EPSILON = 1e-12


def local_context(timestamp_ms):
    """Reporting-only strata from the prediction-cutoff timestamp."""
    local = datetime.fromtimestamp(float(timestamp_ms) / 1000,
                                   ZoneInfo(statistics.CLOCK_TIME_ZONE))
    start = (local.hour // 3) * 3
    return {'hour3': f'{start:02d}:00–{(start + 3) % 24:02d}:00',
            'weekday': WEEKDAYS[local.weekday()]}


def directional_effect(row, before, after):
    """Compare two predictions through the one shared scoring contract."""
    before_hits, after_hits = scoring.hits(row, before), scoring.hits(row, after)
    before_rank, after_rank = scoring.ranked_gain(row, before), scoring.ranked_gain(row, after)
    before_loss = scoring.observed_decision_loss(row, before)
    after_loss = scoring.observed_decision_loss(row, after)
    rank_delta = after_rank['score'] - before_rank['score']
    loss_delta = before_loss - after_loss
    effect = {
        'changed_prediction': before['predictions'] != after['predictions'],
        'rank_score_delta': rank_delta,
        'first_action_loss_delta': loss_delta,
    }
    for level in ('accepted', 'exact'):
        for cutoff in (1, 4):
            before_hit, after_hit = before_hits[level][cutoff], after_hits[level][cutoff]
            effect[f'{level}_top{cutoff}_rescue'] = bool(after_hit and not before_hit)
            effect[f'{level}_top{cutoff}_lost'] = bool(before_hit and not after_hit)
    effect['rank_score_improved'] = rank_delta > EPSILON
    effect['rank_score_decreased'] = rank_delta < -EPSILON
    effect['first_action_improved'] = loss_delta > EPSILON
    effect['first_action_worsened'] = loss_delta < -EPSILON
    effect['helped_any'] = any(effect[name] for name in HELP_METRICS)
    effect['harmed_any'] = any(effect[name] for name in HARM_METRICS)
    return effect


def case_context(row, observation):
    """Coarse reporting context only; target identities remain out of the report."""
    current = observation.get('target') or {}
    truth = row.get('truth_metadata') or {}
    return {
        **local_context(observation['timestamp']),
        'split': row['split'],
        'evidence_grade': row['evidence_grade'],
        'current_kind': current.get('kind', 'unknown'),
        'next_kind': truth.get('kind', 'unknown'),
    }


def summarize_direction(cases, fields, any_field):
    """Count one direction without cancelling it against the opposite direction."""
    selected = [case for case in cases if case[any_field]]
    return {
        'any_n': len(selected),
        'metrics': {field: sum(case[field] for case in cases) for field in fields},
        'positions': [case['position'] for case in selected],
    }


def directional_contexts(cases, fields, any_field):
    """Group observed effects by pre-cutoff clock and coarse target type."""
    keys = ('hour3', 'weekday', 'split', 'evidence_grade', 'current_kind', 'next_kind')
    groups = defaultdict(lambda: {'n': 0, 'metrics': Counter()})
    for case in cases:
        if not case[any_field]:
            continue
        key = tuple(case[name] for name in keys)
        groups[key]['n'] += 1
        groups[key]['metrics'].update(name for name in fields if case[name])
    return [dict(zip(keys, key), n=value['n'], metrics=dict(sorted(value['metrics'].items())))
            for key, value in sorted(groups.items(), key=lambda item: (-item[1]['n'], item[0]))]


def weekday_denominators(cases, any_field):
    """Keep each observed-effect count alongside its fixed eligible denominator."""
    total = Counter(case['weekday'] for case in cases)
    affected = Counter(case['weekday'] for case in cases if case[any_field])
    return [{'weekday': weekday, 'affected_n': affected[weekday], 'eligible_n': total[weekday]}
            for weekday in WEEKDAYS if affected[weekday]]


def summarize_configuration(config, cases):
    return {
        'config': config,
        'eligible_n': len(cases),
        'changed_prediction_n': sum(case['changed_prediction'] for case in cases),
        'helped': summarize_direction(cases, HELP_METRICS, 'helped_any'),
        'harmed': summarize_direction(cases, HARM_METRICS, 'harmed_any'),
        'both_directions_n': sum(case['helped_any'] and case['harmed_any'] for case in cases),
        'helped_weekdays': weekday_denominators(cases, 'helped_any'),
        'harmed_weekdays': weekday_denominators(cases, 'harmed_any'),
        'helped_contexts': directional_contexts(cases, HELP_METRICS, 'helped_any'),
        'harmed_contexts': directional_contexts(cases, HARM_METRICS, 'harmed_any'),
    }


def metrics_text(metrics):
    return '、'.join(METRIC_LABELS[name] for name in metrics) or '—'


def contexts_text(contexts):
    if not contexts:
        return '—'
    return '；'.join(
        f"{item['n']} 条：{item['weekday']}／{item['hour3']}／{item['split']}／"
        f"{item['current_kind']}→{item['next_kind']}（{metrics_text(item['metrics'])}）"
        for item in contexts
    )


def weekday_text(values):
    return '；'.join(f"{item['weekday']} {item['affected_n']}/{item['eligible_n']}" for item in values) or '—'


def report(summary):
    lines = [
        '# 星期上下文：逐样本正反方向诊断', '',
        summary['score_scope'], '',
        '范围只包含纯 `clock_weekday` 特征的两档预先定义重加权；不包含 3 小时时段或时段＋星期组合。每条结论是同一合法样本相对当前目标统计基线的配对变化，而不是星期的因果证明。',
        '时钟输入只取预测截点的 `America/Los_Angeles` 本地星期；后续目标类型仅在预测后用于统一评分和这份诊断报告。模型输入不含后台状态、库存、近期有序路径或结果标签。', '',
        '## 总览', '',
        '| 配置 | 合法样本 | 排名变化 | 任一正向 | 任一负向 | 两者同时出现 | 接受 Top1 救回/丢失 | 接受 Top4 救回/丢失 | 精确 Top1 救回/丢失 | 精确 Top4 救回/丢失 | 位置分提高/降低 | 首项损失降低/升高 |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|',
    ]
    pairs = (
        ('accepted_top1_rescue', 'accepted_top1_lost'),
        ('accepted_top4_rescue', 'accepted_top4_lost'),
        ('exact_top1_rescue', 'exact_top1_lost'),
        ('exact_top4_rescue', 'exact_top4_lost'),
        ('rank_score_improved', 'rank_score_decreased'),
        ('first_action_improved', 'first_action_worsened'),
    )
    for item in summary['configurations']:
        help_metrics, harm_metrics = item['helped']['metrics'], item['harmed']['metrics']
        cells = [f"{help_metrics[good]}/{harm_metrics[bad]}" for good, bad in pairs]
        lines.append(
            f"| 星期（floor={item['config']['floor']:g}） | {item['eligible_n']} | "
            f"{item['changed_prediction_n']} | {item['helped']['any_n']} | {item['harmed']['any_n']} | "
            f"{item['both_directions_n']} | " + ' | '.join(cells) + ' |')
    lines += ['', '## 哪些情景观察到正向', '']
    for item in summary['configurations']:
        lines += [
            f"- `clock_weekday`（floor={item['config']['floor']:g}）："
            f"按星期为 {weekday_text(item['helped_weekdays'])}；{contexts_text(item['helped_contexts'])}。",
        ]
    lines += ['', '## 哪些情景观察到负向', '']
    for item in summary['configurations']:
        lines += [
            f"- `clock_weekday`（floor={item['config']['floor']:g}）："
            f"按星期为 {weekday_text(item['harmed_weekdays'])}；{contexts_text(item['harmed_contexts'])}。",
        ]
    lines += ['', '## 解读边界', '',
        '- 同一星期在本数据中与日期、任务和作息共线，且各桶很小；“有效／有害”仅描述这次冻结回放中的观察结果，不构成稳定的因果规则。',
        '- 如果要让星期特征进入后续研究，应预先固定一个 floor 和可识别的情景条件，再只在新自然样本上检验；不能按真实结果对每个样本挑选配置。',
        '- 本轮不改变模型选择、采集、训练、推荐或运行时行为；统一评分、样本合法性审查、层级匹配与输出上限均未变。', '',
    ]
    return '\n'.join(lines)


def assert_clock_only_inputs(inputs):
    forbidden = {'backgrounds', 'inventory_state', 'inventory_quality'}
    expected = {'position', 'index', 'timestamp', 'origin', 'app', 'annotation', 'target',
                'current_sequence', 'background_through_index', 'inventory_through_index'}
    for q in inputs:
        if set(q) != expected or forbidden & set(q):
            raise ValueError('Weekday diagnostic input crossed the clock-only boundary')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--baseline-run', type=Path, required=True)
    parser.add_argument('--clock-run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    dataset = args.dataset.expanduser().resolve()
    baseline_run = args.baseline_run.expanduser().resolve()
    clock_run = args.clock_run.expanduser().resolve()
    out = args.output.expanduser().resolve()
    root = ROOT.resolve()
    if out.exists() or out == root or not out.is_relative_to(root):
        parser.error('Use a new private SmartApp training directory')

    sources = Sources()
    data = sources.read(dataset/'dataset-private.json.gz')
    dataset_run = sources.read(dataset/'run.json')
    baseline_meta = sources.read(baseline_run/'run.json')
    clock_meta = sources.read(clock_run/'run.json')
    clock_plan = sources.read(clock_run/'plan.json')
    if Path(baseline_meta['dataset']).resolve() != dataset or Path(clock_meta['dataset']).resolve() != dataset:
        raise ValueError('Requested runs do not use the frozen dataset')
    snapshot = Path(dataset_run['snapshot']).resolve()
    events = sources.check(snapshot/'events.jsonl')
    event_hash = sha(events)
    if event_hash != dataset_run['snapshot_hashes']['events.jsonl']:
        raise ValueError('Raw event snapshot differs from the frozen dataset')

    source_bytes = {name: Path(__file__).with_name(name).read_bytes() for name in SOURCE_NAMES}
    for name in SOURCE_NAMES[1:]:
        expected = clock_plan['source_hashes'].get(name)
        if expected and hashlib.sha256(source_bytes[name]).hexdigest() != expected:
            raise ValueError('Clock source no longer matches the frozen clock run: '+name)

    models = sources.read(baseline_run/'models.json')
    base = deepcopy(models['statistics/none']['config'])
    expected_baseline = {int(position): value for position, value in
                         sources.read(prediction_file(baseline_run))['statistics/none'].items()}
    configs = [config for config in configurations(base) if config['profile'] == 'clock_weekday']
    if configs != [config for config in clock_plan['clock_configurations']
                   if config['profile'] == 'clock_weekday'] or len(configs) != 2:
        raise ValueError('Weekday-only grid differs from the frozen clock run')

    inputs = foreground_inputs(data)
    assert_clock_only_inputs(inputs)
    homes, asof_index = asof_homes(snapshot, data['observations'])
    training = build_training(data, inputs)
    os.umask(0o077)
    out.mkdir(mode=0o700, parents=True)
    rows = scoring.prepare_evaluation(data, out)
    observations = {q['position']: q for q in data['observations']}
    background = statistics.BackgroundIndex(inputs, profiles=PROFILES)
    all_queries = statistics.prepare_queries(inputs, training, rows, homes, background)
    baseline_outputs = predictions(all_queries, base)
    if baseline_outputs != expected_baseline:
        mismatch = [position for position in expected_baseline
                    if baseline_outputs.get(position) != expected_baseline[position]]
        raise ValueError('Current-only baseline parity failed: '+str(mismatch[:10]))

    plan = {
        'version': VERSION,
        'purpose': 'post-hoc weekday-only paired help-and-harm diagnostic',
        'base_config': base,
        'weekday_configurations': configs,
        'time_zone': statistics.CLOCK_TIME_ZONE,
        'input_allowlist': ['position', 'index', 'timestamp', 'origin', 'app', 'annotation', 'target'],
        'current_sequence_length': 1,
        'ordered_history_features': False,
        'background_or_inventory_features': False,
        'selection_changed': False,
        'outcome_metadata_used_only_for_scoring_and_diagnostic_reporting': True,
        'scoring_contract': scoring.contract(),
        'dataset_sources': dict(sources.accessed),
        'events_sha256': event_hash,
        'clock_run': str(clock_run),
        'source_hashes': {name: hashlib.sha256(value).hexdigest() for name, value in source_bytes.items()},
    }
    write(out/'plan.json', plan)

    results, diagnostic_cases, probes = [], {}, {}
    probe_rows = [next(row for row in rows if row['split'] == split)
                  for split in ('validation', 'test', 'new_test')]
    for config in configs:
        outputs = predictions(all_queries, config)
        cases = []
        for row in rows:
            if not scoring.eligible(row):
                continue
            before = scoring.prediction_at(baseline_outputs, row['position'])
            after = scoring.prediction_at(outputs, row['position'])
            cases.append({'position': row['position'], **case_context(row, observations[row['position']]),
                          **directional_effect(row, before, after)})
        results.append(summarize_configuration(config, cases))
        diagnostic_cases[config['id']] = [case for case in cases if case['helped_any'] or case['harmed_any']]
        probes[config['id']] = {row['position']: outputs[row['position']] for row in probe_rows}

    memory = statistics.StatisticalMemory()
    for item in training:
        if item['ready_index'] > asof_index:
            raise ValueError('Training record exceeds final input cutoff')
        memory.learn(inputs[item['position']], item['record'], item['ready_index'])
    memory_path = out/'model-memory-private.json.gz'
    with gzip.open(memory_path, 'wt') as target:
        json.dump({'inputs': inputs, 'training': memory.training, 'asof_index': asof_index,
                   'records': len(memory.training), 'navigation_records': len(memory.records)}, target,
                  ensure_ascii=False)
    memory_path.chmod(0o600)
    with gzip.open(memory_path, 'rt') as source:
        restored = json.load(source)
    restored_background = statistics.BackgroundIndex(restored['inputs'], profiles=PROFILES)
    restored_queries = statistics.prepare_queries(restored['inputs'], restored['training'], probe_rows,
                                                  homes, restored_background)
    for config in configs:
        for row in probe_rows:
            if restored_queries[row['position']].predict(config) != probes[config['id']][row['position']]:
                raise ValueError('Saved weekday diagnostic model replay differs')

    summary = {
        'version': VERSION,
        'score_scope': scoring.report_scope(rows),
        'baseline_config': base,
        'configurations': results,
        'baseline_prediction_parity': True,
        'model_reload_parity': True,
        'collection_or_runtime_changed': False,
    }
    write(out/'summary.json', summary)
    write(out/'weekday-effects-private.json', {
        'version': VERSION,
        'identifiers_omitted': 'only cutoff position and coarse time/type strata are retained',
        'cases_by_config': diagnostic_cases,
    })
    report_path = out/'report-private.md'
    report_path.write_text(report(summary), encoding='utf-8')
    report_path.chmod(0o600)
    for name, value in source_bytes.items():
        path = out/('source-'+name)
        path.write_bytes(value)
        path.chmod(0o600)

    sources.verify_unchanged()
    if sha(events) != event_hash:
        raise ValueError('Raw event snapshot changed during replay')
    for name, value in source_bytes.items():
        if Path(__file__).with_name(name).read_bytes() != value:
            raise ValueError('Research source changed during replay: '+name)
    write(out/'run.json', {
        'complete': True,
        'version': VERSION,
        'created_at': datetime.now(timezone.utc).isoformat(),
        'dataset': str(dataset),
        'baseline_run': str(baseline_run),
        'clock_run': str(clock_run),
        'snapshot': str(snapshot),
        'events_sha256': event_hash,
        'input_sources': dict(sources.accessed),
        'source_hashes_verified_before_after': True,
        'baseline_prediction_parity': True,
        'model_reload_parity': True,
        'collection_or_runtime_changed': False,
        'raw_modified': False,
    })
    write(out/'artifact-hashes.json', {path.name: sha(path) for path in sorted(out.iterdir()) if path.is_file()})
    if any(path.stat().st_mode & 0o777 != 0o600 for path in out.iterdir() if path.is_file()):
        raise ValueError('Private artifact permissions changed')
    print(json.dumps({
        'complete': True,
        'output': str(out),
        'configurations': [{
            'id': item['config']['id'], 'helped_any': item['helped']['any_n'],
            'harmed_any': item['harmed']['any_n'], 'both_directions': item['both_directions_n'],
        } for item in results],
    }, ensure_ascii=False))


if __name__ == '__main__':
    main()
