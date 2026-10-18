"""Diagnose sample-level upside from local hour and weekday clock contexts.

This is a post-hoc, offline diagnostic.  It evaluates every predeclared clock
configuration against the frozen current-only baseline, but never chooses a
configuration, changes the runtime, or uses outcome metadata as model input.
Only the prediction cutoff's timestamp supplies the clock features.
"""
import argparse
from collections import Counter
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


VERSION = 'clock-context-positive-cases-v1'
SOURCE_NAMES = (
    'analyze_clock_context_positives.py', 'context_statistics.py',
    'evaluate_clock_context.py', 'evaluation_contract.py',
    'compare_semantic_trajectory.py', 'navigation_hierarchy.py',
    'evaluate_context_knn.py', 'review_historical_models.py',
)
POSITIVE_METRICS = (
    'accepted_top1_rescue', 'accepted_top4_rescue',
    'exact_top1_rescue', 'exact_top4_rescue',
    'rank_score_improved', 'first_action_improved',
)
METRIC_LABELS = {
    'accepted_top1_rescue': '接受 Top1 救回',
    'accepted_top4_rescue': '接受 Top4 救回',
    'exact_top1_rescue': '精确 Top1 救回',
    'exact_top4_rescue': '精确 Top4 救回',
    'rank_score_improved': '综合位置分提高',
    'first_action_improved': '首项实际损失降低',
}
WEEKDAYS = ('星期一', '星期二', '星期三', '星期四', '星期五', '星期六', '星期日')
EPSILON = 1e-12


def local_context(timestamp_ms):
    """Return reporting-only local clock strata; this is not outcome metadata."""
    local = datetime.fromtimestamp(float(timestamp_ms) / 1000,
                                   ZoneInfo(statistics.CLOCK_TIME_ZONE))
    start = (local.hour // 3) * 3
    return {'hour3': f'{start:02d}:00–{(start + 3) % 24:02d}:00',
            'weekday': WEEKDAYS[local.weekday()]}


def paired_effect(row, before, after):
    """Score one already-audited row with the common scoring contract."""
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
            effect[f'{level}_top{cutoff}_rescue'] = bool(
                after_hits[level][cutoff] and not before_hits[level][cutoff])
    effect['rank_score_improved'] = rank_delta > EPSILON
    effect['first_action_improved'] = loss_delta > EPSILON
    effect['positive_any'] = any(effect[name] for name in POSITIVE_METRICS)
    return effect


def case_context(row, observation):
    """Keep only coarse, non-identifier outcome strata in diagnostic artifacts."""
    current = observation.get('target') or {}
    truth = row.get('truth_metadata') or {}
    return {
        **local_context(observation['timestamp']),
        'split': row['split'],
        'evidence_grade': row['evidence_grade'],
        'current_kind': current.get('kind', 'unknown'),
        'next_kind': truth.get('kind', 'unknown'),
    }


def summarize_positive_cases(cases):
    """Aggregate upside only; losses are intentionally outside this diagnostic."""
    result = {
        'eligible_n': len(cases),
        'changed_prediction_n': sum(case['changed_prediction'] for case in cases),
        'positive_any_n': sum(case['positive_any'] for case in cases),
        'positive_metrics': {},
    }
    for metric in POSITIVE_METRICS:
        positive = [case for case in cases if case[metric]]
        result['positive_metrics'][metric] = {
            'n': len(positive),
            'positive_rank_score_total': sum(max(0., case['rank_score_delta']) for case in positive),
            'positive_first_action_loss_total': sum(max(0., case['first_action_loss_delta']) for case in positive),
            'positions': [case['position'] for case in positive],
        }
    return result


def positive_time_strata(cases, metric):
    """Report pre-cutoff hour and weekday denominators for each positive subset."""
    result = {}
    for field in ('hour3', 'weekday'):
        total = Counter(case[field] for case in cases)
        positive = Counter(case[field] for case in cases if case[metric])
        result[field] = [
            {'value': value, 'eligible_n': total[value], 'positive_n': positive[value]}
            for value in sorted(positive)
        ]
    return result


def positive_context_patterns(cases, metric):
    """List coarse contexts for observed positive rows without target identities."""
    fields = ('hour3', 'weekday', 'split', 'evidence_grade', 'current_kind', 'next_kind')
    counts = Counter(tuple(case[field] for field in fields) for case in cases if case[metric])
    return [dict(zip(fields, key), n=value) for key, value in sorted(
        counts.items(), key=lambda item: (-item[1], item[0]))]


def configuration_summary(config, cases):
    summary = summarize_positive_cases(cases)
    summary.update({
        'config': config,
        'time_strata': {metric: positive_time_strata(cases, metric) for metric in POSITIVE_METRICS},
        'positive_context_patterns': {metric: positive_context_patterns(cases, metric)
                                      for metric in POSITIVE_METRICS},
    })
    return summary


def grid_positive_union(cases_by_config):
    """De-duplicate samples helped by any predeclared configuration.

    This is descriptive only: it must not be interpreted as an oracle that can
    pick a different configuration after seeing the outcome.
    """
    by_position = {}
    metric_positions = {metric: set() for metric in POSITIVE_METRICS}
    for cases in cases_by_config.values():
        for case in cases:
            existing = by_position.setdefault(case['position'], case)
            for field in ('hour3', 'weekday', 'split', 'evidence_grade', 'current_kind', 'next_kind'):
                if existing[field] != case[field]:
                    raise ValueError('Clock configurations disagree on an input stratum')
            for metric in POSITIVE_METRICS:
                if case[metric]:
                    metric_positions[metric].add(case['position'])
    return {
        'unique_positive_any_n': len(by_position),
        'positive_metric_n': {metric: len(positions) for metric, positions in metric_positions.items()},
        'unique_case_contexts': [by_position[position] for position in sorted(by_position)],
    }


def profile_label(config):
    labels = {
        'clock_hour3': '3 小时时段',
        'clock_weekday': '星期',
        'clock_hour3_weekday': '时段＋星期',
    }
    return labels[config['profile']]


def compact_patterns(summary, metric):
    patterns = summary['positive_context_patterns'][metric]
    if not patterns:
        return '—'
    return '；'.join(
        f"{item['n']} 次：{item['hour3']}／{item['weekday']}／{item['split']}／"
        f"{item['current_kind']}→{item['next_kind']}"
        for item in patterns
    )


def report(summary):
    union = summary['grid_unique_positive']
    lines = [
        '# 时间与星期：逐样本正向作用诊断', '',
        summary['score_scope'], '',
        '这是纯离线、事后诊断：所有 6 个预先定义的本地时段／星期配置分别与同一当前目标统计基线逐项比较。此报告**不**按净增益选择模型，也不允许按真实结果在样本间切换配置。',
        '“正向”只指同一合法评测样本中，时钟候选相对基线出现接受或精确 Top1/Top4 救回、统一综合位置分提高，或首项实际损失降低。损失不参与本报告的筛选或结论。',
        '时钟仅来自预测截点的 `America/Los_Angeles` 本地时间和星期；模型输入不含后台应用、窗口、标签页、库存、近期有序路径或结果标签。', '',
        '## 全部配置去重后的存在性结论', '',
        f"- 在 270 条合法样本中，**{union['unique_positive_any_n']} 条**至少被一个预先定义的时钟配置正向改善。",
        f"- 去重后：接受 Top1 救回 {union['positive_metric_n']['accepted_top1_rescue']} 条；"
        f"接受 Top4 救回 {union['positive_metric_n']['accepted_top4_rescue']} 条；"
        f"精确 Top1 救回 {union['positive_metric_n']['exact_top1_rescue']} 条；"
        f"综合位置分提高 {union['positive_metric_n']['rank_score_improved']} 条。",
        '- 这是“至少有正向作用”的存在性结果，不是可按真实结果选择配置的线上可用策略。', '',
        '## 正向样本计数', '',
        '| 配置 | 合法样本 | 排名发生变化 | 任一正向 | 接受 Top1 救回 | 接受 Top4 救回 | 精确 Top1 救回 | 精确 Top4 救回 | 综合位置分提高 | 首项损失降低 |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|',
    ]
    for item in summary['configurations']:
        config = item['config']; metrics = item['positive_metrics']
        lines.append(
            f"| {profile_label(config)}（floor={config['floor']:g}） | {item['eligible_n']} | "
            f"{item['changed_prediction_n']} | {item['positive_any_n']} | "
            + ' | '.join(str(metrics[name]['n']) for name in POSITIVE_METRICS) + ' |')

    lines += ['', '## 正向样本的时间／星期位置', '']
    found = False
    for item in summary['configurations']:
        config = item['config']
        for metric in POSITIVE_METRICS:
            n = item['positive_metrics'][metric]['n']
            if not n:
                continue
            found = True
            lines += [
                f"- `{config['id']}` 的{METRIC_LABELS[metric]}：{n} 条；"
                f"{compact_patterns(item, metric)}。",
            ]
    if not found:
        lines.append('- 所有预定义时钟配置均未在任何合法样本上产生本报告定义的正向变化。')

    lines += ['', '## 解读边界', '',
        '- 同一时段／星期桶内只有重复正向样本，才可能成为后续可检验的情景假设；单个事后救回案例只证明“发生过”，不证明稳定的局部规律。',
        '- 本轮不改模型选择、采集、训练、推荐或运行时行为；如需继续验证，应在未来自然产生的新样本上预先固定一个候选与一个时段条件。',
        '- 统一评分、合法性审查、层级匹配与输出上限均保持不变；基线逐截点预测和已保存的最佳时钟候选逐截点预测都已复现。', '',
    ]
    return '\n'.join(lines)


def assert_clock_only_inputs(inputs):
    forbidden = {'backgrounds', 'inventory_state', 'inventory_quality'}
    expected = {'position', 'index', 'timestamp', 'origin', 'app', 'annotation', 'target',
                'current_sequence', 'background_through_index', 'inventory_through_index'}
    for q in inputs:
        if set(q) != expected:
            raise ValueError('Clock diagnostic input allowlist changed')
        if forbidden & set(q):
            raise ValueError('Clock diagnostic materialized forbidden context data')


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
    if Path(baseline_meta['dataset']).resolve() != dataset:
        raise ValueError('Baseline and requested dataset differ')
    clock_meta = sources.read(clock_run/'run.json')
    clock_plan = sources.read(clock_run/'plan.json')
    clock_selection = sources.read(clock_run/'selection.json')
    clock_predictions = sources.read(clock_run/'predictions-private.json')
    if Path(clock_meta['dataset']).resolve() != dataset:
        raise ValueError('Clock run and requested dataset differ')
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
    if len(expected_baseline) != len(data['evaluation']):
        raise ValueError('Frozen baseline does not cover every evaluation cutoff')
    configs = configurations(base)
    if configs != clock_plan['clock_configurations']:
        raise ValueError('Predeclared clock grid differs from the frozen clock run')

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
        mismatches = [position for position in expected_baseline
                      if baseline_outputs.get(position) != expected_baseline[position]]
        raise ValueError('Current-only baseline parity failed: '+str(mismatches[:10]))

    plan = {
        'version': VERSION,
        'purpose': 'post-hoc sample-level positive-effect diagnostic for local-hour and weekday contexts',
        'base_config': base,
        'clock_configurations': configs,
        'time_zone': statistics.CLOCK_TIME_ZONE,
        'input_allowlist': ['position', 'index', 'timestamp', 'origin', 'app', 'annotation', 'target'],
        'current_sequence_length': 1,
        'ordered_history_features': False,
        'background_or_inventory_features': False,
        'selection_changed': False,
        'losses_used_for_filtering_or_conclusion': False,
        'outcome_metadata_used_only_for_scoring_and_diagnostic_reporting': True,
        'positive_effect_definition': list(POSITIVE_METRICS),
        'scoring_contract': scoring.contract(),
        'dataset_sources': dict(sources.accessed),
        'events_sha256': event_hash,
        'clock_run': str(clock_run),
        'clock_run_config': clock_selection['best_clock']['config'],
        'source_hashes': {name: hashlib.sha256(value).hexdigest() for name, value in source_bytes.items()},
    }
    write(out/'plan.json', plan)

    results, cases_by_config, probes = [], {}, {}
    probe_rows = [next(row for row in rows if row['split'] == split)
                  for split in ('validation', 'test', 'new_test')]
    for config in configs:
        outputs = predictions(all_queries, config)
        config_id = config['id']
        if config_id == clock_selection['best_clock']['config']['id']:
            saved = {int(position): value for position, value in
                     clock_predictions.get('clock/'+config_id, {}).items()}
            if saved != outputs:
                raise ValueError('Saved best-clock predictions do not replay')
        cases = []
        for row in rows:
            if not scoring.eligible(row):
                continue
            before = scoring.prediction_at(baseline_outputs, row['position'])
            after = scoring.prediction_at(outputs, row['position'])
            cases.append({'position': row['position'], **case_context(row, observations[row['position']]),
                          **paired_effect(row, before, after)})
        results.append(configuration_summary(config, cases))
        cases_by_config[config_id] = [case for case in cases if case['positive_any']]
        probes[config_id] = {row['position']: outputs[row['position']] for row in probe_rows}

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
            position = row['position']
            if restored_queries[position].predict(config) != probes[config['id']][position]:
                raise ValueError('Saved clock diagnostic model replay differs')

    summary = {
        'version': VERSION,
        'score_scope': scoring.report_scope(rows),
        'baseline_config': base,
        'configurations': results,
        'grid_unique_positive': grid_positive_union(cases_by_config),
        'baseline_prediction_parity': True,
        'saved_best_clock_prediction_parity': True,
        'model_reload_parity': True,
        'collection_or_runtime_changed': False,
    }
    write(out/'summary.json', summary)
    write(out/'positive-cases-private.json', {
        'version': VERSION,
        'identifiers_omitted': 'only cutoff position and coarse time/type strata are retained',
        'cases_by_config': cases_by_config,
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
        'saved_best_clock_prediction_parity': True,
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
        'positive_any': {item['config']['id']: item['positive_any_n'] for item in results},
        'accepted_top4_rescues': {item['config']['id']: item['positive_metrics']['accepted_top4_rescue']['n']
                                  for item in results},
    }, ensure_ascii=False))


if __name__ == '__main__':
    main()
