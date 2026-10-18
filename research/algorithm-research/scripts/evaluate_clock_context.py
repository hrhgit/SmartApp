"""Evaluate local hour and weekday contexts without reading background state.

The model keeps the existing current-target conditional statistics intact and
only redistributes mature votes according to the prediction cutoff's local
clock.  Configurations are selected on the fixed early-validation segment;
later and appended rows are evaluated only after the selection is written.
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

import context_statistics as statistics
import evaluation_contract as scoring
from compare_semantic_trajectory import training_rows
from evaluate_context_knn import admitted, synthetic
from navigation_hierarchy import ENTRY, learning_record
from review_historical_models import ROOT, Sources, sha, write


VERSION = 'clock-context-v1'
PROFILES = ('clock_hour3', 'clock_weekday', 'clock_hour3_weekday')
FLOORS = (.75, .5)
SOURCE_NAMES = (
    'context_statistics.py', 'evaluate_clock_context.py', 'evaluation_contract.py',
    'compare_semantic_trajectory.py', 'navigation_hierarchy.py',
    'evaluate_context_knn.py', 'review_historical_models.py',
)


def configurations(base):
    """A small predeclared grid: hour, weekday, and their equal-weight union."""
    if base.get('mode') != 'none' or base.get('profile') is not None:
        raise ValueError('Clock comparison requires a current-only statistical baseline')
    result = []
    for profile in PROFILES:
        for floor in FLOORS:
            result.append({**deepcopy(base), 'id': f'clock_{profile}_floor{floor:g}',
                           'mode': 'clock', 'profile': profile, 'floor': floor, 'power': 1.,
                           'clock_time_zone': statistics.CLOCK_TIME_ZONE})
    return result


def selection_key(metrics):
    """Use the established early-validation ordering and no later result."""
    return tuple(metrics[group][name] for group, name in (
        ('original', 'top4_hits'), ('expanded', 'top4_hits'),
        ('original', 'top1_hits'), ('expanded', 'top1_hits'),
        ('all', 'top4_hits'), ('all', 'top1_hits'),
    ))


def select(grid):
    if not grid:
        raise ValueError('No configurations to select')
    # max keeps the first predeclared configuration on a complete tie.
    return max(grid, key=lambda row: selection_key(row['validation']))


def foreground_inputs(data):
    """Build the permitted length-one inputs without materializing backgrounds."""
    fields = ('position', 'index', 'timestamp', 'origin', 'app', 'annotation', 'target')
    result = []
    for position, observation in enumerate(data['observations']):
        if observation['position'] != position:
            raise ValueError('Observation positions are not contiguous')
        q = {name: deepcopy(observation[name]) for name in fields}
        q.update(current_sequence=[q['origin']], background_through_index=q['index'],
                 inventory_through_index=q['index'])
        result.append(q)
    return result


def asof_homes(snapshot, observations):
    """Resolve only an already observed ChatGPT entry target at each cutoff."""
    by_index = {q['index']: q for q in observations}
    catalog, homes = {}, {}
    with (snapshot/'events.jsonl').open() as source:
        for index, line in enumerate(source):
            event = json.loads(line)
            if event['kind'] == 'target.update':
                target = event['data']['target']
                if admitted(target) and not synthetic(target):
                    catalog[target['id']] = target
                else:
                    catalog.pop(target['id'], None)
            if index not in by_index:
                continue
            observation = by_index[index]
            if observation['anchor_id'] != event['id'] or observation['session'] != event['sessionID']:
                raise ValueError('Frozen prediction input anchor changed')
            homes[observation['position']] = next((key for key, target in sorted(catalog.items())
                if target['kind'] == 'page' and target.get('locator') == ENTRY), None)
    if len(homes) != len(observations):
        raise ValueError('Missing as-of homepage resolution')
    return homes, index


def build_training(data, inputs):
    stream = [row for row in training_rows(data, 'hard_kind') if row['head'] == 0]
    result = []
    for row in stream:
        q = inputs[row['position']]
        if q['index'] > row['ready_index'] or row['record']['target_evidence_index'] > row['ready_index']:
            raise ValueError('Training evidence is unavailable at maturity')
        result.append({'position': row['position'], 'ready_index': row['ready_index'],
                       'record': learning_record(row, q)})
    return result


def predictions(queries, config):
    return {position: query.predict(config) for position, query in queries.items()}


def cell(metric, field):
    return f"{metric[field+'_hits']}/{metric['n']} ({metric[field]:.1%})" if metric['n'] else '—'


def clock_coverage(rows, observations):
    zone = statistics._CLOCK_ZONE
    result = {}
    for split in ('validation', 'test', 'new_test'):
        values = [observations[row['position']] for row in rows if row['split'] == split]
        weekdays = Counter()
        hour_bins = Counter()
        for q in values:
            local = datetime.fromtimestamp(q['timestamp'] / 1000, zone)
            weekdays[str(local.weekday())] += 1
            hour_bins[str((local.hour // 3) * 3)] += 1
        result[split] = {'n': len(values), 'weekday_counts': dict(sorted(weekdays.items())),
                         'three_hour_bin_counts': dict(sorted(hour_bins.items(), key=lambda item: int(item[0])))}
    return result


def report(summary):
    baseline, clock = summary['baseline'], summary['best_clock']
    selection = summary['selection']
    def row(scope, model, result, cohort):
        metric = result['cohorts'][cohort]
        return (f"| {scope} | {model} | {metric['n']} | {cell(metric, 'top1')} | {cell(metric, 'top4')} | "
                f"{cell(metric['exact'], 'top1')} | {cell(metric['exact'], 'top4')} | "
                f"{metric['rank_score']['mean'] * 100:.2f} |")
    lines = [
        '# 本地时间与星期上下文：离线对照', '',
        summary['score_scope'], '',
        '新增特征只取预测截点的本地钟点与星期（固定 `America/Los_Angeles`）；不读取后台应用、窗口、标签页、库存、近期有序路径或结果标签。历史训练仍只包含此前成熟的前台到达。',
        f"固定的当前目标统计基线为 `{baseline['config']['id']}`；时钟候选中最优的是 `{clock['config']['id']}`。早期验证的联合选择结果为 `{selection['config']['id']}`（{'采用时钟特征' if selection['uses_clock'] else '保留基线'}）。", '',
        '## 主结果', '',
        '| 范围 | 模型 | 样本 | 接受 Top1 | 接受 Top4 | 精确 Top1 | 精确 Top4 | 综合分/100 |',
        '|---|---|---:|---:|---:|---:|---:|---:|',
    ]
    for cohort, label in (
        ('evidence_total', '有证据合计'), ('old_all', '既有固定段全部'),
        ('new_all', '新增后段全部'), ('all', '完整固定验证'),
    ):
        lines.append(row(label, '仅当前统计基线', baseline, cohort))
        lines.append(row(label, '最佳时钟候选', clock, cohort))
    pair = summary['pairs']['all']
    lines += [
        '', '## 配对变化（时钟模型相对基线）', '',
        f"- 完整固定验证：Top1 修正 {pair['top1']['helped']}、损失 {pair['top1']['hurt']}，Top4 修正 {pair['top4']['helped']}、损失 {pair['top4']['hurt']}。",
        f"- 新增后段：Top1 净变化 {summary['pairs']['new_all']['top1']['net_hits']:+d}，Top4 净变化 {summary['pairs']['new_all']['top4']['net_hits']:+d}。",
        '', '## 选择与覆盖', '',
        '- 只用早期 `validation` 段按原始有证据 Top4、扩展有证据 Top4、Top1、全部样本顺序选择；时钟候选和无时钟基线一起参与选择，选择文件在计算其他时段指标前写入。',
        '- 三种上下文为：平滑的 3 小时本地钟点、精确星期、以及两者的等权相似度。每种各比较温和（floor=0.75）和较强（floor=0.5）重加权；其余统计参数固定。',
        f"- 评测时段的星期／3小时桶覆盖已保存到 summary：`{json.dumps(summary['clock_coverage'], ensure_ascii=False)}`。",
        '- 数据仅跨很短的自然行为窗口，星期与日期、作息和任务会共线；即使本轮有增益也只是继续收集新数据的候选，不构成部署证据。',
        '- 统一评分、成员审查、层级匹配与输出上限均未改动；基线逐截点重放与冻结预测完全一致。',
        '',
    ]
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--baseline-run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    out, dataset, baseline_run = (args.output.expanduser().resolve(), args.dataset.expanduser().resolve(),
                                  args.baseline_run.expanduser().resolve())
    root = ROOT.resolve()
    if out.exists() or out == root or not out.is_relative_to(root):
        parser.error('Use a new private SmartApp training directory')

    sources = Sources()
    data = sources.read(dataset/'dataset-private.json.gz')
    dataset_run = sources.read(dataset/'run.json')
    baseline_meta = sources.read(baseline_run/'run.json')
    if Path(baseline_meta['dataset']).resolve() != dataset:
        raise ValueError('Baseline and requested dataset differ')
    models = sources.read(baseline_run/'models.json')
    base = deepcopy(models['statistics/none']['config'])
    expected_all = sources.read(baseline_run/'predictions-private.json.gz')['statistics/none']
    expected = {int(position): value for position, value in expected_all.items()}
    if len(expected) != len(data['evaluation']):
        raise ValueError('Frozen baseline does not cover every evaluation cutoff')
    snapshot = Path(dataset_run['snapshot']).resolve()
    event_hash = sha(snapshot/'events.jsonl')
    if event_hash != dataset_run['snapshot_hashes']['events.jsonl']:
        raise ValueError('Raw event snapshot differs from the frozen dataset')

    source_bytes = {name: Path(__file__).with_name(name).read_bytes() for name in SOURCE_NAMES}
    inputs = foreground_inputs(data)
    homes, asof_index = asof_homes(snapshot, data['observations'])
    training = build_training(data, inputs)
    os.umask(0o077)
    out.mkdir(mode=0o700, parents=True)
    rows = scoring.prepare_evaluation(data, out)
    parts = scoring.cohorts(rows)
    test_index = min(inputs[row['position']]['index'] for row in rows if row['split'] == 'test')
    validation = [row for row in rows if row['split'] == 'validation' and
                  data['observations'][row['position']]['ready_index'] < test_index]
    if not validation or not any(row['evidence_grade'] == 'original' for row in validation):
        raise ValueError('No eligible early validation evidence')
    configs = configurations(base)
    plan = {
        'version': VERSION,
        'purpose': 'offline local-hour and weekday conditional-statistics ablation',
        'baseline_config': base,
        'clock_configurations': configs,
        'time_zone': statistics.CLOCK_TIME_ZONE,
        'clock_features': 'prediction-cutoff timestamp only; no background or inventory context',
        'current_sequence_length': 1,
        'input_allowlist': ['position', 'index', 'timestamp', 'origin', 'app', 'annotation', 'target'],
        'ordered_history_features': False,
        'background_or_inventory_features': False,
        'training_cleaning': 'hard_kind head0 mature navigation records',
        'selection': 'baseline plus clock candidates: validation original Top4, expanded Top4, original Top1, expanded Top1, all Top4, all Top1; first configuration on tie',
        'validation_positions': [row['position'] for row in validation],
        'late_or_new_scores_used_for_selection': False,
        'cohort_positions': {name: [row['position'] for row in values] for name, values in parts.items()},
        'scoring_contract': scoring.contract(),
        'dataset_sources': dict(sources.accessed),
        'events_sha256': event_hash,
        'source_hashes': {name: hashlib.sha256(value).hexdigest() for name, value in source_bytes.items()},
    }
    write(out/'plan.json', plan)

    # Only clock profiles are indexed: this path never accesses historical
    # backgrounds or inventories, even though older datasets may contain them.
    background = statistics.BackgroundIndex(inputs, profiles=PROFILES)
    all_queries = statistics.prepare_queries(inputs, training, rows, homes, background)
    baseline_outputs = predictions(all_queries, base)
    if baseline_outputs != expected:
        mismatches = [position for position in expected if baseline_outputs.get(position) != expected[position]]
        raise ValueError('Current-only baseline parity failed: '+str(mismatches[:10]))
    validation_queries = statistics.prepare_queries(inputs, training, validation, homes, background)
    baseline_validation = scoring.grouped_metrics(validation, baseline_outputs)
    grid = []
    for config in configs:
        output = predictions(validation_queries, config)
        grid.append({'config': config, 'validation': scoring.grouped_metrics(validation, output)})
    best_clock = deepcopy(select(grid))
    overall = deepcopy(select([{'config': base, 'validation': baseline_validation, 'kind': 'baseline'}] + grid))
    uses_clock = overall['config']['id'] != base['id']
    write(out/'validation-grid.json', grid)
    write(out/'selection.json', {'best_clock': best_clock, 'overall': overall,
                                 'overall_uses_clock': uses_clock, 'late_or_new_scores_used': False})

    clock_outputs = predictions(all_queries, best_clock['config'])
    baseline = {'config': base, 'validation': baseline_validation,
                'cohorts': {name: scoring.measure(values, baseline_outputs) for name, values in parts.items()}}
    best_clock_result = {'config': best_clock['config'], 'validation': best_clock['validation'],
                         'cohorts': {name: scoring.measure(values, clock_outputs) for name, values in parts.items()}}
    pairs = {name: scoring.paired_topk(values, baseline_outputs, clock_outputs)
             for name, values in parts.items()}

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
    manifest = {'version': VERSION, 'config': best_clock['config'], 'time_zone': statistics.CLOCK_TIME_ZONE,
                'memory_file': memory_path.name, 'memory_sha256': sha(memory_path),
                'current_sequence_length': 1, 'background_or_inventory_features': False,
                'deployment': False}
    write(out/'model-clock.json', manifest)
    with gzip.open(memory_path, 'rt') as source:
        restored = json.load(source)
    restored_background = statistics.BackgroundIndex(restored['inputs'], profiles=PROFILES)
    probe_rows = [validation[0], next(row for row in rows if row['split'] == 'test'),
                  next(row for row in rows if row['split'] == 'new_test')]
    restored_queries = statistics.prepare_queries(restored['inputs'], restored['training'], probe_rows,
                                                  homes, restored_background)
    for row in probe_rows:
        position = row['position']
        if restored_queries[position].predict(best_clock['config']) != clock_outputs[position]:
            raise ValueError('Saved clock model replay differs')

    summary = {
        'version': VERSION,
        'score_scope': scoring.report_scope(rows),
        'cohort_sizes': scoring.cohort_sizes(parts),
        'baseline': baseline,
        'best_clock': best_clock_result,
        'selection': {'config': overall['config'], 'uses_clock': uses_clock,
                      'best_clock_config': best_clock['config']},
        'pairs': pairs,
        'clock_coverage': clock_coverage(rows, data['observations']),
        'scoring_contract': scoring.contract(),
        'baseline_prediction_parity': True,
        'model_reload_parity': True,
        'collection_or_runtime_changed': False,
    }
    write(out/'summary.json', summary)
    write(out/'paired-comparisons.json', pairs)
    write(out/'predictions-private.json', {'statistics/none': baseline_outputs,
                                           'clock/'+best_clock['config']['id']: clock_outputs})
    report_path = out/'report-private.md'
    report_path.write_text(report(summary), encoding='utf-8')
    report_path.chmod(0o600)
    for name, value in source_bytes.items():
        path = out/('source-'+name)
        path.write_bytes(value)
        path.chmod(0o600)

    sources.verify_unchanged()
    if sha(snapshot/'events.jsonl') != event_hash:
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
    result = best_clock_result['cohorts']['all']
    print(json.dumps({'complete': True, 'output': str(out), 'best_clock': best_clock['config']['id'],
                      'overall_selected': overall['config']['id'], 'overall_uses_clock': uses_clock,
                      'accepted_top4': [baseline['cohorts']['all']['top4_hits'], result['top4_hits']],
                      'exact_top4': [baseline['cohorts']['all']['exact']['top4_hits'], result['exact']['top4_hits']]},
                     ensure_ascii=False))


if __name__ == '__main__':
    main()
