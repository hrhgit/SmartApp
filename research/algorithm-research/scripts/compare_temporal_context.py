"""Train/evaluate time-context models on an unchanged private causal benchmark."""
import argparse
from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone
import gzip
import hashlib
import json
import os
from pathlib import Path
import statistics
import time
import evaluation_contract as scoring

import evaluate_context_knn as io
from compare_semantic_trajectory import groups, training_rows
from navigation_hierarchy import learning_record, service, SERVICE
from summarize_tuned_history import verify_run
from temporal_context import VERSION, ContextSimilarity, TemporalNavigationModel, build_contexts


SOURCES = ('evaluation_contract.py', 'known_inactivity.py', 'compare_temporal_context.py', 'temporal_context.py', 'compare_semantic_trajectory.py',
           'navigation_hierarchy.py', 'semantic_online.py', 'semantic_annotations.py',
           'diagnose_action_granularity.py', 'trajectory_supervision.py',
           'evaluate_context_knn.py', 'summarize_tuned_history.py')


def configurations():
    configs = [{'id': 'current_baseline', 'mode': 'baseline', 'windows': [],
                'sequence_weight': 0., 'floor': 1.}]
    for floor in (.1, .5):
        configs.append({'id': f'sequence_f{floor}', 'mode': 'sequence', 'windows': [],
                        'sequence_weight': 1., 'floor': floor})
    for mode in ('frequency', 'rich'):
        for windows in ([5], [15], [60], [5, 15, 60]):
            for seq in (0., .25):
                for floor in (.1, .5):
                    configs.append({'id': f'{mode}_w' + '-'.join(map(str, windows)) + f'_s{seq}_f{floor}',
                                    'mode': mode, 'windows': windows, 'sequence_weight': seq, 'floor': floor})
    return configs


def model_inputs(data, contexts):
    # Never pass reference-stream x, source inventory or evaluation metadata.
    return [{k: q[k] for k in ('position', 'index', 'timestamp', 'origin', 'annotation', 'target')}
            | {'temporal': contexts[q['position']]} for q in data['observations']]


def replay(observations, stream, evaluation, config, homes, similarity=None):
    model = TemporalNavigationModel(config, similarity)
    cursor, predictions, raw, rankable, latencies = 0, {}, {}, {}, []
    for row in sorted(evaluation, key=lambda r:r['position']):
        pos = row['position']; q = observations[pos]
        while cursor < len(stream) and stream[cursor]['ready_index'] < q['index']:
            before = stream[cursor]; x = observations[before['position']]
            if x['index'] >= q['index'] or x['temporal']['through_index'] != x['index']:
                raise ValueError('Future feature or input entered training')
            if before['record']['target_evidence_index'] > before['ready_index']:
                raise ValueError('Outcome metadata was not available at training maturity')
            model.learn(x, learning_record(before, x))
            cursor += 1
        start = time.perf_counter()
        ranked = model.ranked(q, n=10**9)
        latencies.append(1000*(time.perf_counter()-start))
        raw[pos] = ranked[:4]
        resolved = [homes.get(pos) if k == SERVICE and homes.get(pos) else k for k in ranked]
        predictions[pos] = resolved[:4]
        rankable[pos] = resolved
    return predictions, raw, rankable, latencies


def metric(rows, predictions, rankable):
    outputs = {pos: {'predictions': values, 'rankable': rankable[pos]} for pos, values in predictions.items()}
    return scoring.measure(rows, outputs)


def metrics(rows, predictions, rankable):
    return {name: metric(part, predictions, rankable) for name, part in groups(rows).items()}


def selection_key(scores):
    # Only fixed early validation members; accepted original evidence Top4 first.
    return tuple(scores[group][metric_name] for group,metric_name in (
        ('original', 'top4_hits'), ('expanded', 'top4_hits'),
        ('original', 'top1_hits'), ('expanded', 'top1_hits'),
        ('all', 'top4_hits'), ('all', 'top1_hits')))


def choose(results, ids=None):
    available = [r for r in results if ids is None or r['config']['id'] in ids]
    # Stable prespecified ordering wins ties; late metrics are never read.
    return max(available, key=lambda r:selection_key(r['validation']))


def pct(value):
    return '—' if value is None else f'{value:.1%}'


def report(summary):
    chosen = summary['selected_temporal']
    lines = ['# 时间背景模型训练对照', '',
             '后台采集保持不变；本轮所有模型均不读取后台库存。当前目标、语义共享、目标层级、清洗与评分标签固定。',
             '历史转移按早先的时间背景重新加权；当前目标支持强度与全局回退保持原值。输出不受近期目标集合限制。', '',
             f"比较 {summary['config_count']} 组配置，按较早验证段的固定有证据样本接受 Top4 优先选参；同分按既定指标及配置顺序处理。",
             f"选出的时间背景配置：`{chosen['config']['id']}`。", '',
             '| 评测范围 | 模型 | 样本 | 接受 Top1 | 接受 Top4 | 返回预测比例 |',
             '|---|---|---:|---:|---:|---:|']
    for split,group,label in [('validation','original','较早验证／原始有证据'),
                              ('test','original','较晚段／原始有证据'),
                              ('full','original','全部原始有证据'),
                              ('test','all','较晚段／全部'),
                              ('full','expanded','全部扩展有证据'),
                              ('full','all','全部固定截点')]:
        for name,result in [('当前模型基线',summary['baseline']),('时间背景',chosen)]:
            m = result[split][group]
            lines.append(f"| {label} | {name} | {m['n']} | {pct(m['top1'])} ({m['top1_hits']}) | {pct(m['top4'])} ({m['top4_hits']}) | {pct(m['prediction_coverage'])} |")
    lines += ['', '## 输入与可用性', '',
              '- 时间窗口比较 5、15、60 分钟及三者组合；访问计数只计已确认的独立到达。',
              '- 丰富时间背景包括衰减次数、距最近访问的衰减值、已观察的非空闲停留；另比较保留最近 2–5 步有序后缀。',
              '- 初始/恢复快照不计新增到达；补全不重复计数，也不把具体页面的停留倒填到尚未知道页面时。',
              '- 暂停、采集失效、会话切换和已观察到的空闲不跨越累计停留；未知时段不当作确认没有行为。',
              '- 观测比例仅指已记录的非空闲前台连续时间，不等于用户注意力或所有来源完整。',
              '- 背景不是最近访问目标的推荐名单：仍可预测任何已有成熟行为标签的目标。', '',
              '## 特征消融', '',
              '| 配置家族（各自由验证段选择） | 验证有证据 Top4 | 较晚有证据 Top4 | 较晚全部 Top4 |',
              '|---|---:|---:|---:|']
    for mode, result in summary['selected_by_mode'].items():
        lines.append(f"| {mode} | {pct(result['validation']['original']['top4'])} | {pct(result['test']['original']['top4'])} | {pct(result['test']['all']['top4'])} |")
    lines += ['', '不同家族均使用同样的清洗和标签；家族间比较包含各自验证选参的影响。全部配置和固定参数对照保存在 results.json。', '',
              '## 限制与审计', '',
              '- 该固定数据集此前已多次查看，较晚段只用于本轮按时间选参的开发检查，不能称全新独立测试。',
              '- 原始／扩展有证据分组依据真实后续目标事后定义，不能当作线上接受集或置信保证。',
              '- 服务级与具体会话级分别评分；服务命中不计为精确会话命中。窗口输出仍遵循既定所属应用回退口径。',
              '- 评分截点统一审查合法性，排除原因另存，不能从本次数据推断所有自然使用时刻的覆盖率。',
              '- 模型产物保存在个人训练目录，没有替换正在运行的模型或改动采集数据库。',
              f"- 原始事件与数据集哈希校验通过；旧运行时基线预测不一致数：{summary['baseline_parity_mismatches']}。", '',
              '## 时间特征覆盖', '',
              f"提取审计：`{json.dumps(summary['feature_audit'], ensure_ascii=False)}`", '',
              f"评测截点窗口覆盖：`{json.dumps(summary['context_coverage'], ensure_ascii=False)}`", '']
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--baseline-run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    out = args.output.expanduser().resolve()
    if not out.is_relative_to(io.PRIVATE_ROOT.resolve()) or out == io.PRIVATE_ROOT.resolve() or out.exists():
        parser.error('Use a new private SmartApp training directory')
    provenance, dataset_hashes = verify_run(args.dataset)
    _, baseline_hashes = verify_run(args.baseline_run)
    snapshot = Path(provenance['snapshot'])
    expected = provenance['migration']['current']['events.jsonl']
    if io.sha(snapshot/'events.jsonl') != expected:
        raise ValueError('Raw event snapshot differs from frozen dataset')
    with gzip.open(args.dataset/'dataset-private.json.gz', 'rt') as f:
        data = json.load(f)
    data['evaluation'] = scoring.prepare_evaluation(data)
    events = io.load_lines(snapshot/'events.jsonl')
    sources = {name: Path(__file__).with_name(name).read_bytes() for name in SOURCES}
    contexts, homes, audit = build_contexts(events, data['observations'], io.KNOWN_TEST_SESSIONS)
    observations = model_inputs(data, contexts)
    stream = [r for r in training_rows(data, 'hard_kind') if r['head'] == 0]
    configs = configurations()
    test = [r for r in data['evaluation'] if r['split'] == 'test']
    test_index = min(observations[r['position']]['index'] for r in test)
    validation = [r for r in data['evaluation'] if r['split'] == 'validation'
                  and data['observations'][r['position']]['ready_index'] < test_index]
    if not validation or not any(r['evidence_grade'] == 'original' for r in validation):
        raise ValueError('No mature validation evidence before test boundary')
    os.umask(0o077)
    out.mkdir(mode=0o700, parents=True, exist_ok=False)
    scoring.prepare_evaluation(data, out)
    plan = {'version': VERSION, 'configs': configs, 'label_cleaning': 'hard_kind head0 unchanged',
            'primary_score': 'unified audited acceptance; original evidence Top4',
            'selection': 'validation original Top4, expanded Top4, original Top1, expanded Top1, all Top4, all Top1; first config on tie',
            'validation_n': len(validation), 'test_n': len(test), 'fixed_evaluation_n': len(data['evaluation']),
            'validation_labels_mature_before_index': test_index, 'fresh_holdout': False,
            'background_features': False, 'current_target_and_navigation_hierarchy_fixed': True,
            'collection_changed': False, 'runtime_installation_changed': False,
            'candidate_or_recent_target_output_restriction': False,
            'source_hashes': {n: hashlib.sha256(b).hexdigest() for n,b in sources.items()},
            'dataset_hashes': dataset_hashes, 'baseline_hashes': baseline_hashes}
    io.write_json(out/'plan.json', plan)
    with gzip.open(out/'inputs-private.json.gz', 'wt') as f:
        json.dump(observations, f, ensure_ascii=False)
    results, all_predictions, all_raw, all_rankable = [], {}, {}, {}
    similarity = ContextSimilarity()
    for config in configs:
        pred, raw, rankable, ms = replay(observations, stream, data['evaluation'], config, homes, similarity)
        result = {'config': config, 'validation': metrics(validation, pred, rankable),
                  'prediction_ms_median': statistics.median(ms), 'prediction_ms_max': max(ms)}
        results.append(result)
        all_predictions[config['id']], all_raw[config['id']], all_rankable[config['id']] = pred, raw, rankable
        print(json.dumps({'finished':config['id'], 'validation_original': result['validation']['original']}), flush=True)
    selected = choose(results, {c['id'] for c in configs if c['mode'] != 'baseline'})
    by_mode = {mode: choose(results, {c['id'] for c in configs if c['mode'] == mode})
               for mode in ('sequence', 'frequency', 'rich')}
    # Persist selection before evaluating late/full metrics.
    io.write_json(out/'selection.json', {'selected_temporal': selected, 'selected_by_mode': by_mode,
                                       'selection_uses_test_or_future_labels': False})
    frozen = json.loads((args.baseline_run/'predictions-private.json').read_text())
    baseline_expected = frozen['v2_exact']
    mismatches = sum(all_predictions['current_baseline'][r['position']] != baseline_expected[str(r['position'])]
                     for r in data['evaluation'])
    if mismatches:
        raise ValueError(f'Frozen current baseline parity failed: {mismatches}')
    for result in results:
        key = result['config']['id']; pred, rankable = all_predictions[key], all_rankable[key]
        result['test'] = metrics(test, pred, rankable)
        result['full'] = metrics(data['evaluation'], pred, rankable)
        result['exact_full'] = {name: m['exact'] for name, m in result['full'].items()}
    coverage = {}
    for minutes in (5,15,60):
        rows = [contexts[r['position']]['windows'][str(minutes)] for r in data['evaluation']]
        coverage[str(minutes)] = {'n': len(rows),
            'with_confirmed_history': sum(r['confirmed_arrivals'] > 0 for r in rows),
            'median_confirmed_arrivals': statistics.median(r['confirmed_arrivals'] for r in rows),
            'median_observed_nonidle_fraction': statistics.median(r['observed_nonidle_fraction'] for r in rows)}
    summary = {'version': VERSION, 'config_count': len(configs), 'baseline_parity_mismatches': mismatches,
               'baseline': results[0], 'selected_temporal': selected, 'selected_by_mode': by_mode,
               'feature_audit': audit, 'context_coverage': coverage,
               'fresh_holdout': False, 'collection_changed': False, 'runtime_changed': False}
    io.write_json(out/'results.json', results)
    io.write_json(out/'summary.json', summary)
    io.write_json(out/'predictions-private.json', all_predictions)
    io.write_json(out/'raw-predictions-private.json', all_raw)
    key = selected['config']['id']
    cases = [{'position':r['position'], 'sample_id':r['sample_id'], 'split':r['split'],
              'evidence_grade':r['evidence_grade'], 'truth':r['truth_action'],
              'baseline': all_predictions['current_baseline'][r['position']],
              'temporal': all_predictions[key][r['position']],
              'current': observations[r['position']]['target']}
             for r in data['evaluation']]
    io.write_json(out/'cases-private.json', cases)
    # A trained retrieval/statistics model is its mature weighted feature memory.
    # Save the final snapshot state, clearly separate from earlier replay predictions.
    trained = [{'ready_index':r['ready_index'], 'input': observations[r['position']],
                'record': learning_record(r, observations[r['position']])} for r in stream
               if r['ready_index'] < len(events)]
    with gzip.open(out/'model-private.json.gz', 'wt') as f:
        json.dump({'version': VERSION, 'config': selected['config'], 'records': trained,
                   'asof_event_index':len(events)-1, 'asof_timestamp_ms':events[-1]['timestampMs'],
                   'background_features':False, 'deployment':False}, f, ensure_ascii=False)
    # Reload through the public learning path and verify one final-state query.
    fitted, restored = TemporalNavigationModel(selected['config']), TemporalNavigationModel(selected['config'])
    for r in trained:
        fitted.learn(r['input'], r['record'])
    with gzip.open(out/'model-private.json.gz', 'rt') as f:
        saved = json.load(f)
    for r in saved['records']:
        restored.learn(r['input'], r['record'])
    probe = observations[-1]
    if fitted.ranked(probe) != restored.ranked(probe):
        raise ValueError('Saved model replay differs')
    (out/'report-private.md').write_text(report(summary), encoding='utf-8')
    if (verify_run(args.dataset)[1] != dataset_hashes or verify_run(args.baseline_run)[1] != baseline_hashes
            or io.sha(snapshot/'events.jsonl') != expected):
        raise ValueError('Input changed during training')
    for name, content in sources.items():
        if Path(__file__).with_name(name).read_bytes() != content:
            raise ValueError('Source changed during training')
        (out/('source-'+name)).write_bytes(content)
    io.write_json(out/'run.json', {'version': VERSION, 'complete': True,
        'dataset':str(args.dataset.resolve()), 'dataset_hashes':dataset_hashes,
        'baseline_run':str(args.baseline_run.resolve()), 'baseline_hashes':baseline_hashes,
        'snapshot':str(snapshot), 'events_sha256':expected,
        'source_hashes':plan['source_hashes'], 'created_at':datetime.now(timezone.utc).isoformat(),
        'selected_config': selected['config'], 'trained_records':len(trained), 'model_reload_parity':True,
        'input_and_source_hashes_verified_before_after':True,
        'test_previously_seen':True, 'background_features':False, 'raw_modified':False,
        'collection_or_runtime_changed':False})
    io.write_json(out/'artifact-hashes.json', {p.name:io.sha(p) for p in out.iterdir() if p.is_file()})
    for path in out.iterdir():
        if path.stat().st_mode & 0o777 != 0o600:
            raise ValueError('Private artifact permissions must be 0600')
    print(json.dumps({'output':str(out), 'selected': selected['config'],
                      'baseline_original':results[0]['full']['original'],
                      'temporal_original':selected['full']['original'],
                      'late_baseline_original':results[0]['test']['original'],
                      'late_temporal_original':selected['test']['original']}, ensure_ascii=False))


if __name__ == '__main__':
    main()
