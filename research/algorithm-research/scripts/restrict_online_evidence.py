#!/usr/bin/env python3
"""Restrict an online scorer to destinations learned from the same current origin.

This is a prediction-time gate, not the retrospectively selected scoring cohort.
Every training-cleaning variant builds its own causal gate. The fixed benchmark,
its five-second answers, and its current candidate inventory remain unchanged.
"""
from collections import defaultdict
import argparse
import json
import os
from pathlib import Path

import compare_online_models as comparison
import evaluate_context_knn as core_io
from fused_online_models import FusedModel
from summarize_tuned_history import verify_run


class EvidenceRestrictedModel:
    """Incrementally learn normally; score only previously learned origin edges."""

    def __init__(self, base):
        self.base = base
        self.labels_by_origin = defaultdict(set)
        self.seen_outcomes = set()

    def learn(self, record):
        if 'outcome_id' in record:
            identity = (record['origin'], record['outcome_id'])
            if identity in self.seen_outcomes:
                return False
        else:
            identity = None
        result = self.base.learn(record)
        self.labels_by_origin[record['origin']].add(record['label'])
        if identity is not None:
            self.seen_outcomes.add(identity)
        return result

    def scores(self, query):
        allowed = self.labels_by_origin.get(query['origin'], ())
        # Score the original candidate list, preserving the component's scores.
        # Filtering before calling the base could alter candidate-relative scores.
        return {label: score for label, score in self.base.scores(query).items()
                if label in allowed}


def allowed_at_cutoffs(items, stream, inputs):
    """Query-safe origin edges; never read truth or benchmark membership."""
    cursor = 0
    by_origin = defaultdict(set)
    output = {}
    for index in sorted(inputs):
        query = items[index]
        while cursor < len(stream) and stream[cursor]['ready_index'] < query.index:
            row = stream[cursor]
            assert row['position'] < index
            assert row['input_index'] < query.index and row['outcome_index'] < query.index
            record = row['record']
            by_origin[record['origin']].add(record['label'])
            cursor += 1
        output[index] = frozenset(by_origin.get(inputs[index]['origin'], ()))
    return output


def restricted_predictions(core, predictions, allowed, items):
    assert predictions.keys() == allowed.keys()
    result = {}
    for index, prediction in predictions.items():
        query = items[index]
        raw = [key for key in prediction['raw_ranking']
               if comparison.semantic_candidate(key, query.candidates[key]) in allowed[index]]
        result[index] = {'sample_id': query.sample_id, 'raw_ranking': raw,
                         'predictions': core.select_four(raw, query.candidates)}
    return result


def build(config):
    import online_model_suite
    import stat_model_suite
    if config['family'] == 'same_origin_evidence_gate':
        return EvidenceRestrictedModel(build(config['parameters']['base']))
    if config['family'] == 'rank_fusion_ensemble':
        parameters = config['parameters']
        return FusedModel(build(parameters['left']), build(parameters['right']),
                          parameters['right_weight'], parameters['rank_offset'])
    return (stat_model_suite if config['id'].startswith('stat_') else online_model_suite).build(config)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base-runs', nargs='+', type=Path, required=True)
    parser.add_argument('--fusion-run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    out = args.output.resolve()
    if not out.is_relative_to(core_io.PRIVATE_ROOT.resolve()) or out == core_io.PRIVATE_ROOT.resolve():
        parser.error('Use a new private training run')
    sources = {}
    records = []
    expected = None
    for run in args.base_runs:
        run = run.resolve()
        provenance, hashes = verify_run(run)
        sources[str(run)] = hashes
        expected = expected or provenance['expansion_run']
        assert expected == provenance['expansion_run']
        for result in json.loads((run/'metrics.json').read_text()):
            records.append({**result, 'prediction_source': str(run/f"predictions-{result['training_cleaning']}-{result['config']['id']}-private.json"),
                            'prediction_format': 'base'})
    fusion_run = args.fusion_run.resolve()
    fusion_provenance, fusion_hashes = verify_run(fusion_run)
    sources[str(fusion_run)] = fusion_hashes
    assert fusion_provenance['expansion_run'] == expected
    fusion = json.loads((fusion_run/'selection.json').read_text())['best_current_only']
    assert fusion['config']['family'] == 'rank_fusion_ensemble'
    fusion_ablation = json.loads((fusion_run/'cleaning-ablation.json').read_text())[fusion['config']['id']]
    for result in fusion_ablation:
        records.append({**result, 'config': fusion['config'],
                        'prediction_source': str(fusion_run/f"final-predictions-{result['training_cleaning']}-{fusion['config']['id']}-private.json"),
                        'prediction_format': 'fusion'})
    identities = [(result['config']['id'], result['training_cleaning']) for result in records]
    assert len(identities) == len(set(identities))
    core, items, cohorts, streams, inputs, mappings, provenance = comparison.load_dataset(Path(expected))
    gates = {cleaning: allowed_at_cutoffs(items, streams[cleaning], inputs)
             for cleaning in ('5', '3', 'raw')}
    os.umask(0o077)
    out.mkdir(parents=True, mode=0o700, exist_ok=False)
    plan = {'version': 'same-origin-evidence-gate-v1', 'source_runs': sources,
            'gate': 'At least one matured unique actual outcome with this semantic destination from query current origin.',
            'gate_training_cleaning': 'Same factor as learner: raw,3,5 independently. Never import five-second evidence into another factor.',
            'gate_position': 'After unchanged component scores/raw ordering, before hierarchy selection.',
            'maturity': 'ready_index strictly less than query.index; wrapper learns only through causal replay.',
            'inputs': 'Current semantic origin and learned labels; no current truth, membership, grades or case IDs.',
            'evaluation': 'Fixed original5s62 evidence rows and original3s64 sensitivity rows; candidate omissions remain failures.',
            'search': 'Apply one fixed gate to every base replay plus previous best current-only fusion under all3 cleanings.',
            'base_replays': len(records)-3, 'fusion_replays': 3,
            'test_previously_seen': True, 'new_holdout': False, 'candidate_inventory_changed': False}
    core_io.write_json(out/'plan.json', plan)
    core_io.write_json(out/'input-audit.json', provenance)
    core_io.write_json(out/'gates-private.json', {cleaning: {str(index): sorted(labels)
                        for index, labels in values.items()} for cleaning, values in gates.items()})

    def read_prediction(record):
        values = json.loads(Path(record['prediction_source']).read_text())
        if record['prediction_format'] == 'base':
            values = values['predictions']
        return {int(index): value for index, value in values.items()}

    results = []
    source_records = {}
    for index, record in enumerate(records):
        cleaning = record['training_cleaning']
        config = {'id': 'restricted--' + record['config']['id'], 'family': 'same_origin_evidence_gate',
                  'feature_scope': record['config']['feature_scope'],
                  'parameters': {'base': record['config'], 'minimum_same_origin_outcomes': 1}}
        predictions = restricted_predictions(core, read_prediction(record), gates[cleaning], items)
        result = {'training_cleaning': cleaning, 'config': config,
                  'scores': {threshold: comparison.score_groups(rows, predictions)
                             for threshold, rows in cohorts.items()},
                  'before_gate_scores': record['scores'],
                  'prediction_source': record['prediction_source']}
        results.append(result)
        source_records[(config['id'], cleaning)] = record
        if (index+1) % 100 == 0 or index+1 == len(records):
            print(json.dumps({'stage': 'gate', 'done': index+1, 'total': len(records),
                              'best_top4': max(r['scores']['5']['supported']['top4_hits'] for r in results)}), flush=True)
    results.sort(key=comparison.selection_key)
    best = results[0]
    pure = next(result for result in results if result['config']['feature_scope'] == 'current_only')
    selected = {result['config']['id']: result for result in (best, pure)}
    for index, query in inputs.items():
        query['candidate_key_order'] = mappings[index]
    ablations = {}
    selected_errors = {}
    for identity, representative in selected.items():
        ablations[identity] = []
        for cleaning in ('5', '3', 'raw'):
            record = source_records[(identity, cleaning)]
            expected_predictions = restricted_predictions(core, read_prediction(record), gates[cleaning], items)
            actual, timings = comparison.replay(core, build(representative['config']), items,
                                                streams[cleaning], inputs, mappings)
            assert actual.keys() == expected_predictions.keys()
            assert all(actual[index]['raw_ranking'] == expected_predictions[index]['raw_ranking'] and
                       actual[index]['predictions'] == expected_predictions[index]['predictions']
                       for index in actual), 'Cached/live evidence gate mismatch'
            scores = {threshold: comparison.score_groups(rows, actual) for threshold, rows in cohorts.items()}
            row = {'training_cleaning': cleaning, 'scores': scores, 'timing': timings}
            ablations[identity].append(row)
            for result in results:
                if result['config']['id'] == identity and result['training_cleaning'] == cleaning:
                    assert result['scores'] == scores
                    result['timing'] = timings
            core_io.write_json(out/f'final-predictions-{cleaning}-{identity}-private.json', actual)
            selected_errors[cleaning+':'+identity] = comparison.diagnose(cohorts['5'], actual)
    selection = {'best_development': best, 'best_current_only': pure,
                 'number_replays': len(results), 'base_replays': len(records)-3,
                 'fusion_replays': 3, 'fixed_primary_n': 62, 'fixed_candidate_ceiling': 56,
                 'streaming_cached_parity': True}
    core_io.write_json(out/'selection.json', selection)
    core_io.write_json(out/'metrics.json', results)
    core_io.write_json(out/'cleaning-ablation.json', ablations)
    core_io.write_json(out/'selected-errors-private.json', selected_errors)
    core_io.write_json(out/'benchmark-membership.json', json.loads((Path(expected)/'benchmark-membership.json').read_text()))
    lines = ['# 同当前起点的成熟证据约束', '',
             f'对{len(records)-3}个基础模型回放与既有最佳融合的3种清洗训练加入同一可在线计算的限制。',
             '只有当前起点曾经成熟地转移到过的语义目标，才保留其模型分数；过滤在最终层级去重之前执行。每种训练清洗使用自身历史，完全不读取本次真实下一步或评分成员。',
             '这是在已查看开发数据上比较的新结构；没有新增独立留出测试。候选库存、62个主评分成员和5秒答案均不变。', '',
             '| 选择 | 训练清洗 | Top4/62 | Top1/62 | 原晚38 Top4 |',
             '|---|---|---:|---:|---:|']
    for name, result in [('所有输入范围最优', best), ('仅当前界面最优', pure)]:
        metric = result['scores']['5']['supported']
        late = result['scores']['5']['late_supported']
        lines.append(f"| {name} | {result['training_cleaning']} | {metric['top4_hits']}/62 ({metric['top4']:.1%}) | {metric['top1_hits']}/62 | {late['top4_hits']}/38 |")
    lines += ['', '## 当前界面结构', '', '```json', json.dumps(pure['config'], ensure_ascii=False, indent=2), '```', '',
              '当前界面 → 底层模型原样打分 → 保留从此起点学过的目标 → 唯一候选映射与层级去重 → 最多4条。',
              '新行为标签成熟后，底层模型和当前起点的目标集合一起增量更新；无同起点证据时可以不返回建议。', '',
              '## 固定结构的清洗对照', '',
              '| 训练标签清洗 | 固定5秒主集 Top4 | 原生3秒64条 Top4 |', '|---|---:|---:|']
    for row in ablations[pure['config']['id']]:
        lines.append(f"| {row['training_cleaning']} | {row['scores']['5']['supported']['top4_hits']}/62 | {row['scores']['3']['supported']['top4_hits']}/64 |")
    lines += ['', '清洗层仍独立：仅跳过完整观测且短于阈值的到达，窗口/标签页归属判断正常导航，未知停留和中断不当短停留；保留原始数据，满原30秒观察窗后才能学习。', '',
              '## 证据与边界', '',
              json.dumps(pure['scores']['5']['grades'], ensure_ascii=False, indent=2), '',
              '6个原始候选遗漏仍留在主分母；56/62是固定库存上限。此限制根据过去支持筛可推荐目标，并不能预先识别事后按真实答案选出的有证据评分样本。',
              '所有选中结构在三种训练清洗下均重新逐条增量回放，并与缓存过滤后的完整排序及Top4逐条核对。实际计时记录于cleaning-ablation.json。', '',
              '未改原始采集、此前run或线上菜单。']
    (out/'report-private.md').write_text('\n'.join(lines)+'\n')
    for name in ('restrict_online_evidence.py', 'compare_online_models.py', 'stat_model_suite.py',
                 'online_model_suite.py', 'fused_online_models.py', 'test_restrict_online_evidence.py'):
        (out/('source-'+name)).write_bytes(Path(__file__).with_name(name).read_bytes())
    core_io.write_json(out/'run.json', {'version': 'same-origin-evidence-gate-v1', **provenance,
                                      'source_runs': sources, 'raw_data_modified': False,
                                      'live_model_updated': False, 'test_previously_seen': True,
                                      'complete': True, 'streaming_cached_parity': True})
    core_io.write_json(out/'artifact-hashes.json', {path.name: core_io.sha(path) for path in out.iterdir() if path.is_file()})
    print(json.dumps({'complete': True, 'output': str(out), 'best': {
        'config': best['config'], 'training_cleaning': best['training_cleaning'],
        'main': best['scores']['5']['supported']}, 'current_only': {
        'config': pure['config'], 'training_cleaning': pure['training_cleaning'],
        'main': pure['scores']['5']['supported']}}), flush=True)


if __name__ == '__main__':
    main()
