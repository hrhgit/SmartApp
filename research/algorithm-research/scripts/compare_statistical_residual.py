"""Train statistical-prior residual rankers with delayed, causal feedback."""
import argparse
from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone
import gzip
import importlib
import json
import os
from pathlib import Path
import sys
import evaluation_contract as scoring
from evaluation_contract import measure, paired_topk

from diagnose_background_contributions import PRIVATE_ROOT, checked_run, sha, write_json


def configurations():
    return [{'id': f'{objective}_{"context" if context else "no_context"}_lr{lr}',
             'objective': objective, 'context': context, 'lr': lr, 'rehearsal': 7, 'seed': 20260923}
            for objective in ('top2_hinge', 'cross_entropy')
            for context in (True, False) for lr in (.03, .1)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fusion-run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    out, source = args.output.resolve(), args.fusion_run.resolve()
    if out.exists() or not out.is_relative_to(PRIVATE_ROOT.resolve()) or out == PRIVATE_ROOT.resolve():
        parser.error('Use a new private training run')
    meta, source_hashes = checked_run(source)
    dataset, statrun, learnrun = (Path(meta[k]) for k in ('dataset', 'statistics_run', 'learning_run'))
    data_meta, data_hashes = checked_run(dataset)
    _, stat_hashes = checked_run(statrun)
    _, learn_hashes = checked_run(learnrun)
    os.umask(0o077); sys.dont_write_bytecode = True; out.mkdir(mode=0o700)
    for path in source.glob('*.py'): (out/path.name).write_bytes(path.read_bytes())
    for name in ('statistical_residual.py', 'compare_statistical_residual.py', 'test_statistical_residual.py'):
        (out/name).write_bytes(Path(__file__).with_name(name).read_bytes())
    # Import the frozen dependencies used by both original branches, not newer cleaning code.
    sys.path.insert(0, str(out))
    cs = importlib.import_module('context_statistics')
    comp = importlib.import_module('compare_multibackground')
    residual = importlib.import_module('statistical_residual')
    diagnosis = importlib.import_module('diagnose_background_contributions')
    cohorts = scoring.cohorts
    import numpy as np
    from scipy.sparse import vstack, save_npz, load_npz
    with gzip.open(dataset/'dataset-private.json.gz', 'rt') as f: data = json.load(f)
    with gzip.open(learnrun/'inputs-private.json.gz', 'rt') as f: inputs = json.load(f)
    for q in inputs: q.update(inventory_state={}, inventory_quality={}, inventory_through_index=q['index'])
    config = json.loads((statrun/'model-time.json').read_text())['config']
    homes, asof = diagnosis.asof_homes(Path(meta['snapshot']), data['observations'], comp.io,
                                      importlib.import_module('navigation_hierarchy').ENTRY)
    rows = scoring.prepare_evaluation(data, out); groups = cohorts(rows)
    test_index = min(inputs[r['position']]['index'] for r in rows if r['split'] == 'test')
    validation = [r for r in rows if r['split'] == 'validation' and
                  data['observations'][r['position']]['ready_index'] < test_index]
    parts = {'original': [r for r in validation if r['evidence_grade'] == 'original'],
             'expanded': [r for r in validation if r['evidence_grade'] != 'unsupported'], 'all': validation}
    plan = {'statistical_prior_config': config, 'configurations': configurations(),
            'fusion_coefficient': None, 'formula': 'log(statistical_probability) + learned_correction',
            'selection': 'early validation original Top2, Top1; expanded Top2, Top1; original Top4; all Top2, Top1',
            'validation_positions': [r['position'] for r in validation],
            'current_sequence_length': 1, 'inventory_features': False,
            'training_scope': 'all unique mature navigation relations, not truth-selected evaluation evidence',
            'new_outcomes': 'skip residual training if outcome was unavailable in the original prediction vocabulary; still learn it in statistics',
            'rehearsal': 'up to seven mature examples using their frozen original priors and vocabularies',
            'top2_hinge': 'unit margin against second highest incorrect target',
            'source_runs': {str(source): source_hashes, str(dataset): data_hashes,
                            str(statrun): stat_hashes, str(learnrun): learn_hashes}}
    write_json(out/'plan.json', plan)
    stream = [r for r in comp.training_rows(data, 'hard_kind') if r['head'] == 0]
    memory = cs.StatisticalMemory(); background = cs.BackgroundIndex(inputs)
    examples, training, baseline = {}, [], {}; cursor = 0
    def advance_statistics(cutoff):
        nonlocal cursor
        while cursor < len(stream) and stream[cursor]['ready_index'] < cutoff:
            row = stream[cursor]; before = inputs[row['position']]
            assert before['index'] <= row['ready_index'] < cutoff
            assert row['record']['target_evidence_index'] <= row['ready_index']
            n = len(memory.hierarchy.records)
            memory.learn(before, comp.learning_record(row, before), row['ready_index'])
            if len(memory.hierarchy.records) > n:
                record = memory.hierarchy.records[-1]
                if record['navigationDisposition'] == 'navigation':
                    example = examples[row['position']]
                    label = example['mapping'].get(record['navigationLabel'], record['navigationLabel'])
                    training.append({'position': row['position'], 'input_index': before['index'],
                                     'ready_index': row['ready_index'], 'label': label, 'weight': record['weight'],
                                     'key': [record['origin'], record['label'], record['outcomeID']]})
            cursor += 1
    for q in inputs:
        advance_statistics(q['index'])
        prepared = memory.prepare(q, homes[q['position']], background)
        scores = prepared.scores(config)
        mapping = {prepared.labels[i]: prepared.emitted[j] for i, j in zip(prepared.keep, prepared.output_ids)}
        descriptions = {}
        for record in memory.hierarchy.records:
            emitted = mapping.get(record['navigationLabel'])
            if emitted is None: continue
            target = record.get('destination') or {}
            desc = descriptions.setdefault(emitted, {'aliases': set(), 'owner': target.get('bundleID', ''),
                                                     'kind': target.get('kind', 'unknown')})
            desc['aliases'].add(record['label'])
        descriptions = {k: {**v, 'aliases': sorted(v['aliases'])} for k, v in descriptions.items()}
        x = residual.encode_features(q, scores, descriptions)
        x['mapping'] = mapping; examples[q['position']] = x
        baseline[q['position']] = prepared.predict(config)
        if (q['position']+1) % 250 == 0:
            print(json.dumps({'phase': 'causal_training_features', 'inputs': q['position']+1}), flush=True)
    advance_statistics(asof+1)
    assert cursor == len(stream)
    expected = json.loads((source/'predictions-private.json').read_text())
    assert all(baseline[r['position']] == expected['learning_0.0'][str(r['position'])] for r in rows)
    zero = residual.ResidualRanker(configurations()[0])
    assert all(comp.bundle(zero.scores(examples[r['position']]), homes[r['position']]) == baseline[r['position']] for r in rows)
    # Serialize all original feature matrices: replay never uses final priors on old examples.
    positions = sorted(examples); offsets = [0]
    for pos in positions: offsets.append(offsets[-1]+len(examples[pos]['labels']))
    matrix = vstack([examples[pos]['X'] for pos in positions], format='csr')
    save_npz(out/'causal-features.npz', matrix)
    with gzip.open(out/'causal-examples-private.json.gz', 'wt') as f:
        json.dump({'positions': positions, 'offsets': offsets, 'training': training, 'homes': homes,
                   'examples': {pos: {'labels': x['labels'], 'prior': x['prior'].tolist(), 'mapping': x['mapping']}
                                for pos, x in examples.items()}}, f)
    restored_matrix = load_npz(out/'causal-features.npz')
    assert (restored_matrix != matrix).nnz == 0
    print(json.dumps({'phase': 'features_ready', 'training_relations': len(training),
                      'statistical_records': len(memory.records), 'baseline_parity': True}), flush=True)
    outputs = {'statistics': {r['position']: baseline[r['position']] for r in rows}}
    results = [{'id': 'statistics', 'config': None,
                'validation': {g: measure(rs, outputs['statistics']) for g, rs in parts.items()}}]
    audits = {}; saved = {}
    eval_positions = {r['position'] for r in rows}
    for c in configurations():
        model = residual.ResidualRanker(c); cursor = 0; pred = {}
        for q in inputs:
            while cursor < len(training) and training[cursor]['ready_index'] < q['index']:
                model.learn(training[cursor], examples, q['index']); cursor += 1
            if q['position'] in eval_positions:
                pred[q['position']] = comp.bundle(model.scores(examples[q['position']]), homes[q['position']])
        while cursor < len(training):
            model.learn(training[cursor], examples, asof+1); cursor += 1
        outputs[c['id']] = pred
        result = {'id': c['id'], 'config': c, 'validation': {g: measure(rs, pred) for g, rs in parts.items()}}
        results.append(result); audits[c['id']] = dict(model.audit)
        state = model.save(out, 'model-'+c['id']); loaded = residual.ResidualRanker.load(state)
        for pos in eval_positions:
            assert model.scores(examples[pos]) == loaded.scores(examples[pos])
        # Check that restoring optimizer, deduplication and rehearsal RNG permits identical next learning.
        resume = deepcopy(training[-1]); resume['key'] = ['synthetic-reload-check']+resume['key']
        model.learn(resume, examples, asof+1); loaded.learn(resume, examples, asof+1)
        assert np.array_equal(model.weights, loaded.weights) and np.array_equal(model.accumulator, loaded.accumulator)
        saved[c['id']] = state.name
        print(json.dumps({'phase': 'model_trained', 'id': c['id'], 'validation': result['validation']['original'],
                          'audit': audits[c['id']], 'reload_resume_parity': True}), flush=True)
    chosen = residual.select_top2(results)
    learned = residual.select_top2(results[1:])
    # Save selection before inspecting any aggregate, late or new metrics.
    write_json(out/'validation-grid.json', results)
    write_json(out/'selection.json', {'selected': chosen, 'best_learned': learned, 'late_or_new_used': False})
    for weight in ('.3', '.5', '1.0'):
        oldname = 'learning_0'+weight if weight.startswith('.') else 'learning_'+weight
        outputs['previous_'+oldname] = {int(k): v for k, v in expected[oldname].items()}
    stat_outputs = json.loads((statrun/'predictions-private.json').read_text())
    outputs['current_only_statistics'] = {int(k): v for k, v in stat_outputs['none'].items()}
    scores = {name: {g: measure(rs, pred) for g, rs in groups.items()} for name, pred in outputs.items()}
    pairings = {name: {g: paired_topk(rs, outputs['statistics'], pred) for g, rs in groups.items()}
                for name, pred in outputs.items()}
    summary = {'selected': chosen['id'], 'best_learned': learned['id'], 'scores': scores, 'paired': pairings,
               'cohort_sizes': scoring.cohort_sizes(groups), 'scoring_contract': scoring.contract(),
               'audits': audits, 'saved_models': saved,
               'baseline_parity': True, 'zero_residual_parity': True, 'reload_resume_parity': True,
               'feature_cache_reload_parity': True, 'runtime_or_collection_changed': False}
    write_json(out/'summary.json', summary); write_json(out/'predictions-private.json', outputs)
    display = list(dict.fromkeys(['statistics', learned['id'], chosen['id'], 'previous_learning_0.3',
                                 'previous_learning_0.5', 'previous_learning_1.0', 'current_only_statistics']))
    def cell(m, k): return f"{m[f'top{k}_hits']}/{m['n']} ({m[f'top{k}']:.1%})"
    lines = ['# 统计基础分数上的情景残差学习：Top2 实验', '',
             scoring.report_scope(rows), '',
             f"早期验证选择：{chosen['id']}。最佳学习方案：{learned['id']}。", '',
             '| 模型 | Top1 | Top2 | Top4 |', '|---|---:|---:|---:|']
    for name in display:
        m = scores[name]['evidence_total']; lines.append('| '+name+' | '+' | '.join(cell(m, k) for k in (1, 2, 4))+' |')
    lines += ['', '## 机制与训练边界', '',
              '- 最终排序分数 = log(时间背景统计分数)＋学习修正。修正从零开始，初始预测逐样本复现统计基线；不搜索固定融合比例。统计基础参数保持上轮已选配置。',
              '- 比较前二边界 hinge 与整体 cross entropy，情景特征开/关，学习率 0.03/0.1，共 8 组。模型仍有学习率等训练超参数；避免的是两套独立预测之后的比例调参。',
              '- 情景使用 5/60 分钟、24 小时及累计频次、最近访问、停留、覆盖，以及短长差异和当前目标交互。前台序列长度 1，无后台库存或顺序后缀。',
              '- 训练使用全部符合清洗规则的成熟导航关系，不用有证据评测分组筛训练数据。每个训练输入的统计分数、特征与输出词表在原截点冻结，之后才接收成熟标签。复习旧样本沿用原分数，不允许重新算入自己的结果。',
              '- 截点尚不能输出的标签仍进入统计记忆，残差阶段明确跳过该样本，不将标签补入过去的输出集合。合法评测样本包括当时不可输出的目标；服务接受成绩与精确成绩由统一入口同时保存。',
              '- 优化目标是 surrogate loss，不保证实际 Top2 提升；仅用较早验证选择，选择保存后才计算其余成绩。数据此前被多轮研究查看，不能称全新独立测试。', '',
              '## 分组成绩', '', '| 模型 | 范围 | Top1 | Top2 | Top4 |', '|---|---|---:|---:|---:|']
    for name in ('statistics', learned['id']):
        for group in ('old_evidence', 'late_old_evidence', 'new_evidence', 'expanded_total'):
            lines.append('| '+name+' | '+group+' | '+' | '.join(cell(scores[name][group], k) for k in (1, 2, 4))+' |')
    lines += ['', '## 验证与持久化', '',
              '- 原时间统计 234 条预测及零修正基线完全一致；每个学习模型保存参数、AdaGrad、成熟训练索引、去重与复习随机状态，重载分数和继续更新一致。',
              '- 保存全部 1328 个截点的因果稀疏特征矩阵、原统计先验和词表；所有输入/源码/权重均记录哈希。产物仅写私有训练目录。',
              '- 未修改采集、已完成 run、运行中模型或安装包。', '']
    (out/'report-private.md').write_text('\n'.join(lines))
    for path, hashes in ((source, source_hashes), (dataset, data_hashes), (statrun, stat_hashes), (learnrun, learn_hashes)):
        assert checked_run(path)[1] == hashes
    write_json(out/'run.json', {'complete': True, 'version': 'statistical-residual-top2-v1',
                               'created_at': datetime.now(timezone.utc).isoformat(), 'source': str(source),
                               'dataset': str(dataset), 'snapshot': meta['snapshot'], 'selected': chosen['id'],
                               'raw_modified': False, 'runtime_changed': False})
    write_json(out/'artifact-hashes.json', {p.name: sha(p) for p in sorted(out.iterdir()) if p.is_file()})
    assert all(p.stat().st_mode & 0o777 == 0o600 for p in out.iterdir() if p.is_file())
    print(json.dumps({'complete': True, 'output': str(out), 'selected': chosen['id'], 'best_learned': learned['id'],
                      'evidence_total': {name: scores[name]['evidence_total'] for name in display}}), flush=True)


if __name__ == '__main__': main()
