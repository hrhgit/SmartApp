"""Tune time-context and inventory-context statistics on frozen early validation."""
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

from diagnose_background_contributions import (PRIVATE_ROOT, asof_homes, checked_run,
    freeze_dependencies, paired, sha, write_json)


LABELS = {'original': '原统计参数／仅当前', 'none': '调参／仅当前',
          'time': '调参／当前＋时间背景', 'inventory': '调参／当前＋后台状态',
          'time_shared_core': '时间背景／通用参数与仅当前组相同',
          'inventory_shared_core': '后台状态／通用参数与仅当前组相同',
          'time_without_context': '时间组所选参数／移除时间背景',
          'inventory_without_context': '后台组所选参数／移除后台状态'}


cohorts = scoring.cohorts


def cell(metric, key):
    return f"{metric[key+'_hits']}/{metric['n']} ({metric[key]:.1%})" if metric['n'] else '—'


def report(summary):
    results = summary['results']
    total = summary['cohort_sizes']['evidence_total']
    all_n = summary['cohort_sizes']['all']
    lines = ['# 时间背景与后台状态：纯统计模型调参对照', '',
        summary['score_scope'],
        '前台序列长度为 1。以下全部是历史条件统计及统计回退，不使用 softmax、神经网络或它们的输出。', '',
        '| 模型 | 有证据合计 Top1 | 有证据合计 Top4 | 返回覆盖 | 接受目标可输出覆盖 |',
        '|---|---:|---:|---:|---:|']
    for name in ('original', 'none', 'time', 'inventory'):
        m = results[name]['accepted']['evidence_total']['all']
        lines.append(f"| {LABELS[name]} | {cell(m,'top1')} | {cell(m,'top4')} | {m['prediction_coverage']:.1%} | {m['rankable_truth_n']}/{m['n']} |")
    lines += ['', '## 选参规则与算法', '',
        '- 使用上轮不可变快照及冻结时间背景、后台状态输入，不追加数据，不重做清洗或改评分标签。仅在截点前已成熟的标签才能进入历史计数。',
        '- 历史记录只按其当时的背景与当前背景比较，不把它的后续目标或未来状态用于相似度。时间背景是 5/60 分钟、24 小时与累计历史的无序统计；没有前一步、有序后缀或候选集合限制。',
        '- 先按当前起点匹配与语义支持分别收集转移票数，再按背景相似度分配票数。背景权重因来源质量而折扣；缺失来源保持中性，已观察为空单独表示。两个支持组保留原证据总量；全局与近期目的地回退继续保留全部已成熟输出类别。',
        '- 核函数为 1 − (1 − floor) × 可比较覆盖 × (1 − 相似度^power)。各尺度/来源先独立归一化，再平均，避免后台对象数量主导。',
        '- 通用网格：语义支持强度 16/4/64；近期目的地回退权重 0.25/0/0.5；历史票数半衰期不衰减/24 小时/168 小时；全局回退强度固定为 1。',
        '- 时间背景的 6 组特征为近期/长期/组合 × 频次/频次＋最近访问衰减＋停留。后台的 6 组特征为全部来源/浏览器与目录/应用与窗口 × 目标身份/完整已观察语义特征。两组各有同样 972 组参数，另有 27 组仅当前对照。背景 floor 取 0.75/0.5/0.25，power 取 1/2。',
        '- 只用旧早期验证中经过统一合法性审查的样本选参（分母见 validation）。优先原证据 Top4，再扩展证据 Top4、原证据 Top1、扩展证据 Top1、全部 Top4、全部 Top1；同分按预先固定网格顺序。选参文件先写入，再计算旧较晚和新增成绩。',
        '- 这批旧、新数据此前都已在研究中查看；本轮虽未拿较晚/新增分数调参，仍不能称为全新独立测试。1971 组配置相对较小验证集有选择过拟合风险，最终优势需要新自然数据确认。', '',
        '## 各组所选参数', '',
        '| 组 | 背景特征 | floor / power | 语义强度 | 近期回退权重 | 历史半衰期（小时，0 为不衰减） | 验证有证据 Top1 / Top4 |',
        '|---|---|---:|---:|---:|---:|---:|']
    for name in ('none', 'time', 'inventory'):
        r = results[name]; c = r['config']; m = r['validation']['original']
        lines.append(f"| {LABELS[name]} | {c['profile'] or '无'} | {c['floor']} / {c['power']} | {c['semantic_strength']} | {c['recency_mix']} | {c['half_life_hours']} | {m['top1_hits']}/{m['n']} / {m['top4_hits']}/{m['n']} |")
    lines += ['', '## 排除通用参数差异的对照', '',
        '背景组独立选参，直接相减会同时包含通用参数的差异。下面固定各组参数、仅移除背景，以及固定为仅当前组所选通用参数后再比较背景。', '',
        '| 模型 | 有证据合计 Top1 | 有证据合计 Top4 |', '|---|---:|---:|']
    for name in ('none', 'time_shared_core', 'inventory_shared_core', 'time', 'time_without_context', 'inventory', 'inventory_without_context'):
        m = results[name]['accepted']['evidence_total']['all']
        lines.append(f"| {LABELS[name]} | {cell(m,'top1')} | {cell(m,'top4')} |")
    lines += ['', '## 时段及证据强弱细分', '',
        '| 范围 | 模型 | Top1 | Top4 |', '|---|---|---:|---:|']
    for cohort, title in [('old_evidence', '旧固定证据'), ('late_old_evidence', '旧较晚证据'),
                           ('new_evidence', '新增证据'), ('expanded_total', '含较弱语义支持的全部证据'),
                           ('new_all', '新增全部（辅助诊断）')]:
        for name in ('original', 'none', 'time', 'inventory'):
            m = results[name]['accepted'][cohort]['all']
            lines.append(f"| {title} | {LABELS[name]} | {cell(m,'top1')} | {cell(m,'top4')} |")
    lines += ['', '## 复核及产物', '',
        '- 逐样本原参数预测、背景关闭对照与旧统计模型核对；精确目标和服务级命中分开保存。',
        '- 总分、逐样本预测、配对修正/损失、全部验证网格、选择文件、统计模型记忆和可重载配置均保存在本私有 run。',
        '- 重载后按原成熟边界回放，核对验证段和新增段预测一致。源快照、原 run 和源码均在前后校验哈希，文件 0600、目录 0700。',
        '- 当前记录约 5.35 天，不代表更长期习惯；来源质量不足与输出层级限制继续保留，不通过删除困难样本提高分数。',
        '- 采集设置、运行中模型及安装包均未改变。', '']
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment', type=Path, required=True)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    out = args.output.expanduser().resolve()
    if out.exists() or out == PRIVATE_ROOT.resolve() or not out.is_relative_to(PRIVATE_ROOT.resolve()):
        parser.error('Use a new private training run')
    experiment, dataset = args.experiment.resolve(), args.dataset.resolve()
    exp_run, exp_hashes = checked_run(experiment)
    _, data_hashes = checked_run(dataset)
    if Path(exp_run['dataset']).resolve() != dataset:
        raise ValueError('Mismatched frozen dataset')
    snapshot = Path(exp_run['snapshot'])
    if sha(snapshot/'events.jsonl') != exp_run['events_sha256']:
        raise ValueError('Raw snapshot changed')
    os.umask(0o077); sys.dont_write_bytecode = True; out.mkdir(mode=0o700)
    dependencies = freeze_dependencies(experiment, dataset, out)
    local_sources = {}
    for name in ('context_statistics.py', 'compare_context_statistics.py',
                 'diagnose_background_contributions.py', 'test_context_statistics.py'):
        source = Path(__file__).with_name(name)
        local_sources[name] = sha(source)
        (out/name).write_bytes(source.read_bytes())
    sys.path.insert(0, str(out))
    cs = importlib.import_module('context_statistics')
    comparison = importlib.import_module('compare_multibackground')
    with gzip.open(dataset/'dataset-private.json.gz', 'rt') as f:
        data = json.load(f)
    with gzip.open(experiment/'inputs-private.json.gz', 'rt') as f:
        inputs = json.load(f)
    for q, observation in zip(inputs, data['observations']):
        if any(q[k] != observation[k] for k in ('position', 'index', 'origin', 'annotation', 'target')):
            raise ValueError('Input identity changed')
        q.update(inventory_state=observation['background_state'], inventory_quality=observation['source_quality'],
                 inventory_through_index=observation['index'])
    homes, asof_index = asof_homes(snapshot, data['observations'], comparison.io, cs.ENTRY)
    stream = [r for r in comparison.training_rows(data, 'hard_kind') if r['head'] == 0]
    training = []
    for row in stream:
        q = inputs[row['position']]
        if q['index'] > row['ready_index'] or row['record']['target_evidence_index'] > row['ready_index']:
            raise ValueError('Invalid evidence maturity')
        training.append({'position': row['position'], 'ready_index': row['ready_index'],
                         'record': comparison.learning_record(row, q)})
    rows = scoring.prepare_evaluation(data, out); parts = cohorts(rows)
    test_index = min(inputs[r['position']]['index'] for r in rows if r['split'] == 'test')
    validation = [r for r in rows if r['split'] == 'validation' and
                  data['observations'][r['position']]['ready_index'] < test_index]
    configs = cs.configurations()
    plan = {'configs': configs, 'counts': dict(Counter(c['mode'] for c in configs)),
        'selection_uses_only_old_early_validation': True, 'validation_positions': [r['position'] for r in validation],
        'current_sequence_length': 1, 'ordered_past_features': False, 'neural_predictions_used': False,
        'training_cleaning_changed': False, 'scoring_contract': scoring.contract(), 'data_added': False,
        'dataset_hashes': data_hashes, 'experiment_hashes': exp_hashes,
        'source_dependencies': dependencies, 'new_source_hashes': local_sources,
        'cohort_positions': {k: [r['position'] for r in v] for k, v in parts.items()}}
    write_json(out/'plan.json', plan)
    background = cs.BackgroundIndex(inputs)
    queries = cs.prepare_queries(inputs, training, validation, homes, background)
    expected = {int(k): v for k, v in json.loads((experiment/'predictions-private.json').read_text())['current_statistics'].items()}
    original = configs[0]
    for pos, query in queries.items():
        if query.predict(original) != expected[pos]:
            raise ValueError('Original validation prediction parity failed at '+str(pos))
    print(json.dumps({'phase': 'validation_search', 'configurations': len(configs),
                      'validation_n': len(validation), 'original_parity': True}), flush=True)
    grid = []
    for i, config in enumerate(configs):
        predictions = {pos: query.predict(config) for pos, query in queries.items()}
        grid.append({'config': config, 'validation': scoring.grouped_metrics(validation, predictions)})
        if (i+1) % 100 == 0:
            print(json.dumps({'validated': i+1, 'total': len(configs)}), flush=True)
    selected = {mode: deepcopy(cs.select_validation(grid, mode)) for mode in ('none', 'time', 'inventory')}
    core = selected['none']['config']
    selected.update({mode+'_shared_core': deepcopy(cs.select_validation(grid, mode, core)) for mode in ('time', 'inventory')})
    selected['original'] = deepcopy(grid[0])
    for mode in ('time', 'inventory'):
        c = selected[mode]['config']
        selected[mode+'_without_context'] = deepcopy(next(r for r in grid if r['config']['mode'] == 'none'
            and all(r['config'][k] == c[k] for k in ('semantic_strength', 'global_strength', 'recency_mix', 'half_life_hours'))))
    write_json(out/'validation-grid.json', grid)
    write_json(out/'selection.json', {'selected': selected, 'late_or_new_scores_used': False})
    print(json.dumps({'phase': 'selection_frozen', 'configs': {k: v['config'] for k, v in selected.items()}}), flush=True)
    # Free search-only component caches before later replay. No later score was inspected above.
    del queries, grid
    background.cache.clear()
    queries = cs.prepare_queries(inputs, training, rows, homes, background)
    outputs = {name: {pos: query.predict(r['config']) for pos, query in queries.items()} for name, r in selected.items()}
    if outputs['original'] != expected:
        mismatch = [pos for pos in expected if outputs['original'][pos] != expected[pos]]
        raise ValueError('Original full prediction parity failed: '+str(mismatch))
    for pos, query in queries.items():
        for mode in ('time', 'inventory'):
            disabled = {**selected[mode]['config'], 'floor': 1.}
            if query.predict(disabled) != outputs[mode+'_without_context'][pos]:
                raise ValueError('Zero background effect changed predictions')
    results = {}
    for name, predictions in outputs.items():
        results[name] = {**selected[name],
            'accepted': {group: scoring.grouped_metrics(members, predictions) for group, members in parts.items()},
            'exact': {group: {g: m['exact'] for g, m in scoring.grouped_metrics(members, predictions).items()} for group, members in parts.items()}}
    pairs = {}
    for before, after in [('original', 'none'), ('none', 'time'), ('none', 'inventory'),
        ('inventory', 'time'), ('time_without_context', 'time'), ('inventory_without_context', 'inventory'),
        ('none', 'time_shared_core'), ('none', 'inventory_shared_core')]:
        pairs[before+'__to__'+after] = {group: scoring.paired_topk(members, outputs[before], outputs[after])
                                      for group, members in parts.items()}
    memory = cs.StatisticalMemory()
    for item in training:
        if item['ready_index'] <= asof_index:
            memory.learn(inputs[item['position']], item['record'], item['ready_index'])
    memory_path = out/'model-memory-private.json.gz'
    with gzip.open(memory_path, 'wt') as f:
        json.dump({'inputs': inputs, 'training': memory.training, 'asof_index': asof_index,
                   'records': len(memory.training), 'navigation_records': len(memory.records)}, f, ensure_ascii=False)
    for mode in ('none', 'time', 'inventory'):
        write_json(out/('model-'+mode+'.json'), {'version': 'conditional-background-statistics-v1',
            'config': selected[mode]['config'], 'memory_file': memory_path.name,
            'memory_sha256': sha(memory_path), 'current_sequence_length': 1, 'deployment': False})
    with gzip.open(memory_path, 'rt') as f:
        restored = json.load(f)
    probe_rows = [validation[0], next(r for r in rows if r['split'] == 'new_test'), rows[-1]]
    restored_background = cs.BackgroundIndex(restored['inputs'])
    reloaded = cs.prepare_queries(restored['inputs'], restored['training'], probe_rows, homes, restored_background)
    for mode in ('none', 'time', 'inventory'):
        manifest = json.loads((out/('model-'+mode+'.json')).read_text())
        if manifest['memory_sha256'] != sha(memory_path):
            raise ValueError('Saved model memory checksum failed')
        for pos, query in reloaded.items():
            if query.predict(manifest['config']) != outputs[mode][pos]:
                raise ValueError('Saved model reload prediction differs')
    summary = {'cohort_sizes': scoring.cohort_sizes(parts), 'scoring_contract': scoring.contract(), 'score_scope': scoring.report_scope(rows), 'results': results,
        'grid_counts': plan['counts'], 'original_prediction_parity': True, 'disabled_context_parity': True,
        'reload_parity': True, 'training_records': len(memory.training), 'navigation_records': len(memory.records),
        'new_data_used_to_choose_config': False, 'collection_or_runtime_changed': False}
    write_json(out/'summary.json', summary)
    write_json(out/'paired-comparisons.json', pairs)
    write_json(out/'predictions-private.json', outputs)
    (out/'report-private.md').write_text(report(summary))
    if checked_run(experiment)[1] != exp_hashes or checked_run(dataset)[1] != data_hashes:
        raise ValueError('Frozen run changed')
    if sha(snapshot/'events.jsonl') != exp_run['events_sha256']:
        raise ValueError('Raw snapshot changed')
    for name, digest in local_sources.items():
        if sha(Path(__file__).with_name(name)) != digest:
            raise ValueError('Research source changed during run')
    for item in dependencies.values():
        if sha(Path(item['source'])) != item['sha256']:
            raise ValueError('Dependency source changed during run')
    write_json(out/'run.json', {'complete': True, 'version': 'conditional-background-statistics-v1',
        'created_at': datetime.now(timezone.utc).isoformat(), 'snapshot': str(snapshot),
        'dataset': str(dataset), 'experiment': str(experiment), 'dataset_hashes': data_hashes,
        'experiment_hashes': exp_hashes, 'raw_sha256': exp_run['events_sha256'],
        'source_dependencies': dependencies, 'new_source_hashes': local_sources,
        'source_hashes_verified_before_after': True, 'model_reload_parity': True,
        'collection_or_runtime_changed': False, 'raw_modified': False})
    write_json(out/'artifact-hashes.json', {p.name: sha(p) for p in sorted(out.iterdir()) if p.is_file()})
    if any(p.stat().st_mode & 0o777 != 0o600 for p in out.iterdir() if p.is_file()):
        raise ValueError('Private artifact permissions changed')
    print(json.dumps({'complete': True, 'output': str(out), 'evidence_n': len(parts['evidence_total']),
        'scores': {name: results[name]['accepted']['evidence_total']['all'] for name in ('original','none','time','inventory')}}), flush=True)


if __name__ == '__main__':
    main()
