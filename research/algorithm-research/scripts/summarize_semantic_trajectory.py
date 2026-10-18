"""Replay selected architectures and produce a private Chinese experiment report."""
import argparse
from collections import Counter, defaultdict
import gzip
import json
import os
from pathlib import Path

import evaluate_context_knn as io
from compare_semantic_trajectory import replay, evaluate, groups, training_rows, select_single
from semantic_online import SemanticStatistics, rank, blend
from semantic_online_classifier import build
from semantic_annotations import compare_annotations
from summarize_tuned_history import verify_run
from evaluate_evidence_model import DENIED_BUNDLES


def reconstruct(data, config, baseline):
    base = {int(pos): {label:1./(20+i) for i,label in enumerate(row['raw_ranking'], 1)}
            for pos,row in baseline.items()}
    if config['family'] == 'historical_unrestricted_control':
        distributions = {r['position']: {h:base[r['position']] for h in range(4)} for r in data['evaluation']}
    else:
        if config['family'] == 'semantic_statistics':
            model = SemanticStatistics(**{k:config[k] for k in
                ('semantic_strength','global_strength','half_life_hours','recency_mix')})
        else:
            model = build(config)
        distributions = replay(data, config['policy'], model)
        mix = config.get('baseline_rrf_mix', 0.)
        if mix:
            distributions = {pos: {h:blend([
                {label:1./(20+i) for i,label in enumerate(rank(scores, len(scores)), 1)}, base[pos]],
                [1-mix,mix]) for h,scores in heads.items()} for pos,heads in distributions.items()}
    scores, predictions = evaluate(data, distributions, config['future_mix'])
    return scores, predictions, distributions


def pct(value):
    return '—' if value is None else '%.1f%%' % (100*value)


def cell(metric):
    return '%s/%s（%s）' % (metric['top4_hits'], metric['n'], pct(metric['top4']))


def config_text(config):
    return json.dumps(config, ensure_ascii=False, sort_keys=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    private_root = Path.home()/'Library/Application Support/SmartApp/training'
    if private_root.resolve() not in args.output.resolve().parents:
        raise ValueError('Private training output required')
    provenance, source_hashes = verify_run(args.run)
    dataset = Path(provenance['dataset'])
    if verify_run(dataset)[1] != provenance['dataset_hashes']:
        raise ValueError('Dataset provenance changed')
    with gzip.open(dataset/'dataset-private.json.gz', 'rt') as f:
        data = json.load(f)
    descriptive = json.loads((dataset/'descriptive-private.json').read_text())
    denied_keys = {key for key,meta in descriptive['target_metadata'].items()
                   if (meta.get('bundleID') or meta.get('_bundle') or
                       (meta.get('id','')[4:] if meta.get('kind') == 'app' else None)) in DENIED_BUNDLES}
    descriptive_exclusions = {}
    for field in ('independent_arrivals_by_kind','background_inventory_observations_by_kind'):
        descriptive_exclusions[field] = sum(count for counts in descriptive[field].values()
                                            for key,count in counts.items() if key in denied_keys)
        descriptive[field] = {kind:{key:count for key,count in counts.items() if key not in denied_keys}
                              for kind,counts in descriptive[field].items()}
    descriptive['target_metadata'] = {key:meta for key,meta in descriptive['target_metadata'].items() if key not in denied_keys}
    descriptive['legacy_ineligible_excluded_counts'] = descriptive_exclusions
    baseline_path = Path(provenance['baseline_run'])
    if verify_run(baseline_path)[1] != provenance['baseline_hashes']:
        raise ValueError('Historical control changed')
    baseline = json.loads((baseline_path/'predictions-private.json').read_text())
    selected = json.loads((args.run/'selection.json').read_text())
    cached = json.loads((args.run/'selected-predictions-private.json').read_text())
    results = json.loads((args.run/'results.json').read_text())
    verified, distributions = {}, {}
    for objective, winner in selected.items():
        scores, predictions, distributions[objective] = reconstruct(data, winner['config'], baseline)
        if json.loads(json.dumps(scores)) != winner['scores'] or json.loads(json.dumps(predictions)) != cached[objective]:
            raise ValueError('Independent direct replay disagrees with cached search result')
        verified[objective] = len(predictions)
    group_rows = groups(data['evaluation'])
    truth_alignment = {}
    for name,rows in group_rows.items():
        alignment = Counter()
        for row in rows:
            labels = data['future'][str(row['position'])]['labels']
            alignment['missing' if not labels else 'same' if labels[0]['action_key'] == row['truth_action'] else 'different'] += 1
        truth_alignment[name] = dict(alignment)
    control = next(r for r in results if r['config']['family'] == 'historical_unrestricted_control')
    single, future = selected['single'], selected['future']
    methods = defaultdict(list)
    for result in results:
        label = result['config']['family'] + ('+旧模型融合' if result['config']['baseline_rrf_mix'] else '')
        methods[label].append(result)
    method_best = {k:max(v, key=select_single) for k,v in methods.items()}
    stats = max((r for r in results if r['config']['family'] == 'semantic_statistics'
                 and r['config']['baseline_rrf_mix'] == 0.), key=select_single)
    reference = stats['config']
    cleaning_ablation, semantic_ablation = [], []
    for result in results:
        c = result['config']
        if c['family'] != 'semantic_statistics' or c['baseline_rrf_mix']:
            continue
        if all(c[k] == reference[k] for k in reference if k != 'policy'):
            cleaning_ablation.append(result)
        if all(c[k] == reference[k] for k in reference if k != 'semantic_strength'):
            semantic_ablation.append(result)
    stream = training_rows(data, single['config']['policy'])
    old_stream = training_rows(data, 'original_cleaning') if single['config'].get('baseline_rrf_mix') else []
    errors = []
    for row in group_rows['expanded']:
        pos = row['position']; actual = row['truth_action']; q = data['observations'][pos]
        shown = cached['single'][str(pos)]['single']
        if actual in shown:
            continue
        current_component = [s for s in stream if s['head'] == 0 and s['ready_index'] < q['index']]
        old_component = [s for s in old_stream if s['head'] == 0 and s['ready_index'] < q['index']]
        past = current_component + old_component
        same = [s for s in past if data['observations'][s['position']]['origin'] == q['origin']]
        exact = [s for s in same if s['label'] == actual]
        semantic = [s for s in past if s['label'] == actual and compare_annotations(
            q['annotation'], data['observations'][s['position']]['annotation'])['level'] in ('exact','strong','weak')]
        learned = any(s['label'] == actual for s in past)
        ordering = rank(distributions['single'][pos][0], len(distributions['single'][pos][0]))
        reason = ('not_learned_under_selected_cleaning' if not learned else
                  'exact_origin_evidence_ranked_below_four' if exact else
                  'semantic_evidence_ranked_below_four' if semantic else 'only_other_origin_history_after_cleaning')
        errors.append({'position': pos, 'sample_id': row['sample_id'], 'grade': row['evidence_grade'],
            'current': q['target'], 'truth_action': actual, 'truth_metadata': row['truth_metadata'],
            'prediction': shown, 'rank': ordering.index(actual)+1 if actual in ordering else None,
            'reason': reason, 'exact_support': len({s['outcome_id'] for s in exact}),
            'same_origin_history': len({s['outcome_id'] for s in same}),
            'semantic_support': len({s['outcome_id'] for s in semantic}),
            'semantic_origin_contexts': len({data['observations'][s['position']]['origin'] for s in semantic}),
            'component_actual_support': {
                'new_or_selected_component': len({s['outcome_id'] for s in current_component if s['label'] == actual}),
                'historical_fusion_component': len({s['outcome_id'] for s in old_component if s['label'] == actual})},
            'same_origin_next_counts': dict(Counter(label for label,_ in {(s['label'],s['outcome_id']) for s in same}))})
    ledger_stats = {}
    for policy, ledgers in data['trajectory_ledgers'].items():
        observed = [x for row in ledgers for x in row.get('observed', [])]
        ledger_stats[policy] = {'anchors': len(ledgers), 'observed_supervision_rows': len(observed),
            'dropped_rows': sum(len(row.get('dropped', [])) for row in ledgers),
            'retained_rows': len(data['streams'][policy]),
            'downweighted_rows': sum(r['record']['weight'] < 1 for r in data['streams'][policy]),
            'stop_reasons': dict(Counter(row.get('stop_reason') for row in ledgers)),
            'dwell_status': dict(Counter(x['dwell']['status'] for x in observed))}
    lines = ['# 语义增强、数据清洗与多步预测实验', '',
        '本报告是已多次查看的数据上的开发实验。按历史顺序先预测、后消费严格成熟的标签；参数选择仍使用同一开发评分集，不能当成独立泛化准确率。', '',
        '## 评分与输入边界', '',
        '- 单步保留原 5 秒评分目标及 162 个截点，原有证据的 62 条保持固定；语义新增样本单列。',
        '- 语义关系只丰富起点表示，答案仍核对精确目标。普通不可重开窗口回退到所属应用，网页/目录保留原定位符。',
        '- 当前模型输入仍以当前界面为主；后台状态保留在数据中，本轮没有调入后台相似度。无候选名单、当前库存或同起点硬过滤。',
        '- 增量标签只在原 30 秒观察窗成熟后可用；每个起点/真实到达/预测头去重，不把多个语义标签当成多条行为。',
        '- 证据分组根据实际答案回看已成熟的历史，属于事后诊断，不表示线上能提前保证该样本一定可预测。', '',
        '旧无候选控制模型保留上轮参数：近期访问与在线自适应 Hoeffding 树各占一半的倒数排名融合；已去掉同起点硬门控。新模型与它的融合也只使用预测前已生成的全量历史类别排名。', '',
        '## 语义证据覆盖', '', '| 分组 | 条数 |', '|---|---:|']
    for name, rows in group_rows.items():
        lines.append('| %s | %d |' % (name, len(rows)))
    lines += ['', 'exact 为当前身份/已接受的应用窗口类别支持；strong 为明确课程、项目、仓库或具体目录范围；weak 可由同站点或受范围约束的名称相似形成。仅类型、PDF 扩展名、通用下载角色不能形成新增证据。相似度与强弱阈值是启发式规则，未校准为概率。', '',
              '## 单步结果', '', '| 评分集 | 旧无候选限制模型 Top4 | 本次最优 Top4 | 本次 Top1 |', '|---|---:|---:|---:|']
    for name in ('original','added_exact','semantic_strong','semantic_weak','expanded','late_original','all'):
        m = single['scores'][name]['single']
        lines.append('| %s | %s | %s | %s/%s |' % (name, cell(control['scores'][name]['single']), cell(m), m['top1_hits'], m['n']))
    lines += ['', '本次最优配置：`'+config_text(single['config'])+'`。先比原 62 条 Top4，再比扩展组 Top4，再比 Top1。', '',
              '| 模型家族各自最佳 | 原 62 条 Top4 | 扩展组 Top4 |', '|---|---:|---:|']
    for name,result in method_best.items():
        lines.append('| %s | %s | %s |' % (name,cell(result['scores']['original']['single']),cell(result['scores']['expanded']['single'])))
    lines += ['', '## 清洗与结构分开比较', '',
              '下表固定同一统计模型结构，只替换训练清洗，单步评分答案与成员始终不变。多步评分始终使用 resolved 原始独立到达链。', '',
              '固定结构：`'+config_text({k:v for k,v in reference.items() if k != 'policy'})+'`。', '',
              '| 训练清洗 | 原 62 条 Top4 | 扩展组 Top4 |', '|---|---:|---:|']
    for result in cleaning_ablation:
        lines.append('| %s | %s | %s |' % (result['config']['policy'],cell(result['scores']['original']['single']),cell(result['scores']['expanded']['single'])))
    lines += ['', '- original_cleaning：上轮按类型的硬阈值稳定目标标签；未来头仍用 resolved 独立到达。',
        '- resolved：同次到达补全合并，不按短停留删除。',
        '- hard_kind：确认短于应用/窗口 5 秒、具体资源 2 秒的到达从训练中跳过；未知/删失时长保留。',
        '- soft_kind：短停留按时长比例降低权重，最低 0.1；不直接删除。',
        '- soft_quickcheck：仅利用预测前已完成的同起点快速返回习惯，缓和短停留降权。',
        '- soft_overlap：在 soft_kind 基础上，按输入前 30 秒同起点自然 anchor 的数量降低密集观测权重；不删除真实到达。', '',
        '这几种标签和权重是学习策略，不是认定用户误选。未读取的背景不能当成空，标签观测中断也不当成停留短。', '',
        '| 固定清洗下语义强度 | 原 62 条 Top4 | 扩展组 Top4 |', '|---|---:|---:|']
    for result in semantic_ablation:
        lines.append('| %s | %s | %s |' % (result['config']['semantic_strength'],cell(result['scores']['original']['single']),cell(result['scores']['expanded']['single'])))
    lines += ['', '## 多步结果', '',
        '按原始顺序建立独立的第 1/2/3 步预测头；真实 A→B→A 返回保留为步骤。不足三步的记录屏蔽缺失步。短停留清洗不从第 4 步补齐。',
        '原162条中，resolved 原始第1步与固定5秒清洗单步答案相同 %d 条、不同 %d 条、不可观察 %d 条。因此下方“第1步”分数也不能直接当作上表单步分数的增减；原62条中两种答案不同 %d 条。' %
            (truth_alignment['all'].get('same',0),truth_alignment['all'].get('different',0),truth_alignment['all'].get('missing',0),truth_alignment['original'].get('different',0)),
        '面向菜单的四个目标把各步分布加权融合，再去重排序。指标“未来目标覆盖”按第一/二/三步 1、0.5、0.25 计权，同一目标只计其首次出现；它与单步 Top4 不同。', '',
        '多步沿用扩展的75个起点以固定比较范围；这75条的证据由单步5秒答案定义，不表示其第2/3步也已经各自具备历史支持。完整三步样本更少，必须结合下方人数解读。', '',
        '| 扩展证据组 | 旧单步菜单 | 新多步菜单 |', '|---|---:|---:|']
    a, b = control['scores']['expanded']['future'], future['scores']['expanded']['future']
    for label,key in [('未来目标加权覆盖率','weighted_distinct_target_coverage'), ('未来任一步至少命中一次','any_future_hit_rate'), ('完整三步子集加权覆盖率','full_three_weighted_coverage')]:
        lines.append('| %s | %s | %s |' % (label,pct(a[key]),pct(b[key])))
    lines += ['', '扩展组可观察到未来的样本 %d 条，其中完整观察三步 %d 条。' % (b['observed_future_n'], b['fully_observed_three_n']),
        '多步最佳配置：`'+config_text(future['config'])+'`。', '',
        '| 独立预测头 | 可观察样本数 | Top4 | Top1 |', '|---|---:|---:|---:|']
    for step,m in b['steps'].items():
        lines.append('| 第 %s 步 | %s | %s/%s（%s） | %s/%s |' % (step,m['n'],m['top4_hits'],m['n'],pct(m['top4']),m['top1_hits'],m['n']))
    lines += ['', '## 剩余错误与复核', '',
        '扩展组单步未进前四：%d 条。按可观测的失败位置分组：`%s`。这不是对真实用户意图的因果判断。' % (len(errors),json.dumps(dict(Counter(e['reason'] for e in errors)),ensure_ascii=False)),
        '具体起点、正确目标、排名、同起点支持及相似历史保存在 errors-private.json。',
        '两个入选配置已脱离搜索组件缓存重新完整回放，每个配置 %d 个截点，排序、逐样本预测与分组分数全部一致。' % verified['single'],
        '共比较 %d 个配置（含未来头配比；相同单步配置随配比重复，不代表同样数量的独立模型家族）。所有完整 URL/路径与目标明细仅保存在私有 run。未部署到运行中的推荐菜单。' % len(results), '',
        '## 数据与复现', '',
        '- 数据集：`'+str(dataset)+'`', '- 比较 run：`'+str(args.run)+'`',
        '- 数据包含授权候选字段清理的显式迁移链；176 个旧查询时间及输入已重建核对，不宣称旧事件每个其他字段已获得逐字密码学证明。',
        '- 独立到达、后台库存计数及完整目标元数据见本汇总 descriptive-private.json，二者不能相加。旧抽取排行中的不合格辅助对象已仅在派生报告中排除，原冻结数据集未改写。',
        '- 已有62条原有证据集和新增语义证据的 exact/strong/weak 见 evidence-private.json；计数单位仍是实际历史到达。', '',
        '## 私有目标排行', '', '以下包含个人完整定位符，只可在本机私有报告中审阅。每种类型列前 10 项；后台观察次数是状态被观察的次数，不是独立到达，也不能与到达次数相加。', '']
    for title,field in [('独立到达','independent_arrivals_by_kind'),('后台库存观察','background_inventory_observations_by_kind')]:
        lines += ['### '+title, '', '| 类型 | 目标或完整定位符 | 计数 |','|---|---|---:|']
        for kind in ('app','window','page','directory','file'):
            counts = descriptive[field].get(kind, {})
            if not counts:
                lines.append('| %s | 无记录 | 0 |' % kind)
            for key,count in sorted(counts.items(),key=lambda kv:(-kv[1],kv[0]))[:10]:
                meta = descriptive['target_metadata'].get(key, {})
                display = meta.get('locator') or meta.get('name') or meta.get('bundleID') or key
                display = str(display).replace('|','\\|').replace('\n',' ').replace('`','\\`')
                lines.append('| %s | %s | %s |' % (kind,display,count))
        lines.append('')
    os.umask(0o077); args.output.mkdir(mode=0o700,parents=True,exist_ok=False)
    (args.output/'report-private.md').write_text('\n'.join(lines))
    io.write_json(args.output/'errors-private.json', errors)
    io.write_json(args.output/'cleaning-audit.json', ledger_stats)
    io.write_json(args.output/'descriptive-private.json', descriptive)
    io.write_json(args.output/'summary.json', {'selected': selected, 'method_best': method_best,
        'cleaning_ablation': cleaning_ablation, 'semantic_ablation': semantic_ablation,
        'single_vs_resolved_step1_truth_alignment': truth_alignment,
        'direct_replay_verified': verified, 'error_reasons': dict(Counter(e['reason'] for e in errors))})
    (args.output/'source-summarize_semantic_trajectory.py').write_bytes(Path(__file__).read_bytes())
    io.write_json(args.output/'run.json', {'complete': True, 'source_run': str(args.run),
        'source_hashes': source_hashes, 'direct_replay_verified': verified, 'test_previously_seen': True})
    io.write_json(args.output/'artifact-hashes.json', {p.name:io.sha(p) for p in args.output.iterdir() if p.is_file()})
    print(json.dumps({'verified':verified,'error_reasons':dict(Counter(e['reason'] for e in errors))}))


if __name__ == '__main__':
    main()
