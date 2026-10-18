"""Fixed-configuration, chronological feature ablations of a completed run.

This is post-hoc diagnosis, not a new model-selection round. All variants are
refit from scratch; a removed feature stays absent during learning and scoring.
The deployed model and completed source experiment are never modified.
"""
import argparse
import ast
from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone
import gzip
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import sys
import evaluation_contract as scoring
import time
from unittest.mock import patch


FAMILIES = ('browser', 'finder', 'qspace', 'desktop_windows', 'running_apps')
PRIVATE_ROOT = Path.home()/'Library/Application Support/SmartApp/training'
VARIANTS = (
    'full', 'inventory_only', 'time_and_inventory',
    'without_current_features', 'without_semantics', 'without_interactions',
    'without_frequency', 'without_recency', 'without_dwell', 'without_coverage',
    'without_recent5m', 'without_recent60m', 'without_long24h',
    'without_recorded_history', 'without_recent_vs_history',
    'inventory_without_browser', 'inventory_without_finder',
    'inventory_without_qspace', 'inventory_without_desktop_windows',
    'inventory_without_running_apps',
)
LABELS = {
    'full': '当前＋时间背景', 'inventory_only': '当前＋后台状态',
    'time_and_inventory': '当前＋时间背景＋后台状态',
    'without_current_features': '学习分支的当前目标、所属应用、类型、语义及交互',
    'without_semantics': '当前目标语义标签', 'without_interactions': '当前×背景交互项',
    'without_frequency': '访问频次及其衍生统计', 'without_recency': '最近访问衰减',
    'without_dwell': '非空闲停留及其覆盖比例', 'without_coverage': '时间背景覆盖标记',
    'without_recent5m': '5 分钟背景及短长差异',
    'without_recent60m': '60 分钟背景', 'without_long24h': '24 小时背景',
    'without_recorded_history': '累计背景及短长差异',
    'without_recent_vs_history': '5 分钟与累计频次的差异',
    'inventory_without_browser': '后台浏览器状态',
    'inventory_without_finder': '后台 Finder 状态',
    'inventory_without_qspace': '后台 QSpace 状态',
    'inventory_without_desktop_windows': '后台普通窗口状态',
    'inventory_without_running_apps': '后台运行应用状态',
}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n')
    path.chmod(0o600)


def checked_run(path):
    hashes = json.loads((path/'artifact-hashes.json').read_text())
    for name, digest in hashes.items():
        if Path(name).name != name or sha(path/name) != digest:
            raise ValueError('Completed run artifact changed: '+name)
    run = json.loads((path/'run.json').read_text())
    return run, hashes


def freeze_dependencies(experiment, dataset, destination):
    """Prefer the experiment's actual sources, then dataset sources, then imports.

    Some old runs did not bundle transitive helpers. Freeze any missing helpers
    too; exact baseline and full-model parity are mandatory before interpretation.
    """
    local = Path(__file__).parent
    pending = ['compare_multibackground', 'multibackground_model']
    copied = {}
    destination.mkdir(mode=0o700, exist_ok=True)
    while pending:
        name = pending.pop()
        if name in copied:
            continue
        choices = (experiment/('source-'+name+'.py'),
                   dataset/('source-'+name+'.py'), local/(name+'.py'))
        source = next((p for p in choices if p.is_file()), None)
        if source is None:
            continue  # standard library / installed numerical dependency
        payload = source.read_bytes()
        target = destination/(name+'.py')
        target.write_bytes(payload)
        target.chmod(0o600)
        copied[name] = {'source': str(source), 'sha256': sha(source)}
        for node in ast.walk(ast.parse(payload)):
            if isinstance(node, ast.Import):
                pending.extend(n.name.split('.')[0] for n in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                pending.append(node.module.split('.')[0])
    return copied


def inventory_features(q, excluded=()):
    if q['inventory_through_index'] != q['index']:
        raise ValueError('Inventory must be frozen at the prediction cutoff')
    state, quality = q['inventory_state'], q['inventory_quality']
    result = {}
    # Five independent source families share a fixed total content budget of 4,
    # matching four temporal scales. Missing families do not boost other sources.
    for family in FAMILIES:
        if family in excluded:
            continue
        prefix = 'inventory/'+family+'/'
        reliability = quality.get(family, 0.)
        if not math.isfinite(reliability) or not 0 <= reliability <= 1:
            raise ValueError('Invalid inventory source quality')
        if family not in state or reliability <= 0:
            result[prefix+'unavailable'] = .1
            continue
        positive = {k: v for k, v in state[family].items()
                    if math.isfinite(v) and v > 0}
        mass = sum(positive.values())
        if mass:
            result.update({prefix+'token/'+k: .8*reliability*v/mass
                           for k, v in positive.items()})
        else:
            result[prefix+'observed_empty'] = .1
        result[prefix+'quality'] = .1*reliability
    return result


def remove_background_feature(key, variant):
    parts = key.split('/')
    if variant == 'without_frequency':
        return (parts[0] == 'recent_vs_history' or
                len(parts) > 1 and parts[1] in {'count', 'observed_count', 'log_arrivals'})
    if variant == 'without_recency':
        return len(parts) > 1 and parts[1] in {'recent', 'observed_recent'}
    if variant == 'without_dwell':
        return len(parts) > 1 and parts[1] in {'dwell', 'observed_dwell', 'observed_nonidle_fraction'}
    if variant == 'without_coverage':
        return (len(parts) > 1 and
                (parts[1].startswith('observed_') or parts[1] == 'left_truncated'))
    if variant.startswith('without_'):
        scale = variant.removeprefix('without_')
        if parts[0] == scale:
            return True
        # The cross-scale contrast cannot retain a removed scale's information.
        if scale in {'recent5m', 'recorded_history'} and parts[0] == 'recent_vs_history':
            return True
    return False


def diagnostic_features(q, config, base_features):
    variant = config['diagnostic_variant']
    if variant not in VARIANTS:
        raise ValueError('Unknown diagnostic variant')
    base_config = {k: v for k, v in config.items() if k != 'diagnostic_variant'}
    if variant == 'inventory_only':
        base_config['backgrounds'] = 'none'
    original = base_features(q, base_config)
    # Recreate the exact original interaction namespace after filtering raw
    # groups, so removed information cannot leak through an interaction feature.
    foreground = {k: v for k, v in original.items()
                  if not k.startswith(('background/', 'interaction/'))}
    background = {k.removeprefix('background/'): v for k, v in original.items()
                  if k.startswith('background/')}
    background = {k: v for k, v in background.items()
                  if not remove_background_feature(k, variant)}
    if variant in {'inventory_only', 'time_and_inventory'} or variant.startswith('inventory_without_'):
        excluded = (variant.removeprefix('inventory_without_'),) if variant.startswith('inventory_without_') else ()
        background.update(inventory_features(q, excluded))
    if variant == 'without_semantics':
        foreground = {k: v for k, v in foreground.items() if not k.startswith('semantic=')}
    if variant == 'without_current_features':
        foreground = {}
    result = {**foreground, **{'background/'+k: v for k, v in background.items() if v}}
    if variant not in {'without_current_features', 'without_interactions'}:
        result.update({'interaction/'+q['origin']+'/'+k: .5*v
                       for k, v in background.items() if v})
    return result


paired = scoring.paired_topk


def cohort_rows(data):
    rows = data['evaluation']
    return {
        'old_evidence': [r for r in rows if r['evidence_grade'] == 'original'],
        'old_late_evidence': [r for r in rows if r['split'] == 'test' and r['evidence_grade'] == 'original'],
        'new_all': [r for r in rows if r['split'] == 'new_test'],
        'new_evidence': [r for r in rows if r['split'] == 'new_test' and r.get('new_exact_history')],
        'new_expanded': [r for r in rows if r['split'] == 'new_test' and r['evidence_grade'] != 'unsupported'],
        'old_all': [r for r in rows if r['split'] != 'new_test'],
    }


def asof_homes(snapshot, observations, core, entry):
    by_index = {q['index']: q for q in observations}
    catalog, homes = {}, {}
    with (snapshot/'events.jsonl').open() as f:
        for index, line in enumerate(f):
            event = json.loads(line)
            if event['kind'] == 'target.update':
                target = event['data']['target']
                if core.admitted(target) and not core.synthetic(target):
                    catalog[target['id']] = target
                else:
                    catalog.pop(target['id'], None)
            if index in by_index:
                q = by_index[index]
                if q['anchor_id'] != event['id'] or q['session'] != event['sessionID']:
                    raise ValueError('Prediction input anchor changed')
                homes[q['position']] = next((k for k, t in sorted(catalog.items())
                    if t['kind'] == 'page' and t.get('locator') == entry), None)
    if len(homes) != len(observations):
        raise ValueError('Missing as-of home entry')
    return homes, index


def build_report(result):
    lines = ['# 时间背景、后台状态及输入贡献诊断', '',
        '固定上轮早期验证选定的交互 softmax、学习率 0.1、种子与训练顺序。所有新变体从头按时间重训；标签仅在成熟后进入学习。前台序列长度保持 1。',
        '本轮是看过测试结果后的解释性分析，没有据此重新选模型或部署。统一审查评测合法性，采用服务接受主分并保留精确诊断。', '',
        '## 后台与时间背景的同架构比较', '',
        '后台状态直接使用原冻结数据内截至截点已观察的应用、普通窗口、浏览器、Finder 和 QSpace 状态；按来源归一化并保留缺失/空/质量标记。后台内容总预算固定为 4，与四个时间尺度相当，未调参。该表示方式与预算本身仍会影响结果。',
        '下表统一使用 50% 学习分支＋50% 原统计分支；融合系数不是贡献率。', '',
        '| 输入 | 旧合法有证据 Top1 / Top4 | 旧较晚合法有证据 Top1 / Top4 | 新增合法全部 Top1 / Top4 |',
        '|---|---:|---:|---:|']
    for name in ('baseline', 'current_only', 'recent_only', 'long_only', 'full', 'inventory_only', 'time_and_inventory'):
        label = {'baseline': '旧统计模型', 'current_only': '仅当前目标',
                 'recent_only': '当前＋近期', 'long_only': '当前＋长期'}.get(name, LABELS.get(name, name))
        cells = []
        for cohort in ('old_evidence', 'old_late_evidence', 'new_all'):
            m = result['metrics'][name]['mix50'][cohort]['all']
            cells.append(f"{m['top1_hits']} / {m['top4_hits']}")
        lines.append('| '+label+' | '+' | '.join(cells)+' |')
    lines += ['', '## 各项的条件增益', '',
        '每次移除一项后重新训练，其余配置固定。“贡献”定义为完整输入减去移除该项的结果，单位为净增加的命中数或百分点；同时列出修正/损失。不同项互相重叠，不能相加为 100%，也不是用户行为的因果归因。',
        '移除频次时连同短长频次差异与到达总量移除；移除 5 分钟或累计背景时也移除两者的差异。移除停留时同时移除停留覆盖比例。',
        '当前目标与语义的消融只作用于学习分支；旧统计分支、训练去重和固定输出语义仍保留当前目标，因此不代表整个系统的当前目标总贡献。', '',
        '| 移除项 | 旧有证据净 Top1 / Top4 | 新增全部净 Top1 / Top4 | 新增 Top4 修正 / 损失 |',
        '|---|---:|---:|---:|']
    for variant in VARIANTS:
        if variant in {'full', 'inventory_only', 'time_and_inventory'}:
            continue
        pair = result['contributions'][variant]['mix50']
        old, new = pair['old_evidence'], pair['new_all']
        lines.append(f"| {LABELS[variant]} | {old['top1']['net_hits']:+d} / {old['top4']['net_hits']:+d} | {new['top1']['net_hits']:+d} / {new['top4']['net_hits']:+d} | {new['top4']['helped']} / {new['top4']['hurt']} |")
    lines += ['', '后台各来源的消融以“当前＋时间＋后台”为完整组；其他项以“当前＋时间”为完整组。完整 Top1/Top4 百分比、返回覆盖、精确目标可输出覆盖和纯学习分支结果保存在 results.json。', '',
        '## 解释与复核', '',
        '- 相同命中率可能来自不同样本的相互抵消；逐样本的修正与损失保存在 paired-comparisons.json。',
        '- 只测试已固定的一个架构与一个随机种子，不把单次小样本差异声称为普遍、稳定或显著提升；时间背景最多约 5.35 天。',
        '- 有证据分组依据真实后续目标，是诊断分组，不是线上置信保证。',
        '- 服务级结果单独保存；本报告展示统一接受命中。哈希特征共享、归一化与输出限制均是这套实现的一部分，不能把系数当作贡献百分比。',
        '- 保留原统计分支的融合可能掩盖学习分支的差异；因此同时保存纯学习结果。',
        '- 输入只取原冻结截点，后台库存不限制标签类别或输出范围，未输入近期有序序列；原始采集、运行模型及上轮产物均未修改。',
        '- 原统计及完整时间模型均与上一轮全部逐样本预测和可输出集合精确一致；每种消融的样本、成熟时间及优化器更新次数一致。', '']
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment', type=Path, required=True)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    out = args.output.expanduser().resolve()
    if not out.is_relative_to(PRIVATE_ROOT.resolve()) or out == PRIVATE_ROOT.resolve() or out.exists():
        parser.error('Use a new private training run')
    experiment = args.experiment.expanduser().resolve()
    dataset = args.dataset.expanduser().resolve()
    exp_run, exp_hashes = checked_run(experiment)
    data_run, data_hashes = checked_run(dataset)
    if Path(exp_run['dataset']).resolve() != dataset:
        raise ValueError('Experiment and dataset differ')
    snapshot = Path(exp_run['snapshot'])
    if sha(snapshot/'events.jsonl') != exp_run['events_sha256']:
        raise ValueError('Raw snapshot changed')
    os.umask(0o077)
    sys.dont_write_bytecode = True
    out.mkdir(mode=0o700)
    sources = freeze_dependencies(experiment, dataset, out)
    sys.path.insert(0, str(out))
    comparison = importlib.import_module('compare_multibackground')
    multi = importlib.import_module('multibackground_model')
    summary = json.loads((experiment/'summary.json').read_text())
    config = deepcopy(summary['selected_combined']['config'])
    if config['family'] != 'interaction_softmax' or config['backgrounds'] != 'combined':
        raise ValueError('This diagnostic requires the selected interaction model')
    with gzip.open(dataset/'dataset-private.json.gz', 'rt') as f:
        data = json.load(f)
    data['evaluation'] = scoring.prepare_evaluation(data, out)
    with gzip.open(experiment/'inputs-private.json.gz', 'rt') as f:
        inputs = json.load(f)
    previous = {name: {int(pos): row for pos, row in rows.items()} for name, rows in
                json.loads((experiment/'predictions-private.json').read_text()).items()}
    if len(inputs) != len(data['observations']):
        raise ValueError('Input count changed')
    for q, observation in zip(inputs, data['observations']):
        if any(q[k] != observation[k] for k in ('position', 'index', 'origin', 'target', 'annotation')):
            raise ValueError('Frozen input changed')
        q.update(inventory_state=observation['background_state'],
                 inventory_quality=observation['source_quality'],
                 inventory_through_index=observation['index'])
    homes, asof_index = asof_homes(snapshot, data['observations'], comparison.io, multi.ENTRY)
    stream = [r for r in comparison.training_rows(data, 'hard_kind') if r['head'] == 0]
    cohorts = cohort_rows(data)
    plan = {'purpose': 'post-hoc fixed-config attribution; no selection or deployment',
        'configuration': config, 'variants': list(VARIANTS),
        'source_dependencies': sources, 'dataset_hashes': data_hashes,
        'experiment_hashes': exp_hashes, 'inventory_family_content_budget': .8,
        'refit_each_variant': True, 'current_sequence_length': 1,
        'ordered_past_features': False, 'background_inventory_limits_output': False,
        'cohorts': {k: [r['position'] for r in v] for k, v in cohorts.items()}}
    write_json(out/'plan.json', plan)  # fixed before any new scores are computed
    base_scores, base_outputs = comparison.replay_baseline(inputs, stream, data['evaluation'], homes)
    if base_outputs != previous['current_statistics']:
        raise ValueError('Old baseline parity failed')
    outputs = {'baseline': {1.: base_outputs, .5: base_outputs}}
    for name, mode in [('current_only', 'none'), ('recent_only', 'recent'), ('long_only', 'long')]:
        outputs[name] = {w: previous[f"interaction_softmax_{mode}_rich_lr{config['lr']}_{suffix}"]
                         for w, suffix in [(1., 'pure'), (.5, 'mix50')]}
    expected = {w: previous[config['id'].rsplit('_', 1)[0]+'_'+suffix]
                for w, suffix in [(1., 'pure'), (.5, 'mix50')]}
    base_features = multi.features
    def features(q, c):
        return diagnostic_features(q, c, base_features)
    training = {}
    print(json.dumps({'phase': 'diagnosis_started', 'variants': len(VARIANTS),
                      'evaluation_n': len(data['evaluation']), 'baseline_parity': True}), flush=True)
    for variant in VARIANTS:
        c = {**config, 'diagnostic_variant': variant}
        if variant == 'full' and any(features(q, c) != base_features(q, config) for q in inputs):
            raise ValueError('Full feature parity failed')
        started = time.perf_counter()
        with patch.object(multi, 'features', features):
            model, predictions, _ = comparison.replay_joint(inputs, stream, data['evaluation'],
                homes, base_scores, c, asof_index)
        if variant == 'full' and predictions != expected:
            raise ValueError('Full model chronological prediction parity failed')
        updates = {'records': len(model.training_rows), 'updates': model.classifier.update_count,
                   'classes': len(model.classifier.labels)}
        if training and updates != training['full']:
            raise ValueError('Ablation changed training membership')
        training[variant] = updates
        outputs[variant] = predictions
        write_json(out/('predictions-'+variant+'-private.json'), predictions)
        print(json.dumps({'finished': variant, 'seconds': round(time.perf_counter()-started, 2),
                          'training': updates}), flush=True)
    metrics = {}
    for name, distributions in outputs.items():
        metrics[name] = {}
        for weight, predictions in distributions.items():
            key = 'pure' if weight == 1. else 'mix50'
            metrics[name][key] = {cohort: scoring.grouped_metrics(rows, predictions) for cohort, rows in cohorts.items()}
            metrics[name][key+'_exact'] = {cohort: {g: m['exact'] for g, m in scoring.grouped_metrics(rows, predictions).items()}
                                          for cohort, rows in cohorts.items()}
    contributions = {}
    for variant in VARIANTS:
        if variant in {'full', 'inventory_only', 'time_and_inventory'}:
            continue
        full = 'time_and_inventory' if variant.startswith('inventory_without_') else 'full'
        contributions[variant] = {('pure' if w == 1. else 'mix50'):
            {cohort: paired(rows, outputs[variant][w], outputs[full][w])
             for cohort, rows in cohorts.items()} for w in (1., .5)}
    pairs = {}
    for reference, alternative in [('baseline', 'full'), ('current_only', 'full'),
        ('current_only', 'inventory_only'), ('inventory_only', 'full'),
        ('full', 'time_and_inventory'), ('inventory_only', 'time_and_inventory'),
        ('current_only', 'recent_only'), ('current_only', 'long_only')]:
        pairs[reference+'__to__'+alternative] = {('pure' if w == 1. else 'mix50'):
            {cohort: paired(rows, outputs[reference][w], outputs[alternative][w])
             for cohort, rows in cohorts.items()} for w in (1., .5)}
    result = {'config': config, 'metrics': metrics, 'contributions': contributions,
              'training': training, 'baseline_parity': True, 'full_model_parity': True,
              'interpretation': 'conditional refit performance difference, not additive causal shares',
              'recorded_days': summary['recorded_days']}
    write_json(out/'results.json', result)
    write_json(out/'paired-comparisons.json', pairs)
    (out/'report-private.md').write_text(build_report(result))
    script = Path(__file__).read_bytes()
    (out/('source-'+Path(__file__).name)).write_bytes(script)
    for source in sources.values():
        if sha(Path(source['source'])) != source['sha256']:
            raise ValueError('A source changed during replay')
    if checked_run(experiment)[1] != exp_hashes or checked_run(dataset)[1] != data_hashes:
        raise ValueError('Source experiment changed')
    if sha(snapshot/'events.jsonl') != exp_run['events_sha256']:
        raise ValueError('Raw snapshot changed')
    write_json(out/'run.json', {'complete': True, 'created_at': datetime.now(timezone.utc).isoformat(),
        'experiment': str(experiment), 'dataset': str(dataset), 'snapshot': str(snapshot),
        'source_dependencies': sources, 'experiment_hashes': exp_hashes, 'dataset_hashes': data_hashes,
        'collection_or_runtime_changed': False, 'raw_modified': False, 'model_selection': False,
        'baseline_and_full_model_parity': True, 'source_hashes_verified_before_after': True})
    hashes = {str(p.relative_to(out)): sha(p) for p in sorted(out.rglob('*')) if p.is_file()}
    write_json(out/'artifact-hashes.json', hashes)
    for path in out.rglob('*'):
        if path.stat().st_mode & 0o777 != (0o700 if path.is_dir() else 0o600):
            raise ValueError('Private artifact permissions changed')
    print(json.dumps({'complete': True, 'output': str(out), 'variants': len(VARIANTS)}), flush=True)


if __name__ == '__main__':
    main()
