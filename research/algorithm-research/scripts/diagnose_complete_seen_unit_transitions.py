#!/usr/bin/env python3
"""Diagnose next-step accuracy from any current unit seen before one fixed cut.

Units use the canonical current ``origin`` supplied to the predictor: apps,
window classes, concrete pages, directories, and files share one cohort.  This
is not the realized-next-target-conditioned evidence subset.
"""
import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import evaluation_contract as scoring
from diagnose_complete_seen_page_transitions import (
    RUN_VERSION,
    audit_signature,
    digest,
    expected_complete_scope,
    fraction,
    markdown_cell,
    percent,
)
from page_transition_diagnostics import (
    UNIT_KINDS,
    fixed_split_unit_cohorts,
    fixed_split_unit_history,
    source_unit_rows,
)
from review_historical_models import ROOT, Sources, sha, write


UNIT_RUN_VERSION = RUN_VERSION + '-unit-v1'


def unit_catalog(data):
    """Build private, stable display metadata for canonical source units."""
    catalog = {}
    for observation in data['observations']:
        target = observation['target']
        kind, unit_id = target.get('kind'), observation['origin']
        if kind not in UNIT_KINDS:
            continue
        if kind in {'page', 'directory', 'file'}:
            display = target.get('locator') or target.get('name') or unit_id
        elif kind == 'window':
            display = unit_id
        else:
            display = target.get('name') or unit_id
        old = catalog.get(unit_id)
        if old:
            if old['kind'] != kind:
                raise ValueError('One canonical unit has conflicting target kinds')
            old['display_variants'].add(display)
            if target.get('bundleID'):
                old['bundle_ids'].add(target['bundleID'])
        else:
            catalog[unit_id] = {'unit_id': unit_id, 'kind': kind,
                                'display_variants': {display},
                                'bundle_ids': {target['bundleID']} if target.get('bundleID') else set()}
    for value in catalog.values():
        value['display_variants'] = sorted(value['display_variants'])
        value['display'] = min(value['display_variants'], key=lambda item: (len(item), item))
        value['bundle_ids'] = sorted(value['bundle_ids'])
    return catalog


def score(metric):
    value = metric['rank_score']['mean']
    return f'{100*value:.2f}' if value is not None else '—'


def render(dataset, prediction, model, boundary, metrics, per_unit, source_units, groups):
    order = (
        ('all_complete_validation', '完整固定验证段（全部单位）'),
        ('current_unit_seen_before_split', '当前单位在切点前已出现（主新口径）'),
        ('current_unit_not_seen_before_split', '当前单位在切点前未出现'),
        ('seen_current_unit_arrivals_1', '已见单位：切点前出现 1 次'),
        ('seen_current_unit_arrivals_2_3', '已见单位：切点前出现 2–3 次'),
        ('seen_current_unit_arrivals_4_plus', '已见单位：切点前出现 4+ 次'),
        ('seen_current_unit_kind_app', '已见单位：应用'),
        ('seen_current_unit_kind_window', '已见单位：应用窗口类别'),
        ('seen_current_unit_kind_page', '已见单位：具体网页'),
        ('seen_current_unit_kind_directory', '已见单位：目录'),
        ('seen_current_unit_kind_file', '已见单位：文件'),
        ('seen_current_unit_to_page', '已见单位 → 页面（事后目标类型细分）'),
    )
    lines = [
        '# 已见当前单位 → 下一步：完整固定验证段诊断',
        '',
        f'本报告复用完整验证数据 `{dataset}` 与冻结预测 `{prediction}` 的同一固定时间切点 **{boundary}**。完整后段候选为 277 条，统一合法性审查后为 270 条；模型、预测、标签、清洗与评分契约均不变。',
        '',
        f'评分模型：`{model}`。单位是模型在预测截点使用的当前来源标识，覆盖应用、应用窗口类别、具体网页、目录和文件。主新口径只在完整后段中重新分组：同一当前单位必须在切点前已有独立确认到达。入组不读取切点后的到达、真实下一跳、历史同起点转移是否命中、候选库存或模型输出。',
        '',
        '## 汇总',
        '',
        '| 分组 | 候选 / 合法 | 接受 Top1 | 接受 Top4 | 精确 Top1 | 精确 Top4 | 输出覆盖 | 综合分 |',
        '|---|---:|---:|---:|---:|---:|---:|---:|',
    ]
    for key, label in order:
        metric = metrics[key]
        lines.append(
            f"| {label} | {metric['audit']['candidate_n']} / {metric['n']} | "
            f"{fraction(metric, 'accepted', 1)} | {fraction(metric, 'accepted', 4)} | "
            f"{fraction(metric, 'exact', 1)} | {fraction(metric, 'exact', 4)} | "
            f"{percent(metric['prediction_coverage'])} | {score(metric)} |"
        )
    lines += [
        '',
        '“已见”仅指当前来源单位在切点前发生过，不代表其下一跳曾重复出现；因此它不等于按真实下一跳历史支持定义的 `evidence_total`。窗口使用模型输入的稳定窗口类别，不把瞬态窗口实例当作跨时段可重复单位。最后一行的目标是页面仅在结果发生后可知，不能作为预测前入组条件。',
        '',
        '## 按来源单位',
        '',
        '下表只包含主新口径中的来源单位。具体网页与目录/文件定位符仅保留在此私有 run；样本数很小时只作诊断，不作为单位级稳定结论。',
        '',
        '| 来源单位 | 类别 | 切点前到达次数 | 验证样本 | 接受 Top1 | 接受 Top4 | 精确 Top1 | 精确 Top4 | 综合分 |',
        '|---|---|---:|---:|---:|---:|---:|---:|---:|',
    ]
    for value in per_unit:
        metric = value['metrics']
        lines.append(
            f"| `{markdown_cell(value['display'])}` | {value['kind']} | {value['pre_split_arrivals']} | {metric['n']} | "
            f"{fraction(metric, 'accepted', 1)} | {fraction(metric, 'accepted', 4)} | "
            f"{fraction(metric, 'exact', 1)} | {fraction(metric, 'exact', 4)} | "
            f"{score(metric)} |"
        )
    lines += [
        '',
        '## 固定性检查',
        '',
        f"- 固定后段成员：{len(groups['all_complete_validation'])} 条候选；主新口径：{len(groups['current_unit_seen_before_split'])} 条。",
        f'- 主新口径包含 {len(source_units)} 个不同来源单位；逐单位成员、私有定位符、切点前到达计数和完整指标已写入 `per-unit-private.json`。',
        '- 评分复算后的完整后段 comparison key 与冻结模型报告的 `statistics/none` 全量评分一致；这证明分割、合法性审查、预测与评分口径没有被重置。',
        '',
    ]
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset-run', type=Path, required=True)
    parser.add_argument('--prediction-run', type=Path, required=True)
    parser.add_argument('--model', default='statistics/none')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    dataset, prediction, output = (args.dataset_run.resolve(), args.prediction_run.resolve(), args.output.resolve())
    if output.exists() or output == ROOT.resolve() or not output.is_relative_to(ROOT.resolve()):
        parser.error('Use a new private training directory')

    sources = Sources()
    data = sources.read(dataset/'dataset-private.json.gz')
    dataset_plan = sources.read(dataset/'plan.json')
    dataset_audit = sources.read(dataset/'evaluation-audit-private.json')
    prediction_plan = sources.read(prediction/'plan.json')
    prediction_audit = sources.read(prediction/'evaluation-audit-private.json')
    frozen_metrics = sources.read(prediction/'metrics-private.json')
    predictions = sources.read(prediction/'predictions-private.json.gz')
    boundary = expected_complete_scope(dataset_plan, prediction_plan)
    if audit_signature(dataset_audit) != audit_signature(prediction_audit):
        raise ValueError('Dataset and prediction run do not have the same immutable validation membership')
    if args.model not in predictions or args.model not in frozen_metrics:
        raise ValueError('Requested frozen model is not present in the prediction run')

    rows = scoring.prepare_evaluation(data)
    rebuilt_audit = scoring.audit_summary(rows)
    if (rebuilt_audit['candidate_n'], rebuilt_audit['eligible_n'], rebuilt_audit['excluded_n']) != (
        dataset_audit['candidate_n'], dataset_audit['eligible_n'], dataset_audit['excluded_n']):
        raise ValueError('Current scoring audit no longer matches the frozen complete-validation membership')
    history = fixed_split_unit_history(data, rows, boundary)
    groups = fixed_split_unit_cohorts(rows, history)
    metrics = {name: scoring.measure(group, predictions[args.model]) for name, group in groups.items()}
    frozen_all = frozen_metrics[args.model].get('all')
    if not frozen_all or metrics['all_complete_validation']['comparison_key'] != frozen_all.get('comparison_key'):
        raise ValueError('Full complete-validation rescore does not match the frozen model result')

    catalog = unit_catalog(data)
    source_units = source_unit_rows(rows, history)
    per_unit = []
    for unit_id, unit_rows in source_units.items():
        details = {history[row['position']]['pre_split_current_unit_arrivals'] for row in unit_rows}
        if len(details) != 1 or unit_id not in catalog:
            raise ValueError('Source-unit membership has inconsistent private metadata')
        per_unit.append({
            **catalog[unit_id],
            'pre_split_arrivals': details.pop(),
            'candidate_positions': [row['position'] for row in unit_rows],
            'metrics': scoring.measure(unit_rows, predictions[args.model]),
        })
    per_unit.sort(key=lambda value: (-value['metrics']['n'], value['unit_id']))

    membership = {
        'boundary_event_index': boundary,
        'complete_validation_candidate_positions': [row['position'] for row in rows],
        'complete_validation_legal_positions': [row['position'] for row in rows if scoring.eligible(row)],
        'groups': {name: [row['position'] for row in group] for name, group in groups.items()},
        'source_units': {unit_id: [row['position'] for row in group] for unit_id, group in source_units.items()},
    }
    os.umask(0o077)
    output.mkdir(mode=0o700)
    write(output/'plan.json', {
        'version': UNIT_RUN_VERSION,
        'dataset_run': str(dataset),
        'prediction_run': str(prediction),
        'model': args.model,
        'evaluation_scope': 'complete_two_cohorts',
        'boundary_event_index': boundary,
        'same_complete_validation_membership': True,
        'candidate_n': dataset_audit['candidate_n'],
        'legal_n': dataset_audit['eligible_n'],
        'unit_definition': 'current canonical origin used by the predictor; app, window class, page, directory, or file',
        'unit_membership_uses_post_split_arrivals': False,
        'unit_membership_uses_realized_next_target': False,
        'unit_to_page_is_after_outcome_target_kind_diagnostic': True,
        'score': 'current unified evaluation_contract; accepted destination plus exact sibling diagnostic and rank_score',
        'membership_sha256': digest(membership),
        'frozen_all_comparison_key': frozen_all['comparison_key'],
        'recomputed_all_comparison_key': metrics['all_complete_validation']['comparison_key'],
    })
    write(output/'metrics-private.json', {'model': args.model, 'groups': metrics})
    write(output/'membership-private.json', membership)
    write(output/'per-unit-private.json', {'model': args.model, 'units': per_unit})
    report = render(dataset, prediction, args.model, boundary, metrics, per_unit, source_units, groups)
    (output/'report-private.md').write_text(report)
    (output/'report-private.md').chmod(0o600)
    for path in (Path(__file__), Path(__file__).with_name('page_transition_diagnostics.py'),
                 Path(__file__).with_name('diagnose_complete_seen_page_transitions.py'),
                 Path(__file__).with_name('evaluation_contract.py')):
        (output/('source-'+path.name)).write_bytes(path.read_bytes())
        (output/('source-'+path.name)).chmod(0o600)
    sources.verify_unchanged()
    write(output/'run.json', {
        'complete': True,
        'created_at': datetime.now(timezone.utc).isoformat(),
        'version': UNIT_RUN_VERSION,
        'dataset_run': str(dataset),
        'prediction_run': str(prediction),
        'model': args.model,
        'boundary_event_index': boundary,
        'source_files': sources.accessed,
        'raw_modified': False,
        'model_or_predictions_modified': False,
    })
    write(output/'artifact-hashes.json', {path.name: sha(path) for path in output.iterdir() if path.is_file()})
    print(json.dumps({
        'output': str(output),
        'model': args.model,
        'boundary_event_index': boundary,
        'all': metrics['all_complete_validation'],
        'seen_current_unit': metrics['current_unit_seen_before_split'],
        'source_unit_n': len(source_units),
    }, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
