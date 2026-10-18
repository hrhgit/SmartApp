#!/usr/bin/env python3
"""Diagnose next-step accuracy from pages seen before the complete-validation cut.

The cohort is not the outcome-conditioned ``evidence_total`` subset.  It first
keeps the exact common complete-validation tail, then selects only rows whose
current exact page had an independently observed arrival before that tail's
fixed time boundary.  Existing predictions are re-scored; no model, label, or
membership is changed.
"""
import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import evaluation_contract as scoring
from page_transition_diagnostics import (
    VERSION,
    fixed_split_page_cohorts,
    fixed_split_page_history,
    source_page_rows,
)
from review_historical_models import ROOT, Sources, sha, write


RUN_VERSION = VERSION + '-complete-fixed-split-v1'


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                     separators=(',', ':')).encode()).hexdigest()


def percent(value):
    return f'{value:.1%}' if value is not None else '—'


def fraction(metric, level, cutoff):
    value = metric if level == 'accepted' else metric['exact']
    return f"{value[f'top{cutoff}_hits']}/{value['n']}（{percent(value[f'top{cutoff}'])}）" if value['n'] else '—'


def markdown_cell(value):
    return str(value or '—').replace('|', '\\|').replace('\n', ' ').replace('`', '\\`')


def audit_signature(audit):
    required = {'candidate_n', 'eligible_n', 'excluded_n', 'membership_sha256', 'rows'}
    if not required <= set(audit):
        raise ValueError('Validation audit lacks immutable membership fields')
    return {
        'candidate_n': audit['candidate_n'],
        'eligible_n': audit['eligible_n'],
        'excluded_n': audit['excluded_n'],
        'membership_sha256': audit['membership_sha256'],
        'rows': [(row['position'], row['sample_id'], row['eligible'], row['reason'], row['row_sha256'])
                 for row in audit['rows']],
    }


def page_catalog(data):
    """Return private, exact locator metadata keyed by page ID."""
    catalog = {}
    for observation in data['observations']:
        target = observation['target']
        if target.get('kind') != 'page':
            continue
        page_id, locator = target['id'], target.get('locator') or target.get('name')
        old = catalog.get(page_id)
        if old and old.get('locator') and locator and old['locator'] != locator:
            raise ValueError('One page identity has conflicting exact locators')
        catalog[page_id] = {'page_id': page_id, 'locator': locator,
                            'origin': target.get('origin'), 'bundle_id': target.get('bundleID')}
    return catalog


def expected_complete_scope(dataset_plan, prediction_plan):
    if dataset_plan.get('evaluation_scope') != 'complete_two_cohorts':
        raise ValueError('Dataset is not the common complete-validation scope')
    if prediction_plan.get('evaluation_scope') != 'complete_two_cohorts':
        raise ValueError('Predictions are not the common complete-validation scope')
    boundary = dataset_plan.get('boundary_event_index')
    if not isinstance(boundary, int):
        raise ValueError('Dataset lacks its fixed validation boundary')
    prediction_boundary = (prediction_plan.get('scope') or {}).get('boundary_event_index')
    if prediction_boundary != boundary:
        raise ValueError('Prediction run uses a different validation boundary')
    return boundary


def render(dataset, prediction, model, boundary, metrics, per_page, source_pages, groups):
    order = (
        ('all_complete_validation', '完整固定验证段（全部）'),
        ('current_exact_page', '当前为具体页面（全部）'),
        ('current_exact_page_seen_before_split', '当前具体页面在切点前已出现（主新口径）'),
        ('current_exact_page_not_seen_before_split', '当前具体页面在切点前未出现'),
        ('seen_current_page_arrivals_1', '已见页面：切点前出现 1 次'),
        ('seen_current_page_arrivals_2_3', '已见页面：切点前出现 2–3 次'),
        ('seen_current_page_arrivals_4_plus', '已见页面：切点前出现 4+ 次'),
        ('seen_current_page_to_page', '已见页面 → 页面（事后目标类型细分）'),
    )
    lines = [
        '# 已见具体页面 → 下一步：完整固定验证段诊断',
        '',
        f'本报告复用完整验证数据 `{dataset}` 与冻结预测 `{prediction}` 的同一固定时间切点 **{boundary}**。完整后段候选为 277 条，统一合法性审查后为 270 条；模型、预测、标签、清洗与评分契约均不变。',
        '',
        f'评分模型：`{model}`。主新口径只在这条完整后段中重新分组：当前目标须为精确页面，且相同页面 ID 在切点前已有独立确认到达。入组不读取切点后的到达、真实下一跳、历史同起点转移是否命中、候选库存或模型输出。',
        '',
        '## 汇总',
        '',
        '| 分组 | 候选 / 合法 | 接受 Top1 | 接受 Top4 | 精确 Top1 | 精确 Top4 | 输出覆盖 | 综合分 |',
        '|---|---:|---:|---:|---:|---:|---:|---:|',
    ]
    for key, label in order:
        metric = metrics[key]
        audit = metric['audit']
        lines.append(
            f"| {label} | {audit['candidate_n']} / {metric['n']} | "
            f"{fraction(metric, 'accepted', 1)} | {fraction(metric, 'accepted', 4)} | "
            f"{fraction(metric, 'exact', 1)} | {fraction(metric, 'exact', 4)} | "
            f"{percent(metric['prediction_coverage'])} | {100*metric['rank_score']['mean']:.2f} |"
        )
    lines += [
        '',
        '“已见”只描述来源页在切点前发生过，不代表其下一跳曾重复出现；因此它与基于真实下一跳定义的 `evidence_total` 互不替代。最后一行的“页面→页面”仅在结果发生后才可知，不能作为预测前的入组条件。',
        '',
        '## 按来源页面',
        '',
        '下表只包含主新口径中的来源页。URL 仅保留在此私有 run；每行样本数很小时只作诊断，不作为页面级稳定结论。',
        '',
        '| 来源页面（精确 locator） | 切点前到达次数 | 验证样本 | 接受 Top1 | 接受 Top4 | 精确 Top1 | 精确 Top4 | 综合分 |',
        '|---|---:|---:|---:|---:|---:|---:|---:|',
    ]
    for value in per_page:
        metric = value['metrics']
        lines.append(
            f"| `{markdown_cell(value['locator'])}` | {value['pre_split_arrivals']} | {metric['n']} | "
            f"{fraction(metric, 'accepted', 1)} | {fraction(metric, 'accepted', 4)} | "
            f"{fraction(metric, 'exact', 1)} | {fraction(metric, 'exact', 4)} | "
            f"{100*metric['rank_score']['mean']:.2f} |"
        )
    lines += [
        '',
        '## 固定性检查',
        '',
        f"- 固定后段成员：{len(groups['all_complete_validation'])} 条候选；页面来源：{len(groups['current_exact_page'])} 条；主新口径：{len(groups['current_exact_page_seen_before_split'])} 条。",
        f'- 主新口径包含 {len(source_pages)} 个不同来源页面；逐页成员、精确 locator、切点前到达计数和完整指标已写入 `per-page-private.json`。',
        '- 评分复算后的完整后段 comparison key 与冻结模型报告的 `statistics/none` 全量评分一致；这证明分割、合法性审查、预测与评分口径没有被重置。',
        '',
    ]
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset-run', type=Path, required=True,
                        help='Complete validation dataset run with the fixed boundary and rows')
    parser.add_argument('--prediction-run', type=Path, required=True,
                        help='Complete validation prediction run on the same dataset')
    parser.add_argument('--model', default='statistics/none',
                        help='Frozen prediction model to score; defaults to the no-context statistics baseline')
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
    history = fixed_split_page_history(data, rows, boundary)
    groups = fixed_split_page_cohorts(rows, history)
    metrics = {name: scoring.measure(group, predictions[args.model]) for name, group in groups.items()}
    frozen_all = frozen_metrics[args.model].get('all')
    if not frozen_all or metrics['all_complete_validation']['comparison_key'] != frozen_all.get('comparison_key'):
        raise ValueError('Full complete-validation rescore does not match the frozen model result')

    catalog = page_catalog(data)
    source_pages = source_page_rows(rows, history)
    per_page = []
    for page_id, page_rows in source_pages.items():
        details = {history[row['position']]['pre_split_current_page_arrivals'] for row in page_rows}
        if len(details) != 1 or page_id not in catalog:
            raise ValueError('Source-page membership has inconsistent private metadata')
        per_page.append({
            **catalog[page_id],
            'source_page_id': page_id,
            'pre_split_arrivals': details.pop(),
            'candidate_positions': [row['position'] for row in page_rows],
            'metrics': scoring.measure(page_rows, predictions[args.model]),
        })
    per_page.sort(key=lambda value: (-value['metrics']['n'], value['source_page_id']))

    membership = {
        'boundary_event_index': boundary,
        'complete_validation_candidate_positions': [row['position'] for row in rows],
        'complete_validation_legal_positions': [row['position'] for row in rows if scoring.eligible(row)],
        'groups': {name: [row['position'] for row in group] for name, group in groups.items()},
        'source_pages': {page_id: [row['position'] for row in group] for page_id, group in source_pages.items()},
    }
    os.umask(0o077)
    output.mkdir(mode=0o700)
    write(output/'plan.json', {
        'version': RUN_VERSION,
        'dataset_run': str(dataset),
        'prediction_run': str(prediction),
        'model': args.model,
        'evaluation_scope': 'complete_two_cohorts',
        'boundary_event_index': boundary,
        'same_complete_validation_membership': True,
        'candidate_n': dataset_audit['candidate_n'],
        'legal_n': dataset_audit['eligible_n'],
        'source_definition': 'current exact page identity was an independent confirmed page arrival strictly before the fixed validation boundary',
        'source_membership_uses_post_split_arrivals': False,
        'source_membership_uses_realized_next_target': False,
        'page_to_page_is_after_outcome_target_kind_diagnostic': True,
        'score': 'current unified evaluation_contract; accepted destination plus exact sibling diagnostic and rank_score',
        'membership_sha256': digest(membership),
        'frozen_all_comparison_key': frozen_all['comparison_key'],
        'recomputed_all_comparison_key': metrics['all_complete_validation']['comparison_key'],
    })
    write(output/'metrics-private.json', {'model': args.model, 'groups': metrics})
    write(output/'membership-private.json', membership)
    write(output/'per-page-private.json', {'model': args.model, 'pages': per_page})
    report = render(dataset, prediction, args.model, boundary, metrics, per_page, source_pages, groups)
    (output/'report-private.md').write_text(report)
    (output/'report-private.md').chmod(0o600)
    for path in (Path(__file__), Path(__file__).with_name('page_transition_diagnostics.py'),
                 Path(__file__).with_name('evaluation_contract.py')):
        (output/('source-'+path.name)).write_bytes(path.read_bytes())
        (output/('source-'+path.name)).chmod(0o600)
    sources.verify_unchanged()
    write(output/'run.json', {
        'complete': True,
        'created_at': datetime.now(timezone.utc).isoformat(),
        'version': RUN_VERSION,
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
        'seen_current_page': metrics['current_exact_page_seen_before_split'],
        'source_page_n': len(source_pages),
    }, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
