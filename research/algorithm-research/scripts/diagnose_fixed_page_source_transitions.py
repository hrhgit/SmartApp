#!/usr/bin/env python3
"""Score current-page source cohorts on the frozen ordered-current test split.

This does not rerun, tune, or alter the historical model.  It reuses the exact
88-row ``all-test-5s`` membership and the already frozen per-row exact Top1/
Top4 scores from the existing same-current-transition audit.  The new cohort is
therefore directly comparable to that audit while remaining source-defined at
prediction time.
"""
import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

from page_transition_diagnostics import VERSION, fixed_test_cohorts, frozen_test_metrics
from review_historical_models import ROOT, Sources, sha, write


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                     separators=(',', ':')).encode()).hexdigest()


def read_jsonl(sources, path):
    path = sources.check(path)
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def percent(value):
    return f'{value:.1%}' if value is not None else '—'


def hits(metric, key):
    return f"{metric[key+'_hits']}/{metric['n']}（{percent(metric[key])}）" if metric['n'] else '—'


def render(benchmark, metrics, evidence, fixed_n):
    order = (
        ('all_fixed_test', '固定 late-test 全部样本'),
        ('current_page', '当前是具体页面'),
        ('current_page_with_mature_outgoing', '当前页面已有成熟同页出边'),
        ('current_page_without_mature_outgoing', '当前页面没有成熟同页出边'),
        ('current_page_outgoing_1', '已有 1 条成熟同页出边'),
        ('current_page_outgoing_2_3', '已有 2–3 条成熟同页出边'),
        ('current_page_outgoing_4_plus', '已有 4+ 条成熟同页出边'),
        ('current_page_to_page', '页面 → 页面（事后目标类型细分）'),
        ('current_page_to_page_with_mature_outgoing', '页面 → 页面，且已有成熟同页出边'),
        ('current_page_with_realized_transition_support', '页面且实际下一步已有同起点支持（原有事后口径子集）'),
    )
    lines = [
        '# 已见当前具体页面 → 下一步：固定测试集诊断',
        '',
        f'本报告严格复用 [既有同起点转移审计](<{benchmark}/report-private.md>) 的 5 秒固定 late-test：**{fixed_n} 条**，成员、清洗、预测和每行精确 Top1/Top4 均未改变。新指标只重新分组，不重新训练或重新评分。',
        '',
        '“当前页面已有成熟同页出边”是预测时可识别的保守定义：该具体页面此前已经作为前台起点，且至少一条独立后续行为已在当前截点前成熟。它不把后台标签、候选库存或真实下一步是否重复当作入组条件。',
        '',
        '## 同一固定测试集上的结果',
        '',
        '| 分组 | 样本数 | 占固定测试 | 精确 Top1 | 精确 Top4 | 真实目标已在候选中 |',
        '|---|---:|---:|---:|---:|---:|',
    ]
    for key, label in order:
        metric = metrics[key]
        lines.append(f"| {label} | {metric['n']} | {percent(metric['population_coverage'])} | "
                     f"{hits(metric, 'top1')} | {hits(metric, 'top4')} | "
                     f"{metric['candidate_present_n']}/{metric['n']}（{percent(metric['candidate_present'])}） |")
    lines += [
        '',
        '## 与原有“已有同起点转移”指标的关系',
        '',
        f"原有指标在相同 {fixed_n} 条固定测试中，事后挑出实际下一步已有同起点支持的 **{evidence['n']} 条**：精确 Top1 {hits(evidence, 'top1')}，Top4 {hits(evidence, 'top4')}。",
        '新指标的“已有成熟同页出边”只要求当前页面本身有历史下一步，不要求历史下一步恰好等于本次真实答案；因此它是可在预测前识别的、更宽的诊断分母。最后一行则把原有事后定义限制到页面来源，供解释用，不可作为线上置信门槛。',
        '',
        '## 边界',
        '',
        '- 本报告沿用旧审计的精确 Top1/Top4，不把后来的统一服务回退综合分混入，避免与原口径不可比。',
        '- 当前页面仅按已保存的具体页面身份判断；窗口仍遵循旧审计的应用窗口类别规则。',
        '- 固定测试曾被查看过；本报告是同一历史测试的诊断性重分组，不是新的独立泛化结论。',
        '',
    ]
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--benchmark-run', type=Path, required=True,
                        help='The frozen ordered-current evidence audit containing all-test-5s-private.jsonl')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    benchmark, output = args.benchmark_run.resolve(), args.output.resolve()
    if output.exists() or output == ROOT.resolve() or not output.is_relative_to(ROOT.resolve()):
        parser.error('Use a new private training directory')

    sources = Sources()
    manifest = sources.read(benchmark/'artifact-hashes.json')
    for name in manifest:
        sources.check(benchmark/name)
    plan = sources.read(benchmark/'plan.json')
    if plan.get('version') != 'ordered-current-evidence-audit-v1' or plan.get('primary_threshold_seconds') != 5:
        raise ValueError('Expected the frozen 5-second ordered-current evidence audit')

    all_rows = read_jsonl(sources, benchmark/'all-test-5s-private.jsonl')
    evidence_rows = read_jsonl(sources, benchmark/'evidence-test-5s-private.jsonl')
    if not all_rows:
        raise ValueError('Fixed test split is empty')
    all_ids = {row['sample_id'] for row in all_rows}
    evidence_ids = {row['sample_id'] for row in evidence_rows}
    expected_evidence_ids = {row['sample_id'] for row in all_rows if row['diagnosis_after_outcome'].get('eligible')}
    if len(all_ids) != len(all_rows) or evidence_ids != expected_evidence_ids:
        raise ValueError('Existing evidence cohort is not an exact subset of the fixed test split')

    groups = fixed_test_cohorts(all_rows)
    fixed_n = len(all_rows)
    metrics = {name: frozen_test_metrics(rows, fixed_n) for name, rows in groups.items()}
    evidence = frozen_test_metrics(evidence_rows, fixed_n)
    if metrics['all_fixed_test'] != frozen_test_metrics(all_rows, fixed_n):
        raise ValueError('Fixed test aggregation changed')

    os.umask(0o077)
    output.mkdir(mode=0o700)
    membership = {'all_test_sample_ids': [row['sample_id'] for row in all_rows],
                  'evidence_sample_ids': [row['sample_id'] for row in evidence_rows]}
    write(output/'plan.json', {
        'version': VERSION,
        'benchmark_run': str(benchmark),
        'fixed_test_file': 'all-test-5s-private.jsonl',
        'same_fixed_test_membership': True,
        'fixed_test_n': fixed_n,
        'source_definition': 'current exact page has one or more mature deduplicated same-current outgoing outcomes before its cutoff',
        'source_membership_uses_realized_next_target': False,
        'page_to_page_is_after_outcome_target_kind_diagnostic': True,
        'realized_transition_support_is_after_outcome_diagnostic': True,
        'score': 'reuse frozen exact Top1/Top4 and candidate audit from the existing ordered-current test rows; no rescore',
        'benchmark_membership_sha256': digest(membership),
    })
    write(output/'metrics-private.json', {'groups': metrics, 'existing_realized_transition_metric': evidence})
    write(output/'membership-private.json', membership)
    (output/'report-private.md').write_text(render(benchmark, metrics, evidence, fixed_n))
    for path in (Path(__file__), Path(__file__).with_name('page_transition_diagnostics.py')):
        (output/('source-'+path.name)).write_bytes(path.read_bytes())
    sources.verify_unchanged()
    write(output/'run.json', {
        'complete': True,
        'created_at': datetime.now(timezone.utc).isoformat(),
        'version': VERSION,
        'benchmark_run': str(benchmark),
        'fixed_test_n': fixed_n,
        'same_fixed_test_membership': True,
        'source_files': sources.accessed,
        'raw_modified': False,
        'model_or_predictions_modified': False,
    })
    for path in output.iterdir():
        if path.is_file():
            path.chmod(0o600)
    write(output/'artifact-hashes.json', {path.name: sha(path) for path in output.iterdir() if path.is_file()})
    print(json.dumps({'output': str(output), 'fixed_test_n': fixed_n,
                      'page_source_metrics': metrics['current_page_with_mature_outgoing'],
                      'existing_transition_metrics': evidence}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
