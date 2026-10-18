"""Compare saved recommendations on supported, unsupported and all legal rows."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import gzip
import json
import math
import os
from pathlib import Path

import evaluation_contract as scoring
from rescore_predictions import sha, verify


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True, help='Frozen prediction run')
    parser.add_argument('--reference', type=Path, required=True, help='Current common-scoring run')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    source, reference, out = args.source.resolve(), args.reference.resolve(), args.output.resolve()
    root = (Path.home()/'Library/Application Support/SmartApp/training').resolve()
    if out.exists() or out == root or not out.is_relative_to(root):
        parser.error('Use a new private training run')
    meta = json.loads((source/'run.json').read_text()); dataset = Path(meta['dataset'])
    sources = [source, reference, dataset]; hashes = {str(p): verify(p) for p in sources}
    prior = json.loads((reference/'summary.json').read_text())
    if prior['scoring_contract'] != scoring.contract():
        raise ValueError('Reference must first be re-scored with the current common contract')
    with gzip.open(dataset/'dataset-private.json.gz', 'rt') as f:
        data = json.load(f)
    with gzip.open(source/'predictions-private.json.gz', 'rt') as f:
        predictions = json.load(f)
    os.umask(0o077); out.mkdir(mode=0o700)
    def write(name, value):
        (out/name).write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n')
    rows = scoring.prepare_evaluation(data, out)
    parts = scoring.cohorts(rows); original_groups = list(parts)
    primary = {r['position'] for r in parts['evidence_total']}
    parts.update({
        'outside_primary': [r for r in rows if r['position'] not in primary],
        'additional_support': [r for r in rows if r['position'] not in primary and r['evidence_grade'] != 'unsupported'],
        'unsupported': [r for r in rows if r['evidence_grade'] == 'unsupported'],
        'test_without_validation': [r for r in rows if r['split'] in ('test', 'new_test')],
        'test_unsupported': [r for r in rows if r['split'] in ('test', 'new_test') and r['evidence_grade'] == 'unsupported'],
    })
    membership = {group: {r['position'] for r in part if scoring.eligible(r)} for group, part in parts.items()}
    partitions = [membership[g] for g in ('evidence_total', 'additional_support', 'unsupported')]
    assert set.union(*partitions) == membership['all']
    assert sum(map(len, partitions)) == len(membership['all'])
    assert membership['outside_primary'] == membership['additional_support'] | membership['unsupported']
    plan = {'source': str(source), 'reference': str(reference), 'dataset': str(dataset),
        'source_hashes': hashes, 'scoring_contract': scoring.contract(),
        'membership_independent_of_predictions': True,
        'unsupported_definition': 'existing evidence_grade == unsupported; no support under this diagnostic definition, not proof of no learnable cues',
        'outside_primary_definition': 'complement of the agreed evidence_total; includes additional and weak support',
        'test_definition': 'test and new_test splits; excludes selection validation, but already inspected historically',
        'home_gain': scoring.FALLBACK_GAIN,
        'predictions_changed': False, 'trained': False, 'fresh_holdout': False, 'runtime_changed': False}
    write('plan.json', plan)
    results = {}; rank_scores = {}
    for group, part in parts.items():
        results[group] = {}; rank_scores[group] = {}
        for name, pred in predictions.items():
            m = scoring.measure(part, pred); results[group][name] = m
            if group in original_groups:
                assert m == prior['results'][name][group]
            rank_scores[group][name] = {'total': m['rank_score']['total'], 'n': m['n'],
                'score100': 100*m['rank_score']['mean'] if m['n'] else None}
        scoring.require_comparable(*results[group].values())
    # The whole-set gain must be the sum of mutually exclusive subgroup gains.
    for name in predictions:
        total = sum(rank_scores[g][name]['total'] for g in ('evidence_total', 'additional_support', 'unsupported'))
        assert math.isclose(total, rank_scores['all'][name]['total'], abs_tol=1e-10)
    scopes = {g: {'audit': scoring.audit_summary(part),
                  'eligible_splits': dict(Counter(r['split'] for r in part if scoring.eligible(r))),
                  'eligible_evidence_grades': dict(Counter(r['evidence_grade'] for r in part if scoring.eligible(r))),
                  'positions': sorted(membership[g])} for g, part in parts.items()}
    summary = {'scope': scoring.report_scope(rows), 'results': results, 'rank_scores': rank_scores,
        'home_gain': scoring.FALLBACK_GAIN, 'scopes': scopes, 'scoring_contract': scoring.contract(),
        'canonical_cohorts_match_reference': True, 'partition_totals_verified': True,
        'predictions_changed': False, 'trained': False, 'fresh_holdout': False}
    write('summary.json', summary)
    names = ['existing_order', 'fixed', 'context_frequency', 'global_frequency']
    sizes = {g: len(v) for g, v in membership.items()}
    candidate_n = scopes['all']['audit']['candidate_n']
    excluded_n = scopes['all']['audit']['excluded_n']
    validation_n = scopes['all']['eligible_splits'].get('validation', 0)
    lines = ['# 全部合法样本与证据不足样本对照', '', summary['scope'], '',
        '所有方案使用同一评分入口、同一合法性审查、同一预测截点。未训练或改动推荐顺序，未重新选择模型。', '',
        f'评分：精确匹配当前标签收益 1，正确首页回退收益 {scoring.FALLBACK_GAIN:g}，错误 0；分别取 Top1 至 Top4 最佳收益后平均。', '',
        '## 有证据合计及扩大范围后的综合分', '',
        '单元格为同一收益定义下的百分制综合分，分母按列注明。', '',
        f"| 策略 | 主证据{sizes['evidence_total']} | 主证据外{sizes['outside_primary']} | 无已记录支持{sizes['unsupported']} | 其他支持{sizes['additional_support']} | 全部合法{sizes['all']} |",
        '|---|---:|---:|---:|---:|---:|']
    for name in names:
        cells = []
        for group in ('evidence_total', 'outside_primary', 'unsupported', 'additional_support', 'all'):
            cells.append(f"{rank_scores[group][name]['score100']:.2f}")
        lines.append('| '+name+' | '+' | '.join(cells)+' |')
    for group in ('evidence_total', 'unsupported', 'all', 'test_without_validation', 'test_unsupported'):
        n = len(membership[group])
        lines += ['', f'## {group}：{n} 条', '',
            '| 策略 | 接受 Top1/2/4 命中数 | 精确 Top1/2/4 命中数 | 总收益 / 满分 | 综合分 |',
            '|---|---|---|---:|---:|']
        for name in names:
            m = results[group][name]; v = rank_scores[group][name]
            counts = [' / '.join(str(level[f'top{k}_hits']) for k in (1, 2, 4)) for level in (m, m['exact'])]
            lines.append('| '+name+' | '+' | '.join(counts)+f" | {v['total']:g}/{n} | {v['score100']:.2f} |")
    lines += ['', '## 新旧时段：全部合法样本', '',
        f"| 策略 | 旧{sizes['old_all']}条 | 新{sizes['new_all']}条 |", '|---|---:|---:|']
    for name in names:
        cells = []
        for group in ('old_all', 'new_all'):
            cells.append(f"{rank_scores[group][name]['score100']:.2f}")
        lines.append('| '+name+' | '+' | '.join(cells)+' |')
    lines += ['', '## 口径与解释', '',
        f"- {sizes['all']} 条是现有数据集中的全部合法评测截点，不是全部原始采集事件。原{candidate_n}条中{excluded_n}条不合法标签仍按公共规则排除。",
        f"- 三个互斥组为主证据{sizes['evidence_total']}、其他支持{sizes['additional_support']}、无已记录支持{sizes['unsupported']}；主证据外{sizes['outside_primary']}不能全部称为无证据。其他支持的等级计数：{scopes['additional_support']['eligible_evidence_grades']}。",
        '- 无已记录支持只表示既定历史支持检验未发现证据，不表示输入中必然没有线索，也不表示目标一定从未见过。分组依赖真实结果的事后诊断，不能直接作为线上置信判断。',
        f"- 全部{sizes['all']}条包含{validation_n}条历史选参验证样本；另列{sizes['test_without_validation']}条较晚/新增测试样本，以及其中{sizes['test_unsupported']}条无已记录支持样本。它们也已经被多轮研究，不是全新独立测试。",
        '- 排名变化来自扩大评测范围，不能宣称模型退化或进步；全部采用同一收益定义，未重新生成推荐。',
        '- “精确”是对当前评分标签的精确匹配；本轮普通窗口已归并所属应用，不能把这些命中宣称为精确窗口实例命中。',
        '- 每条样本的具体标签、成员与排除原因保存在私有审查文件，原始采集与运行时均未修改。']
    (out/'report-private.md').write_text('\n'.join(lines)+'\n')
    for p in sources:
        assert verify(p) == hashes[str(p)]
    for name in ('compare_evidence_scopes.py', 'rescore_predictions.py'):
        (out/name).write_bytes(Path(__file__).with_name(name).read_bytes())
    write('run.json', {'complete': True, 'created_at': datetime.now(timezone.utc).isoformat(), **plan})
    write('artifact-hashes.json', {p.name: sha(p) for p in out.iterdir() if p.is_file()})
    assert all(p.stat().st_mode & 0o777 == 0o600 for p in out.iterdir() if p.is_file())
    print(json.dumps({'output': str(out), 'counts': {g: len(v) for g, v in membership.items()},
                     'scores': {g: rank_scores[g] for g in ('unsupported', 'all', 'test_without_validation', 'test_unsupported')}}, ensure_ascii=False))


if __name__ == '__main__':
    main()
