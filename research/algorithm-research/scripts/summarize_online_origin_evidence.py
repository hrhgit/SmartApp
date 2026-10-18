#!/usr/bin/env python3
"""Score frozen online-shadow forecasts on a prospective origin-history cohort.

The cohort reuses the project's existing ``prospective_gate.has_origin_history``
meaning: at a prediction cutoff, the current semantic origin has at least one
unique mature historical outcome.  Membership is formed before reading the
current truth or either model's ranking.  This script only aggregates already
frozen forecasts: it never trains, selects, or changes a model.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import gzip
import hashlib
import json
import math
import os
from pathlib import Path

import evaluate_context_knn as io
import evaluation_contract as scoring
from online_sequence_rnn_shadow import (PRIVATE_ROOT, _safe_data, sequence_inputs,
                                        supervision_stream)
from summarize_tuned_history import verify_run
from temporal_context import build_contexts


VERSION = 'online-origin-history-evidence-summary-v1'
SOURCE_FILES = (
    'summarize_online_origin_evidence.py',
    'online_sequence_rnn_shadow.py',
    'evaluation_contract.py',
    'temporal_context.py',
    'known_inactivity.py',
    'compare_semantic_trajectory.py',
    'summarize_tuned_history.py',
    'evaluate_context_knn.py',
)
TOPK = (1, 2, 4)


def _private_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    path.chmod(0o600)


def _source_bytes():
    directory = Path(__file__).resolve().parent
    return {name: (directory / name).read_bytes() for name in SOURCE_FILES}


def _private_output(path):
    path = path.expanduser().resolve()
    root = PRIVATE_ROOT.resolve()
    if path == root or not path.is_relative_to(root) or path.exists():
        raise ValueError('Use a new directory under SmartApp private training storage')
    return path


def origin_history_membership(rows, stream, inputs, causal_ledger=None):
    """Build the existing prospective ``has_origin_history`` gate causally.

    The membership pass reads only source origin, outcome identity and maturity.
    It deliberately has no parameter for truth, evidence grade, predictions,
    candidate inventory, or model scores.
    """
    history = defaultdict(set)
    membership, cursor = {}, 0
    ordered = sorted(rows, key=lambda row: inputs[row['position']]['index'])
    ledger = {row['position']: row for row in causal_ledger or ()}
    bins = Counter()
    for row in ordered:
        query = inputs[row['position']]
        while cursor < len(stream) and stream[cursor]['ready_index'] < query['index']:
            record = stream[cursor]
            if not (record['position'] < row['position'] and record['input_index'] < query['index']
                    and record['ready_index'] < query['index']):
                raise ValueError('Future or same-cutoff feedback entered evidence membership')
            history[record['origin']].add(record['outcome_id'])
            cursor += 1
        if ledger:
            saved = ledger.get(row['position'])
            if saved is None or saved['mature_updates_at_cutoff'] != cursor:
                raise ValueError('Prospective evidence replay differs from frozen causal ledger')
        count = len(history[query['origin']])
        membership[row['position']] = {
            'has_origin_history': count > 0,
            'mature_unique_origin_outcomes': count,
            'mature_updates_at_cutoff': cursor,
        }
        bins['0' if count == 0 else '1' if count == 1 else '2_3' if count <= 3 else '4_plus'] += 1
    if set(membership) != {row['position'] for row in rows}:
        raise ValueError('Missing evidence membership at an evaluation cutoff')
    return membership, {
        'definition': 'prospective_gate.has_origin_history: mature unique historical outcome count from current origin > 0',
        'membership_sha256': scoring.digest([
            [position, value['has_origin_history'], value['mature_unique_origin_outcomes']]
            for position, value in sorted(membership.items())
        ]),
        'all_evaluation_count_bins': dict(bins),
        'causal_ledger_checked': bool(ledger),
    }


def cohort_rows(rows, membership, scope):
    """Return all legal rows and its prospective, output-independent subset."""
    if scope not in {'causal_full', 'fixed_test'}:
        raise ValueError('Unknown evidence scope: ' + scope)
    selected = [row for row in rows if scope == 'causal_full' or row['split'] == 'test']
    legal = [row for row in selected if scoring.eligible(row)]
    evidence = [row for row in legal if membership[row['position']]['has_origin_history']]
    counts = Counter()
    for row in legal:
        count = membership[row['position']]['mature_unique_origin_outcomes']
        counts['0' if count == 0 else '1' if count == 1 else '2_3' if count <= 3 else '4_plus'] += 1
    return {'all_legal': legal, 'origin_history': evidence}, {
        'scope': scope,
        'evaluation_n': len(selected),
        'legal_n': len(legal),
        'origin_history_n': len(evidence),
        'origin_history_coverage': len(evidence) / len(legal) if legal else None,
        'origin_history_count_bins_among_legal': dict(counts),
        'legality_exclusions': dict(Counter(row['_scoring']['reason'] for row in selected
                                             if not scoring.eligible(row))),
    }


def _change(before, after):
    if after and not before:
        return 'helped'
    if before and not after:
        return 'hurt'
    if after:
        return 'both_correct'
    return 'both_wrong'


def paired_help_harm(rows, reference, alternative, membership):
    """Aggregate and minimally audit per-sample help/harm without target IDs."""
    counts = {level: {k: Counter() for k in TOPK} for level in ('accepted', 'exact')}
    rank = Counter()
    cases = []
    for row in rows:
        if not scoring.eligible(row):
            continue
        before = scoring.prediction_at(reference, row['position'])
        after = scoring.prediction_at(alternative, row['position'])
        before_hits = scoring.hits(row, before)
        after_hits = scoring.hits(row, after)
        case = {
            'position': row['position'], 'sample_id': row['sample_id'], 'split': row['split'],
            'has_origin_history': membership[row['position']]['has_origin_history'],
            'mature_unique_origin_outcomes': membership[row['position']]['mature_unique_origin_outcomes'],
            'accepted': {}, 'exact': {},
        }
        for level in counts:
            for k in TOPK:
                change = _change(before_hits[level][k], after_hits[level][k])
                counts[level][k][change] += 1
                case[level][f'top{k}'] = change
        before_rank = scoring.ranked_gain(row, before)['score']
        after_rank = scoring.ranked_gain(row, after)['score']
        rank_change = 'helped' if after_rank > before_rank else 'hurt' if after_rank < before_rank else 'unchanged'
        rank[rank_change] += 1
        case['rank_score_change'] = rank_change
        cases.append(case)
    summary = {
        'n': len(cases),
        **{level: {f'top{k}': {**counts[level][k],
                                'net': counts[level][k]['helped'] - counts[level][k]['hurt']}
                     for k in TOPK}
           for level in counts},
        'rank_score': {**rank, 'net': rank['helped'] - rank['hurt']},
    }
    return summary, cases


def _pct(value):
    return '—' if value is None else f'{value:.1%}'


def _triple(metric):
    return ' / '.join(_pct(metric[f'top{k}']) for k in TOPK)


def render_report(summary):
    lines = [
        '# 预测前当前来源历史证据 cohort：冻结在线 RNN 补充评测', '',
        '这是对既有冻结预测的纯评分补充：不训练、不重放模型更新、不重新选参，也不改写原 shadow run。',
        '成员定义复用项目既有 `prospective_gate.has_origin_history`：预测截点前，当前来源已有至少一条成熟的独立后续行为。成员资格只用成熟度、来源与独立 outcome ID；不读取本次真实下一步、evidence_grade、模型预测、命中或候选。',
        '它不同于以真实下一步为条件的回顾性诊断 evidence；本报告的 cohort 可在预测时识别，但不应被解释为成功概率或线上自动接受门。', '',
        '## 成员与覆盖', '',
        '| 范围 | 评测行 | 合法样本 | 有当前来源历史 | cohort 覆盖率 | 预测前成熟 outcome 数分布（合法样本） | 标签合法性排除 |',
        '|---|---:|---:|---:|---:|---|---|',
    ]
    names = {'causal_full': '全程因果', 'fixed_test': '固定测试'}
    for scope, value in summary['cohorts'].items():
        lines.append('| ' + ' | '.join([
            names[scope], str(value['evaluation_n']), str(value['legal_n']),
            str(value['origin_history_n']), _pct(value['origin_history_coverage']),
            json.dumps(value['origin_history_count_bins_among_legal'], ensure_ascii=False),
            json.dumps(value['legality_exclusions'], ensure_ascii=False),
        ]) + ' |')
    lines += [
        '', '## 统一评分结果', '',
        '| 范围 | cohort | 模型 | 样本 | cohort 覆盖率 | 输出覆盖率 | 服务接受 Top1 / Top2 / Top4 | 精确 Top1 / Top2 / Top4 | RankScore |',
        '|---|---|---|---:|---:|---:|---|---|---:|',
    ]
    for scope, cohorts in summary['scores'].items():
        for cohort, values in cohorts.items():
            coverage = summary['cohorts'][scope]['origin_history_coverage'] if cohort == 'origin_history' else 1.0
            for model, metric in values.items():
                label = '单步统计基线' if model == 'baseline' else '冻结在线 RNN'
                lines.append('| ' + ' | '.join([
                    names[scope], '全部合法' if cohort == 'all_legal' else '预测前当前来源历史',
                    label, str(metric['n']), _pct(coverage), _pct(metric['prediction_coverage']),
                    _triple(metric), _triple(metric['exact']), _pct(metric['rank_score']['mean']),
                ]) + ' |')
    lines += [
        '', '## 样本级 help/harm（RNN 相对单步统计）', '',
        '| 范围 | cohort | 口径 | TopK | 帮助 | 伤害 | 净变化 | 双正确 | 双错误 |',
        '|---|---|---|---:|---:|---:|---:|---:|---:|',
    ]
    for scope, cohorts in summary['help_harm'].items():
        for cohort, value in cohorts.items():
            for level, label in (('accepted', '服务接受'), ('exact', '精确目标')):
                for k in TOPK:
                    item = value[level][f'top{k}']
                    lines.append(f"| {names[scope]} | {'全部合法' if cohort == 'all_legal' else '预测前当前来源历史'} | "
                                 f"{label} | {k} | {item['helped']} | {item['hurt']} | {item['net']} | "
                                 f"{item['both_correct']} | {item['both_wrong']} |")
            rank = value['rank_score']
            lines.append(f"- {names[scope]}／{'全部合法' if cohort == 'all_legal' else '预测前当前来源历史'} RankScore："
                         f"帮助 {rank['helped']}、伤害 {rank['hurt']}、不变 {rank['unchanged']}、净 {rank['net']}。")
    lines += [
        '', '## 审计与限制', '',
        '- 所有评分使用冻结 run 相同的 `evaluation_contract`，并先逐字段复算全程逐截点分数；服务接受与精确目标始终并列，Top2 不省略。',
        '- 私有 `help-harm-private.json` 仅含 sample ID、预测前成员资格和帮助/伤害类别，不含真值、预测、候选或目标定位符。',
        '- 原始数据、采集范围、运行时模型和选参均未改动；固定测试此前已看过，结果是描述性 shadow 证据。',
    ]
    return '\n'.join(lines) + '\n'


def _seal(output):
    for path in output.iterdir():
        if path.is_file():
            path.chmod(0o600)
    hashes = {path.name: io.sha(path) for path in output.iterdir()
              if path.is_file() and path.name != 'artifact-hashes.json'}
    _private_json(output/'artifact-hashes.json', hashes)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', required=True, type=Path)
    parser.add_argument('--model-run', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    dataset = args.dataset.expanduser().resolve()
    model_run = args.model_run.expanduser().resolve()
    output = _private_output(args.output)

    dataset_meta, dataset_hashes = verify_run(dataset)
    model_meta, model_hashes = verify_run(model_run)
    if not model_meta.get('complete') or model_meta.get('deployment'):
        raise ValueError('Requires a completed non-deployment shadow model run')
    if Path(model_meta['dataset']).resolve() != dataset:
        raise ValueError('Model run and supplied dataset differ')
    selection = json.loads((model_run/'selection.json').read_text())
    if selection.get('selection_uses_test_labels') is not False:
        raise ValueError('Model selection provenance must be fixed before test scoring')
    if Path(scoring.__file__).read_bytes() != (model_run/'evaluation_contract.py').read_bytes():
        raise ValueError('Live scoring contract differs from frozen model-run contract')

    with gzip.open(dataset/'dataset-private.json.gz', 'rt') as handle:
        data = _safe_data(json.load(handle))
    snapshot = Path(dataset_meta['snapshot']).expanduser().resolve()
    if io.sha(snapshot/'events.jsonl') != dataset_meta['migration']['current']['events.jsonl']:
        raise ValueError('Frozen raw event snapshot provenance mismatch')
    contexts, _, history_audit = build_contexts(io.load_lines(snapshot/'events.jsonl'),
                                                data['observations'], io.KNOWN_TEST_SESSIONS)
    inputs = sequence_inputs(data['observations'], contexts)
    stream, training_audit = supervision_stream(data, inputs)
    rows = scoring.prepare_evaluation(data)
    rows.sort(key=lambda row: row['position'])

    saved_results = json.loads((model_run/'results.json').read_text())
    if training_audit['stream_sha256'] != saved_results['training']['stream_sha256']:
        raise ValueError('Prospective evidence stream differs from frozen model replay')
    causal_ledger = json.loads((model_run/'causal-ledger-private.json').read_text())
    membership, membership_audit = origin_history_membership(rows, stream, inputs, causal_ledger)

    with gzip.open(model_run/'predictions-private.json.gz', 'rt') as handle:
        forecasts = json.load(handle)
    expected_positions = {str(row['position']) for row in rows}
    if set(forecasts) != {'baseline', 'selected_rnn'} or any(set(value) != expected_positions
                                                               for value in forecasts.values()):
        raise ValueError('Frozen forecast membership is incomplete or unexpected')
    for forecast_name, result_name in (('baseline', 'baseline'), ('selected_rnn', 'selected')):
        if scoring.measure(rows, forecasts[forecast_name]) != saved_results['models'][result_name]['metrics']['full_all']:
            raise ValueError('Frozen predictions no longer reproduce original unified full score')

    cohort_audit, scores, helps, cases = {}, {}, {}, {}
    for scope in ('causal_full', 'fixed_test'):
        cohorts, audit = cohort_rows(rows, membership, scope)
        cohort_audit[scope] = audit
        scores[scope] = {
            name: {model: scoring.measure(part, forecasts[model])
                   for model in ('baseline', 'selected_rnn')}
            for name, part in cohorts.items()
        }
        helps[scope], cases[scope] = {}, {}
        for name, part in cohorts.items():
            help_summary, help_cases = paired_help_harm(part, forecasts['baseline'], forecasts['selected_rnn'], membership)
            helps[scope][name], cases[scope][name] = help_summary, help_cases
            scoring.require_comparable(scores[scope][name]['baseline'], scores[scope][name]['selected_rnn'])

    summary = {
        'version': VERSION,
        'aggregate_only': True,
        'model_retrained': False,
        'model_reselected': False,
        'membership_uses_truth_or_predictions': False,
        'source_model_run': str(model_run),
        'source_dataset': str(dataset),
        'membership': membership_audit,
        'training_stream': training_audit,
        'cohorts': cohort_audit,
        'scores': scores,
        'help_harm': helps,
        'temporal_history_audit': dict(history_audit),
        'frozen_full_score_reproduced': True,
        'top2_required_everywhere': True,
    }

    source_bytes = _source_bytes()
    os.umask(0o077)
    output.mkdir(mode=0o700, parents=True, exist_ok=False)
    plan = {
        'version': VERSION,
        'purpose': 'prospective current-origin historical-evidence cohort scoring of frozen forecasts',
        'existing_definition': membership_audit['definition'],
        'membership_inputs_only': ['current origin', 'mature unique historical outcome ID', 'ready_index', 'input index'],
        'membership_forbidden_inputs': ['current truth', 'evidence grade', 'prediction', 'hit', 'candidate inventory'],
        'aggregate_only': True, 'model_retrained': False, 'model_reselected': False,
        'model_run': str(model_run), 'model_artifact_hashes': model_hashes,
        'dataset': str(dataset), 'dataset_artifact_hashes': dataset_hashes,
        'source_hashes': {name: hashlib.sha256(value).hexdigest() for name, value in source_bytes.items()},
        'scoring_contract': scoring.contract(),
    }
    _private_json(output/'plan.json', plan)
    _private_json(output/'origin-history-evidence-summary-private.json', summary)
    # Per-sample audit excludes actual target, forecasts and candidate contents.
    _private_json(output/'help-harm-private.json', cases)
    report = output/'report-private.md'
    report.write_text(render_report(summary))
    report.chmod(0o600)
    _private_json(output/'run.json', {
        'complete': True, 'version': VERSION, 'aggregate_only': True,
        'model_retrained': False, 'model_reselected': False,
        'membership_uses_truth_or_predictions': False,
        'source_model_run': str(model_run), 'source_dataset': str(dataset),
        'collection_changed': False, 'runtime_model_changed': False, 'deployment': False,
    })
    for name, value in source_bytes.items():
        path = output/('source-' + name)
        path.write_bytes(value)
        path.chmod(0o600)

    if verify_run(dataset)[1] != dataset_hashes or verify_run(model_run)[1] != model_hashes:
        raise ValueError('Source run changed while evidence cohort was summarized')
    if any((Path(__file__).resolve().parent/name).read_bytes() != value
           for name, value in source_bytes.items()):
        raise ValueError('Aggregation source changed while running')
    _seal(output)
    print(json.dumps({
        'output': str(output),
        'causal_origin_history_n': cohort_audit['causal_full']['origin_history_n'],
        'fixed_test_origin_history_n': cohort_audit['fixed_test']['origin_history_n'],
        'aggregate_only': True,
    }, ensure_ascii=False))


if __name__ == '__main__':
    main()
