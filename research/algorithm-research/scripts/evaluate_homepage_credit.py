"""Audit homepage credit and causal evidence expansion on an unchanged cohort.

Predictions, ground truth, cleaning, splits and input-similarity rules stay frozen.
An exact resource or its known canonical service homepage earns homepage credit;
a different specific conversation does not. Evidence expansion is an outcome-
conditioned diagnostic, not an online confidence or acceptance rule.
"""
import argparse
from collections import Counter
import gzip
import hashlib
import json
import os
from pathlib import Path

from compare_semantic_trajectory import groups
from navigation_hierarchy import ENTRY, SERVICE, VERSION, disposition, service
from prepare_semantic_trajectory import freeze_evidence, sorted_stream
from semantic_annotations import compare_annotations
from summarize_tuned_history import verify_run


def homepage_hit(prediction, truth, truth_service, known_home_ids):
    return prediction == truth or (
        truth_service == SERVICE and prediction in known_home_ids)


def score(rows, predictions, predicate):
    n = len(rows)
    first = sum(bool(predictions[str(r['position'])]) and
                predicate(predictions[str(r['position'])][0], r) for r in rows)
    four = sum(any(predicate(p, r) for p in predictions[str(r['position'])][:4])
               for r in rows)
    return {'n': n, 'top1_hits': first, 'top4_hits': four,
            'top1': first / n if n else None, 'top4': four / n if n else None}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', required=True, type=Path)
    parser.add_argument('--evaluation', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    root = (Path.home()/'Library/Application Support/SmartApp/training').resolve()
    out = args.output.expanduser().resolve()
    if not out.is_relative_to(root) or out == root or out.exists():
        parser.error('Use a new private training directory')
    provenance, dataset_hashes = verify_run(args.dataset)
    evaluation_provenance, evaluation_hashes = verify_run(args.evaluation)
    if Path(evaluation_provenance['dataset']).resolve() != args.dataset.resolve():
        raise ValueError('Predictions and evidence use different datasets')
    old_summary = json.loads((args.evaluation/'summary.json').read_text())
    if old_summary['version'] != VERSION:
        raise ValueError('Unexpected deployed-model evaluation version')
    with gzip.open(args.dataset/'dataset-private.json.gz', 'rt') as handle:
        data = json.load(handle)
    predictions = json.loads((args.evaluation/'predictions-private.json').read_text())
    reference = sorted_stream(data['reference_stream'])
    observations = data['observations']
    queries = sorted(data['evaluation'], key=lambda r: r['position'])

    # Reproduce the old evidence classification before adding service equivalence.
    reproduced = freeze_evidence(queries, observations, reference, compare_annotations)
    for before, after in zip(queries, reproduced):
        for field in ('evidence_grade', 'evidence_count', 'witnesses'):
            if before[field] != after[field]:
                raise ValueError('Frozen input evidence rules changed: ' + field)

    raw = Path(provenance['snapshot'])/'events.jsonl'
    raw_hash = hashlib.sha256(raw.read_bytes()).hexdigest()
    if raw_hash != provenance['migration']['current']['events.jsonl']:
        raise ValueError('Frozen observations changed')
    # Metadata for reference outcomes comes from their own maturity cutoffs;
    # homepage availability comes from each query cutoff, never a later catalog.
    operations = sorted([(r['ready_index'], 'reference', i) for i, r in enumerate(reference)] +
                        [(observations[r['position']]['index'], 'query', r['position']) for r in queries])
    catalog, reference_services, homes = {}, {}, {}
    with raw.open() as handle:
        events = enumerate(handle)
        upcoming = next(events, None)
        for cutoff, kind, key in operations:
            while upcoming is not None and upcoming[0] <= cutoff:
                _, line = upcoming
                event = json.loads(line)
                if event['kind'] == 'target.update':
                    target = event['data']['target']
                    catalog[target['id']] = target
                upcoming = next(events, None)
            if kind == 'reference':
                label = reference[key]['record']['label']
                if label.startswith('page:') and label not in catalog:
                    raise ValueError('Reference page metadata missing at maturity')
                reference_services[key] = service(catalog.get(label))
            else:
                homes[key] = {target_id for target_id, target in catalog.items()
                              if target.get('kind') == 'page' and target.get('locator') == ENTRY}

    # The score credits only a known homepage, never an unavailable abstract action.
    assert homepage_hit('exact', 'exact', SERVICE, set())
    assert homepage_hit('home', 'conversation', SERVICE, {'home'})
    assert not homepage_hit('wrong-conversation', 'conversation', SERVICE, {'home'})
    assert not homepage_hit('home', 'unrelated', None, {'home'})
    assert not homepage_hit(SERVICE, 'conversation', SERVICE, set())

    os.umask(0o077)
    out.mkdir(mode=0o700)

    def write(name, value):
        (out/name).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')

    policy = {
        'version': 'homepage-credit-evidence-v1', 'model_version': VERSION,
        'scoring': 'Exact destination OR its known canonical ChatGPT homepage; wrong specific conversation is not a hit.',
        'scope': 'Only ordinary ChatGPT conversation/home routes admitted by existing navigation_hierarchy.service.',
        'evidence': 'Existing reference stream and input relations; expand only destination equivalence, requiring genuine navigation.',
        'maturity': 'Reference ready_index strictly before query index; historical output metadata at reference readiness.',
        'selection': 'Prior support for eventual truth; diagnostic only, never conditioned on predicted hits.',
        'unchanged': ['162 evaluation queries', 'ground truth', 'predictions', 'cleaning', 'chronological splits', 'old cohort membership'],
    }
    write('analysis-plan.json', policy)
    enriched, added, cursor, history = [], [], 0, []
    priority = {'weak': 1, 'strong': 2, 'exact': 3}
    for row in queries:
        q = observations[row['position']]
        while cursor < len(reference) and reference[cursor]['ready_index'] < q['index']:
            history.append((cursor, reference[cursor]))
            cursor += 1
        truth_service = service(row['truth_metadata'])
        witnesses = {}
        if row['evidence_grade'] == 'unsupported' and truth_service:
            for index, past in history:
                if reference_services[index] != truth_service:
                    continue
                if not past['input_index'] < q['index'] or not past['position'] < row['position']:
                    raise ValueError('Future evidence entered a query')
                before = observations[past['position']]
                if disposition(past['record']['label'], before['annotation']['identity']) != 'navigation':
                    continue
                relation = ({'level': 'exact', 'score': 1., 'matched_tags': []}
                            if before['origin'] == q['origin'] else
                            compare_annotations(q['annotation'], before['annotation']))
                if relation['level'] not in priority:
                    continue
                witness = {'source_position': past['position'], 'outcome_id': past['record']['outcome_id'],
                           'ready_index': past['ready_index'], 'relation': relation,
                           'destination_service': truth_service,
                           'destination_relation': 'exact' if past['record']['label'] == row['truth_action'] else 'same_service'}
                identity = witness['outcome_id']
                prior = witnesses.get(identity)
                if prior is None or (priority[relation['level']], relation['score']) > (
                        priority[prior['relation']['level']], prior['relation']['score']):
                    witnesses[identity] = witness
        new_row = {**row, 'homepage_evidence_witnesses': list(witnesses.values())}
        if witnesses:
            new_row['homepage_evidence_grade'] = max(
                (w['relation']['level'] for w in witnesses.values()), key=priority.get)
            added.append(new_row)
        enriched.append(new_row)

    original = [r for r in enriched if r['evidence_grade'] == 'original']
    previous = [r for r in enriched if r['evidence_grade'] != 'unsupported']
    expanded = previous + added
    cohorts = {'original': original, 'previous_expanded': previous, 'new_service_supported': added,
               'expanded': expanded, 'all': enriched}
    for name, members in list(cohorts.items()):
        cohorts['late_test_' + name] = [r for r in members if r['split'] == 'test']
    scores = {}
    for version in ('v1', 'v2'):
        exact = predictions[version + '_exact']
        coarse = predictions[version + '_service']
        predicates = {
            'exact': lambda p, r: p == r['truth_action'],
            'homepage': lambda p, r: homepage_hit(p, r['truth_action'], service(r['truth_metadata']), homes[r['position']]),
            'service': lambda p, r: p == (service(r['truth_metadata']) or r['truth_action']),
        }
        for metric, predicate in predicates.items():
            source = coarse if metric == 'service' else exact
            scores[version + '_' + metric] = {name: score(members, source, predicate)
                                              for name, members in cohorts.items()}
        for metric in ('exact', 'service'):
            for name, members in groups(enriched).items():
                source = coarse if metric == 'service' else exact
                if score(members, source, predicates[metric]) != old_summary['scores'][version + '_' + metric][name]:
                    raise ValueError('Previously reported score changed')

    summary = {'policy': policy, 'scores': scores,
               'evidence_counts': {k: len(v) for k, v in cohorts.items()},
               'new_evidence_input_strength': dict(Counter(r['homepage_evidence_grade'] for r in added)),
               'queries_with_known_homepage': sum(bool(homes[r['position']]) for r in enriched),
               'chatgpt_truth_counts': {k: sum(service(r['truth_metadata']) == SERVICE for r in v) for k, v in cohorts.items()},
               'original_evidence_reproduction_mismatches': 0,
               'original_score_reproduction_mismatches': 0,
               'limit': 'Previously inspected data, prequential online updates; no new held-out accuracy or online confidence claim.'}
    write('summary.json', summary)
    write('evidence-private.json', enriched)
    write('homepage-availability-private.json', {str(k): sorted(v) for k, v in homes.items()})
    write('run.json', {'complete': True, 'dataset': str(args.dataset.resolve()),
                      'dataset_hashes': dataset_hashes, 'evaluation': str(args.evaluation.resolve()),
                      'evaluation_hashes': evaluation_hashes, 'raw_events_sha256': raw_hash,
                      'raw_data_modified': False, 'runtime_modified': False})
    lines = ['# 首页计分与有证据样本扩展', '',
             '当前版本：`' + VERSION + '`。复用冻结预测；不重训、不调参、不更改测试真值。', '',
             '首页计分：命中原具体目的地，或命中该服务已知的标准首页，均得分。推荐了同服务的另一个具体对话不计分。仅覆盖已实现的 ChatGPT 普通对话/首页，不扩展到其他网站、项目或自定义 GPT。', '',
             '| 样本范围 | 条数 | Top1 | Top4 |', '|---|---:|---:|---:|']
    names = {'original': '原固定有证据集', 'previous_expanded': '此前语义扩展集',
             'new_service_supported': '本次新增服务证据样本', 'expanded': '合并后的有证据集', 'all': '全部原评测样本',
             'late_test_original': '后段 test：原固定集', 'late_test_previous_expanded': '后段 test：此前扩展集',
             'late_test_new_service_supported': '后段 test：本次新增', 'late_test_expanded': '后段 test：合并扩展集',
             'late_test_all': '后段 test：全部'}
    for key, title in names.items():
        result = scores['v2_homepage'][key]
        def cell(metric):
            return '—' if not result['n'] else f"{result[metric + '_hits']}/{result['n']} = {result[metric]:.2%}"
        lines.append(f"| {title} | {result['n']} | {cell('top1')} | {cell('top4')} |")
    lines += ['', '扩展只改变输出层的历史支持定义：以前必须见过该具体 URL，现在同服务的其他合格页面也能提供首页层支持。输入相似度、参考训练流和成熟时间边界保持原样；新增证据排除原地返回或仅回到所属应用的标签。', '',
              '原始 162 条评测不增加；扩展的是其中“有历史证据”的诊断子集。所有证据在预测截点之前成熟，且按独立 outcome 去重。分组会参考实际后续目标，所以不是预测前可识别的线上接受集，也不是置信度保证。', '',
              '62 条与此前扩展集包含 validation 和 test；后段 test 已单列。旧数据已经用于研究，因此这些数值是因果回放成绩，不能称为新数据泛化准确率。新增样本的分数不能与旧分母直接比较来宣称模型提升。', '',
              '可核验细节见 `summary.json`、`evidence-private.json`、`homepage-availability-private.json`、`run.json`。原数据、旧实验和已安装模型未修改。', '']
    (out/'report-private.md').write_text('\n'.join(lines))
    for name in ('evaluate_homepage_credit.py', 'navigation_hierarchy.py', 'prepare_semantic_trajectory.py',
                 'semantic_annotations.py', 'compare_semantic_trajectory.py', 'summarize_tuned_history.py'):
        (out/('source-' + name)).write_bytes((Path(__file__).parent/name).read_bytes())
    write('artifact-hashes.json', {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                 for p in out.iterdir() if p.is_file()})
    print(json.dumps({'run': str(out), 'counts': summary['evidence_counts'],
                      'new_evidence_input_strength': summary['new_evidence_input_strength'],
                      'queries_with_known_homepage': summary['queries_with_known_homepage'],
                      'current_homepage_scores': scores['v2_homepage'],
                      'current_service_scores': scores['v2_service'],
                      'previous_homepage_scores': scores['v1_homepage']}))


if __name__ == '__main__':
    main()
