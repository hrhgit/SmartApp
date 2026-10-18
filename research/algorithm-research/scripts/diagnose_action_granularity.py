"""Audit action aliasing and conversation granularity without changing runtime data.

The frozen replay retains the original cohort, labels and maturity boundaries.
Ablations are diagnostic, not a new model selection or a production fix.
All personal inputs/results are written only to a new private training run.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import gzip
import hashlib
import json
import itertools
import math
import os
from pathlib import Path
import sqlite3
import subprocess
from urllib.parse import urlsplit

from compare_semantic_trajectory import groups, training_rows
from semantic_annotations import annotate_target
from semantic_online import SemanticStatistics, rank
from summarize_tuned_history import verify_run


KINDS = {'site', 'url_parent', 'local_directory_scope', 'observed_parent_id',
         'name_token', 'course_scope', 'project_scope', 'repository_scope'}


class NativeSimilarity:
    """Swift SemanticProfile equivalent, including native profiles without basis."""
    def score(self, left, right):
        a, b = left['identity'], right['identity']
        if a['kind'] == b['kind'] and a['kind'] != 'window':
            if a.get('locator') is not None and a['locator'] == b.get('locator'):
                return 1.
            if a.get('locator') is None and b.get('locator') is None:
                field = 'bundleID' if a['kind'] == 'app' else 'id'
                if a.get(field) is not None and a[field] == b.get(field):
                    return 1.
        right_tags = {t['key']: t for t in right['tags']}
        similarity = 0.
        for t in left['tags']:
            other = right_tags.get(t['key'])
            if t['kind'] not in KINDS or other is None:
                continue
            strength = min(t['strength'], other['strength'])
            if t['kind'] == 'local_directory_scope' and strength < .35:
                continue
            similarity = max(similarity, strength * min(t['confidence'], other['confidence']))
        return similarity ** 2


def model():
    return SemanticStatistics(semantic_strength=16, global_strength=1,
                              recency_mix=.25, similarity=NativeSimilarity())


def action(target):
    if target.get('kind') in {'app', 'window'}:
        owner = target.get('bundleID') or target.get('_bundle')
        return 'app:' + owner if owner else None
    return target.get('id') if target.get('locator') else None


def query(target, timestamp):
    origin = ('window_class:' + target['bundleID']
              if target['kind'] == 'window' else target['id'])
    return {'origin': origin, 'annotation': annotate_target(target), 'timestamp': timestamp}


def route(target, host):
    if target.get('kind') != 'page':
        return None
    p = urlsplit(target.get('locator', ''))
    if p.hostname != host:
        return None
    parts = p.path.strip('/').split('/') if p.path.strip('/') else []
    if not parts:
        return 'home'
    if len(parts) == 2 and parts[0] == 'c':
        return 'conversation'
    # Project/custom-GPT/account routes are intentionally NOT assumed equivalent.
    return 'other_route'


def metric(rows, predictions):
    n = len(rows)
    top1 = sum(predictions[r['position']][:1] == [r['truth_action']] for r in rows)
    top4 = sum(r['truth_action'] in predictions[r['position']] for r in rows)
    return {'n': n, 'top1_hits': top1, 'top4_hits': top4,
            'top1': top1 / n if n else None, 'top4': top4 / n if n else None}


def frozen_audit(data, selected, bundle):
    stream = [r for r in training_rows(data, 'hard_kind') if r['head'] == 0]
    original, without_alias = model(), model()
    predictions = {name: {} for name in ('raw_model', 'runtime_id_filter', 'train_without_alias',
                                        'action_mask_only', 'train_without_alias_and_mask')}
    self_rows, noops, mismatch, cursor = [], Counter(), 0, 0
    for r in stream:
        q = data['observations'][r['position']]
        if r['label'] == action(q['target']):
            ledger = data['trajectory_ledgers']['hard_kind'][r['position']]
            self_rows.append({'position': r['position'], 'origin': q['origin'],
                              'label': r['label'], 'observed_step': r['record']['observed_step'],
                              'same_target_id': q['target']['id'] == r['record']['target']['id'],
                              'earlier_dropped': [v for v in ledger['dropped']
                                                  if v['observed_step'] < r['record']['observed_step']]})
    for r in sorted(data['evaluation'], key=lambda v: v['position']):
        q = data['observations'][r['position']]
        while cursor < len(stream) and stream[cursor]['ready_index'] < q['index']:
            before = stream[cursor]
            x = data['observations'][before['position']]
            assert x['index'] < q['index']
            kwargs = dict(step=0, weight=before['weight'], outcome_id=before['outcome_id'],
                          outcome_timestamp=before['outcome_timestamp'])
            original.learn(x, before['label'], **kwargs)
            if action(x['target']) != before['label']:
                without_alias.learn(x, before['label'], **kwargs)
            cursor += 1
        scores = original.predict(q, 0)
        alias = action(q['target'])
        raw = rank(scores)
        mismatch += raw != selected[str(r['position'])]['single']
        predictions['raw_model'][r['position']] = raw
        predictions['runtime_id_filter'][r['position']] = rank({k: v for k, v in scores.items() if k != q['target']['id']})
        predictions['action_mask_only'][r['position']] = rank({k: v for k, v in scores.items() if k != alias})
        cleaned_scores = without_alias.predict(q, 0)
        predictions['train_without_alias'][r['position']] = rank(cleaned_scores)
        predictions['train_without_alias_and_mask'][r['position']] = rank({k: v for k, v in cleaned_scores.items() if k != alias})
        # Only coarse app/window inputs are called execution aliases here.
        if q['target']['kind'] in {'app', 'window'}:
            noops['coarse_inputs'] += 1
            noops['current_app_in_runtime_top4'] += alias in predictions['runtime_id_filter'][r['position']]
            noops['current_app_is_frozen_truth'] += alias == r['truth_action']
            noops['current_app_top4_after_training_alias_removal'] += alias in predictions['train_without_alias'][r['position']]
            if q['target'].get('bundleID') == bundle:
                noops['focus_inputs'] += 1
                noops['focus_current_app_in_runtime_top4'] += alias in predictions['runtime_id_filter'][r['position']]
    if mismatch:
        raise ValueError(f'Frozen winner replay mismatch: {mismatch}')
    summary = {'baseline_parity_mismatches': mismatch, 'execution_alias_audit': dict(noops),
               'self_action_training_records': len(self_rows),
               'self_action_after_dropped_steps': sum(bool(v['earlier_dropped']) for v in self_rows),
               'self_action_same_target_id': sum(v['same_target_id'] for v in self_rows),
               'focus_self_action_records': sum(v['label'] == 'app:' + bundle for v in self_rows),
               'scores': {name: {g: metric(rows, p) for g, rows in groups(data['evaluation']).items()}
                          for name, p in predictions.items()}}
    return summary, {'self_action_rows': self_rows, 'predictions': predictions}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, default=Path.home()/'Library/Application Support/SmartApp')
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--comparison', type=Path, required=True)
    parser.add_argument('--bundle', required=True)
    parser.add_argument('--host', required=True)
    parser.add_argument('--native-cli', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    root, out = args.data_root.expanduser().resolve(), args.output.expanduser().resolve()
    private = (Path.home()/'Library/Application Support/SmartApp/training').resolve()
    if not out.is_relative_to(private) or out == private or out.exists():
        parser.error('Use a new private SmartApp training run')
    os.umask(0o077)
    out.mkdir(mode=0o700)
    def write(name, data):
        (out/name).write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n')

    state_bytes = (root/'models/semantic-statistics-hard-kind-v1/state.json').read_bytes()
    (out/'model-state-private.json').write_bytes(state_bytes)
    state = json.loads(state_bytes)
    with sqlite3.connect((root/'context-v2.sqlite').as_uri()+'?mode=ro', uri=True) as conn:
        conn.execute('BEGIN')
        max_seq = conn.execute('select max(seq) from events').fetchone()[0]
        catalog = {}
        for raw, in conn.execute("select json from events where kind='target.update' order by seq"):
            target = json.loads(raw)['data'].get('target')
            if target:
                catalog[target['id']] = target
        contexts = [json.loads(raw) for raw, in conn.execute(
            "select json from events where kind='context.change' and app_bundle_id=? order by seq desc limit 40", (args.bundle,))]
    write('target-catalog-private.json', catalog)
    write('recent-focus-contexts-private.json', contexts)
    current, cleaned_current = model(), model()
    for rec in state['records']:
        x = {'origin': rec['origin'], 'annotation': rec['annotation'], 'timestamp': rec['outcomeTimestamp']}
        kwargs = dict(step=0, weight=rec['weight'], outcome_id=rec['outcomeID'], outcome_timestamp=rec['outcomeTimestamp'])
        current.learn(x, rec['label'], **kwargs)
        if rec['label'] != action(rec['annotation']['identity']):
            cleaned_current.learn(x, rec['label'], **kwargs)
    now = datetime.now(timezone.utc)
    app = catalog['app:' + args.bundle]
    window = next(catalog[e['data']['targetID']] for e in contexts
                  if catalog.get(e['data']['targetID'], {}).get('kind') == 'window')
    targets = [app, {**window, 'bundleID': args.bundle}]
    pages = {k: v for k, v in catalog.items() if route(v, args.host)}
    web_labels = {r['label'] for r in state['records']} & pages.keys()
    conversations = [annotate_target(t) for t in pages.values() if route(t, args.host) == 'conversation']
    similarities = [math.sqrt(NativeSimilarity().score(a, b)) for a,b in itertools.combinations(conversations, 2)]
    details, queries = [], []
    for index, target in enumerate(targets):
        q = query(target, now.timestamp()*1000)
        exact, semantic, global_prior, recency = current.components(q, 0)
        scores = current.predict(q, 0)
        order = rank(scores, len(scores))
        cleaned_order = rank(cleaned_current.predict(q, 0), len(scores))
        details.append({'current': target, 'origin': q['origin'], 'self_action_rank': order.index(action(target))+1,
                        'self_action_rank_after_training_alias_removal': cleaned_order.index(action(target))+1 if action(target) in cleaned_order else None,
                        'exact_records': sum(exact.values()),
                        'self_exact_votes': exact.get(action(target), 0),
                        'homepage_exact_votes': sum(exact.get(k, 0) for k, t in pages.items() if route(t, args.host) == 'home'),
                        'web_family_score': sum(scores.get(k, 0) for k in web_labels),
                        'web_family_exact_votes': sum(exact.get(k, 0) for k in web_labels),
                        'top10': [{'target': catalog.get(k, {'id': k}), 'score': scores[k],
                                   'exact': exact.get(k, 0), 'semantic_prior': semantic.get(k, 0),
                                   'global_prior': global_prior.get(k, 0), 'recency': recency.get(k, 0)} for k in order[:10]],
                        'homepage_ranks': [{'target': t, 'rank': order.index(k)+1 if k in order else None,
                                            'score': scores.get(k, 0)} for k,t in pages.items() if route(t,args.host) == 'home']})
        queries.append({'position': index, 'index': 1, 'origin': q['origin'], 'target': target, 'expected': order[:4]})
    write('current-query-audit-private.json', details)
    write('native-fixture-private.json', {'updates': [{'readyIndex': 0, 'record': r} for r in state['records']],
                                         'queries': queries, 'annotations': []})
    native = subprocess.run([str(args.native_cli.resolve()), 'validate-model', '--input', str(out/'native-fixture-private.json'),
                             '--output', str(out/'native-result-private.json')], check=True, capture_output=True, text=True)
    native_result = json.loads(native.stdout)
    if native_result['rankingMismatches'] or native_result['annotationMismatches']:
        raise ValueError('Native replay differs')
    _, dataset_hashes = verify_run(args.dataset)
    _, comparison_hashes = verify_run(args.comparison)
    with gzip.open(args.dataset/'dataset-private.json.gz', 'rt') as f:
        data = json.load(f)
    selected = json.loads((args.comparison/'selected-predictions-private.json').read_text())['single']
    frozen, frozen_private = frozen_audit(data, selected, args.bundle)
    write('frozen-audit-private.json', frozen_private)
    summary = {'captured_at_utc': now.isoformat(), 'database_max_seq': max_seq,
               'model_records': len(state['records']), 'native_parity': native_result,
               'focus_queries': [{k: v for k, v in d.items() if k not in {'current', 'top10', 'homepage_ranks'}} for d in details],
               'known_web_targets': len(pages), 'learned_web_labels': len(web_labels),
               'known_web_routes': dict(Counter(route(t, args.host) for t in pages.values())),
               'learned_web_routes': dict(Counter(route(pages[k], args.host) for k in web_labels)),
               'conversation_pair_similarity': {'pairs': len(similarities), 'min': min(similarities, default=None),
                                                 'max': max(similarities, default=None),
                                                 'strong_pairs': sum(v >= .7 for v in similarities)},
               'frozen_replay': frozen,
               'limits': ['Current-model counterfactuals, not a reconstruction of a dismissed menu.',
                          'Latest catalog is used only for current-model diagnosis, never frozen replay.',
                          'Rank scores are not calibrated intent probabilities.',
                          'Action-alias labels can describe real returns or different windows; this audit does not relabel raw events.',
                          'Frozen labels/cohort unchanged; training deletion and masks are ablations only.',
                          'Service-level success is not measured by the exact-URL benchmark.']}
    write('summary.json', summary)
    write('run.json', {'complete': True, 'dataset': str(args.dataset), 'comparison': str(args.comparison),
                       'dataset_hashes': dataset_hashes, 'comparison_hashes': comparison_hashes,
                       'arguments': {k: str(v) for k,v in vars(args).items()},
                       'runtime_modified': False, 'raw_data_modified': False, 'test_previously_seen': True})
    (out/'source-diagnose_action_granularity.py').write_bytes(Path(__file__).read_bytes())
    write('artifact-hashes.json', {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in out.iterdir() if p.is_file()})
    print(json.dumps({'run': str(out), 'model_records': summary['model_records'], 'native_parity': native_result,
                      'known_web_targets': len(pages), 'learned_web_labels': len(web_labels),
                      'self_action_training_records': frozen['self_action_training_records'],
                      'self_action_after_dropped_steps': frozen['self_action_after_dropped_steps']}))


if __name__ == '__main__':
    main()
