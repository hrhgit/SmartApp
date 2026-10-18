"""Fixed-cohort, causal v1/v2 comparison and native parity; private outputs only."""
import argparse
from collections import Counter
from copy import deepcopy
import gzip
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import subprocess

from compare_semantic_trajectory import groups, training_rows
from diagnose_action_granularity import NativeSimilarity, metric
from navigation_hierarchy import NavigationHierarchy, VERSION, SERVICE, ENTRY, learning_record, origin, service
from semantic_annotations import annotate_target
from semantic_online import SemanticStatistics, rank
from summarize_tuned_history import verify_run


def run_native(cli, out, name, fixture):
    path = out/(name+'-fixture-private.json')
    path.write_text(json.dumps(fixture))
    response = subprocess.run([str(cli.resolve()), 'validate-model', '--input', str(path),
                               '--output', str(out/(name+'-native-private.json'))],
                              capture_output=True, text=True, check=True)
    result = json.loads(response.stdout)
    if result['rankingMismatches'] or result['annotationMismatches']:
        raise ValueError('Native parity failed: ' + name)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', required=True, type=Path)
    parser.add_argument('--comparison', required=True, type=Path)
    parser.add_argument('--native-cli', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--focus-bundle', required=True)
    args = parser.parse_args()
    root = Path.home()/'Library/Application Support/SmartApp'
    out = args.output.expanduser().resolve()
    if not out.is_relative_to((root/'training').resolve()) or out == (root/'training').resolve() or out.exists():
        parser.error('Use a new private training directory')
    os.umask(0o077)
    out.mkdir(mode=0o700)
    def write(name, value):
        (out/name).write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n')

    provenance, hashes = verify_run(args.dataset)
    _, comparison_hashes = verify_run(args.comparison)
    with gzip.open(args.dataset/'dataset-private.json.gz', 'rt') as handle:
        data = json.load(handle)
    selected = json.loads((args.comparison/'selected-predictions-private.json').read_text())['single']
    rows = [r for r in training_rows(data, 'hard_kind') if r['head'] == 0]
    updates = [{'readyIndex': r['ready_index'], 'record': learning_record(r, data['observations'][r['position']])} for r in rows]
    model = NavigationHierarchy()
    baseline = SemanticStatistics(semantic_strength=16, global_strength=1, recency_mix=.25, similarity=NativeSimilarity())
    predictions = {k: {} for k in ['v1_exact', 'v2_exact', 'v1_service', 'v2_service']}
    queries, catalog, cursor, aliases = [], {}, 0, Counter()
    events_path = Path(provenance['snapshot'])/'events.jsonl'
    expected_hash = provenance['migration']['current']['events.jsonl']
    if hashlib.sha256(events_path.read_bytes()).hexdigest() != expected_hash:
        raise ValueError('Frozen raw events changed')
    with events_path.open() as handle:
        events = enumerate(handle)
        next_event = next(events, None)
        for row in sorted(data['evaluation'], key=lambda r:r['position']):
            pos = row['position']; q = data['observations'][pos]
            while next_event is not None and next_event[0] <= q['index']:
                index, line = next_event; event = json.loads(line)
                if index == q['index'] and event['id'] != q['anchor_id']:
                    raise ValueError('Historical event indices changed')
                if event['kind'] == 'target.update':
                    target = event['data']['target']; catalog[target['id']] = target
                next_event = next(events, None)
            while cursor < len(updates) and updates[cursor]['readyIndex'] < q['index']:
                r = updates[cursor]['record']
                assert data['observations'][rows[cursor]['position']]['index'] < q['index']
                model.learn(r)
                baseline.learn({'origin': r['origin'], 'annotation': r['annotation'], 'timestamp': r['outcomeTimestamp']},
                    r['label'], step=0, weight=r['weight'], outcome_id=r['outcomeID'], outcome_timestamp=r['outcomeTimestamp'])
                cursor += 1
            old = rank(baseline.predict(q, 0))
            if old != selected[str(pos)]['single']:
                raise ValueError('Frozen v1 predictions changed')
            new = model.ranked(q['target'])
            # Only exact pre-cutoff metadata can resolve the abstract service action.
            home = next((k for k,t in sorted(catalog.items()) if t.get('locator') == ENTRY and t.get('kind') == 'page'), None)
            resolved = [home if k == SERVICE and home else k for k in new]
            predictions['v1_exact'][pos], predictions['v2_exact'][pos] = old, resolved
            predictions['v1_service'][pos] = [service(catalog.get(k)) or k for k in old]
            predictions['v2_service'][pos] = [service(catalog.get(k)) or k for k in new]
            queries.append({'position': pos, 'index': q['index'], 'origin': q['origin'], 'target': q['target'], 'expected': new})
            if q['target']['kind'] in {'app', 'window'}:
                self_label = 'app:' + q['target']['bundleID']
                aliases['coarse_inputs'] += 1
                aliases['v1_current_app_top4'] += self_label in old
                aliases['v2_current_app_top4'] += self_label in new
    grouped = groups(data['evaluation'])
    scores = {}
    for name, preds in predictions.items():
        scores[name] = {}
        for group, members in grouped.items():
            transformed = [{**r, 'truth_action': service(r['truth_metadata']) or r['truth_action']} for r in members] if name.endswith('_service') else members
            scores[name][group] = metric(transformed, preds)
    parity = run_native(args.native_cli, out, 'frozen', {'modelVersion': VERSION, 'updates': updates, 'queries': queries, 'annotations': []})
    complete = NavigationHierarchy(r['record'] for r in updates)
    write('derived-training-private.json', complete.records)
    write('predictions-private.json', predictions)

    # Separate present-time reconstruction; never used by the historical replay.
    state_bytes = (root/'models/semantic-statistics-hard-kind-v1/state.json').read_bytes()
    (out/'legacy-state-private.json').write_bytes(state_bytes)
    state = json.loads(state_bytes)
    catalog, contexts = {}, []
    with sqlite3.connect((root/'context-v2.sqlite').as_uri()+'?mode=ro', uri=True) as conn:
        conn.execute('BEGIN')
        for raw, in conn.execute("select json from events where kind='target.update' order by seq"):
            t = json.loads(raw)['data']['target']; catalog[t['id']] = t
        contexts = [json.loads(raw) for raw, in conn.execute("select json from events where kind='context.change' and app_bundle_id=? order by seq desc limit 50", (args.focus_bundle,))]
    enriched = deepcopy(state['records'])
    for r in enriched:
        t = catalog.get(r['label'])
        if t and t['kind'] != 'window' and t.get('admission'):
            r['destination'] = annotate_target(t)['identity']
    present = NavigationHierarchy(enriched)
    app = catalog['app:' + args.focus_bundle]
    window = next(catalog[e['data']['targetID']] for e in contexts if catalog.get(e['data']['targetID'], {}).get('kind') == 'window')
    window = {**window, 'bundleID': args.focus_bundle}
    current_queries = [{'position': i, 'index': 1, 'origin': origin(t), 'target': t, 'expected': present.ranked(t)} for i,t in enumerate([app,window])]
    current_parity = run_native(args.native_cli, out, 'current', {'modelVersion': VERSION, 'updates': [{'readyIndex': 0, 'record': r} for r in enriched], 'queries': current_queries, 'annotations': []})
    write('current-derived-records-private.json', present.records)
    write('current-catalog-private.json', catalog)
    write('current-queries-private.json', current_queries)
    summary = {'version': VERSION, 'baseline_parity_mismatches': 0, 'native_frozen': parity, 'native_current': current_parity,
               'scores': scores, 'current_app_actions': dict(aliases),
               'frozen_records': len(complete.records), 'frozen_dispositions': dict(Counter(r['navigationDisposition'] for r in complete.records)),
               'current_records': len(present.records), 'current_dispositions': dict(Counter(r['navigationDisposition'] for r in present.records)),
               'service_supervision_records': sum(r.get('navigationLabel') == SERVICE for r in present.records),
               'limits': ['Exact and service metrics are different objectives.', 'Previously viewed cohort; no unseen generalization claim.',
                          'Specificity rule fixed before replay; not a calibrated confidence guarantee.', 'Length-one current input baseline retained.']}
    write('summary.json', summary)
    write('run.json', {'complete': True, 'dataset': str(args.dataset), 'dataset_hashes': hashes, 'comparison_hashes': comparison_hashes,
                      'raw_events_sha256': expected_hash, 'arguments': {k: str(v) for k,v in vars(args).items()},
                      'raw_data_modified': False, 'runtime_modified': False})
    for name in ['evaluate_navigation_hierarchy.py', 'navigation_hierarchy.py', 'diagnose_action_granularity.py', 'semantic_online.py', 'semantic_annotations.py']:
        (out/('source-'+name)).write_bytes((Path(__file__).parent/name).read_bytes())
    write('artifact-hashes.json', {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in out.iterdir() if p.is_file()})
    print(json.dumps({'run': str(out), 'native_frozen': parity, 'native_current': current_parity,
                      'frozen_dispositions': summary['frozen_dispositions'], 'current_app_actions': dict(aliases),
                      'scores_original': {k:v['original'] for k,v in scores.items()}}))


if __name__ == '__main__':
    main()
