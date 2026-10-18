"""Freeze one new chronological validation set before any model comparison."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import os
from pathlib import Path

import describe_dwell as dwell
import evaluate_context_knn as core
import evaluation_contract as scoring
from evaluate_dwell_cleaning import derive_label
from evaluate_evidence_model import DENIED_BUNDLES, clean_input
from prepare_semantic_trajectory import inputs_at_anchors
from prepare_multibackground_dataset import checked_snapshot, require_prefix, evidence_for_new
from semantic_annotations import annotate_target
from trajectory_supervision import TrajectoryEvents, derive_trajectory, action_key
from review_historical_models import ROOT, Sources, sha, write


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--snapshot', type=Path, required=True)
    parser.add_argument('--previous-snapshot', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(); out = args.output.resolve()
    if out.exists() or out == ROOT.resolve() or not out.is_relative_to(ROOT.resolve()):
        parser.error('Use a new private training directory')
    sources = Sources()
    hashes, manifest = checked_snapshot(args.snapshot)
    events = core.load_lines(sources.check(args.snapshot/'events.jsonl'))
    previous = core.load_lines(sources.check(args.previous_snapshot/'events.jsonl'))
    require_prefix(previous, events)
    boundary = len(previous)
    samples = core.load_lines(sources.check(args.snapshot/'navigation-samples.jsonl'))
    if len(events) != manifest['events'] or len(samples) != manifest['samples']:
        raise ValueError('Snapshot counts mismatch')
    os.umask(0o077); out.mkdir(mode=0o700)
    write(out/'plan.json', {'boundary_event_index': boundary,
        'boundary_timestamp_ms': previous[-1]['timestampMs'],
        'selection_rule': 'all eligible natural prediction anchors after the previous complete snapshot',
        'training': 'common hard_kind first retained navigation; strict maturity; incremental updates during validation',
        'scoring': 'stable 5s destination within observed active 30s horizon; common legality audit',
        'model_selection': 'historical parameters frozen; no new validation tuning',
        'raw_snapshot': str(args.snapshot), 'snapshot_hashes': hashes,
        'previous_snapshot': str(args.previous_snapshot), 'scoring_contract': scoring.contract()})
    items, extraction, *_ = core.make_observations(events, samples, core.KNOWN_TEST_SESSIONS)
    blocked = {core.key_for(e['data']['target'], e['sessionID']) for e in events
        if e['kind'] == 'target.update' and e['data']['target'].get('bundleID') in DENIED_BUNDLES}
    items = [clean_input(q, blocked) for q in items if q.current_app not in DENIED_BUNDLES]
    observations = inputs_at_anchors(core, items, events, annotate_target)
    event_index = {e['id']: i for i, e in enumerate(events)}
    lookup = TrajectoryEvents({e['id']: e for e in events}, event_index)
    visits, _, visit_audit = dwell.extract_visits(events, samples, core.KNOWN_TEST_SESSIONS)
    by_visit = {v['arrival_event_id']: v for v in visits}
    stream, reference, evaluation, ledger = [], [], [], []
    audit = Counter()
    for pos, q in enumerate(items):
        labels, details = derive_trajectory(q, by_visit, lookup, event_index, policy='hard_kind')
        ledger.append({'position': pos, **details})
        for label in labels:
            if label['target'].get('bundleID') not in DENIED_BUNDLES:
                stream.append({'position': pos, 'input_index': q.index,
                               'ready_index': q.ready_index, 'record': label})
        scored, why = derive_label(q, by_visit, lookup, 5)
        if q.index >= boundary:
            audit[why['status']] += 1
        if scored is None or scored.truth_meta.get('bundleID') in DENIED_BUNDLES:
            continue
        truth = action_key(scored.truth_meta)
        if truth is None:
            continue
        reference.append({'position': pos, 'input_index': q.index, 'ready_index': q.ready_index,
            'record': {'origin': observations[pos]['origin'], 'label': truth,
                'outcome_id': scored.next_event_id,
                'outcome_timestamp': lookup[scored.next_event_id]['timestampMs']}})
        if q.index >= boundary:
            evaluation.append({'position': pos, 'sample_id': q.sample_id, 'split': 'new_test',
                'truth_action': truth, 'truth_metadata': scored.truth_meta, 'input_index': q.index,
                'label_ready_index': q.ready_index, 'evidence_grade': 'unsupported', 'strict': q.strict})
    evidence_for_new(evaluation, observations, reference)
    stream.sort(key=lambda r: (r['ready_index'], r['position'], r['record']['observed_step']))
    data = {'observations': observations, 'evaluation': evaluation, 'streams': {'hard_kind': stream},
            'reference_stream': reference, 'original_cleaning_stream': []}
    rows = scoring.prepare_evaluation(data, out)
    if not evaluation or not any(scoring.eligible(r) for r in rows):
        raise ValueError('No new legal validation observations')
    write(out/'dataset-private.json.gz', data)
    write(out/'label-ledger-private.json.gz', ledger)
    write(out/'observation-objects-private.json.gz', [q.__dict__ for q in items])
    summary = {'scope': scoring.report_scope(rows), 'sizes': scoring.cohort_sizes(scoring.cohorts(rows)),
        'observations': len(observations), 'new_observations': sum(q.index >= boundary for q in items),
        'training_rows': len(stream), 'new_label_audit': dict(audit), 'extraction': extraction,
        'visit_audit': visit_audit, 'boundary_event_index': boundary, 'new_events': len(events)-boundary,
        'validation_range_utc': [core.utc(observations[r['position']]['timestamp']) for r in (evaluation[0], evaluation[-1])],
        'all_validation_after_previous_snapshot': all(r['input_index'] >= boundary for r in evaluation)}
    write(out/'summary.json', summary)
    sources.verify_unchanged()
    for path in Path(__file__).parent.glob('*.py'):
        (out/('source-'+path.name)).write_bytes(path.read_bytes())
    write(out/'run.json', {'complete': True, 'created_at': datetime.now(timezone.utc).isoformat(),
        'snapshot': str(args.snapshot.resolve()), 'snapshot_hashes': hashes,
        'previous_snapshot': str(args.previous_snapshot.resolve()), 'source_files': sources.accessed,
        'boundary_event_index': boundary, 'scoring_contract': scoring.contract(), 'raw_modified': False})
    write(out/'artifact-hashes.json', {p.name: sha(p) for p in out.iterdir() if p.is_file()})
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == '__main__':
    main()
