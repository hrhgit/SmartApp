"""Prepare a private causal dataset with semantic annotations and future labels.

Collection remains untouched. A narrowly audited, user-authorized removal of
candidateTargetIDs is the only allowed difference from the frozen raw source.
"""
import argparse
from collections import Counter, defaultdict, deque
import copy
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
from types import SimpleNamespace

import evaluate_context_knn as io
from known_inactivity import annotate_events, active_time, is_boundary
from diagnose_dwell_errors import frozen_core
from evaluate_evidence_model import clean_input, DENIED_BUNDLES
from evaluate_ordered_current import current_context
from reopenable_actions import truth_action_key, truth_metadata
from summarize_tuned_history import verify_run

VERSION = 'semantic-trajectory-dataset-v2'
SOURCE_FILES = ('known_inactivity.py', 'prepare_semantic_trajectory.py', 'semantic_annotations.py',
    'trajectory_supervision.py', 'describe_dwell.py', 'evaluate_context_knn.py',
    'diagnose_dwell_errors.py', 'evaluate_evidence_model.py',
    'evaluate_ordered_current.py', 'reopenable_actions.py',
    'compare_online_models.py', 'summarize_tuned_history.py')


def checked_redacted_snapshot(snapshot, expected):
    required = {'events.jsonl', 'navigation-samples.jsonl', 'target-history.jsonl', 'manifest.json'}
    allowed = required | {'candidate-list-redaction.json'}
    if set(expected) != required:
        raise ValueError('Unexpected original snapshot contract')
    actual = {}
    for line in (snapshot/'SHA256SUMS').read_text().splitlines():
        digest, name = line.split(maxsplit=1)
        name = name.strip().lstrip('*')
        if (name not in allowed or name in actual or Path(name).name != name
                or io.sha(snapshot/name) != digest):
            raise ValueError('Invalid current snapshot checksum')
        actual[name] = digest
    if not required <= actual.keys():
        raise ValueError('Missing required snapshot checksum')
    differences = {name for name, digest in expected.items() if actual.get(name) != digest}
    audit = {'expected_original': expected, 'current': actual, 'changed_files': sorted(differences),
             'checksum_manifest_hash': io.sha(snapshot/'SHA256SUMS')}
    if differences:
        if differences != {'events.jsonl'}:
            raise ValueError('Unexpected raw-source change')
        if 'candidate-list-redaction.json' not in actual:
            raise ValueError('Redaction marker must be included in current checksums')
        marker = json.loads((snapshot/'candidate-list-redaction.json').read_text())
        changed = marker['changed_file']
        if not (marker['operation'] == 'remove-collected-candidate-lists-v1'
                and marker['user_authorized'] is True
                and marker.get('old_training_provenance_requires_explicit_redaction_migration') is True
                and Path(changed['path']).resolve() == (snapshot/'events.jsonl').resolve()
                and changed['sha256_before'] == expected['events.jsonl']
                and changed['sha256_after'] == actual['events.jsonl']
                and changed['changed_rows'] == 1):
            raise ValueError('Redaction lineage does not explain source change')
        audit['authorized_redaction'] = marker
    return audit


def contains_collected_candidates(value):
    if isinstance(value, dict):
        return 'candidateTargetIDs' in value or any(contains_collected_candidates(v) for v in value.values())
    return isinstance(value, list) and any(contains_collected_candidates(v) for v in value)


def inputs_at_anchors(core, items, events, annotate):
    by_index = {q.index: (i, q) for i, q in enumerate(items)}
    if len(by_index) != len(items):
        raise ValueError('Duplicate input cutoff')
    catalog, session, result = {}, None, []
    for index, event in enumerate(events):
        if event['sessionID'] != session:
            catalog, session = {}, event['sessionID']
        if event['kind'] == 'target.update':
            target = event['data']['target']
            if core.admitted(target) and not core.synthetic(target):
                catalog[target['id']] = target
            else:
                catalog.pop(target['id'], None)
        if index not in by_index:
            continue
        position, q = by_index[index]
        if event['id'] != q.anchor_id or event['sessionID'] != q.session:
            raise ValueError('Input anchor identity/order mismatch')
        meta = copy.deepcopy(catalog.get(event['data'].get('targetID'), {}))
        if not meta or core.key_for(meta, q.session) != q.current:
            raise ValueError('Current-target metadata is not available at cutoff')
        meta['bundleID'] = q.current_app or meta.get('bundleID')
        context = current_context(q)
        if context is None:
            raise ValueError('Unresolved current origin')
        origin = context.targets[0]
        kind = 'window_class' if origin.startswith('window_class:') else origin.split(':', 1)[0]
        result.append({'position': position, 'index': q.index, 'ready_index': q.ready_index,
            'sample_id': q.sample_id, 'anchor_id': q.anchor_id, 'session': q.session,
            'segment': q.segment, 'origin': origin, 'app': q.current_app or '',
            'timestamp': q.timestamp, 'monotonic_ms': event['monotonicMs'],
            'target': meta, 'annotation': annotate(meta),
            'x': {'origin='+origin: 1., 'app='+(q.current_app or ''): 1., 'kind='+kind: 1., 'bias': 1.},
            'background_state': q.background, 'source_quality': q.source_quality,
            'coverage': q.coverage, 'strict': q.strict})
    if len(result) != len(items):
        raise ValueError('Missing as-of current inputs')
    return result


def anchor_overlap_counts(core, events, samples, observations, excluded_sessions, window_ms=30000):
    """Count prior *input* anchors without conditioning on their future labels.

    Every admitted natural sample anchor participates, including samples with
    censored/missing later outcomes. Session/capture boundaries reset history.
    The interval is (current monotonic time - 30 seconds, current input index).
    """
    events = annotate_events(events)
    natural = {s['anchorEventID'] for s in samples if s.get('origin') == 'natural_observed'}
    queries = {q['index']: q for q in observations}
    result, catalog, recent, session = {}, {}, defaultdict(deque), None
    for index, event in enumerate(events):
        if event['sessionID'] != session:
            session, catalog, recent = event['sessionID'], {}, defaultdict(deque)
        if is_boundary(event):
            recent.clear()
        if session in excluded_sessions:
            continue
        if event['kind'] == 'target.update':
            meta = event['data']['target']
            if (core.admitted(meta) and not core.synthetic(meta)
                    and meta.get('bundleID') not in DENIED_BUNDLES):
                catalog[meta['id']] = meta
            else:
                catalog.pop(meta['id'], None)
        if event['kind'] != 'context.change' or event['id'] not in natural:
            continue
        meta = catalog.get(event['data'].get('targetID'))
        if not meta:
            recent.clear()
            continue
        owner = event.get('app', {}).get('bundleID') or meta.get('bundleID')
        if owner in DENIED_BUNDLES:
            recent.clear()
            continue
        origin = ('window_class:' + owner if meta['kind'] == 'window' and owner
                  else None if meta['kind'] == 'window' else core.key_for(meta, session))
        if origin is None:
            recent.clear()
            continue
        now = active_time(event)
        prior = recent[origin]
        while prior and prior[0][0] <= now - window_ms:
            prior.popleft()
        if index in queries:
            query = queries[index]
            if query['origin'] != origin or query['anchor_id'] != event['id']:
                raise ValueError('Overlap anchor semantics differ from model input')
            result[query['position']] = {'prior_anchor_count': len(prior),
                'multiplier': 1 / math.sqrt(1 + len(prior)), 'window_ms': window_ms,
                'input_index': index, 'prior_max_index': prior[-1][1] if prior else None,
                'basis': 'past_natural_input_anchors_same_origin_no_outcome_filter'}
        prior.append((now, index))
    if set(result) != {q['position'] for q in observations}:
        raise ValueError('Missing causal overlap history')
    return result


def soft_overlap_labels(labels, overlap):
    """Change training weight only; preserve all arrivals, steps and targets."""
    result = copy.deepcopy(labels)
    for label in result:
        label['pre_overlap_weight'] = label['weight']
        label['weight'] *= overlap['multiplier']
        label['pre_overlap_basis'] = label['basis']
        label['basis'] = 'soft_kind_times_past_same_origin_anchor_overlap'
        label['overlap'] = dict(overlap)
    return result


def sorted_stream(stream):
    result = sorted(stream, key=lambda r: (r['ready_index'], r['position'],
                    r['record'].get('observed_step', 0),
                    r['record'].get('arrival_event_id', r['record'].get('outcome_id', ''))))
    for row in result:
        if not row['input_index'] < row['ready_index']:
            raise ValueError('Training outcome matured before its input')
    return result


def retained_reference_stream(stream):
    allowed = {'origin', 'app', 'timestamp', 'x', 'label', 'outcome_timestamp', 'outcome_id'}
    temporal = {'position', 'input_index', 'ready_index', 'outcome_index'}
    return sorted_stream([{**{k: v for k, v in row.items() if k in temporal},
                           'record': {k: v for k, v in row['record'].items() if k in allowed}}
                          for row in stream])


def freeze_evidence(rows, observations, reference_stream, compare):
    history, cursor, result = [], 0, []
    reference_stream = sorted_stream(reference_stream)
    priority = {'weak': 1, 'strong': 2, 'exact': 3}
    for row in sorted(rows, key=lambda r:r['position']):
        q = observations[row['position']]
        while cursor < len(reference_stream) and reference_stream[cursor]['ready_index'] < q['index']:
            history.append(reference_stream[cursor]); cursor += 1
        witnesses = {}
        for past in history:
            if past['input_index'] >= q['index'] or past['position'] >= row['position']:
                raise ValueError('Evidence input is not strictly before query')
            if past['record']['label'] != row['truth_action']:
                continue
            before = observations[past['position']]
            if before['origin'] == q['origin']:
                relation = {'level': 'exact', 'score': 1., 'matched_tags': []}
            else:
                relation = compare(q['annotation'], before['annotation'])
            if relation['level'] not in ('exact', 'strong', 'weak'):
                continue
            witness = {'source_position': past['position'], 'outcome_id': past['record']['outcome_id'],
                       'ready_index': past['ready_index'], 'relation': relation}
            key = past['record']['outcome_id']
            if key not in witnesses or (priority[relation['level']], relation['score']) > (
                    priority[witnesses[key]['relation']['level']], witnesses[key]['relation']['score']):
                witnesses[key] = witness
        original = row['diagnosis_after_outcome']['eligible']
        levels = {w['relation']['level'] for w in witnesses.values()}
        grade = ('original' if original else 'added_exact' if 'exact' in levels else
                 'semantic_strong' if 'strong' in levels else 'semantic_weak' if 'weak' in levels else 'unsupported')
        result.append({**copy.deepcopy(row), 'evidence_grade': grade, 'evidence_count': len(witnesses),
                       'witnesses': list(witnesses.values())})
    return result


def supervision_duplicates(stream, observations):
    groups, counts = defaultdict(set), Counter()
    for row in stream:
        record = row['record']
        key = (record.get('observed_step', 1), observations[row['position']]['origin'],
               record.get('arrival_event_id', record.get('outcome_id')))
        groups[key].add(record.get('action_key', record.get('label')))
        counts[key] += 1
    return {'unique_head_origin_outcomes': len(groups),
            'duplicate_rows_same_head_origin_outcome': sum(n - 1 for n in counts.values()),
            'conflicting_label_groups': sum(len(labels) > 1 for labels in groups.values())}


def step_coverage(future, positions):
    masks = [future[position]['audit']['step_mask'] for position in positions]
    return {'n': len(masks), 'available_by_step': {
                str(step + 1): sum(mask[step] for mask in masks) for step in range(3)},
            'observed_step_count_distribution': dict(Counter(str(sum(mask)) for mask in masks)),
            'all_three_observed': sum(all(mask) for mask in masks)}


def main():
    from semantic_annotations import annotate_target, compare_annotations
    from trajectory_supervision import derive_trajectory, PastQuickcheckHistory, TrajectoryEvents
    import describe_dwell as dwell
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    private_root = Path.home()/'Library/Application Support/SmartApp/training'
    if (not args.output.resolve().is_relative_to(private_root.resolve())
            or args.output.resolve() == private_root.resolve()):
        raise ValueError('Dataset output must be a new private SmartApp training run')
    if args.output.exists():
        raise ValueError('Refusing to overwrite an existing dataset run')
    source_bytes = {name: Path(__file__).with_name(name).read_bytes() for name in SOURCE_FILES}
    source_code_hashes = {name: hashlib.sha256(data).hexdigest() for name, data in source_bytes.items()}
    provenance, source_hashes = verify_run(args.source_run)
    base = Path(provenance['cleaning_run'])
    base_provenance, base_hashes = verify_run(base)
    if base_hashes != provenance['cleaning_hashes']:
        raise ValueError('Frozen cleaning provenance mismatch')
    snapshot = Path(provenance['snapshot'])
    migration = checked_redacted_snapshot(snapshot, provenance['input_hashes'])
    events = io.load_lines(snapshot/'events.jsonl')
    samples = io.load_lines(snapshot/'navigation-samples.jsonl')
    if migration.get('authorized_redaction') and len(events) != migration['authorized_redaction']['changed_file']['rows']:
        raise ValueError('Redaction changed event count')
    if contains_collected_candidates(events):
        raise ValueError('Redacted collection still contains candidate lists')
    core = frozen_core(base)
    items, observation_audit, arrivals, inventories, display = core.make_observations(
        events, samples, set(base_provenance['excluded_sessions']))
    if observation_audit != provenance['observation_audit']:
        raise ValueError('Frozen observation extraction differs after redaction')
    blocked = {core.key_for(e['data']['target'], e['sessionID']) for e in events
               if e['kind'] == 'target.update' and e['data']['target'].get('bundleID') in DENIED_BUNDLES}
    items = [clean_input(q, blocked) for q in items]
    with gzip.open(args.source_run/'worker-input-private.json.gz', 'rt') as f:
        previous = json.load(f)
    if len(items) != previous['item_count']:
        raise ValueError('Original observation membership changed')
    for pos, old in previous['query_items'].items():
        q = items[int(pos)]
        for key in ('index', 'ready_index', 'anchor_id', 'sample_id', 'session', 'current',
                    'current_app', 'next_event_id', 'foreground', 'background', 'history',
                    'history_keys', 'source_quality', 'reference_max_index', 'timestamp'):
            if json.loads(json.dumps(getattr(q, key))) != old[key]:
                raise ValueError('Redaction changed prediction timing or current state: '+key)
    observations = inputs_at_anchors(core, items, events, annotate_target)
    overlap = anchor_overlap_counts(core, events, samples, observations,
                                    set(base_provenance['excluded_sessions']))
    old_items = {int(i): SimpleNamespace(**q) for i, q in previous['query_items'].items()}
    rows = [{**r, 'truth_action': truth_action_key(r, old_items), 'truth_metadata': truth_metadata(r, old_items)}
            for r in previous['cohorts']['5']]
    event_index = {e['id']: i for i, e in enumerate(events)}
    if len(event_index) != len(events):
        raise ValueError('Duplicate raw event identity')
    for stream in previous['streams'].values():
        for row in stream:
            query = items[row['position']]
            if (row['input_index'] != query.index or row['ready_index'] != query.ready_index
                    or event_index[row['record']['outcome_id']] != row['outcome_index']):
                raise ValueError('Frozen stream temporal indices changed after redaction')
    reference_stream = retained_reference_stream(previous['streams']['5'])
    original_cleaning_stream = retained_reference_stream(previous['streams']['kind_app5_resource2'])
    evidence = freeze_evidence(rows, observations, reference_stream, compare_annotations)
    if sum(r['evidence_grade'] == 'original' for r in evidence) != 62 or len(evidence) != 162:
        raise ValueError('Original evaluation cohort changed')
    # The original extraction code is deterministic over authorized current raw
    # events; no future metadata enters observations above.
    dwell.core = core
    visits, app_visits, visit_audit = dwell.extract_visits(events, samples, set(base_provenance['excluded_sessions']))
    visits_by_id = {v['arrival_event_id']: v for v in visits}
    events_by_id = TrajectoryEvents({e['id']: e for e in events}, event_index)
    policies = ['resolved', 'hard_kind', 'soft_kind', 'soft_quickcheck', 'soft_overlap']
    streams, ledgers = {p: [] for p in policies}, {p: [] for p in policies}
    future = {}
    quickcheck_history = PastQuickcheckHistory(visits, event_index)
    for position, query in enumerate(items):
        labels_by_policy, audits_by_policy = {}, {}
        for policy in policies:
            if policy == 'soft_overlap':
                labels = soft_overlap_labels(labels_by_policy['soft_kind'], overlap[position])
                audit = copy.deepcopy(audits_by_policy['soft_kind'])
                audit.update(policy={'id': policy, 'base': 'soft_kind', 'window_ms': 30000},
                             overlap=overlap[position])
                audit['observed'] = soft_overlap_labels(audit['observed'], overlap[position])
            else:
                labels, audit = derive_trajectory(query, visits_by_id, events_by_id, event_index,
                    policy=policy, history=quickcheck_history if policy == 'soft_quickcheck' else None)
            labels_by_policy[policy], audits_by_policy[policy] = labels, audit
            ledgers[policy].append({'position': position, **audit})
            if policy == 'resolved':
                future[position] = {'labels': labels, 'audit': audit}
            for label in labels:
                if not (query.index < label['arrival_index'] <= label['target_evidence_index']
                        <= label['ready_index'] == query.ready_index):
                    raise ValueError('Trajectory label evidence violates maturity boundary')
                streams[policy].append({'position': position, 'input_index': query.index,
                    'ready_index': label['ready_index'], 'record': label})
    streams = {name: sorted_stream(stream) for name, stream in streams.items()}
    if checked_redacted_snapshot(snapshot, provenance['input_hashes']) != migration:
        raise ValueError('Snapshot changed during dataset preparation')
    if any(io.sha(Path(__file__).with_name(name)) != digest for name, digest in source_code_hashes.items()):
        raise ValueError('Source code changed during dataset preparation')
    os.umask(0o077)
    args.output.mkdir(parents=True, mode=0o700, exist_ok=False)
    payload = {'observations': observations, 'evaluation': evidence, 'future': future,
               'streams': streams, 'reference_stream': reference_stream,
               'original_cleaning_stream': original_cleaning_stream,
               'trajectory_ledgers': ledgers, 'overlap_history': overlap}
    with gzip.open(args.output/'dataset-private.json.gz', 'wt') as f:
        json.dump(payload, f, ensure_ascii=False)
    io.write_json(args.output/'evidence-private.json', evidence)
    io.write_json(args.output/'descriptive-private.json', {'independent_arrivals_by_kind': arrivals,
        'background_inventory_observations_by_kind': inventories, 'target_metadata': display,
        'counts_are_separate_not_additive': True,
        'arrival_count_basis': 'admitted context.change excluding resolution_update and controlled sessions',
        'resolution_updates_do_not_increment_arrival_counts': True})
    summary = {'original_observations': len(items), 'evaluation_n': len(rows),
        'evidence_grades': dict(Counter(r['evidence_grade'] for r in evidence)),
        'original_cohort_without_current_reference_witness': sum(
            r['evidence_grade'] == 'original' and r['evidence_count'] == 0 for r in evidence),
        'visit_extraction': visit_audit, 'observation_audit': observation_audit,
        'streams': {p: len(s) for p,s in streams.items()}, 'source_parity_queries': len(previous['query_items']),
        'steps_all_observations': step_coverage(future, range(len(items))),
        'steps_evaluation': step_coverage(future, [r['position'] for r in evidence]),
        'steps_by_evidence_grade': {grade: step_coverage(future,
            [r['position'] for r in evidence if r['evidence_grade'] == grade])
            for grade in ('original', 'added_exact', 'semantic_strong', 'semantic_weak', 'unsupported')},
        'supervision_duplicate_audit': {name: supervision_duplicates(stream, observations)
            for name, stream in {**streams, 'original_cleaning_stream': original_cleaning_stream}.items()},
        'overlap_input_count_distribution': dict(Counter(str(v['prior_anchor_count']) for v in overlap.values())),
        'overlap_is_training_weight_only': True,
        'same_outcome_evidence_witnesses_deduplicated': True,
        'trajectory_rows_preserved_before_model_head_deduplication': True,
        'future_steps_bounded_by_original_30s_horizon': True,
        'input_feature_metadata_is_asof': True}
    io.write_json(args.output/'summary.json', summary)
    for name, data in source_bytes.items():
        (args.output/('source-'+name)).write_bytes(data)
    (args.output/'frozen-core-source.py').write_bytes((base/'core-source.py').read_bytes())
    io.write_json(args.output/'source-hashes.json', source_code_hashes)
    if checked_redacted_snapshot(snapshot, provenance['input_hashes']) != migration:
        raise ValueError('Snapshot changed during artifact writing')
    if any(io.sha(Path(__file__).with_name(name)) != digest for name, digest in source_code_hashes.items()):
        raise ValueError('Source code changed during artifact writing')
    io.write_json(args.output/'run.json', {'version': VERSION, 'complete': True, 'source_run': str(args.source_run),
        'source_artifact_hashes': source_hashes, 'snapshot': str(snapshot), 'migration': migration,
        'source_hashes': source_code_hashes, 'frozen_core_hash': base_hashes['core-source.py'],
        'raw_modified': False, 'candidate_lists_used': False, 'test_previously_seen': True,
        'snapshot_and_source_hashes_verified_before_and_after': True,
        'fixed_primary_truth_and_membership': True,
        'future_labels_and_evidence_diagnosis_are_not_model_inputs': True,
        'model_output_not_restricted_by_inventory_or_origin_support': True})
    io.write_json(args.output/'artifact-hashes.json', {p.name: io.sha(p) for p in args.output.iterdir() if p.is_file()})
    if args.output.stat().st_mode & 0o777 != 0o700:
        raise ValueError('Private output directory permissions differ from 0700')
    for path in args.output.rglob('*'):
        if path.stat().st_mode & 0o777 != (0o700 if path.is_dir() else 0o600):
            raise ValueError('Private artifact permission mismatch')
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == '__main__':
    main()
