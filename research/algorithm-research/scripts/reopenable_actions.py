"""As-of executable action candidates and explicit multi-level evaluation.

Generic window destinations become activate-or-launch-app actions only when an
admitted application target was already known. Pages, directories and files keep
their exact locator identity. No operation is executed by this module.

The frozen replay resets its target catalog at every session. This adapter may
restore previously admitted app/locator targets from earlier target.update
events, while honoring subsequent explicit rejection or loss of qualification.
Historical locators are a reopening proxy, not proof of current reachability.
"""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
import json
from urllib.parse import urlsplit

import compare_online_models as comparison
from evaluate_evidence_model import DENIED_BUNDLES
from summarize_tuned_history import verify_run


VERSION = 'reopenable-actions-v1'
CONCRETE_KINDS = {'page', 'directory', 'file'}


def bundle_of(meta):
    """Read only explicit ownership; a URL does not imply a browser owner."""
    bundle = meta.get('_bundle') or meta.get('bundleID')
    if bundle:
        return bundle
    if meta.get('kind') == 'app' and str(meta.get('id', '')).startswith('app:'):
        return meta['id'][4:] or None
    return None


def valid_locator(meta):
    locator = meta.get('locator')
    if not isinstance(locator, str) or not locator:
        return False
    kind = meta.get('kind')
    parsed = urlsplit(locator)
    if kind == 'page':
        return parsed.scheme.lower() in {'http', 'https'} and bool(parsed.hostname)
    if kind in {'directory', 'file'}:
        return locator.startswith('/') or (parsed.scheme == 'file' and parsed.path.startswith('/'))
    return False


def _qualified(core, meta):
    if not core.admitted(meta) or core.synthetic(meta) or bundle_of(meta) in DENIED_BUNDLES:
        return False
    admission = meta.get('admission', {})
    basis, version = admission.get('basis'), admission.get('policyVersion')
    kind, owner = meta.get('kind'), bundle_of(meta)
    # Match the admitted evidence categories in TargetEligibility.swift; an
    # arbitrary truthy basis (including an unknown or negative reason) is not
    # sufficient for restoring a target beyond the frozen session catalog.
    if kind == 'app':
        return bool(owner and basis in {'standalone_application', 'independent_work_window'})
    if kind == 'window':
        return bool(owner and (
            basis in {'standard_work_window', 'supported_browser_window'}
            or (basis == 'supported_finder_window' and owner == 'com.apple.finder'
                and version in {'target-eligibility-v2', 'target-eligibility-v3', 'target-eligibility-v4'})
            or (basis == 'supported_qspace_window' and owner == 'com.jinghaoshe.qspace.pro'
                and version in {'target-eligibility-v3', 'target-eligibility-v4'})))
    if kind == 'page':
        return basis == 'supported_browser_page' and valid_locator(meta)
    if kind in {'directory', 'file'}:
        if not valid_locator(meta) or urlsplit(meta['locator']).scheme != 'file':
            return False
        return bool(basis == 'registered_local_target'
            or (kind == 'file' and basis == 'observed_window_document' and owner
                and version == 'target-eligibility-v4')
            or (kind == 'directory' and basis == 'observed_finder_directory'
                and owner == 'com.apple.finder'
                and version in {'target-eligibility-v2', 'target-eligibility-v3', 'target-eligibility-v4'})
            or (kind == 'directory' and basis == 'observed_qspace_directory'
                and owner == 'com.jinghaoshe.qspace.pro'
                and version in {'target-eligibility-v3', 'target-eligibility-v4'}))
    return False


def _app_key(meta):
    bundle = bundle_of(meta)
    # A conflicting app ID/owner is not a confirmed launchable application.
    if meta.get('kind') != 'app' or not bundle or meta.get('id') != 'app:' + bundle:
        return None
    return meta['id']


def _reopenable_key(core, meta):
    if not _qualified(core, meta):
        return None
    if meta['kind'] == 'app':
        return _app_key(meta)
    if meta['kind'] in CONCRETE_KINDS and valid_locator(meta):
        return meta['id']
    return None


def action_label(label):
    if label.startswith('window_class:'):
        return 'app:' + label[len('window_class:'):]
    return label


def _load_events(core, provenance, events):
    if events is None:
        snapshot = Path(provenance['snapshot'])
        hashes, _ = core.verified_snapshot(snapshot)
        if hashes != provenance['input_hashes']:
            raise ValueError('Action reconstruction snapshot hashes changed')
        events = core.load_lines(snapshot / 'events.jsonl')
    excluded = provenance.get('excluded_sessions')
    if excluded is None and provenance.get('cleaning_run'):
        source, hashes = verify_run(Path(provenance['cleaning_run']))
        if provenance.get('cleaning_hashes') is not None and hashes != provenance['cleaning_hashes']:
            raise ValueError('Cleaning provenance changed')
        excluded = source.get('excluded_sessions', [])
    return events, set(excluded or [])


def _plan(meta, evidence, source_window_keys):
    kind = meta['kind']
    if kind == 'app':
        return {'type': 'activate_or_launch_app', 'target_level': 'app',
                'bundle_id': bundle_of(meta),
                'observed_window_candidates': sorted(source_window_keys),
                'guarantees_specific_window': False, 'guarantees_new_window': False,
                'reachability': 'historically_admitted_application_not_runtime_verified',
                'evidence': evidence}
    return {'type': {'page': 'open_url', 'directory': 'open_directory', 'file': 'open_file'}[kind],
            'target_level': kind, 'locator': meta['locator'],
            'reachability': 'asof_known_locator_not_runtime_verified', 'evidence': evidence}


def adapt_dataset(core, items, cohorts, streams, provenance, *, events=None):
    """Return action items, prediction inputs/mappings, streams, private audit.

    ``events`` is an optional already-verified event list for callers retaining
    the frozen snapshot in memory. Indices must match the supplied observations.
    Training labels are converted only after an admitted owning app is known by
    their label-maturity index. Input current/current_app remain unchanged.
    """
    events, excluded_sessions = _load_events(core, provenance, events)
    by_index = {}
    for position, query in enumerate(items):
        by_index.setdefault(query.index, []).append((position, query))
        if query.index >= len(events) or events[query.index]['id'] != query.anchor_id:
            raise ValueError('Action event indices do not match observation anchors')
        if query.reference_max_index > query.index:
            raise ValueError('Future input reference')
    maturity_indices = {row['ready_index'] for stream in streams.values() for row in stream}
    checkpoints = sorted(set(by_index) | maturity_indices)
    registry, rejected, window_owners, latest_updates = {}, set(), {}, {}
    apps_at_maturity = {}
    action_items = [None] * len(items)
    positions_audit, totals = {}, Counter()
    cursor = 0
    for cutoff in checkpoints:
        if cutoff >= len(events):
            raise ValueError('Maturity index outside verified events')
        while cursor <= cutoff:
            event = events[cursor]
            if event['kind'] == 'target.update' and event.get('sessionID') not in excluded_sessions:
                meta = deepcopy(event['data']['target'])
                target_id = meta.get('id')
                if target_id:
                    latest_updates[target_id] = cursor
                    key = _reopenable_key(core, meta)
                    if key:
                        registry[target_id] = {'meta': meta, 'index': cursor,
                                               'session': event.get('sessionID')}
                        rejected.discard(target_id)
                    else:
                        registry.pop(target_id, None)
                        if not _qualified(core, meta) or meta.get('kind') in CONCRETE_KINDS | {'app'}:
                            rejected.add(target_id)
                        elif meta.get('kind') == 'window':
                            rejected.discard(target_id)
                    if meta.get('kind') == 'window':
                        window_key = core.key_for(meta, event.get('sessionID', ''))
                        owner = bundle_of(meta) if _qualified(core, meta) else None
                        if owner:
                            window_owners[window_key] = owner
                        else:
                            window_owners.pop(window_key, None)
            cursor += 1
        known_apps = {bundle_of(entry['meta']): entry
                      for entry in registry.values() if entry['meta']['kind'] == 'app'}
        if cutoff in maturity_indices:
            apps_at_maturity[cutoff] = set(known_apps)
        for position, query in by_index.get(cutoff, []):
            original = query.candidates
            admitted_apps = dict(known_apps)
            # Current frozen candidate metadata may carry an admitted app even
            # when a synthetic unit fixture omits its target.update event.
            for key, meta in original.items():
                if (meta.get('id') not in rejected and _qualified(core, meta)
                        and _app_key(meta)):
                    admitted_apps[bundle_of(meta)] = {
                        'meta': meta, 'index': latest_updates.get(meta['id']),
                        'session': query.session, 'candidate_key': key}
            proposed = {key: {'meta': entry['meta'], 'target_update_index': entry['index'],
                              'known_session': entry['session'], 'source': 'historical_target_update'}
                        for key, entry in registry.items()}
            for key, meta in original.items():
                if meta.get('id') in rejected:
                    continue
                action_key = _reopenable_key(core, meta)
                if action_key:
                    proposed[action_key] = {'meta': meta, 'target_update_index': latest_updates.get(meta['id']),
                                            'known_session': query.session, 'source': 'current_candidate',
                                            'original_candidate_key': key}
            window_keys = {}
            for key, meta in original.items():
                if (meta.get('kind') != 'window' or not _qualified(core, meta)
                        or meta.get('id') in rejected):
                    continue
                owner = bundle_of(meta)
                if owner in admitted_apps:
                    app = admitted_apps[owner]
                    app_key = _app_key(app['meta'])
                    if app_key not in proposed:
                        proposed[app_key] = {'meta': app['meta'], 'target_update_index': app['index'],
                                             'known_session': app['session'], 'source': 'asof_admitted_owning_app'}
                    window_keys.setdefault(app_key, []).append(key)
                else:
                    totals['window_candidates_without_admitted_app'] += 1
            candidates, key_map, evidence_by_key = {}, {}, {}
            for key, source in proposed.items():
                # Only exclude an identical concrete current destination. An app
                # action remains available when current is a page or window.
                if key == query.current:
                    continue
                meta = deepcopy(source['meta'])
                meta['_bundle'] = bundle_of(meta)
                evidence = {k: v for k, v in source.items() if k != 'meta'}
                if evidence['target_update_index'] is not None:
                    assert evidence['target_update_index'] <= query.index
                meta['_action_plan'] = _plan(meta, evidence, window_keys.get(key, []))
                candidates[key] = meta
                evidence_by_key[key] = evidence
            for key, meta in original.items():
                if key in candidates:
                    key_map[key] = key
                elif (meta.get('kind') == 'window' and bundle_of(meta)
                      and meta.get('id') not in rejected and _qualified(core, meta)):
                    app_key = 'app:' + bundle_of(meta)
                    if app_key in candidates:
                        key_map[key] = app_key
            # A current-window arrival can collapse to an app that deliberately
            # remains a candidate. It is current state, not a previous visit:
            # filter before aggregation so it cannot overwrite an older visit.
            removed_at_cutoff = [key for key, timestamp in query.last_visits.items()
                                 if timestamp == query.timestamp]
            removed_after_cutoff = [key for key, timestamp in query.last_visits.items()
                                    if timestamp > query.timestamp]
            prior_visits = {key: timestamp for key, timestamp in query.last_visits.items()
                            if timestamp < query.timestamp}
            last_visits = dict(prior_visits)
            asof_owners = {**window_owners, **{key: bundle_of(meta) for key, meta in original.items()
                                             if meta.get('kind') == 'window' and bundle_of(meta)}}
            for key, timestamp in prior_visits.items():
                if key in asof_owners:
                    app_key = 'app:' + asof_owners[key]
                    if app_key in candidates:
                        last_visits[app_key] = max(last_visits.get(app_key, timestamp), timestamp)
            adapted = replace(query, candidates=candidates, last_visits=last_visits,
                              recency={key: last_visits[key] for key in candidates if key in last_visits})
            adapted._reopen_original_candidates = deepcopy(original)
            adapted._reopen_prediction_key_map = key_map
            adapted._reopen_original_truth_meta = deepcopy(query.truth_meta)
            action_items[position] = adapted
            current_action_keys = set(key_map.values())
            added = sorted(set(candidates) - current_action_keys)
            restored_apps = [key for key in added if candidates[key]['kind'] == 'app']
            restored_concrete = [key for key in added if candidates[key]['kind'] in CONCRETE_KINDS]
            totals.update(original_candidate_mentions=len(original), action_candidate_mentions=len(candidates),
                          restored_app_mentions=len(restored_apps), restored_concrete_mentions=len(restored_concrete),
                          collapsed_window_mentions=sum(key != value for key, value in key_map.items()),
                          visit_entries_at_cutoff_removed=len(removed_at_cutoff),
                          visit_entries_after_cutoff_removed=len(removed_after_cutoff))
            positions_audit[str(position)] = {
                'input_event_index': query.index, 'original_candidate_count': len(original),
                'action_candidate_count': len(candidates), 'current_candidate_action_count': len(current_action_keys),
                'added_app_keys': restored_apps, 'added_concrete_keys': restored_concrete,
                'window_to_app': {key: value for key, value in key_map.items() if key != value},
                'candidate_evidence': evidence_by_key,
                'recency_filter': {'strictly_before_prediction': True,
                                   'removed_at_cutoff_keys': removed_at_cutoff,
                                   'removed_after_cutoff_keys': removed_after_cutoff},
            }
    action_streams, stream_audits = {}, {}
    for name, stream in streams.items():
        converted, counts = [], Counter()
        for row in stream:
            label = row['record']['label']
            mapped = action_label(label)
            if label.startswith('window_class:'):
                owner = label[len('window_class:'):]
                if owner not in apps_at_maturity[row['ready_index']]:
                    counts['excluded_window_without_admitted_app_at_maturity'] += 1
                    continue
                counts['window_labels_to_app'] += 1
            elif label.startswith('window@'):
                counts['excluded_window_without_semantic_owner'] += 1
                continue
            if mapped.startswith('app:') and mapped[4:] in DENIED_BUNDLES:
                counts['excluded_denied_app'] += 1
                continue
            record = {**row['record'], 'label': mapped}
            converted.append({**row, 'record': record})
        action_streams[name] = converted
        stream_audits[name] = {'original': len(stream), 'retained': len(converted), **dict(counts)}
    positions = sorted({row['position'] for rows in cohorts.values() for row in rows})
    inputs, mappings = {}, {}
    for position in positions:
        inputs[position], mappings[position] = comparison.model_input(action_items[position])
        inputs[position]['candidate_key_order'] = mappings[position]
    cohort_audit = {}
    for name, rows in cohorts.items():
        supported = [row for row in rows if row['diagnosis_after_outcome']['eligible']]
        old_missing = []
        for row in supported:
            if row['diagnosis_after_outcome']['candidate_present']:
                continue
            query = action_items[row['position']]
            key = truth_action_key(row, action_items)
            old_missing.append({'position': row['position'], 'sample_id': row['sample_id'],
                                'original_key': row['observed_outcome']['key'], 'action_key': key,
                                'action_candidate_present': key in query.candidates,
                                'candidate_evidence': positions_audit[str(row['position'])]['candidate_evidence'].get(key)})
        cohort_audit[name] = {
            'supported_n': len(supported), 'old_missing_mappings': old_missing,
            'action_candidate_missing': sum(truth_action_key(row, action_items)
                                            not in action_items[row['position']].candidates for row in supported),
        }
    audit = {'version': VERSION, 'current_origin_unchanged': True, 'raw_data_modified': False,
             'launch_performed': False, 'historical_reachability_verified': False,
             'historical_restoration': 'admitted app or exact locator at/before query index; later rejection removes it',
             'excluded_sessions': sorted(excluded_sessions), 'totals': dict(totals),
             'positions': positions_audit, 'streams': stream_audits, 'cohorts': cohort_audit}
    return action_items, inputs, mappings, action_streams, audit


def truth_metadata(row, items):
    query = items[row['position']]
    observed = row['observed_outcome']
    if observed.get('metadata'):
        return observed['metadata']
    if observed['key'] == query.truth:
        return getattr(query, '_reopen_original_truth_meta', query.truth_meta)
    return getattr(query, '_reopen_original_candidates', query.candidates).get(observed['key'], {})


def truth_action_key(row, items):
    key = row['observed_outcome']['key']
    meta = truth_metadata(row, items)
    if meta.get('kind') == 'window' and bundle_of(meta):
        return 'app:' + bundle_of(meta)
    return key


def prediction_action_key(key, query):
    """Rescore an old exact-window prediction only using its as-of candidate map."""
    return getattr(query, '_reopen_prediction_key_map', {}).get(key, key)


def same_foreground_app_ambiguous(row, items):
    query = items[row['position']]
    meta = truth_metadata(row, items)
    owner = bundle_of(meta)
    return bool(meta.get('kind') in {'app', 'window'} and owner and owner == query.current_app)


def _metric(rows, predictions, items, level):
    ranks, present = [], 0
    unknown_truth_app = unknown_prediction_app = 0
    app_upper_ranks, app_top1_upper = [], []
    for row in rows:
        query = items[row['position']]
        shown = predictions[row['position']]['predictions']
        if len(shown) > 4 or len(set(shown)) != len(shown):
            raise ValueError('Predictions must contain at most four distinct actions')
        truth = row['observed_outcome']['key']
        if level == 'actionable':
            truth = truth_action_key(row, items)
            ranked = [prediction_action_key(key, query) for key in shown]
            covered = truth in query.candidates
        elif level == 'exact_original':
            ranked = shown
            covered = truth in getattr(query, '_reopen_original_candidates', query.candidates)
        else:
            truth = bundle_of(truth_metadata(row, items))
            if not truth:
                unknown_truth_app += 1
                continue
            original = getattr(query, '_reopen_original_candidates', {})
            ranked = [bundle_of(query.candidates.get(key) or original.get(key) or {}) for key in shown]
            if any(owner is None for owner in ranked):
                unknown_prediction_app += 1
            known_rank = ranked.index(truth) + 1 if truth in ranked else 0
            unknown_rank = ranked.index(None) + 1 if None in ranked else 0
            possible_ranks = [rank for rank in (known_rank, unknown_rank) if rank]
            app_upper_ranks.append(min(possible_ranks) if possible_ranks else 0)
            app_top1_upper.append(bool(ranked) and ranked[0] in (truth, None))
            covered = any(bundle_of(meta) == truth for meta in query.candidates.values())
        ranks.append(ranked.index(truth) + 1 if truth in ranked else 0)
        present += bool(covered)
    n = len(ranks)
    result = {'n': n, 'top4_hits': sum(0 < rank <= 4 for rank in ranks),
              'top1_hits': sum(rank == 1 for rank in ranks),
              'top4': sum(0 < rank <= 4 for rank in ranks) / n if n else None,
              'top1': sum(rank == 1 for rank in ranks) / n if n else None,
              'mrr4': sum(1 / rank if rank else 0 for rank in ranks) / n if n else None,
              'candidate_present': present, 'candidate_missing': n - present}
    if level == 'app_level':
        result.update(n_total=len(rows), unknown_truth_app=unknown_truth_app,
                      unknown_prediction_app=unknown_prediction_app,
                      unknown_total=unknown_truth_app + unknown_prediction_app,
                      top4_lower=result['top4'], top1_lower=result['top1'],
                      top4_upper_hits=sum(0 < rank <= 4 for rank in app_upper_ranks),
                      top1_upper_hits=sum(app_top1_upper),
                      top4_upper=sum(0 < rank <= 4 for rank in app_upper_ranks) / n if n else None,
                      top1_upper=sum(app_top1_upper) / n if n else None,
                      mrr4_upper=sum(1 / rank if rank else 0 for rank in app_upper_ranks) / n if n else None,
                      evaluation='fixed known-truth-owner denominator; top1/top4 are confirmed lower bounds; unknown slots define upper bounds')
    return result


def _groups(rows, predictions, items, level):
    evidence = [row for row in rows if row['diagnosis_after_outcome']['eligible']]
    metric = lambda group: _metric(group, predictions, items, level)
    def covered(row):
        query = items[row['position']]
        if level == 'exact_original':
            return row['observed_outcome']['key'] in getattr(query, '_reopen_original_candidates', query.candidates)
        return truth_action_key(row, items) in query.candidates
    result = {'all': metric(rows), 'supported': metric(evidence),
              'late_supported': metric([row for row in evidence if row['split'] == 'test']),
              'early_supported': metric([row for row in evidence if row['split'] == 'validation']),
              'covered_supported': metric([row for row in evidence if covered(row)]),
              'strict_supported': metric([row for row in evidence
                  if row['evaluation']['strict_input_and_label_coverage']]),
              'grades': {grade: metric([row for row in evidence if row['diagnosis_after_outcome']['grade'] == grade])
                         for grade in ('strong', 'repeated', 'single')},
              'by_original_kind': {kind: metric([row for row in evidence
                  if truth_metadata(row, items).get('kind', 'unknown') == kind])
                  for kind in ('app', 'window', 'page', 'directory', 'file', 'unknown')},
              'exact_representable_supported': metric([row for row in evidence
                  if truth_metadata(row, items).get('kind') in CONCRETE_KINDS | {'app'}]),
              'concrete_resource_supported': metric([row for row in evidence
                  if truth_metadata(row, items).get('kind') in CONCRETE_KINDS]),
              'same_foreground_app_ambiguous': metric([row for row in evidence
                  if same_foreground_app_ambiguous(row, items)]),
              'without_same_foreground_app_ambiguous': metric([row for row in evidence
                  if not same_foreground_app_ambiguous(row, items)])}
    return result


def score_groups(rows, predictions, items):
    """Actionable main score; exact originals and known-owner app scores separate."""
    result = _groups(rows, predictions, items, 'actionable')
    result['exact_original'] = _groups(rows, predictions, items, 'exact_original')
    result['app_level'] = _groups(rows, predictions, items, 'app_level')
    result['scoring_contract'] = {
        'main': 'window→owning app; original app and concrete page/directory/file remain exact',
        'window_app_action_is_not_exact_window_hit': True,
        'unknown_app_ownership': 'unknown truth owners explicitly excluded; fixed known-truth-owner denominator; unknown shown owners produce score bounds',
        'same_foreground_app': 'kept in primary score; potentially no-op action also reported separately',
    }
    return result


def annotate_predictions(predictions, items):
    """Attach inspectable action plans to shown slots; never execute an action."""
    result = deepcopy(predictions)
    for position, row in result.items():
        query = items[int(position)]
        plans = []
        for key in row['predictions']:
            action_key = prediction_action_key(key, query)
            meta = query.candidates.get(action_key)
            if meta is None:
                raise ValueError('Shown action lacks as-of executable candidate evidence')
            plan = deepcopy(meta['_action_plan'])
            plan.update(candidate_key=key, action_key=action_key,
                        same_foreground_app_may_be_noop=(meta['kind'] == 'app'
                            and bundle_of(meta) == query.current_app))
            plans.append(plan)
        row['action_plans'] = plans
    return result
