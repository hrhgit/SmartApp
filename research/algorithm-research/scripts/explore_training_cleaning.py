"""Fixed, independently configured training-label cleaning experiments.

Returns private audit ledgers; performs no writes and never reads evaluation
answers or scores. Every label keeps its original complete 30-second maturity.
The caller owns the fixed scoring cohort and any later action-label conversion.
"""

from collections import Counter, defaultdict
from dataclasses import replace
import json
import math
from pathlib import Path
import types

from evaluate_evidence_model import DENIED_BUNDLES, semantic_label
from evaluate_ordered_current import current_context
from summarize_tuned_history import verify_run


VERSION = 'training-cleaning-exploration-v1'


def policies():
    result = [{'id': 'raw', 'mode': 'raw'},
              {'id': 'resolved', 'mode': 'resolved_arrival'},
              {'id': '3', 'mode': 'frozen', 'seconds': 3},
              {'id': '5', 'mode': 'frozen', 'seconds': 5}]
    result += [{'id': f'dwell{seconds}', 'mode': 'all_short', 'seconds': seconds}
               for seconds in (1, 2, 8, 10)]
    for name, mode in [('nav', 'same_tab_only'), ('return', 'return_only'),
                       ('confirmed', 'navigation_and_confirmed_switch')]:
        result += [{'id': f'{name}{seconds}', 'mode': mode, 'seconds': seconds}
                   for seconds in (3, 5)]
    result += [
        {'id': 'kind_app2_resource5', 'mode': 'all_short',
         'kind_seconds': {'app': 2, 'window': 2, 'page': 5, 'directory': 5, 'file': 5}},
        {'id': 'kind_app5_resource2', 'mode': 'all_short',
         'kind_seconds': {'app': 5, 'window': 5, 'page': 2, 'directory': 2, 'file': 2}},
    ]
    for scope, quantile in [('global', .10), ('kind', .10), ('kind', .25)]:
        result.append({'id': f'adaptive_{scope}_q{int(quantile * 100)}', 'mode': 'all_short',
                       'adaptive': {'scope': scope, 'quantile': quantile, 'min_history': 20,
                                    'min_seconds': 1, 'max_seconds': 8, 'fallback_seconds': 3}})
    names = {'raw': '原始下一次到达', 'resolved': '同次到达补全，不过滤短停留',
             '3': '冻结原3秒清洗', '5': '冻结原5秒清洗',
             'kind_app2_resource5': '应用窗口2秒，具体资源5秒',
             'kind_app5_resource2': '应用窗口5秒，具体资源2秒',
             'adaptive_global_q10': '过去全部完成停留的10%分位数',
             'adaptive_kind_q10': '过去同类完成停留的10%分位数',
             'adaptive_kind_q25': '过去同类完成停留的25%分位数'}
    for p in result:
        if p['id'] not in names:
            prefix = {'all_short': '全部短停留', 'same_tab_only': '仅确证同标签页导航',
                      'return_only': '仅快速A-B-A返回',
                      'navigation_and_confirmed_switch': '确证导航与跨目标切换'}[p['mode']]
            names[p['id']] = prefix + f"，阈值{p['seconds']}秒"
        p['name_zh'] = names[p['id']]
    assert len(result) == 19 and len({p['id'] for p in result}) == 19
    return result


def quantile(values, probability):
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = int(position)
    upper = math.ceil(position)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def threshold_for(policy, kind, history):
    """history must contain only completed intervals visible at input cutoff."""
    if policy['mode'] == 'resolved_arrival':
        return 0, {'scope': 'no_dwell_filter'}
    adaptive = policy.get('adaptive')
    if adaptive:
        values = history.get(kind if adaptive['scope'] == 'kind' else '*', ())
        used = len(values) >= adaptive['min_history']
        estimate = quantile(values, adaptive['quantile']) if used else adaptive['fallback_seconds']
        seconds = min(adaptive['max_seconds'], max(adaptive['min_seconds'], estimate))
        return seconds, {'scope': adaptive['scope'], 'completed_prior_intervals': len(values),
                         'quantile': adaptive['quantile'], 'estimated_seconds': estimate,
                         'used_past_distribution': used}
    seconds = policy.get('kind_seconds', {}).get(kind, policy.get('seconds', 3))
    return seconds, {'scope': 'fixed_kind' if 'kind_seconds' in policy else 'fixed'}


def _confirmed_switch(core, visit, target):
    category = visit.get('navigation_class')
    if category in {'application_switch', 'browser_window_switch', 'same_window_tab_switch'}:
        return True
    other = visit.get('next_target')
    if not other or target.get('kind') != other.get('kind'):
        return False
    kind = target['kind']
    if kind == 'window':
        return core.key_for(target, visit['session']) != core.key_for(other, visit['session'])
    if kind in {'directory', 'file'}:
        return bool(target.get('locator') and other.get('locator')
                    and target['locator'] != other['locator'])
    return False


def derive_label(core, query, visits_by_id, events_by_id, event_index, policy, history=None):
    """A causal, traceable derivation; selective rules retain unproven transients.

    all_short requires observed dwell >= threshold. Selective rules remove only
    the named *completed* short transitions; they retain other observed arrivals
    without claiming those arrivals prove stable intention.
    """
    history = history or {}
    deadline = events_by_id[query.anchor_id]['monotonicMs'] + 30000
    event_id = query.next_event_id
    skipped, seen = [], set()

    def reject(status, **extra):
        return None, {'status': status, 'skipped': skipped, **extra}

    while True:
        if event_id in seen:
            return reject('cyclic_visit_chain')
        seen.add(event_id)
        visit = visits_by_id.get(event_id)
        if not visit or visit['session'] != query.session or visit.get('left_censored'):
            return reject('missing_or_unlinked_visit')
        start = visit['arrival_start_monotonic_ms']
        if start > deadline or event_index[event_id] > query.ready_index:
            return reject('no_target_within_original_horizon')
        target = visit['initial_target']
        target_evidence_id = event_id
        for refinement in visit.get('refinements', ()):
            if (refinement['monotonic_ms'] <= deadline
                    and event_index[refinement['event_id']] <= query.ready_index):
                target, start = refinement['target'], refinement['monotonic_ms']
                target_evidence_id = refinement['event_id']
        if (target.get('_bundle') or target.get('bundleID')) in DENIED_BUNDLES:
            return reject('denied_destination_boundary')
        key = core.key_for(target, query.session)
        if key == query.current:
            return reject('returned_to_current_target')
        seconds, threshold_audit = threshold_for(policy, target['kind'], history)
        end = visit.get('end_monotonic_ms')
        known_duration = isinstance(end, (int, float)) and math.isfinite(end) and end >= start
        observed_seconds = max(0., (min(end, deadline) - start) / 1000) if known_duration else None
        exit_id = visit.get('end_event_id')
        exit_observed = (known_duration and end <= deadline
                         and exit_id in event_index and event_index[exit_id] <= query.ready_index)
        completed_short = bool(exit_observed and visit.get('complete')
                               and observed_seconds is not None and observed_seconds < seconds)
        category = visit.get('navigation_class', 'unknown') if exit_observed else 'exit_not_observed_in_horizon'
        return_to_previous = False
        if completed_short:
            before, after = visit.get('previous_target'), visit.get('next_target')
            return_to_previous = bool(before and after and
                core.key_for(before, query.session) == core.key_for(after, query.session))
        mode = policy['mode']
        skip = completed_short and (
            mode == 'all_short'
            or (mode == 'same_tab_only' and category == 'same_tab_navigation')
            or (mode == 'return_only' and return_to_previous)
            or (mode == 'navigation_and_confirmed_switch'
                and (category == 'same_tab_navigation' or _confirmed_switch(core, visit, target))))
        detail = {'event_id': event_id, 'target': target, 'target_evidence_event_id': target_evidence_id,
                  'threshold_seconds': seconds, 'threshold_basis': threshold_audit,
                  'observed_seconds_within_horizon': observed_seconds,
                  'completed_exit_within_horizon': bool(exit_observed and visit.get('complete')),
                  'navigation_class': category, 'returns_to_previous': return_to_previous}
        if skip:
            skipped.append({**detail, 'departure_event_id': exit_id,
                            'category': 'quick_return' if mode == 'return_only'
                            else 'navigation_intermediate' if category == 'same_tab_navigation'
                            else 'quick_return' if return_to_previous else 'suspected_temporary_target'})
            event_id = exit_id
            continue
        if mode == 'all_short' and (observed_seconds is None or observed_seconds < seconds):
            if not known_duration:
                status = 'unknown_dwell_not_treated_as_short'
            elif not exit_observed:
                status = 'insufficient_dwell_before_30s'
            else:
                status = 'observation_boundary_not_treated_as_short'
            return reject(status, considered=detail)
        evidence_time = start + seconds * 1000 if mode == 'all_short' else start
        # Selective filtering needs the observed exit if it retains a completed
        # transition because that transition did not match the selected category.
        if mode not in {'all_short', 'resolved_arrival'} and exit_observed:
            evidence_time = max(evidence_time, end)
        assert evidence_time <= deadline
        meta = {**target, '_bundle': target.get('bundleID')}
        modified = replace(query, truth=key, truth_meta=meta, next_event_id=event_id,
                           offset_ms=visit['arrival_start_monotonic_ms']
                           - events_by_id[query.anchor_id]['monotonicMs'])
        return modified, {'status': 'accepted_target', 'target_event_id': event_id,
            'target': target, 'skipped': skipped, 'considered': detail,
            'evidence_available_monotonic_ms': evidence_time,
            'basis': 'observed_minimum_dwell' if mode == 'all_short'
                     else 'same_arrival_refinement_within_horizon' if mode == 'resolved_arrival'
                     else 'observed_arrival_not_removed_by_selected_rule',
            'stable_intention_claimed': False, 'changed': key != query.truth}


def _frozen_dwell(base, core):
    module = types.ModuleType('cleaning_exploration_frozen_dwell')
    path = base / 'dwell-source.py'
    exec(compile(path.read_text(), str(path), 'exec'), module.__dict__)
    module.core = core
    return module


def _record(query, target, outcome_id, core, event_times):
    context = current_context(query)
    if context is None:
        return None
    origin = context.targets[0]
    app = query.current_app or ''
    kind = 'window_class' if origin.startswith('window_class:') else origin.split(':', 1)[0]
    x = {'origin=' + origin: 1., 'app=' + app: 1., 'kind=' + kind: 1., 'bias': 1.}
    meta = {**target, '_bundle': target.get('_bundle') or target.get('bundleID')}
    labeled = replace(query, truth=core.key_for(target, query.session), truth_meta=meta)
    return {'origin': origin, 'app': app, 'timestamp': query.timestamp, 'x': x,
            'label': semantic_label(labeled, 'window_class'),
            'outcome_timestamp': event_times[outcome_id], 'outcome_id': outcome_id}


class PastDwellHistory:
    """Adaptive statistics advance only to each source prediction's input index."""

    def __init__(self, visits, event_index):
        self.completed = sorted(((event_index[v['end_event_id']], v) for v in visits
            if v.get('complete') and not v.get('left_censored')
            and not any((t.get('_bundle') or t.get('bundleID')) in DENIED_BUNDLES
                        for t in (v.get('initial_target') or {}, v.get('target') or {},
                                  v.get('next_target') or {}))
            and isinstance(v.get('observed_seconds'), (int, float))
            and math.isfinite(v['observed_seconds']) and v['observed_seconds'] >= 0),
            key=lambda pair: pair[0])
        self.values = defaultdict(list)
        self.cursor = 0
        self.last_cutoff = -1
        self.through_index = None

    def advance(self, input_index):
        if input_index < self.last_cutoff:
            raise ValueError('Adaptive history cannot move backwards in time')
        while self.cursor < len(self.completed) and self.completed[self.cursor][0] <= input_index:
            index, visit = self.completed[self.cursor]
            self.values['*'].append(visit['observed_seconds'])
            self.values[visit['target']['kind']].append(visit['observed_seconds'])
            self.through_index = index
            self.cursor += 1
        self.last_cutoff = input_index
        return self.values


def build_cleaning_streams(core, items, provenance):
    """Return {policy_id: pending records}, audit including private decision ledgers.

    provenance is compare_online_models.load_dataset's verified provenance. No
    filesystem mutation occurs; caller must save private ledgers only in a new run.
    """
    base = Path(provenance['cleaning_run'])
    source, source_hashes = verify_run(base)
    if provenance.get('cleaning_hashes') != source_hashes:
        raise ValueError('Cleaning source hashes differ from verified provenance')
    snapshot = Path(provenance['snapshot'])
    input_hashes, _ = core.verified_snapshot(snapshot)
    if input_hashes != provenance['input_hashes'] or input_hashes != source['input_hashes']:
        raise ValueError('Snapshot hashes differ from verified provenance')
    events = core.load_lines(snapshot / 'events.jsonl')
    samples = core.load_lines(snapshot / 'navigation-samples.jsonl')
    dwell = _frozen_dwell(base, core)
    visits, _, extraction_audit = dwell.extract_visits(events, samples, set(source['excluded_sessions']))
    events_by_id = {e['id']: e for e in events}
    event_index = {e['id']: i for i, e in enumerate(events)}
    event_times = {e['id']: e['timestampMs'] for e in events}
    visits_by_id = {v['arrival_event_id']: v for v in visits}
    frozen_ledger = json.loads((base / 'labels-private.json').read_text())
    assert all(len(frozen_ledger[t]) == len(items) for t in ('3', '5'))
    past_history = PastDwellHistory(visits, event_index)
    configs = policies()
    pending = {p['id']: [] for p in configs}
    ledgers = {p['id']: [] for p in configs}
    assert all(left.index < right.index for left, right in zip(items, items[1:]))
    for position, query in enumerate(items):
        history = past_history.advance(query.index)
        for policy in configs:
            ident = policy['id']
            if policy['mode'] == 'raw':
                target, outcome_id = query.truth_meta, query.next_event_id
                decision = {'status': 'accepted_raw', 'target': target, 'target_event_id': outcome_id,
                            'basis': 'original_observed_arrival', 'skipped': []}
                labeled = query
            elif policy['mode'] == 'frozen':
                row = frozen_ledger[ident][position]
                assert row['sample_id'] == query.sample_id and row['position'] == position
                assert row['label_ready_event_index'] == query.ready_index
                decision = dict(row)
                target, outcome_id = row.get('target'), row.get('target_event_id')
                labeled = query if row['status'] == 'stable_target' else None
            else:
                labeled, decision = derive_label(core, query, visits_by_id, events_by_id,
                                                  event_index, policy, history)
                target, outcome_id = decision.get('target'), decision.get('target_event_id')
            row = {'position': position, 'sample_id': query.sample_id,
                   'input_through_event_index': query.index, 'label_ready_event_index': query.ready_index,
                   'adaptive_history_through_event_index': past_history.through_index, **decision}
            if query.current_app in DENIED_BUNDLES:
                row['training_exclusion'] = 'denied_origin'
            elif labeled is not None:
                bundle = target.get('_bundle') or target.get('bundleID')
                if bundle in DENIED_BUNDLES:
                    row['training_exclusion'] = 'denied_destination'
                else:
                    record = _record(query, target, outcome_id, core, event_times)
                    if record is None:
                        row['training_exclusion'] = 'unknown_origin'
                    else:
                        assert query.index < event_index[outcome_id] <= query.ready_index
                        pending[ident].append({'position': position, 'input_index': query.index,
                            'ready_index': query.ready_index, 'outcome_index': event_index[outcome_id],
                            'record': record})
                        row['training_included_before_dedup'] = True
            ledgers[ident].append(row)
    streams, summaries = {}, {}
    for policy in configs:
        ident = policy['id']
        seen, stream = set(), []
        for row in sorted(pending[ident], key=lambda r: (r['ready_index'], r['position'])):
            key = row['record']['origin'], row['record']['outcome_id']
            if key not in seen:
                seen.add(key)
                stream.append(row)
                ledgers[ident][row['position']]['training_included_after_dedup'] = True
            else:
                ledgers[ident][row['position']]['training_exclusion'] = 'duplicate_origin_outcome'
        streams[ident] = stream
        summaries[ident] = {'eligible_before_dedup': len(pending[ident]),
            'unique_origin_outcomes': len(stream), 'duplicates_removed': len(pending[ident]) - len(stream),
            'status_counts': dict(Counter(r['status'] for r in ledgers[ident])),
            'exclusion_counts': dict(Counter(r['training_exclusion'] for r in ledgers[ident]
                                            if 'training_exclusion' in r)),
            'skipped_visit_count': sum(len(r.get('skipped', ())) for r in ledgers[ident])}
    audit = {'version': VERSION, 'policies': configs, 'summaries': summaries,
             'ledger_private': ledgers, 'visit_extraction_audit': extraction_audit,
             'complete_dwell_descriptive_only': {kind: {'n': len(values),
                 'p10_seconds': quantile(values, .1), 'p25_seconds': quantile(values, .25),
                 'median_seconds': quantile(values, .5)} for kind, values in past_history.values.items()},
             'cleaning_source_hashes': source_hashes, 'input_hashes': input_hashes,
             'fixed_evaluation_membership_changed': False, 'raw_data_modified': False,
             'original_30s_maturity_preserved': True, 'adaptive_fit_cutoff': 'original prediction input index',
             'adaptive_eligibility': 'Admitted independent completed visits; denied targets and exits excluded.',
             'selective_rule_note': 'Unproven transients are retained as observed arrivals, not confirmed intention.'}
    return streams, audit
