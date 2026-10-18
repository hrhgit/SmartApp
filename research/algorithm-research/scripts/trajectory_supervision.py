"""Causal, weighted supervision for the first three independent arrivals.

Input visits are the admitted, episode-level output of describe_dwell.extract_visits.
Resolution updates refine one episode; they never create extra trajectory steps.
The evaluation policy is always ``resolved``. Other policies change training
weights/retention only, preserving observed_step and the fixed first-three limit.
No candidate set, predictions, later catalog, or body text is used here.
"""
from collections import Counter, defaultdict
from copy import deepcopy
from bisect import bisect_right
import math
from urllib.parse import urlsplit

import evaluate_context_knn as core
from known_inactivity import annotate_events, active_time, is_boundary
from evaluate_evidence_model import DENIED_BUNDLES
from reopenable_actions import _qualified


VERSION = 'trajectory-supervision-v2'
KIND_SECONDS = {'app': 5., 'window': 5., 'page': 2., 'directory': 2., 'file': 2.}


class TrajectoryEvents(dict):
    """Optional read-only-in-use event lookup with fast bounded interval scans.

    Wrap once as TrajectoryEvents(events_by_id, event_index) for many queries.
    Ordinary dictionaries remain supported, with a linear interval lookup.
    Reconstruct this wrapper after any deliberate source-data repair.
    """
    def __init__(self, events_by_id, event_index):
        super().__init__(events_by_id)
        ordered = sorted((event_index[key], value) for key, value in self.items())
        self.indices = [i for i, _ in ordered]
        self.ordered = annotate_events([event for _, event in ordered])
        self.update({event['id']: event for event in self.ordered})

    def between(self, lower, upper):
        return self.ordered[bisect_right(self.indices, lower):bisect_right(self.indices, upper)]


def _episode_boundary(visit, begin_index, through_index, deadline, events, event_index):
    initial_ids = {visit['initial_target'].get('id')}
    owner = _owner(visit['initial_target'])
    if owner:
        initial_ids.add('app:' + owner)
    for event in events:
        ei = event_index[event['id']]
        if not begin_index < ei <= through_index or active_time(event) > deadline:
            continue
        kind, data = event['kind'], event.get('data', {})
        if event.get('sessionID') != visit['session'] or is_boundary(event):
            return {'reason': 'explicit_observation_boundary', 'event_index': ei}
        if event.get('continuityDisposition') == 'suspended':
            continue
        active_ids = initial_ids | {r['target'].get('id') for r in visit.get('refinements', ())
            if event_index.get(r['event_id'], math.inf) <= ei and r.get('active_ms', r['monotonic_ms']) <= deadline}
        if kind == 'target.update' and data.get('target', {}).get('id') in active_ids:
            if not _qualified(core, data['target']):
                return {'reason': 'explicit_active_target_revocation', 'event_index': ei}
        if kind == 'source.status' and data.get('observationStatus') in {'unavailable', 'stale', 'initializing'}:
            source = event.get('source')
            if source and (source == visit.get('source') or
                           (core.family(source) != 'other' and core.family(source) == visit.get('source_family'))):
                return {'reason': 'explicit_foreground_source_boundary', 'event_index': ei}
    return None


def policies():
    return [
        {'id': 'resolved', 'mode': 'resolved', 'max_steps': 3},
        {'id': 'hard_kind', 'mode': 'hard_kind', 'max_steps': 3,
         'kind_seconds': dict(KIND_SECONDS)},
        {'id': 'soft_kind', 'mode': 'soft_kind', 'max_steps': 3,
         'kind_seconds': dict(KIND_SECONDS), 'minimum_weight': .1},
        {'id': 'soft_quickcheck', 'mode': 'soft_quickcheck', 'max_steps': 3,
         'kind_seconds': dict(KIND_SECONDS), 'minimum_weight': .1, 'min_history': 5},
    ]


def _owner(target):
    explicit, observed = target.get('bundleID'), target.get('_bundle')
    if explicit and observed and explicit != observed:
        return None
    owner = observed or explicit
    if not owner and target.get('kind') == 'app' and target.get('id', '').startswith('app:'):
        owner = target['id'][4:]
    return owner if isinstance(owner, str) and owner and owner not in DENIED_BUNDLES else None


def action_key(target):
    """Concrete resources stay exact; generic windows fall back to their owner."""
    if not isinstance(target, dict):
        return None
    if target.get('bundleID') in DENIED_BUNDLES or target.get('_bundle') in DENIED_BUNDLES:
        return None
    kind, key = target.get('kind'), target.get('id')
    if kind in {'app', 'window'}:
        owner = _owner(target)
        if kind == 'app' and owner and key != 'app:' + owner:
            return None
        return 'app:' + owner if owner else None
    if kind not in {'page', 'directory', 'file'} or not isinstance(key, str) or not key:
        return None
    locator = target.get('locator')
    if not isinstance(locator, str) or not locator:
        return None
    try:
        parsed = urlsplit(locator)
        valid = (parsed.scheme.lower() in {'http', 'https'} and bool(parsed.hostname)
                 if kind == 'page' else locator.startswith('/')
                 or (parsed.scheme == 'file' and parsed.path.startswith('/')))
    except ValueError:
        return None
    return key if valid else None


def _semantic_origin(target):
    if target.get('kind') == 'window':
        owner = _owner(target)
        return 'window_class:' + owner if owner else None
    return action_key(target)


def origin_key(query):
    current = query.current
    if current.startswith('window@'):
        return 'window_class:' + query.current_app if query.current_app else None
    return current


def _finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _policy(value):
    if isinstance(value, str):
        name = 'resolved' if value in {'unfiltered', 'resolved_arrival'} else value
        value = next((p for p in policies() if p['id'] == name), None)
        if value is None:
            raise ValueError('Unknown trajectory policy')
    result = deepcopy(value)
    if result.get('mode') not in {'resolved', 'hard_kind', 'soft_kind', 'soft_quickcheck'}:
        raise ValueError('Unknown trajectory mode')
    count = result.get('max_steps', 3)
    if isinstance(count, bool) or not isinstance(count, int) or not 1 <= count <= 3:
        raise ValueError('Trajectory must preserve one to three original steps')
    result['max_steps'] = count
    floor = result.get('minimum_weight', .1)
    if not _finite(floor) or not 0 < floor <= 1:
        raise ValueError('minimum_weight must be in (0, 1]')
    thresholds = {**KIND_SECONDS, **result.get('kind_seconds', {})}
    if any(not _finite(t) or t <= 0 for t in thresholds.values()):
        raise ValueError('Dwell thresholds must be finite and positive')
    result['kind_seconds'] = thresholds
    if result.get('min_history', 5) < 1:
        raise ValueError('min_history must be positive')
    return result


def _anchor_visit(query, visits_by_id, events_by_id):
    if query.anchor_id in visits_by_id:
        return visits_by_id[query.anchor_id]
    linked = events_by_id[query.anchor_id].get('data', {}).get('resolvesContextEventID')
    if linked in visits_by_id:
        return visits_by_id[linked]
    # Older exports may omit resolvesContextEventID, but the extractor retains
    # the explicit refinement-to-episode link. This lookup uses no final target.
    return next((v for v in visits_by_id.values() if any(
        r['event_id'] == query.anchor_id for r in v.get('refinements', ()))), None)


def _quickcheck_history(query, history):
    if history is None:
        return {'origin': origin_key(query), 'through_index': None, 'n': 0, 'quick_returns': 0}
    if hasattr(history, 'advance'):
        history = history.advance(query)
    value = dict(history)
    if value.get('origin') != origin_key(query):
        raise ValueError('Quick-check history belongs to a different input origin')
    through = value.get('through_index')
    if through is not None and (not isinstance(through, int) or through > query.index):
        raise ValueError('Quick-check history contains post-input outcomes')
    n, returns = value.get('n', 0), value.get('quick_returns', 0)
    if not isinstance(n, int) or not isinstance(returns, int) or not 0 <= returns <= n:
        raise ValueError('Invalid quick-check history counts')
    if n and through is None:
        raise ValueError('Nonempty quick-check history needs an as-of index')
    return value


def derive_trajectory(query, visits_by_id, events_by_id, event_index,
                      policy='resolved', history=None):
    """Return ``(labels, audit)`` with fixed observed_step and explicit masks.

    All retained labels mature at query.ready_index (the original complete
    30-second window), even if arrival evidence appeared earlier. A caller may
    learn them only before a later prediction whose index is strictly greater.
    Hard cleaning never searches a fourth episode to replace a dropped step.
    Unknown/right-censored dwell is not a short visit. Missing future arrivals
    are masked, and are not negative labels or confirmed "no next action".
    """
    p = _policy(policy)
    if not isinstance(events_by_id, TrajectoryEvents):
        events_by_id = TrajectoryEvents(events_by_id, event_index)
    anchor = events_by_id[query.anchor_id]
    if event_index[query.anchor_id] != query.index or query.ready_index < query.index:
        raise ValueError('Prediction input/maturity indices are inconsistent')
    deadline = active_time(anchor) + 30000
    window_events = (events_by_id.between(query.index, query.ready_index)
                     if isinstance(events_by_id, TrajectoryEvents) else
                     [event for key, event in events_by_id.items()
                      if query.index < event_index[key] <= query.ready_index])
    window_events = sorted(window_events, key=lambda event: event_index[event['id']])
    prior = _quickcheck_history(query, history) if p['mode'] == 'soft_quickcheck' else None
    audit = {'version': VERSION, 'policy': p, 'input_index': query.index,
        'ready_index': query.ready_index, 'deadline_monotonic_ms': deadline,
        'step_mask': [False] * p['max_steps'], 'training_step_mask': [False] * p['max_steps'],
        'dropped': [], 'observed': [], 'stable_intention_claimed': False,
        'evaluation_policy': 'resolved', 'absent_steps_are_negative': False,
        'history': prior, 'stop_reason': None}
    labels = []

    def stop(reason):
        audit['stop_reason'] = reason
        audit['observed_count'] = sum(audit['step_mask'])
        audit['retained_count'] = len(labels)
        audit['masked_steps'] = [i + 1 for i, known in enumerate(audit['step_mask']) if not known]
        return labels, audit

    source_visit = _anchor_visit(query, visits_by_id, events_by_id)
    if source_visit is None:
        return stop('unlinked_input_episode')
    source_end = event_index.get(source_visit.get('end_event_id'))
    source_end_time = source_visit.get('end_active_ms', source_visit.get('end_monotonic_ms'))
    if source_end is None or not _finite(source_end_time):
        return stop('unknown_input_episode_exit')
    if source_end > query.ready_index or source_end_time > deadline:
        return stop('original_30s_horizon')
    boundary = _episode_boundary(source_visit, query.index, source_end - 1, deadline,
                                 window_events, event_index)
    if boundary:
        return stop(boundary['reason'])
    if source_visit.get('end_reason') != 'different_observed_destination':
        return stop('input_episode_observation_boundary:' + str(source_visit.get('end_reason', 'unknown')))
    event_id = source_visit.get('end_event_id')
    seen = set()
    segment = source_visit.get('segment')
    for step in range(1, p['max_steps'] + 1):
        if event_id in seen:
            return stop('cyclic_visit_chain')
        seen.add(event_id)
        visit = visits_by_id.get(event_id)
        if visit is None:
            return stop('missing_or_unlinked_followup')
        if visit['session'] != query.session or (segment is not None and visit.get('segment') != segment):
            return stop('session_or_segment_boundary')
        if visit.get('left_censored'):
            return stop('left_censored_followup')
        index = event_index.get(event_id)
        start = visit.get('arrival_start_active_ms', visit.get('arrival_start_monotonic_ms'))
        if index is None or index <= query.index or not _finite(start):
            return stop('invalid_arrival_evidence')
        if index > query.ready_index or start > deadline:
            return stop('original_30s_horizon')
        event = events_by_id.get(event_id)
        if not event or event.get('sessionID') != query.session:
            return stop('missing_arrival_event')
        end_id = visit.get('end_event_id')
        end_index, end_time = event_index.get(end_id), visit.get('end_active_ms', visit.get('end_monotonic_ms'))
        through = min(query.ready_index, end_index - 1) if end_index is not None else query.ready_index
        boundary = _episode_boundary(visit, index, through, deadline, window_events, event_index)
        target = deepcopy(visit['initial_target'])
        if action_key(target) is None:
            return stop('denied_or_unusable_destination_boundary')
        evidence_id, target_start = event_id, start
        used_refinements = []
        for refinement in sorted(visit.get('refinements', ()),
                                 key=lambda r: event_index.get(r['event_id'], math.inf)):
            rid, rtime = refinement['event_id'], refinement.get('active_ms', refinement['monotonic_ms'])
            ri = event_index.get(rid)
            if ri is None or ri > query.ready_index or rtime > deadline:
                continue
            if boundary and ri >= boundary['event_index']:
                continue
            if ri < index or rtime < start:
                return stop('invalid_refinement_order')
            refinement_event = events_by_id.get(rid)
            if not refinement_event or refinement_event.get('sessionID') != query.session:
                return stop('missing_or_cross_session_refinement')
            close_index = event_index.get(visit.get('end_event_id'))
            if close_index is not None and close_index <= query.ready_index and ri >= close_index:
                return stop('refinement_after_episode_exit')
            if action_key(refinement['target']) is None:
                return stop('denied_or_unusable_refinement_boundary')
            # An explicit resolution belongs to this episode, even when it changes
            # the exact locator. It does not create a fourth/extra observation.
            target, evidence_id, target_start = deepcopy(refinement['target']), rid, rtime
            used_refinements.append(rid)
        key = action_key(target)
        if key is None:
            return stop('denied_or_unusable_destination_boundary')
        exit_known = bool(end_index is not None and end_index <= query.ready_index
            and _finite(end_time) and target_start <= end_time <= deadline)
        completed = bool(exit_known and visit.get('complete')
                         and visit.get('end_reason') == 'different_observed_destination')
        if boundary:
            # The arrival preceding a gap remains observed. Its measured dwell
            # and later linked arrivals do not survive the explicit boundary.
            completed, exit_known = False, False
        seconds = (end_time - target_start) / 1000 if completed else None
        # An arrival with no exit inside the frozen window remains a valid
        # positive, but absence of an exit alone is not a duration measurement.
        lower = (end_time - target_start) / 1000 if exit_known else None
        threshold = p['kind_seconds'][target['kind']]
        short = completed and seconds < threshold
        weight, basis = 1., 'resolved_independent_arrival'
        keep = True
        if p['mode'] == 'hard_kind' and short:
            keep, weight, basis = False, 0., 'completed_short_visit_removed'
        elif p['mode'] in {'soft_kind', 'soft_quickcheck'} and short:
            weight = max(p.get('minimum_weight', .1), min(1., seconds / threshold))
            basis = 'completed_short_visit_dwell_weight'
            if p['mode'] == 'soft_quickcheck':
                if prior['n'] >= p.get('min_history', 5):
                    rate = prior['quick_returns'] / prior['n']
                    weight += (1. - weight) * rate
                    basis = 'past_same_origin_quick_return_rate_softens_dwell_weight'
                else:
                    basis = 'insufficient_past_same_origin_history_fallback_soft_kind'
        elif p['mode'] != 'resolved':
            basis = 'known_arrival_unknown_or_censored_dwell_retained' if not completed else 'completed_nonshort_visit'
        label = {'observed_step': step, 'action_key': key, 'target': target,
            'event_id': event_id, 'arrival_event_id': event_id,
            'target_evidence_event_id': evidence_id, 'refinement_event_ids': used_refinements,
            'arrival_timestamp': event['timestampMs'], 'arrival_timestamp_ms': event['timestampMs'],
            'arrival_monotonic_ms': start, 'arrival_index': index,
            'target_evidence_index': event_index[evidence_id],
            'ready_index': query.ready_index, 'weight': weight, 'basis': basis,
            'dwell': {'seconds': seconds, 'lower_bound_seconds': lower,
                'threshold_seconds': threshold, 'completed_exit_within_horizon': completed,
                'exit_observed_within_horizon': exit_known,
                'status': 'complete' if completed else 'censored_or_unknown',
                'observation_boundary': boundary,
                'end_event_id': end_id if exit_known else None},
            'original_kind': target['kind'], 'destination_app': 'app:' + _owner(target) if _owner(target) else None}
        audit['step_mask'][step - 1] = True
        audit['observed'].append(deepcopy(label))
        if keep:
            labels.append(label)
            audit['training_step_mask'][step - 1] = True
        else:
            audit['dropped'].append(deepcopy(label))
        if boundary:
            return stop(boundary['reason'])
        if step == p['max_steps']:
            return stop('first_independent_step_limit')
        if not exit_known:
            return stop('original_30s_horizon' if end_index is not None else 'unknown_followup')
        if not completed:
            return stop('observation_boundary:' + str(visit.get('end_reason', 'unknown')))
        event_id = end_id
    return stop('first_independent_step_limit')


class PastQuickcheckHistory:
    """Incremental, completed A→B→A habits visible at each input cutoff.

    Counts are conditioned on the previous target A using the same stable
    current-interface representation (window class or exact resource). A short
    middle B is a quick return only if its independent exit to A was observed.
    The present query's later return is never used for its own training weight.
    """
    def __init__(self, visits, event_index):
        self.completed = []
        for visit in visits:
            before, middle, after = (visit.get(k) for k in ('previous_target', 'target', 'next_target'))
            end = event_index.get(visit.get('end_event_id'))
            seconds = visit.get('observed_seconds')
            if (not visit.get('complete') or visit.get('left_censored')
                    or visit.get('end_reason') != 'different_observed_destination'
                    or end is None or not _finite(seconds) or seconds < 0
                    or not all(isinstance(t, dict) and action_key(t) for t in (before, middle, after))):
                continue
            if any(event_index.get(r['event_id'], math.inf) >= end
                   for r in visit.get('refinements', ())):
                continue
            origin = _semantic_origin(before)
            quick = (action_key(before) == action_key(after)
                     and action_key(middle) != action_key(before)
                     and seconds < KIND_SECONDS[middle['kind']])
            self.completed.append((end, origin, bool(quick)))
        self.completed.sort(key=lambda row: row[0])
        self.cursor, self.last_cutoff = 0, -1
        self.counts = defaultdict(Counter)
        self.through = {}

    def advance(self, query):
        cutoff = query.index
        if cutoff < self.last_cutoff:
            raise ValueError('Quick-check history cannot move backward')
        while self.cursor < len(self.completed) and self.completed[self.cursor][0] <= cutoff:
            index, origin, quick = self.completed[self.cursor]
            self.counts[origin]['n'] += 1
            self.counts[origin]['quick_returns'] += int(quick)
            self.through[origin] = index
            self.cursor += 1
        self.last_cutoff = cutoff
        origin = origin_key(query)
        return {'origin': origin, 'through_index': self.through.get(origin),
                'n': self.counts[origin]['n'], 'quick_returns': self.counts[origin]['quick_returns'],
                'input_index': cutoff, 'source': 'completed_prior_same_origin_visits'}


def past_quickcheck_history(visits, event_index, query):
    """Convenience snapshot; reuse PastQuickcheckHistory for ordered many-query runs."""
    return PastQuickcheckHistory(visits, event_index).advance(query)
