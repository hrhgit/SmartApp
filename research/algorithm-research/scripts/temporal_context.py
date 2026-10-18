"""Causal foreground time context and incremental conditional statistics.

Inventory, clipboard, menu and future labels are not feature sources. Contexts
are frozen at each input event; later refinements never rewrite stored inputs.
"""
from collections import Counter
from copy import deepcopy
import math

import evaluate_context_knn as core
from known_inactivity import InactivityClock
from diagnose_action_granularity import NativeSimilarity
from navigation_hierarchy import NavigationHierarchy, SERVICE, ENTRY, disposition, service
from semantic_online import SemanticStatistics, normalize, rank
from trajectory_supervision import action_key, _semantic_origin


VERSION = 'foreground-temporal-context-v2'
WINDOWS = (5, 15, 60)


def target_key(target):
    return _semantic_origin(target)


def tokens(target):
    key = target_key(target)
    result = {'target:' + key: 1.} if key else {}
    if target.get('bundleID'):
        result['owner:' + target['bundleID']] = .25
    return result


class TemporalHistory:
    """A single forward pass; counters contain confirmed changes only.

    First/resumed observations establish state but do not assert a switch.
    Refinements replace the episode's known identity only from that event on;
    dwell before refinement keeps the earlier, coarser identity. Time summaries
    may retain earlier verified visits across a gap, but ordered suffixes and
    dwell never bridge the gap. Idle time is not foreground-use duration.
    """
    def __init__(self, excluded_sessions=()):
        self.excluded = set(excluded_sessions)
        self.session = None
        self.catalog = {}
        self.current = None
        self.episodes, self.spans, self.sequence = [], [], []
        self.idle = False
        self.index = -1
        self.segment = 0
        self.last_boundary_ms = None
        self.audit = Counter()
        self.inactivity = InactivityClock()
        self.paused = False

    def _related(self, source):
        if self.current is None:
            return False
        own = self.current['source']
        return source == own or (core.family(source) != 'other' and core.family(source) == core.family(own))

    def _span(self, event):
        if self.current is None or self.idle or self.paused:
            return None
        c = self.current
        elapsed = event['monotonicMs'] - c['span_mono']
        wall = event['timestampMs'] - c['span_ms']
        # Reject clock discontinuities instead of assigning them to a target.
        if elapsed < 0 or wall < 0 or abs(wall - elapsed) > 2000:
            return None
        if elapsed == 0:
            return None
        return {'start_ms': c['span_ms'], 'end_ms': event['timestampMs'],
                'seconds': elapsed / 1000, 'target': deepcopy(c['target']),
                'through_index': self.index}

    def _close_span(self, event):
        span = self._span(event)
        if span:
            self.spans.append(span)
        if self.current is not None:
            self.current['span_ms'] = event['timestampMs']
            self.current['span_mono'] = event['monotonicMs']

    def _boundary(self, event, reason):
        if self.current is not None:
            # A delayed failure/gap cannot prove dwell up to its delivery time.
            self._close_span(self.current['last_proof'])
        self.current = None
        self.sequence = []
        self.segment += 1
        self.last_boundary_ms = event['timestampMs']
        self.audit['boundary:' + reason] += 1

    def _start(self, event, target, confirmed):
        episode = {'arrival_id': event['id'], 'arrival_index': self.index,
                   'arrival_ms': event['timestampMs'], 'identity_known_ms': event['timestampMs'],
                   'target': deepcopy(target), 'confirmed': confirmed,
                   'evidence_index': self.index}
        self.episodes.append(episode)
        self.sequence.append(episode)
        self.current = {'episode': episode, 'target': deepcopy(target), 'source': event['source'],
                        'span_ms': event['timestampMs'], 'span_mono': event['monotonicMs'],
                        'last_proof': event, 'entrance': event.get('data', {}).get('entranceID'),
                        'linked_ids': {event['id']}}
        self.audit['confirmed_arrivals' if confirmed else 'initial_or_resumed_states'] += 1

    def consume(self, event, index):
        if index <= self.index:
            raise ValueError('Temporal history must consume strictly increasing event indices')
        self.index = index
        if event['sessionID'] != self.session:
            self._boundary(event, 'session')
            self.session, self.catalog, self.idle = event['sessionID'], {}, False
        if self.session in self.excluded:
            self.audit['excluded_validation_events'] += 1
            return
        event = self.inactivity.consume(event)
        kind, data = event['kind'], event.get('data', {})
        disposition = event['continuityDisposition']
        if disposition == 'suspended':
            if not self.paused:
                self._close_span(event)
            self.paused = True
            self.audit['known_inactivity_events'] += 1
            return
        if self.paused and disposition == 'resumed' and self.current:
            self.current['span_ms'] = event['timestampMs']
            self.current['span_mono'] = event['monotonicMs']
            self.current['last_proof'] = event
            self.idle = False
        self.paused = False
        if kind == 'target.update':
            target = data['target']
            if core.admitted(target) and not core.synthetic(target) and action_key(target):
                self.catalog[target['id']] = deepcopy(target)
            else:
                self.catalog.pop(target['id'], None)
                if self.current and self.current['target']['id'] == target['id']:
                    self._boundary(event, 'target_revoked')
            return
        if kind in {'capture.gap', 'session.end'} or disposition == 'boundary':
            self._boundary(event, kind)
            self.idle = False
            return
        if kind == 'source.status':
            if self._related(event['source']):
                status = data.get('observationStatus')
                if status in {'unavailable', 'stale', 'initializing'}:
                    self._boundary(event, 'foreground_source_unavailable')
                elif status == 'ready':
                    self.current['last_proof'] = event
            return
        if kind == 'idle.change':
            self._close_span(event)
            self.idle = bool(data.get('idle'))
            return
        if kind != 'context.change':
            return
        target = deepcopy(self.catalog.get(data.get('targetID')))
        if target is None:
            self._boundary(event, 'unadmitted_context')
            return
        target['bundleID'] = event.get('app', {}).get('bundleID') or target.get('bundleID')
        if not action_key(target):
            self._boundary(event, 'unresolved_owner')
            return
        resolution = data.get('reason') == 'resolution_update'
        recovery = data.get('resolutionEvidence') == 'source_recovered_behavior_unobserved'
        if recovery:
            self._boundary(event, 'resumed_resolution')
        current = self.current
        linked = resolution and current and not recovery and (
            data.get('resolvesContextEventID') in current['linked_ids']
            or (not data.get('resolvesContextEventID') and
                current['target'].get('bundleID') == target.get('bundleID')))
        if linked:
            self._close_span(event)
            current['target'] = deepcopy(target)
            current['source'] = event['source']
            current['entrance'] = data.get('entranceID')
            current['last_proof'] = event
            current['linked_ids'].add(event['id'])
            current['episode'].update(target=deepcopy(target), identity_known_ms=event['timestampMs'],
                                      evidence_index=index)
            self.audit['refinements_without_extra_count'] += 1
            return
        if resolution and current:
            self._boundary(event, 'unlinked_resolution')
            current = None
        if current and current['target']['id'] == target['id'] and current['entrance'] == data.get('entranceID'):
            current['last_proof'] = event
            current['linked_ids'].add(event['id'])
            self.audit['repeated_foreground_observations'] += 1
            return
        confirmed = current is not None and not resolution
        if current:
            previous = current['target']
            if (previous['kind'] in {'page', 'directory', 'file'} and target['kind'] in {'app', 'window'}
                    and previous.get('bundleID') == target.get('bundleID')):
                # Losing the child locator does not establish a new destination.
                self._boundary(event, 'granularity_loss')
                confirmed = False
            else:
                self._close_span(event)
        self._start(event, target, confirmed)

    def snapshot(self, event):
        now = event['timestampMs']
        spans = self.spans + ([span] if (span := self._span(event)) else [])
        windows = {}
        for minutes in WINDOWS:
            duration = minutes * 60000
            start = now - duration
            count, decayed, recent, dwell = Counter(), Counter(), Counter(), Counter()
            arrivals, known_seconds = 0, 0.
            for episode in self.episodes:
                if not episode['confirmed'] or not start <= episode['arrival_ms'] <= now:
                    continue
                arrivals += 1
                # Counts use the known episode once; recency of a newly resolved
                # concrete identity starts when that identity became observable.
                age = now - max(episode['arrival_ms'], episode['identity_known_ms'])
                weight = 2. ** (-age / (duration / 2))
                for token, value in tokens(episode['target']).items():
                    count[token] += value
                    decayed[token] += value * weight
                    recent[token] = max(recent[token], value * weight)
            for span in spans:
                overlap = max(0., min(now, span['end_ms']) - max(start, span['start_ms']))
                width = span['end_ms'] - span['start_ms']
                seconds = span['seconds'] * overlap / width if width > 0 else 0.
                known_seconds += seconds
                for token, value in tokens(span['target']).items():
                    dwell[token] += value * seconds
            windows[str(minutes)] = {'count': dict(count), 'decayed_count': dict(decayed),
                                     'recent': dict(recent), 'dwell': dict(dwell),
                                     'confirmed_arrivals': arrivals, 'observed_nonidle_seconds': known_seconds,
                                     'observed_nonidle_fraction': min(1., known_seconds / (minutes * 60))}
        sequence = [target_key(e['target']) for e in self.sequence[-5:]]
        return {'through_index': self.index, 'timestamp': now, 'windows': windows,
                'sequence': sequence, 'sequence_length': len(sequence),
                'segment': self.segment, 'idle': self.idle,
                'last_boundary_ms': self.last_boundary_ms,
                'duration_basis': 'observed_continuity_clipped_at_gaps_excluding_observed_idle'}


def build_contexts(events, observations, excluded_sessions=()):
    by_index = {q['index']: q for q in observations}
    if len(by_index) != len(observations):
        raise ValueError('Duplicate input indices')
    history = TemporalHistory(excluded_sessions)
    contexts, homes, catalog = {}, {}, {}
    # Service resolution uses the pre-cutoff catalog only, outside model input.
    for index, event in enumerate(events):
        history.consume(event, index)
        if event['kind'] == 'target.update':
            target = event['data']['target']
            if core.admitted(target) and not core.synthetic(target):
                catalog[target['id']] = target
            else:
                catalog.pop(target['id'], None)
        if index in by_index:
            q = by_index[index]
            if event['id'] != q['anchor_id'] or event['sessionID'] != q['session']:
                raise ValueError('Temporal input does not match frozen anchor')
            contexts[q['position']] = history.snapshot(event)
            homes[q['position']] = next((key for key, t in sorted(catalog.items())
                                       if t['kind'] == 'page' and t.get('locator') == ENTRY), None)
    if len(contexts) != len(observations):
        raise ValueError('Missing temporal inputs')
    return contexts, homes, dict(history.audit)


def cosine(a, b):
    if not a or not b:
        return None
    denominator = math.sqrt(sum(v*v for v in a.values()) * sum(v*v for v in b.values()))
    return sum(v * b.get(k, 0.) for k, v in a.items()) / denominator if denominator else None


def ordered_similarity(a, b):
    """Match ordered suffixes of length 2..5; length-one is the base origin."""
    values = []
    for length in range(2, 6):
        if len(a) >= length and len(b) >= length:
            values.append(float(a[-length:] == b[-length:]))
    return sum(values) / len(values) if values else None


class ContextSimilarity:
    def __init__(self):
        self.cache = {}

    def score(self, left, right, config):
        key = (config['mode'], tuple(config.get('windows', ())), config.get('sequence_weight', 0.),
               left['through_index'], right['through_index'])
        if key in self.cache:
            return self.cache[key]
        mode = config['mode']
        terms = []
        sequence_weight = 1. if mode == 'sequence' else config.get('sequence_weight', 0.)
        if mode != 'sequence':
            channels = ['count'] if mode == 'frequency' else ['decayed_count', 'recent', 'dwell']
            for minutes in config['windows']:
                a, b = left['windows'][str(minutes)], right['windows'][str(minutes)]
                values = [value for channel in channels
                          if (value := cosine(a[channel], b[channel])) is not None]
                if values:
                    terms.append(sum(values) / len(values))
        timed = sum(terms) / len(terms) if terms else None
        sequence = ordered_similarity(left['sequence'], right['sequence'])
        available = [(timed, 1-sequence_weight), (sequence, sequence_weight)]
        available = [(v,w) for v,w in available if v is not None and w > 0]
        value = sum(v*w for v,w in available) / sum(w for _,w in available) if available else None
        self.cache[key] = value
        return value


class TemporalStatistics(SemanticStatistics):
    """Reweight mature transition evidence by its earlier temporal context.

    Current-origin and semantic evidence retain their original total strength;
    only the distribution within each changes. Global/recency backoff stays the
    same, so the learned output vocabulary is never limited to recent targets.
    """
    def __init__(self, config, context_similarity=None):
        super().__init__(semantic_strength=16, global_strength=1, recency_mix=.25,
                         similarity=NativeSimilarity())
        self.config = config
        self.context_similarity = context_similarity or ContextSimilarity()

    def components(self, q, step=0):
        original = super().components(q, step)
        if self.config['mode'] == 'baseline':
            return original
        exact, votes = Counter(), {}
        for before, label, weight, _, outcome_id in self.records[step]:
            same = before['origin'] == q['origin']
            relation = 1. if same else self.similarity.score(q['annotation'], before['annotation'])
            if relation <= 0:
                continue
            similarity = self.context_similarity.score(q['temporal'], before['temporal'], self.config)
            factor = (1. if similarity is None else self.config['floor'] +
                      (1-self.config['floor']) * similarity**2)
            if same:
                exact[label] += weight * factor
            else:
                key = (label, outcome_id)
                votes[key] = max(votes.get(key, 0.), weight * relation * factor)
        semantic = Counter()
        for (label, _), value in votes.items():
            semantic[label] += value
        # Preserve the baseline evidence mass, including weak semantic matches.
        exact = {k:v*sum(original[0].values()) for k,v in normalize(exact).items()}
        semantic = {k:v*sum(original[1].values()) for k,v in normalize(semantic).items()}
        return exact, semantic, original[2], original[3]


class TemporalNavigationModel:
    def __init__(self, config, context_similarity=None):
        self.config = config
        self.hierarchy = NavigationHierarchy()
        self.statistics = TemporalStatistics(config, context_similarity)

    def learn(self, q, record):
        before = len(self.hierarchy.records)
        self.hierarchy.learn(record)
        if len(self.hierarchy.records) == before:
            return
        r = self.hierarchy.records[-1]
        if r['navigationDisposition'] != 'navigation':
            return
        current = {'origin': r['navigationOrigin'], 'annotation': q['annotation'],
                   'timestamp': q['timestamp'], 'temporal': q['temporal']}
        self.statistics.learn(current, r['navigationLabel'], step=0, weight=r['weight'],
                              outcome_id=r['outcomeID'], outcome_timestamp=r['outcomeTimestamp'])

    def scores(self, q):
        # Explicit whitelist: background_state, truth and evidence grades cannot enter.
        current = {'origin': service(q['annotation']['identity']) or q['origin'],
                   'annotation': q['annotation'], 'timestamp': q['timestamp'], 'temporal': q['temporal']}
        result = Counter()
        for label, score in self.statistics.predict(current, 0).items():
            selected = (self.hierarchy.specific(label, q['origin']) or label) if label == SERVICE else label
            if disposition(selected, q['annotation']['identity']) != 'navigation':
                continue
            if selected == SERVICE and q['target'].get('locator') == ENTRY:
                continue
            result[selected] += score
        return dict(result)

    def ranked(self, q, n=4):
        return rank(self.scores(q), n)
