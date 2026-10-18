"""Causal conditional statistics over current targets and unordered contexts.

No neural output or candidate inventory filter is used. Contexts redistribute
historical votes within exact-origin and semantic support; global and recency
backoff retain the complete mature navigation vocabulary.
"""
from collections import Counter
from copy import deepcopy
from datetime import datetime
import itertools
import math
from zoneinfo import ZoneInfo

import numpy as np
from scipy.sparse import csr_matrix

from diagnose_action_granularity import NativeSimilarity
from navigation_hierarchy import NavigationHierarchy, SERVICE, ENTRY, disposition, service


SCALES = ('recent5m', 'recent60m', 'long24h', 'recorded_history')
CHANNELS = ('count', 'recent', 'dwell')
FAMILIES = ('browser', 'finder', 'qspace', 'desktop_windows', 'running_apps')
CLOCK_TIME_ZONE = 'America/Los_Angeles'
_CLOCK_ZONE = ZoneInfo(CLOCK_TIME_ZONE)
_CLOCK_HOUR_WIDTHS = {'hour3': 3.0}
PROFILES = {}
for scope, scales in [('recent', SCALES[:2]), ('long', SCALES[2:]), ('combined', SCALES)]:
    for channels in ('count', 'rich'):
        PROFILES['time_'+scope+'_'+channels] = tuple(
            'time/'+scale+'/'+channel for scale in scales
            for channel in (('count',) if channels == 'count' else CHANNELS))
for scope, families in [('all', FAMILIES), ('resources', FAMILIES[:3]), ('apps_windows', FAMILIES[3:])]:
    for channels in ('identity', 'rich'):
        PROFILES['inventory_'+scope+'_'+channels] = tuple(
            'inventory/'+family+'/'+channels for family in families)
# These are prediction-cutoff features, not recent-activity summaries.  The
# fixed zone makes an offline replay independent of the machine that runs it;
# ZoneInfo also handles daylight-saving transitions from the stored timestamp.
PROFILES.update({
    'clock_hour3': ('clock/hour3/feature',),
    'clock_weekday': ('clock/weekday/feature',),
    'clock_hour3_weekday': ('clock/hour3/feature', 'clock/weekday/feature'),
})


def clock_values(timestamp_ms, scope):
    """Nonnegative local-clock features derived only from one input timestamp."""
    timestamp_ms = float(timestamp_ms)
    if not math.isfinite(timestamp_ms):
        raise ValueError('Invalid prediction timestamp for clock context')
    local = datetime.fromtimestamp(timestamp_ms / 1000, _CLOCK_ZONE)
    if scope == 'weekday':
        return {f'weekday:{local.weekday()}': 1.}
    width = _CLOCK_HOUR_WIDTHS.get(scope)
    if width is None:
        raise ValueError('Unknown clock context scope: '+scope)
    hour = local.hour + local.minute / 60 + local.second / 3600 + local.microsecond / 3_600_000_000
    return {f'hour:{center:02d}': math.exp(-.5 * (min(abs(hour-center), 24-abs(hour-center)) / width) ** 2)
            for center in range(24)}


def configurations():
    result = []
    # Prior/default values first, then a bounded grid. Stable ordering breaks ties.
    cores = list(itertools.product((16., 4., 64.), (.25, 0., .5), (0., 24., 168.)))
    for mode in ('none', 'time', 'inventory'):
        profiles = [None] if mode == 'none' else [p for p in PROFILES if p.startswith(mode+'_')]
        for profile in profiles:
            kernels = [(1., 1.)] if mode == 'none' else itertools.product((.75, .5, .25), (1., 2.))
            for floor, power in kernels:
                for semantic, recency, half_life in cores:
                    result.append({'id': f'{mode}-{len(result):04d}', 'mode': mode,
                        'profile': profile, 'floor': floor, 'power': power,
                        'semantic_strength': semantic, 'global_strength': 1.,
                        'recency_mix': recency, 'half_life_hours': half_life})
    return result


def validate_input(q):
    if q['current_sequence'] != [q['origin']]:
        raise ValueError('Foreground sequence must have length one')
    if q['background_through_index'] != q['index'] or q['inventory_through_index'] != q['index']:
        raise ValueError('Context is not frozen at the current cutoff')


def channel_values(q, group):
    """Return a nonnegative content vector and an explicit reliability weight."""
    kind, scope, channel = group.split('/')
    if kind == 'time':
        profile = q['backgrounds'][scope]
        values = profile[channel]
        available = profile['available_span_seconds']
        requested = profile.get('requested_span_seconds')
        reliability = min(1., available/requested) if requested else float(available > 0)
    elif kind == 'clock':
        if channel != 'feature':
            raise ValueError('Unknown clock context channel: '+channel)
        values, reliability = clock_values(q['timestamp'], scope), 1.
    else:
        state, quality = q['inventory_state'], q['inventory_quality']
        reliability = quality.get(scope, 0.) if scope in state else 0.
        values = state.get(scope, {})
        if reliability > 0 and not values:
            return {'observed_empty': 1.}, reliability
        if channel == 'identity':
            values = {k: v for k, v in values.items()
                      if k.startswith(('target:', 'window-class:', 'selected:target:',
                                       'selected:window-class:', 'pinned:target:'))}
    if not math.isfinite(reliability) or not 0 <= reliability <= 1:
        raise ValueError('Invalid source reliability')
    values = {k: v for k, v in values.items() if math.isfinite(v) and v > 0}
    # Empty temporal activity is uninformative, not evidence of complete coverage.
    return values, reliability if values else 0.


class BackgroundIndex:
    """Lossless sparse cosine index. Vocabulary construction fits no statistics.

    Query-to-history similarities only use the two frozen observations. The
    caller supplies mature record positions; future examples never get votes.
    Missing sources stay neutral instead of masquerading as observed emptiness.
    """
    def __init__(self, inputs, profiles=None):
        self.inputs = inputs
        self.groups, self.cache = {}, {}
        self.profiles = tuple(PROFILES if profiles is None else profiles)
        unknown = sorted(set(self.profiles) - set(PROFILES))
        if unknown:
            raise ValueError('Unknown context profile: '+', '.join(unknown))
        for i, q in enumerate(inputs):
            validate_input(q)
            if q['position'] != i:
                raise ValueError('Input position mismatch')
        for group in sorted({g for profile in self.profiles for g in PROFILES[profile]}):
            rows = [channel_values(q, group) for q in inputs]
            columns = {k: i for i, k in enumerate(sorted({k for values, _ in rows for k in values}))}
            row_ids, col_ids, values = [], [], []
            for i, (row, _) in enumerate(rows):
                norm = math.sqrt(sum(v*v for v in row.values()))
                for k, v in row.items():
                    row_ids.append(i); col_ids.append(columns[k]); values.append(v/norm)
            matrix = csr_matrix((values, (row_ids, col_ids)), shape=(len(inputs), len(columns)))
            self.groups[group] = (matrix, np.asarray([r for _, r in rows]))

    def similarities(self, position, profile):
        if profile not in self.profiles:
            raise ValueError('Profile was not indexed: '+profile)
        key = position, profile
        if key not in self.cache:
            weighted = np.zeros(len(self.inputs)); available = weighted.copy()
            groups = PROFILES[profile]
            for group in groups:
                matrix, reliability = self.groups[group]
                shared = np.minimum(reliability[position], reliability)
                similarity = (matrix[position] @ matrix.T).toarray()[0]
                weighted += shared*np.clip(similarity, 0., 1.)
                available += shared
            similarity = np.divide(weighted, available, out=np.ones_like(weighted), where=available > 0)
            self.cache[key] = similarity, available/len(groups)
        return self.cache[key]

    def factors(self, position, before_positions, config):
        if config['profile'] is None or config['floor'] == 1:
            return np.ones(len(before_positions))
        similarity, reliability = self.similarities(position, config['profile'])
        s, r = similarity[before_positions], reliability[before_positions]
        return 1.-(1.-config['floor'])*r*(1.-s**config['power'])


def normalize(values):
    total = values.sum()
    return values/total if total else values.copy()


class StatisticalMemory:
    def __init__(self):
        self.hierarchy = NavigationHierarchy()
        self.records = []
        self.training = []

    def learn(self, q, record, ready_index):
        validate_input(q)
        if ready_index < q['index']:
            raise ValueError('Label matures before its input')
        count = len(self.hierarchy.records)
        base_count = len(self.hierarchy.base.records[0])
        self.hierarchy.learn(record)
        if len(self.hierarchy.records) == count:
            return
        self.training.append({'position': q['position'], 'ready_index': ready_index,
                              'record': deepcopy(record)})
        r = self.hierarchy.records[-1]
        # Several concrete origins can share one service origin. The native
        # statistics deduplicate again after that fold; retain the same votes.
        if len(self.hierarchy.base.records[0]) > base_count:
            self.records.append({'position': q['position'], 'origin': r['navigationOrigin'],
                'annotation': q['annotation'], 'label': r['navigationLabel'],
                'weight': r['weight'], 'timestamp': r['outcomeTimestamp'],
                'outcome': r['outcomeID'], 'ready_index': ready_index})

    def prepare(self, q, home, background):
        if any(r['ready_index'] >= q['index'] for r in self.training):
            raise ValueError('Unmatured evidence reached a prediction')
        return PreparedQuery(q, self.records, self.hierarchy, home, background)


class PreparedQuery:
    def __init__(self, q, records, hierarchy, home, background):
        validate_input(q)
        self.q, self.background = q, background
        self.labels = sorted({r['label'] for r in records})
        label_index = {label: i for i, label in enumerate(self.labels)}
        n = len(self.labels)
        current = service(q['annotation']['identity']) or q['origin']
        self.positions = np.asarray([r['position'] for r in records], dtype=np.int64)
        self.label_ids = np.asarray([label_index[r['label']] for r in records], dtype=np.int64)
        self.weights = np.asarray([r['weight'] for r in records])
        self.age_hours = np.asarray([max(0., q['timestamp']-r['timestamp'])/3600000 for r in records])
        self.exact = np.asarray([r['origin'] == current for r in records], dtype=bool)
        similarity = NativeSimilarity()
        annotation_cache = {}
        relation = []
        for record, same in zip(records, self.exact):
            key = id(record['annotation'])
            if not same and key not in annotation_cache:
                annotation_cache[key] = similarity.score(q['annotation'], record['annotation'])
            relation.append(0. if same else annotation_cache[key])
        self.relation = np.asarray(relation)
        self.confidence = max(relation, default=0.)
        groups, group_labels, group_ids, seen, global_ids = {}, [], [], set(), []
        last = {}
        for i, record in enumerate(records):
            key = record['label'], record['outcome']
            if key not in groups:
                groups[key] = len(groups); group_labels.append(label_index[record['label']])
            group_ids.append(groups[key])
            if record['outcome'] not in seen:
                global_ids.append(i); seen.add(record['outcome'])
            last[record['label']] = max(last.get(record['label'], -math.inf), record['timestamp'])
        self.group_ids = np.asarray(group_ids, dtype=np.int64)
        self.group_labels = np.asarray(group_labels, dtype=np.int64)
        self.global_ids = np.asarray(global_ids, dtype=np.int64)
        recency = np.zeros(n)
        for i, label in enumerate(sorted(last, key=lambda k: (-last[k], k))):
            recency[label_index[label]] = 1/(i+1)
        self.recency = normalize(recency)
        mapped = []
        for label in self.labels:
            selected = (hierarchy.specific(label, q['origin']) or label) if label == SERVICE else label
            if disposition(selected, q['annotation']['identity']) != 'navigation':
                mapped.append(None)
            elif selected == SERVICE and q['target'].get('locator') == ENTRY:
                mapped.append(None)
            else:
                mapped.append(selected)
        self.emitted = sorted({label for label in mapped if label is not None})
        output_ids = {label: i for i, label in enumerate(self.emitted)}
        self.keep = np.asarray([i for i, label in enumerate(mapped) if label is not None], dtype=np.int64)
        self.output_ids = np.asarray([output_ids[mapped[i]] for i in self.keep], dtype=np.int64)
        self.home = home
        self.parts_cache, self.decay_cache = {}, {}

    def _semantic(self, weights):
        grouped = np.zeros(len(self.group_labels))
        np.maximum.at(grouped, self.group_ids, weights*self.relation)
        return np.bincount(self.group_labels, weights=grouped, minlength=len(self.labels))

    def components(self, config):
        half = config['half_life_hours']
        key = config['profile'], config['floor'], config['power'], half
        if key in self.parts_cache:
            return self.parts_cache[key]
        if half not in self.decay_cache:
            weights = self.weights*(2.**(-self.age_hours/half) if half else 1.)
            exact = np.bincount(self.label_ids, weights=weights*self.exact, minlength=len(self.labels))
            semantic = normalize(self._semantic(weights))*self.confidence
            global_counts = np.bincount(self.label_ids[self.global_ids],
                weights=weights[self.global_ids], minlength=len(self.labels))
            self.decay_cache[half] = weights, exact, semantic, normalize(global_counts)
        weights, exact, semantic, global_counts = self.decay_cache[half]
        if config['profile'] is not None and config['floor'] < 1:
            factors = self.background.factors(self.q['position'], self.positions, config)
            conditioned = weights*factors
            weighted_exact = np.bincount(self.label_ids, weights=conditioned*self.exact, minlength=len(self.labels))
            exact = normalize(weighted_exact)*exact.sum()
            semantic = normalize(self._semantic(conditioned))*semantic.sum()
        parts = exact, semantic, global_counts, self.recency
        self.parts_cache[key] = parts
        return parts

    def scores(self, config):
        exact, semantic, global_counts, recency = self.components(config)
        votes = exact+config['semantic_strength']*semantic+config['global_strength']*global_counts
        scores = (1-config['recency_mix'])*normalize(votes)+config['recency_mix']*recency
        emitted = np.bincount(self.output_ids, weights=scores[self.keep], minlength=len(self.emitted))
        return dict(zip(self.emitted, map(float, emitted)))

    def predict(self, config):
        scores = self.scores(config)
        raw = sorted(scores, key=lambda k: (-scores[k], k))
        resolved = list(dict.fromkeys(self.home if label == SERVICE and self.home else label for label in raw))
        return {'predictions': resolved[:4], 'raw': raw[:4], 'rankable': resolved}


def prepare_queries(inputs, training, rows, homes, background):
    """Replay unique mature records. The input stream is maturity-sorted."""
    memory = StatisticalMemory(); cursor = 0; prepared = {}
    for row in sorted(rows, key=lambda r: r['position']):
        q = inputs[row['position']]
        while cursor < len(training) and training[cursor]['ready_index'] < q['index']:
            item = training[cursor]; before = inputs[item['position']]
            if before['index'] >= q['index']:
                raise ValueError('Future input reached learning')
            memory.learn(before, item['record'], item['ready_index']); cursor += 1
        prepared[q['position']] = memory.prepare(q, homes[q['position']], background)
    return prepared


def select_validation(results, mode, core=None):
    def key(row):
        m = row['validation']
        return tuple(m[g][v] for g, v in (
            ('original', 'top4_hits'), ('expanded', 'top4_hits'),
            ('original', 'top1_hits'), ('expanded', 'top1_hits'),
            ('all', 'top4_hits'), ('all', 'top1_hits')))
    candidates = [r for r in results if r['config']['mode'] == mode and
        (core is None or all(r['config'][k] == core[k] for k in
            ('semantic_strength', 'global_strength', 'recency_mix', 'half_life_hours')))]
    return max(candidates, key=key)
