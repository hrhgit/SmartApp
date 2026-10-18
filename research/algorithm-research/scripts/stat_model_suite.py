"""Incrementally updated, current-interface statistical recommendation models.

The replay owns label maturity and the fixed evaluation membership.  These models
see only mature training transitions and prediction-time candidate labels.  The
length-one origin is an opaque semantic key: no history sequence, test answer,
background inventory similarity, or future label vocabulary is read here.

All configurations use the same memory ledger; the names denote scoring rules,
not separate claims about independently implemented learning algorithms.  The
finite-count window is also the exact-current, k-most-recent-neighbour baseline.
"""

from collections import Counter, defaultdict
from dataclasses import dataclass
import math


HOUR_MS = 3_600_000.0
TIE_EPSILON = 1e-9


def configurations():
    """A fixed, modest grid; no dataset inspection or adaptive tuning here."""
    configs = []

    def add(family, suffix, **parameters):
        configs.append({'id': f'stat_{family}_{suffix}', 'family': family,
                        'parameters': parameters})

    for scope in ('conditional', 'app', 'global'):
        add('frequency', scope, scope=scope)
        add('recency', scope, scope=scope)
    for k in (1, 2, 4, 8, 16, 32, 64, 128):
        add('recent_memory', str(k), k=k)
    for hours in (1 / 12, .25, 1, 4, 12, 24, 48, 72):
        add('time_window', f'{hours:g}h', hours=hours)
    for hours in (1 / 30, 1 / 12, .25, 1, 4, 12, 24, 48, 96):
        add('time_decay', f'{hours:g}h', half_life_hours=hours)
    for half_life in (1, 2, 4, 8, 16, 32, 64, 128):
        add('event_decay', str(half_life), half_life_events=half_life)
    for recency in ('rank', 'decay_4h'):
        for weight in (.1, .25, .5, .75, .9):
            add('count_recency', f'{recency}_{weight:g}', recency=recency,
                recency_weight=weight)
    for offset in (0, 3, 10):
        for weight in (.25, .5, .75):
            add('rank_fusion', f'{offset}_{weight:g}', rank_offset=offset,
                recency_weight=weight)
    for prior in ('global_frequency', 'global_decay_6h', 'app_frequency'):
        for alpha in (.1, 1, 3, 10, 30, 100):
            add('bayesian_smoothing', f'{prior}_{alpha:g}', prior=prior,
                alpha=alpha)
    for conditional in ('frequency', 'decay_4h', 'decay_24h'):
        for weight in (.1, .25, .5, .75):
            add('global_mixture', f'{conditional}_{weight:g}',
                conditional=conditional, global_weight=weight)
    for width in (1, 3, 6, 12):
        for half_life in (0, 6, 24):
            add('clock_kernel', f'{width}h_{half_life}h',
                hour_bandwidth=width, half_life_hours=half_life)
    add('candidate_recency', 'observed')
    for mode in ('probability', 'rank'):
        for weight in (.1, .25, .5, .75, .9):
            add('count_candidate_recency', f'{mode}_{weight:g}', mode=mode,
                recency_weight=weight)
    assert len({x['id'] for x in configs}) == len(configs)
    return configs


@dataclass(frozen=True)
class Experience:
    origin: object
    app: object
    label: str
    timestamp: float
    outcome_timestamp: float
    serial: int


def _probability(weights):
    total = sum(weights.values())
    return {key: value / total for key, value in weights.items()} if total else {}


def _ranks(weights, latest=None):
    latest = latest or {}
    return {key: i + 1 for i, key in enumerate(sorted(
        weights, key=lambda key: (-weights[key], -latest.get(key, -math.inf), key)))}


def _mix(left, right, right_weight):
    return {key: (1 - right_weight) * left.get(key, 0) + right_weight * right.get(key, 0)
            for key in left.keys() | right.keys()}


def _latest(records):
    result = {}
    for record in records:
        result[record.label] = max(result.get(record.label, -math.inf),
                                   record.outcome_timestamp)
    return result


def _frequency(records):
    return Counter(record.label for record in records)


def _time_decay(records, now, half_life_hours):
    # A common age factor cancels from rankings and normalized probabilities.
    # Anchor at the newest mature outcome so a long inactive gap does not make
    # every historical weight underflow to zero.
    anchor = max((record.outcome_timestamp for record in records), default=now)
    result = defaultdict(float)
    for record in records:
        result[record.label] += 2.0 ** (
            -(anchor - record.outcome_timestamp) / (half_life_hours * HOUR_MS))
    return dict(result)


def _reciprocal_recency(latest, offset=0):
    return {key: 1 / (offset + rank) for key, rank in _ranks(latest).items()}


class StatisticalModel:
    """Learn immediately from each confirmed transition; accept unseen labels."""

    def __init__(self, config):
        self.config = config
        self.by_origin = defaultdict(list)
        self.by_app = defaultdict(list)
        self.global_records = []
        self._conditional_ids = set()
        self._app_ids = set()
        self._global_ids = set()
        self._serial = 0
        self._latest_outcome = -math.inf

    def learn(self, record):
        # outcome_id allows global/app priors to count actual arrivals once even
        # when one arrival is a mature response to multiple origin observations.
        event_id = record.get('outcome_id')
        if event_id is None:
            event_id = ('anonymous_training_record', self._serial)
        identity = (record['origin'], event_id)
        if identity in self._conditional_ids:
            return False
        timestamp = float(record['timestamp'])
        outcome_timestamp = float(record['outcome_timestamp'])
        if not math.isfinite(timestamp) or not math.isfinite(outcome_timestamp):
            raise ValueError('non-finite training timestamp')
        if outcome_timestamp < timestamp:
            raise ValueError('outcome precedes its source observation')
        experience = Experience(record['origin'], record.get('app'),
                                record['label'], timestamp, outcome_timestamp,
                                self._serial)
        self._serial += 1
        self._conditional_ids.add(identity)
        self.by_origin[experience.origin].append(experience)
        app_identity = (experience.app, event_id)
        if app_identity not in self._app_ids:
            self.by_app[experience.app].append(experience)
            self._app_ids.add(app_identity)
        if event_id not in self._global_ids:
            self.global_records.append(experience)
            self._global_ids.add(event_id)
        self._latest_outcome = max(self._latest_outcome, outcome_timestamp)
        return True

    def support_counts(self, origin):
        """Ungraded historical evidence only; no dependence on a query answer."""
        return dict(_frequency(self.by_origin.get(origin, ())))

    def scores(self, query):
        now = float(query['timestamp'])
        if not math.isfinite(now):
            raise ValueError('non-finite prediction timestamp')
        if self._latest_outcome >= now:
            raise ValueError('training outcome is not strictly before prediction')
        labels = set(query['labels'])
        records = self.by_origin.get(query['origin'], ())
        app_records = self.by_app.get(query.get('app'), ())
        family = self.config['family']
        p = self.config['parameters']
        latest = _latest(records)
        counts = _frequency(records)

        if family in ('frequency', 'recency'):
            scoped = {'conditional': records, 'app': app_records,
                      'global': self.global_records}[p['scope']]
            latest = _latest(scoped)
            weights = (_frequency(scoped) if family == 'frequency'
                       else _reciprocal_recency(latest))
        elif family == 'recent_memory':
            # Nearest in recency among identical-current observations, with a
            # frequency vote; sorting handles delayed arrivals learned out of order.
            selected = sorted(records, key=lambda r: (r.outcome_timestamp, r.serial))[-p['k']:]
            weights = _frequency(selected)
            latest = _latest(selected)
        elif family == 'time_window':
            selected = [r for r in records if now - r.outcome_timestamp <= p['hours'] * HOUR_MS]
            weights = _frequency(selected)
            latest = _latest(selected)
        elif family == 'time_decay':
            weights = _time_decay(records, now, p['half_life_hours'])
        elif family == 'event_decay':
            weights = defaultdict(float)
            ordered = sorted(records, key=lambda r: (r.outcome_timestamp, r.serial), reverse=True)
            for age, record in enumerate(ordered):
                weights[record.label] += 2.0 ** (-age / p['half_life_events'])
        elif family == 'count_recency':
            anchor = max(latest.values(), default=now)
            recent = (_reciprocal_recency(latest) if p['recency'] == 'rank'
                      else {key: 2.0 ** (-(anchor - ts) / (4 * HOUR_MS))
                            for key, ts in latest.items()})
            weights = _mix(_probability(counts), _probability(recent), p['recency_weight'])
        elif family == 'rank_fusion':
            count_rank = {key: 1 / (p['rank_offset'] + rank)
                          for key, rank in _ranks(counts, latest).items()}
            weights = _mix(count_rank, _reciprocal_recency(latest, p['rank_offset']),
                           p['recency_weight'])
        elif family == 'bayesian_smoothing':
            if p['prior'] == 'global_frequency':
                prior = _probability(_frequency(self.global_records))
            elif p['prior'] == 'app_frequency':
                prior = _probability(_frequency(app_records))
            else:
                prior = _probability(_time_decay(self.global_records, now, 6))
            denominator = sum(counts.values()) + p['alpha']
            weights = {key: (counts.get(key, 0) + p['alpha'] * prior.get(key, 0)) / denominator
                       for key in counts.keys() | prior.keys()}
        elif family == 'global_mixture':
            conditional = counts
            if p['conditional'] != 'frequency':
                hours = 4 if p['conditional'] == 'decay_4h' else 24
                conditional = _time_decay(records, now, hours)
            weights = _mix(_probability(conditional),
                           _probability(_frequency(self.global_records)), p['global_weight'])
        elif family == 'clock_kernel':
            # Circular distance needs no timezone assumption: any fixed UTC offset
            # cancels. This is a daily-time kernel, not previous-page history.
            weights = defaultdict(float)
            anchor = max(latest.values(), default=now)
            for record in records:
                distance = abs((now - record.timestamp) / HOUR_MS) % 24
                distance = min(distance, 24 - distance)
                weight = math.exp(-.5 * (distance / p['hour_bandwidth']) ** 2)
                if p['half_life_hours']:
                    weight *= 2.0 ** (-(anchor - record.outcome_timestamp)
                                     / (p['half_life_hours'] * HOUR_MS))
                weights[record.label] += weight
        elif family in ('candidate_recency', 'count_candidate_recency'):
            observed = {key: float(ts) for key, ts in query.get('candidate_last_seen', {}).items()
                        if key in labels}
            if any(not math.isfinite(ts) or ts >= now for ts in observed.values()):
                raise ValueError('candidate visit must be strictly before prediction')
            recent = _reciprocal_recency(observed)
            if family == 'candidate_recency':
                weights, latest = recent, observed
            elif p['mode'] == 'probability':
                weights = _mix(_probability(counts), _probability(recent), p['recency_weight'])
            else:
                ranks = {key: 1 / rank for key, rank in _ranks(counts, latest).items()}
                weights = _mix(ranks, recent, p['recency_weight'])
        else:
            raise ValueError(f'unknown statistical family: {family}')

        # Conditional frequency preserves the old count/latest-outcome ordering.
        # Bound the perturbation by the smallest nonzero score difference so it
        # cannot reverse close decay/kernel scores. Semantic IDs break any exact
        # remaining floating-point ties in the caller. Unavailable historical
        # labels have remained in all confidence denominators above.
        times = {timestamp: rank + 1 for rank, timestamp in enumerate(sorted(set(latest.values())))}
        positive = {key: float(value) for key, value in weights.items() if key in labels and value > 0}
        distinct = sorted(set(positive.values()))
        margin = min((right - left for left, right in zip(distinct, distinct[1:])), default=1.0)
        epsilon = min(TIE_EPSILON, margin / 4)
        return {key: value + epsilon * times.get(latest.get(key), 0) / (len(times) + 1)
                for key, value in positive.items()}


def build(config):
    return StatisticalModel(config)
