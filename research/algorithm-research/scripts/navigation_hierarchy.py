"""Native-parity implementation of semantic-navigation-hierarchy-v2.

Only derived supervision changes. Original locators, observed outcomes, input
metadata and maturity remain intact. No candidate inventories are accepted.
"""
from collections import Counter
from copy import deepcopy
import math
from urllib.parse import urlsplit, parse_qsl
from uuid import UUID

from diagnose_action_granularity import NativeSimilarity
from semantic_annotations import annotate_target
from semantic_online import SemanticStatistics, rank

VERSION = 'semantic-navigation-hierarchy-v2'
SERVICE = 'service:https://chatgpt.com/'
ENTRY = 'https://chatgpt.com/'
QUERY_KEYS = {'messageId', 'src', 'historySearchQuery', 'no_universal_links'}


def service(identity):
    if not identity or identity.get('kind') != 'page' or not identity.get('locator'):
        return None
    try:
        url = urlsplit(identity['locator'])
        if (url.scheme.lower() != 'https' or url.hostname != 'chatgpt.com'
                or url.port not in {None, 443} or url.username is not None or url.password is not None
                or any(k not in QUERY_KEYS for k, _ in parse_qsl(url.query, keep_blank_values=True))
                or url.fragment.startswith('/')):
            return None
        if url.path in {'', '/'}:
            return SERVICE
        parts = url.path.split('/')
        if len(parts) == 3 and parts[:2] == ['', 'c'] and len(parts[2]) == 36:
            UUID(parts[2])
            return SERVICE
    except (ValueError, TypeError):
        pass
    return None


def disposition(label, identity):
    if label == identity.get('id'):
        return 'current_destination'
    if identity.get('bundleID') and label == 'app:' + identity['bundleID']:
        return 'same_application_unspecified'
    return 'navigation'


def origin(target):
    return 'window_class:' + target['bundleID'] if target['kind'] == 'window' else target['id']


class NavigationHierarchy:
    def __init__(self, records=()):
        self.records, self.seen = [], set()
        self.base = SemanticStatistics(semantic_strength=16, global_strength=1,
                                       recency_mix=.25, similarity=NativeSimilarity())
        for record in records:
            self.learn(record)

    def learn(self, raw):
        if not (math.isfinite(raw['weight']) and raw['weight'] > 0
                and math.isfinite(raw['outcomeTimestamp']) and raw['origin'] and raw['label']):
            return
        key = raw['origin'], raw['label'], raw['outcomeID']
        if key in self.seen:
            return
        self.seen.add(key)
        record = deepcopy(raw)
        record['navigationDisposition'] = disposition(raw['label'], raw['annotation']['identity'])
        record['navigationOrigin'] = service(raw['annotation']['identity']) or raw['origin']
        record['navigationLabel'] = (service(raw.get('destination')) or raw['label']) if record['navigationDisposition'] == 'navigation' else None
        self.records.append(record)
        if record['navigationDisposition'] != 'navigation':
            return
        before = {'origin': service(raw['annotation']['identity']) or raw['origin'],
                  'annotation': raw['annotation'], 'timestamp': raw['outcomeTimestamp']}
        self.base.learn(before, service(raw.get('destination')) or raw['label'], step=0,
                        weight=raw['weight'], outcome_id=raw['outcomeID'], outcome_timestamp=raw['outcomeTimestamp'])

    def specific(self, group, exact_origin):
        seen, counts = set(), Counter()
        for r in self.records:
            if (r['origin'] != exact_origin or r['navigationDisposition'] != 'navigation'
                    or service(r.get('destination')) != group or r['outcomeID'] in seen):
                continue
            seen.add(r['outcomeID'])
            counts[r['label']] += min(1, r['weight'])
        total = sum(counts.values())
        if total < 4:
            return None
        best = rank(counts, 1)[0]
        target = next(r.get('destination') for r in self.records if r['label'] == best)
        if not target or not urlsplit(target.get('locator', '')).path.startswith('/c/'):
            return None
        p, z2 = counts[best] / total, 1.96 ** 2
        lower = (p + z2 / (2 * total) - 1.96 * math.sqrt(p * (1 - p) / total + z2 / (4 * total ** 2))) / (1 + z2 / total)
        return best if lower > .5 else None

    def scores(self, current):
        annotation, exact_origin = annotate_target(current), origin(current)
        q = {'origin': service(annotation['identity']) or exact_origin, 'annotation': annotation, 'timestamp': 0}
        result = Counter()
        for label, score in self.base.predict(q, 0).items():
            selected = (self.specific(label, exact_origin) or label) if label == SERVICE else label
            if disposition(selected, annotation['identity']) != 'navigation':
                continue
            if selected == SERVICE and current.get('locator') == ENTRY:
                continue
            result[selected] += score
        return dict(result)

    def ranked(self, current, n=4):
        return rank(self.scores(current), n)


def learning_record(row, observation):
    target = row['record']['target']
    return {'origin': observation['origin'], 'annotation': observation['annotation'],
            'label': row['label'], 'outcomeID': row['outcome_id'],
            'outcomeTimestamp': row['outcome_timestamp'], 'weight': row['weight'],
            'source': 'research_seed', 'destination': annotate_target(target)['identity'],
            'observedStep': row['record']['observed_step']}
