"""Incremental, weighted destination statistics with soft semantic sharing.

One record is one observed origin/outcome/head pair. The replay owner enforces
strict label maturity and deduplication. No inventory or future label vocabulary
is accepted. Similarity enriches the origin; destination labels remain exact.
"""
from collections import Counter
import json
import math

from semantic_annotations import compare_annotations


def normalize(values):
    total = sum(max(0., v) for v in values.values())
    return {k: max(0., v) / total for k, v in values.items()} if total else {}


def rank(scores, n=4):
    return sorted(scores, key=lambda k: (-scores[k], k))[:n]


def blend(distributions, weights):
    result = Counter()
    for distribution, weight in zip(distributions, weights):
        for key, value in normalize(distribution).items():
            result[key] += weight * value
    return normalize(result)


class SimilarityCache:
    def __init__(self):
        self.values = {}
        self.signatures = {}

    def score(self, left, right):
        # A title can change while origin identity is stable; cache annotations,
        # not just origin IDs, so later metadata never rewrites an earlier input.
        for annotation in (left, right):
            if id(annotation) not in self.signatures:
                self.signatures[id(annotation)] = (annotation, json.dumps(annotation, sort_keys=True, ensure_ascii=False))
        a = self.signatures[id(left)][1]
        b = self.signatures[id(right)][1]
        key = (a, b) if a <= b else (b, a)
        if key not in self.values:
            self.values[key] = compare_annotations(left, right)['score'] ** 2
        return self.values[key]


class SemanticStatistics:
    def __init__(self, semantic_strength=0., global_strength=1., half_life_hours=0.,
                 recency_mix=0., similarity=None):
        self.semantic_strength = semantic_strength
        self.global_strength = global_strength
        self.half_life_ms = half_life_hours * 3600000
        self.recency_mix = recency_mix
        self.similarity = similarity or SimilarityCache()
        self.records = {0: [], 1: [], 2: [], 3: []}
        self.seen = set()

    def learn(self, q, label, weight=1., step=1, outcome_id=None, outcome_timestamp=None):
        if step not in self.records or not math.isfinite(weight) or weight < 0:
            raise ValueError('Invalid observed step or training weight')
        key = (step, q['origin'], label, outcome_id)
        if outcome_id is not None and key in self.seen:
            return
        self.seen.add(key)
        self.records[step].append((q, label, weight, q['timestamp'] if outcome_timestamp is None else outcome_timestamp, outcome_id))

    def components(self, q, step=1):
        exact, semantic, global_counts, last = Counter(), Counter(), Counter(), {}
        global_seen, semantic_votes = set(), {}
        semantic_confidence = 0.
        similarities = {}
        for before, label, weight, outcome_time, outcome_id in self.records[step]:
            elapsed = max(0., q['timestamp'] - outcome_time)
            decayed = weight * (2. ** (-elapsed / self.half_life_ms)
                                if self.half_life_ms else 1.)
            if before['origin'] == q['origin']:
                exact[label] += decayed
            elif self.semantic_strength:
                identity = id(before['annotation'])
                if identity not in similarities:
                    similarities[identity] = self.similarity.score(q['annotation'], before['annotation'])
                strength = similarities[identity]
                semantic_confidence = max(semantic_confidence, strength)
                # The same future arrival can be supervised from more than one
                # origin. It still contributes at most its strongest vote here.
                key = (label, outcome_id if outcome_id is not None else id(before))
                semantic_votes[key] = max(semantic_votes.get(key, 0.), decayed * strength)
            # Overlapping supervision windows cannot multiply global popularity.
            if outcome_id is None or outcome_id not in global_seen:
                global_counts[label] += decayed
                global_seen.add(outcome_id)
            last[label] = max(last.get(label, -math.inf), outcome_time)
        # Recency is rank-based, so a single recent event cannot dominate by
        # timestamp magnitude. Every learned destination remains rankable.
        recency = {label: 1. / (i + 1) for i, label in enumerate(
            sorted(last, key=lambda k: (-last[k], k)))}
        for (label, _), weight in semantic_votes.items():
            semantic[label] += weight
        # A single weak site match must not become a unit-strength prior merely
        # because it was the only match. Preserve the strongest relation weight.
        semantic_prior = {key: value * semantic_confidence for key,value in normalize(semantic).items()}
        return exact, semantic_prior, normalize(global_counts), normalize(recency)

    def predict(self, q, step=1):
        exact, semantic, global_prior, recency = self.components(q, step)
        scores = Counter(exact)
        for source, strength in ((semantic, self.semantic_strength), (global_prior, self.global_strength)):
            for label, value in source.items():
                scores[label] += strength * value
        # Include the complete learned vocabulary even when smoothing is zero.
        for _, label, _, _, _ in self.records[step]:
            scores.setdefault(label, 0.)
        return blend([scores, recency], [1. - self.recency_mix, self.recency_mix])


def scores_from_components(parts, semantic_strength=0., global_strength=1., recency_mix=0.):
    exact, semantic, global_prior, recency = parts
    scores = Counter(exact)
    for source, strength in ((semantic, semantic_strength), (global_prior, global_strength)):
        for label, value in source.items():
            scores[label] += strength * value
    for label in global_prior:
        scores.setdefault(label, 0.)
    return blend([scores, recency], [1. - recency_mix, recency_mix])
