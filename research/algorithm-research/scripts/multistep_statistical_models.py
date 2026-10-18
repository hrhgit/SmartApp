"""Causal foreground-sequence statistical models for offline research.

The model receives only an as-of sequence of confirmed foreground destination
keys and mature transition labels supplied by the replay owner.  It never
enumerates an inventory, accepts a candidate list, or reads evaluation truth
from a prediction query.  The highest-order distribution backs off through
shorter suffixes and then to the mature global vocabulary.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import math


HOUR_MS = 3_600_000.0
TIE_EPSILON = 1e-12


def _normalize(values: dict[str, float]) -> dict[str, float]:
    total = math.fsum(value for value in values.values() if value > 0)
    if not total:
        return {}
    return {key: max(0.0, value) / total for key, value in values.items() if value > 0}


def _mix(left: dict[str, float], right: dict[str, float], right_weight: float) -> dict[str, float]:
    weight = min(1.0, max(0.0, float(right_weight)))
    keys = left.keys() | right.keys()
    return {key: (1.0 - weight) * left.get(key, 0.0) + weight * right.get(key, 0.0)
            for key in keys}


@dataclass(frozen=True)
class Transition:
    sequence: tuple[str, ...]
    label: str
    input_timestamp: float
    outcome_timestamp: float
    outcome_id: str
    serial: int


def configurations() -> list[dict]:
    """A small predeclared grid; the held-out test set never selects from it."""
    configs = [
        {
            "id": "sequence/markov2-count",
            "family": "markov",
            "order": 2,
            "alpha": 0.0,
            "backoff_tau": 2.0,
            "half_life_hours": 0.0,
            "recency_mix": 0.0,
            "min_context_support": 0.0,
        },
        {
            "id": "sequence/markov2-smoothed",
            "family": "markov",
            "order": 2,
            "alpha": 1.0,
            "backoff_tau": 4.0,
            "half_life_hours": 0.0,
            "recency_mix": 0.0,
            "min_context_support": 0.0,
        },
        {
            "id": "sequence/markov2-recent",
            "family": "markov",
            "order": 2,
            "alpha": 1.0,
            "backoff_tau": 8.0,
            "half_life_hours": 24.0,
            "recency_mix": 0.25,
            "min_context_support": 0.0,
        },
        {
            "id": "sequence/markov3-smoothed",
            "family": "markov",
            "order": 3,
            "alpha": 1.0,
            "backoff_tau": 4.0,
            "half_life_hours": 0.0,
            "recency_mix": 0.0,
            "min_context_support": 0.0,
        },
        {
            "id": "sequence/markov3-recent",
            "family": "markov",
            "order": 3,
            "alpha": 2.0,
            "backoff_tau": 8.0,
            "half_life_hours": 24.0,
            "recency_mix": 0.25,
            "min_context_support": 0.0,
        },
        {
            "id": "sequence/variable-suffix",
            "family": "variable_suffix",
            "order": 3,
            "alpha": 1.0,
            "backoff_tau": 6.0,
            "half_life_hours": 24.0,
            "recency_mix": 0.15,
            "min_context_support": 2.0,
        },
    ]
    if len({config["id"] for config in configs}) != len(configs):
        raise AssertionError("Duplicate sequence configuration")
    return configs


class SequenceStatisticalModel:
    """Interpolated n-gram/Markov statistics with causal suffix backoff."""

    def __init__(self, config: dict):
        self.config = dict(config)
        self.order = int(self.config["order"])
        if self.order < 1:
            raise ValueError("Sequence order must be positive")
        self.records: dict[int, defaultdict[tuple[str, ...], list[Transition]]] = {
            length: defaultdict(list) for length in range(1, self.order + 1)
        }
        self.seen_by_context: dict[int, defaultdict[tuple[str, ...], set[str]]] = {
            length: defaultdict(set) for length in range(1, self.order + 1)
        }
        self.global_records: list[Transition] = []
        self.seen_global: set[str] = set()
        self.seen_full: set[tuple[tuple[str, ...], str]] = set()
        self.vocabulary: set[str] = set()
        self.latest_outcome = -math.inf
        self._serial = 0

    def learn(self, record: dict) -> bool:
        """Add one mature transition, deduplicating repeated observations."""
        sequence = tuple(record.get("sequence") or ())
        origin = record.get("origin")
        label = record.get("label")
        outcome_id = record.get("outcome_id")
        if not sequence or not all(isinstance(value, str) and value for value in sequence):
            raise ValueError("A nonempty foreground sequence is required")
        if origin is not None and sequence[-1] != origin:
            raise ValueError("Sequence must end at the current origin")
        if not isinstance(label, str) or not label:
            raise ValueError("Transition label is required")
        if not isinstance(outcome_id, str) or not outcome_id:
            raise ValueError("Independent outcome ID is required")
        input_timestamp = float(record["input_timestamp"])
        outcome_timestamp = float(record["outcome_timestamp"])
        if not math.isfinite(input_timestamp) or not math.isfinite(outcome_timestamp):
            raise ValueError("Transition timestamps must be finite")
        if outcome_timestamp < input_timestamp:
            raise ValueError("Outcome precedes its input")

        full_identity = (sequence, outcome_id)
        if full_identity in self.seen_full:
            return False
        self.seen_full.add(full_identity)
        transition = Transition(sequence, label, input_timestamp, outcome_timestamp,
                                outcome_id, self._serial)
        self._serial += 1
        self.vocabulary.add(label)
        self.latest_outcome = max(self.latest_outcome, outcome_timestamp)
        accepted = False
        for length in range(1, min(self.order, len(sequence)) + 1):
            suffix = sequence[-length:]
            seen = self.seen_by_context[length][suffix]
            if outcome_id in seen:
                continue
            seen.add(outcome_id)
            self.records[length][suffix].append(transition)
            accepted = True
        if outcome_id not in self.seen_global:
            self.seen_global.add(outcome_id)
            self.global_records.append(transition)
        return accepted

    def _weight(self, timestamp: float, now: float) -> float:
        half_life = float(self.config.get("half_life_hours", 0.0))
        if half_life <= 0:
            return 1.0
        age = max(0.0, now - timestamp)
        return 2.0 ** (-age / (half_life * HOUR_MS))

    def _weighted_counts(self, records: list[Transition], now: float) -> dict[str, float]:
        counts: defaultdict[str, float] = defaultdict(float)
        for record in records:
            if record.outcome_timestamp >= now:
                raise ValueError("Future transition reached a prediction")
            counts[record.label] += self._weight(record.outcome_timestamp, now)
        return dict(counts)

    def _global_distribution(self, now: float) -> dict[str, float]:
        return _normalize(self._weighted_counts(self.global_records, now))

    def _suffix_distribution(self, suffix: tuple[str, ...], lower: dict[str, float], now: float) -> dict[str, float]:
        length = len(suffix)
        records = self.records[length].get(suffix, [])
        if not records:
            return lower
        counts = self._weighted_counts(records, now)
        support = math.fsum(counts.values())
        alpha = max(0.0, float(self.config.get("alpha", 0.0)))
        if lower and alpha:
            denominator = support + alpha
            empirical = {label: (counts.get(label, 0.0) + alpha * lower.get(label, 0.0)) / denominator
                         for label in self.vocabulary}
        else:
            empirical = _normalize(counts)
        if not empirical:
            return lower
        minimum = float(self.config.get("min_context_support", 0.0))
        if self.config.get("family") == "variable_suffix" and support < minimum:
            return lower
        tau = max(0.0, float(self.config.get("backoff_tau", 0.0)))
        interpolation = 1.0 if tau == 0 else support / (support + tau)
        return _mix(lower, empirical, interpolation)

    def _recency_distribution(self, now: float) -> dict[str, float]:
        latest: dict[str, float] = {}
        for record in self.global_records:
            if record.outcome_timestamp >= now:
                raise ValueError("Future transition reached recency features")
            latest[record.label] = max(latest.get(record.label, -math.inf), record.outcome_timestamp)
        ordered = sorted(latest, key=lambda label: (-latest[label], label))
        return _normalize({label: 1.0 / (index + 1) for index, label in enumerate(ordered)})

    def scores(self, query: dict) -> dict[str, float]:
        """Return scores for the mature vocabulary; no query candidate list is read."""
        now = float(query["timestamp"])
        if not math.isfinite(now):
            raise ValueError("Prediction timestamp must be finite")
        if self.latest_outcome >= now:
            raise ValueError("Training outcome is not strictly before prediction")
        sequence = tuple(query.get("sequence") or ())
        if not sequence or not all(isinstance(value, str) and value for value in sequence):
            return {}
        if query.get("origin") is not None and sequence[-1] != query["origin"]:
            raise ValueError("Prediction sequence does not end at current origin")
        distribution = self._global_distribution(now)
        max_length = min(self.order, len(sequence))
        for length in range(1, max_length + 1):
            distribution = self._suffix_distribution(sequence[-length:], distribution, now)
        recency_mix = float(self.config.get("recency_mix", 0.0))
        if recency_mix:
            distribution = _mix(distribution, self._recency_distribution(now), recency_mix)
        positive = {label: score for label, score in distribution.items()
                    if label in self.vocabulary and score > 0 and math.isfinite(score)}
        return positive

    def ranked(self, query: dict) -> list[str]:
        scores = self.scores(query)
        return sorted(scores, key=lambda label: (-scores[label], label))

    def state_summary(self) -> dict:
        return {
            "order": self.order,
            "vocabulary": len(self.vocabulary),
            "independent_outcomes": len(self.seen_global),
            "records_by_order": {
                str(length): sum(len(rows) for rows in self.records[length].values())
                for length in self.records
            },
            "contexts_by_order": {str(length): len(self.records[length]) for length in self.records},
        }


def build(config: dict) -> SequenceStatisticalModel:
    return SequenceStatisticalModel(config)
