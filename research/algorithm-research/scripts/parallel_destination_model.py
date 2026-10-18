"""Incremental model over parallel input and output destination layers.

Each confirmed outcome contributes once to each conditional view of its path.
The views are smoothed into one estimate at prediction time; they are never
added as independent outcome probabilities or extra behavior examples.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
import math
from typing import Mapping

import hierarchical_fallback_policy as hierarchy_policy
from parallel_destination_data import FrequencyInput, Layer, ParallelArrival, semantic_key


@dataclass(frozen=True)
class LayerPrediction:
    scores: Mapping[str, float]
    ranked: tuple[str, ...]
    input_layers: tuple[str, ...]


def _origin_order(layer: Layer) -> tuple[int, int, str]:
    # An owning application is the coarsest context.  Then use content parents
    # from broad to narrow, and finally the observed exact current destination.
    if "owner_app" in layer.path:
        return 0, 0, layer.key
    if layer.path:
        return 1, -len(layer.path), layer.key
    return 2, 0, layer.key


class ParallelDestinationModel:
    def __init__(self, *, shrinkage: float = 5.0) -> None:
        if not math.isfinite(shrinkage) or shrinkage <= 0:
            raise ValueError("Shrinkage must be finite and positive")
        self.shrinkage = float(shrinkage)
        self.seen: dict[str, tuple] = {}
        self.global_n = 0
        self.origin_n: Counter[str] = Counter()
        self.global_paths: Counter[tuple[str, tuple[str, ...]]] = Counter()
        self.origin_paths: dict[str, Counter[tuple[str, tuple[str, ...]]]] = defaultdict(Counter)
        self.global_by_key: dict[str, Counter[tuple[str, ...]]] = defaultdict(Counter)
        self.origin_by_key: dict[str, dict[str, Counter[tuple[str, ...]]]] = defaultdict(
            lambda: defaultdict(Counter)
        )

    def learn(self, arrival: ParallelArrival) -> bool:
        identity = (
            arrival.input_event_index, arrival.mature_event_index,
            tuple(layer.key for layer in arrival.current),
            tuple((layer.key, layer.path) for layer in arrival.destination),
        )
        previous = self.seen.get(arrival.outcome_id)
        if previous is not None:
            if previous != identity:
                raise ValueError("One arrival ID cannot produce conflicting layers")
            return False
        exact = [layer for layer in arrival.destination if not layer.path]
        if len(exact) != 1:
            raise ValueError("One real outcome must have one exact layer")
        self.seen[arrival.outcome_id] = identity
        self.global_n += 1
        unique_outputs = {(layer.key, layer.path) for layer in arrival.destination}
        for item in unique_outputs:
            self.global_paths[item] += 1
            self.global_by_key[item[0]][item[1]] += 1
        for origin in {layer.key for layer in arrival.current}:
            self.origin_n[origin] += 1
            for item in unique_outputs:
                self.origin_paths[origin][item] += 1
                self.origin_by_key[origin][item[0]][item[1]] += 1
        return True

    @staticmethod
    def _gain(paths: Mapping[tuple[str, ...], int],
              policy: hierarchy_policy.FallbackPolicy) -> float:
        return math.fsum(
            count * policy.path_utility(path)
            for path, count in paths.items()
        )

    def predict(self, current: tuple[Layer, ...],
                hierarchy: hierarchy_policy.Hierarchy,
                policy: hierarchy_policy.FallbackPolicy, *,
                current_action: str | None = None,
                current_app: str | None = None,
                frequency: FrequencyInput | None = None,
                recent_window: int | None = None) -> LayerPrediction:
        if recent_window not in (None, 0, 1, 2):
            raise ValueError("Unknown recent frequency window")
        if recent_window is not None and frequency is None:
            raise ValueError("Recent frequency ranking requires an as-of input")
        blocked = hierarchy_policy.blocked_current_actions(
            hierarchy, current_action=current_action, current_app=current_app,
        )
        origins = tuple(dict.fromkeys(
            layer.key for layer in sorted(current, key=_origin_order)
        ))
        scores: dict[str, float] = {}
        for action in hierarchy.executable - blocked:
            metadata = hierarchy.metadata.get(action)
            if metadata is None:
                continue
            key = semantic_key(metadata)
            value = self._gain(self.global_by_key.get(key, {}), policy) / self.global_n if self.global_n else 0.0
            if frequency is not None and recent_window is not None:
                n = frequency.total[recent_window]
                if n:
                    value = (frequency.gain(key, recent_window, policy)
                             + self.shrinkage * value) / (n + self.shrinkage)
            for origin in origins:
                n = self.origin_n[origin]
                if n:
                    value = (self._gain(self.origin_by_key[origin].get(key, {}), policy)
                             + self.shrinkage * value) / (n + self.shrinkage)
            if value > 0:
                scores[action] = value
        ranked = tuple(sorted(scores, key=lambda action: (
            -scores[action], -hierarchy.specificity(action), action,
        )))
        return LayerPrediction(scores, ranked, origins)
