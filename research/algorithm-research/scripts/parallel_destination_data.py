"""Causal parallel destination layers for one confirmed foreground arrival.

One arrival remains one behavior.  Its observed exact destination and its
structurally supported parents are simultaneous labels, not extra visits.
The same representation is used for the known current input and mature output.
No catalog or future outcome is allowed into the prediction input.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from typing import Iterable, Mapping

import hierarchical_fallback_policy as hierarchy_policy


@dataclass(frozen=True)
class Layer:
    key: str
    path: tuple[str, ...]
    kind: str
    evidence: str


@dataclass(frozen=True)
class ParallelArrival:
    outcome_id: str
    input_event_index: int
    mature_event_index: int
    current: tuple[Layer, ...]
    destination: tuple[Layer, ...]


FREQUENCY_WINDOWS_MS = (300_000, 3_600_000, 86_400_000)


@dataclass(frozen=True)
class FrequencyInput:
    """As-of counts of independent, mature arrivals at every destination layer.

    Windows are overlapping views of the same arrivals, never extra visits.
    The final slot is the complete observed history. All keys stay in memory.
    """

    cutoff_ms: float
    total: tuple[int, int, int, int]
    paths: tuple[Mapping[str, Mapping[tuple[str, ...], int]], ...]

    def count(self, key: str, scale: int) -> int:
        return sum(self.paths[scale].get(key, {}).values())

    def gain(self, key: str, scale: int,
             policy: hierarchy_policy.FallbackPolicy) -> float:
        return math.fsum(count * policy.path_utility(path)
                         for path, count in self.paths[scale].get(key, {}).items())


class FrequencyHistory:
    """Build query inputs only from outcomes that matured before the cutoff."""

    def __init__(self) -> None:
        self.records: list[tuple[float, tuple[Layer, ...]]] = []
        self.seen: set[str] = set()

    def observe(self, arrival: ParallelArrival, timestamp_ms: float) -> None:
        if not math.isfinite(timestamp_ms):
            raise ValueError("Arrival timestamp must be finite")
        if arrival.outcome_id in self.seen:
            raise ValueError("One independent arrival cannot be counted twice")
        if self.records and timestamp_ms < self.records[-1][0]:
            raise ValueError("Frequency arrivals must be observed in time order")
        self.seen.add(arrival.outcome_id)
        self.records.append((timestamp_ms, arrival.destination))

    def snapshot(self, cutoff_ms: float) -> FrequencyInput:
        if not math.isfinite(cutoff_ms):
            raise ValueError("Prediction cutoff must be finite")
        paths: list[dict[str, dict[tuple[str, ...], int]]] = [dict() for _ in range(4)]
        totals = [0, 0, 0, 0]
        for timestamp, layers in self.records:
            if timestamp >= cutoff_ms:
                raise ValueError("Current or future arrival entered frequency input")
            age = cutoff_ms - timestamp
            for scale in (*[i for i, window in enumerate(FREQUENCY_WINDOWS_MS)
                            if age <= window], 3):
                totals[scale] += 1
                for layer in layers:
                    by_path = paths[scale].setdefault(layer.key, {})
                    by_path[layer.path] = by_path.get(layer.path, 0) + 1
        return FrequencyInput(cutoff_ms, tuple(totals), tuple(paths))


def semantic_key(metadata: Mapping) -> str:
    """Stable in-memory identity across recorded and derived parent IDs.

    Raw locators are hashed in this key.  The original locator remains only in
    the caller's as-of metadata for execution and cannot enter an artifact.
    """
    if metadata.get("kind") == "window":
        bundle = metadata.get("bundleID")
        if isinstance(bundle, str) and bundle:
            return "window_class:" + bundle
    web = hierarchy_policy._web_location(metadata)
    if web is not None:
        origin, segments, query, fragment = web
        value = ["page", origin, segments, query, fragment]
    else:
        file_location = hierarchy_policy._file_location(metadata)
        if file_location is not None:
            value = [metadata.get("kind"), file_location]
        else:
            identifier = metadata.get("id")
            if not isinstance(identifier, str) or not identifier:
                raise ValueError("Layer has no valid identity")
            return identifier
    digest = hashlib.sha256(json.dumps(
        value, ensure_ascii=False, separators=(",", ":"),
    ).encode("utf-8")).hexdigest()
    return "layer:" + str(value[0]) + ":" + digest


def layers_for(hierarchy: hierarchy_policy.Hierarchy, target_id: str) -> tuple[Layer, ...]:
    """One exact layer and every independently justified ancestor in a DAG."""
    if target_id not in hierarchy.metadata:
        raise ValueError("Target is absent from its as-of hierarchy")
    selected: dict[str, Layer] = {}
    for identifier, path in hierarchy.paths(target_id).items():
        metadata = hierarchy.metadata.get(identifier)
        if not metadata:
            continue
        key = semantic_key(metadata)
        evidence = "observed_exact" if identifier == target_id else (
            "structural_parent" if metadata.get("_derived_parent") is True
            else "confirmed_relation"
        )
        layer = Layer(key, path, str(metadata.get("kind", "unknown")), evidence)
        previous = selected.get(key)
        if previous is None or (len(path), evidence) < (len(previous.path), previous.evidence):
            selected[key] = layer
    exact_key = semantic_key(hierarchy.metadata[target_id])
    if exact_key not in selected or selected[exact_key].path:
        raise ValueError("The exact destination layer was lost")
    return tuple(sorted(selected.values(), key=lambda item: (
        len(item.path), item.path, item.key,
    )))


def asof_hierarchy(catalog: Iterable[Mapping], current: Mapping | None = None
                   ) -> hierarchy_policy.Hierarchy:
    metadata = {str(item["id"]): item for item in catalog if item.get("id")}
    if current and current.get("id"):
        metadata[str(current["id"])] = current
    # An observed foreground child's bundle confirms its owner as a parallel
    # input/output layer, even when the app had no separate target.update.
    # This is a semantic training node; action eligibility remains the caller's
    # independent as-of catalog decision.
    for item in tuple(metadata.values()):
        bundle = item.get("bundleID")
        if isinstance(bundle, str) and bundle:
            owner = "app:" + bundle
            metadata.setdefault(owner, {"id": owner, "kind": "app", "bundleID": bundle})
    return hierarchy_policy.build_hierarchy(metadata.values(),
                                             executable=metadata.keys(),
                                             include_owner_apps=True)


def focused_hierarchy(catalog: Iterable[Mapping], target: Mapping
                      ) -> hierarchy_policy.Hierarchy:
    """Build only this observed object's proven lineage from an as-of catalog."""
    items = {str(item["id"]): item for item in catalog if item.get("id")}
    items[str(target["id"])] = target
    selected = {str(target["id"]): target}
    web = hierarchy_policy._web_location(target)
    file_location = hierarchy_policy._file_location(target)
    for identifier, item in items.items():
        candidate_web = hierarchy_policy._web_location(item)
        if web is not None and candidate_web is not None:
            origin, segments, _, _ = web
            other_origin, other_segments, query, fragment = candidate_web
            if (origin == other_origin and not query and not fragment
                    and segments[:len(other_segments)] == other_segments):
                selected[identifier] = item
        candidate_file = hierarchy_policy._file_location(item)
        if (file_location is not None and candidate_file is not None
                and item.get("kind") == "directory"):
            host, path = file_location
            other_host, other_path = candidate_file
            if host == other_host and (path == other_path or path.startswith(
                    other_path.rstrip("/") + "/")):
                selected[identifier] = item
    pointer = target
    seen = {str(target["id"])}
    while isinstance(pointer.get("parentID"), str):
        parent_id = pointer["parentID"]
        if parent_id in seen or parent_id not in items:
            break
        seen.add(parent_id)
        pointer = items[parent_id]
        selected[parent_id] = pointer
    bundle = target.get("bundleID")
    if isinstance(bundle, str) and "app:" + bundle in items:
        selected["app:" + bundle] = items["app:" + bundle]
    return asof_hierarchy(selected.values(), target)


def make_arrival(*, outcome_id: str, input_event_index: int,
                 mature_event_index: int, current: Mapping, destination: Mapping,
                 input_catalog: Iterable[Mapping], mature_catalog: Iterable[Mapping]
                 ) -> ParallelArrival:
    """Build input from the cutoff and output only after its real maturity."""
    if not isinstance(outcome_id, str) or not outcome_id:
        raise ValueError("One confirmed outcome ID is required")
    if not input_event_index < mature_event_index:
        raise ValueError("The output must mature after the input cutoff")
    if not current.get("id") or not destination.get("id"):
        raise ValueError("Both observed endpoints need identities")
    input_tree = focused_hierarchy(input_catalog, current)
    mature_tree = focused_hierarchy(mature_catalog, destination)
    return ParallelArrival(
        outcome_id, input_event_index, mature_event_index,
        layers_for(input_tree, str(current["id"])),
        layers_for(mature_tree, str(destination["id"])),
    )
