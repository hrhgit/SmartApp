"""Foreground-only, as-of sequence inputs for private offline experiments.

One confirmed arrival contributes one token and one frequency record.  Its
verified parents are parallel attributes of that token, never extra visits.
Bridges are read only when their *after* observation has arrived; a mask then
describes the missing interval without asserting physical continuity or dwell.
"""

from __future__ import annotations

import bisect
import collections
import math
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Iterable
from zoneinfo import ZoneInfo

import numpy as np

import parallel_destination_data as parallel
import run_offline_batch_rankers as runner


NUMERIC_WIDTH = 23
CANDIDATE_WIDTH = 8
MAX_LAYERS = 8
WINDOWS_MS = (300_000, 3_600_000, 86_400_000)
NON_ARRIVAL_REASONS = {"resolution_update", "browser_history", "browser_resync"}
MASK_KEYS = {"legacy_short_gap": "observation:legacy_short_gap",
             "excluded_scope": "observation:excluded_scope"}


@dataclass(frozen=True)
class Step:
    key: str
    event_index: int
    timestamp_ms: int
    monotonic_ms: int | None
    layers: tuple[str, ...]
    source_quality: tuple[float, float, float, float, float, float, float]
    marker_kind: str = ""
    marker_duration_ms: int | None = None
    marker_same_exact: bool = False
    marker_active_elapsed_ms: int | None = None
    marker_observed_pre_gap_ms: int | None = None
    marker_begin_index: int | None = None
    marker_end_index: int | None = None
    source_id: str | None = None
    source_status: str | None = None

    @property
    def is_arrival(self) -> bool:
        return not self.marker_kind


@dataclass(frozen=True)
class RichPrefix:
    steps: tuple[Step, ...]
    current_key: str
    cutoff_event_index: int
    cutoff_timestamp_ms: int

    @property
    def keys(self) -> tuple[str, ...]:
        return tuple(step.key for step in self.steps)

    @property
    def arrival_count(self) -> int:
        return sum(step.is_arrival for step in self.steps)


@dataclass(frozen=True)
class ArrivalRecord:
    event_index: int
    timestamp_ms: int
    exact_layer: str
    layers: tuple[str, ...]


@dataclass(frozen=True)
class RichIndex:
    by_event_index: dict[int, RichPrefix]
    records: tuple[ArrivalRecord, ...]
    audit: dict[str, int]
    candidate_cache: dict[tuple[int, tuple[str, ...], int, int],
                          tuple[np.ndarray, np.ndarray]] = field(
        default_factory=dict, compare=False, repr=False)
    frequency_cache: dict[tuple[int, int], tuple[dict[str, tuple[int, int, int, int]],
                                                 dict[str, tuple[int, int, int, int]]]] = field(
        default_factory=dict, compare=False, repr=False)
    layer_cache: dict[tuple[Any, ...], tuple[str, ...]] = field(
        default_factory=dict, compare=False, repr=False)


def _timestamp(event: dict[str, Any]) -> int:
    value = event.get("timestampMs")
    return int(value) if isinstance(value, (int, float)) and math.isfinite(value) else 0


def _monotonic(event: dict[str, Any]) -> int | None:
    value = event.get("monotonicMs")
    return int(value) if isinstance(value, (int, float)) and math.isfinite(value) else None


def _quality(event: dict[str, Any]) -> tuple[float, float, float, float, float, float, float]:
    coverage = (event.get("data") or {}).get("inputCoverage")
    if not isinstance(coverage, dict) or coverage.get("status") != "foreground_only":
        return (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0)
    sources = coverage.get("sources")
    if not isinstance(sources, list) or not sources:
        return (0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0)
    counts = collections.Counter(item.get("status") for item in sources if isinstance(item, dict))
    total = max(1, sum(counts.values()))
    known = {"ready", "initializing", "unavailable", "failed", "stale"}
    unknown = sum(count for status, count in counts.items() if status not in known)
    return (counts["ready"] / total, counts["initializing"] / total,
            counts["unavailable"] / total, 1.0,
            counts["failed"] / total, counts["stale"] / total, unknown / total)


def _relevant_source(event: dict[str, Any]) -> tuple[str | None, str | None]:
    source_id = event.get("source")
    if not isinstance(source_id, str) or not source_id:
        return None, None
    coverage = (event.get("data") or {}).get("inputCoverage")
    sources = coverage.get("sources") if isinstance(coverage, dict) else None
    if not isinstance(sources, list):
        return source_id, None
    matches = [entry.get("status") for entry in sources
               if isinstance(entry, dict) and entry.get("source") == source_id]
    return source_id, str(matches[-1]) if len(matches) == 1 else None


def _layers(meta: dict[str, Any], history: runner.TargetHistory, index: int) -> tuple[str, ...]:
    """Resolve the current observed target's ancestry using only old metadata."""
    catalog = [meta]
    pointer = meta
    seen = {str(meta.get("id"))}
    while isinstance(pointer.get("parentID"), str):
        parent_id = pointer["parentID"]
        if parent_id in seen:
            break
        parent = history.metadata_at(parent_id, index)
        if not isinstance(parent, dict):
            break
        catalog.append(parent)
        seen.add(parent_id)
        pointer = parent
    try:
        tree = parallel.focused_hierarchy(catalog, meta)
        layers = parallel.layers_for(tree, str(meta["id"]))
        return tuple(layer.key for layer in layers)
    except (ValueError, KeyError, TypeError):
        return (str(meta["id"]),)


def _cached_layers(meta: dict[str, Any], history: runner.TargetHistory,
                   event_index: int, index: RichIndex) -> tuple[str, ...]:
    """Cache immutable as-of metadata versions, including explicit parents."""
    original = history.metadata_at(meta.get("id"), event_index)
    pointer = original if isinstance(original, dict) else meta
    versions: list[int] = [id(original) if original is not None else 0]
    seen: set[str] = set()
    while isinstance(pointer.get("parentID"), str):
        parent_id = pointer["parentID"]
        if parent_id in seen:
            break
        seen.add(parent_id)
        parent = history.metadata_at(parent_id, event_index)
        versions.append(id(parent) if parent is not None else 0)
        if parent is None:
            break
        pointer = parent
    key = (meta.get("id"), meta.get("kind"), meta.get("locator"),
           meta.get("bundleID"), meta.get("parentID"), tuple(versions))
    if key not in index.layer_cache:
        index.layer_cache[key] = _layers(meta, history, event_index)
    return index.layer_cache[key]


def build_rich_index(events: list[dict[str, Any]], history: runner.TargetHistory,
                     sequence_length: int, bridges: Iterable[Any] = ()) -> RichIndex:
    """Rebuild a bounded prefix only after the bridge's second endpoint exists."""
    if sequence_length < 2:
        raise ValueError("Sequence length must retain at least two arrivals")
    bridges = tuple(bridges)
    bridge_after = {int(bridge.after_index): bridge for bridge in bridges}
    if len(bridge_after) != len(bridges):
        raise ValueError("More than one bridge ends at the same event")
    by_event: dict[int, RichPrefix] = {}
    records: list[ArrivalRecord] = []
    audit: collections.Counter[str] = collections.Counter()
    active: tuple[Step, ...] = ()
    current_session: str | None = None

    for index, event in enumerate(events):
        session = event.get("sessionID")
        if not isinstance(session, str) or not session:
            active = ()
            current_session = None
        elif current_session is None:
            current_session = session
        elif session != current_session:
            active = ()
            current_session = session
            audit["session_boundary"] += 1
        kind = event.get("kind")
        bridge = bridge_after.get(index)
        if bridge is not None:
            before = by_event.get(int(bridge.before_index))
            if (before is None or bridge.kind not in MASK_KEYS
                    or events[int(bridge.before_index)].get("sessionID") != session):
                raise ValueError("Bridge has no earlier confirmed same-session endpoint")
            marker = Step(MASK_KEYS[bridge.kind], index, _timestamp(event), _monotonic(event),
                          (), _quality(event), bridge.kind,
                          int(bridge.duration_ms) if bridge.duration_ms is not None else None,
                          bool(bridge.same_exact_target),
                          int(bridge.active_elapsed_ms) if getattr(bridge, "active_elapsed_ms", None) is not None
                          else None,
                          int(bridge.observed_pre_gap_ms)
                          if getattr(bridge, "observed_pre_gap_ms", None) is not None else None,
                          int(bridge.begin_index), int(bridge.end_index),
                          *_relevant_source(event))
            active = (*before.steps, marker)[-sequence_length:]
            audit["restored_" + bridge.kind] += 1
        elif kind in {"capture.gap", "session.start", "session.end", "observation.scope"}:
            active = ()
            audit["hard_boundary_" + str(kind).replace(".", "_")] += 1
            if kind == "session.end":
                current_session = None
            continue

        if kind != "context.change":
            continue
        data = event.get("data")
        if not isinstance(data, dict) or not runner.FOREGROUND_COVERAGE_VERSION == (
                data.get("inputCoverage") or {}).get("version") or not _quality(event)[3]:
            audit["context_without_foreground_coverage"] += 1
            continue
        reason = data.get("reason")
        target_id = data.get("targetID")
        meta = history.metadata_at(target_id, index)
        action = runner.action_from_meta(meta, catalog_only=False)
        if action is None or action.kind not in runner.ACTIONABLE_KINDS:
            audit["context_without_actionable_target"] += 1
            continue
        previous = next((step for step in reversed(active) if step.is_arrival), None)
        same_raw = (bridge is not None and bool(bridge.same_exact_target))
        if same_raw and (previous is None or previous.key != action.key):
            raise ValueError("Same-target bridge disagrees with its previous action")
        if reason in NON_ARRIVAL_REASONS and bridge is None:
            audit["non_arrival_context"] += 1
            continue
        if not same_raw and (previous is None or previous.key != action.key or bridge is not None):
            assert meta is not None
            layer_keys = _layers(meta, history, index)
            step = Step(action.key, index, _timestamp(event), _monotonic(event),
                        layer_keys, _quality(event), source_id=_relevant_source(event)[0],
                        source_status=_relevant_source(event)[1])
            active = (*active, step)[-sequence_length:]
            records.append(ArrivalRecord(index, step.timestamp_ms, layer_keys[0], layer_keys))
            audit["independent_arrivals"] += 1
        elif same_raw:
            audit["same_exact_bridge_without_new_arrival"] += 1
        else:
            audit["consecutive_duplicate_context"] += 1
        if active:
            by_event[index] = RichPrefix(active, action.key, index, _timestamp(event))
    return RichIndex(by_event, tuple(records), dict(sorted(audit.items())))


def _quality_usable_for_dwell(step: Step) -> bool:
    # Hundreds of unrelated sources can be initializing.  Only the source
    # that produced this foreground observation proves its own readiness.
    return bool(step.source_quality[3] and step.source_id and step.source_status == "ready")


def _trusted_interval(events: list[dict[str, Any]], start: int, end: int,
                      source_id: str | None) -> bool:
    """Only uninterrupted observed time can be called a dwell portion."""
    if start < 0 or end <= start or end >= len(events) or not source_id:
        return False
    for index in range(start + 1, end):
        event = events[index]
        kind = event.get("kind")
        if kind in {"idle.change", "capture.gap", "observation.scope",
                    "session.start", "session.end"}:
            return False
        if kind == "source.status" and event.get("source") == source_id:
            status = (event.get("data") or {}).get("observationStatus")
            if status != "ready":
                return False
        if kind == "context.change" and event.get("source") == source_id:
            if _relevant_source(event)[1] != "ready":
                return False
    return True


def sequence_numeric(prefix: RichPrefix, events: list[dict[str, Any]],
                     timezone: str = "America/Los_Angeles") -> np.ndarray:
    """Encode only facts known by this prefix's cutoff; missing dwell is masked."""
    steps = prefix.steps
    values = np.zeros((len(steps), NUMERIC_WIDTH), dtype=np.float32)
    zone = ZoneInfo(timezone)
    for i, step in enumerate(steps):
        local = datetime.fromtimestamp(step.timestamp_ms / 1000, tz=zone)
        hour = local.hour + local.minute / 60.0
        weekday = local.weekday()
        values[i, :4] = (math.sin(2 * math.pi * hour / 24), math.cos(2 * math.pi * hour / 24),
                         math.sin(2 * math.pi * weekday / 7), math.cos(2 * math.pi * weekday / 7))
        values[i, 8:12] = step.source_quality[:4]
        values[i, 20:23] = step.source_quality[4:]
        values[i, 18] = math.log1p(len(step.layers))
        if step.marker_kind:
            values[i, 12] = float(step.marker_kind == "legacy_short_gap")
            values[i, 13] = float(step.marker_kind == "excluded_scope")
            if step.marker_duration_ms is not None and step.marker_duration_ms >= 0:
                values[i, 14] = math.log1p(step.marker_duration_ms / 1000)
                values[i, 15] = 1.0
            if step.marker_same_exact:
                previous_index = next((j for j in range(i - 1, -1, -1)
                                       if steps[j].is_arrival), None)
                if previous_index is not None:
                    previous = steps[previous_index]
                    pre_safe = (step.marker_begin_index is not None
                                and _trusted_interval(events, previous.event_index,
                                                      step.marker_begin_index,
                                                      previous.source_id)
                                and _quality_usable_for_dwell(previous))
                    post_safe = (step.marker_end_index is not None
                                 and _trusted_interval(events, step.marker_end_index,
                                                       step.event_index, step.source_id)
                                 and _quality_usable_for_dwell(step))
                    if (pre_safe and post_safe and step.marker_active_elapsed_ms is not None
                            and step.marker_active_elapsed_ms >= 0):
                        values[previous_index, 16] = math.log1p(step.marker_active_elapsed_ms / 1000)
                        values[previous_index, 17] = 1.0  # inferred same-target active time, not confirmed dwell
                    if (pre_safe and step.marker_observed_pre_gap_ms is not None
                            and step.marker_observed_pre_gap_ms >= 0):
                        values[previous_index, 6] = math.log1p(step.marker_observed_pre_gap_ms / 1000)
                        values[previous_index, 7] = 1.0  # observed portion only; masked interval is excluded
            continue
        previous = next((old for old in reversed(steps[:i]) if old.is_arrival), None)
        if previous is not None and i > 0 and steps[i - 1].is_arrival:
            elapsed = step.timestamp_ms - previous.timestamp_ms
            if elapsed >= 0:
                values[i, 4] = math.log1p(elapsed / 1000)
                values[i, 5] = 1.0
        # A preceding arrival's dwell is confirmed only when this observed
        # arrival closes it without a gap, exclusion, idle marker, or bad clock.
        if previous is not None and i > 0 and steps[i - 1].is_arrival:
            if (previous.monotonic_ms is not None and step.monotonic_ms is not None
                    and step.monotonic_ms >= previous.monotonic_ms
                    and _quality_usable_for_dwell(previous)
                    and _quality_usable_for_dwell(step)
                    and _trusted_interval(events, previous.event_index,
                                          step.event_index, previous.source_id)):
                duration = step.monotonic_ms - previous.monotonic_ms
                values[i - 1, 6] = math.log1p(duration / 1000)
                values[i - 1, 7] = 1.0
        elif previous is not None and i > 0 and steps[i - 1].marker_same_exact:
            marker = steps[i - 1]
            if (marker.monotonic_ms is not None and step.monotonic_ms is not None
                    and step.monotonic_ms >= marker.monotonic_ms
                    and _quality_usable_for_dwell(marker)
                    and _quality_usable_for_dwell(step)
                    and _trusted_interval(events, marker.event_index,
                                          step.event_index, marker.source_id)):
                post_ms = step.monotonic_ms - marker.monotonic_ms
                pre_ms = marker.marker_observed_pre_gap_ms
                previous_index = next(j for j in range(i - 1, -1, -1) if steps[j].is_arrival)
                pre_safe = (marker.marker_begin_index is not None
                            and _quality_usable_for_dwell(previous)
                            and _trusted_interval(events, previous.event_index,
                                                  marker.marker_begin_index,
                                                  previous.source_id))
                resume_safe = (marker.marker_end_index is not None
                               and _trusted_interval(events, marker.marker_end_index,
                                                     marker.event_index, marker.source_id))
                # Keep separately observed portions; never reclassify an idle
                # or failed pre-gap span as a complete dwell after the fact.
                observed_ms = post_ms + (pre_ms if pre_safe and pre_ms is not None and pre_ms >= 0 else 0)
                values[previous_index, 6] = math.log1p(observed_ms / 1000)
                values[previous_index, 7] = 1.0
                if (pre_safe and resume_safe and marker.marker_active_elapsed_ms is not None
                        and marker.marker_active_elapsed_ms >= 0):
                    values[previous_index, 16] = math.log1p((marker.marker_active_elapsed_ms + post_ms) / 1000)
                    values[previous_index, 17] = 1.0
        values[i, 19] = 1.0  # confirmed foreground-arrival token
    # The current target's future dwell is never filled from a later event.
    return values


def path_frequency(index: RichIndex, cutoff_event_index: int, cutoff_ms: int
                   ) -> tuple[dict[str, tuple[int, int, int, int]],
                              dict[str, tuple[int, int, int, int]]]:
    """Count earlier independent arrivals once, including each proven parent."""
    cache_key = (cutoff_event_index, cutoff_ms)
    cached = index.frequency_cache.get(cache_key)
    if cached is not None:
        return cached
    path: dict[str, list[int]] = collections.defaultdict(lambda: [0, 0, 0, 0])
    direct: dict[str, list[int]] = collections.defaultdict(lambda: [0, 0, 0, 0])
    for record in index.records:
        if record.event_index >= cutoff_event_index:
            break
        if record.timestamp_ms > cutoff_ms:
            continue
        age = cutoff_ms - record.timestamp_ms
        scales = [slot for slot, window in enumerate(WINDOWS_MS) if age <= window] + [3]
        for slot in scales:
            direct[record.exact_layer][slot] += 1
            for key in set(record.layers):
                path[key][slot] += 1
    result = ({key: tuple(values) for key, values in path.items()},
              {key: tuple(values) for key, values in direct.items()})
    index.frequency_cache[cache_key] = result
    return result


def candidate_inputs(group: runner.QueryGroup, index: RichIndex, history: runner.TargetHistory,
                     bucket: Any, max_layers: int = MAX_LAYERS,
                     bucket_count: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Return temporary parallel-layer buckets and multiscale visit counts."""
    cache_key = (group.row.input_event_index, tuple(action.key for action in group.actions),
                 max_layers, bucket_count)
    cached = index.candidate_cache.get(cache_key)
    if cached is not None:
        return cached
    path, direct = path_frequency(index, group.row.input_event_index, group.row.input_timestamp_ms)
    layer_ids = np.zeros((len(group.actions), max_layers), dtype=np.int64)
    numeric = np.zeros((len(group.actions), CANDIDATE_WIDTH), dtype=np.float32)
    for i, action in enumerate(group.actions):
        layers = _cached_layers(action.meta, history, group.row.input_event_index, index)
        for j, key in enumerate(layers[:max_layers]):
            layer_ids[i, j] = bucket(key)
        exact = layers[0]
        numeric[i, :4] = np.log1p(path.get(exact, (0, 0, 0, 0)))
        numeric[i, 4:] = np.log1p(direct.get(exact, (0, 0, 0, 0)))
    index.candidate_cache[cache_key] = (layer_ids, numeric)
    return layer_ids, numeric


def cutoff_features(prefix: RichPrefix, events: list[dict[str, Any]],
                    timezone: str = "America/Los_Angeles") -> dict[str, float]:
    """Small shared numeric feature set for one prediction cutoff."""
    local = datetime.fromtimestamp(prefix.cutoff_timestamp_ms / 1000, tz=ZoneInfo(timezone))
    hour = local.hour + local.minute / 60.0
    weekday = local.weekday()
    quality = _quality(events[prefix.cutoff_event_index])
    numeric = sequence_numeric(prefix, events, timezone)
    closed = [float(numeric[i, 6]) for i in range(len(prefix.steps)) if numeric[i, 7] > 0]
    marker = next((step for step in reversed(prefix.steps) if step.marker_kind), None)
    gap_duration = (marker.marker_duration_ms if marker is not None else None)
    current_index = next((i for i in range(len(prefix.steps) - 1, -1, -1)
                          if prefix.steps[i].is_arrival), None)
    inferred_active = (float(numeric[current_index, 16]) if current_index is not None
                       and numeric[current_index, 17] > 0 else 0.0)
    has_inferred_active = bool(current_index is not None and numeric[current_index, 17] > 0)
    return {
        "time_hour_sin": math.sin(2 * math.pi * hour / 24),
        "time_hour_cos": math.cos(2 * math.pi * hour / 24),
        "time_weekday_sin": math.sin(2 * math.pi * weekday / 7),
        "time_weekday_cos": math.cos(2 * math.pi * weekday / 7),
        "last_confirmed_dwell_log_s": closed[-1] if closed else 0.0,
        "has_confirmed_dwell": float(bool(closed)),
        "quality_ready_fraction": quality[0],
        "quality_syncing_fraction": quality[1],
        "quality_unavailable_fraction": quality[2],
        "quality_confirmed": quality[3],
        "quality_failed_fraction": quality[4],
        "quality_stale_fraction": quality[5],
        "quality_unknown_fraction": quality[6],
        "has_legacy_short_gap": float(marker is not None and marker.marker_kind == "legacy_short_gap"),
        "has_excluded_scope_gap": float(marker is not None and marker.marker_kind == "excluded_scope"),
        "known_gap_log_s": (math.log1p(gap_duration / 1000)
                            if gap_duration is not None and gap_duration >= 0 else 0.0),
        "known_gap_duration": float(gap_duration is not None and gap_duration >= 0),
        "same_target_inferred_active_log_s": inferred_active,
        "has_same_target_inferred_active": float(has_inferred_active),
    }


def candidate_feature_rows(group: runner.QueryGroup, index: RichIndex,
                           history: runner.TargetHistory, events: list[dict[str, Any]],
                           prefix: RichPrefix) -> list[dict[str, float]]:
    """Numeric rows in exact group.actions order for a shared batch ranker."""
    if prefix.cutoff_event_index != group.row.input_event_index:
        raise ValueError("Feature prefix must be at this prediction cutoff")
    common = cutoff_features(prefix, events)
    path, direct = path_frequency(index, group.row.input_event_index, group.row.input_timestamp_ms)
    result: list[dict[str, float]] = []
    for action in group.actions:
        layers = _cached_layers(action.meta, history, group.row.input_event_index, index)
        exact = layers[0]
        path_counts = path.get(exact, (0, 0, 0, 0))
        direct_counts = direct.get(exact, (0, 0, 0, 0))
        row = dict(common)
        row["hierarchy_layer_count_log"] = math.log1p(len(layers))
        for slot, name in enumerate(("5m", "1h", "24h", "all")):
            row["path_count_" + name + "_log"] = math.log1p(path_counts[slot])
            row["direct_count_" + name + "_log"] = math.log1p(direct_counts[slot])
        result.append(row)
    return result


def step_layer_ids(prefix: RichPrefix, bucket: Any,
                   max_layers: int = MAX_LAYERS) -> np.ndarray:
    result = np.zeros((len(prefix.steps), max_layers), dtype=np.int64)
    for i, step in enumerate(prefix.steps):
        for j, key in enumerate(step.layers[:max_layers]):
            result[i, j] = bucket(key)
    return result
