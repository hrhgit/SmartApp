"""Read-only, causal cleaning of foreground navigation for local experiments.

The original export and fixed test membership are never rewritten.  A bridge
means that two *eligible* foreground destinations may be adjacent in a derived
recommendation sequence.  It does not assert that no physical actions occurred
inside the masked interval, and its time is never observed dwell.
"""

from __future__ import annotations

import collections
import dataclasses
import math
from dataclasses import dataclass
from typing import Any, Iterable
from uuid import UUID

import run_offline_batch_rankers as offline


VERSION = "cleaned-foreground-data-v1"
NON_ARRIVAL_REASONS = {"resolution_update", "browser_history", "browser_resync"}
EXCLUDED_CATEGORIES = {"excluded_app", "excluded_page", "private_browser", "secure_input"}
BOUNDARY_KINDS = {"capture.gap", "observation.scope", "session.start", "session.end"}
SHORT_GAP_MS = 1000.0


@dataclass(frozen=True)
class Confirmed:
    index: int
    event_id: str
    target_id: str
    action: offline.Action
    parent: offline.Action | None
    input_quality: str
    source_statuses: tuple[str, ...]


@dataclass(frozen=True)
class Bridge:
    """Auditable interval connecting confirmed endpoints without filling it."""

    kind: str  # legacy_short_gap or excluded_scope
    begin_event_id: str
    end_event_id: str
    before_event_id: str
    after_event_id: str
    begin_index: int
    end_index: int
    before_index: int
    after_index: int
    duration_ms: float
    resume_delay_ms: float
    category: str | None
    same_exact_target: bool
    same_action: bool
    train_eligible: bool
    before_target_id: str
    after_target_id: str
    before_quality: str
    after_quality: str
    before_source_statuses: tuple[str, ...]
    after_source_statuses: tuple[str, ...]
    active_elapsed_ms: float
    observed_pre_gap_ms: float


@dataclass(frozen=True)
class RowEvidence:
    source: str  # observed_first_step, legacy_short_gap, excluded_scope
    anchor_event_id: str
    outcome_event_id: str
    boundary_begin_event_id: str | None
    boundary_end_event_id: str | None
    interval_duration_ms: float | None
    first_step_confirmed: bool
    full_horizon_observed: bool
    inferred_next_eligible: bool
    input_quality: str
    outcome_quality: str
    current_target_id: str
    truth_target_id: str
    current_hierarchy: dict[str, Any] | None = None
    truth_hierarchy: dict[str, Any] | None = None
    input_source_statuses: tuple[str, ...] = ()
    outcome_source_statuses: tuple[str, ...] = ()
    active_elapsed_ms: float | None = None


@dataclass(frozen=True)
class CleaningResult:
    salvaged_train: tuple[offline.Observation, ...]
    salvaged_test_diagnostic: tuple[offline.Observation, ...]
    stitched_train: tuple[offline.Observation, ...]
    bridges: tuple[Bridge, ...]
    row_evidence: dict[str, RowEvidence]
    audit: dict[str, int]


def _quality(event: dict[str, Any]) -> str:
    coverage = (event.get("data") or {}).get("inputCoverage")
    if not isinstance(coverage, dict):
        return "missing"
    if coverage.get("version") != offline.FOREGROUND_COVERAGE_VERSION:
        return "wrong_version"
    return str(coverage.get("status") or "missing")


def _source_statuses(event: dict[str, Any]) -> tuple[str, ...]:
    coverage = (event.get("data") or {}).get("inputCoverage")
    sources = coverage.get("sources") if isinstance(coverage, dict) else None
    if not isinstance(sources, list):
        return ()
    return tuple(sorted(str(source.get("status") or "missing") for source in sources
                        if isinstance(source, dict)))


def _hierarchy(event: dict[str, Any], confirmed: Confirmed,
               history: offline.TargetHistory) -> dict[str, Any]:
    meta = history.metadata_at(confirmed.target_id, confirmed.index) or {}
    app = event.get("app")
    bundle = app.get("bundleID") if isinstance(app, dict) else None
    if not bundle:
        bundle = confirmed.action.bundle
    return {
        "destinationTargetID": confirmed.target_id,
        "kind": meta.get("kind"),
        "containerTargetID": meta.get("parentID"),
        "appTargetID": "app:" + bundle if isinstance(bundle, str) and bundle else None,
        "locator": meta.get("locator"),
    }


def _confirmed(event: dict[str, Any], index: int,
               history: offline.TargetHistory) -> Confirmed | None:
    if event.get("kind") != "context.change" or _quality(event) != "foreground_only":
        return None
    data = event.get("data") or {}
    reason = data.get("reason")
    if not isinstance(reason, str) or not reason or reason in NON_ARRIVAL_REASONS:
        return None
    target_id = data.get("targetID")
    if not isinstance(target_id, str) or not target_id:
        return None
    meta = history.metadata_at(target_id, index)
    action = offline.action_from_meta(meta, catalog_only=True)
    if action is None or action.kind not in offline.ACTIONABLE_KINDS:
        return None
    event_id = event.get("id")
    if not isinstance(event_id, str) or not event_id:
        return None
    parent = action if action.kind == "app" else offline.foreground_parent_action(event)
    return Confirmed(index, event_id, target_id, action, parent, _quality(event),
                     _source_statuses(event))


def _navigation(before: Confirmed, after: Confirmed) -> bool:
    if before.action.key == after.action.key:
        return False
    bundle = before.action.meta.get("bundleID")
    return not (isinstance(bundle, str) and after.action.key == "app:" + bundle)


def _observation(sample_id: str, split: str, position: int, before: Confirmed,
                 after: Confirmed, events: list[dict[str, Any]],
                 history: offline.TargetHistory) -> offline.Observation:
    catalog = history.catalog_at(before.index)
    prior_exact = after.action.key in catalog
    prior_parent = bool(after.action.kind in offline.CONCRETE_KINDS
                        and before.parent is not None and after.parent is not None
                        and after.parent.key != before.parent.key
                        and after.parent.key in catalog)
    return offline.Observation(
        sample_id=sample_id, split=split, position=position,
        input_event_index=before.index, completed_event_index=after.index,
        input_timestamp_ms=int(events[before.index]["timestampMs"]),
        outcome_timestamp_ms=int(events[after.index]["timestampMs"]),
        outcome_event_id=after.event_id,
        current=before.action, truth=after.action,
        current_parent=before.parent, truth_parent=after.parent,
        has_prior_exact_action=prior_exact, has_prior_parent_action=prior_parent,
    )


def _valid_pair_time(start: dict[str, Any], end: dict[str, Any],
                     max_ms: float | None) -> float | None:
    try:
        monotonic = float(end["monotonicMs"]) - float(start["monotonicMs"])
        wall = float(end["timestampMs"]) - float(start["timestampMs"])
    except (KeyError, ValueError, TypeError):
        return None
    if not all(math.isfinite(value) and value >= 0 for value in (monotonic, wall)):
        return None
    if max_ms is not None and (monotonic > max_ms or wall > max_ms):
        return None
    return monotonic


def _increasing_sequence(*events: dict[str, Any]) -> bool:
    values = [event.get("sequence") for event in events]
    return all(isinstance(value, int) for value in values) and all(
        earlier < later for earlier, later in zip(values, values[1:]))


def _valid_scope_id(value: Any) -> bool:
    try:
        UUID(value)
        return True
    except (ValueError, TypeError, AttributeError):
        return False


def _valid_scope_marker(event: dict[str, Any], phase: str) -> bool:
    data = event.get("data") or {}
    return (event.get("kind") == "observation.scope"
            and event.get("source") == "lifecycle"
            and event.get("app") is None
            and isinstance(data, dict)
            and set(data) == {"scopeVersion", "scopeID", "phase", "category"}
            and data.get("scopeVersion") == "excluded-foreground-v1"
            and data.get("phase") == phase
            and _valid_scope_id(data.get("scopeID"))
            and data.get("category") in EXCLUDED_CATEGORIES)


def _find_before(events: list[dict[str, Any]], confirmations: dict[int, Confirmed],
                 begin_index: int) -> Confirmed | None:
    session = events[begin_index].get("sessionID")
    for index in range(begin_index - 1, -1, -1):
        event = events[index]
        if event.get("sessionID") != session or event.get("kind") in BOUNDARY_KINDS:
            return None
        if event.get("kind") == "context.change":
            return confirmations.get(index)
    return None


def _find_end_and_after(events: list[dict[str, Any]], confirmations: dict[int, Confirmed],
                        begin_index: int, kind: str) -> tuple[int, Confirmed] | None:
    begin = events[begin_index]
    session = begin.get("sessionID")
    data = begin.get("data") or {}
    end_index: int | None = None
    for index in range(begin_index + 1, len(events)):
        event = events[index]
        event_kind = event.get("kind")
        if event.get("sessionID") != session or event_kind in {"session.start", "session.end"}:
            return None
        if end_index is None:
            if event_kind == "context.change":
                return None
            if event_kind in {"capture.gap", "observation.scope"}:
                end_data = event.get("data") or {}
                if kind == "legacy_short_gap":
                    if event_kind != "capture.gap" or end_data.get("reason") != "observation_resumed":
                        return None
                else:
                    if (not _valid_scope_marker(event, "end")
                            or end_data.get("scopeID") != data.get("scopeID")
                            or end_data.get("category") != data.get("category")):
                        return None
                end_index = index
                continue
        else:
            if event_kind in BOUNDARY_KINDS:
                return None
            if event_kind == "context.change":
                after = confirmations.get(index)
                return (end_index, after) if after is not None else None
    return None


def _bridges(events: list[dict[str, Any]], history: offline.TargetHistory,
             test_boundary_index: int, audit: collections.Counter[str]) -> tuple[Bridge, ...]:
    confirmations = {index: candidate for index, event in enumerate(events)
                     if (candidate := _confirmed(event, index, history)) is not None}
    result: list[Bridge] = []
    for begin_index, begin in enumerate(events):
        data = begin.get("data") or {}
        kind: str | None = None
        if begin.get("kind") == "capture.gap" and data.get("reason") == "observation_suspended":
            # Paired modern pauses have their own continuity semantics. Legacy
            # records carry no cause or ID and may bridge only under this much
            # narrower local rule.
            if any(key in data for key in ("pauseID", "causeID", "pauseVersion", "causeCategory")):
                continue
            kind = "legacy_short_gap"
        elif _valid_scope_marker(begin, "begin"):
            kind = "excluded_scope"
        if kind is None:
            continue
        audit[f"{kind}_begins"] += 1
        before = _find_before(events, confirmations, begin_index)
        pair = _find_end_and_after(events, confirmations, begin_index, kind)
        if before is None or pair is None:
            audit[f"{kind}_rejected_missing_endpoint_or_boundary"] += 1
            continue
        end_index, after = pair
        end = events[end_index]
        maximum = SHORT_GAP_MS if kind == "legacy_short_gap" else None
        duration = _valid_pair_time(begin, end, maximum)
        delay = _valid_pair_time(end, events[after.index], maximum)
        observed_pre_gap = _valid_pair_time(events[before.index], begin, None)
        total = _valid_pair_time(events[before.index], events[after.index], None)
        if (duration is None or delay is None or observed_pre_gap is None or total is None
                or not _increasing_sequence(events[before.index], begin, end, events[after.index])):
            audit[f"{kind}_rejected_time"] += 1
            continue
        if not (before.index < begin_index < end_index < after.index):
            audit[f"{kind}_rejected_order"] += 1
            continue
        bridge = Bridge(
            kind=kind, begin_event_id=begin["id"], end_event_id=end["id"],
            before_event_id=before.event_id, after_event_id=after.event_id,
            begin_index=begin_index, end_index=end_index,
            before_index=before.index, after_index=after.index,
            duration_ms=duration, resume_delay_ms=delay,
            category=data.get("category") if kind == "excluded_scope" else None,
            same_exact_target=before.target_id == after.target_id,
            same_action=before.action.key == after.action.key,
            train_eligible=after.index < test_boundary_index,
            before_target_id=before.target_id, after_target_id=after.target_id,
            before_quality=before.input_quality, after_quality=after.input_quality,
            before_source_statuses=before.source_statuses,
            after_source_statuses=after.source_statuses,
            active_elapsed_ms=total - duration,
            observed_pre_gap_ms=observed_pre_gap,
        )
        result.append(bridge)
        audit[f"{kind}_accepted"] += 1
        if bridge.same_exact_target:
            audit[f"{kind}_same_exact_target"] += 1
        elif bridge.same_action:
            audit[f"{kind}_same_action"] += 1
    return tuple(result)


def derive_cleaning(events: list[dict[str, Any]], samples: list[dict[str, Any]],
                    history: offline.TargetHistory, *, test_boundary_index: int,
                    existing_train_rows: Iterable[offline.Observation] = (),
                    existing_test_rows: Iterable[offline.Observation] = ()) -> CleaningResult:
    """Return additional train rows and bridge evidence; never mutate inputs.

    `test_boundary_index` is the already-frozen split's first test input index.
    Additional test-period first-step rows are diagnostic only, never used in
    fixed-member scoring or batch-model training.  A row matures only when its
    observed next target appears strictly before that boundary.
    """
    if not 0 <= test_boundary_index <= len(events):
        raise ValueError("Invalid fixed test boundary")
    by_id = {event.get("id"): index for index, event in enumerate(events)}
    if len(by_id) != len(events) or None in by_id:
        raise ValueError("Events need unique IDs")
    sample_ids = [sample.get("id") for sample in samples]
    if (any(not isinstance(sample_id, str) or not sample_id for sample_id in sample_ids)
            or len(sample_ids) != len(set(sample_ids))):
        raise ValueError("Navigation samples need unique IDs")
    train_rows = tuple(existing_train_rows)
    test_rows = tuple(existing_test_rows)
    used_outcomes = {row.outcome_event_id for row in train_rows + test_rows}
    used_sample_ids = {row.sample_id for row in train_rows + test_rows}
    evidence: dict[str, RowEvidence] = {}
    audit: collections.Counter[str] = collections.Counter()
    salvaged_train: list[offline.Observation] = []
    salvaged_test: list[offline.Observation] = []
    stitched_train: list[offline.Observation] = []

    for sample in samples:
        if sample.get("status") != "incomplete" or not sample.get("outcomes"):
            continue
        audit["incomplete_with_observed_first_outcome"] += 1
        sample_id = sample.get("id")
        if not isinstance(sample_id, str) or not sample_id or sample_id in used_sample_ids:
            audit["first_step_rejected_duplicate_or_missing_sample"] += 1
            continue
        coverage = sample.get("inputCoverage") or {}
        if (sample.get("schemaVersion") != 4
                or sample.get("labelRuleVersion") != "observed-foreground-sequence-30s-v4"
                or sample.get("inventoryEventIDs")
                or not isinstance(coverage, dict)
                or coverage.get("version") != offline.FOREGROUND_COVERAGE_VERSION
                or coverage.get("status") != "foreground_only"):
            audit["first_step_rejected_sample_contract"] += 1
            continue
        first = sample["outcomes"][0]
        before_index = by_id.get(sample.get("anchorEventID"))
        after_index = by_id.get(first.get("eventID"))
        if before_index is None or after_index is None or after_index <= before_index:
            audit["first_step_rejected_event_identity"] += 1
            continue
        before_event, after_event = events[before_index], events[after_index]
        if (sample.get("sessionID") != before_event.get("sessionID")
                or sample.get("sessionID") != after_event.get("sessionID")
                or sample.get("inputThroughSequence") != before_event.get("sequence")
                or sample.get("inputCutoffMs") != before_event.get("monotonicMs")
                or not _increasing_sequence(before_event, after_event)
                or first.get("targetID") != (after_event.get("data") or {}).get("targetID")
                or sample.get("anchorEventID") != before_event.get("id")):
            audit["first_step_rejected_mismatched_provenance"] += 1
            continue
        between = events[before_index + 1:after_index]
        if (any(event.get("sessionID") != sample["sessionID"]
                or event.get("kind") in BOUNDARY_KINDS for event in between)
                or _valid_pair_time(before_event, after_event, None) is None):
            audit["first_step_rejected_crosses_boundary"] += 1
            continue
        before = _confirmed(before_event, before_index, history)
        after = _confirmed(after_event, after_index, history)
        if before is None or after is None or not _navigation(before, after):
            audit["first_step_rejected_non_navigation"] += 1
            continue
        if after.event_id in used_outcomes:
            audit["first_step_rejected_duplicate_outcome"] += 1
            continue
        if after_index < test_boundary_index:
            split = "train"
            position = len(train_rows) + len(salvaged_train)
        elif before_index >= test_boundary_index:
            split = "test_diagnostic"
            position = len(train_rows) + len(test_rows) + len(salvaged_test)
        else:
            audit["first_step_guard_crosses_test_boundary"] += 1
            continue
        row = _observation(sample_id, split, position, before, after, events, history)
        (salvaged_train if split == "train" else salvaged_test).append(row)
        used_outcomes.add(after.event_id)
        used_sample_ids.add(sample_id)
        hierarchy = sample.get("hierarchy")
        outcome_hierarchy = first.get("hierarchy")
        evidence[sample_id] = RowEvidence(
            source="observed_first_step", anchor_event_id=before.event_id,
            outcome_event_id=after.event_id, boundary_begin_event_id=None,
            boundary_end_event_id=None, interval_duration_ms=None,
            first_step_confirmed=True, full_horizon_observed=False,
            inferred_next_eligible=False, input_quality=before.input_quality,
            outcome_quality=after.input_quality, current_target_id=before.target_id,
            truth_target_id=after.target_id,
            current_hierarchy=hierarchy if isinstance(hierarchy, dict) else None,
            truth_hierarchy=outcome_hierarchy if isinstance(outcome_hierarchy, dict) else None,
            input_source_statuses=before.source_statuses,
            outcome_source_statuses=after.source_statuses,
            active_elapsed_ms=_valid_pair_time(before_event, after_event, None),
        )
        audit[f"first_step_{split}"] += 1

    bridges = _bridges(events, history, test_boundary_index, audit)
    for bridge in bridges:
        if not bridge.train_eligible or bridge.same_exact_target or bridge.same_action:
            continue
        before = _confirmed(events[bridge.before_index], bridge.before_index, history)
        after = _confirmed(events[bridge.after_index], bridge.after_index, history)
        if before is None or after is None or not _navigation(before, after):
            audit[f"{bridge.kind}_rejected_non_navigation"] += 1
            continue
        if bridge.active_elapsed_ms > 30_000:
            audit[f"{bridge.kind}_outside_30s_active_horizon"] += 1
            continue
        if after.event_id in used_outcomes:
            audit[f"{bridge.kind}_duplicate_outcome"] += 1
            continue
        sample_id = f"derived:{bridge.kind}:{bridge.begin_event_id}:{bridge.after_event_id}"
        row = _observation(sample_id, "train", len(train_rows) + len(salvaged_train) + len(stitched_train),
                           before, after, events, history)
        stitched_train.append(row)
        used_outcomes.add(after.event_id)
        evidence[sample_id] = RowEvidence(
            source=bridge.kind, anchor_event_id=before.event_id,
            outcome_event_id=after.event_id,
            boundary_begin_event_id=bridge.begin_event_id,
            boundary_end_event_id=bridge.end_event_id,
            interval_duration_ms=bridge.duration_ms,
            first_step_confirmed=True, full_horizon_observed=False,
            inferred_next_eligible=True,
            input_quality=before.input_quality, outcome_quality=after.input_quality,
            current_target_id=before.target_id, truth_target_id=after.target_id,
            current_hierarchy=_hierarchy(events[before.index], before, history),
            truth_hierarchy=_hierarchy(events[after.index], after, history),
            input_source_statuses=before.source_statuses,
            outcome_source_statuses=after.source_statuses,
            active_elapsed_ms=bridge.active_elapsed_ms,
        )
        audit[f"{bridge.kind}_train_navigation_row"] += 1
    audit["total_extra_train_rows"] = len(salvaged_train) + len(stitched_train)
    return CleaningResult(tuple(salvaged_train), tuple(salvaged_test), tuple(stitched_train),
                          bridges, evidence, dict(sorted(audit.items())))


def private_ledger_records(result: CleaningResult) -> tuple[dict[str, Any], ...]:
    """Records for a permission-restricted *private training run* only.

    The ledger can contain target IDs and full locators. Do not copy these
    records into source-controlled reports or share them as aggregate results.
    No excluded foreground object identity or recommendation candidate list is
    present. The original snapshot remains the source of truth.
    """
    records: list[dict[str, Any]] = []
    for group, rows in (("train_first_step", result.salvaged_train),
                        ("test_first_step_diagnostic", result.salvaged_test_diagnostic),
                        ("train_inferred_bridge", result.stitched_train)):
        for row in rows:
            records.append({"recordType": group, "sampleID": row.sample_id,
                            "evidence": dataclasses.asdict(result.row_evidence[row.sample_id])})
    for bridge in result.bridges:
        records.append({"recordType": "continuity_bridge", "bridge": dataclasses.asdict(bridge)})
    return tuple(records)
