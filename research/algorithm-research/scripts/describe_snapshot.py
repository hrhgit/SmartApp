#!/usr/bin/env python3
"""Build a privacy-preserving descriptive report for a SmartApp v2 export.

The program deliberately emits aggregates only.  It never copies raw events,
target names, locators, target IDs, browser IDs, or source-instance IDs into
the output run directory.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from typing import Any, Iterable


ANALYSIS_VERSION = "descriptive-v1"
REQUIRED_FILES = (
    "events.jsonl",
    "target-history.jsonl",
    "navigation-samples.jsonl",
    "manifest.json",
)
TARGET_KINDS = {"app", "window", "page", "directory", "file"}
SAMPLE_STATUSES = {"observed_horizon", "incomplete", "pending"}
SAMPLE_ORIGINS = {"natural_observed", "recommendation_related"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create descriptives from a SmartApp v2 export snapshot."
    )
    parser.add_argument("--snapshot", type=Path, required=True, help="Immutable v2 export directory")
    parser.add_argument("--output", type=Path, required=True, help="New private training run directory")
    parser.add_argument(
        "--include-targets",
        action="store_true",
        help="Include concrete target names and locators in a private detailed report and CSV.",
    )
    parser.add_argument(
        "--target-limit",
        type=int,
        default=25,
        help="Number of concrete targets to show in the report (default: 25; CSV remains complete).",
    )
    args = parser.parse_args()
    if args.target_limit < 1:
        parser.error("--target-limit must be at least 1")
    return args


def error(message: str) -> None:
    print(f"error: {message}", file=sys.stderr)
    raise SystemExit(2)


def as_mapping(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def known(value: Any, allowed: set[str], fallback: str = "unknown") -> str:
    return value if isinstance(value, str) and value in allowed else fallback


def source_family(source: Any) -> str:
    """Keep source-instance identifiers (PID, browser window UUID) out of output."""
    if not isinstance(source, str):
        return "other"
    if source.startswith("browser-extension:") or source.startswith("browser:") or source == "browser_automation":
        return "browser"
    if source.startswith("finder:"):
        return "finder"
    if source.startswith("qspace:"):
        return "qspace"
    if source.startswith("ax:") or source == "accessibility":
        return "accessibility_window"
    if source in {"lifecycle", "running_apps"}:
        return "application_lifecycle"
    if source in {"workspace", "windows", "apps"}:
        return "workspace"
    if source == "target_catalog":
        return "target_catalog"
    if source in {"pasteboard_metadata", "clipboard"}:
        return "clipboard_metadata"
    if source == "system_idle":
        return "system_idle"
    return "other"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_snapshot(snapshot: Path) -> dict[str, Any]:
    if not snapshot.is_dir():
        error(f"snapshot is not a directory: {snapshot}")
    missing = [name for name in REQUIRED_FILES if not (snapshot / name).is_file()]
    if missing:
        error("snapshot is missing required file(s): " + ", ".join(missing))

    checksum_file = snapshot / "SHA256SUMS"
    if not checksum_file.is_file():
        return {"status": "not_present", "files": {}}

    checksums: dict[str, str] = {}
    pattern = re.compile(r"^([0-9a-fA-F]{64})\s+\*?(.+)$")
    for line_number, raw_line in enumerate(checksum_file.read_text(encoding="utf-8").splitlines(), start=1):
        if not raw_line.strip():
            continue
        match = pattern.match(raw_line)
        if not match:
            error(f"cannot parse SHA256SUMS line {line_number}")
        expected, name = match.groups()
        candidate = snapshot / name
        if not candidate.is_file():
            error(f"checksum references a missing file: {name}")
        actual = sha256(candidate)
        if actual.lower() != expected.lower():
            error(f"checksum mismatch for {name}")
        checksums[name] = actual
    if not checksums:
        error("SHA256SUMS is present but has no checksum entries")
    return {"status": "verified", "files": checksums}


def read_json_lines(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            if not raw_line.strip():
                continue
            try:
                value = json.loads(raw_line)
            except json.JSONDecodeError as exc:
                error(f"invalid JSON in {path.name} line {line_number}: {exc.msg}")
            if not isinstance(value, dict):
                error(f"non-object JSON value in {path.name} line {line_number}")
            yield value


def quantile(values: list[float], proportion: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = (len(ordered) - 1) * proportion
    low = math.floor(index)
    high = math.ceil(index)
    if low == high:
        return ordered[low]
    return ordered[low] + (ordered[high] - ordered[low]) * (index - low)


def rounded(value: float | None, digits: int = 1) -> float | None:
    return round(value, digits) if value is not None else None


def share(numerator: int | float, denominator: int | float) -> float | None:
    return round(100 * numerator / denominator, 1) if denominator else None


def summarize_distribution(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"count": 0, "min": None, "median": None, "p90": None, "p99": None, "max": None}
    return {
        "count": len(values),
        "min": rounded(min(values)),
        "median": rounded(float(median(values))),
        "p90": rounded(quantile(values, 0.90)),
        "p99": rounded(quantile(values, 0.99)),
        "max": rounded(max(values)),
    }


def utc_time(timestamp_ms: int | float | None) -> str | None:
    if not isinstance(timestamp_ms, (int, float)):
        return None
    return datetime.fromtimestamp(timestamp_ms / 1000, tz=timezone.utc).isoformat().replace("+00:00", "Z")


def utc_date(timestamp_ms: int | float | None) -> str | None:
    rendered = utc_time(timestamp_ms)
    return rendered[:10] if rendered else None


def top_share(counts: Counter[Any], limit: int) -> float | None:
    total = sum(counts.values())
    return share(sum(count for _, count in counts.most_common(limit)), total)


def analyze_events(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    event_kinds: Counter[str] = Counter()
    source_families: Counter[str] = Counter()
    sessions: dict[str, dict[str, Any]] = {}
    apps: set[str] = set()
    target_by_id: dict[str, dict[str, str | None]] = {}
    target_updates_by_kind: Counter[str] = Counter()
    target_locator_by_kind: dict[str, Counter[str]] = defaultdict(Counter)
    context_records: list[dict[str, Any]] = []
    capture_gap_sessions: set[str] = set()
    capture_gaps = 0
    inventory_entrance_counts: list[float] = []
    browser_state_count = 0
    browser_foreground_count = 0
    browser_tab_counts: list[float] = []
    first_timestamp: int | float | None = None
    last_timestamp: int | float | None = None
    target_update_records = 0
    total_events = 0

    for event in read_json_lines(path):
        total_events += 1
        kind = event.get("kind") if isinstance(event.get("kind"), str) else "unknown"
        event_kinds[kind] += 1
        source_families[source_family(event.get("source"))] += 1
        timestamp = event.get("timestampMs")
        if isinstance(timestamp, (int, float)):
            first_timestamp = timestamp if first_timestamp is None else min(first_timestamp, timestamp)
            last_timestamp = timestamp if last_timestamp is None else max(last_timestamp, timestamp)
        session_id = event.get("sessionID")
        if not isinstance(session_id, str):
            session_id = "__missing_session__"
        session = sessions.setdefault(session_id, {"events": 0, "first": None, "last": None})
        session["events"] += 1
        if isinstance(timestamp, (int, float)):
            session["first"] = timestamp if session["first"] is None else min(session["first"], timestamp)
            session["last"] = timestamp if session["last"] is None else max(session["last"], timestamp)

        app = as_mapping(event.get("app"))
        bundle_id = app.get("bundleID")
        if isinstance(bundle_id, str):
            apps.add(bundle_id)
        data = as_mapping(event.get("data"))

        if kind == "target.update":
            target_update_records += 1
            target = as_mapping(data.get("target"))
            target_id = target.get("id")
            target_kind = known(target.get("kind"), TARGET_KINDS)
            target_updates_by_kind[target_kind] += 1
            if isinstance(target_id, str):
                target_by_id[target_id] = {
                    "kind": target_kind,
                    "name": target.get("name") if isinstance(target.get("name"), str) else "(未命名目标)",
                    "locator": target.get("locator") if isinstance(target.get("locator"), str) else None,
                    "origin": target.get("origin") if isinstance(target.get("origin"), str) else None,
                    "bundle_id": target.get("bundleID") if isinstance(target.get("bundleID"), str) else None,
                }
            target_locator_by_kind[target_kind]["with_locator" if isinstance(target.get("locator"), str) else "without_locator"] += 1

        if kind == "context.change":
            target_id = data.get("targetID") if isinstance(data.get("targetID"), str) else None
            reason = data.get("reason") if isinstance(data.get("reason"), str) else None
            monotonic = event.get("monotonicMs")
            context_records.append(
                {
                    "session": session_id,
                    "timestamp": timestamp if isinstance(timestamp, (int, float)) else None,
                    "monotonic": monotonic if isinstance(monotonic, (int, float)) else None,
                    "target_id": target_id,
                    "resolution_update": reason == "resolution_update",
                    "recommendation_related": isinstance(event.get("decisionID"), str),
                }
            )

        if kind == "capture.gap":
            capture_gaps += 1
            capture_gap_sessions.add(session_id)

        if kind == "inventory.change":
            entrances = as_list(data.get("entrances"))
            inventory_entrance_counts.append(float(len(entrances)))
            browser_state = as_mapping(data.get("browserState"))
            if browser_state:
                browser_state_count += 1
                if browser_state.get("browserForeground") is True:
                    browser_foreground_count += 1
                tab_count = sum(len(as_list(as_mapping(window).get("tabs"))) for window in as_list(browser_state.get("windows")))
                browser_tab_counts.append(float(tab_count))

    anchors = [record for record in context_records if not record["resolution_update"]]
    contexts_by_session: dict[str, list[dict[str, Any]]] = defaultdict(list)
    target_activity: Counter[str] = Counter()
    context_target_kinds: Counter[str] = Counter()
    anchors_by_date: Counter[str] = Counter()
    anchors_by_hour: Counter[str] = Counter()
    recommendation_related_anchors = 0
    anchors_missing_target = 0
    for record in anchors:
        contexts_by_session[record["session"]].append(record)
        target_id = record["target_id"]
        if isinstance(target_id, str):
            target_activity[target_id] += 1
            context_target_kinds[target_by_id.get(target_id, {}).get("kind", "unknown")] += 1
        else:
            anchors_missing_target += 1
            context_target_kinds["unknown"] += 1
        if record["recommendation_related"]:
            recommendation_related_anchors += 1
        timestamp = record["timestamp"]
        if isinstance(timestamp, (int, float)):
            day = utc_date(timestamp)
            if day:
                anchors_by_date[day] += 1
            hour = datetime.fromtimestamp(timestamp / 1000, tz=timezone.utc).strftime("%H:00Z")
            anchors_by_hour[hour] += 1

    transitions: Counter[tuple[str, str]] = Counter()
    inter_anchor_gaps_ms: list[float] = []
    self_transitions = 0
    valid_transition_count = 0
    session_anchor_counts: list[float] = []
    session_unique_target_counts: list[float] = []
    sessions_with_anchors = 0
    for records in contexts_by_session.values():
        ordered = sorted(
            records,
            key=lambda record: (
                record["monotonic"] if isinstance(record["monotonic"], (int, float)) else float("inf"),
                record["timestamp"] if isinstance(record["timestamp"], (int, float)) else float("inf"),
            ),
        )
        sessions_with_anchors += 1
        session_anchor_counts.append(float(len(ordered)))
        session_unique_target_counts.append(float(len({record["target_id"] for record in ordered if isinstance(record["target_id"], str)})))
        previous: dict[str, Any] | None = None
        for record in ordered:
            if previous:
                current_monotonic = record["monotonic"]
                prior_monotonic = previous["monotonic"]
                if isinstance(current_monotonic, (int, float)) and isinstance(prior_monotonic, (int, float)):
                    gap = current_monotonic - prior_monotonic
                    if gap >= 0:
                        inter_anchor_gaps_ms.append(float(gap))
                target_id = record["target_id"]
                prior_target_id = previous["target_id"]
                if isinstance(target_id, str) and isinstance(prior_target_id, str):
                    valid_transition_count += 1
                    transitions[(prior_target_id, target_id)] += 1
                    if target_id == prior_target_id:
                        self_transitions += 1
            previous = record

    session_spans_ms = [
        float(session["last"] - session["first"])
        for session in sessions.values()
        if isinstance(session["first"], (int, float)) and isinstance(session["last"], (int, float))
    ]
    target_activity_counts = list(target_activity.values())
    mapping_coverage = share(len(anchors) - anchors_missing_target, len(anchors))
    detailed_targets: list[dict[str, Any]] = []
    for target_id, arrivals in target_activity.items():
        target = target_by_id.get(target_id)
        if not target:
            continue
        detailed_targets.append({
            "kind": target["kind"],
            "name": target["name"],
            "locator": target["locator"],
            "origin": target["origin"],
            "bundle_id": target["bundle_id"],
            "arrivals": arrivals,
        })
    detailed_targets.sort(key=lambda target: (-int(target["arrivals"]), str(target["kind"]), str(target["name"])))
    for rank, target in enumerate(detailed_targets, start=1):
        target["rank"] = rank

    return {
        "event_count": total_events,
        "time_window_utc": {"first": utc_time(first_timestamp), "last": utc_time(last_timestamp)},
        "event_kinds": dict(event_kinds.most_common()),
        "source_families": dict(source_families.most_common()),
        "sessions": {
            "count": len(sessions),
            "with_context_anchors": sessions_with_anchors,
            "without_context_anchors": len(sessions) - sessions_with_anchors,
            "unique_observed_applications": len(apps),
            "events_per_session": summarize_distribution([float(session["events"]) for session in sessions.values()]),
            "observed_span_ms": summarize_distribution(session_spans_ms),
        },
        "targets": {
            "target_update_records": target_update_records,
            "unique_target_ids": len(target_by_id),
            "target_update_records_by_kind": dict(target_updates_by_kind.most_common()),
            "locator_coverage_by_kind": {
                kind: {
                    "with_locator": counts["with_locator"],
                    "without_locator": counts["without_locator"],
                    "percent_with_locator": share(counts["with_locator"], sum(counts.values())),
                }
                for kind, counts in sorted(target_locator_by_kind.items())
            },
        },
        "observed_arrivals": {
            "context_change_events": len(context_records),
            "resolution_updates_excluded_from_anchors": len(context_records) - len(anchors),
            "independent_context_anchors": len(anchors),
            "anchors_missing_target_reference": anchors_missing_target,
            "anchors_with_target_reference_percent": mapping_coverage,
            "recommendation_related_anchors": recommendation_related_anchors,
            "context_target_kind_counts": dict(context_target_kinds.most_common()),
            "anchors_by_utc_date": dict(sorted(anchors_by_date.items())),
            "top_utc_hours_by_anchor_count": dict(anchors_by_hour.most_common(6)),
            "active_target_count": len(target_activity),
            "target_activity": {
                "single_observation_targets": sum(1 for count in target_activity_counts if count == 1),
                "single_observation_target_percent": share(sum(1 for count in target_activity_counts if count == 1), len(target_activity_counts)),
                "top_1_percent": top_share(target_activity, 1),
                "top_5_percent": top_share(target_activity, 5),
                "top_10_percent": top_share(target_activity, 10),
                "observations_per_active_target": summarize_distribution([float(value) for value in target_activity_counts]),
            },
            "transitions": {
                "observed_pairs_with_target_references": valid_transition_count,
                "unique_directed_edges": len(transitions),
                "self_transition_count": self_transitions,
                "self_transition_percent": share(self_transitions, valid_transition_count),
                "top_1_edge_percent": top_share(transitions, 1),
                "top_5_edge_percent": top_share(transitions, 5),
                "inter_anchor_gap_ms": summarize_distribution(inter_anchor_gaps_ms),
            },
            "anchors_per_session": summarize_distribution(session_anchor_counts),
            "unique_targets_per_active_session": summarize_distribution(session_unique_target_counts),
        },
        "observation_quality": {
            "capture_gap_events": capture_gaps,
            "sessions_with_capture_gap": len(capture_gap_sessions),
            "inventory_change_events": len(inventory_entrance_counts),
            "entrances_per_inventory_snapshot": summarize_distribution(inventory_entrance_counts),
            "browser_state_inventory_events": browser_state_count,
            "browser_state_foreground_events": browser_foreground_count,
            "browser_tabs_per_state_snapshot": summarize_distribution(browser_tab_counts),
        },
    }, detailed_targets


def analyze_samples(path: Path) -> dict[str, Any]:
    statuses: Counter[str] = Counter()
    input_coverage: Counter[str] = Counter()
    outcome_coverage: Counter[str] = Counter()
    origins: Counter[str] = Counter()
    sample_sessions: set[str] = set()
    outcome_counts: list[float] = []
    recent_context_counts: list[float] = []
    activity_context_counts: list[float] = []
    inventory_counts: list[float] = []
    samples_with_outcomes = 0
    complete_with_outcomes = 0
    total = 0
    for sample in read_json_lines(path):
        total += 1
        status = known(sample.get("status"), SAMPLE_STATUSES, "other")
        origin = known(sample.get("origin"), SAMPLE_ORIGINS, "other")
        statuses[status] += 1
        input_coverage[(sample.get("inputCoverage") or {}).get("status", "unknown")] += 1
        outcome_coverage[sample.get("outcomeCoverage") or "unknown"] += 1
        origins[origin] += 1
        session_id = sample.get("sessionID")
        if isinstance(session_id, str):
            sample_sessions.add(session_id)
        outcome_count = len(as_list(sample.get("outcomes")))
        outcome_counts.append(float(outcome_count))
        recent_context_counts.append(float(len(as_list(sample.get("recentContextEventIDs")))))
        activity_context_counts.append(float(len(as_list(sample.get("activityContextEventIDs")))))
        inventory_counts.append(float(len(as_list(sample.get("inventoryEventIDs")))))
        if outcome_count:
            samples_with_outcomes += 1
            if status == "observed_horizon":
                complete_with_outcomes += 1
    return {
        "count": total,
        "sessions": len(sample_sessions),
        "status_counts": dict(statuses.most_common()),
        "input_coverage_counts": dict(input_coverage.most_common()),
        "outcome_coverage_counts": dict(outcome_coverage.most_common()),
        "origin_counts": dict(origins.most_common()),
        "samples_with_at_least_one_outcome": samples_with_outcomes,
        "samples_with_at_least_one_outcome_percent": share(samples_with_outcomes, total),
        "complete_samples_with_at_least_one_outcome": complete_with_outcomes,
        "complete_samples_with_at_least_one_outcome_percent": share(
            complete_with_outcomes, statuses["observed_horizon"]
        ),
        "outcomes_per_sample": summarize_distribution(outcome_counts),
        "recent_context_references_per_sample": summarize_distribution(recent_context_counts),
        "activity_context_references_per_sample": summarize_distribution(activity_context_counts),
        "inventory_references_per_sample": summarize_distribution(inventory_counts),
    }


def validate_snapshot_counts(events: dict[str, Any], samples: dict[str, Any], target_history_count: int, manifest: dict[str, Any]) -> dict[str, bool | None]:
    def same_manifest(name: str, actual: int) -> bool | None:
        expected = manifest.get(name)
        return expected == actual if isinstance(expected, int) else None

    return {
        "events_match_manifest": same_manifest("events", events["event_count"]),
        "samples_match_manifest": same_manifest("samples", samples["count"]),
        "target_history_matches_target_updates": target_history_count == events["targets"]["target_update_records"],
        "target_history_match_manifest": same_manifest("targetRecords", target_history_count),
    }


def format_number(value: int | float | None) -> str:
    if value is None:
        return "—"
    if isinstance(value, float) and value.is_integer():
        value = int(value)
    return f"{value:,}" if isinstance(value, int) else f"{value:,.1f}"


def format_percent(value: float | None) -> str:
    return "—" if value is None else f"{value:.1f}%"


def format_duration_ms(value: float | None) -> str:
    if value is None:
        return "—"
    if value < 1_000:
        return f"{value:.0f} ms"
    if value < 60_000:
        return f"{value / 1_000:.1f} s"
    if value < 3_600_000:
        return f"{value / 60_000:.1f} min"
    return f"{value / 3_600_000:.1f} h"


def table(headers: list[str], rows: Iterable[Iterable[str]]) -> list[str]:
    rendered_rows = [list(row) for row in rows]
    result = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    result.extend("| " + " | ".join(row) + " |" for row in rendered_rows)
    return result


def markdown_cell(value: Any) -> str:
    return str(value).replace("\\", "\\\\").replace("|", "\\|").replace("\r", "").replace("\n", "<br>")


def concrete_target_label(target: dict[str, Any]) -> str:
    locator = target.get("locator")
    return markdown_cell(locator if isinstance(locator, str) and locator else target.get("name", "(未命名目标)"))


def report(metrics: dict[str, Any], detailed_targets: list[dict[str, Any]] | None = None,
           target_limit: int = 25) -> str:
    events = metrics["events"]
    targets = events["targets"]
    arrivals = events["observed_arrivals"]
    quality = events["observation_quality"]
    sessions = events["sessions"]
    samples = metrics["samples"]
    detail_mode = detailed_targets is not None
    sequence_section = 4 if detail_mode else 3
    quality_section = sequence_section + 1
    implications_section = quality_section + 1
    privacy_note = (
        "- 详细目标：按你的明确要求，报告及私有 CSV 含目标名称和完整 URL/路径；仍不输出目标 ID、标签页 ID、进程 ID 或原始 source 实例标识。"
        if detail_mode else
        "- 隐私边界：本报告只含聚合统计；不输出目标名称、URL、路径、目标 ID、标签页 ID、进程 ID 或原始 source 实例标识。"
    )
    lines = [
        "# SmartApp v2 数据描述性分析",
        "",
        f"- 分析版本：`{ANALYSIS_VERSION}`",
        f"- 输入快照：`{metrics['input']['snapshot_name']}`",
        f"- 生成时间（UTC）：{metrics['generated_at_utc']}",
        f"- 快照校验：{metrics['input']['integrity']['status']}",
        privacy_note,
        "",
        "## 结论摘要",
        "",
        f"本快照覆盖 {events['time_window_utc']['first']} 至 {events['time_window_utc']['last']}，共有 {format_number(events['event_count'])} 条原始事件、{format_number(sessions['count'])} 个会话和 {format_number(arrivals['independent_context_anchors'])} 个独立前台目标锚点。",
        f"其中 {format_number(arrivals['resolution_updates_excluded_from_anchors'])} 条 `resolution_update` 已从独立到达统计中排除；导出的序列样本为 {format_number(samples['count'])} 条，完整观察窗口占 {format_percent(share(samples['status_counts'].get('observed_horizon', 0), samples['count']))}。",
        "",
        "这些量足以开始可复现的时间切分和近期访问/转移基线评估，但仍应把结论当作单用户、有限会话的早期信号，而非稳定泛化证据。",
        "",
        "## 1. 数据规模与事件组成",
        "",
    ]
    lines.extend(table(["指标", "数值"], [
        ["原始事件", format_number(events["event_count"])],
        ["采集会话", format_number(sessions["count"])],
        ["有独立目标锚点的会话", format_number(sessions["with_context_anchors"])],
        ["观察到的应用数量（仅计数）", format_number(sessions["unique_observed_applications"])],
        ["目标更新记录 / 唯一目标 ID", f"{format_number(targets['target_update_records'])} / {format_number(targets['unique_target_ids'])}"],
        ["全部 context.change", format_number(arrivals["context_change_events"])],
        ["可作独立序列锚点的 context.change", format_number(arrivals["independent_context_anchors"])],
        ["可关联目标引用的锚点", format_percent(arrivals["anchors_with_target_reference_percent"])],
    ]))
    lines.extend(["", "事件类型按原始写入事件计数；心跳与库存快照是观测健康度/上下文，不应直接当作到达标签。", ""])
    lines.extend(table(["事件类型", "事件数", "占全部事件"], [
        [kind, format_number(count), format_percent(share(count, events["event_count"]))]
        for kind, count in events["event_kinds"].items()
    ]))
    lines.extend(["", "## 2. 采集来源与目标形态", ""])
    lines.extend(table(["归一化来源族", "事件数", "占全部事件"], [
        [family, format_number(count), format_percent(share(count, events["event_count"]))]
        for family, count in events["source_families"].items()
    ]))
    lines.extend(["", "这里已将浏览器窗口 UUID、Accessibility PID 等实例标识合并为来源族。", ""])
    lines.extend(table(["目标类型", "目标更新记录", "前台锚点"], [
        [
            kind,
            format_number(targets["target_update_records_by_kind"].get(kind, 0)),
            format_number(arrivals["context_target_kind_counts"].get(kind, 0)),
        ]
        for kind in sorted(set(targets["target_update_records_by_kind"]) | set(arrivals["context_target_kind_counts"]))
    ]))
    lines.extend(["", "定位字段覆盖（只报告有无，不展示定位内容）：", ""])
    lines.extend(table(["目标类型", "有定位字段", "无定位字段", "覆盖率"], [
        [
            kind,
            format_number(values["with_locator"]),
            format_number(values["without_locator"]),
            format_percent(values["percent_with_locator"]),
        ]
        for kind, values in targets["locator_coverage_by_kind"].items()
    ]))
    if detailed_targets is not None:
        total_arrivals = arrivals["independent_context_anchors"]
        lines.extend([
            "",
            "## 3. 具体到达目标（私有详细版）",
            "",
            f"下表显示前 {min(target_limit, len(detailed_targets))} 个实际前台到达目标；完整的 {len(detailed_targets)} 个活跃目标清单见同目录 `concrete-target-arrivals.csv`。页面和目录优先显示完整 locator。",
            "",
        ])
        lines.extend(table(["总排名", "类型", "具体目标", "到达次数", "占全部锚点"], [
            [
                format_number(target["rank"]),
                markdown_cell(target["kind"]),
                concrete_target_label(target),
                format_number(target["arrivals"]),
                format_percent(share(target["arrivals"], total_arrivals)),
            ]
            for target in detailed_targets[:target_limit]
        ]))
        for kind in ("app", "window", "page", "directory", "file"):
            leaders = [target for target in detailed_targets if target["kind"] == kind][:5]
            if not leaders:
                continue
            lines.extend(["", f"### {kind} 目标 Top {len(leaders)}", ""])
            lines.extend(table(["排名", "具体目标", "到达次数"], [
                [format_number(target["rank"]), concrete_target_label(target), format_number(target["arrivals"])]
                for target in leaders
            ]))
    activity = arrivals["target_activity"]
    transitions = arrivals["transitions"]
    lines.extend(["", f"## {sequence_section}. 序列、长尾与节奏", ""])
    lines.extend(table(["指标", "数值"], [
        ["活跃目标数（至少一次独立到达）", format_number(arrivals["active_target_count"])],
        ["只出现一次的活跃目标", f"{format_number(activity['single_observation_targets'])} ({format_percent(activity['single_observation_target_percent'])})"],
        ["最常见 1 / 5 / 10 个目标的到达占比", " / ".join(format_percent(activity[key]) for key in ("top_1_percent", "top_5_percent", "top_10_percent"))],
        ["可观察到的目标对转移 / 唯一有向边", f"{format_number(transitions['observed_pairs_with_target_references'])} / {format_number(transitions['unique_directed_edges'])}"],
        ["同目标连续观测比例", format_percent(transitions["self_transition_percent"])],
        ["最常见 1 / 5 条边的转移占比", " / ".join(format_percent(transitions[key]) for key in ("top_1_edge_percent", "top_5_edge_percent"))],
        ["每个活跃会话的锚点数（中位数 / P90）", f"{format_number(arrivals['anchors_per_session']['median'])} / {format_number(arrivals['anchors_per_session']['p90'])}"],
        ["相邻锚点间隔（中位数 / P90）", f"{format_duration_ms(transitions['inter_anchor_gap_ms']['median'])} / {format_duration_ms(transitions['inter_anchor_gap_ms']['p90'])}"],
    ]))
    lines.extend(["", "按 UTC 日期的独立锚点数：", ""])
    lines.extend(table(["UTC 日期", "独立锚点"], [[day, format_number(count)] for day, count in arrivals["anchors_by_utc_date"].items()]))
    lines.extend(["", "相邻锚点间隔是观察时间差，不是任务完成时间，也不能推出用户注意力或意图。", ""])
    lines.extend([f"## {quality_section}. 观察质量与导出序列样本", ""])
    lines.extend(table(["指标", "数值"], [
        ["capture.gap 事件", format_number(quality["capture_gap_events"])],
        ["发生过 gap 的会话", format_number(quality["sessions_with_capture_gap"])],
        ["库存快照", format_number(quality["inventory_change_events"])],
        ["每个库存快照的入口数（中位数 / P90）", f"{format_number(quality['entrances_per_inventory_snapshot']['median'])} / {format_number(quality['entrances_per_inventory_snapshot']['p90'])}"],
        ["含浏览器状态的库存快照", format_number(quality["browser_state_inventory_events"])],
        ["浏览器前台状态快照", format_number(quality["browser_state_foreground_events"])],
        ["导出序列样本", format_number(samples["count"])],
        ["跨过后续观察时限的样本（不代表状态覆盖完整）", format_number(samples["status_counts"].get("observed_horizon", 0))],
        ["后续观察被截断的样本", format_number(samples["status_counts"].get("incomplete", 0))],
        ["输入覆盖达标 / 部分缺失 / 无法确认", " / ".join(format_number(samples["input_coverage_counts"].get(key, 0)) for key in ["complete", "partial", "unknown"])],
        ["后续覆盖状态", json.dumps(samples["outcome_coverage_counts"], ensure_ascii=False)],
        ["跨过观察时限且有后续行为的样本", f"{format_number(samples['complete_samples_with_at_least_one_outcome'])} ({format_percent(samples['complete_samples_with_at_least_one_outcome_percent'])})"],
        ["每样本 outcomes（中位数 / P90）", f"{format_number(samples['outcomes_per_sample']['median'])} / {format_number(samples['outcomes_per_sample']['p90'])}"],
    ]))
    lines.extend(["", "完整性/一致性检查：", ""])
    checks = metrics["consistency_checks"]
    lines.extend(table(["检查", "结果"], [
        [label.replace("_", " "), "通过" if value is True else "不适用" if value is None else "不一致"]
        for label, value in checks.items()
    ]))
    lines.extend([
        "",
        f"## {implications_section}. 对下一阶段算法研究的含义",
        "",
        "1. 以完整会话或时间块做训练/验证切分；不能随机打散相邻锚点，因为它们共享历史和未来观察窗口。",
        "2. 评估时只用锚点时刻已知的上下文和库存。后台标签页/目录库存可作候选或上下文，但不当作用户到达；所有 outcomes 严格留在标签侧。",
        "3. 分别检查 inputCoverage 和 outcomeCoverage；旧数据 unknown 不代表全部观测无效，也不能冒充完整状态。把缺口、会话边界及截断保留为未知，不能变成负例或 `__stay__`。",
        "4. 先建立受可达库存约束的近期访问与一阶转移基线，再根据时间外验证收益决定是否增加个性化在线学习。长尾目标应报告 coverage/recall，而不是只看热门目标准确率。",
        "5. 这份报告不把观察到的后续目标称为唯一最终意图；它只是可用于排序评估的多步、带时间顺序的结果。",
        "",
    ])
    return "\n".join(lines)


def write_private(path: Path, contents: str) -> None:
    path.write_text(contents, encoding="utf-8")
    os.chmod(path, 0o600)


def write_concrete_target_arrivals(path: Path, targets: list[dict[str, Any]], total_arrivals: int) -> None:
    """Write the explicit target view only after --include-targets was requested."""
    fields = ["rank", "target_kind", "target_name", "locator", "origin", "bundle_id", "arrival_count", "arrival_share_percent"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for target in targets:
            writer.writerow({
                "rank": target["rank"],
                "target_kind": target["kind"],
                "target_name": target["name"],
                "locator": target["locator"] or "",
                "origin": target["origin"] or "",
                "bundle_id": target["bundle_id"] or "",
                "arrival_count": target["arrivals"],
                "arrival_share_percent": share(target["arrivals"], total_arrivals),
            })
    os.chmod(path, 0o600)


def main() -> None:
    args = parse_args()
    snapshot = args.snapshot.expanduser().resolve()
    output = args.output.expanduser().resolve()
    if output.exists():
        error(f"output already exists; choose a new run directory: {output}")
    integrity = verify_snapshot(snapshot)
    try:
        manifest = json.loads((snapshot / "manifest.json").read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        error(f"invalid manifest.json: {exc.msg}")
    if not isinstance(manifest, dict):
        error("manifest.json must contain a JSON object")

    events, detailed_targets = analyze_events(snapshot / "events.jsonl")
    samples = analyze_samples(snapshot / "navigation-samples.jsonl")
    target_history_count = sum(1 for _ in read_json_lines(snapshot / "target-history.jsonl"))
    metrics: dict[str, Any] = {
        "analysis_version": ANALYSIS_VERSION,
        "generated_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "input": {
            "snapshot_name": snapshot.name,
            "integrity": integrity,
            "schema_version": manifest.get("schemaVersion"),
            "target_eligibility_policy_version": manifest.get("targetEligibilityPolicyVersion"),
            "prediction_horizon_ms": manifest.get("predictionHorizonMs"),
            "events_excluded_from_samples": manifest.get("eventsExcludedFromSamples"),
            "concrete_target_details_included": args.include_targets,
        },
        "events": events,
        "samples": samples,
        "target_history_records": target_history_count,
        "consistency_checks": validate_snapshot_counts(events, samples, target_history_count, manifest),
    }
    # The documented training root may not exist yet.  Python applies ``mode``
    # only to the leaf when it creates intermediate directories, so explicitly
    # tighten the run parent before creating the new immutable run directory.
    output.parent.mkdir(parents=True, mode=0o700, exist_ok=True)
    os.chmod(output.parent, 0o700)
    output.mkdir(mode=0o700)
    os.chmod(output, 0o700)
    write_private(output / "metrics.json", json.dumps(metrics, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    write_private(output / "descriptive-analysis.md", report(
        metrics, detailed_targets if args.include_targets else None, args.target_limit
    ))
    if args.include_targets:
        write_concrete_target_arrivals(
            output / "concrete-target-arrivals.csv",
            detailed_targets,
            events["observed_arrivals"]["independent_context_anchors"],
        )
    print(f"Wrote {'detailed' if args.include_targets else 'aggregate-only'} analysis to: {output}")


if __name__ == "__main__":
    main()
