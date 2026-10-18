#!/usr/bin/env python3
"""Compare causal multi-step foreground statistics with ``statistics/none``.

This is an offline-only research replay.  The fixed test membership comes from
``offline-batch-split-v1``.  Hyperparameters are selected on an earlier
time-forward slice of the training membership; the fixed test is then scored
once after retraining on all mature training members.  A separate prequential
diagnostic may consume mature test labels, but it never selects a model.

Only foreground context changes are used to build a sequence.  Resolution
updates are refinements, not arrivals.  A real observation boundary clears the
sequence; known sleep/lock pauses remain resumable only when the collector's
continuity protocol marks them as such.  No inventory, background state,
candidate list, page body, or future target metadata is passed to a model.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import statistics
import sys
import time
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import context_statistics
import evidence_cohort
import evaluation_contract as scoring
import multistep_statistical_models as sequence_models
import run_offline_batch_rankers as offline
from known_inactivity import annotate_events
from navigation_hierarchy import ENTRY, disposition
from semantic_annotations import annotate_target


VERSION = "multistep-statistical-v1"
PRIVATE_ROOT = Path.home() / "Library/Application Support/SmartApp/training"
BASELINE_NAME = "statistics/none"
BASELINE_CONFIG = {
    "id": "none-0025",
    "family": "context_statistics",
    "mode": "none",
    "profile": None,
    "floor": 1.0,
    "power": 1.0,
    "semantic_strength": 64.0,
    "global_strength": 1.0,
    "recency_mix": 0.5,
    "half_life_hours": 24.0,
}
ACTIONABLE_KINDS = {"app", "page", "directory", "file"}


@dataclass(frozen=True)
class PreparedRow:
    observation: Any
    sequence: tuple[str, ...]
    segment_id: int | None


class NoBackground:
    """Fail loudly if the no-context baseline tries to consume state."""

    def factors(self, *args, **kwargs):
        raise AssertionError("statistics/none must not read background features")


class StatisticsNoneModel:
    """Adapter for the existing one-step unordered ``statistics/none`` model."""

    def __init__(self):
        self.memory = context_statistics.StatisticalMemory()

    @staticmethod
    def query(row: PreparedRow) -> dict:
        observation = row.observation
        current = observation.current
        if current is None:
            raise ValueError("A baseline query needs an admitted current target")
        annotation = annotate_target(current.meta)
        index = int(observation.input_event_index)
        return {
            "position": int(observation.position),
            "index": index,
            "timestamp": float(observation.input_timestamp_ms),
            "origin": current.key,
            "target": current.meta,
            "annotation": annotation,
            "current_sequence": [current.key],
            "background_through_index": index,
            "inventory_through_index": index,
        }

    @staticmethod
    def record(row: PreparedRow) -> dict:
        observation = row.observation
        current, truth = observation.current, observation.truth
        if current is None or truth is None:
            raise ValueError("A baseline record needs current and truth targets")
        return {
            "origin": current.key,
            "annotation": annotate_target(current.meta),
            "label": truth.key,
            "outcomeID": observation.outcome_event_id,
            "outcomeTimestamp": float(observation.outcome_timestamp_ms),
            "weight": 1.0,
            "source": "natural_or_selected_foreground",
            "destination": annotate_target(truth.meta)["identity"],
            "observedStep": 0,
        }

    def learn(self, row: PreparedRow) -> bool:
        query = self.query(row)
        before = len(self.memory.training)
        self.memory.learn(query, self.record(row), int(row.observation.completed_event_index))
        return len(self.memory.training) > before

    def predict(self, row: PreparedRow, home: str | None) -> dict:
        query = self.query(row)
        return self.memory.prepare(query, home, NoBackground()).predict(BASELINE_CONFIG)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def write_private_json(path: Path, value: Any) -> None:
    if path.exists():
        raise ValueError(f"Immutable artifact already exists: {path}")
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")
    path.chmod(0o600)


def write_private_jsonl(path: Path, values: list[dict]) -> None:
    if path.exists():
        raise ValueError(f"Immutable artifact already exists: {path}")
    with path.open("x", encoding="utf-8") as stream:
        for value in values:
            stream.write(json.dumps(value, ensure_ascii=False, sort_keys=True))
            stream.write("\n")
    path.chmod(0o600)


def contains_key(value: Any, key: str) -> bool:
    if isinstance(value, dict):
        return key in value or any(contains_key(item, key) for item in value.values())
    if isinstance(value, list):
        return any(contains_key(item, key) for item in value)
    return False


def build_foreground_sequences(events: list[dict], history: Any) -> tuple[dict[int, tuple[tuple[str, ...], int]], dict]:
    """Return as-of sequences keyed by event index, without crossing boundaries."""
    annotated = annotate_events(events)
    contexts: dict[int, tuple[tuple[str, ...], int]] = {}
    sequence: list[str] = []
    session = None
    segment_id = -1
    audit = Counter()

    for index, event in enumerate(annotated):
        if event.get("sessionID") != session:
            session = event.get("sessionID")
            sequence = []
            segment_id += 1
            audit["session_boundaries"] += 1

        if (event.get("kind") in {"capture.gap", "session.end"}
                and event.get("continuityDisposition") == "boundary"):
            sequence = []
            segment_id += 1
            audit["observation_boundaries"] += 1

        if event.get("kind") != "context.change":
            continue
        data = event.get("data") or {}
        reason = data.get("reason")
        if reason == "resolution_update":
            audit["resolution_updates_skipped"] += 1
            if event.get("continuityDisposition") == "boundary":
                sequence = []
                segment_id += 1
                audit["unlinked_resolution_boundaries"] += 1
            continue

        target = history.metadata_at(data.get("targetID"), index)
        action = offline.action_from_meta(target, catalog_only=False)
        if action is None or action.kind not in ACTIONABLE_KINDS:
            sequence = []
            segment_id += 1
            audit["unresolved_or_ineligible_context_boundaries"] += 1
            continue

        sequence.append(action.key)
        contexts[index] = (tuple(sequence), segment_id)
        audit["foreground_arrivals"] += 1
        if len(sequence) > 1:
            audit["arrivals_with_history"] += 1
        if len(sequence) > audit["max_sequence_length"]:
            audit["max_sequence_length"] = len(sequence)

    audit["segments"] = segment_id + 1
    audit["sequence_contexts"] = len(contexts)
    audit["sequence_length_histogram"] = dict(sorted(Counter(len(value[0]) for value in contexts.values()).items()))
    return contexts, dict(audit)


def prepare_rows(observations: list[Any], contexts: dict[int, tuple[tuple[str, ...], int]]) -> list[PreparedRow]:
    result = []
    for observation in observations:
        context = contexts.get(int(observation.input_event_index))
        if context is None:
            result.append(PreparedRow(observation, (), None))
            continue
        sequence, segment_id = context
        if observation.current is not None and sequence[-1] != observation.current.key:
            raise ValueError("As-of sequence current target disagrees with split materialization")
        result.append(PreparedRow(observation, sequence, segment_id))
    return result


def load_dataset(snapshot: Path, split_dir: Path) -> tuple[list[PreparedRow], list[PreparedRow], Any, dict, dict]:
    train, test, history, input_audit = offline.load_observations(snapshot, split_dir)
    events = read_jsonl(snapshot / "events.jsonl")
    samples = read_jsonl(snapshot / "navigation-samples.jsonl")
    manifest = read_json(snapshot / "manifest.json")
    if manifest.get("stateCoverageVersion") != "foreground-coverage-v2":
        raise ValueError("The multi-step replay requires the foreground-only coverage contract")
    if any(sample.get("inventoryEventIDs") for sample in samples):
        raise ValueError("Inventory references are not allowed in this research input")
    if any(contains_key(event, "candidateTargetIDs") for event in events):
        raise ValueError("Collected candidate lists are not allowed in this research input")
    contexts, sequence_audit = build_foreground_sequences(events, history)
    train_rows = prepare_rows(train, contexts)
    test_rows = prepare_rows(test, contexts)
    if train_rows and test_rows:
        if max(r.observation.completed_event_index for r in train_rows) >= min(r.observation.input_event_index for r in test_rows):
            raise ValueError("Fixed train/test split permits maturity leakage")
    input_audit = {
        **input_audit,
        "snapshot_manifest": {
            "schemaVersion": manifest.get("schemaVersion"),
            "sampleSchemaVersion": manifest.get("sampleSchemaVersion"),
            "stateCoverageVersion": manifest.get("stateCoverageVersion"),
            "events": manifest.get("events"),
            "samples": manifest.get("samples"),
        },
        "samples_with_inventory_references": 0,
        "events_with_candidate_lists": 0,
        "train_rows_without_sequence_context": sum(not row.sequence for row in train_rows),
        "test_rows_without_sequence_context": sum(not row.sequence for row in test_rows),
    }
    return train_rows, test_rows, history, input_audit, sequence_audit


def navigation_training_row(row: PreparedRow) -> bool:
    observation = row.observation
    if observation.current is None or observation.truth is None or not row.sequence:
        return False
    if observation.current.kind not in ACTIONABLE_KINDS or observation.truth.kind not in ACTIONABLE_KINDS:
        return False
    return disposition(observation.truth.key, observation.current.meta) == "navigation"


def sequence_record(row: PreparedRow) -> dict:
    observation = row.observation
    if observation.current is None or observation.truth is None:
        raise ValueError("Sequence training row lacks action metadata")
    return {
        "sequence": row.sequence,
        "origin": observation.current.key,
        "label": observation.truth.key,
        "input_timestamp": float(observation.input_timestamp_ms),
        "outcome_timestamp": float(observation.outcome_timestamp_ms),
        "outcome_id": observation.outcome_event_id,
        "segment_id": row.segment_id,
    }


def home_key(history: Any, event_index: int) -> str | None:
    """Resolve only an already-observed ChatGPT entry action at this cutoff."""
    catalog = history.catalog_at(event_index)
    for action in sorted(catalog.values(), key=lambda value: value.key):
        if action.meta.get("locator") == ENTRY:
            return action.key
    return None


def prediction_for(model_kind: str, model: Any, row: PreparedRow, history: Any) -> dict:
    observation = row.observation
    if observation.current is None or not row.sequence:
        return {"predictions": [], "rankable": []}
    if model_kind == "baseline":
        return model.predict(row, home_key(history, int(observation.input_event_index)))
    query = {
        "sequence": row.sequence,
        "origin": observation.current.key,
        "timestamp": float(observation.input_timestamp_ms),
        "segment_id": row.segment_id,
    }
    scores = model.scores(query)
    ranked = sorted(scores, key=lambda label: (-scores[label], label))
    ranked = [label for label in ranked if disposition(label, observation.current.meta) == "navigation"]
    return {"predictions": ranked[:4], "rankable": ranked}


def timing_summary(values: list[float]) -> dict:
    ordered = sorted(values)
    return {
        "n": len(values),
        "median_ms": statistics.median(values) if values else 0.0,
        "p95_ms": ordered[int(0.95 * (len(ordered) - 1))] if values else 0.0,
        "total_ms": math.fsum(values),
    }


def replay(model_kind: str, config: dict | None, training_rows: list[PreparedRow],
           query_rows: list[PreparedRow], history: Any) -> tuple[dict[int, dict], dict]:
    model = StatisticsNoneModel() if model_kind == "baseline" else sequence_models.build(config or {})
    stream = [row for row in training_rows if navigation_training_row(row)]
    stream.sort(key=lambda row: (row.observation.completed_event_index,
                                 row.observation.input_event_index, row.observation.position))
    cursor = 0
    outputs: dict[int, dict] = {}
    learned = 0
    predict_times: list[float] = []
    learn_times: list[float] = []
    for row in sorted(query_rows, key=lambda item: (item.observation.input_event_index, item.observation.position)):
        cutoff = int(row.observation.input_event_index)
        now = float(row.observation.input_timestamp_ms)
        while cursor < len(stream) and stream[cursor].observation.completed_event_index < cutoff:
            mature = stream[cursor]
            if mature.observation.input_event_index >= cutoff:
                raise ValueError("A future input entered the training stream")
            if mature.observation.outcome_timestamp_ms >= now:
                raise ValueError("A future timestamp entered the training stream")
            started = time.perf_counter()
            model.learn(mature) if model_kind == "baseline" else model.learn(sequence_record(mature))
            learn_times.append((time.perf_counter() - started) * 1000.0)
            learned += 1
            cursor += 1
        started = time.perf_counter()
        outputs[int(row.observation.position)] = prediction_for(model_kind, model, row, history)
        predict_times.append((time.perf_counter() - started) * 1000.0)
    return outputs, {
        "model_kind": model_kind,
        "stream_rows": len(stream),
        "learned_rows": learned,
        "learn": timing_summary(learn_times),
        "predict": timing_summary(predict_times),
        "queries": len(query_rows),
    }


def evaluation_rows(rows: list[PreparedRow], evidence_name: str) -> list[dict]:
    observations, evaluations = [], []
    for row in rows:
        observation = row.observation
        current = observation.current.meta if observation.current is not None else {}
        truth = observation.truth.meta if observation.truth is not None else {}
        current_parent = observation.current_parent.meta if observation.current_parent is not None else None
        truth_parent = observation.truth_parent.meta if observation.truth_parent is not None else None
        observations.append({
            "position": int(observation.position),
            "index": int(observation.input_event_index),
            "ready_index": int(observation.completed_event_index),
            "target": current,
            "parent_action": current_parent,
        })
        evaluations.append({
            "position": int(observation.position),
            "sample_id": observation.sample_id,
            "truth_action": observation.truth.key if observation.truth is not None else None,
            "truth_metadata": truth,
            "split": evidence_name,
            "evidence_grade": evidence_name,
            "truth_parent_metadata": truth_parent,
            "prior_evidence": {
                "exact_target": observation.has_prior_exact_action,
                "parent_action": observation.has_prior_parent_action,
            },
        })
    return scoring.prepare_evaluation({"observations": observations, "evaluation": evaluations})


def score(rows: list[PreparedRow], outputs: dict[int, dict], evidence_name: str) -> tuple[dict, list[dict]]:
    audited = evaluation_rows(rows, evidence_name)
    return scoring.measure(audited, outputs), audited


def inner_split(rows: list[PreparedRow]) -> tuple[list[PreparedRow], list[PreparedRow], list[PreparedRow], dict]:
    boundaries = sorted({row.observation.input_event_index for row in rows})
    options = []
    for boundary in boundaries[1:]:
        fit = [row for row in rows if row.observation.completed_event_index < boundary]
        validation = [row for row in rows if row.observation.input_event_index >= boundary]
        guard = [row for row in rows if row.observation.input_event_index < boundary <= row.observation.completed_event_index]
        if not fit or not validation:
            continue
        if max(r.observation.completed_event_index for r in fit) >= min(r.observation.input_event_index for r in validation):
            continue
        usable = len(fit) + len(validation)
        ratio_error = abs(len(fit) / usable - 0.75)
        options.append(((ratio_error, -usable, boundary), fit, validation, guard, boundary))
    if not options:
        raise ValueError("No causal inner validation split exists")
    _, fit, validation, guard, boundary = min(options, key=lambda item: item[0])
    return fit, validation, guard, {
        "fit": len(fit),
        "validation": len(validation),
        "guard": len(guard),
        "boundary_event_index": boundary,
        "rule": "fit labels complete before validation inputs; guard rows overlap the boundary",
    }


def selection_key(metrics: dict, config_id: str) -> tuple:
    return (
        -metrics["top4_hits"],
        -metrics["exact"]["top4_hits"],
        -(metrics["rank_score"]["mean"] or 0.0),
        -metrics["top2_hits"],
        -metrics["exact"]["top2_hits"],
        config_id,
    )


def select_configurations(train_rows: list[PreparedRow], history: Any, configs: list[dict]) -> tuple[dict, dict]:
    fit, validation, guard, split = inner_split(train_rows)
    results = {}
    audited = evaluation_rows(validation, "inner_validation")
    for config in configs:
        outputs, timing = replay("sequence", config, fit, validation, history)
        metrics = scoring.measure(audited, outputs)
        leaderboards = scoring.score_leaderboards(audited, {config["id"]: outputs})
        primary_board, primary_metrics = scoring.primary_leaderboard_metric(leaderboards, config["id"])
        results[config["id"]] = {
            "config": config,
            "metrics": metrics,
            "leaderboards": leaderboards,
            "selection_leaderboard": primary_board,
            "selection_metrics": primary_metrics,
            "timing": timing,
        }
    selected = min(results.values(), key=lambda item: selection_key(item["selection_metrics"], item["config"]["id"]))
    return {"split": split, "selected": selected, "results": results}, {
        "fit": fit,
        "validation": validation,
        "guard": guard,
    }


def paired_help_harm(rows: list[PreparedRow], baseline: dict[int, dict], alternative: dict[int, dict],
                     arm: str, model_name: str, audited: list[dict] | None = None) -> tuple[dict, list[dict]]:
    audited = evaluation_rows(rows, arm) if audited is None else audited
    baseline_metric = scoring.measure(audited, baseline)
    alternative_metric = scoring.measure(audited, alternative)
    scoring.require_comparable(baseline_metric, alternative_metric)
    levels = {level: {str(k): Counter() for k in (1, 2, 4)} for level in ("accepted", "exact")}
    rank_counts = Counter()
    details = []
    for row in audited:
        if not scoring.eligible(row):
            continue
        position = row["position"]
        before = scoring.hits(row, scoring.prediction_at(baseline, position))
        after = scoring.hits(row, scoring.prediction_at(alternative, position))
        base_rank = scoring.ranked_gain(row, scoring.prediction_at(baseline, position))
        alt_rank = scoring.ranked_gain(row, scoring.prediction_at(alternative, position))
        rank_delta = alt_rank["score"] - base_rank["score"]
        rank_counts["better" if rank_delta > 1e-12 else "worse" if rank_delta < -1e-12 else "tied"] += 1
        detail = {
            "arm": arm,
            "model": model_name,
            "sample_id": row["sample_id"],
            "position": position,
            "rank_score_delta": rank_delta,
        }
        for level in ("accepted", "exact"):
            detail[level] = {}
            for k in (1, 2, 4):
                before_hit, after_hit = before[level][k], after[level][k]
                counts = levels[level][str(k)]
                counts["helped"] += int(after_hit and not before_hit)
                counts["hurt"] += int(before_hit and not after_hit)
                counts["both_correct"] += int(after_hit and before_hit)
                counts["both_wrong"] += int(not after_hit and not before_hit)
                counts["baseline_hits"] += int(before_hit)
                counts["alternative_hits"] += int(after_hit)
                detail[level][str(k)] = {"baseline": before_hit, "alternative": after_hit,
                                         "helped": bool(after_hit and not before_hit),
                                         "hurt": bool(before_hit and not after_hit)}
        details.append(detail)
    return {
        "n": sum(1 for row in audited if scoring.eligible(row)),
        "accepted": {key: {**value, "net": value["helped"] - value["hurt"]}
                      for key, value in levels["accepted"].items()},
        "exact": {key: {**value, "net": value["helped"] - value["hurt"]}
                  for key, value in levels["exact"].items()},
        "rank_score": {**rank_counts, "net": rank_counts["better"] - rank_counts["worse"]},
        "baseline_comparison_key": baseline_metric["comparison_key"],
        "alternative_comparison_key": alternative_metric["comparison_key"],
    }, details


def prospective_evidence_metrics(rows: list[PreparedRow], training_rows: list[PreparedRow],
                                 outputs: dict[str, dict[int, dict]], arm: str) -> tuple[dict, list[dict]]:
    """Score all models on one pre-cutoff, baseline-anchored evidence cohort."""
    audited = evaluation_rows(rows, arm)
    cohort = evidence_cohort.build(rows, training_rows, outputs[BASELINE_NAME])
    groups = {"evidence_total": ["strong", "repeated", "single"]}
    for grade_name in ("strong", "repeated", "single", "none"):
        groups["evidence_" + grade_name] = [grade_name]
    metrics: dict[str, dict[str, dict]] = {}
    for group_name, grade_names in groups.items():
        subset = [row for row in audited
                  if cohort.grade_by_position.get(int(row["position"])) in grade_names]
        metrics[group_name] = {name: scoring.measure(subset, model_outputs)
                               for name, model_outputs in outputs.items()}
        scoring.require_comparable(*metrics[group_name].values())

    eligible_grade_counts = Counter(
        cohort.grade_by_position[int(row["position"])]
        for row in audited if scoring.eligible(row)
    )
    audit = {
        **cohort.audit,
        "eligible_grade_counts": {name: eligible_grade_counts.get(name, 0)
                                   for name in evidence_cohort.GRADES},
        "eligible_evidence_total": sum(eligible_grade_counts.get(name, 0)
                                        for name in ("strong", "repeated", "single")),
    }
    paired: dict[str, dict] = {}
    details: list[dict] = []
    evidence_rows = [row for row in audited
                     if cohort.grade_by_position.get(int(row["position"])) != "none"]
    for name, alternative in outputs.items():
        if name == BASELINE_NAME:
            continue
        summary, model_details = paired_help_harm(
            rows, outputs[BASELINE_NAME], alternative, arm, name, audited=evidence_rows
        )
        paired[name] = summary
        details.extend(model_details)
    return {"audit": audit, "metrics": metrics, "paired": paired}, details


def percent(value: float | None) -> str:
    return "—" if value is None else f"{100.0 * value:.1f}%"


def metric_line(name: str, metrics: dict) -> str:
    exact = metrics["exact"]
    return (f"| {name} | {metrics['n']} | "
            f"{metrics['top1_hits']}/{metrics['n']} ({percent(metrics['top1'])}) | "
            f"{metrics['top2_hits']}/{metrics['n']} ({percent(metrics['top2'])}) | "
            f"{metrics['top4_hits']}/{metrics['n']} ({percent(metrics['top4'])}) | "
            f"{exact['top1_hits']}/{metrics['n']} ({percent(exact['top1'])}) | "
            f"{exact['top2_hits']}/{metrics['n']} ({percent(exact['top2'])}) | "
            f"{exact['top4_hits']}/{metrics['n']} ({percent(exact['top4'])}) | "
            f"{percent(metrics['rank_score']['mean'])} | "
            f"{percent(metrics['prediction_coverage'])} | "
            f"{percent(metrics['rankable_truth_coverage'])} |")


def leaderboard_order(metrics: dict[str, dict]) -> list[tuple[str, dict]]:
    """Sort by the declared useful Top4 value, then position and exactness."""
    return sorted(metrics.items(), key=lambda item: (
        -(item[1]["actionable_topk"]["top4_gain"] or 0.0),
        -(item[1]["rank_score"]["mean"] or 0.0),
        -item[1]["exact"]["top4_hits"],
        -item[1]["top4_hits"],
        item[0],
    ))


def leaderboard_metric_line(rank: int, name: str, metrics: dict) -> str:
    exact, usable = metrics["exact"], metrics["actionable_topk"]
    n = metrics["n"]
    return (f"| {rank} | {name} | {n} | "
            f"{metrics['top1_hits']}/{n} ({percent(metrics['top1'])}) | "
            f"{metrics['top2_hits']}/{n} ({percent(metrics['top2'])}) | "
            f"{metrics['top4_hits']}/{n} ({percent(metrics['top4'])}) | "
            f"{exact['top1_hits']}/{n} ({percent(exact['top1'])}) | "
            f"{exact['top2_hits']}/{n} ({percent(exact['top2'])}) | "
            f"{exact['top4_hits']}/{n} ({percent(exact['top4'])}) | "
            f"{percent(usable['top4_gain'])} | "
            f"{usable['top4_parent_fallback_used_n']}/{n} ({percent(usable['top4_parent_fallback_used'])}) | "
            f"{percent(metrics['rank_score']['mean'])} | "
            f"{percent(metrics['prediction_coverage'])} | "
            f"{percent(metrics['rankable_truth_coverage'])} |")


def append_leaderboard_tables(lines: list[str], arm: str, leaderboards: dict[str, dict], audit: dict) -> None:
    """Render both fixed-membership scoreboards without exposing target identities."""
    lines += [
        f"## {arm}：两套评分榜单",
        "",
        "排序先看“可用入口 Top4 收益”，再看位置敏感的 RankScore、精确 Top4 和服务接受 Top4。具体网页/目录命中收益是 1；只有预测前已确认、可启动、且不是当前前台所属应用的正确上级应用入口才给 0.6。它不会写成精确命中，也不会改变服务接受或精确 Top1/Top2/Top4。",
        "",
        "### 主榜：预测前已有可用入口",
        "",
        f"本榜 {audit['prior_supported_n']}/{audit['legal_n']} 条合法样本。成员条件是：真实后续目标本身在预测前已确认，或它的正确上级应用入口在预测前已确认且可启动。这个条件只在事后分组评分时使用，不喂给模型、不筛模型输出，也不能作为线上是否展示的开关。",
        "",
        "| 排名 | 模型 | 合法 n | 接受 Top1 | 接受 Top2 | 接受 Top4 | 精确 Top1 | 精确 Top2 | 精确 Top4 | 可用入口 Top4 收益 | 上级入口实际帮助 | RankScore@4 | 输出覆盖 | 可排序真值覆盖 |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for rank, (name, metrics) in enumerate(leaderboard_order(leaderboards[scoring.LEADERBOARD_PRIOR]), 1):
        lines.append(leaderboard_metric_line(rank, name, metrics))
    lines += [
        "",
        "### 完整榜：全部合法样本",
        "",
        f"本榜保留全部 {audit['legal_n']} 条合法样本，其中 {audit['prior_unsupported_n']} 条在预测前没有已确认的具体入口或合格上级入口；它们反映覆盖缺口，保留用于产品评估，但不作为模型排序的主依据。",
        "",
        "| 排名 | 模型 | 合法 n | 接受 Top1 | 接受 Top2 | 接受 Top4 | 精确 Top1 | 精确 Top2 | 精确 Top4 | 可用入口 Top4 收益 | 上级入口实际帮助 | RankScore@4 | 输出覆盖 | 可排序真值覆盖 |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for rank, (name, metrics) in enumerate(leaderboard_order(leaderboards[scoring.LEADERBOARD_ALL]), 1):
        lines.append(leaderboard_metric_line(rank, name, metrics))
    lines += ["", f"榜单成员审计：{json.dumps(audit, ensure_ascii=False, sort_keys=True)}", ""]


def evidence_metric_line(arm: str, name: str, group: str, metrics: dict) -> str:
    exact = metrics["exact"]
    return (f"| {arm} | {name} | {group} | {metrics['n']} | "
            f"{metrics['top1_hits']}/{metrics['n']} ({percent(metrics['top1'])}) | "
            f"{metrics['top2_hits']}/{metrics['n']} ({percent(metrics['top2'])}) | "
            f"{metrics['top4_hits']}/{metrics['n']} ({percent(metrics['top4'])}) | "
            f"{exact['top1_hits']}/{metrics['n']} ({percent(exact['top1'])}) | "
            f"{exact['top2_hits']}/{metrics['n']} ({percent(exact['top2'])}) | "
            f"{exact['top4_hits']}/{metrics['n']} ({percent(exact['top4'])}) | "
            f"{percent(metrics['rank_score']['mean'])} | "
            f"{percent(metrics['prediction_coverage'])} | "
            f"{percent(metrics['rankable_truth_coverage'])} |")


def write_report(path: Path, plan: dict, selection: dict, fixed: dict, prequential: dict,
                 leaderboards: dict, leaderboard_audits: dict, evidence_metrics: dict,
                 paired: dict, sequence_audit: dict, input_audit: dict) -> None:
    lines = [
        "# 多步前台时间序列统计模型离线研究",
        "",
        "本 run 只用于 SmartApp 私有离线研究，不修改采集器、线上模型或运行数据库。模型输入是预测截点前确认的前台到达序列；`resolution_update` 不新增到达，真正观察边界切断序列。未使用后台状态、打开库存、独立候选清单或页面正文。",
        "",
        f"固定数据 split：train {input_audit.get('train_members', '—')}，test {input_audit.get('test_members', '—')}；测试只在内部时间前向选参后评分一次。主对照是现有单步无序列 `{BASELINE_NAME}`；prequential 只作为另列诊断，不参与选参。",
        "",
        "服务接受列按统一评分入口归并 ChatGPT 服务，精确列保持具体目标诊断；`RankScore@4` 同样来自统一评分入口。覆盖率分别是有输出覆盖和真实目标出现在模型可排序输出中的覆盖。",
        "",
    ]
    append_leaderboard_tables(lines, "固定测试", leaderboards["fixed_test"], leaderboard_audits["fixed_test"])
    lines += [
        "## Prequential 诊断（非固定测试）",
        "",
        "每个测试截点先预测，再只消费已经成熟的前台后续标签；因此该表反映在线增量回放，不是冻结模型的独立测试，也不用于选择模型。",
        "",
    ]
    append_leaderboard_tables(lines, "Prequential", leaderboards["prequential"], leaderboard_audits["prequential"])
    lines += ["", "## 预测前可识别的历史证据 cohort", "",
              "主表仍是全部合法样本。以下 cohort 在预测截点先由 `statistics/none` 的首个输出和当时已成熟的同起点后续历史确定，再把同一成员交给所有模型评分；不读取真实下一步、不按命中筛样。强支持为支持至少 3 次且占该起点全部成熟后续的 70% 以上，重复支持为至少 2 次但未达强支持，一次支持为 1 次。", "",
              "| arm | 模型 | cohort | 合法 n | 接受 Top1 | 接受 Top2 | 接受 Top4 | 精确 Top1 | 精确 Top2 | 精确 Top4 | RankScore@4 | 输出覆盖 | 可排序真值覆盖 |",
              "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for arm in ("fixed_test", "prequential"):
        result = evidence_metrics[arm]
        lines.append(f"| {arm} | — | audit | {result['audit']['eligible_evidence_total']} | — | — | — | — | — | — | — | — | — |")
        for group in ("evidence_total", "evidence_strong", "evidence_repeated", "evidence_single", "evidence_none"):
            for name, metrics in result["metrics"][group].items():
                lines.append(evidence_metric_line(arm, name, group, metrics))
    lines += ["", "cohort audit（只保存聚合计数与成员哈希）："]
    for arm in ("fixed_test", "prequential"):
        lines.append(f"- {arm}: {json.dumps(evidence_metrics[arm]['audit'], ensure_ascii=False, sort_keys=True)}")
    lines += ["", "## evidence cohort 相对 `statistics/none` 的样本级 help/harm", "",
              "以下仅在同一预测前 evidence_total 成员内逐样本比较；详情写入私有 JSONL，不保存预测列表或候选列表。", "",
              "| arm | 模型 | n | accepted T1 help/hurt | accepted T2 help/hurt | accepted T4 help/hurt | exact T1 help/hurt | exact T2 help/hurt | exact T4 help/hurt | RankScore better/worse/tied |",
              "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for arm in ("fixed_test", "prequential"):
        for name, value in evidence_metrics[arm]["paired"].items():
            accepted, exact, rank = value["accepted"], value["exact"], value["rank_score"]
            cells = [f"{accepted[str(k)]['helped']}/{accepted[str(k)]['hurt']}" for k in (1, 2, 4)]
            cells += [f"{exact[str(k)]['helped']}/{exact[str(k)]['hurt']}" for k in (1, 2, 4)]
            cells.append(f"{rank['better']}/{rank['worse']}/{rank['tied']}")
            lines.append(f"| {arm} | {name} | {value['n']} | " + " | ".join(cells) + " |")
    lines += ["", "## 相对 `statistics/none` 的样本级 help/harm", "",
              "helped/hurt 按同一合法样本逐条比较，不按模型命中后筛选分母；`rank_score` 的 better/worse/tied 也按样本保留。详情只保存于私有 JSONL，不保存模型候选列表。", "",
              "| arm | 模型 | n | accepted T1 help/hurt | accepted T2 help/hurt | accepted T4 help/hurt | exact T1 help/hurt | exact T2 help/hurt | exact T4 help/hurt | RankScore better/worse/tied |",
              "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for key, value in paired.items():
        arm, name = key.split("::", 1)
        a, e = value["accepted"], value["exact"]
        r = value["rank_score"]
        cells = [f"{a[str(k)]['helped']}/{a[str(k)]['hurt']}" for k in (1, 2, 4)]
        cells += [f"{e[str(k)]['helped']}/{e[str(k)]['hurt']}" for k in (1, 2, 4)]
        cells.append(f"{r['better']}/{r['worse']}/{r['tied']}")
        lines.append(f"| {arm} | {name} | {value['n']} | " + " | ".join(cells) + " |")
    lines += [
        "",
        "## 因果与数据审计",
        "",
        f"- 序列审计：{json.dumps(sequence_audit, ensure_ascii=False, sort_keys=True)}",
        f"- 输入审计：{json.dumps(input_audit, ensure_ascii=False, sort_keys=True)}",
        f"- 内部选参：选择 `{selection['selected']['config']['id']}`；主标准为 `{selection['selected']['selection_leaderboard']}` 榜的服务接受 Top4、精确 Top4、RankScore、Top2，再按固定 ID 平局。",
        f"- 选参边界：{json.dumps(selection['split'], ensure_ascii=False, sort_keys=True)}",
        f"- 评分契约摘要：rules_sha256={scoring.contract()['rules_sha256']}，所有表均由 `evaluation_contract.py` 重算。",
        "- 已查看的本机数据属于开发/探索性回放；结果不构成新用户数据上的独立泛化保证，也不代表已经部署。",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")
    path.chmod(0o600)
def artifact_hashes(output: Path) -> dict[str, str]:
    result = {}
    for path in sorted(output.rglob("*")):
        if path.is_file() and path.name not in {"artifact-hashes.json", "run.json"}:
            result[str(path.relative_to(output))] = sha256(path)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config-ids", nargs="*", help="Optional research configuration allowlist")
    args = parser.parse_args()

    snapshot = args.snapshot.expanduser().resolve()
    split_dir = args.split.expanduser().resolve()
    output = args.output.expanduser().resolve()
    private = PRIVATE_ROOT.resolve()
    if output == private or not output.is_relative_to(private):
        parser.error("Output must be a new private SmartApp training directory")
    if output.exists():
        parser.error("Output directory must not already exist")

    configs = sequence_models.configurations()
    if args.config_ids:
        wanted = set(args.config_ids)
        configs = [config for config in configs if config["id"] in wanted]
        if not configs or len(wanted) != len(configs):
            parser.error("Unknown or empty --config-ids")

    os.umask(0o077)
    output.mkdir(parents=True, mode=0o700)
    output.chmod(0o700)

    train_rows, test_rows, history, input_audit, sequence_audit = load_dataset(snapshot, split_dir)
    all_rows = train_rows + test_rows
    selection, inner_rows = select_configurations(train_rows, history, configs)
    fit_rows, validation_rows = inner_rows["fit"], inner_rows["validation"]

    plan = {
        "version": VERSION,
        "scope": "local_offline_statistical_research_only",
        "snapshot": str(snapshot),
        "split": str(split_dir),
        "foreground_only": True,
        "input": "as-of confirmed foreground arrival sequence; no background state or inventory",
        "sequence_boundaries": "session and true observation boundaries reset; paired known sleep/lock pause can continue only through collector continuity evidence",
        "resolution_updates_are_arrival": False,
        "candidate_lists_persisted": False,
        "model_outputs_persisted": False,
        "runtime_or_online_model_modified": False,
        "baseline": {"name": BASELINE_NAME, "config": BASELINE_CONFIG,
                     "sequence_length": 1, "unordered": True},
        "sequence_configurations": configs,
        "selection_protocol": "inner causal fit/validation split; fixed test untouched until final replay",
        "fixed_test_protocol": "train on all mature split members, freeze state, score all legal test rows through evaluation_contract",
        "prequential_protocol": "secondary replay; predict before consuming labels with completed_event_index < cutoff",
        "evidence_cohort_protocol": "common prospective cohort anchored to statistics/none first output; mature same-origin support only; never uses realized truth or model hit",
        "leaderboard_protocol": "retain all legal rows and a prior-supported main leaderboard; parent-app fallback gets fixed partial score only when it was already a confirmed launchable action different from the current app",
        "scoring_contract": scoring.contract(),
    }
    write_private_json(output / "plan.json", plan)
    write_private_json(output / "input-audit.json", input_audit)
    write_private_json(output / "sequence-audit.json", sequence_audit)
    write_private_json(output / "inner-selection.json", {
        "selected": selection["selected"],
        "split": selection["split"],
        "configuration_results": selection["results"],
        "selection_rule": "prior-supported leaderboard when nonempty: service Top4, exact Top4, RankScore@4, service Top2, exact Top2, stable configuration ID",
    })

    validation_outputs = {}
    baseline_validation, baseline_validation_timing = replay("baseline", None, fit_rows, validation_rows, history)
    baseline_validation_score, _ = score(validation_rows, baseline_validation, "inner_validation")
    validation_outputs[BASELINE_NAME] = {"metrics": baseline_validation_score, "timing": baseline_validation_timing}
    for config in configs:
        item = selection["results"][config["id"]]
        validation_outputs[config["id"]] = {"metrics": item["metrics"], "timing": item["timing"]}
    write_private_json(output / "validation-metrics.json", validation_outputs)

    fixed_outputs: dict[str, dict[int, dict]] = {}
    fixed_timing: dict[str, dict] = {}
    fixed_outputs[BASELINE_NAME], fixed_timing[BASELINE_NAME] = replay(
        "baseline", None, train_rows, test_rows, history)
    for config in configs:
        fixed_outputs[config["id"]], fixed_timing[config["id"]] = replay(
            "sequence", config, train_rows, test_rows, history)

    prequential_outputs: dict[str, dict[int, dict]] = {}
    prequential_timing: dict[str, dict] = {}
    prequential_outputs[BASELINE_NAME], prequential_timing[BASELINE_NAME] = replay(
        "baseline", None, all_rows, test_rows, history)
    for config in configs:
        prequential_outputs[config["id"]], prequential_timing[config["id"]] = replay(
            "sequence", config, all_rows, test_rows, history)

    fixed_audited = evaluation_rows(test_rows, "fixed_test")
    prequential_audited = evaluation_rows(test_rows, "prequential_diagnostic")
    fixed_metrics = {name: scoring.measure(fixed_audited, outputs)
                     for name, outputs in fixed_outputs.items()}
    prequential_metrics = {name: scoring.measure(prequential_audited, outputs)
                           for name, outputs in prequential_outputs.items()}
    scoring.require_comparable(*fixed_metrics.values())
    scoring.require_comparable(*prequential_metrics.values())
    leaderboards = {
        "fixed_test": scoring.score_leaderboards(fixed_audited, fixed_outputs),
        "prequential": scoring.score_leaderboards(prequential_audited, prequential_outputs),
    }
    leaderboard_audits = {
        "fixed_test": scoring.leaderboard_audit(fixed_audited),
        "prequential": scoring.leaderboard_audit(prequential_audited),
    }
    write_private_json(output / "fixed-test-metrics.json", fixed_metrics)
    write_private_json(output / "prequential-metrics.json", prequential_metrics)
    write_private_json(output / "score-leaderboards.json", {
        "primary": scoring.PRIMARY_LEADERBOARD,
        "audits": leaderboard_audits,
        "metrics": leaderboards,
    })
    write_private_json(output / "replay-timing.json", {"fixed_test": fixed_timing, "prequential": prequential_timing})

    evidence_results: dict[str, dict] = {}
    evidence_details: list[dict] = []
    for arm, outputs, training in (
        ("fixed_test", fixed_outputs, train_rows),
        ("prequential", prequential_outputs, all_rows),
    ):
        result, details = prospective_evidence_metrics(test_rows, training, outputs, arm)
        evidence_results[arm] = result
        evidence_details.extend(details)
    write_private_json(output / "evidence-cohort-metrics.json", evidence_results)
    write_private_jsonl(output / "evidence-cohort-help-harm-details-private.jsonl", evidence_details)

    paired_summary, paired_details = {}, []
    for arm, outputs in (("fixed_test", fixed_outputs), ("prequential", prequential_outputs)):
        baseline = outputs[BASELINE_NAME]
        for name, alternative in outputs.items():
            if name == BASELINE_NAME:
                continue
            summary, details = paired_help_harm(test_rows, baseline, alternative, arm, name)
            paired_summary[f"{arm}::{name}"] = summary
            paired_details.extend(details)
    write_private_json(output / "paired-help-harm.json", paired_summary)
    write_private_jsonl(output / "paired-help-harm-details-private.jsonl", paired_details)

    forecast_digests = {
        "fixed_test": {name: scoring.digest(outputs) for name, outputs in fixed_outputs.items()},
        "prequential": {name: scoring.digest(outputs) for name, outputs in prequential_outputs.items()},
        "note": "Hashes cover transient actual model outputs; prediction lists are not persisted.",
    }
    write_private_json(output / "forecast-digests-private.json", forecast_digests)
    write_report(output / "report-private.md", plan, selection, fixed_metrics, prequential_metrics,
                 leaderboards, leaderboard_audits, evidence_results, paired_summary, sequence_audit, input_audit)

    source_names = [
        "compare_multistep_statistical.py", "multistep_statistical_models.py",
        "evidence_cohort.py",
        "evaluation_contract.py", "prepare_offline_batch_split.py", "run_offline_batch_rankers.py",
        "context_statistics.py", "semantic_annotations.py", "navigation_hierarchy.py", "known_inactivity.py",
    ]
    source_hashes = {}
    for name in source_names:
        source = SCRIPT_DIR / name
        target = output / ("source-" + name)
        shutil.copyfile(source, target)
        target.chmod(0o600)
        source_hashes[name] = sha256(source)

    run = {
        "version": VERSION,
        "complete": True,
        "snapshot": str(snapshot),
        "split": str(split_dir),
        "source_hashes": source_hashes,
        "foreground_only": True,
        "candidate_lists_persisted": False,
        "model_outputs_persisted": False,
        "runtime_or_online_model_modified": False,
        "fixed_test_scored_once": True,
        "prequential_is_secondary_diagnostic": True,
        "evidence_cohort_scored_with_shared_membership": True,
        "evidence_cohort_membership_uses_realized_truth": False,
        "evidence_cohort_membership_uses_model_hit": False,
        "primary_scoreboard": scoring.PRIMARY_LEADERBOARD,
        "scoreboard_audits": leaderboard_audits,
        "selected_sequence_config": selection["selected"]["config"],
        "fixed_test_metrics": {name: {
            "n": metrics["n"], "accepted_top1": metrics["top1"], "accepted_top2": metrics["top2"],
            "accepted_top4": metrics["top4"], "exact_top1": metrics["exact"]["top1"],
            "exact_top2": metrics["exact"]["top2"], "exact_top4": metrics["exact"]["top4"],
            "rank_score": metrics["rank_score"]["mean"],
        } for name, metrics in fixed_metrics.items()},
        "scoring_contract": scoring.contract(),
    }
    write_private_json(output / "run.json", run)
    write_private_json(output / "artifact-hashes.json", artifact_hashes(output))
    print(json.dumps({
        "status": "complete",
        "output": str(output),
        "train_members": len(train_rows),
        "test_members": len(test_rows),
        "selected_sequence_config": selection["selected"]["config"]["id"],
        "fixed_models": list(fixed_metrics),
    }, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
