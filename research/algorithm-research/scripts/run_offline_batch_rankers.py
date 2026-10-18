#!/usr/bin/env python3
"""Run sealed, foreground-only offline ranker experiments.

The runner intentionally keeps the application's online v3 model untouched.
It reconstructs an action catalog only in memory from target updates visible at
each prediction cutoff, trains batch models once, and writes aggregate reports
plus private model parameters.  It never exports raw targets or a candidate
list.  The held-out test membership is read only after inner time-forward model
selection has been completed.

Models compared on one scoring contract:

* a frozen no-context statistical prior;
* a batch conditional-logit residual over that prior;
* a CatBoost YetiRankPairwise residual over that prior;
* a hybrid LightFM residual with a learned (not hand-set) calibration scale.

All target identifiers used by model features are one-way hashed before model
construction.  Full locators remain in memory only long enough to validate an
already-observed, reopenable action and to apply the agreed service hierarchy.
"""

from __future__ import annotations

import argparse
import bisect
import collections
import copy
import dataclasses
import hashlib
import json
import math
import os
import pickle
import platform
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable
from urllib.parse import urlsplit

import numpy as np
from scipy.optimize import minimize, minimize_scalar
from scipy.sparse import coo_matrix, csr_matrix, vstack


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import evaluation_contract as evaluation
import prepare_offline_batch_split as split


VERSION = "offline-batch-rankers-v1"
PRIVATE_ROOT = Path.home() / "Library/Application Support/SmartApp/training"
FOREGROUND_COVERAGE_VERSION = "foreground-coverage-v2"
ACTIONABLE_KINDS = {"app", "window", "page", "directory", "file"}
CONCRETE_KINDS = {"page", "directory", "file"}
HASH_PERSON = b"OffRankV1"
FEATURE_PERSON = b"OffFeatV1"
LINEAR_DIMENSIONS = 4096
MAX_TRAINING_CANDIDATES = 48
SMOOTHING = 1.0
CAT_COLUMNS = (
    "origin_token",
    "candidate_token",
    "origin_candidate_token",
    "origin_kind",
    "candidate_kind",
    "origin_bundle_token",
    "candidate_bundle_token",
    "same_bundle",
    "candidate_service",
)
NUMERIC_COLUMNS = (
    "prior_log_probability",
    "global_log_count",
    "conditional_log_count",
    "recent_1h_log_count",
    "recent_24h_log_count",
    "recency_1h",
    "recency_24h",
    "catalog_log_size",
)


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        if hasattr(hashlib, "file_digest"):
            return hashlib.file_digest(stream, "sha256").hexdigest()
        digest = hashlib.sha256()
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
        return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def write_private_json(path: Path, value: Any) -> None:
    if path.exists():
        raise ValueError(f"Immutable artifact already exists: {path}")
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")
    path.chmod(0o600)


def write_private_text(path: Path, text: str) -> None:
    if path.exists():
        raise ValueError(f"Immutable artifact already exists: {path}")
    path.write_text(text, encoding="utf-8")
    path.chmod(0o600)


def token(value: str | None) -> str:
    """Stable opaque feature value; never store a target or locator verbatim."""
    if not value:
        return "none"
    return hashlib.blake2b(value.encode("utf-8"), digest_size=12, person=HASH_PERSON).hexdigest()


def feature_index(name: str) -> tuple[int, float]:
    digest = hashlib.blake2b(name.encode("utf-8"), digest_size=8, person=FEATURE_PERSON).digest()
    number = int.from_bytes(digest, "little")
    return number % LINEAR_DIMENSIONS, 1.0 if number >> 63 else -1.0


def owner(meta: dict[str, Any]) -> str | None:
    value = meta.get("bundleID") or meta.get("_bundle")
    if isinstance(value, str) and value:
        return value
    if meta.get("kind") == "app" and isinstance(meta.get("id"), str) and meta["id"].startswith("app:"):
        return meta["id"][4:] or None
    return None


def valid_locator(meta: dict[str, Any]) -> bool:
    locator = meta.get("locator")
    if not isinstance(locator, str) or not locator:
        return False
    kind = meta.get("kind")
    parsed = urlsplit(locator)
    if kind == "page":
        return parsed.scheme.lower() in {"http", "https"} and bool(parsed.hostname)
    if kind in {"directory", "file"}:
        return locator.startswith("/") or (parsed.scheme == "file" and parsed.path.startswith("/"))
    return False


def admitted(meta: dict[str, Any]) -> bool:
    """Mirror the recorded eligibility evidence without discovering anything new."""
    admission = meta.get("admission")
    if not isinstance(admission, dict):
        return False
    basis = admission.get("basis")
    kind = meta.get("kind")
    bundle = owner(meta)
    if kind == "app":
        return bool(bundle and basis in {"standalone_application", "independent_work_window"})
    if kind == "window":
        return bool(bundle and basis in {
            "standard_work_window", "supported_browser_window", "supported_finder_window",
            "supported_qspace_window", "independent_work_window",
        })
    if kind == "page":
        return basis == "supported_browser_page" and valid_locator(meta)
    if kind == "directory":
        return basis in {"registered_local_target", "observed_finder_directory", "observed_qspace_directory"} and valid_locator(meta)
    if kind == "file":
        return basis in {"registered_local_target", "observed_window_document"} and valid_locator(meta)
    return False


@dataclass(frozen=True)
class Action:
    key: str
    meta: dict[str, Any]
    kind: str
    bundle: str | None


def action_from_meta(meta: dict[str, Any] | None, *, catalog_only: bool) -> Action | None:
    """Normalize a confirmed destination to an executable action.

    A window deliberately becomes its owning app action.  The fresh synthetic
    metadata does not claim reopening the original transient window; concrete
    page/directory/file locators retain their exact IDs for scoring only.
    """
    if not isinstance(meta, dict):
        return None
    kind = meta.get("kind")
    raw_id = meta.get("id")
    if not isinstance(raw_id, str) or not raw_id:
        return None
    bundle = owner(meta)
    if kind == "window":
        if not bundle or not admitted(meta):
            return None
        action_meta = {"id": "app:" + bundle, "kind": "app", "bundleID": bundle,
                       "admission": {"basis": "window_action_fallback", "policyVersion": VERSION}}
        return Action(action_meta["id"], action_meta, "app", bundle)
    if kind == "app":
        if not bundle or not admitted(meta):
            return None
        action_meta = {"id": "app:" + bundle, "kind": "app", "bundleID": bundle,
                       "admission": copy.deepcopy(meta.get("admission", {}))}
        return Action(action_meta["id"], action_meta, "app", bundle)
    if kind in CONCRETE_KINDS:
        if not admitted(meta):
            return None
        return Action(raw_id, copy.deepcopy(meta), kind, bundle)
    if catalog_only:
        return None
    # Keep a malformed/unsupported truth visible to the audit rather than
    # silently removing a legal split member.  It can never enter a catalog.
    return Action(raw_id, copy.deepcopy(meta), str(kind or "unknown"), bundle)


def foreground_parent_action(event: dict[str, Any] | None) -> Action | None:
    """Use only the app recorded on this same foreground observation.

    This does not inspect another window, tab, or application inventory.  The
    returned action is evaluation-only until the separately reconstructed
    pre-cutoff catalog proves that this app was already a launchable action.
    """
    if not isinstance(event, dict) or event.get("kind") != "context.change":
        return None
    app = event.get("app")
    bundle = app.get("bundleID") if isinstance(app, dict) else None
    if not isinstance(bundle, str) or not bundle:
        return None
    meta = {
        "id": "app:" + bundle,
        "kind": "app",
        "bundleID": bundle,
        "admission": {"basis": "foreground_associated_application", "policyVersion": VERSION},
    }
    return Action(meta["id"], meta, "app", bundle)


@dataclass(frozen=True)
class Observation:
    sample_id: str
    split: str
    position: int
    input_event_index: int
    completed_event_index: int
    input_timestamp_ms: int
    outcome_timestamp_ms: int
    outcome_event_id: str
    current: Action | None
    truth: Action | None
    current_parent: Action | None = None
    truth_parent: Action | None = None
    has_prior_exact_action: bool = False
    has_prior_parent_action: bool = False


class TargetHistory:
    """Read-only as-of target metadata and action reconstruction."""

    def __init__(self, events: list[dict[str, Any]]):
        self.events = events
        self.update_events: list[tuple[int, dict[str, Any]]] = []
        self.by_target: dict[str, list[tuple[int, dict[str, Any]]]] = collections.defaultdict(list)
        for index, event in enumerate(events):
            if event.get("kind") != "target.update":
                continue
            meta = event.get("data", {}).get("target")
            if not isinstance(meta, dict) or not isinstance(meta.get("id"), str):
                continue
            self.update_events.append((index, meta))
            self.by_target[meta["id"]].append((index, meta))
        self.update_indices = [index for index, _ in self.update_events]

    def metadata_at(self, target_id: str | None, index: int) -> dict[str, Any] | None:
        if not isinstance(target_id, str):
            return None
        entries = self.by_target.get(target_id, [])
        positions = [item[0] for item in entries]
        at = bisect.bisect_right(positions, index) - 1
        return entries[at][1] if at >= 0 else None

    def catalog_at(self, index: int) -> dict[str, Action]:
        """Derive a temporary action catalog only from past target updates."""
        latest: dict[str, dict[str, Any]] = {}
        stop = bisect.bisect_right(self.update_indices, index)
        for _, meta in self.update_events[:stop]:
            latest[meta["id"]] = meta
        actions: dict[str, Action] = {}
        for meta in latest.values():
            action = action_from_meta(meta, catalog_only=True)
            if action is not None and action.kind in CONCRETE_KINDS | {"app"}:
                actions[action.key] = action
        return actions


def verify_split(split_dir: Path, snapshot: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, str]]:
    run = read_json(split_dir / "run.json")
    if run.get("version") != split.SPLIT_VERSION or not run.get("complete"):
        raise ValueError("Split is not a completed offline-batch-split-v1 run")
    if Path(run.get("snapshot", "")).resolve() != snapshot.resolve():
        raise ValueError("Split points to a different snapshot")
    snapshot_hashes, _ = split.snapshot_hashes(snapshot)
    if snapshot_hashes != run.get("snapshotHashes"):
        raise ValueError("Snapshot hashes differ from split provenance")
    stored = read_json(split_dir / "artifact-hashes.json")
    for name in ("train-membership-private.jsonl", "test-membership-private.jsonl", "split-report.json"):
        path = split_dir / name
        if stored.get(name) != sha256(path):
            raise ValueError(f"Split artifact hash failed: {name}")
    return (read_jsonl(split_dir / "train-membership-private.jsonl"),
            read_jsonl(split_dir / "test-membership-private.jsonl"), snapshot_hashes)


def load_observations(snapshot: Path, split_dir: Path) -> tuple[list[Observation], list[Observation], TargetHistory, dict[str, Any]]:
    train_members, test_members, hashes = verify_split(split_dir, snapshot)
    events = read_jsonl(snapshot / "events.jsonl")
    samples = {sample["id"]: sample for sample in read_jsonl(snapshot / "navigation-samples.jsonl")}
    by_event = {event["id"]: event for event in events}
    event_index = {event["id"]: index for index, event in enumerate(events)}
    if len(by_event) != len(events):
        raise ValueError("Duplicate event IDs in snapshot")
    history = TargetHistory(events)

    def materialize(members: list[dict[str, Any]], group: str, offset: int) -> list[Observation]:
        result: list[Observation] = []
        for local_position, member in enumerate(members):
            sample = samples.get(member.get("sampleID"))
            anchor = by_event.get(member.get("anchorEventID"))
            if sample is None or anchor is None:
                raise ValueError("Split membership has an unknown sample or anchor")
            outcomes = sample.get("outcomes") or []
            if not outcomes or outcomes[0].get("eventID") not in by_event:
                raise ValueError("Split member lacks an observed outcome event")
            outcome = outcomes[0]
            outcome_event = by_event[outcome["eventID"]]
            input_index = event_index[anchor["id"]]
            outcome_index = event_index[outcome["eventID"]]
            if input_index != member.get("inputEventIndex") or outcome_index <= input_index:
                raise ValueError("Split chronology does not match snapshot")
            current_meta = history.metadata_at(anchor.get("data", {}).get("targetID"), input_index)
            truth_meta = history.metadata_at(outcome.get("targetID"), outcome_index)
            current = action_from_meta(current_meta, catalog_only=False)
            truth = action_from_meta(truth_meta, catalog_only=False)
            current_parent = current if current is not None and current.kind == "app" else foreground_parent_action(anchor)
            truth_parent = truth if truth is not None and truth.kind == "app" else foreground_parent_action(outcome_event)
            catalog = history.catalog_at(input_index)
            has_prior_exact_action = truth is not None and truth.key in catalog
            has_prior_parent_action = bool(
                truth is not None and truth.kind in CONCRETE_KINDS
                and truth_parent is not None and current_parent is not None
                and truth_parent.key != current_parent.key and truth_parent.key in catalog
            )
            result.append(Observation(
                sample_id=sample["id"], split=group, position=offset + local_position,
                input_event_index=input_index, completed_event_index=member["completedEventIndex"],
                input_timestamp_ms=int(anchor["timestampMs"]), outcome_timestamp_ms=int(outcome_event["timestampMs"]),
                outcome_event_id=outcome["eventID"], current=current, truth=truth,
                current_parent=current_parent, truth_parent=truth_parent,
                has_prior_exact_action=has_prior_exact_action,
                has_prior_parent_action=has_prior_parent_action,
            ))
        return result

    train = materialize(train_members, "train", 0)
    test = materialize(test_members, "test", len(train))
    if len({row.sample_id for row in train + test}) != len(train) + len(test):
        raise ValueError("Split memberships overlap")
    if train and test and max(row.completed_event_index for row in train) >= min(row.input_event_index for row in test):
        raise ValueError("Split has train/test maturity leakage")
    audit = {
        "snapshot_hashes": hashes,
        "train_members": len(train),
        "test_members": len(test),
        "missing_current_metadata": sum(row.current is None for row in train + test),
        "missing_truth_metadata": sum(row.truth is None for row in train + test),
        "current_action_kinds": dict(collections.Counter(
            row.current.kind if row.current else "missing" for row in train + test)),
        "truth_action_kinds": dict(collections.Counter(
            row.truth.kind if row.truth else "missing" for row in train + test)),
        "missing_current_parent": sum(row.current_parent is None for row in train + test),
        "missing_truth_parent": sum(row.truth_parent is None for row in train + test),
        "prior_exact_action": sum(row.has_prior_exact_action for row in train + test),
        "prior_parent_action": sum(row.has_prior_parent_action for row in train + test),
        "prior_supported": sum(row.has_prior_exact_action or row.has_prior_parent_action for row in train + test),
    }
    return train, test, history, audit


class HistoryStats:
    """Independent, mature foreground arrivals only; no test-period updates."""

    def __init__(self) -> None:
        self.global_counts: collections.Counter[str] = collections.Counter()
        self.conditional_counts: collections.Counter[tuple[str, str]] = collections.Counter()
        self.origin_counts: collections.Counter[str] = collections.Counter()
        self.records: list[tuple[int, str, str]] = []
        self.last_seen: dict[str, int] = {}
        self.seen_outcomes: set[str] = set()
        self._summary_cache: dict[tuple[str, int], dict[str, Any]] = {}

    def add(self, row: Observation) -> bool:
        if row.outcome_event_id in self.seen_outcomes:
            return False
        if row.current is None or row.truth is None:
            return False
        if row.current.kind not in ACTIONABLE_KINDS or row.truth.kind not in ACTIONABLE_KINDS:
            return False
        self.seen_outcomes.add(row.outcome_event_id)
        self.global_counts[row.truth.key] += 1
        self.conditional_counts[row.current.key, row.truth.key] += 1
        self.origin_counts[row.current.key] += 1
        self.records.append((row.outcome_timestamp_ms, row.current.key, row.truth.key))
        self.last_seen[row.truth.key] = max(self.last_seen.get(row.truth.key, -1), row.outcome_timestamp_ms)
        self._summary_cache.clear()
        return True

    @classmethod
    def from_rows(cls, rows: Iterable[Observation]) -> "HistoryStats":
        result = cls()
        for row in sorted(rows, key=lambda item: (item.completed_event_index, item.input_event_index, item.sample_id)):
            result.add(row)
        return result

    def summary(self, current_key: str, timestamp_ms: int) -> dict[str, Any]:
        key = current_key, timestamp_ms
        cached = self._summary_cache.get(key)
        if cached is not None:
            return cached
        recent_1h: collections.Counter[str] = collections.Counter()
        recent_24h: collections.Counter[str] = collections.Counter()
        for seen_at, _, target in self.records:
            if seen_at > timestamp_ms:
                raise ValueError("Future behavior reached a feature state")
            age = timestamp_ms - seen_at
            if age <= 3_600_000:
                recent_1h[target] += 1
            if age <= 86_400_000:
                recent_24h[target] += 1
        value = {"recent_1h": recent_1h, "recent_24h": recent_24h}
        self._summary_cache[key] = value
        return value


def action_features(row: Observation, action: Action, stats: HistoryStats, catalog_size: int) -> tuple[float, dict[str, Any]]:
    if row.current is None:
        raise ValueError("Cannot create a prediction feature without a current action")
    if catalog_size < 1:
        raise ValueError("Candidate catalog must be nonempty")
    current = row.current
    summary = stats.summary(current.key, row.input_timestamp_ms)
    global_count = stats.global_counts[action.key]
    conditional_count = stats.conditional_counts[current.key, action.key]
    recent_1h = summary["recent_1h"][action.key]
    recent_24h = summary["recent_24h"][action.key]
    probability = (global_count + SMOOTHING) / (sum(stats.global_counts.values()) + SMOOTHING * catalog_size)
    last = stats.last_seen.get(action.key)
    age = math.inf if last is None else max(0, row.input_timestamp_ms - last)
    service = evaluation.service(action.meta) is not None
    values = {
        "origin_token": token(current.key),
        "candidate_token": token(action.key),
        "origin_candidate_token": token(current.key + "\0" + action.key),
        "origin_kind": current.kind,
        "candidate_kind": action.kind,
        "origin_bundle_token": token(current.bundle),
        "candidate_bundle_token": token(action.bundle),
        "same_bundle": "yes" if current.bundle and current.bundle == action.bundle else "no",
        "candidate_service": "yes" if service else "no",
        "prior_log_probability": math.log(probability),
        "global_log_count": math.log1p(global_count),
        "conditional_log_count": math.log1p(conditional_count),
        "recent_1h_log_count": math.log1p(recent_1h),
        "recent_24h_log_count": math.log1p(recent_24h),
        "recency_1h": 0.0 if math.isinf(age) else math.exp(-age / 3_600_000),
        "recency_24h": 0.0 if math.isinf(age) else math.exp(-age / 86_400_000),
        "catalog_log_size": math.log(catalog_size),
    }
    return values["prior_log_probability"], values


@dataclass
class QueryGroup:
    row: Observation
    actions: list[Action]
    prior: np.ndarray
    features: list[dict[str, Any]]
    truth_index: int | None


def choose_training_actions(group: QueryGroup) -> QueryGroup:
    if group.truth_index is None or len(group.actions) <= MAX_TRAINING_CANDIDATES:
        return group
    truth = group.actions[group.truth_index]
    order = sorted(range(len(group.actions)), key=lambda index: (-group.prior[index], group.actions[index].key))
    chosen: set[int] = {group.truth_index}
    chosen.update(order[:24])
    for index, action in enumerate(group.actions):
        if action.kind == truth.kind or (truth.bundle and action.bundle == truth.bundle):
            chosen.add(index)
            if len(chosen) >= 36:
                break
    remainder = [index for index in range(len(group.actions)) if index not in chosen]
    remainder.sort(key=lambda index: token(group.row.sample_id + "\0" + group.actions[index].key))
    chosen.update(remainder[:max(0, MAX_TRAINING_CANDIDATES - len(chosen))])
    selected = sorted(chosen, key=lambda index: (-group.prior[index], group.actions[index].key))
    truth_index = selected.index(group.truth_index)
    return QueryGroup(group.row, [group.actions[index] for index in selected], group.prior[selected],
                      [group.features[index] for index in selected], truth_index)


def make_group(row: Observation, stats: HistoryStats, history: TargetHistory, *, training: bool) -> QueryGroup:
    catalog = history.catalog_at(row.input_event_index)
    actions = [catalog[key] for key in sorted(catalog)]
    if row.current is None or not actions:
        return QueryGroup(row, [], np.array([], dtype=float), [], None)
    prior, features = zip(*(action_features(row, action, stats, len(actions)) for action in actions))
    truth_index = None
    if row.truth is not None:
        for index, action in enumerate(actions):
            if action.key == row.truth.key:
                truth_index = index
                break
    result = QueryGroup(row, actions, np.asarray(prior, dtype=float), list(features), truth_index)
    return choose_training_actions(result) if training else result


def build_causal_training_groups(rows: list[Observation], history: TargetHistory) -> tuple[list[QueryGroup], dict[str, int]]:
    stats = HistoryStats()
    groups: list[QueryGroup] = []
    audit: collections.Counter[str] = collections.Counter()
    maturity = sorted(rows, key=lambda item: (item.completed_event_index, item.input_event_index, item.sample_id))
    cursor = 0
    seen_relations: set[tuple[str, str]] = set()
    for row in sorted(rows, key=lambda item: (item.input_event_index, item.sample_id)):
        while cursor < len(maturity) and maturity[cursor].completed_event_index < row.input_event_index:
            if stats.add(maturity[cursor]):
                audit["mature_unique_arrivals"] += 1
            else:
                audit["mature_duplicate_or_unusable_arrivals"] += 1
            cursor += 1
        group = make_group(row, stats, history, training=True)
        if group.row.truth is None or group.row.current is None:
            audit["unusable_action_metadata"] += 1
        elif group.truth_index is None:
            audit["truth_not_in_asof_catalog"] += 1
        elif len(group.actions) < 2:
            audit["fewer_than_two_training_actions"] += 1
        else:
            identity = (group.row.current.key, group.row.outcome_event_id)
            if identity in seen_relations:
                audit["duplicate_origin_outcome_relation"] += 1
            else:
                seen_relations.add(identity)
                groups.append(group)
                audit["ranker_groups"] += 1
    return groups, dict(audit)


def build_static_groups(rows: list[Observation], stats: HistoryStats, history: TargetHistory) -> list[QueryGroup]:
    return [make_group(row, stats, history, training=False) for row in rows]


def encode_linear(feature_rows: list[dict[str, Any]]) -> csr_matrix:
    indices: list[int] = []
    values: list[float] = []
    indptr = [0]
    for row in feature_rows:
        encoded: collections.Counter[int] = collections.Counter()
        for name in CAT_COLUMNS:
            index, sign = feature_index("cat/" + name + "=" + str(row[name]))
            encoded[index] += sign
        for name in NUMERIC_COLUMNS:
            index, sign = feature_index("num/" + name)
            encoded[index] += sign * float(row[name])
        index, sign = feature_index("bias")
        encoded[index] += sign
        for index in sorted(encoded):
            if encoded[index]:
                indices.append(index)
                values.append(encoded[index])
        indptr.append(len(indices))
    return csr_matrix((values, indices, indptr), shape=(len(feature_rows), LINEAR_DIMENSIONS), dtype=float)


@dataclass
class LinearResidual:
    weights: np.ndarray
    l2: float
    optimization: dict[str, Any]

    def correction(self, group: QueryGroup) -> np.ndarray:
        return np.asarray(encode_linear(group.features) @ self.weights).ravel() if group.actions else np.array([])


def train_linear(groups: list[QueryGroup], l2: float) -> LinearResidual:
    differences: list[csr_matrix] = []
    for group in groups:
        assert group.truth_index is not None
        matrix = encode_linear(group.features)
        for index in range(len(group.actions)):
            if index != group.truth_index:
                differences.append(matrix[group.truth_index] - matrix[index])
    if not differences:
        raise ValueError("Conditional-logit training has no positive/negative pairs")
    X = vstack(differences, format="csr")

    def objective(weights: np.ndarray) -> tuple[float, np.ndarray]:
        margins = np.asarray(X @ weights).ravel()
        loss = float(np.logaddexp(0.0, -margins).sum() + 0.5 * l2 * np.dot(weights, weights))
        probabilities = 1.0 / (1.0 + np.exp(np.clip(margins, -60.0, 60.0)))
        gradient = -np.asarray(X.T @ probabilities).ravel() + l2 * weights
        return loss, gradient

    result = minimize(objective, np.zeros(LINEAR_DIMENSIONS), jac=True, method="L-BFGS-B",
                      options={"maxiter": 150, "ftol": 1e-9, "maxls": 30})
    if not np.isfinite(result.x).all() or not np.isfinite(result.fun):
        raise ValueError("Conditional-logit optimization became nonfinite")
    return LinearResidual(result.x.astype(float), l2, {
        "success": bool(result.success), "status": int(result.status), "message": str(result.message),
        "iterations": int(result.nit), "objective": float(result.fun), "pair_count": int(X.shape[0]),
    })


def cat_frame(groups: list[QueryGroup]):
    import pandas as pd

    rows: list[dict[str, Any]] = []
    labels: list[int] = []
    group_ids: list[int] = []
    baselines: list[float] = []
    for group_id, group in enumerate(groups):
        if group.truth_index is None:
            continue
        for index, feature in enumerate(group.features):
            rows.append({name: feature[name] for name in CAT_COLUMNS + NUMERIC_COLUMNS})
            labels.append(int(index == group.truth_index))
            group_ids.append(group_id)
            baselines.append(float(group.prior[index]))
    return pd.DataFrame(rows, columns=CAT_COLUMNS + NUMERIC_COLUMNS), labels, group_ids, baselines


@dataclass
class CatBoostResidual:
    model: Any
    config: dict[str, Any]

    def correction(self, group: QueryGroup) -> np.ndarray:
        if not group.actions:
            return np.array([])
        import pandas as pd
        frame = pd.DataFrame([{name: row[name] for name in CAT_COLUMNS + NUMERIC_COLUMNS}
                              for row in group.features], columns=CAT_COLUMNS + NUMERIC_COLUMNS)
        return np.asarray(self.model.predict(frame), dtype=float)


def train_catboost(groups: list[QueryGroup], config: dict[str, Any]) -> CatBoostResidual:
    from catboost import CatBoostRanker, Pool

    frame, labels, group_ids, baselines = cat_frame(groups)
    if not labels or len(set(group_ids)) < 2:
        raise ValueError("CatBoost training needs at least two ranked queries")
    pool = Pool(frame, label=labels, group_id=group_ids, baseline=baselines,
                cat_features=list(CAT_COLUMNS), feature_names=list(CAT_COLUMNS + NUMERIC_COLUMNS))
    model = CatBoostRanker(
        loss_function="YetiRankPairwise",
        eval_metric="NDCG:top=4",
        iterations=config["iterations"], depth=config["depth"], learning_rate=config["learning_rate"],
        l2_leaf_reg=config["l2_leaf_reg"], random_seed=config["seed"], thread_count=1,
        allow_writing_files=False, verbose=False,
    )
    model.fit(pool)
    return CatBoostResidual(model, copy.deepcopy(config))


def user_tokens(action: Action) -> list[str]:
    return ["bias", "origin=" + token(action.key), "kind=" + action.kind,
            "bundle=" + token(action.bundle)]


def item_tokens(action: Action) -> list[str]:
    return ["bias", "target=" + token(action.key), "kind=" + action.kind,
            "bundle=" + token(action.bundle), "service=" + ("yes" if evaluation.service(action.meta) else "no")]


def sparse_feature_rows(rows: list[list[str]], vocabulary: dict[str, int]) -> csr_matrix:
    data: list[float] = []
    row_indices: list[int] = []
    col_indices: list[int] = []
    for row_index, values in enumerate(rows):
        for value in sorted(set(values)):
            column = vocabulary.get(value)
            if column is not None:
                data.append(1.0)
                row_indices.append(row_index)
                col_indices.append(column)
    return csr_matrix((data, (row_indices, col_indices)), shape=(len(rows), len(vocabulary)), dtype=np.float32)


@dataclass
class LightFMResidual:
    model: Any
    user_vocabulary: dict[str, int]
    item_vocabulary: dict[str, int]
    scale: float
    config: dict[str, Any]

    def correction(self, group: QueryGroup) -> np.ndarray:
        if group.row.current is None or not group.actions:
            return np.array([])
        user_features = sparse_feature_rows([user_tokens(group.row.current)], self.user_vocabulary)
        item_features = sparse_feature_rows([item_tokens(action) for action in group.actions], self.item_vocabulary)
        return np.asarray(self.model.predict(
            np.zeros(len(group.actions), dtype=np.int32), np.arange(len(group.actions), dtype=np.int32),
            user_features=user_features, item_features=item_features, num_threads=1,
        ), dtype=float) * self.scale


def train_lightfm(rows: list[Observation], config: dict[str, Any], *, scale: float = 1.0) -> LightFMResidual:
    from lightfm import LightFM

    valid = [row for row in rows if row.current is not None and row.truth is not None
             and row.current.kind in ACTIONABLE_KINDS and row.truth.kind in ACTIONABLE_KINDS]
    if not valid:
        raise ValueError("LightFM training has no usable actions")
    users = {action: index for index, action in enumerate(sorted({row.current.key for row in valid if row.current}))}
    items = {action: index for index, action in enumerate(sorted({row.truth.key for row in valid if row.truth}))}
    user_meta = {row.current.key: row.current for row in valid if row.current is not None}
    item_meta = {row.truth.key: row.truth for row in valid if row.truth is not None}
    user_values = [user_tokens(user_meta[key]) for key in sorted(users, key=users.get)]
    item_values = [item_tokens(item_meta[key]) for key in sorted(items, key=items.get)]
    user_vocabulary = {value: index for index, value in enumerate(sorted({value for values in user_values for value in values}))}
    item_vocabulary = {value: index for index, value in enumerate(sorted({value for values in item_values for value in values}))}
    user_features = sparse_feature_rows(user_values, user_vocabulary)
    item_features = sparse_feature_rows(item_values, item_vocabulary)
    counts: collections.Counter[tuple[int, int]] = collections.Counter(
        (users[row.current.key], items[row.truth.key]) for row in valid if row.current and row.truth)
    pairs = sorted(counts)
    interactions = coo_matrix((np.ones(len(pairs), dtype=np.float32),
                               ([pair[0] for pair in pairs], [pair[1] for pair in pairs])),
                              shape=(len(users), len(items)), dtype=np.float32)
    weights = coo_matrix((np.asarray([counts[pair] for pair in pairs], dtype=np.float32),
                          ([pair[0] for pair in pairs], [pair[1] for pair in pairs])),
                         shape=(len(users), len(items)), dtype=np.float32)
    model = LightFM(loss=config["loss"], no_components=config["no_components"], learning_rate=config["learning_rate"],
                    user_alpha=config["user_alpha"], item_alpha=config["item_alpha"], random_state=config["seed"])
    model.fit(interactions, user_features=user_features, item_features=item_features, sample_weight=weights,
              epochs=config["epochs"], num_threads=1, verbose=False)
    return LightFMResidual(model, user_vocabulary, item_vocabulary, scale, copy.deepcopy(config))


def fit_lightfm_scale(model: LightFMResidual, groups: list[QueryGroup]) -> tuple[float, dict[str, Any]]:
    usable = [group for group in groups if group.truth_index is not None and len(group.actions) > 1]
    if not usable:
        return 0.0, {"calibration_groups": 0, "status": "no_rankable_calibration_groups"}
    raws = [model.correction(group) / model.scale for group in usable]

    def objective(scale: float) -> float:
        total = 0.0
        for group, raw in zip(usable, raws):
            assert group.truth_index is not None
            logits = group.prior + scale * raw
            total += float(np.logaddexp.reduce(logits) - logits[group.truth_index])
        return total / len(usable)

    result = minimize_scalar(objective, bounds=(0.0, 64.0), method="bounded", options={"xatol": 1e-5})
    selected = float(result.x) if result.success and math.isfinite(result.x) else 0.0
    return selected, {"calibration_groups": len(usable), "status": "optimized" if result.success else "optimizer_failed",
                      "objective": float(objective(selected)), "scale": selected}


def predictions(groups: list[QueryGroup], correction: Callable[[QueryGroup], np.ndarray]) -> dict[int, dict[str, Any]]:
    result: dict[int, dict[str, Any]] = {}
    for group in groups:
        values = correction(group)
        if len(values) != len(group.actions):
            raise ValueError("Model did not return one score per as-of action")
        if len(values) and not np.isfinite(values).all():
            raise ValueError("Model returned nonfinite rank scores")
        totals = group.prior + values
        order = sorted(range(len(group.actions)), key=lambda index: (-totals[index], group.actions[index].key))
        raw = [group.actions[index].key for index in order]
        result[group.row.position] = {"predictions": raw[:4], "rankable": raw}
    return result


def baseline_correction(group: QueryGroup) -> np.ndarray:
    return np.zeros(len(group.actions), dtype=float)


def evaluation_rows(rows: list[Observation], known_action_metadata: Iterable[dict[str, Any]] = ()) -> list[dict[str, Any]]:
    observations = []
    evaluation_rows = []
    for row in rows:
        current_meta = row.current.meta if row.current is not None else {}
        truth_meta = row.truth.meta if row.truth is not None else {}
        current_parent = row.current_parent.meta if row.current_parent is not None else None
        truth_parent = row.truth_parent.meta if row.truth_parent is not None else None
        observations.append({"position": row.position, "index": row.input_event_index,
                             "ready_index": row.completed_event_index, "target": current_meta,
                             "parent_action": current_parent})
        evaluation_rows.append({"position": row.position, "sample_id": row.sample_id,
                                "truth_action": row.truth.key if row.truth is not None else None,
                                "truth_metadata": truth_meta, "split": row.split,
                                "evidence_grade": "offline_batch_all_legal_members",
                                "truth_parent_metadata": truth_parent,
                                "prior_evidence": {
                                    "exact_target": row.has_prior_exact_action,
                                    "parent_action": row.has_prior_parent_action,
                                }})
    payload: dict[str, Any] = {"observations": observations, "evaluation": evaluation_rows}
    known = [copy.deepcopy(target) for target in known_action_metadata]
    if any(not isinstance(target, dict) or not isinstance(target.get("id"), str) for target in known):
        raise ValueError("Known scoring actions require target metadata with stable IDs")
    if known:
        # Evaluation-only metadata for actions that were actually emitted.  It
        # lets the shared scorer recognize a known service homepage without
        # persisting a candidate catalog or exposing it to a predictor.
        payload["streams"] = {
            "emitted_actions": [{"record": {"target": target}} for target in known],
        }
    return evaluation.prepare_evaluation(payload)


def score_groups(rows: list[Observation], groups: list[QueryGroup], models: dict[str, Callable[[QueryGroup], np.ndarray]]) -> tuple[dict[str, Any], dict[str, dict[int, dict[str, Any]]], list[dict[str, Any]]]:
    audit_rows = evaluation_rows(rows)
    by_position = {group.row.position for group in groups}
    if by_position != {row.position for row in rows}:
        raise ValueError("Prediction groups do not match the evaluation membership")
    outputs = {name: predictions(groups, scorer) for name, scorer in models.items()}
    scores = {name: evaluation.measure(audit_rows, output) for name, output in outputs.items()}
    evaluation.require_comparable(*scores.values())
    return scores, outputs, audit_rows


def score_leaderboards(audit_rows: list[dict[str, Any]],
                       outputs: dict[str, dict[int, dict[str, Any]]]) -> dict[str, dict[str, dict[str, Any]]]:
    """Keep all legal rows and the prior-supported comparison board together."""
    return evaluation.score_leaderboards(audit_rows, outputs)


def causal_three_way(rows: list[Observation]) -> tuple[list[Observation], list[Observation], list[Observation], dict[str, Any]]:
    """Find a time-forward fit/calibration/selection split with no maturity leak."""
    boundaries = sorted({row.input_event_index for row in rows})
    # A dense but bounded grid is enough for a 60/20/20 time split and avoids
    # turning split selection itself into a cubic scan over local history.
    stride = max(1, len(boundaries) // 48)
    candidates = sorted(set(boundaries[1:-1:stride] + [boundaries[1], boundaries[-2]]))
    options = []
    for first in candidates:
        for second in candidates:
            if second <= first:
                continue
            fit = [row for row in rows if row.completed_event_index < first]
            calibration = [row for row in rows if row.input_event_index >= first and row.completed_event_index < second]
            selection = [row for row in rows if row.input_event_index >= second]
            used = set(row.sample_id for row in fit + calibration + selection)
            if not fit or not calibration or not selection or len(used) != len(fit) + len(calibration) + len(selection):
                continue
            usable = len(used)
            ratio_error = abs(len(fit) / usable - 0.60) + abs((len(fit) + len(calibration)) / usable - 0.80)
            options.append(((ratio_error, -usable, first, second), fit, calibration, selection, used, first, second))
    if not options:
        raise ValueError("Cannot construct a causal inner three-way split")
    _, fit, calibration, selection, used, first, second = min(options, key=lambda option: option[0])
    guards = [row for row in rows if row.sample_id not in used]
    if max(row.completed_event_index for row in fit) >= min(row.input_event_index for row in calibration):
        raise AssertionError("Fit labels reach calibration inputs")
    if max(row.completed_event_index for row in calibration) >= min(row.input_event_index for row in selection):
        raise AssertionError("Calibration labels reach selection inputs")
    audit = {"fit": len(fit), "calibration": len(calibration), "selection": len(selection), "guard": len(guards),
             "fit_boundary_event_index": first, "selection_boundary_event_index": second,
             "rule": "fit labels mature before calibration inputs; calibration labels mature before selection inputs"}
    return fit, calibration, selection, audit


def metric_key(score: dict[str, Any], config_id: str) -> tuple[Any, ...]:
    exact = score["exact"]
    return (-score["top4_hits"], -exact["top4_hits"], -score["rank_score"]["mean"],
            -score["top1_hits"], config_id)


def run_inner_selection(fit: list[Observation], calibration: list[Observation], selection: list[Observation],
                        history: TargetHistory) -> tuple[dict[str, Any], dict[str, Any]]:
    """Select one configuration per family without touching the sealed test set."""
    fit_groups, fit_audit = build_causal_training_groups(fit, history)
    frozen_fit_stats = HistoryStats.from_rows(fit)
    calibration_groups = build_static_groups(calibration, frozen_fit_stats, history)
    selection_groups = build_static_groups(selection, frozen_fit_stats, history)
    base_scores, base_outputs, base_audited = score_groups(selection, selection_groups, {"baseline": baseline_correction})
    base_leaderboards = score_leaderboards(base_audited, base_outputs)
    base_selection_board, base_selection_metrics = evaluation.primary_leaderboard_metric(base_leaderboards, "baseline")
    results: dict[str, list[dict[str, Any]]] = {"linear": [], "catboost": [], "lightfm": []}

    for l2 in (0.05, 0.5, 5.0):
        model = train_linear(fit_groups, l2)
        scores, outputs, audited = score_groups(selection, selection_groups, {"candidate": model.correction})
        leaderboards = score_leaderboards(audited, outputs)
        selection_board, selection_metrics = evaluation.primary_leaderboard_metric(leaderboards, "candidate")
        results["linear"].append({"id": f"linear-l2-{l2:g}", "config": {"l2": l2},
                                  "score": scores["candidate"], "leaderboards": leaderboards,
                                  "selection_leaderboard": selection_board,
                                  "selection_metrics": selection_metrics,
                                  "optimization": model.optimization})

    cat_configs = [
        {"id": "catboost-d3-r20", "depth": 3, "iterations": 120, "learning_rate": 0.05, "l2_leaf_reg": 8.0, "seed": 20260928},
        {"id": "catboost-d4-r20", "depth": 4, "iterations": 120, "learning_rate": 0.05, "l2_leaf_reg": 20.0, "seed": 20260928},
        {"id": "catboost-d5-r40", "depth": 5, "iterations": 160, "learning_rate": 0.04, "l2_leaf_reg": 40.0, "seed": 20260928},
    ]
    for config in cat_configs:
        model = train_catboost(fit_groups, config)
        scores, outputs, audited = score_groups(selection, selection_groups, {"candidate": model.correction})
        leaderboards = score_leaderboards(audited, outputs)
        selection_board, selection_metrics = evaluation.primary_leaderboard_metric(leaderboards, "candidate")
        results["catboost"].append({"id": config["id"], "config": config, "score": scores["candidate"],
                                     "leaderboards": leaderboards, "selection_leaderboard": selection_board,
                                     "selection_metrics": selection_metrics})

    light_configs = [
        {"id": "lightfm-bpr-8", "loss": "bpr", "no_components": 8, "learning_rate": 0.05,
         "user_alpha": 1e-4, "item_alpha": 1e-4, "epochs": 40, "seed": 20260928},
        {"id": "lightfm-warp-8", "loss": "warp", "no_components": 8, "learning_rate": 0.05,
         "user_alpha": 1e-4, "item_alpha": 1e-4, "epochs": 40, "seed": 20260928},
        {"id": "lightfm-warp-16", "loss": "warp", "no_components": 16, "learning_rate": 0.04,
         "user_alpha": 5e-4, "item_alpha": 5e-4, "epochs": 50, "seed": 20260928},
    ]
    for config in light_configs:
        model = train_lightfm(fit, config)
        scale, calibration_audit = fit_lightfm_scale(model, calibration_groups)
        model.scale = scale
        scores, outputs, audited = score_groups(selection, selection_groups, {"candidate": model.correction})
        leaderboards = score_leaderboards(audited, outputs)
        selection_board, selection_metrics = evaluation.primary_leaderboard_metric(leaderboards, "candidate")
        results["lightfm"].append({"id": config["id"], "config": config, "scale": scale,
                                    "calibration": calibration_audit, "score": scores["candidate"],
                                    "leaderboards": leaderboards, "selection_leaderboard": selection_board,
                                    "selection_metrics": selection_metrics})

    selected = {family: min(candidates, key=lambda item: metric_key(item["selection_metrics"], item["id"]))
                for family, candidates in results.items()}
    summary = {"version": VERSION, "test_accessed": False,
               "selection_rule": "maximize prior-supported (or all-legal if empty) service-accepted Top4, exact Top4, RankScore, then accepted Top1; stable ID breaks ties",
               "baseline_on_selection": base_scores["baseline"],
               "baseline_selection_leaderboards": base_leaderboards,
               "baseline_selection_leaderboard": base_selection_board,
               "baseline_selection_metrics": base_selection_metrics,
               "fit_training": fit_audit,
               "candidate_results": results, "selected": selected}
    return summary, {"fit_groups": fit_groups, "frozen_fit_stats": frozen_fit_stats,
                     "calibration_groups": calibration_groups, "selection_groups": selection_groups}


def save_models(output: Path, linear: LinearResidual, catboost: CatBoostResidual, lightfm: LightFMResidual) -> dict[str, str]:
    model_dir = output / "models"
    model_dir.mkdir(mode=0o700)
    model_dir.chmod(0o700)
    linear_path = model_dir / "conditional-logit-residual.npz"
    np.savez_compressed(linear_path, weights=linear.weights)
    linear_path.chmod(0o600)
    catboost_path = model_dir / "catboost-yetirank-residual.cbm"
    catboost.model.save_model(str(catboost_path))
    catboost_path.chmod(0o600)
    lightfm_path = model_dir / "lightfm-residual.pkl"
    with lightfm_path.open("xb") as stream:
        pickle.dump(lightfm.model, stream, protocol=pickle.HIGHEST_PROTOCOL)
    lightfm_path.chmod(0o600)
    return {str(path.relative_to(output)): sha256(path) for path in (linear_path, catboost_path, lightfm_path)}


def source_artifacts(output: Path) -> None:
    for source in (Path(__file__), SCRIPT_DIR / "evaluation_contract.py", SCRIPT_DIR / "prepare_offline_batch_split.py"):
        target = output / ("source-" + source.name)
        if target.exists():
            raise ValueError("Duplicate source artifact")
        shutil.copyfile(source, target)
        target.chmod(0o600)


def environment() -> dict[str, Any]:
    modules: dict[str, str | None] = {}
    for name in ("numpy", "scipy", "sklearn", "catboost", "lightfm"):
        try:
            module = __import__(name)
            modules[name] = getattr(module, "__version__", "unknown")
        except Exception:
            modules[name] = None
    freeze = subprocess.run([sys.executable, "-m", "pip", "freeze"], check=False, capture_output=True, text=True)
    return {"python": sys.version, "executable": sys.executable, "platform": platform.platform(),
            "modules": modules, "pip_freeze": freeze.stdout.splitlines() if freeze.returncode == 0 else []}


def artifact_hashes(output: Path) -> dict[str, str]:
    result = {}
    for path in sorted(output.rglob("*")):
        if path.is_file() and path.name not in {"artifact-hashes.json", "run.json"}:
            result[str(path.relative_to(output))] = sha256(path)
    return result


def progress(stage: str, **details: Any) -> None:
    """Safe terminal progress: counts/config IDs only, never target identities."""
    print(json.dumps({"stage": stage, **details}, ensure_ascii=False, sort_keys=True), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    snapshot = args.snapshot.expanduser().resolve()
    split_dir = args.split.expanduser().resolve()
    output = args.output.expanduser().resolve()
    private = PRIVATE_ROOT.resolve()
    if output == private or not output.is_relative_to(private):
        parser.error("Output must be a new private SmartApp training directory")
    if output.exists():
        parser.error("Output directory must not already exist")

    # Fail before creating artifacts if a dependency cannot genuinely run.
    try:
        import catboost  # noqa: F401
        import lightfm  # noqa: F401
    except ImportError as error:
        parser.error(f"Missing offline model dependency: {error}")

    os.umask(0o077)
    output.mkdir(parents=True, mode=0o700)
    output.chmod(0o700)
    try:
        train, test, history, input_audit = load_observations(snapshot, split_dir)
        fit, calibration, selection, inner_split = causal_three_way(train)
        progress("input_verified", train_members=len(train), test_members=len(test), **inner_split)
        plan = {
            "version": VERSION,
            "scope": "local_offline_research_only",
            "snapshot": str(snapshot), "split": str(split_dir),
            "foreground_only": True, "candidate_lists_persisted": False,
            "runtime_or_online_model_modified": False,
            "current_sequence_length": 1,
            "test_protocol": "inner temporal fit/calibration/selection first; test scores exactly once after final retraining",
            "statistical_baseline": "frozen no-context global frequency prior with Laplace smoothing",
            "offline_update_policy": "no weight or count update during test; as-of catalog is reconstructed transiently from observed foreground target history",
            "models": ["conditional_logit_residual", "catboost_yetirank_pairwise_residual", "lightfm_hybrid_residual"],
            "action_semantics": "app/window→app action; page/directory/file exact action; service and prediction-time-known parent-app fallbacks are scored by the shared contract without becoming exact hits",
            "leaderboard_protocol": "keep all legal rows and use the prior-supported board as the main model-comparison board when it is nonempty",
        }
        write_private_json(output / "plan.json", plan)
        write_private_json(output / "input-audit.json", input_audit)
        write_private_json(output / "inner-split.json", inner_split)
        source_artifacts(output)
        write_private_json(output / "environment.json", environment())

        inner, inner_state = run_inner_selection(fit, calibration, selection, history)
        write_private_json(output / "inner-selection.json", inner)
        progress("inner_selection_complete", selected={family: value["id"] for family, value in inner["selected"].items()})

        # The final parameters use every mature training member.  The only
        # calibration scalar retained for LightFM was learned before selection.
        final_groups, final_training_audit = build_causal_training_groups(train, history)
        final_stats = HistoryStats.from_rows(train)
        linear_choice = inner["selected"]["linear"]
        cat_choice = inner["selected"]["catboost"]
        light_choice = inner["selected"]["lightfm"]
        linear = train_linear(final_groups, float(linear_choice["config"]["l2"]))
        catboost = train_catboost(final_groups, cat_choice["config"])
        lightfm = train_lightfm(train, light_choice["config"], scale=float(light_choice["scale"]))
        model_hashes = save_models(output, linear, catboost, lightfm)
        write_private_json(output / "final-training.json", {
            "groups": len(final_groups), "training_audit": final_training_audit,
            "linear": {"config": linear_choice["config"], "optimization": linear.optimization},
            "catboost": {"config": cat_choice["config"]},
            "lightfm": {"config": light_choice["config"], "frozen_inner_calibration_scale": lightfm.scale},
            "model_hashes": model_hashes,
        })
        progress("final_models_fit", ranker_training_groups=len(final_groups))

        # The following is deliberately the first code path that materializes
        # predictions on the held-out membership.
        progress("heldout_scoring_started", test_members=len(test))
        test_groups = build_static_groups(test, final_stats, history)
        test_scores, test_outputs, audited_test = score_groups(test, test_groups, {
            "no_context_statistical": baseline_correction,
            "conditional_logit": linear.correction,
            "catboost_yetirank": catboost.correction,
            "lightfm_hybrid": lightfm.correction,
        })
        test_leaderboards = score_leaderboards(audited_test, test_outputs)
        test_leaderboard_audit = evaluation.leaderboard_audit(audited_test)
        paired = {name: evaluation.paired_topk(audited_test, test_outputs["no_context_statistical"], output)
                  for name, output in test_outputs.items() if name != "no_context_statistical"}
        candidate_audit = {
            "test_members": len(test), "test_groups_with_actions": sum(bool(group.actions) for group in test_groups),
            "test_truth_in_asof_action_catalog": sum(group.truth_index is not None for group in test_groups),
            "test_catalog_action_count": {
                "min": min((len(group.actions) for group in test_groups), default=0),
                "max": max((len(group.actions) for group in test_groups), default=0),
                "mean": sum(len(group.actions) for group in test_groups) / len(test_groups) if test_groups else 0.0,
            },
            "all_predictions_transient": True,
            "raw_predictions_or_candidate_lists_written": False,
        }
        write_private_json(output / "test-metrics.json", test_scores)
        write_private_json(output / "score-leaderboards.json", {
            "primary": evaluation.PRIMARY_LEADERBOARD,
            "audit": test_leaderboard_audit,
            "metrics": test_leaderboards,
        })
        write_private_json(output / "paired-help-harm.json", paired)
        write_private_json(output / "test-audit.json", candidate_audit)
        summary = {
            "version": VERSION, "complete": True, "test_scored_once": True,
            "foreground_only": True, "candidate_lists_persisted": False,
            "raw_predictions_or_candidate_lists_written": False,
            "runtime_or_online_model_modified": False,
            "test_members_requested": len(test), "test_members_after_label_audit": test_scores["no_context_statistical"]["audit"],
            "primary_scoreboard": evaluation.PRIMARY_LEADERBOARD,
            "scoreboard_audit": test_leaderboard_audit,
            "scoreboards": test_leaderboards,
            "models": {name: {"accepted_top1": score["top1"], "accepted_top4": score["top4"],
                               "exact_top1": score["exact"]["top1"], "exact_top4": score["exact"]["top4"],
                               "rank_score": score["rank_score"]["mean"], "prediction_coverage": score["prediction_coverage"]}
                       for name, score in test_scores.items()},
            "selected_configurations": inner["selected"],
        }
        write_private_json(output / "summary.json", summary)
        write_private_json(output / "artifact-hashes.json", artifact_hashes(output))
        write_private_json(output / "run.json", {
            "version": VERSION, "complete": True, "snapshot": str(snapshot), "split": str(split_dir),
            "input_snapshot_hashes": input_audit["snapshot_hashes"], "test_scored_once": True,
            "foreground_only": True, "candidate_lists_persisted": False,
            "raw_predictions_or_candidate_lists_written": False,
            "runtime_or_online_model_modified": False,
        })
        print(json.dumps({"status": "complete", "output": str(output), "test_members": len(test),
                          "models": list(test_scores)}, ensure_ascii=False))
    except Exception:
        # Preserve immutable partial evidence for diagnosis; do not overwrite it.
        raise


if __name__ == "__main__":
    main()
