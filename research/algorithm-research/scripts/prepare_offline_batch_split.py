#!/usr/bin/env python3
"""Create a causal train/test membership split from a foreground-only v4 snapshot.

The input snapshot remains immutable.  This tool writes only compact private
membership ledgers: later trainers reconstruct their as-of features and labels
from the referenced snapshot rather than receiving copied raw history.  It
never writes a recommendation candidate list or touches the running model.
"""

from __future__ import annotations

import argparse
import collections
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
from typing import Any


PRIVATE_ROOT = Path.home() / "Library/Application Support/SmartApp/training"
REQUIRED_SNAPSHOT_FILES = {
    "events.jsonl",
    "target-history.jsonl",
    "navigation-samples.jsonl",
    "manifest.json",
}
SPLIT_VERSION = "offline-batch-split-v1"
FOREGROUND_COVERAGE_VERSION = "foreground-coverage-v2"
LABEL_RULE_VERSION = "observed-foreground-sequence-30s-v4"


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        if hasattr(hashlib, "file_digest"):
            return hashlib.file_digest(stream, "sha256").hexdigest()
        digest = hashlib.sha256()
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
        return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def write_json(path: Path, value: Any) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")
    path.chmod(0o600)


def write_jsonl(path: Path, values: list[dict[str, Any]]) -> None:
    with path.open("x", encoding="utf-8") as stream:
        for value in values:
            stream.write(json.dumps(value, ensure_ascii=False, sort_keys=True))
            stream.write("\n")
    path.chmod(0o600)


def snapshot_hashes(snapshot: Path) -> tuple[dict[str, str], dict[str, Any]]:
    if not snapshot.is_dir():
        raise ValueError(f"Snapshot does not exist: {snapshot}")
    sums = snapshot / "SHA256SUMS"
    if not sums.is_file():
        raise ValueError("Snapshot lacks SHA256SUMS")
    hashes: dict[str, str] = {}
    for line in sums.read_text(encoding="utf-8").splitlines():
        digest, name = line.split(maxsplit=1)
        name = name.lstrip("*")
        if name in REQUIRED_SNAPSHOT_FILES:
            target = snapshot / name
            if not target.is_file() or sha256(target) != digest:
                raise ValueError(f"Snapshot checksum failed: {name}")
            hashes[name] = digest
    if set(hashes) != REQUIRED_SNAPSHOT_FILES:
        raise ValueError("Snapshot checksums do not cover every required file")
    manifest = json.loads((snapshot / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("schemaVersion") != 2 or manifest.get("sampleSchemaVersion") != 4:
        raise ValueError("Requires a context-v2 snapshot with navigation sample schema v4")
    if manifest.get("stateCoverageVersion") != FOREGROUND_COVERAGE_VERSION:
        raise ValueError("Requires the foreground-only coverage contract")
    return hashes, manifest


def rejection_reason(sample: dict[str, Any], events: dict[str, dict[str, Any]],
                     sequence_index: dict[tuple[str, int], int],
                     event_index: dict[str, int]) -> str | None:
    if sample.get("schemaVersion") != 4:
        return "wrong_sample_schema"
    if sample.get("labelRuleVersion") != LABEL_RULE_VERSION:
        return "wrong_label_rule"
    if sample.get("status") != "observed_horizon":
        return "incomplete_horizon"
    coverage = sample.get("inputCoverage") or {}
    if coverage.get("version") != FOREGROUND_COVERAGE_VERSION:
        return "wrong_coverage_contract"
    if coverage.get("status") != "foreground_only":
        return "input_not_foreground_only"
    if sample.get("inventoryEventIDs"):
        return "inventory_reference_present"
    if sample.get("outcomeCoverage") != "observed":
        return "outcome_not_observed"
    anchor = events.get(sample.get("anchorEventID"))
    if anchor is None or anchor.get("kind") != "context.change":
        return "missing_or_invalid_anchor"
    if anchor.get("data", {}).get("reason") == "resolution_update":
        return "resolution_anchor"
    if sample.get("inputThroughSequence") != anchor.get("sequence"):
        return "anchor_sequence_mismatch"
    if sample.get("inputCutoffMs") != anchor.get("monotonicMs"):
        return "anchor_cutoff_mismatch"
    outcomes = sample.get("outcomes") or []
    if not outcomes:
        return "no_observed_outcome"
    input_target = anchor.get("data", {}).get("targetID")
    outcome_target = outcomes[0].get("targetID")
    if not input_target or not outcome_target:
        return "missing_target_identity"
    if input_target == outcome_target:
        return "same_destination"
    session = sample.get("sessionID")
    completion = sample.get("completedThroughSequence")
    if not isinstance(completion, int) or (session, completion) not in sequence_index:
        return "missing_completion_boundary"
    if event_index[sample["anchorEventID"]] > sequence_index[(session, completion)]:
        return "completion_precedes_input"
    return None


def eligible_members(snapshot: Path) -> tuple[list[dict[str, Any]], collections.Counter[str], dict[str, str], dict[str, Any]]:
    hashes, manifest = snapshot_hashes(snapshot)
    events = read_jsonl(snapshot / "events.jsonl")
    samples = read_jsonl(snapshot / "navigation-samples.jsonl")
    if len(events) != manifest.get("events") or len(samples) != manifest.get("samples"):
        raise ValueError("Snapshot manifest count mismatch")
    by_id = {event["id"]: event for event in events}
    if len(by_id) != len(events):
        raise ValueError("Duplicate event ID in snapshot")
    event_index: dict[str, int] = {}
    sequence_index: dict[tuple[str, int], int] = {}
    for index, event in enumerate(events):
        event_index[event["id"]] = index
        sequence = event.get("sequence")
        if not isinstance(sequence, int):
            raise ValueError("Event has no session sequence")
        key = (event["sessionID"], sequence)
        if key in sequence_index:
            raise ValueError("Duplicate session sequence in snapshot")
        sequence_index[key] = index

    rejected: collections.Counter[str] = collections.Counter()
    members: list[dict[str, Any]] = []
    seen_samples: set[str] = set()
    for sample in samples:
        sample_id = sample.get("id")
        if not isinstance(sample_id, str) or not sample_id or sample_id in seen_samples:
            raise ValueError("Missing or duplicate navigation sample ID")
        seen_samples.add(sample_id)
        reason = rejection_reason(sample, by_id, sequence_index, event_index)
        if reason:
            rejected[reason] += 1
            continue
        anchor = by_id[sample["anchorEventID"]]
        completed_event_index = sequence_index[(sample["sessionID"], sample["completedThroughSequence"])]
        completed_event = events[completed_event_index]
        members.append({
            "sampleID": sample_id,
            "anchorEventID": sample["anchorEventID"],
            "inputEventIndex": event_index[sample["anchorEventID"]],
            "inputTimestampMs": anchor["timestampMs"],
            "completedEventIndex": completed_event_index,
            "completedTimestampMs": completed_event["timestampMs"],
            "origin": sample["origin"],
            "outcomeCount": len(sample["outcomes"]),
        })
    members.sort(key=lambda member: (member["inputEventIndex"], member["sampleID"]))
    return members, rejected, hashes, manifest


def split_members(members: list[dict[str, Any]], train_fraction: float) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], int]:
    if len(members) < 2:
        raise ValueError("Need at least two eligible changed-destination samples")
    boundaries = sorted({member["inputEventIndex"] for member in members})
    options: list[tuple[tuple[float, int, int], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], int]] = []
    for boundary in boundaries[1:]:
        train = [member for member in members if member["completedEventIndex"] < boundary]
        test = [member for member in members if member["inputEventIndex"] >= boundary]
        guard = [member for member in members
                 if member["inputEventIndex"] < boundary <= member["completedEventIndex"]]
        usable = len(train) + len(test)
        if not train or not test:
            continue
        ratio_error = abs(len(train) / usable - train_fraction)
        # Prefer the requested ratio, then retain more usable rows, then the earliest tie.
        options.append(((ratio_error, -usable, boundary), train, test, guard, boundary))
    if not options:
        raise ValueError("No non-overlapping causal train/test boundary exists")
    _, train, test, guard, boundary = min(options, key=lambda option: option[0])
    train_ids = {member["sampleID"] for member in train}
    test_ids = {member["sampleID"] for member in test}
    guard_ids = {member["sampleID"] for member in guard}
    if train_ids & test_ids or train_ids & guard_ids or test_ids & guard_ids:
        raise AssertionError("Causal split groups overlap")
    if train_ids | test_ids | guard_ids != {member["sampleID"] for member in members}:
        raise AssertionError("Causal split does not account for every eligible sample")
    if max(member["completedEventIndex"] for member in train) >= min(member["inputEventIndex"] for member in test):
        raise AssertionError("Training maturity reaches into the test input period")
    return train, test, guard, boundary


def origin_counts(members: list[dict[str, Any]]) -> dict[str, int]:
    return dict(sorted(collections.Counter(member["origin"] for member in members).items()))


def utc(timestamp_ms: int) -> str:
    return dt.datetime.fromtimestamp(timestamp_ms / 1_000, tz=dt.timezone.utc).isoformat().replace("+00:00", "Z")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--train-fraction", type=float, default=0.8)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if not 0.5 < args.train_fraction < 1:
        parser.error("--train-fraction must be strictly between 0.5 and 1")
    output = args.output.expanduser().resolve()
    if output == PRIVATE_ROOT.resolve() or not output.is_relative_to(PRIVATE_ROOT.resolve()):
        parser.error("Output must be a new private SmartApp training directory")
    if output.exists():
        parser.error("Output already exists")

    members, rejected, hashes, manifest = eligible_members(args.snapshot.resolve())
    train, test, guard, boundary = split_members(members, args.train_fraction)
    boundary_event = read_jsonl(args.snapshot.resolve() / "events.jsonl")[boundary]
    report = {
        "version": SPLIT_VERSION,
        "snapshot": str(args.snapshot.resolve()),
        "snapshotHashes": hashes,
        "requestedTrainFraction": args.train_fraction,
        "actualTrainFraction": len(train) / (len(train) + len(test)),
        "boundary": {
            "eventIndex": boundary,
            "timestampMs": boundary_event["timestampMs"],
            "timestampUTC": utc(boundary_event["timestampMs"]),
            "rule": "training samples complete before boundary; test inputs begin at or after boundary",
        },
        "eligibleChangedDestinationSamples": len(members),
        "rejectedSamples": dict(sorted(rejected.items())),
        "train": {"count": len(train), "origins": origin_counts(train)},
        "test": {"count": len(test), "origins": origin_counts(test)},
        "boundaryExcluded": {"count": len(guard), "reason": "input began before boundary but its 30-second active observation horizon completed at or after it"},
        "frontOnly": True,
        "rawSnapshotModified": False,
        "runningModelModified": False,
        "candidateListsPersisted": False,
        "modelTrainingPerformed": False,
    }
    if args.dry_run:
        print(json.dumps(report, ensure_ascii=False, sort_keys=True))
        return

    os.umask(0o077)
    output.mkdir(parents=True, mode=0o700)
    write_json(output / "split-report.json", report)
    write_jsonl(output / "train-membership-private.jsonl", train)
    write_jsonl(output / "test-membership-private.jsonl", test)
    write_jsonl(output / "boundary-excluded-private.jsonl", guard)
    source = Path(__file__)
    (output / ("source-" + source.name)).write_bytes(source.read_bytes())
    (output / ("source-" + source.name)).chmod(0o600)
    write_json(output / "run.json", {
        "version": SPLIT_VERSION,
        "complete": True,
        "snapshot": str(args.snapshot.resolve()),
        "snapshotHashes": hashes,
        "manifestSchemaVersion": manifest["schemaVersion"],
        "sampleSchemaVersion": manifest["sampleSchemaVersion"],
        "frontOnly": True,
        "rawSnapshotModified": False,
        "runningModelModified": False,
    })
    write_json(output / "artifact-hashes.json", {
        path.name: sha256(path) for path in sorted(output.iterdir()) if path.is_file()
    })
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
