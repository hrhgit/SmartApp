#!/usr/bin/env python3
"""Train the selected tiny path MLP on all matured foreground history.

This is a final deployment fit, not a new holdout evaluation.  It reuses the
shared as-of input and feedback adapters from the fixed study, but includes
all eligible completed samples in the immutable snapshot plus the previously
approved first-step and strict short-gap cleaning.  One confirmed arrival
causes at most one update.  Neither candidate catalogs nor raw target IDs are
persisted in the model artifact.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Sequence

import numpy as np

import cleaned_foreground_data as cleaning
import prepare_offline_batch_split as split
import run_break_cleaning_study as old_study
import run_offline_batch_rankers as offline
import run_shared_model_input_study as study
import sequence_asof_features as asof
import shared_model_adapters as adapters
import shared_model_input as shared
import shared_online_path_mlp as online


MODEL_VERSION = "shared-online-path-mlp-v1"
RUN_VERSION = "full-history-shared-online-path-mlp-v1"
HIDDEN = 16
LEARNING_RATE = 0.001
NEGATIVES = 8
SEED = 20260929
CATEGORICAL_BUCKETS = 512


def materialize_all(snapshot: Path, events: list[dict[str, Any]],
                    members: Sequence[dict[str, Any]],
                    history: offline.TargetHistory) -> list[offline.Observation]:
    """Reconstruct every eligible complete sample, with no train/test split.

    This mirrors the fixed-split loader's observation reconstruction and uses
    its target admission and current-foreground application rules.  A sample
    is never treated as evidence about an unopened background target.
    """
    sample_rows = offline.read_jsonl(snapshot / "navigation-samples.jsonl")
    samples = {sample["id"]: sample for sample in sample_rows}
    by_event = {event["id"]: event for event in events}
    event_index = {event["id"]: index for index, event in enumerate(events)}
    if len(by_event) != len(events) or len(samples) != len(sample_rows):
        raise ValueError("Duplicate source event or sample identity")
    result: list[offline.Observation] = []
    for position, member in enumerate(members):
        sample = samples.get(member.get("sampleID"))
        anchor = by_event.get(member.get("anchorEventID"))
        if sample is None or anchor is None:
            raise ValueError("Complete member has no sample or foreground anchor")
        outcomes = sample.get("outcomes") or []
        if not outcomes or outcomes[0].get("eventID") not in by_event:
            raise ValueError("Complete member has no confirmed first outcome")
        outcome = outcomes[0]
        outcome_event = by_event[outcome["eventID"]]
        input_index = event_index[anchor["id"]]
        outcome_index = event_index[outcome["eventID"]]
        if (input_index != member.get("inputEventIndex")
                or not input_index < outcome_index <= member["completedEventIndex"]):
            raise ValueError("Complete member chronology changed")
        current_meta = history.metadata_at(anchor.get("data", {}).get("targetID"), input_index)
        truth_meta = history.metadata_at(outcome.get("targetID"), outcome_index)
        current = offline.action_from_meta(current_meta, catalog_only=False)
        truth = offline.action_from_meta(truth_meta, catalog_only=False)
        current_parent = (current if current is not None and current.kind == "app"
                          else offline.foreground_parent_action(anchor))
        truth_parent = (truth if truth is not None and truth.kind == "app"
                        else offline.foreground_parent_action(outcome_event))
        catalog = history.catalog_at(input_index)
        has_prior_exact = truth is not None and truth.key in catalog
        has_prior_parent = bool(
            truth is not None and truth.kind in offline.CONCRETE_KINDS
            and truth_parent is not None and current_parent is not None
            and truth_parent.key != current_parent.key and truth_parent.key in catalog)
        result.append(offline.Observation(
            sample_id=sample["id"], split="full_history", position=position,
            input_event_index=input_index,
            completed_event_index=member["completedEventIndex"],
            input_timestamp_ms=int(anchor["timestampMs"]),
            outcome_timestamp_ms=int(outcome_event["timestampMs"]),
            outcome_event_id=outcome["eventID"], current=current, truth=truth,
            current_parent=current_parent, truth_parent=truth_parent,
            has_prior_exact_action=has_prior_exact,
            has_prior_parent_action=has_prior_parent,
        ))
    return result


def _optimizer_state(model: online.SharedOnlineSetMLP) -> dict[str, Any]:
    """Keep Adam's accumulated state so live updates can continue the fit."""
    names = ("layer1Weights", "layer1Bias", "layer2Weights", "layer2Bias")
    parameters = (model.model[0].weight, model.model[0].bias,
                  model.model[2].weight, model.model[2].bias)
    values: dict[str, Any] = {}
    steps: set[int] = set()
    for name, parameter in zip(names, parameters):
        state = model.optimizer.state.get(parameter)
        if not state or set(state) != {"step", "exp_avg", "exp_avg_sq"}:
            raise ValueError("Adam state is incomplete after final training")
        first = state["exp_avg"].detach().cpu().numpy().astype(np.float32, copy=True)
        second = state["exp_avg_sq"].detach().cpu().numpy().astype(np.float32, copy=True)
        step = int(state["step"].item())
        if (first.shape != tuple(parameter.shape) or second.shape != first.shape
                or not np.isfinite(first).all() or not np.isfinite(second).all()
                or step <= 0):
            raise ValueError("Invalid accumulated Adam state")
        steps.add(step)
        if name == "layer2Weights":
            first = first.reshape(-1)
            second = second.reshape(-1)
        elif name == "layer2Bias":
            first = first.reshape(())
            second = second.reshape(())
        values[name] = {
            "expAvg": first.tolist(),
            "expAvgSq": second.tolist(),
            "step": step,
        }
    if len(steps) != 1 or next(iter(steps)) != model.counts["optimizer_updates"]:
        raise ValueError("Adam steps disagree with confirmed model updates")
    return values


def export_weights(model: online.SharedOnlineSetMLP, *, identity_sha256: str,
                   trained_through_event_id: str,
                   training_audit: dict[str, Any]) -> dict[str, Any]:
    """Serialize only private numeric weights and the strict feature contract."""
    first = model.model[0]
    second = model.model[2]
    w1 = first.weight.detach().cpu().numpy().astype(np.float32, copy=True)
    b1 = first.bias.detach().cpu().numpy().astype(np.float32, copy=True)
    w2 = second.weight.detach().cpu().numpy().astype(np.float32, copy=True)
    b2 = second.bias.detach().cpu().numpy().astype(np.float32, copy=True)
    expected_width = len(online.NUMERIC_FIELDS) + CATEGORICAL_BUCKETS
    if (w1.shape != (HIDDEN, expected_width) or b1.shape != (HIDDEN,)
            or w2.shape != (1, HIDDEN) or b2.shape != (1,)
            or not all(np.isfinite(value).all() for value in (w1, b1, w2, b2))):
        raise ValueError("Unexpected or nonfinite trained network parameters")
    categorical = list(shared.CANDIDATE_CATEGORICAL_COLUMNS)
    schema = {
        "featureSchema": shared.SCHEMA_VERSION,
        "numericFields": list(online.NUMERIC_FIELDS),
        "categoricalFields": categorical,
        "categoricalBuckets": CATEGORICAL_BUCKETS,
        "tokenHashPerson": offline.HASH_PERSON.decode("ascii"),
        "tokenHashBytes": 12,
        "hashPerson": online.HASH_PERSON.decode("ascii"),
        "hashBytes": 8,
        "hashIndexByteOrder": "little",
        "hashSignRule": "positive_if_low_bit_of_first_byte_is_one",
        "timezone": "America/Los_Angeles",
    }
    schema_hash = hashlib.sha256(json.dumps(
        schema, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
    return {
        "format": "online-path-mlp-f32-v1",
        "modelVersion": MODEL_VERSION,
        **schema,
        "featureSchemaSHA256": schema_hash,
        "hidden": HIDDEN,
        "learningRate": LEARNING_RATE,
        "negativeLimit": NEGATIVES,
        "seed": SEED,
        "beta1": 0.9,
        "beta2": 0.999,
        "epsilon": 1e-8,
        "layer1Weights": w1.tolist(),
        "layer1Bias": b1.tolist(),
        "layer2Weights": w2.reshape(-1).tolist(),
        "layer2Bias": float(b2[0]),
        "trainedThroughEventID": trained_through_event_id,
        "identityKeySHA256": identity_sha256,
        "trainingAudit": training_audit,
        "optimizerState": _optimizer_state(model),
    }


def score_exported(weights: dict[str, Any], features: np.ndarray) -> np.ndarray:
    """Independent NumPy forward pass for artifact round-trip verification."""
    w1 = np.asarray(weights["layer1Weights"], dtype=np.float32)
    b1 = np.asarray(weights["layer1Bias"], dtype=np.float32)
    w2 = np.asarray(weights["layer2Weights"], dtype=np.float32)
    b2 = np.float32(weights["layer2Bias"])
    return np.maximum(features @ w1.T + b1, np.float32(0)) @ w2 + b2


def train(snapshot: Path, data_root: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    snapshot_hashes, manifest = split.snapshot_hashes(snapshot)
    members, rejected, member_hashes, _ = split.eligible_members(snapshot)
    if member_hashes != snapshot_hashes:
        raise ValueError("Snapshot changed during member selection")
    events = offline.read_jsonl(snapshot / "events.jsonl")
    samples = offline.read_jsonl(snapshot / "navigation-samples.jsonl")
    if not events or len(events) != manifest["events"] or len(samples) != manifest["samples"]:
        raise ValueError("Snapshot manifest differs from raw event/sample count")
    history = offline.TargetHistory(events)
    complete = materialize_all(snapshot, events, members, history)
    derived = cleaning.derive_cleaning(
        events, samples, history, test_boundary_index=len(events),
        existing_train_rows=complete)
    cleaned, weights, cleaning_audit = old_study.training_arm(
        complete, derived, "both", boundary_index=len(events))
    rich_index = asof.build_rich_index(events, history, 16, derived.bridges)
    batch = study.build_batch(cleaned, (),
                              shared.SharedInputBuilder(events, history, rich_index),
                              boundary_index=len(events), weights=weights)
    model = online.SharedOnlineSetMLP(HIDDEN, LEARNING_RATE, negatives=NEGATIVES,
                                      seed=SEED, categorical_buckets=CATEGORICAL_BUCKETS)
    gate = adapters.MaturedFeedbackGate()
    counts: Counter[str] = Counter()
    probes: list[tuple[int, np.ndarray]] = []
    for case in sorted(batch.train, key=lambda row: (
            row.outcome.completed_event_index, row.cutoff.cutoff_event_index, row.position)):
        example = adapters.adapt_feedback_training(
            case.cutoff, case.outcome, family="candidate_mlp",
            learning_event_index=len(events), gate=gate)
        if not example.labels.useful_indices:
            counts["no_useful_action"] += 1
            continue
        update = model.learn(example.prediction, example.labels.useful_indices,
                             weight=float(case.weight))
        counts["matured_unique_arrivals"] += 1
        counts["optimizer_updates"] += bool(update["updated"])
        counts["inferred_bridge_arrivals"] += case.cutoff.sample_id.startswith("derived:")
        counts["new_exact_after_feedback"] += case.outcome.exact_candidate_id is None
        if not update["updated"]:
            counts["skipped_" + update["reason"]] += 1
        if len(probes) < 3 and case.cutoff.action_ids:
            prediction = adapters.adapt(case.cutoff, family="candidate_mlp")
            probes.append((case.position, model._candidate_matrix(prediction)[:1].numpy().copy()))
    model_audit = model.audit()
    if counts["optimizer_updates"] != model_audit.get("optimizer_updates", 0):
        raise AssertionError("Unique-arrival and model update counts disagree")
    identity_key = data_root / "identity-v2.key"
    if not identity_key.is_file():
        raise ValueError("The historical target identity key is unavailable")
    weight_audit = {
        "complete_eligible_samples": len(complete),
        "cleaned_cases": len(batch.train),
        "matured_unique_arrivals": counts["matured_unique_arrivals"],
        "optimizer_updates": counts["optimizer_updates"],
        "inferred_bridge_arrivals": counts["inferred_bridge_arrivals"],
        "new_exact_after_feedback": counts["new_exact_after_feedback"],
        "skipped_no_negative_action": counts["skipped_no_negative_action"],
        "one_pass_in_completion_order": True,
        "test_members_used_for_final_fit": True,
        "holdout_score_for_final_weights": None,
    }
    exported = export_weights(
        model, identity_sha256=offline.sha256(identity_key),
        trained_through_event_id=events[-1]["id"], training_audit=weight_audit)
    parity = []
    for position, matrix in probes:
        expected = model.model(model.torch.from_numpy(matrix)).detach().cpu().numpy().reshape(-1)
        observed = score_exported(exported, matrix).reshape(-1)
        if not np.allclose(expected, observed, rtol=1e-5, atol=1e-4):
            raise ValueError("Serialized weights do not reproduce model scores")
        parity.append({"samplePosition": position, "featureVector": matrix[0].tolist(),
                       "expectedScore": float(expected[0])})
    final_hashes, _ = split.snapshot_hashes(snapshot)
    if final_hashes != snapshot_hashes:
        raise ValueError("Source snapshot changed during full-history training")
    audit = {
        "version": RUN_VERSION,
        "snapshot": str(snapshot),
        "snapshotHashes": snapshot_hashes,
        "eventCount": len(events),
        "sampleCount": len(samples),
        "completeRejectedReasons": dict(sorted(rejected.items())),
        "cleaning": cleaning_audit,
        "cleaningDerivation": derived.audit,
        "sharedBatch": batch.audit,
        "richIndex": rich_index.audit,
        "model": model_audit,
        "fit": weight_audit,
        "rawSnapshotModified": False,
        "candidateListsPersisted": False,
        "runningModelModified": False,
    }
    return exported, audit, {"version": MODEL_VERSION, "probes": parity}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--data-root", type=Path,
                        default=Path.home() / "Library/Application Support/SmartApp")
    args = parser.parse_args()
    snapshot = args.snapshot.expanduser().resolve()
    output = args.output.expanduser().resolve()
    data_root = args.data_root.expanduser().resolve()
    training_root = (data_root / "training").resolve()
    if output == training_root or not output.is_relative_to(training_root) or output.exists():
        parser.error("Output must be a new directory inside private SmartApp training storage")
    if snapshot == data_root or not snapshot.is_relative_to(
            (data_root / "exports/context-v2").resolve()):
        parser.error("Snapshot must be an immutable private context-v2 export")
    os.umask(0o077)
    weights, audit, parity = train(snapshot, data_root)
    output.mkdir(parents=True, mode=0o700)
    offline.write_private_json(output / "weights.json", weights)
    offline.write_private_json(output / "training-audit.json", audit)
    offline.write_private_json(output / "native-parity-private.json", parity)
    source = Path(__file__)
    (output / ("source-" + source.name)).write_bytes(source.read_bytes())
    (output / ("source-" + source.name)).chmod(0o600)
    hashes = {path.name: offline.sha256(path) for path in output.iterdir()
              if path.is_file()}
    offline.write_private_json(output / "artifact-hashes.json", hashes)
    offline.write_private_json(output / "run.json", {
        "version": RUN_VERSION, "complete": True,
        "snapshot": str(snapshot), "snapshotHashes": audit["snapshotHashes"],
        "weightsSHA256": hashes["weights.json"],
        "featureSchemaSHA256": weights["featureSchemaSHA256"],
        "trainedThroughEventID": weights["trainedThroughEventID"],
        "trainingAudit": weights["trainingAudit"],
        "holdoutScoreForFinalWeights": None,
        "installedInApp": False,
    })
    print(json.dumps({"output": str(output), "eventCount": audit["eventCount"],
                      "completeEligible": weights["trainingAudit"]["complete_eligible_samples"],
                      "trainedArrivals": weights["trainingAudit"]["matured_unique_arrivals"],
                      "optimizerUpdates": weights["trainingAudit"]["optimizer_updates"]},
                     ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
