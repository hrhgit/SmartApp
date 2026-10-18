#!/usr/bin/env python3
"""Re-score frozen sequence-shadow models on a prospective evidence cohort.

This is an aggregate-only supplement to a completed ``offline-sequence-shadow``
run.  It never trains, calibrates, selects a model, or writes a candidate list
or raw prediction.  Frozen checkpoints are replayed in memory solely to reuse
the common evaluation contract.

The evidence cohort follows the existing fixed-split current-unit definition:
the current canonical unit used by the predictor must have at least one
independently confirmed foreground arrival strictly before the outer train/test
boundary.  Membership reads only the current action and pre-boundary arrivals;
it never reads the realized next target, hit/miss result, post-boundary arrival,
or an as-of candidate catalog.
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import evaluation_contract as evaluation
import run_offline_batch_rankers as runner
import run_offline_sequence_shadow as shadow


VERSION = "offline-sequence-evidence-eval-v1"
PRIVATE_ROOT = Path.home() / "Library/Application Support/SmartApp/training"


def independent_arrival_counts_before_boundary(events: list[dict[str, Any]], prefixes: shadow.PrefixIndex,
                                               boundary_event_index: int) -> tuple[collections.Counter[str], dict[str, int]]:
    """Count only independent foreground arrivals strictly before one fixed cut.

    The boundary/reset logic mirrors the frozen sequence input policy.  It
    deliberately does not treat duplicate observer messages, recovery records,
    candidate availability, or test-period actions as independent history.
    """
    if not isinstance(boundary_event_index, int) or boundary_event_index < 1:
        raise ValueError("A positive fixed outer boundary event index is required")
    counts: collections.Counter[str] = collections.Counter()
    audit: collections.Counter[str] = collections.Counter()
    current_session: str | None = None
    last_action: str | None = None
    for index, event in enumerate(events):
        session = event.get("sessionID")
        if not isinstance(session, str) or not session:
            if current_session is not None or last_action is not None:
                audit["hard_boundary_missing_session"] += 1
            current_session = None
            last_action = None
        elif current_session is None:
            current_session = session
        elif current_session != session:
            audit["hard_boundary_session_change"] += 1
            current_session = session
            last_action = None

        kind = event.get("kind")
        if kind in shadow.HARD_BOUNDARY_KINDS:
            audit["hard_boundary_" + str(kind).replace(".", "_")] += 1
            last_action = None
            if kind == "session.end":
                current_session = None
            continue
        action_key = prefixes.action_by_event_index.get(index)
        if action_key is None:
            continue
        if action_key == last_action:
            audit["consecutive_duplicate_not_independent"] += 1
            continue
        if index < boundary_event_index:
            counts[action_key] += 1
            audit["pre_boundary_independent_arrival"] += 1
        else:
            audit["post_boundary_arrival_not_used_for_membership"] += 1
        last_action = action_key
    return counts, dict(sorted(audit.items()))


def evidence_cohorts(test: list[runner.Observation], audited_rows: list[dict[str, Any]],
                     pre_boundary_arrivals: collections.Counter[str]) -> tuple[
                         dict[str, list[dict[str, Any]]], dict[str, Any], dict[int, dict[str, Any]]]:
    """Build all memberships from current action plus a frozen pre-split count."""
    by_position = {row.position: row for row in test}
    if len(by_position) != len(test) or set(by_position) != {row["position"] for row in audited_rows}:
        raise ValueError("Test observations and shared evaluation rows must align exactly")
    cohorts: dict[str, list[dict[str, Any]]] = {
        "all_fixed_test": list(audited_rows),
        "current_unit_seen_before_outer_split": [],
        "current_unit_not_seen_before_outer_split": [],
        "seen_current_unit_arrivals_1": [],
        "seen_current_unit_arrivals_2_3": [],
        "seen_current_unit_arrivals_4_plus": [],
    }
    supported_kinds = sorted(runner.ACTIONABLE_KINDS)
    for kind in supported_kinds:
        cohorts["seen_current_unit_kind_" + kind] = []
    membership: collections.Counter[str] = collections.Counter()
    seen_unit_membership: dict[int, dict[str, Any]] = {}
    for audit_row in audited_rows:
        observation = by_position[audit_row["position"]]
        current = observation.current
        if current is None or current.kind not in runner.ACTIONABLE_KINDS:
            membership["missing_or_nonunit_current_action"] += 1
            continue
        count = int(pre_boundary_arrivals[current.key])
        if count <= 0:
            cohorts["current_unit_not_seen_before_outer_split"].append(audit_row)
            membership["current_unit_not_seen_before_outer_split"] += 1
            continue
        seen_unit_membership[audit_row["position"]] = {
            "current_unit_key": current.key,
            "current_unit_kind": current.kind,
            "pre_boundary_independent_arrivals": count,
        }
        cohorts["current_unit_seen_before_outer_split"].append(audit_row)
        cohorts["seen_current_unit_kind_" + current.kind].append(audit_row)
        if count == 1:
            cohorts["seen_current_unit_arrivals_1"].append(audit_row)
            membership["seen_current_unit_arrivals_1"] += 1
        elif count <= 3:
            cohorts["seen_current_unit_arrivals_2_3"].append(audit_row)
            membership["seen_current_unit_arrivals_2_3"] += 1
        else:
            cohorts["seen_current_unit_arrivals_4_plus"].append(audit_row)
            membership["seen_current_unit_arrivals_4_plus"] += 1
        membership["current_unit_seen_before_outer_split"] += 1
    legal = [row for row in audited_rows if evaluation.eligible(row)]
    legal_evidence = [row for row in cohorts["current_unit_seen_before_outer_split"] if evaluation.eligible(row)]
    audit = {
        "definition": (
            "current canonical action unit has at least one independently confirmed foreground arrival "
            "strictly before the fixed outer train/test boundary"
        ),
        "membership_uses_current_action_only": True,
        "membership_uses_realized_next_target": False,
        "membership_uses_prediction_or_hit": False,
        "membership_uses_post_boundary_arrivals": False,
        "candidate_members": {name: len(rows) for name, rows in cohorts.items()},
        "legal_members": {
            "all_fixed_test": len(legal),
            "current_unit_seen_before_outer_split": len(legal_evidence),
        },
        "cohort_coverage_of_legal_test": len(legal_evidence) / len(legal) if legal else None,
        "membership_counts_before_label_audit": dict(sorted(membership.items())),
    }
    return cohorts, audit, seen_unit_membership


def compact_score(score: dict[str, Any]) -> dict[str, Any]:
    exact = score["exact"]
    return {
        "n": score["n"], "audit": score["audit"],
        "accepted": {f"top{k}": score[f"top{k}"] for k in (1, 2, 4)},
        "accepted_hits": {f"top{k}": score[f"top{k}_hits"] for k in (1, 2, 4)},
        "exact": {f"top{k}": exact[f"top{k}"] for k in (1, 2, 4)},
        "exact_hits": {f"top{k}": exact[f"top{k}_hits"] for k in (1, 2, 4)},
        "rank_score": score["rank_score"]["mean"],
        "prediction_coverage": score["prediction_coverage"],
        "rankable_truth_coverage": score["rankable_truth_coverage"],
    }


def per_unit_private_aggregates(rows: list[dict[str, Any]], membership: dict[int, dict[str, Any]],
                                outputs: dict[str, dict[int, dict[str, Any]]]) -> dict[str, Any]:
    """Score each prospective source unit without persisting its action identity.

    The only unit identifier written to disk is ``runner.token``'s opaque,
    stable digest. Grouping keys and rankings remain in process memory.
    """
    grouped: dict[str, dict[str, Any]] = {}
    for row in rows:
        detail = membership.get(row["position"])
        if detail is None:
            raise ValueError("Evidence membership is missing a seen current unit")
        unit_key = detail["current_unit_key"]
        item = grouped.setdefault(unit_key, {
            "kind": detail["current_unit_kind"],
            "pre_boundary_independent_arrivals": detail["pre_boundary_independent_arrivals"],
            "rows": [],
        })
        if (item["kind"] != detail["current_unit_kind"]
                or item["pre_boundary_independent_arrivals"] != detail["pre_boundary_independent_arrivals"]):
            raise ValueError("Current unit has inconsistent pre-boundary membership evidence")
        item["rows"].append(row)

    units: list[dict[str, Any]] = []
    for unit_key, item in grouped.items():
        unit_rows = item["rows"]
        scores = {name: evaluation.measure(unit_rows, model_output) for name, model_output in outputs.items()}
        evaluation.require_comparable(*scores.values())
        units.append({
            "current_unit_token": runner.token(unit_key),
            "current_unit_kind": item["kind"],
            "pre_boundary_independent_arrivals": item["pre_boundary_independent_arrivals"],
            "cohort_member_n": len(unit_rows),
            "legal_member_n": scores["no_context_statistical"]["n"],
            "models": {name: compact_score(score) for name, score in scores.items()},
            "paired_vs_no_context": {
                name: evaluation.paired_topk(unit_rows, outputs["no_context_statistical"], model_output)
                for name, model_output in outputs.items() if name != "no_context_statistical"
            },
        })
    return {
        "scope": "prospective_seen_current_unit_only",
        "current_unit_identifier": "stable opaque token; raw action ID and locator are not persisted",
        "membership_uses_realized_next_target": False,
        "membership_uses_prediction_or_hit": False,
        "membership_uses_post_boundary_arrivals": False,
        "units": sorted(units, key=lambda item: (
            item["current_unit_kind"], item["pre_boundary_independent_arrivals"], item["current_unit_token"])),
    }


def replay_signature(score: dict[str, Any]) -> dict[str, Any]:
    """Full-scope fields that must reproduce the sealed shadow result exactly."""
    exact = score["exact"]
    return {
        "top_hits": {str(k): score[f"top{k}_hits"] for k in (1, 2, 4)},
        "top": {str(k): score[f"top{k}"] for k in (1, 2, 4)},
        "exact_top_hits": {str(k): exact[f"top{k}_hits"] for k in (1, 2, 4)},
        "exact_top": {str(k): exact[f"top{k}"] for k in (1, 2, 4)},
        "rank_score": score["rank_score"], "prediction_coverage": score["prediction_coverage"],
        "rankable_truth_coverage": score["rankable_truth_coverage"],
        "comparison_key": score["comparison_key"], "audit": score["audit"],
    }


def verify_source_run(source_run: Path, snapshot: Path, split_dir: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    """Fail closed if the source run, code, or frozen artifacts drifted."""
    run = runner.read_json(source_run / "run.json")
    summary = runner.read_json(source_run / "summary.json")
    if (run.get("version") != shadow.VERSION or not run.get("complete") or not summary.get("complete")
            or not run.get("test_scored_once") or not summary.get("test_scored_once")):
        raise ValueError("Source is not a completed sealed sequence-shadow run")
    if Path(run.get("snapshot", "")).resolve() != snapshot.resolve() or Path(run.get("split", "")).resolve() != split_dir.resolve():
        raise ValueError("Source run snapshot/split do not match this fixed re-score request")
    stored_hashes = runner.read_json(source_run / "artifact-hashes.json")
    for relative, expected in stored_hashes.items():
        path = source_run / relative
        if not path.is_file() or runner.sha256(path) != expected:
            raise ValueError(f"Frozen source artifact hash failed: {relative}")
    for current in (SCRIPT_DIR / "run_offline_sequence_shadow.py", SCRIPT_DIR / "evaluation_contract.py",
                    SCRIPT_DIR / "run_offline_batch_rankers.py", SCRIPT_DIR / "prepare_offline_batch_split.py"):
        frozen = source_run / ("source-" + current.name)
        if not frozen.is_file() or runner.sha256(frozen) != runner.sha256(current):
            raise ValueError(f"Current source differs from the frozen source artifact: {current.name}")
    return run, summary


def load_frozen_scorers(source_run: Path, selected: dict[str, Any], sequences: dict[int, tuple[str, ...]]) -> dict[str, Any]:
    """Load immutable checkpoint tensors; this code path has no optimizer or fitting."""
    torch, _, _ = shadow.require_torch()
    scorers: dict[str, Any] = {"no_context_statistical": runner.baseline_correction}
    for family in ("rnn", "lstm", "transformer"):
        selected_item = selected.get(family)
        if not isinstance(selected_item, dict):
            raise ValueError(f"Source summary has no selected {family} model")
        checkpoint = source_run / "models" / (family + "-residual.pt")
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if (payload.get("version") != shadow.VERSION or payload.get("family") != family
                or payload.get("config") != selected_item.get("config")):
            raise ValueError(f"Frozen {family} checkpoint provenance mismatch")
        if payload.get("tokenization", {}).get("target_vocabulary_persisted") is not False:
            raise ValueError("Checkpoint does not satisfy the no-target-vocabulary contract")
        model = shadow.build_sequence_model(family, payload["config"]).to("cpu")
        model.load_state_dict(payload["state_dict"])
        model.eval()
        residual = shadow.SequenceResidual(
            model, sequences, int(payload["config"]["bucket_count"]),
            float(payload["frozen_inner_calibration_scale"]),
        )
        scorers[family + "_residual"] = residual.correction
    return scorers


def source_artifacts(output: Path) -> None:
    for source in (Path(__file__), SCRIPT_DIR / "run_offline_sequence_shadow.py",
                   SCRIPT_DIR / "evaluation_contract.py", SCRIPT_DIR / "run_offline_batch_rankers.py",
                   SCRIPT_DIR / "prepare_offline_batch_split.py"):
        target = output / ("source-" + source.name)
        shutil.copyfile(source, target)
        target.chmod(0o600)


def environment() -> dict[str, Any]:
    torch, _, _ = shadow.require_torch()
    freeze = subprocess.run([sys.executable, "-m", "pip", "freeze"], check=False, capture_output=True, text=True)
    return {"python": sys.version, "executable": sys.executable, "platform": platform.platform(),
            "modules": {"torch": torch.__version__}, "torch_device": "cpu",
            "pip_freeze": freeze.stdout.splitlines() if freeze.returncode == 0 else []}


def artifact_hashes(output: Path) -> dict[str, str]:
    return {str(path.relative_to(output)): runner.sha256(path)
            for path in sorted(output.rglob("*"))
            if path.is_file() and path.name not in {"artifact-hashes.json", "run.json"}}


def progress(stage: str, **details: Any) -> None:
    print(json.dumps({"stage": stage, **details}, ensure_ascii=False, sort_keys=True), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run", type=Path, required=True)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source_run = args.source_run.expanduser().resolve()
    snapshot = args.snapshot.expanduser().resolve()
    split_dir = args.split.expanduser().resolve()
    output = args.output.expanduser().resolve()
    private = PRIVATE_ROOT.resolve()
    if output == private or not output.is_relative_to(private) or output.exists():
        parser.error("Output must be a new private SmartApp training directory")
    try:
        shadow.require_torch()
    except RuntimeError as error:
        parser.error(str(error))

    source_run_meta, source_summary = verify_source_run(source_run, snapshot, split_dir)
    train, test, history, input_audit = runner.load_observations(snapshot, split_dir)
    if input_audit["snapshot_hashes"] != source_run_meta.get("input_snapshot_hashes"):
        raise ValueError("Snapshot hashes differ from the sealed source run")
    split_report = runner.read_json(split_dir / "split-report.json")
    boundary = split_report.get("boundary", {}).get("eventIndex")
    if not isinstance(boundary, int) or boundary != min(row.input_event_index for row in test):
        raise ValueError("Split outer boundary does not match the fixed held-out test start")
    events = runner.read_jsonl(snapshot / "events.jsonl")
    sequence_length = int(source_summary["selected_configurations"]["rnn"]["config"]["sequence_length"])
    prefixes = shadow.build_prefix_index(events, history, sequence_length)
    pre_boundary_arrivals, arrival_audit = independent_arrival_counts_before_boundary(events, prefixes, boundary)
    attachments = shadow.attach_sequences(test, prefixes)
    test_groups = runner.build_static_groups(test, runner.HistoryStats.from_rows(train), history)
    audited_rows = runner.evaluation_rows(test)
    cohorts, cohort_audit, seen_unit_membership = evidence_cohorts(test, audited_rows, pre_boundary_arrivals)
    progress("frozen_inputs_verified", test_members=len(test), outer_boundary_event_index=boundary,
             evidence_candidate_members=len(cohorts["current_unit_seen_before_outer_split"]))

    # All rankings below live only in memory.  The original frozen primary score
    # must reproduce before the new, membership-only cohort is summarized.
    scorers = load_frozen_scorers(source_run, source_summary["selected_configurations"], attachments.eligible)
    replay_scores, outputs, replay_audited_rows = runner.score_groups(test, test_groups, scorers)
    expected_scores = runner.read_json(source_run / "test-metrics.json")
    for name, score in replay_scores.items():
        if replay_signature(score) != replay_signature(expected_scores[name]):
            raise ValueError(f"Frozen replay drifted from original fixed test metrics: {name}")
    evaluation.require_comparable(*replay_scores.values())

    metrics: dict[str, dict[str, Any]] = {}
    paired: dict[str, dict[str, Any]] = {}
    for cohort_name, rows in cohorts.items():
        scores = {name: evaluation.measure(rows, model_output) for name, model_output in outputs.items()}
        evaluation.require_comparable(*scores.values())
        metrics[cohort_name] = scores
        paired[cohort_name] = {
            name: evaluation.paired_topk(rows, outputs["no_context_statistical"], model_output)
            for name, model_output in outputs.items() if name != "no_context_statistical"
        }
    per_unit = per_unit_private_aggregates(
        cohorts["current_unit_seen_before_outer_split"], seen_unit_membership, outputs)

    os.umask(0o077)
    output.mkdir(parents=True, mode=0o700)
    output.chmod(0o700)
    source_artifacts(output)
    runner.write_private_json(output / "plan.json", {
        "version": VERSION, "scope": "aggregate_only_fixed_model_evidence_cohort_evaluation",
        "source_run": str(source_run), "snapshot": str(snapshot), "split": str(split_dir),
        "selection_or_training_performed": False,
        "prediction_replay": "frozen checkpoints only; rankings transient in memory",
        "evidence_definition": cohort_audit["definition"],
        "outer_boundary_event_index": boundary,
        "membership_uses_realized_next_target": False,
        "membership_uses_prediction_or_hit": False,
        "membership_uses_post_boundary_arrivals": False,
        "candidate_lists_persisted": False, "raw_predictions_persisted": False,
        "runtime_or_online_model_modified": False,
    })
    runner.write_private_json(output / "input-verification.json", {
        "source_run": str(source_run), "source_run_version": source_run_meta["version"],
        "source_test_scored_once": source_run_meta["test_scored_once"],
        "snapshot_hashes": input_audit["snapshot_hashes"],
        "frozen_source_matches_current": True, "original_fixed_test_scores_reproduced": True,
    })
    runner.write_private_json(output / "evidence-membership-audit.json", {
        "outer_boundary_event_index": boundary,
        "independent_arrival_counting": arrival_audit,
        "cohort_membership": cohort_audit,
        "test_sequence_attachment": attachments.audit,
        "raw_action_or_locator_persisted": False,
    })
    runner.write_private_json(output / "metrics-by-cohort.json", metrics)
    runner.write_private_json(output / "paired-help-harm-by-cohort.json", paired)
    runner.write_private_json(output / "per-unit-private.json", per_unit)
    runner.write_private_json(output / "environment.json", environment())
    summary_cohorts = {}
    for name in ("all_fixed_test", "current_unit_seen_before_outer_split"):
        summary_cohorts[name] = {
            "models": {model: compact_score(score) for model, score in metrics[name].items()},
            "paired_vs_no_context": paired[name],
        }
    runner.write_private_json(output / "summary.json", {
        "version": VERSION, "complete": True,
        "supplemental_only_not_used_for_selection_or_training": True,
        "outer_boundary_event_index": boundary,
        "cohort_definition": cohort_audit["definition"],
        "cohort_coverage_of_legal_test": cohort_audit["cohort_coverage_of_legal_test"],
        "cohorts": summary_cohorts,
        "strata_available_in": "metrics-by-cohort.json",
        "per_unit_private_report": {
            "artifact": "per-unit-private.json", "unit_n": len(per_unit["units"]),
            "identifier": per_unit["current_unit_identifier"],
        },
        "comparison_key": replay_scores["no_context_statistical"]["comparison_key"],
        "candidate_lists_persisted": False, "raw_predictions_persisted": False,
        "runtime_or_online_model_modified": False,
    })
    runner.write_private_json(output / "artifact-hashes.json", artifact_hashes(output))
    runner.write_private_json(output / "run.json", {
        "version": VERSION, "complete": True, "source_run": str(source_run),
        "snapshot": str(snapshot), "split": str(split_dir), "input_snapshot_hashes": input_audit["snapshot_hashes"],
        "selection_or_training_performed": False, "fixed_model_predictions_replayed_in_memory": True,
        "candidate_lists_persisted": False, "raw_predictions_persisted": False,
        "runtime_or_online_model_modified": False,
    })
    progress("complete", output=str(output), legal_evidence_members=cohort_audit["legal_members"]["current_unit_seen_before_outer_split"])


if __name__ == "__main__":
    main()
