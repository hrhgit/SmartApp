#!/usr/bin/env python3
"""Retrain fixed sequence models and compare hierarchical Top-4 policies.

This is a private, offline experiment.  It reuses the fixed outer split and
the architecture choices made by an earlier sealed sequence run, but it never
opens that run's held-out predictions or scores.  RNN, LSTM, and Transformer
weights are fitted again from the outer training membership.  The statistical
prior and each prior-plus-residual vector are logits; the hierarchical policy
receives their softmax only after the current destination and current owner
application have been removed.

Explicit ``parentID`` edges, URL and file-directory ancestors derived from
as-of reopenable children, the approved ChatGPT homepage relation, and an
as-of executable owner application may create a fallback. Derived parent
actions remain transient and are not independent visits or persisted candidates. Outputs contain
aggregate metrics and audit counts only.  Target identities, locators,
candidate catalogs, per-row decisions, model weights, and raw predictions are
kept transiently in memory and are never written.
"""

from __future__ import annotations

import argparse
import collections
import math
import os
import platform
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import evaluation_contract as evaluation
import hierarchical_fallback_policy as hierarchical
import run_offline_batch_rankers as runner
import run_offline_sequence_shadow as sequence


VERSION = "hierarchical-sequence-experiment-v3"
PRIVATE_ROOT = Path.home() / "Library/Application Support/SmartApp/training"
FAMILIES = ("rnn", "lstm", "transformer")
MODEL_NAMES = {
    "rnn": "rnn_residual",
    "lstm": "lstm_residual",
    "transformer": "transformer_residual",
}
ORIGINAL = "original_order"
CURRENT_FILTERED = "current_filtered_exact_order"
POLICIES = (
    hierarchical.FallbackPolicy(
        "product0.6", mode="product", content_retention=0.6,
        app_retention=evaluation.FALLBACK_GAIN,
    ),
    hierarchical.FallbackPolicy(
        "product0.8", mode="product", content_retention=0.8,
        app_retention=evaluation.FALLBACK_GAIN,
    ),
    hierarchical.FallbackPolicy(
        "saturating0.8", mode="saturating", content_retention=0.8,
        app_retention=evaluation.FALLBACK_GAIN, floor=0.4,
    ),
)


@dataclass(frozen=True)
class TransientContext:
    """Per-cutoff hierarchy retained only while aggregate scores are made."""

    group: runner.QueryGroup
    hierarchy: hierarchical.Hierarchy
    current_action: str | None
    current_app: str | None


def progress(stage: str, **details: Any) -> None:
    """Print only aggregate counts and public configuration names."""
    import json

    print(json.dumps({"stage": stage, **details}, ensure_ascii=False, sort_keys=True), flush=True)


def policy_description(policy: hierarchical.FallbackPolicy) -> dict[str, Any]:
    return {
        "name": policy.name,
        "mode": policy.mode,
        "content_retention": policy.content_retention,
        "content_retention_scope": "explicit_and_structurally_derived_url_or_directory_parent_levels",
        "service_home_retention": evaluation.FALLBACK_GAIN,
        "app_retention": policy.app_retention,
        "floor": policy.floor if policy.mode == "saturating" else None,
    }


def validate_selection_source(
    selection_run: Path, snapshot: Path, split_dir: Path,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    """Read only the old run manifest, hash manifest, and inner selection."""
    run = runner.read_json(selection_run / "run.json")
    hashes = runner.read_json(selection_run / "artifact-hashes.json")
    inner_path = selection_run / "inner-selection.json"
    if run.get("version") != sequence.VERSION or not run.get("complete"):
        raise ValueError("Selection source is not a completed sequence shadow run")
    if not run.get("test_scored_once"):
        raise ValueError("Selection source does not document its sealed test protocol")
    if Path(run.get("snapshot", "")).resolve() != snapshot:
        raise ValueError("Selection source used a different snapshot")
    if Path(run.get("split", "")).resolve() != split_dir:
        raise ValueError("Selection source used a different fixed split")
    if hashes.get("inner-selection.json") != runner.sha256(inner_path):
        raise ValueError("Selection source inner-selection hash failed")

    inner = runner.read_json(inner_path)
    if inner.get("version") != sequence.VERSION or inner.get("test_accessed") is not False:
        raise ValueError("Selection source did not select configurations before test access")
    selected = inner.get("selected")
    if not isinstance(selected, dict) or set(selected) != set(FAMILIES):
        raise ValueError("Selection source must contain exactly RNN, LSTM, and Transformer")

    stripped: dict[str, dict[str, Any]] = {}
    sequence_lengths: set[int] = set()
    bucket_counts: set[int] = set()
    for family in FAMILIES:
        item = selected[family]
        if not isinstance(item, dict) or not isinstance(item.get("config"), dict):
            raise ValueError(f"Selection source lacks a valid {family} configuration")
        config = dict(item["config"])
        if item.get("id") != config.get("id"):
            raise ValueError(f"Selection source {family} configuration ID is inconsistent")
        calibration = item.get("frozen_inner_calibration")
        scale = calibration.get("scale") if isinstance(calibration, dict) else None
        if not isinstance(scale, (int, float)) or not math.isfinite(scale) or scale < 0:
            raise ValueError(f"Selection source {family} residual scale is invalid")
        try:
            sequence_lengths.add(int(config["sequence_length"]))
            bucket_counts.add(int(config["bucket_count"]))
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"Selection source {family} lacks sequence dimensions") from error
        stripped[family] = {
            "id": str(item["id"]),
            "config": config,
            "frozen_inner_calibration_scale": float(scale),
        }
    if len(sequence_lengths) != 1 or min(sequence_lengths) < sequence.MIN_SEQUENCE_LENGTH:
        raise ValueError("Selected models must share one valid sequence length")
    if len(bucket_counts) != 1 or min(bucket_counts) < 2:
        raise ValueError("Selected models must share one valid token bucket count")
    return stripped, {
        "source_version": run["version"],
        "source_complete": True,
        "source_test_scored_once": True,
        "inner_selection_test_accessed": False,
        "inner_selection_sha256": hashes["inner-selection.json"],
        "old_test_predictions_or_metrics_read": False,
        "files_read_from_selection_run": [
            "run.json", "artifact-hashes.json", "inner-selection.json",
        ],
        "selected": stripped,
        "sequence_length": next(iter(sequence_lengths)),
        "bucket_count": next(iter(bucket_counts)),
        "input_snapshot_hashes": run.get("input_snapshot_hashes"),
    }


def normalized_truth(context: TransientContext) -> dict[str, Any]:
    truth = context.group.row.truth
    if truth is None:
        return {}
    result = dict(truth.meta)
    # Action normalization can turn a transient window into its executable app.
    # The hierarchy and shared scorer must use that same action identity.
    result["id"] = truth.key
    result["kind"] = truth.kind
    if truth.bundle:
        result["bundleID"] = truth.bundle
    return result


def build_context(group: runner.QueryGroup) -> TransientContext:
    action_keys = [action.key for action in group.actions]
    if len(action_keys) != len(set(action_keys)):
        raise ValueError("As-of executable actions are not unique")
    for action in group.actions:
        if action.meta.get("id") != action.key:
            raise ValueError("Executable action metadata is not normalized to its action identity")
    known = hierarchical.build_hierarchy(
        (action.meta for action in group.actions), executable=action_keys,
        include_owner_apps=True,
    )
    current_action = group.row.current.key if group.row.current is not None else None
    if group.row.current_parent is not None:
        current_app = group.row.current_parent.key
    elif group.row.current is not None and group.row.current.kind == "app":
        current_app = group.row.current.key
    elif group.row.current is not None and group.row.current.bundle:
        current_app = "app:" + group.row.current.bundle
    else:
        current_app = None
    return TransientContext(group, known, current_action, current_app)


def hierarchy_routes(context: TransientContext) -> set[str]:
    """Return fallback edge kinds available for the realized truth, for audit only."""
    truth = normalized_truth(context)
    truth_id = truth.get("id")
    if not truth_id:
        return set()
    expanded = hierarchical.add_outcome(context.hierarchy, truth)
    result: set[str] = set()
    for path in hierarchical.useful_fallback_actions(
        expanded, str(truth_id), POLICIES[0],
        current_action=context.current_action, current_app=context.current_app,
    ).values():
        result.update(path)
    return result


def original_output(context: TransientContext, logits: np.ndarray) -> dict[str, Any]:
    if len(logits) != len(context.group.actions) or not np.isfinite(logits).all():
        raise ValueError("Model returned invalid combined logits")
    order = sorted(
        range(len(context.group.actions)),
        key=lambda index: (-float(logits[index]), context.group.actions[index].key),
    )
    ranked = [context.group.actions[index].key for index in order]
    return {"predictions": ranked[:4], "rankable": ranked}


def current_filtered_output(context: TransientContext, logits: np.ndarray) -> dict[str, Any]:
    """Keep model order while removing actions that cannot be useful now."""
    if len(logits) != len(context.group.actions) or not np.isfinite(logits).all():
        raise ValueError("Model returned invalid combined logits")
    blocked = hierarchical.blocked_current_actions(
        context.hierarchy,
        current_action=context.current_action,
        current_app=context.current_app,
    )
    ranked = [
        action.key for index, action in sorted(
            enumerate(context.group.actions),
            key=lambda item: (-float(logits[item[0]]), item[1].key),
        )
        if action.key not in blocked
    ]
    ranked = hierarchical.deduplicate_equivalent_service_homes(
        ranked, context.hierarchy,
    )
    return {"predictions": ranked[:4], "rankable": ranked}


def greedy_menu_cached(
    posterior: Mapping[str, float], hierarchy: hierarchical.Hierarchy,
    policy: hierarchical.FallbackPolicy, *, size: int = 4,
    current_action: str | None = None, current_app: str | None = None,
    candidate_order: Sequence[str] | None = None,
) -> tuple[list[str], list[float]]:
    """Compatibility wrapper around the shared sparse greedy implementation."""
    if not posterior:
        return [], []
    return hierarchical.greedy_menu(
        posterior, hierarchy, policy, size=size,
        current_action=current_action, current_app=current_app,
        candidate_order=candidate_order,
    )


def hierarchical_output(
    context: TransientContext, logits: np.ndarray, policy: hierarchical.FallbackPolicy,
) -> dict[str, Any]:
    if len(logits) != len(context.group.actions) or not np.isfinite(logits).all():
        raise ValueError("Model returned invalid combined logits")
    blocked = hierarchical.blocked_current_actions(
        context.hierarchy,
        current_action=context.current_action,
        current_app=context.current_app,
    )
    # Removing these outcomes before softmax prevents their mass from leaking
    # into a parent recommendation.  Passing them to greedy_menu also makes the
    # action-side zero utility explicit.
    scores = {
        action.key: float(logits[index])
        for index, action in enumerate(context.group.actions)
        if action.key not in blocked
    }
    ranked = current_filtered_output(context, logits)["rankable"]
    if not scores:
        return {"predictions": [], "rankable": ranked}
    posterior = hierarchical.posterior_from_scores(scores, "logits")
    menu, _ = greedy_menu_cached(
        posterior, context.hierarchy, policy, size=4,
        current_action=context.current_action, current_app=context.current_app,
        candidate_order=ranked,
    )
    return {"predictions": menu, "rankable": ranked}


def infer_transient_outputs(
    groups: Sequence[runner.QueryGroup],
    test_sequences: Mapping[int, tuple[str, ...]],
    fitted: Mapping[str, tuple[Any, float]],
    bucket_count: int,
) -> tuple[
    dict[str, dict[str, dict[int, dict[str, Any]]]],
    dict[int, TransientContext],
    dict[str, Any],
]:
    names = ["no_context_statistical", *(MODEL_NAMES[family] for family in FAMILIES)]
    outputs = {
        name: {variant: {} for variant in (
            ORIGINAL, CURRENT_FILTERED, *(policy.name for policy in POLICIES),
        )}
        for name in names
    }
    contexts: dict[int, TransientContext] = {}
    edge_counts: collections.Counter[str] = collections.Counter()
    empty_after_current_removal = 0
    for group in groups:
        context = build_context(group)
        contexts[group.row.position] = context
        for edges in context.hierarchy.parents.values():
            edge_counts.update(edge.kind for edge in edges)
        blocked = hierarchical.blocked_current_actions(
            context.hierarchy,
            current_action=context.current_action,
            current_app=context.current_app,
        )
        empty_after_current_removal += bool(
            context.group.actions and not any(action.key not in blocked for action in context.group.actions)
        )

        logits_by_model: dict[str, np.ndarray] = {
            "no_context_statistical": np.asarray(group.prior, dtype=float),
        }
        for family in FAMILIES:
            model, scale = fitted[family]
            residual = sequence.raw_sequence_scores(model, group, test_sequences, bucket_count)
            logits_by_model[MODEL_NAMES[family]] = np.asarray(group.prior, dtype=float) + scale * residual

        for model_name, logits in logits_by_model.items():
            outputs[model_name][ORIGINAL][group.row.position] = original_output(context, logits)
            outputs[model_name][CURRENT_FILTERED][group.row.position] = current_filtered_output(
                context, logits,
            )
            for policy in POLICIES:
                outputs[model_name][policy.name][group.row.position] = hierarchical_output(
                    context, logits, policy,
                )
    return outputs, contexts, {
        "cutoffs": len(groups),
        "asof_edge_counts": dict(sorted(edge_counts.items())),
        "empty_catalog_n": sum(not group.actions for group in groups),
        "empty_after_current_outcomes_removed_n": empty_after_current_removal,
        "combined_score_semantics": "prior_plus_scaled_residual_logits",
        "posterior_conversion": "softmax_after_removing_current_semantic_destination_and_current_owner_app",
        "current_filtered_reference": "also_deduplicates_equivalent_confirmed_site_home_ids_and_backfills",
        "probability_calibration": "not_calibrated",
        "open_set_mass": {
            "enabled": False,
            "reason": (
                "No causally validated unknown-outcome estimator is available. Softmax is conditional "
                "on the remaining as-of executable actions; this limitation is audited rather than hidden."
            ),
        },
        "raw_predictions_or_candidate_lists_written": False,
    }


def emitted_service_homes(
    outputs: Mapping[str, Mapping[str, Mapping[int, Mapping[str, Any]]]],
    contexts: Mapping[int, TransientContext],
) -> list[dict[str, Any]]:
    """Return only emitted, as-of-known service home metadata for scoring."""
    homes: dict[str, dict[str, Any]] = {}
    for variants in outputs.values():
        for output in variants.values():
            for position, record in output.items():
                context = contexts[position]
                for identifier in record.get("predictions", ()):
                    meta = context.hierarchy.metadata.get(identifier)
                    if meta is None:
                        raise ValueError("An emitted action lacks as-of metadata")
                    if hierarchical.is_service_home(meta):
                        previous = homes.get(identifier)
                        if previous is not None and previous != meta:
                            raise ValueError("Emitted service-home metadata changed across cutoffs")
                        homes[identifier] = dict(meta)
    return [homes[key] for key in sorted(homes)]


def membership_hash(rows: Iterable[dict[str, Any]]) -> str:
    return evaluation.digest(sorted(row["_scoring"]["row_sha256"] for row in rows))


def build_cohorts(
    audited_rows: list[dict[str, Any]], contexts: Mapping[int, TransientContext], expected_legal_n: int,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    boards = evaluation.leaderboard_groups(audited_rows)
    all_legal = boards[evaluation.LEADERBOARD_ALL]
    if expected_legal_n and len(all_legal) != expected_legal_n:
        raise ValueError(
            f"Fixed split legal membership changed: expected {expected_legal_n}, observed {len(all_legal)}"
        )
    applicable: list[dict[str, Any]] = []
    route_counts: collections.Counter[str] = collections.Counter()
    for row in all_legal:
        routes = hierarchy_routes(contexts[row["position"]])
        if routes:
            applicable.append(row)
            route_counts.update(routes)
    cohorts = {
        "all_legal": all_legal,
        "prior_supported": boards[evaluation.LEADERBOARD_PRIOR],
        "hierarchy_applicable": applicable,
    }
    return cohorts, {
        "cohorts": {
            name: {"n": len(rows), "membership_sha256": membership_hash(rows)}
            for name, rows in cohorts.items()
        },
        "hierarchy_applicable_route_presence_n": dict(sorted(route_counts.items())),
        "membership": {
            "all_legal": "shared prediction-independent label audit",
            "prior_supported": evaluation.RULES["prior_supported_leaderboard"],
            "hierarchy_applicable": {
                "definition": (
                    "Posthoc diagnostic: the realized exact target has at least one as-of executable "
                    "explicit parent, structurally derived URL/directory ancestor, approved homepage, "
                    "equivalent homepage, or "
                    "owner app with positive fallback utility after current-destination/current-app "
                    "exclusion."
                ),
                "uses_realized_truth_only_for_posthoc_grouping": True,
                "never_filters_predictions_or_changes_other_cohorts": True,
            },
        },
    }


def canonical_metrics(
    cohorts: Mapping[str, list[dict[str, Any]]],
    outputs: Mapping[str, Mapping[str, dict[int, dict[str, Any]]]],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for cohort_name, rows in cohorts.items():
        result[cohort_name] = {}
        for model_name, variants in outputs.items():
            measured = {
                variant: evaluation.measure(rows, model_output)
                for variant, model_output in variants.items()
            }
            evaluation.require_comparable(*measured.values())
            result[cohort_name][model_name] = measured
    return {
        "version": VERSION,
        "metric_status": "canonical_shared_scoring_contract",
        "scoring_contract": evaluation.contract(),
        "cohorts": result,
    }


def experimental_score(
    rows: Sequence[dict[str, Any]], output: Mapping[int, dict[str, Any]],
    contexts: Mapping[int, TransientContext], policy: hierarchical.FallbackPolicy,
) -> dict[str, Any]:
    totals = collections.Counter()
    helpful = collections.Counter()
    for row in rows:
        context = contexts[row["position"]]
        score = hierarchical.experimental_rank_score(
            evaluation.prediction_at(output, row["position"])["predictions"],
            normalized_truth(context), context.hierarchy, policy,
            current_action=context.current_action, current_app=context.current_app, cutoff=4,
        )
        for metric, value in score.items():
            totals[metric] += value
            helpful[metric] += int(value > 0)
    n = len(rows)
    return {
        "n": n,
        **{
            metric: {
                "gain_total": totals[metric],
                "mean_gain": totals[metric] / n if n else None,
                "helpful_n": helpful[metric],
                "helpful": helpful[metric] / n if n else None,
            }
            for metric in ("top1", "top2", "top4", "rank_score")
        },
    }


def experimental_metrics(
    cohorts: Mapping[str, list[dict[str, Any]]],
    outputs: Mapping[str, Mapping[str, dict[int, dict[str, Any]]]],
    contexts: Mapping[int, TransientContext],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for cohort_name, rows in cohorts.items():
        result[cohort_name] = {}
        for model_name, variants in outputs.items():
            result[cohort_name][model_name] = {}
            for policy in POLICIES:
                result[cohort_name][model_name][policy.name] = {
                    ORIGINAL: experimental_score(rows, variants[ORIGINAL], contexts, policy),
                    CURRENT_FILTERED: experimental_score(
                        rows, variants[CURRENT_FILTERED], contexts, policy,
                    ),
                    "hierarchical_order": experimental_score(
                        rows, variants[policy.name], contexts, policy,
                    ),
                }
    return {
        "version": VERSION,
        "metric_status": "experimental_multilevel_utility_not_canonical_accuracy",
        "policies": [policy_description(policy) for policy in POLICIES],
        "definition": (
            "For each prefix, keep the single best exact-or-ancestor utility for one realized need; "
            "homepage, parent, and child utility are never added together. RankScore averages Top1 "
            "through Top4 prefix gains."
        ),
        "probability_status": "softmax_of_combined_logits_not_calibrated",
        "cohorts": result,
    }


def paired_binary(
    rows: Sequence[dict[str, Any]], before: Mapping[int, dict[str, Any]],
    after: Mapping[int, dict[str, Any]], level: str,
) -> dict[str, Any]:
    counts = {k: collections.Counter() for k in (1, 2, 4)}
    for row in rows:
        left = evaluation.hits(row, evaluation.prediction_at(before, row["position"]))[level]
        right = evaluation.hits(row, evaluation.prediction_at(after, row["position"]))[level]
        for k in counts:
            counts[k]["helped"] += right[k] and not left[k]
            counts[k]["hurt"] += left[k] and not right[k]
            counts[k]["unchanged"] += left[k] == right[k]
            counts[k]["reference_hits"] += left[k]
            counts[k]["alternative_hits"] += right[k]
    n = len(rows)
    return {
        "n": n,
        **{
            f"top{k}": {
                **dict(counts[k]),
                "net_hits": counts[k]["helped"] - counts[k]["hurt"],
                "delta_percentage_points": (
                    100 * (counts[k]["helped"] - counts[k]["hurt"]) / n if n else None
                ),
            }
            for k in (1, 2, 4)
        },
    }


def paired_experimental(
    rows: Sequence[dict[str, Any]], before: Mapping[int, dict[str, Any]],
    after: Mapping[int, dict[str, Any]], contexts: Mapping[int, TransientContext],
    policy: hierarchical.FallbackPolicy,
) -> dict[str, Any]:
    metrics = ("top1", "top2", "top4", "rank_score")
    counts = {metric: collections.Counter() for metric in metrics}
    totals = {metric: [0.0, 0.0] for metric in metrics}
    for row in rows:
        context = contexts[row["position"]]
        truth = normalized_truth(context)
        left = hierarchical.experimental_rank_score(
            evaluation.prediction_at(before, row["position"])["predictions"],
            truth, context.hierarchy, policy,
            current_action=context.current_action, current_app=context.current_app, cutoff=4,
        )
        right = hierarchical.experimental_rank_score(
            evaluation.prediction_at(after, row["position"])["predictions"],
            truth, context.hierarchy, policy,
            current_action=context.current_action, current_app=context.current_app, cutoff=4,
        )
        for metric in metrics:
            delta = right[metric] - left[metric]
            counts[metric]["helped"] += delta > 1e-12
            counts[metric]["hurt"] += delta < -1e-12
            counts[metric]["unchanged"] += abs(delta) <= 1e-12
            totals[metric][0] += left[metric]
            totals[metric][1] += right[metric]
    n = len(rows)
    return {
        "n": n,
        **{
            metric: {
                **dict(counts[metric]),
                "reference_gain_total": totals[metric][0],
                "alternative_gain_total": totals[metric][1],
                "delta_gain_total": totals[metric][1] - totals[metric][0],
                "delta_mean_gain": (
                    (totals[metric][1] - totals[metric][0]) / n if n else None
                ),
            }
            for metric in metrics
        },
    }


def paired_metrics(
    cohorts: Mapping[str, list[dict[str, Any]]],
    outputs: Mapping[str, Mapping[str, dict[int, dict[str, Any]]]],
    contexts: Mapping[int, TransientContext],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for cohort_name, rows in cohorts.items():
        result[cohort_name] = {}
        for model_name, variants in outputs.items():
            result[cohort_name][model_name] = {}
            for policy in POLICIES:
                original = variants[ORIGINAL]
                filtered = variants[CURRENT_FILTERED]
                hierarchical_order = variants[policy.name]

                def comparison(before: Mapping[int, dict[str, Any]],
                               after: Mapping[int, dict[str, Any]]) -> dict[str, Any]:
                    return {
                        "canonical_accepted": evaluation.paired_topk(rows, before, after),
                        "canonical_exact": paired_binary(rows, before, after, "exact"),
                        "experimental_multilevel": paired_experimental(
                            rows, before, after, contexts, policy,
                        ),
                    }

                result[cohort_name][model_name][policy.name] = {
                    "current_suppression": comparison(original, filtered),
                    "parent_aggregation": comparison(filtered, hierarchical_order),
                    "total": comparison(original, hierarchical_order),
                }
    return {
        "version": VERSION,
        "comparisons": {
            "current_suppression": [ORIGINAL, CURRENT_FILTERED],
            "parent_aggregation": [CURRENT_FILTERED, "hierarchical_order_for_named_policy"],
            "total": [ORIGINAL, "hierarchical_order_for_named_policy"],
        },
        "cohorts": result,
    }


def source_artifacts(output: Path) -> None:
    sources = (
        Path(__file__),
        SCRIPT_DIR / "evaluation_contract.py",
        SCRIPT_DIR / "hierarchical_fallback_policy.py",
        SCRIPT_DIR / "prepare_offline_batch_split.py",
        SCRIPT_DIR / "run_offline_batch_rankers.py",
        SCRIPT_DIR / "run_offline_sequence_shadow.py",
    )
    for source in sources:
        target = output / ("source-" + source.name)
        if target.exists():
            raise ValueError("Duplicate source artifact")
        shutil.copyfile(source, target)
        target.chmod(0o600)


def artifact_hashes(output: Path) -> dict[str, str]:
    return {
        str(path.relative_to(output)): runner.sha256(path)
        for path in sorted(output.rglob("*"))
        if path.is_file() and path.name not in {"artifact-hashes.json", "run.json"}
    }


def environment() -> dict[str, Any]:
    torch, _, _ = sequence.require_torch()
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "modules": {
            "numpy": np.__version__,
            "scipy": __import__("scipy").__version__,
            "torch": torch.__version__,
        },
        "torch_device": "cpu",
    }


def synthetic_self_test() -> dict[str, Any]:
    """Tiny no-training check used by maintainers after syntax compilation."""
    targets = [
        {"id": "parent", "kind": "directory"},
        {"id": "child-a", "kind": "directory", "parentID": "parent"},
        {"id": "child-b", "kind": "directory", "parentID": "parent"},
    ]
    tree = hierarchical.build_hierarchy(targets, executable=[item["id"] for item in targets])
    posterior = hierarchical.posterior_from_scores(
        {"parent": -4.0, "child-a": 0.0, "child-b": -0.1}, "logits",
    )
    menu, values = greedy_menu_cached(posterior, tree, POLICIES[1], size=2)
    reference_menu, reference_values = hierarchical.greedy_menu(
        posterior, tree, POLICIES[1], size=2,
    )
    if menu != reference_menu or not np.allclose(values, reference_values, atol=1e-12, rtol=1e-12):
        raise AssertionError("Cached and public greedy menu implementations disagree")
    if menu[0] != "parent" or len(values) != 2 or values[1] + 1e-12 < values[0]:
        raise AssertionError("Synthetic hierarchy policy check failed")
    current_menu, _ = greedy_menu_cached(
        posterior, tree, POLICIES[1], size=2, current_action="parent",
    )
    if "parent" in current_menu:
        raise AssertionError("Current action was not removed")
    return {"status": "ok", "policy": POLICIES[1].name, "menu_size": len(menu)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--selection-run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-legal-n", type=int, default=139)
    args = parser.parse_args()

    snapshot = args.snapshot.expanduser().resolve()
    split_dir = args.split.expanduser().resolve()
    selection_run = args.selection_run.expanduser().resolve()
    output = args.output.expanduser().resolve()
    private_root = PRIVATE_ROOT.resolve()
    if args.expected_legal_n < 0:
        parser.error("--expected-legal-n cannot be negative")
    if output == private_root or not output.is_relative_to(private_root):
        parser.error("Output must be a new private SmartApp training directory")
    if output.exists():
        parser.error("Output directory must not already exist")
    try:
        sequence.require_torch()
        selected, selection_audit = validate_selection_source(
            selection_run, snapshot, split_dir,
        )
    except (RuntimeError, ValueError, OSError) as error:
        parser.error(str(error))

    sequence_length = int(selection_audit["sequence_length"])
    bucket_count = int(selection_audit["bucket_count"])
    train, test, history, input_audit = runner.load_observations(snapshot, split_dir)
    if selection_audit.get("input_snapshot_hashes") != input_audit["snapshot_hashes"]:
        raise ValueError("Selection source snapshot hashes no longer match the fixed input")
    events = runner.read_jsonl(snapshot / "events.jsonl")
    prefixes = sequence.build_prefix_index(events, history, sequence_length)
    train_attachments = sequence.attach_sequences(train, prefixes)

    os.umask(0o077)
    output.mkdir(parents=True, mode=0o700)
    output.chmod(0o700)
    plan = {
        "version": VERSION,
        "scope": "local_offline_research_only",
        "foreground_only": True,
        "runtime_or_online_model_modified": False,
        "configuration_selection": (
            "Reuse only configurations and frozen inner residual scales from the sealed inner-selection "
            "artifact; do not read old test predictions or test metrics."
        ),
        "training": (
            "Rebuild the no-context prior from outer train and refit RNN, LSTM, and Transformer once "
            "on causally valid outer-train sequence examples."
        ),
        "combined_scores": "statistical prior logits plus frozen-scale neural residual logits",
        "hierarchical_probability_input": (
            "Softmax of combined logits after current semantic action and current owner app are removed; "
            "the resulting probabilities are explicitly uncalibrated."
        ),
        "ablation": (
            "After current semantic destination/application removal, keep one model-ranked ID per "
            "equivalent confirmed site homepage and backfill the reference menu. Compare hierarchical parent "
            "aggregation against that reference separately from the untouched model."
        ),
        "canonical_service_metadata": (
            "Only service-home actions actually emitted by a variant are passed transiently to the "
            "shared scorer; they are never persisted as candidates."
        ),
        "allowed_hierarchy": [
            "explicit_parentID", "structurally_derived_same_origin_url_ancestor",
            "structurally_derived_file_directory_ancestor", "approved_service_home", "owner_app",
        ],
        "forbidden_hierarchy_inference": [
            "same_site_without_path_ancestry", "arbitrary_url_path_truncation",
            "filesystem_scan", "background_inventory",
        ],
        "policies": [policy_description(policy) for policy in POLICIES],
        "cohorts": ["all_legal", "prior_supported", "hierarchy_applicable"],
        "heldout_protocol": "one model inference pass; all orderings are derived transiently from the same logits",
        "persistence": {
            "aggregate_metrics_only": True,
            "target_ids": False,
            "locators": False,
            "candidate_lists": False,
            "raw_predictions": False,
            "model_weights": False,
        },
    }
    runner.write_private_json(output / "plan.json", plan)
    runner.write_private_json(output / "input-audit.json", input_audit)
    runner.write_private_json(output / "selection-source-audit.json", selection_audit)
    runner.write_private_json(output / "sequence-input-audit.json", {
        "prefix_construction": prefixes.audit,
        "train_attachment": train_attachments.audit,
        "sequence_length": sequence_length,
        "bucket_count": bucket_count,
        "raw_target_or_locator_persisted": False,
    })
    runner.write_private_json(output / "environment.json", environment())
    source_artifacts(output)
    progress(
        "input_verified", train_members=len(train), test_members=len(test),
        selected={family: selected[family]["id"] for family in FAMILIES},
    )

    final_groups, final_group_audit = runner.build_causal_training_groups(train, history)
    final_examples, final_example_audit = sequence.examples_from_groups(
        final_groups, train_attachments.eligible, bucket_count,
    )
    fitted: dict[str, tuple[Any, float]] = {}
    final_training: dict[str, Any] = {}
    for family in FAMILIES:
        item = selected[family]
        model, training = sequence.train_sequence_model(family, item["config"], final_examples)
        scale = float(item["frozen_inner_calibration_scale"])
        fitted[family] = (model, scale)
        final_training[family] = {
            "configuration_id": item["id"],
            "config": item["config"],
            "training": training,
            "frozen_inner_calibration_scale": scale,
        }
    runner.write_private_json(output / "final-training.json", {
        "no_context_statistical": "rebuilt from all mature outer-train arrivals",
        "ranker_groups": len(final_groups),
        "ranker_group_audit": final_group_audit,
        "sequence_examples": final_example_audit,
        "models": final_training,
        "model_weights_persisted": False,
    })
    progress("final_models_fit", sequence_training_examples=len(final_examples))

    # The first and only held-out inference pass starts here.  All exact action
    # keys and ranked menus remain transient Python objects until aggregate
    # scoring is complete.
    final_stats = runner.HistoryStats.from_rows(train)
    test_groups = runner.build_static_groups(test, final_stats, history)
    test_attachments = sequence.attach_sequences(test, prefixes)
    outputs, contexts, inference_audit = infer_transient_outputs(
        test_groups, test_attachments.eligible, fitted, bucket_count,
    )
    emitted_homes = emitted_service_homes(outputs, contexts)
    audited_test = runner.evaluation_rows(test, emitted_homes)
    cohorts, cohort_audit = build_cohorts(
        audited_test, contexts, args.expected_legal_n,
    )
    inference_audit["test_attachment"] = test_attachments.audit
    inference_audit["test_members"] = len(test)
    inference_audit["test_groups_with_actions"] = sum(bool(group.actions) for group in test_groups)
    inference_audit["test_truth_in_asof_action_catalog"] = sum(
        group.truth_index is not None for group in test_groups
    )
    inference_audit["emitted_service_home_metadata_n"] = len(emitted_homes)
    inference_audit["emitted_scoring_metadata_persisted"] = False
    runner.write_private_json(output / "hierarchy-audit.json", {
        **inference_audit,
        **cohort_audit,
    })

    canonical = canonical_metrics(cohorts, outputs)
    experimental = experimental_metrics(cohorts, outputs, contexts)
    paired = paired_metrics(cohorts, outputs, contexts)
    runner.write_private_json(output / "canonical-metrics.json", canonical)
    runner.write_private_json(output / "experimental-multilevel-metrics.json", experimental)
    runner.write_private_json(output / "paired-help-harm.json", paired)
    summary = {
        "version": VERSION,
        "complete": True,
        "test_scored_once": True,
        "test_members": len(test),
        "legal_members": len(cohorts["all_legal"]),
        "prior_supported_members": len(cohorts["prior_supported"]),
        "hierarchy_applicable_members": len(cohorts["hierarchy_applicable"]),
        "models": list(outputs),
        "orderings": [ORIGINAL, CURRENT_FILTERED, *(policy.name for policy in POLICIES)],
        "probability_status": "softmax_of_combined_logits_not_calibrated",
        "open_set_mass_estimated": False,
        "canonical_metrics_artifact": "canonical-metrics.json",
        "experimental_metrics_artifact": "experimental-multilevel-metrics.json",
        "paired_metrics_artifact": "paired-help-harm.json",
        "raw_predictions_or_candidate_lists_written": False,
        "model_weights_persisted": False,
    }
    runner.write_private_json(output / "summary.json", summary)
    runner.write_private_json(output / "artifact-hashes.json", artifact_hashes(output))
    runner.write_private_json(output / "run.json", {
        "version": VERSION,
        "complete": True,
        "test_scored_once": True,
        "input_snapshot_hashes": input_audit["snapshot_hashes"],
        "selection_inner_sha256": selection_audit["inner_selection_sha256"],
        "scoring_contract_sha256": evaluation.contract()["rules_sha256"],
        "hierarchical_policy_source_sha256": runner.sha256(
            SCRIPT_DIR / "hierarchical_fallback_policy.py"
        ),
        "foreground_only": True,
        "runtime_or_online_model_modified": False,
        "raw_predictions_or_candidate_lists_written": False,
        "model_weights_persisted": False,
    })
    progress(
        "complete", test_members=len(test), legal_members=len(cohorts["all_legal"]),
        prior_supported_members=len(cohorts["prior_supported"]),
        hierarchy_applicable_members=len(cohorts["hierarchy_applicable"]),
    )


if __name__ == "__main__":
    main()
