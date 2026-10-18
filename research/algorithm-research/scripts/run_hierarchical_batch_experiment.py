#!/usr/bin/env python3
"""Retrain the fixed batch rankers and compare causal hierarchy fallbacks.

This is a private, offline research entry point.  It reuses only the selected
configuration metadata from an earlier inner-selection run; no held-out
prediction is read.  Every prediction is rebuilt from the supplied snapshot
and split, kept in memory, and scored by the current shared evaluation
contract.  The additional multi-level parent utility is reported separately
as an experiment and never replaces the canonical score.

The output contains aggregate metrics, audits, provenance, and a Markdown
report.  Target identifiers, locators, raw predictions, and candidate lists
are deliberately never written.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import math
import os
import platform
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import evaluation_contract as evaluation
import hierarchical_fallback_policy as hierarchy_policy
import run_offline_batch_rankers as runner


VERSION = "hierarchical-batch-experiment-v4"
PRIVATE_ROOT = runner.PRIVATE_ROOT
MODEL_ORDER = (
    "no_context_statistical",
    "conditional_logit",
    "catboost_yetirank",
    "lightfm_hybrid",
)
LOGIT_MODELS = {"no_context_statistical", "conditional_logit"}
RANK_PROXY_MODELS = {"catboost_yetirank", "lightfm_hybrid"}
SCORE_EPSILON = 1e-12


@dataclass(frozen=True)
class PredictionContext:
    """Transient facts used to rank and score one cutoff."""

    group: runner.QueryGroup
    hierarchy: hierarchy_policy.Hierarchy
    current_action: str | None
    current_app: str | None


def _json_sha256(value: Any) -> str:
    return hashlib.sha256(json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
    ).encode("utf-8")).hexdigest()


def _finite_number(value: Any, name: str, *, positive: bool = False) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"Invalid selected {name}") from error
    if not math.isfinite(result) or (positive and result <= 0):
        raise ValueError(f"Invalid selected {name}")
    return result


def _positive_integer(value: Any, name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"Invalid selected {name}")
    try:
        result = int(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"Invalid selected {name}") from error
    if result <= 0 or result != value:
        raise ValueError(f"Invalid selected {name}")
    return result


def load_selected_configs(selection_run: Path, snapshot: Path,
                          split_dir: Path) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    """Read only sealed inner-selection metadata from the earlier batch run."""
    run_path = selection_run / "run.json"
    selection_path = selection_run / "inner-selection.json"
    hashes_path = selection_run / "artifact-hashes.json"
    for path in (run_path, selection_path, hashes_path):
        if not path.is_file():
            raise ValueError(f"Missing selection provenance artifact: {path.name}")

    source_run = runner.read_json(run_path)
    if source_run.get("version") != runner.VERSION or not source_run.get("complete"):
        raise ValueError("Selection source is not a completed offline batch run")
    if Path(source_run.get("snapshot", "")).resolve() != snapshot.resolve():
        raise ValueError("Selection source points to a different snapshot")
    if Path(source_run.get("split", "")).resolve() != split_dir.resolve():
        raise ValueError("Selection source points to a different split")

    stored_hashes = runner.read_json(hashes_path)
    selection_hash = runner.sha256(selection_path)
    if stored_hashes.get(selection_path.name) != selection_hash:
        raise ValueError("Inner-selection artifact hash failed")
    selected = runner.read_json(selection_path).get("selected")
    if not isinstance(selected, dict) or not {"linear", "catboost", "lightfm"} <= set(selected):
        raise ValueError("Inner-selection artifact lacks one or more selected model families")

    linear_entry = selected["linear"]
    cat_entry = selected["catboost"]
    light_entry = selected["lightfm"]
    if not all(isinstance(entry, dict) and isinstance(entry.get("config"), dict)
               for entry in (linear_entry, cat_entry, light_entry)):
        raise ValueError("Selected model configuration is malformed")

    linear = {
        "source_id": str(linear_entry.get("id", "")),
        "config": {"l2": _finite_number(linear_entry["config"].get("l2"), "linear l2", positive=True)},
    }
    cat_config = cat_entry["config"]
    catboost = {
        "source_id": str(cat_entry.get("id", "")),
        "config": {
            "id": str(cat_config.get("id", cat_entry.get("id", ""))),
            "depth": _positive_integer(cat_config.get("depth"), "CatBoost depth"),
            "iterations": _positive_integer(cat_config.get("iterations"), "CatBoost iterations"),
            "learning_rate": _finite_number(cat_config.get("learning_rate"), "CatBoost learning rate", positive=True),
            "l2_leaf_reg": _finite_number(cat_config.get("l2_leaf_reg"), "CatBoost L2", positive=True),
            "seed": int(cat_config.get("seed")),
        },
    }
    light_config = light_entry["config"]
    loss = light_config.get("loss")
    if loss not in {"bpr", "warp", "warp-kos"}:
        raise ValueError("Invalid selected LightFM loss")
    lightfm = {
        "source_id": str(light_entry.get("id", "")),
        "config": {
            "id": str(light_config.get("id", light_entry.get("id", ""))),
            "loss": str(loss),
            "no_components": _positive_integer(light_config.get("no_components"), "LightFM components"),
            "learning_rate": _finite_number(light_config.get("learning_rate"), "LightFM learning rate", positive=True),
            "user_alpha": _finite_number(light_config.get("user_alpha"), "LightFM user alpha", positive=True),
            "item_alpha": _finite_number(light_config.get("item_alpha"), "LightFM item alpha", positive=True),
            "epochs": _positive_integer(light_config.get("epochs"), "LightFM epochs"),
            "seed": int(light_config.get("seed")),
        },
        "scale": _finite_number(light_entry.get("scale"), "LightFM frozen calibration scale", positive=True),
    }
    for family in (linear, catboost, lightfm):
        if not family["source_id"]:
            raise ValueError("Selected model configuration lacks a stable source ID")

    configs = {"linear": linear, "catboost": catboost, "lightfm": lightfm}
    audit = {
        "selection_run": str(selection_run),
        "source_run_version": source_run["version"],
        "source_run_sha256": runner.sha256(run_path),
        "inner_selection_sha256": selection_hash,
        "artifact_hashes_sha256": runner.sha256(hashes_path),
        "files_read": ["run.json", "artifact-hashes.json", "inner-selection.json"],
        "heldout_metrics_or_predictions_read": False,
        "old_test_predictions_used": False,
        "selection_scores_reused": False,
        "selected_configuration_sha256": _json_sha256(configs),
    }
    return configs, audit


def policies() -> dict[str, hierarchy_policy.FallbackPolicy]:
    """The three requested content-parent utility curves."""
    return {
        "product0.6": hierarchy_policy.FallbackPolicy(
            "product0.6", "product", content_retention=0.6,
            app_retention=evaluation.FALLBACK_GAIN,
        ),
        "product0.8": hierarchy_policy.FallbackPolicy(
            "product0.8", "product", content_retention=0.8,
            app_retention=evaluation.FALLBACK_GAIN,
        ),
        "saturating0.8": hierarchy_policy.FallbackPolicy(
            "saturating0.8", "saturating", content_retention=0.8,
            app_retention=evaluation.FALLBACK_GAIN, floor=0.4,
        ),
    }


def _current_actions(row: runner.Observation) -> tuple[str | None, str | None]:
    current_action = row.current.key if row.current is not None else None
    if row.current_parent is not None:
        current_app = row.current_parent.key
    elif row.current is not None and row.current.kind == "app":
        current_app = row.current.key
    elif row.current is not None and row.current.bundle:
        current_app = "app:" + row.current.bundle
    else:
        current_app = None
    return current_action, current_app


def _group_hierarchy(group: runner.QueryGroup) -> hierarchy_policy.Hierarchy:
    executable = [action.key for action in group.actions]
    hierarchy = hierarchy_policy.build_hierarchy(
        [action.meta for action in group.actions], executable=executable,
        include_owner_apps=True,
    )
    original_actions = set(executable)
    derived_actions = hierarchy.executable - original_actions
    if (not original_actions.issubset(hierarchy.executable)
            or derived_actions != set(hierarchy.derived_parent_ids) & hierarchy.executable):
        raise AssertionError("Hierarchy added an action without a derived-parent locator")
    return hierarchy


def _combined_scores(group: runner.QueryGroup,
                     correction: Callable[[runner.QueryGroup], np.ndarray]) -> dict[str, float]:
    values = np.asarray(correction(group), dtype=float)
    if values.shape != group.prior.shape:
        raise ValueError("Model did not return one correction per as-of action")
    totals = group.prior + values
    if totals.size and not np.isfinite(totals).all():
        raise ValueError("Model produced nonfinite combined ranking scores")
    return {action.key: float(totals[index]) for index, action in enumerate(group.actions)}


def greedy_menu_cached(posterior: Mapping[str, float], hierarchy: hierarchy_policy.Hierarchy,
                       policy: hierarchy_policy.FallbackPolicy, *, size: int = 4,
                       current_action: str | None = None,
                       current_app: str | None = None,
                       candidate_order: Sequence[str] | None = None) -> tuple[list[str], list[float]]:
    """Compatibility wrapper around the shared sparse greedy implementation."""
    if not posterior:
        return [], []
    return hierarchy_policy.greedy_menu(
        posterior, hierarchy, policy, size=size,
        current_action=current_action, current_app=current_app,
        candidate_order=candidate_order,
    )


def build_prediction_variants(
    groups: Sequence[runner.QueryGroup],
    scorers: Mapping[str, Callable[[runner.QueryGroup], np.ndarray]],
) -> tuple[
    dict[str, dict[int, dict[str, Any]]],
    dict[int, PredictionContext],
    dict[str, dict[str, Any]],
]:
    """Create all original and hierarchy menus in memory in one test pass."""
    requested_policies = policies()
    outputs: dict[str, dict[int, dict[str, Any]]] = {}
    contexts: dict[int, PredictionContext] = {}
    variants: dict[str, dict[str, Any]] = {}

    for model in MODEL_ORDER:
        if model not in scorers:
            raise ValueError(f"Missing model scorer: {model}")
        name = model + "__original"
        outputs[name] = {}
        variants[name] = {"kind": "original", "model": model, "reference": None,
                          "probability_semantics": None, "policy": None}
        filtered = model + "__current_filtered"
        outputs[filtered] = {}
        variants[filtered] = {
            "kind": "current_filtered_exact", "model": model, "reference": name,
            "probability_semantics": None, "policy": None,
            "equivalent_service_homes_deduplicated": True,
        }

    for model in MODEL_ORDER:
        semantics = ["logits_softmax"]
        if model in RANK_PROXY_MODELS:
            semantics.append("rank_proxy")
        for policy_name in requested_policies:
            for semantic in semantics:
                name = f"{model}__{policy_name}__{semantic}"
                outputs[name] = {}
                variants[name] = {
                    "kind": "hierarchical_rerank",
                    "model": model,
                    "reference": model + "__current_filtered",
                    "probability_semantics": semantic,
                    "policy": policy_name,
                }

    for group in groups:
        position = group.row.position
        hierarchy = _group_hierarchy(group)
        current_action, current_app = _current_actions(group.row)
        contexts[position] = PredictionContext(group, hierarchy, current_action, current_app)
        for model in MODEL_ORDER:
            combined = _combined_scores(group, scorers[model])
            original = sorted(combined, key=lambda key: (-combined[key], key))
            outputs[model + "__original"][position] = {
                "predictions": original[:4], "rankable": original,
            }
            # Evaluation members are real destination changes.  Remove the
            # current semantic destination and its owning app before
            # normalization so their mass cannot make a parent look useful.
            # The reference also collapses equivalent homepage IDs in model
            # order; hierarchy gains therefore do not include duplicate-slot
            # cleanup.  The untouched original order remains a separate base.
            blocked = hierarchy_policy.blocked_current_actions(
                hierarchy, current_action=current_action, current_app=current_app,
            )
            eligible_scores = {key: value for key, value in combined.items() if key not in blocked}
            eligible_rankable_raw = sorted(
                eligible_scores, key=lambda key: (-eligible_scores[key], key)
            )
            eligible_rankable = hierarchy_policy.deduplicate_equivalent_service_homes(
                eligible_rankable_raw, hierarchy,
            )
            outputs[model + "__current_filtered"][position] = {
                "predictions": eligible_rankable[:4], "rankable": eligible_rankable,
            }
            posterior_by_semantic = {
                "logits_softmax": hierarchy_policy.posterior_from_scores(eligible_scores, "logits"),
            }
            if model in RANK_PROXY_MODELS:
                posterior_by_semantic["rank_proxy"] = (
                    hierarchy_policy.rank_proxy_posterior(eligible_scores) if eligible_scores else {}
                )
            for policy_name, policy in requested_policies.items():
                for semantic, posterior in posterior_by_semantic.items():
                    menu, _ = greedy_menu_cached(
                        posterior, hierarchy, policy, size=4,
                        current_action=current_action, current_app=current_app,
                        candidate_order=eligible_rankable,
                    )
                    name = f"{model}__{policy_name}__{semantic}"
                    outputs[name][position] = {
                        "predictions": menu, "rankable": eligible_rankable,
                    }
    return outputs, contexts, variants


def emitted_service_homes(
    outputs: Mapping[str, Mapping[int, Mapping[str, Any]]],
    contexts: Mapping[int, PredictionContext],
) -> list[dict[str, Any]]:
    """Return only emitted, as-of-known service home metadata for scoring."""
    homes: dict[str, dict[str, Any]] = {}
    for variant in outputs.values():
        for position, record in variant.items():
            context = contexts[position]
            for identifier in record.get("predictions", ()):
                meta = context.hierarchy.metadata.get(identifier)
                if meta is None:
                    raise ValueError("An emitted action lacks as-of metadata")
                if hierarchy_policy.is_service_home(meta):
                    previous = homes.get(identifier)
                    if previous is not None and previous != meta:
                        raise ValueError("Emitted service-home metadata changed across cutoffs")
                    homes[identifier] = dict(meta)
    return [homes[key] for key in sorted(homes)]


def training_path_support_audit(rows: Iterable[runner.Observation],
                                history: runner.TargetHistory) -> dict[str, Any]:
    """Aggregate derived ancestor support from outer-train arrivals only."""
    seen: set[str] = set()
    event_indices = {
        event.get("id"): index for index, event in enumerate(history.events)
        if isinstance(event.get("id"), str)
    }
    counts: collections.Counter[str] = collections.Counter()
    distances: collections.Counter[str] = collections.Counter()
    edge_counts: collections.Counter[str] = collections.Counter()
    for row in sorted(rows, key=lambda item: (
            item.completed_event_index, item.input_event_index, item.sample_id)):
        if row.outcome_event_id in seen:
            counts["duplicate_outcomes_skipped"] += 1
            continue
        seen.add(row.outcome_event_id)
        if row.truth is None:
            counts["missing_truth_skipped"] += 1
            continue
        outcome_index = event_indices.get(row.outcome_event_id)
        if outcome_index is None:
            raise ValueError("Outer-train outcome is absent from the verified event stream")
        catalog = history.catalog_at(outcome_index)
        hierarchy = hierarchy_policy.build_hierarchy(
            [action.meta for action in catalog.values()], executable=catalog.keys(),
            include_owner_apps=True,
        )
        expanded = hierarchy_policy.add_outcome(hierarchy, row.truth.meta)
        support = hierarchy_policy.path_support(
            [(row.outcome_event_id, row.truth.key, 1.0)], expanded,
        )
        counts["unique_arrivals"] += int(support["unique_outcomes"]["n"])
        counts["exact_support_total"] += int(math.fsum(support["exact"].values()))
        counts["derived_path_support_total"] += int(math.fsum(support["path"].values()))
        ancestors = list(hierarchy_policy.unique_ancestor_paths(
            expanded, row.truth.key,
        ).items())
        if ancestors:
            counts["arrivals_with_structural_ancestor"] += 1
        executable_ancestors = [(ancestor, path) for ancestor, path in ancestors
                                if ancestor in hierarchy.executable]
        if executable_ancestors:
            counts["arrivals_with_executable_ancestor"] += 1
        counts["executable_ancestor_support_total"] += len(executable_ancestors)
        for _, path in ancestors:
            distances[str(len(path))] += 1
            edge_counts.update(path)
    return {
        **dict(counts),
        "ancestor_distance_counts": dict(sorted(distances.items(), key=lambda item: int(item[0]))),
        "edge_occurrence_counts": dict(sorted(edge_counts.items())),
        "source": "outer_train_independent_confirmed_arrivals_only",
        "path_support_role": (
            "Derived frequency audit and possible future open-set allocation. Known exact-outcome "
            "posteriors are not duplicated; expected utility already aggregates their probability "
            "along structurally derived paths."
        ),
        "persisted_target_identifiers": False,
    }


def hierarchy_applicable(context: PredictionContext) -> bool:
    """Prediction-independent posthoc cohort: truth has a usable fallback action."""
    truth = context.group.row.truth
    if truth is None:
        return False
    expanded = hierarchy_policy.add_outcome(context.hierarchy, truth.meta)
    reference_policy = policies()["product0.6"]
    return bool(hierarchy_policy.useful_fallback_actions(
        expanded, truth.key, reference_policy,
        current_action=context.current_action, current_app=context.current_app,
    ))


def cohort_rows(audited_rows: Sequence[dict[str, Any]],
                contexts: Mapping[int, PredictionContext]) -> dict[str, list[dict[str, Any]]]:
    legal = [row for row in audited_rows if evaluation.eligible(row)]
    prior = [row for row in legal
             if row["_scoring"].get("prior_evidence", {}).get("supported")]
    applicable_positions = {
        position for position, context in contexts.items() if hierarchy_applicable(context)
    }
    applicable = [row for row in legal if row["position"] in applicable_positions]
    return {
        "all_legal": legal,
        "prior_supported": prior,
        "hierarchy_applicable": applicable,
    }


def canonical_metrics(
    cohorts: Mapping[str, Sequence[dict[str, Any]]],
    outputs: Mapping[str, dict[int, dict[str, Any]]],
) -> dict[str, dict[str, dict[str, Any]]]:
    result: dict[str, dict[str, dict[str, Any]]] = {}
    for cohort, rows in cohorts.items():
        board = {name: evaluation.measure(list(rows), output)
                 for name, output in outputs.items()}
        if board:
            evaluation.require_comparable(*board.values())
        result[cohort] = board
    return result


def _paired_counts(before: Mapping[int, bool], after: Mapping[int, bool]) -> dict[str, Any]:
    keys = sorted(set(before) | set(after))
    if set(before) != set(after):
        raise ValueError("Paired score membership differs")
    counts: collections.Counter[str] = collections.Counter()
    for key in keys:
        left, right = bool(before[key]), bool(after[key])
        counts["helped"] += right and not left
        counts["hurt"] += left and not right
        counts["both_correct"] += left and right
        counts["both_wrong"] += not left and not right
    n = len(keys)
    return {
        "n": n, **dict(counts), "net": counts["helped"] - counts["hurt"],
        "delta_percentage_points": (
            100.0 * (counts["helped"] - counts["hurt"]) / n if n else None
        ),
    }


def _paired_values(before: Mapping[int, float], after: Mapping[int, float]) -> dict[str, Any]:
    if set(before) != set(after):
        raise ValueError("Paired score membership differs")
    deltas = [after[key] - before[key] for key in sorted(before)]
    helped = sum(delta > SCORE_EPSILON for delta in deltas)
    hurt = sum(delta < -SCORE_EPSILON for delta in deltas)
    n = len(deltas)
    before_total = math.fsum(before.values())
    after_total = math.fsum(after.values())
    return {
        "n": n,
        "helped": helped,
        "hurt": hurt,
        "unchanged": n - helped - hurt,
        "net_helped_rows": helped - hurt,
        "before_mean": before_total / n if n else None,
        "after_mean": after_total / n if n else None,
        "mean_delta": (after_total - before_total) / n if n else None,
        "total_delta": after_total - before_total,
    }


def canonical_paired(
    cohorts: Mapping[str, Sequence[dict[str, Any]]],
    outputs: Mapping[str, dict[int, dict[str, Any]]],
    variants: Mapping[str, Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for alternative, metadata in variants.items():
        reference = metadata.get("reference")
        if not reference:
            continue
        comparisons: dict[str, Any] = {}
        for cohort, rows in cohorts.items():
            accepted = evaluation.paired_topk(
                list(rows), outputs[reference], outputs[alternative],
            )
            exact: dict[str, Any] = {}
            actionable: dict[str, Any] = {}
            rank_before: dict[int, float] = {}
            rank_after: dict[int, float] = {}
            for k in (1, 2, 4):
                exact_before: dict[int, bool] = {}
                exact_after: dict[int, bool] = {}
                gain_before: dict[int, float] = {}
                gain_after: dict[int, float] = {}
                for row in rows:
                    position = row["position"]
                    left = evaluation.prediction_at(outputs[reference], position)
                    right = evaluation.prediction_at(outputs[alternative], position)
                    exact_before[position] = evaluation.hits(row, left)["exact"][k]
                    exact_after[position] = evaluation.hits(row, right)["exact"][k]
                    gain_before[position] = evaluation.ranked_gain(row, left)["prefix_gains"][k - 1]
                    gain_after[position] = evaluation.ranked_gain(row, right)["prefix_gains"][k - 1]
                    if k == 4:
                        rank_before[position] = evaluation.ranked_gain(row, left)["score"]
                        rank_after[position] = evaluation.ranked_gain(row, right)["score"]
                exact[f"top{k}"] = _paired_counts(exact_before, exact_after)
                actionable[f"top{k}"] = _paired_values(gain_before, gain_after)
            comparisons[cohort] = {
                "accepted": accepted,
                "exact": exact,
                "canonical_actionable_gain": actionable,
                "canonical_rank_score": _paired_values(rank_before, rank_after),
            }
        result[alternative] = {
            "reference": reference,
            "model": metadata["model"],
            "policy": metadata["policy"],
            "probability_semantics": metadata["probability_semantics"],
            "cohorts": comparisons,
        }
    return result


def _aggregate_experimental(records: Mapping[int, Mapping[str, float]],
                            positions: set[int]) -> dict[str, Any]:
    selected = [records[position] for position in sorted(positions)]
    n = len(selected)
    result: dict[str, Any] = {"n": n}
    for metric in ("top1", "top2", "top4", "rank_score"):
        total = math.fsum(record[metric] for record in selected)
        result[metric + "_total"] = total
        result[metric] = total / n if n else None
        result[metric + "_helpful_n"] = sum(record[metric] > 0 for record in selected)
        if metric in {"top1", "top2", "top4"}:
            result[metric + "_success_n"] = result[metric + "_helpful_n"]
    return result


def experimental_metrics_and_pairs(
    cohorts: Mapping[str, Sequence[dict[str, Any]]],
    outputs: Mapping[str, dict[int, dict[str, Any]]],
    contexts: Mapping[int, PredictionContext],
    variants: Mapping[str, Mapping[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Score multi-level parents separately with the experimental utility."""
    requested_policies = policies()
    cohort_positions = {
        name: {row["position"] for row in rows} for name, rows in cohorts.items()
    }
    score_cache: dict[tuple[str, int, tuple[str, ...]], dict[str, float]] = {}

    def row_scores(policy_name: str, output: Mapping[int, dict[str, Any]]) -> dict[int, dict[str, float]]:
        policy = requested_policies[policy_name]
        result: dict[int, dict[str, float]] = {}
        for position in cohort_positions["all_legal"]:
            context = contexts[position]
            truth = context.group.row.truth
            if truth is None:
                raise ValueError("A legal evaluation row lacks truth metadata")
            predictions = tuple(output[position]["predictions"])
            key = (policy_name, position, predictions)
            if key not in score_cache:
                score_cache[key] = hierarchy_policy.experimental_rank_score(
                    predictions, truth.meta, context.hierarchy, policy,
                    current_action=context.current_action,
                    current_app=context.current_app,
                    cutoff=4,
                )
            result[position] = score_cache[key]
        return result

    metrics: dict[str, Any] = {name: {"models": {}} for name in requested_policies}
    pairs: dict[str, Any] = {}
    for alternative, metadata in variants.items():
        if metadata.get("kind") != "hierarchical_rerank":
            continue
        reference = metadata.get("reference")
        if not reference:
            continue
        policy_name = str(metadata["policy"])
        model = str(metadata["model"])
        semantic = str(metadata["probability_semantics"])
        before = row_scores(policy_name, outputs[reference])
        after = row_scores(policy_name, outputs[alternative])
        per_cohort: dict[str, Any] = {}
        per_pair: dict[str, Any] = {}
        for cohort, positions in cohort_positions.items():
            original_metrics = _aggregate_experimental(before, positions)
            reranked_metrics = _aggregate_experimental(after, positions)
            per_cohort[cohort] = {
                "reference": original_metrics,
                "reranked": reranked_metrics,
                "delta": {
                    metric: (
                        reranked_metrics[metric] - original_metrics[metric]
                        if reranked_metrics[metric] is not None else None
                    )
                    for metric in ("top1", "top2", "top4", "rank_score")
                },
            }
            per_pair[cohort] = {
                metric: _paired_values(
                    {position: before[position][metric] for position in positions},
                    {position: after[position][metric] for position in positions},
                )
                for metric in ("top1", "top2", "top4", "rank_score")
            }
        metrics[policy_name]["models"].setdefault(model, {})[semantic] = {
            "reference_variant": reference,
            "reranked_variant": alternative,
            "cohorts": per_cohort,
        }
        pairs[alternative] = {
            "reference": reference,
            "model": model,
            "policy": policy_name,
            "probability_semantics": semantic,
            "cohorts": per_pair,
        }
    for name, policy in requested_policies.items():
        metrics[name]["utility"] = {
            "mode": policy.mode,
            "content_retention": policy.content_retention,
            "content_retention_scope": "explicit_and_structurally_derived_url_or_directory_parent_levels",
            "service_home_retention": evaluation.FALLBACK_GAIN,
            "app_retention": policy.app_retention,
            "floor": policy.floor,
        }
    metrics["definition"] = {
        "status": "path_success_and_experimental_multilevel_utility",
        "canonical_score_replaced": False,
        "exact_gain": 1.0,
        "current_action_gain": 0.0,
        "current_app_gain": 0.0,
        "service_home_gain": evaluation.FALLBACK_GAIN,
        "service_home_gain_fixed_across_policies": True,
        "success": "TopK contains the exact destination or an as-of executable intermediate URL or other confirmed ancestor with positive utility",
        "aggregation": "best utility in each Top1, Top2, and Top4 prefix; one need is never counted twice",
    }
    return metrics, pairs


def _percentage(value: float | None) -> str:
    return "—" if value is None else f"{100.0 * value:.2f}%"


def build_report(
    configs: Mapping[str, Mapping[str, Any]],
    cohorts: Mapping[str, Sequence[dict[str, Any]]],
    canonical: Mapping[str, Mapping[str, Mapping[str, Any]]],
    experimental: Mapping[str, Any],
    variants: Mapping[str, Mapping[str, Any]],
) -> str:
    lines = [
        "# 分层回退批量模型实验",
        "",
        "本报告只包含汇总结果。目标编号、URL、路径、原始预测和候选清单都没有写入磁盘。",
        "统一评分仍使用当前 `evaluation_contract.py`；多级父页面/父目录分值单独列为试验结果。",
        "",
        "## 范围",
        "",
        f"- 全部合法样本：{len(cohorts['all_legal'])}",
        f"- 预测前已有可用依据：{len(cohorts['prior_supported'])}",
        f"- 真值有可用上级路径或等价服务首页：{len(cohorts['hierarchy_applicable'])}（路线拆分见审计）",
        f"- 条件逻辑回归配置：{configs['linear']['source_id']}",
        f"- CatBoost 配置：{configs['catboost']['source_id']}",
        f"- LightFM 配置：{configs['lightfm']['source_id']}",
        "",
        "## 统一评分",
        "",
        "这里的接受率把合法服务首页视为接受命中；首页的0.6部分收益只进入综合位置分、行动收益与损失，精确率另列。多级父目录试验分不会混入这一表。",
    ]
    ordered_variants = [name for name in variants if variants[name]["kind"] == "original"]
    ordered_variants += [name for name in variants if variants[name]["kind"] == "current_filtered_exact"]
    ordered_variants += sorted(
        name for name in variants if variants[name]["kind"] == "hierarchical_rerank"
    )
    for cohort in ("all_legal", "prior_supported", "hierarchy_applicable"):
        lines += ["", f"### {cohort}（n={len(cohorts[cohort])}）", "",
                  "| 方案 | 接受Top1 | 接受Top2 | 接受Top4 | 精确Top1 | 精确Top2 | 精确Top4 | 综合位置分 |",
                  "|---|---:|---:|---:|---:|---:|---:|---:|"]
        for name in ordered_variants:
            score = canonical[cohort][name]
            exact = score["exact"]
            lines.append(
                f"| `{name}` | {_percentage(score['top1'])} | {_percentage(score['top2'])} | "
                f"{_percentage(score['top4'])} | {_percentage(exact['top1'])} | "
                f"{_percentage(exact['top2'])} | {_percentage(exact['top4'])} | "
                f"{_percentage(score['rank_score']['mean'])} |"
            )

    lines += [
        "",
        "## 路径成功命中与部分收益",
        "",
        "本实验的 TopK 成功命中包含真实目标及预测前可用、与真实目标有可靠路径关系的中间网址或父目录；它们只得部分收益。表中“原→新”比较屏蔽当前语义动作、合并等价首页后的顺序与分层菜单；完全同一目标的成绩仍在上表单列。",
        "",
        "| 上级分值 | 模型 | 概率解释 | 样本组 | 成功Top1 原→新 | 成功Top2 原→新 | 成功Top4 原→新 | 部分收益Top1 原→新 | 部分收益Top2 原→新 | 部分收益Top4 原→新 | Top4收益变化 |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for policy_name in ("product0.6", "product0.8", "saturating0.8"):
        for model in MODEL_ORDER:
            model_results = experimental[policy_name]["models"][model]
            for semantic in sorted(model_results):
                for cohort in ("all_legal", "prior_supported", "hierarchy_applicable"):
                    comparison = model_results[semantic]["cohorts"][cohort]
                    left, right = comparison["reference"], comparison["reranked"]
                    delta = comparison["delta"]["top4"]
                    lines.append(
                        f"| {policy_name} | {model} | {semantic} | {cohort} | "
                        f"{left['top1_success_n']}/{left['n']}→{right['top1_success_n']}/{right['n']} | "
                        f"{left['top2_success_n']}/{left['n']}→{right['top2_success_n']}/{right['n']} | "
                        f"{left['top4_success_n']}/{left['n']}→{right['top4_success_n']}/{right['n']} | "
                        f"{_percentage(left['top1'])}→{_percentage(right['top1'])} | "
                        f"{_percentage(left['top2'])}→{_percentage(right['top2'])} | "
                        f"{_percentage(left['top4'])}→{_percentage(right['top4'])} | "
                        f"{_percentage(delta)} |"
                    )
    lines += [
        "",
        "## 解释限制",
        "",
        "- 第一步屏蔽当前地点、当前所属应用及当前首页的等价记录，并把非当前的等价首页按模型顺序保留一个后补足名额；分层方案随后才做父级汇总。原始顺序另行保留。",
        "- `logits_softmax` 把合并后的模型分数按 softmax 归一化。无情景统计和条件逻辑回归按这个解释；CatBoost 与 LightFM 也列出它，但不宣称已经校准。",
        "- `rank_proxy` 只按名次分配递减权重，整体缩放不改变结果。它是 CatBoost 与 LightFM 的并列敏感性检查，不是可靠后验概率。",
        "- 开放世界的未知目标质量本轮没有估计；已知候选的概率没有被偷偷重分给未知子级。",
        "- 上级路径频率只由外层训练到达派生。它不被当成新的前台到达，也不会和具体目标概率重复相加。",
        "- 三条曲线只改变普通父页面/父目录；已约定的服务首页始终是0.6，多级混合路径再乘普通父级的逐级分值。",
        "- `hierarchy_applicable` 是看过真实下一步后的诊断组，只用于分析，不能在线上提前识别为保证正确的样本。",
        "",
    ]
    return "\n".join(lines)


def _artifact_hashes(output: Path) -> dict[str, str]:
    return {
        str(path.relative_to(output)): runner.sha256(path)
        for path in sorted(output.rglob("*"))
        if path.is_file() and path.name not in {"artifact-hashes.json", "run.json"}
    }


def _module_versions() -> dict[str, Any]:
    versions: dict[str, Any] = {"python": sys.version, "platform": platform.platform()}
    for name in ("numpy", "scipy", "sklearn", "catboost", "lightfm"):
        module = __import__(name)
        versions[name] = getattr(module, "__version__", "unknown")
    return versions


def progress(stage: str, **counts: int) -> None:
    """Print counts only; never print targets, paths, or configuration values."""
    if any(isinstance(value, bool) or not isinstance(value, int) for value in counts.values()):
        raise ValueError("Progress fields must be integer counts")
    print(json.dumps({"stage": stage, **counts}, sort_keys=True), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--selection-run", type=Path, required=True,
                        help="Completed 2026-09-28 batch run used only for selected configuration metadata")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    snapshot = args.snapshot.expanduser().resolve()
    split_dir = args.split.expanduser().resolve()
    selection_run = args.selection_run.expanduser().resolve()
    output = args.output.expanduser().resolve()
    private = PRIVATE_ROOT.resolve()
    if output == private or not output.is_relative_to(private):
        parser.error("Output must be a new private SmartApp training directory")
    if output.exists():
        parser.error("Output directory must not already exist")

    try:
        import catboost  # noqa: F401
        import lightfm  # noqa: F401
    except ImportError as error:
        parser.error(f"Missing offline model dependency: {error}")

    configs, selection_audit = load_selected_configs(
        selection_run, snapshot, split_dir,
    )
    train, test, history, input_audit = runner.load_observations(snapshot, split_dir)
    progress("inputs_verified", train_members=len(train), test_members=len(test))

    os.umask(0o077)
    output.mkdir(parents=True, mode=0o700)
    output.chmod(0o700)
    try:
        plan = {
            "version": VERSION,
            "scope": "private_local_offline_research_only",
            "snapshot": str(snapshot),
            "split": str(split_dir),
            "selection_run": str(selection_run),
            "selected_configuration_source_only": True,
            "old_test_predictions_read": False,
            "test_used_for_training_or_selection": False,
            "foreground_only": True,
            "current_action_disabled": True,
            "current_app_disabled": True,
            "hierarchy_posterior_conditioning": (
                "Remove the current semantic destination and current owning app before logits-softmax "
                "or rank-proxy normalization. The filtered reference also keeps only the first "
                "model-ranked ID for each equivalent confirmed site homepage and backfills the menu; keep "
                "the untouched order too so semantic cleanup and parent aggregation stay separate."
            ),
            "candidate_lists_persisted": False,
            "raw_predictions_persisted": False,
            "target_identifiers_or_locators_persisted": False,
            "canonical_evaluation": "current evaluation_contract.py",
            "canonical_service_metadata": (
                "Only service-home actions actually emitted by a variant are passed transiently to the "
                "shared scorer; they are never persisted as candidates."
            ),
            "experimental_evaluation": "hierarchical_fallback_policy.experimental_rank_score",
            "probability_semantics": {
                "no_context_statistical": ["logits_softmax"],
                "conditional_logit": ["logits_softmax"],
                "catboost_yetirank": ["logits_softmax", "rank_proxy"],
                "lightfm_hybrid": ["logits_softmax", "rank_proxy"],
            },
            "open_set_mass": {
                "status": "not_estimated",
                "used_in_prediction": False,
                "reason": "No test truth or later behavior may estimate prediction-time unknown quality.",
            },
        }
        runner.write_private_json(output / "plan.json", plan)
        runner.write_private_json(output / "input-audit.json", input_audit)
        runner.write_private_json(output / "selection-audit.json", selection_audit)
        runner.write_private_json(output / "selected-configurations.json", configs)
        runner.write_private_json(output / "environment.json", _module_versions())

        path_audit = training_path_support_audit(train, history)
        runner.write_private_json(output / "path-support-audit.json", path_audit)

        training_groups, training_audit = runner.build_causal_training_groups(train, history)
        frozen_stats = runner.HistoryStats.from_rows(train)
        linear = runner.train_linear(training_groups, configs["linear"]["config"]["l2"])
        catboost = runner.train_catboost(training_groups, configs["catboost"]["config"])
        lightfm = runner.train_lightfm(
            train, configs["lightfm"]["config"], scale=configs["lightfm"]["scale"],
        )
        runner.write_private_json(output / "training-audit.json", {
            "groups": len(training_groups),
            "causal_group_audit": training_audit,
            "conditional_logit_optimization": linear.optimization,
            "model_parameters_persisted": False,
            "outer_train_only": True,
        })
        progress("models_retrained", training_groups=len(training_groups))

        # This is the only held-out inference pass.  Everything afterward
        # reuses these transient in-memory sequences under fixed score rules.
        test_groups = runner.build_static_groups(test, frozen_stats, history)
        scorers: dict[str, Callable[[runner.QueryGroup], np.ndarray]] = {
            "no_context_statistical": runner.baseline_correction,
            "conditional_logit": linear.correction,
            "catboost_yetirank": catboost.correction,
            "lightfm_hybrid": lightfm.correction,
        }
        outputs, contexts, variants = build_prediction_variants(test_groups, scorers)
        progress("heldout_predictions_built", test_groups=len(test_groups), variants=len(variants))

        emitted_homes = emitted_service_homes(outputs, contexts)
        audited_test = runner.evaluation_rows(test, emitted_homes)
        cohorts = cohort_rows(audited_test, contexts)
        canonical = canonical_metrics(cohorts, outputs)
        canonical_pairs = canonical_paired(cohorts, outputs, variants)
        experimental, experimental_pairs = experimental_metrics_and_pairs(
            cohorts, outputs, contexts, variants,
        )
        cohort_audit = {
            "test_members": len(test),
            "all_legal_n": len(cohorts["all_legal"]),
            "prior_supported_n": len(cohorts["prior_supported"]),
            "hierarchy_applicable_n": len(cohorts["hierarchy_applicable"]),
            "hierarchy_applicable_definition": (
                "Posthoc truth has at least one as-of executable explicit parent, structurally "
                "derived URL or directory ancestor, or equivalent service homepage other than "
                "the current destination and current owning application."
            ),
            "hierarchy_applicable_is_online_confidence_claim": False,
            "test_groups_with_actions": sum(bool(group.actions) for group in test_groups),
            "test_truth_in_asof_action_catalog": sum(group.truth_index is not None for group in test_groups),
            "test_groups_empty_after_current_removal": sum(
                bool(group.actions) and not any(
                    action.key not in hierarchy_policy.blocked_current_actions(
                        contexts[group.row.position].hierarchy,
                        current_action=contexts[group.row.position].current_action,
                        current_app=contexts[group.row.position].current_app,
                    )
                    for action in group.actions
                )
                for group in test_groups
            ),
            "prediction_variants": len(variants),
            "emitted_service_home_metadata_n": len(emitted_homes),
            "emitted_scoring_metadata_persisted": False,
            "raw_predictions_or_candidate_lists_written": False,
        }
        runner.write_private_json(output / "cohort-audit.json", cohort_audit)
        runner.write_private_json(output / "canonical-metrics.json", {
            "contract": evaluation.contract(),
            "variants": variants,
            "cohorts": canonical,
        })
        runner.write_private_json(output / "experimental-multilevel-metrics.json", experimental)
        runner.write_private_json(output / "paired-help-harm.json", {
            "canonical": canonical_pairs,
            "experimental_multilevel": experimental_pairs,
        })
        report = build_report(configs, cohorts, canonical, experimental, variants)
        runner.write_private_text(output / "report-private.md", report)

        source_hashes = {
            "run_hierarchical_batch_experiment.py": runner.sha256(Path(__file__)),
            "hierarchical_fallback_policy.py": runner.sha256(SCRIPT_DIR / "hierarchical_fallback_policy.py"),
            "evaluation_contract.py": runner.sha256(SCRIPT_DIR / "evaluation_contract.py"),
            "run_offline_batch_rankers.py": runner.sha256(SCRIPT_DIR / "run_offline_batch_rankers.py"),
        }
        artifacts = _artifact_hashes(output)
        runner.write_private_json(output / "artifact-hashes.json", artifacts)
        artifact_hash_file = runner.sha256(output / "artifact-hashes.json")
        runner.write_private_json(output / "run.json", {
            "version": VERSION,
            "complete": True,
            "snapshot": str(snapshot),
            "split": str(split_dir),
            "selection_run": str(selection_run),
            "input_snapshot_hashes": input_audit["snapshot_hashes"],
            "evaluation_rules_sha256": evaluation.contract()["rules_sha256"],
            "source_hashes": source_hashes,
            "artifact_hashes_sha256": artifact_hash_file,
            "test_members": len(test),
            "all_legal_n": len(cohorts["all_legal"]),
            "prior_supported_n": len(cohorts["prior_supported"]),
            "hierarchy_applicable_n": len(cohorts["hierarchy_applicable"]),
            "test_inference_passes": 1,
            "test_used_for_training_or_selection": False,
            "candidate_lists_persisted": False,
            "raw_predictions_persisted": False,
            "target_identifiers_or_locators_persisted": False,
            "model_parameters_persisted": False,
        })
        progress(
            "complete",
            test_members=len(test),
            all_legal=len(cohorts["all_legal"]),
            prior_supported=len(cohorts["prior_supported"]),
            hierarchy_applicable=len(cohorts["hierarchy_applicable"]),
        )
    except Exception:
        # Partial private evidence stays immutable for diagnosis.
        raise


if __name__ == "__main__":
    main()
