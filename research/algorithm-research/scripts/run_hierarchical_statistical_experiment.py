#!/usr/bin/env python3
"""Compare hierarchy fallback policies on fixed causal statistical models.

This private offline experiment replays the existing ``statistics/none`` and
``sequence/markov2-count`` models on the frozen outer split.  Their statistical
scores are already normalized exact-outcome distributions.  At each cutoff the
experiment removes the current destination and current owner application,
renormalizes the remaining known actions, and builds a Top-4 menu by maximizing
non-duplicated expected utility over parent paths derived from known children.

The held-out test never selects a model, curve, or threshold.  The original
order, current-filtered order, and three predeclared fallback curves are all
derived in one replay.  Only aggregate metrics, audits, source snapshots, and
hashes are written.  Target identifiers, locators, predictions, candidate
catalogs, and per-row results stay transient in memory.
"""

from __future__ import annotations

import argparse
import collections
from dataclasses import dataclass
import json
import math
import os
import platform
from pathlib import Path
import shutil
import statistics
import sys
import time
from typing import Any, Iterable, Mapping, Sequence


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import context_statistics
import evaluation_contract as evaluation
import hierarchical_fallback_policy as hierarchical
import multistep_statistical_models as sequence_models
from navigation_hierarchy import SERVICE, disposition
import run_offline_batch_rankers as offline
import compare_multistep_statistical as statistical


VERSION = "hierarchical-statistical-experiment-v4"
PRIVATE_ROOT = Path.home() / "Library/Application Support/SmartApp/training"
ORIGINAL = "original_order"
CURRENT_FILTERED = "current_filtered_exact_order"
MODEL_BASELINE = "statistics_none"
MODEL_MARKOV2 = "markov2_count"
MODEL_ORDER = (MODEL_BASELINE, MODEL_MARKOV2)
EPSILON = 1e-12

MARKOV2_CONFIG = next(
    config for config in sequence_models.configurations()
    if config["id"] == "sequence/markov2-count"
)
POLICIES = (
    hierarchical.FallbackPolicy(
        "product0.6", "product", content_retention=0.6,
        app_retention=evaluation.FALLBACK_GAIN,
    ),
    hierarchical.FallbackPolicy(
        "product0.8", "product", content_retention=0.8,
        app_retention=evaluation.FALLBACK_GAIN,
    ),
    hierarchical.FallbackPolicy(
        "saturating0.8", "saturating", content_retention=0.8,
        app_retention=evaluation.FALLBACK_GAIN, floor=0.4,
    ),
)


@dataclass(frozen=True)
class TransientContext:
    """As-of hierarchy and current actions retained only during evaluation."""

    row: statistical.PreparedRow
    hierarchy: hierarchical.Hierarchy
    current_action: str | None
    current_app: str | None


def progress(stage: str, **counts: int) -> None:
    """Print only aggregate counts; never print targets or predictions."""
    if any(isinstance(value, bool) or not isinstance(value, int) for value in counts.values()):
        raise ValueError("Progress fields must be integer counts")
    print(json.dumps({"stage": stage, **counts}, sort_keys=True), flush=True)


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


def normalized_truth(context: TransientContext) -> dict[str, Any]:
    truth = context.row.observation.truth
    if truth is None:
        return {}
    result = dict(truth.meta)
    result["id"] = truth.key
    result["kind"] = truth.kind
    if truth.bundle:
        result["bundleID"] = truth.bundle
    return result


def current_actions(row: statistical.PreparedRow) -> tuple[str | None, str | None]:
    observation = row.observation
    current_action = observation.current.key if observation.current is not None else None
    if observation.current_parent is not None:
        current_app = observation.current_parent.key
    elif observation.current is not None and observation.current.kind == "app":
        current_app = observation.current.key
    elif observation.current is not None and observation.current.bundle:
        current_app = "app:" + observation.current.bundle
    else:
        current_app = None
    return current_action, current_app


def build_context(row: statistical.PreparedRow, history: offline.TargetHistory) -> TransientContext:
    catalog = history.catalog_at(int(row.observation.input_event_index))
    tree = hierarchical.build_hierarchy(
        (action.meta for action in catalog.values()), executable=catalog.keys(),
        include_owner_apps=True,
    )
    current_action, current_app = current_actions(row)
    return TransientContext(row, tree, current_action, current_app)


def baseline_distribution(
    model: statistical.StatisticsNoneModel,
    row: statistical.PreparedRow,
    history: offline.TargetHistory,
) -> tuple[dict[str, Any], dict[str, float]]:
    observation = row.observation
    if observation.current is None or not row.sequence:
        return {"predictions": [], "rankable": []}, {}
    home = statistical.home_key(history, int(observation.input_event_index))
    prepared = model.memory.prepare(
        model.query(row), home, statistical.NoBackground(),
    )
    original = prepared.predict(statistical.BASELINE_CONFIG)
    raw = prepared.scores(statistical.BASELINE_CONFIG)
    resolved: collections.Counter[str] = collections.Counter()
    for action, probability in raw.items():
        emitted = home if action == SERVICE and home else action
        resolved[emitted] += float(probability)
    return original, dict(resolved)


def sequence_distribution(
    model: sequence_models.SequenceStatisticalModel,
    row: statistical.PreparedRow,
) -> tuple[dict[str, Any], dict[str, float]]:
    observation = row.observation
    if observation.current is None or not row.sequence:
        return {"predictions": [], "rankable": []}, {}
    query = {
        "sequence": row.sequence,
        "origin": observation.current.key,
        "timestamp": float(observation.input_timestamp_ms),
        "segment_id": row.segment_id,
    }
    raw = model.scores(query)
    usable = {
        action: float(probability) for action, probability in raw.items()
        if disposition(action, observation.current.meta) == "navigation"
    }
    ranked = sorted(usable, key=lambda action: (-usable[action], action))
    return {"predictions": ranked[:4], "rankable": ranked}, usable


def filtered_output(original: Mapping[str, Any], context: TransientContext) -> dict[str, Any]:
    blocked = hierarchical.blocked_current_actions(
        context.hierarchy,
        current_action=context.current_action,
        current_app=context.current_app,
    )
    ranked = [action for action in original.get("rankable", ()) if action not in blocked]
    ranked = hierarchical.deduplicate_equivalent_service_homes(
        ranked, context.hierarchy,
    )
    return {"predictions": ranked[:4], "rankable": ranked}


def known_posterior(
    scores: Mapping[str, float], context: TransientContext,
) -> tuple[dict[str, float], dict[str, float]]:
    """Condition exact-outcome mass on known, useful actions after suppression."""
    blocked = hierarchical.blocked_current_actions(
        context.hierarchy,
        current_action=context.current_action,
        current_app=context.current_app,
    )
    total = math.fsum(float(value) for value in scores.values())
    if any(not math.isfinite(float(value)) or value < 0 for value in scores.values()):
        raise ValueError("Statistical model returned invalid probability scores")
    retained = {
        action: float(value) for action, value in scores.items()
        if action in context.hierarchy.executable and action not in blocked and value > 0
    }
    retained_mass = math.fsum(retained.values())
    if not retained_mass:
        return {}, {
            "model_probability_mass": total,
            "retained_probability_mass": 0.0,
            "removed_probability_mass": total,
        }
    probability = {action: value / retained_mass for action, value in retained.items()}
    # Make the declared probability semantics exact despite floating-point division.
    last = sorted(probability)[-1]
    probability[last] += 1.0 - math.fsum(probability.values())
    posterior = hierarchical.posterior_from_scores(probability, "probabilities")
    return posterior, {
        "model_probability_mass": total,
        "retained_probability_mass": retained_mass,
        "removed_probability_mass": max(0.0, total - retained_mass),
    }


def hierarchical_output(
    posterior: Mapping[str, float], context: TransientContext,
    policy: hierarchical.FallbackPolicy, rankable: Sequence[str],
) -> dict[str, Any]:
    if not posterior:
        return {"predictions": [], "rankable": list(rankable)}
    menu, _ = hierarchical.greedy_menu(
        posterior, context.hierarchy, policy, size=4,
        current_action=context.current_action, current_app=context.current_app,
        candidate_order=rankable,
    )
    return {"predictions": menu, "rankable": list(rankable)}


def replay(
    training_rows: Sequence[statistical.PreparedRow],
    query_rows: Sequence[statistical.PreparedRow],
    history: offline.TargetHistory,
) -> tuple[
    dict[str, dict[str, dict[int, dict[str, Any]]]],
    dict[int, TransientContext],
    dict[str, Any],
]:
    """Learn both fixed models once and derive every ordering in one test pass."""
    models: dict[str, Any] = {
        MODEL_BASELINE: statistical.StatisticsNoneModel(),
        MODEL_MARKOV2: sequence_models.build(MARKOV2_CONFIG),
    }
    variants = (ORIGINAL, CURRENT_FILTERED, *(policy.name for policy in POLICIES))
    outputs = {
        model: {variant: {} for variant in variants}
        for model in MODEL_ORDER
    }
    stream = [row for row in training_rows if statistical.navigation_training_row(row)]
    stream.sort(key=lambda row: (
        row.observation.completed_event_index,
        row.observation.input_event_index,
        row.observation.position,
    ))
    contexts: dict[int, TransientContext] = {}
    cursor = 0
    learned = collections.Counter()
    predict_ms: dict[str, list[float]] = {model: [] for model in MODEL_ORDER}
    edge_counts: collections.Counter[str] = collections.Counter()
    mass = {
        model: collections.Counter({"rows": 0, "nonempty": 0}) for model in MODEL_ORDER
    }

    for row in sorted(query_rows, key=lambda item: (
            item.observation.input_event_index, item.observation.position)):
        cutoff = int(row.observation.input_event_index)
        now = float(row.observation.input_timestamp_ms)
        while cursor < len(stream) and stream[cursor].observation.completed_event_index < cutoff:
            mature = stream[cursor]
            if mature.observation.input_event_index >= cutoff:
                raise ValueError("A future input entered statistical training")
            if mature.observation.outcome_timestamp_ms >= now:
                raise ValueError("A future timestamp entered statistical training")
            if models[MODEL_BASELINE].learn(mature):
                learned[MODEL_BASELINE] += 1
            if models[MODEL_MARKOV2].learn(statistical.sequence_record(mature)):
                learned[MODEL_MARKOV2] += 1
            cursor += 1

        context = build_context(row, history)
        position = int(row.observation.position)
        contexts[position] = context
        for edges in context.hierarchy.parents.values():
            edge_counts.update(edge.kind for edge in edges)

        for model_name in MODEL_ORDER:
            started = time.perf_counter()
            if model_name == MODEL_BASELINE:
                original, scores = baseline_distribution(models[model_name], row, history)
            else:
                original, scores = sequence_distribution(models[model_name], row)
            filtered = filtered_output(original, context)
            posterior, row_mass = known_posterior(scores, context)
            predict_ms[model_name].append((time.perf_counter() - started) * 1000.0)
            mass[model_name]["rows"] += 1
            mass[model_name]["nonempty"] += bool(posterior)
            for key, value in row_mass.items():
                mass[model_name][key] += value

            outputs[model_name][ORIGINAL][position] = dict(original)
            outputs[model_name][CURRENT_FILTERED][position] = filtered
            for policy in POLICIES:
                outputs[model_name][policy.name][position] = hierarchical_output(
                    posterior, context, policy, filtered["rankable"],
                )

    def timing(values: Sequence[float]) -> dict[str, Any]:
        ordered = sorted(values)
        return {
            "n": len(values),
            "median_ms": statistics.median(values) if values else 0.0,
            "p95_ms": ordered[int(0.95 * (len(ordered) - 1))] if values else 0.0,
            "total_ms": math.fsum(values),
        }

    return outputs, contexts, {
        "training_stream_rows": len(stream),
        "learned_rows": dict(learned),
        "query_rows": len(query_rows),
        "asof_edge_counts": dict(sorted(edge_counts.items())),
        "prediction_timing": {model: timing(values) for model, values in predict_ms.items()},
        "probability_mass": {
            model: {
                key: (int(value) if key in {"rows", "nonempty"} else float(value))
                for key, value in values.items()
            }
            for model, values in mass.items()
        },
        "probability_semantics": "model_native_normalized_exact_outcome_distribution",
        "conditioning": "current semantic destination/current app and non-executable outcomes removed, then renormalized",
        "current_filtered_reference": "equivalent confirmed site-home IDs deduplicated in model order with slots backfilled",
        "probability_calibration": "model-native statistical estimate; calibration not separately validated",
        "open_set_mass": {
            "enabled": False,
            "reason": "No causal prediction-time estimator for unseen exact outcomes is available.",
        },
        "heldout_inference_passes": 1,
        "raw_predictions_or_candidate_lists_written": False,
    }


def emitted_service_homes(
    outputs: Mapping[str, Mapping[str, Mapping[int, Mapping[str, Any]]]],
    contexts: Mapping[int, TransientContext],
) -> list[dict[str, Any]]:
    """Keep only metadata for service homes actually emitted by a variant."""
    homes: dict[str, dict[str, Any]] = {}
    for variants in outputs.values():
        for records in variants.values():
            for position, record in records.items():
                context = contexts[position]
                for identifier in record.get("predictions", ()):
                    meta = context.hierarchy.metadata.get(identifier)
                    if meta is None or not hierarchical.is_service_home(meta):
                        continue
                    previous = homes.get(identifier)
                    if previous is not None and previous != meta:
                        raise ValueError("Emitted service-home metadata changed across cutoffs")
                    homes[identifier] = dict(meta)
    return [homes[key] for key in sorted(homes)]


def hierarchy_routes(context: TransientContext) -> set[str]:
    truth = normalized_truth(context)
    truth_id = truth.get("id")
    if not truth_id:
        return set()
    expanded = hierarchical.add_outcome(context.hierarchy, truth)
    routes: set[str] = set()
    for path in hierarchical.useful_fallback_actions(
        expanded, str(truth_id), POLICIES[0],
        current_action=context.current_action, current_app=context.current_app,
    ).values():
        routes.update(path)
    return routes


def membership_hash(rows: Iterable[dict[str, Any]]) -> str:
    return evaluation.digest(sorted(row["_scoring"]["row_sha256"] for row in rows))


def build_cohorts(
    audited: list[dict[str, Any]], contexts: Mapping[int, TransientContext],
    expected_legal_n: int,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    boards = evaluation.leaderboard_groups(audited)
    legal = boards[evaluation.LEADERBOARD_ALL]
    if expected_legal_n and len(legal) != expected_legal_n:
        raise ValueError(
            f"Fixed split legal membership changed: expected {expected_legal_n}, observed {len(legal)}"
        )
    applicable: list[dict[str, Any]] = []
    route_counts: collections.Counter[str] = collections.Counter()
    for row in legal:
        routes = hierarchy_routes(contexts[row["position"]])
        if routes:
            applicable.append(row)
            route_counts.update(routes)
    cohorts = {
        "all_legal": legal,
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
                    "Posthoc realized truth has an as-of executable explicit or structurally derived ancestor or equivalent "
                    "service homepage with positive utility after current destination/current app "
                    "suppression."
                ),
                "uses_realized_truth_only_for_diagnosis": True,
                "online_confidence_claim": False,
            },
        },
    }


def canonical_metrics(
    cohorts: Mapping[str, Sequence[dict[str, Any]]],
    outputs: Mapping[str, Mapping[str, Mapping[int, Mapping[str, Any]]]],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for cohort, rows in cohorts.items():
        result[cohort] = {}
        for model, variants in outputs.items():
            measured = {
                variant: evaluation.measure(list(rows), records)
                for variant, records in variants.items()
            }
            evaluation.require_comparable(*measured.values())
            result[cohort][model] = measured
    return {
        "version": VERSION,
        "metric_status": "canonical_shared_scoring_contract",
        "scoring_contract": evaluation.contract(),
        "cohorts": result,
    }


def experimental_row_score(
    row: dict[str, Any], output: Mapping[int, Mapping[str, Any]],
    contexts: Mapping[int, TransientContext], policy: hierarchical.FallbackPolicy,
) -> dict[str, float]:
    context = contexts[row["position"]]
    return hierarchical.experimental_rank_score(
        evaluation.prediction_at(output, row["position"])["predictions"],
        normalized_truth(context), context.hierarchy, policy,
        current_action=context.current_action, current_app=context.current_app, cutoff=4,
    )


def aggregate_experimental(
    rows: Sequence[dict[str, Any]], output: Mapping[int, Mapping[str, Any]],
    contexts: Mapping[int, TransientContext], policy: hierarchical.FallbackPolicy,
) -> dict[str, Any]:
    totals = collections.Counter()
    helpful = collections.Counter()
    for row in rows:
        score = experimental_row_score(row, output, contexts, policy)
        for metric, value in score.items():
            totals[metric] += value
            helpful[metric] += value > 0
    n = len(rows)
    return {
        "n": n,
        **{metric: totals[metric] / n if n else None
           for metric in ("top1", "top2", "top4", "rank_score")},
        **{metric + "_total": totals[metric]
           for metric in ("top1", "top2", "top4", "rank_score")},
        **{metric + "_helpful_n": helpful[metric]
           for metric in ("top1", "top2", "top4", "rank_score")},
        **{metric + "_success_n": helpful[metric]
           for metric in ("top1", "top2", "top4")},
    }


def experimental_metrics(
    cohorts: Mapping[str, Sequence[dict[str, Any]]],
    outputs: Mapping[str, Mapping[str, Mapping[int, Mapping[str, Any]]]],
    contexts: Mapping[int, TransientContext],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for cohort, rows in cohorts.items():
        result[cohort] = {}
        for model, variants in outputs.items():
            result[cohort][model] = {
                policy.name: {
                    variant: aggregate_experimental(rows, records, contexts, policy)
                    for variant, records in variants.items()
                }
                for policy in POLICIES
            }
    return {
        "version": VERSION,
        "metric_status": "path_success_and_experimental_multilevel_utility",
        "canonical_score_replaced": False,
        "policies": [policy_description(policy) for policy in POLICIES],
        "definition": (
            "TopK success includes exact actions and as-of executable URL intermediate or other "
            "confirmed ancestor actions with positive utility. Each prefix keeps the single best "
            "partial or exact utility for one need; gains are never added."
        ),
        "cohorts": result,
    }


def paired_boolean(before: Sequence[bool], after: Sequence[bool]) -> dict[str, Any]:
    if len(before) != len(after):
        raise ValueError("Paired score membership differs")
    counts: collections.Counter[str] = collections.Counter()
    for left, right in zip(before, after):
        counts["helped"] += right and not left
        counts["hurt"] += left and not right
        counts["unchanged"] += left == right
        counts["reference_hits"] += left
        counts["alternative_hits"] += right
    n = len(before)
    return {
        "n": n,
        **dict(counts),
        "net_hits": counts["helped"] - counts["hurt"],
        "delta_percentage_points": (
            100.0 * (counts["helped"] - counts["hurt"]) / n if n else None
        ),
    }


def paired_values(before: Sequence[float], after: Sequence[float]) -> dict[str, Any]:
    if len(before) != len(after):
        raise ValueError("Paired score membership differs")
    deltas = [right - left for left, right in zip(before, after)]
    helped = sum(delta > EPSILON for delta in deltas)
    hurt = sum(delta < -EPSILON for delta in deltas)
    n = len(deltas)
    left_total, right_total = math.fsum(before), math.fsum(after)
    return {
        "n": n,
        "helped": helped,
        "hurt": hurt,
        "unchanged": n - helped - hurt,
        "net_helped_rows": helped - hurt,
        "reference_mean": left_total / n if n else None,
        "alternative_mean": right_total / n if n else None,
        "mean_delta": (right_total - left_total) / n if n else None,
        "total_delta": right_total - left_total,
    }


def compare_pair(
    rows: Sequence[dict[str, Any]], before: Mapping[int, Mapping[str, Any]],
    after: Mapping[int, Mapping[str, Any]], contexts: Mapping[int, TransientContext],
    policy: hierarchical.FallbackPolicy,
) -> dict[str, Any]:
    canonical = {level: {} for level in ("accepted", "exact")}
    canonical_rank_before: list[float] = []
    canonical_rank_after: list[float] = []
    experimental_before: dict[str, list[float]] = {
        metric: [] for metric in ("top1", "top2", "top4", "rank_score")
    }
    experimental_after = {metric: [] for metric in experimental_before}
    hit_values = {
        level: {k: [[], []] for k in (1, 2, 4)}
        for level in ("accepted", "exact")
    }
    for row in rows:
        left = evaluation.prediction_at(before, row["position"])
        right = evaluation.prediction_at(after, row["position"])
        left_hits, right_hits = evaluation.hits(row, left), evaluation.hits(row, right)
        for level in ("accepted", "exact"):
            for k in (1, 2, 4):
                hit_values[level][k][0].append(left_hits[level][k])
                hit_values[level][k][1].append(right_hits[level][k])
        canonical_rank_before.append(evaluation.ranked_gain(row, left)["score"])
        canonical_rank_after.append(evaluation.ranked_gain(row, right)["score"])
        left_experimental = experimental_row_score(row, before, contexts, policy)
        right_experimental = experimental_row_score(row, after, contexts, policy)
        for metric in experimental_before:
            experimental_before[metric].append(left_experimental[metric])
            experimental_after[metric].append(right_experimental[metric])
    for level in canonical:
        canonical[level] = {
            f"top{k}": paired_boolean(*hit_values[level][k]) for k in (1, 2, 4)
        }
    return {
        "n": len(rows),
        "canonical": {
            **canonical,
            "rank_score": paired_values(canonical_rank_before, canonical_rank_after),
        },
        "experimental_multilevel": {
            metric: paired_values(experimental_before[metric], experimental_after[metric])
            for metric in experimental_before
        },
    }


def paired_metrics(
    cohorts: Mapping[str, Sequence[dict[str, Any]]],
    outputs: Mapping[str, Mapping[str, Mapping[int, Mapping[str, Any]]]],
    contexts: Mapping[int, TransientContext],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for cohort, rows in cohorts.items():
        result[cohort] = {}
        for model, variants in outputs.items():
            result[cohort][model] = {}
            for policy in POLICIES:
                original = variants[ORIGINAL]
                filtered = variants[CURRENT_FILTERED]
                reranked = variants[policy.name]
                result[cohort][model][policy.name] = {
                    "current_suppression": compare_pair(
                        rows, original, filtered, contexts, policy,
                    ),
                    "parent_aggregation": compare_pair(
                        rows, filtered, reranked, contexts, policy,
                    ),
                    "total": compare_pair(
                        rows, original, reranked, contexts, policy,
                    ),
                }
    return {
        "version": VERSION,
        "comparisons": {
            "current_suppression": [ORIGINAL, CURRENT_FILTERED],
            "parent_aggregation": [CURRENT_FILTERED, "named_hierarchy_policy"],
            "total": [ORIGINAL, "named_hierarchy_policy"],
        },
        "per_row_records_persisted": False,
        "cohorts": result,
    }


def training_path_support_audit(
    rows: Sequence[statistical.PreparedRow], history: offline.TargetHistory,
) -> dict[str, Any]:
    """Aggregate one derived support per structural ancestor, without identities."""
    event_indices = {
        event.get("id"): index for index, event in enumerate(history.events)
        if isinstance(event.get("id"), str)
    }
    seen: set[str] = set()
    counts: collections.Counter[str] = collections.Counter()
    distances: collections.Counter[str] = collections.Counter()
    edge_counts: collections.Counter[str] = collections.Counter()
    for prepared in sorted(rows, key=lambda item: (
            item.observation.completed_event_index,
            item.observation.input_event_index,
            item.observation.position)):
        row = prepared.observation
        if row.outcome_event_id in seen:
            counts["duplicate_outcomes_skipped"] += 1
            continue
        seen.add(row.outcome_event_id)
        if row.truth is None:
            counts["missing_truth_skipped"] += 1
            continue
        outcome_index = event_indices.get(row.outcome_event_id)
        if outcome_index is None:
            raise ValueError("Training outcome is absent from the verified event stream")
        catalog = history.catalog_at(outcome_index)
        tree = hierarchical.build_hierarchy(
            (action.meta for action in catalog.values()), executable=catalog.keys(),
            include_owner_apps=True,
        )
        expanded = hierarchical.add_outcome(tree, row.truth.meta)
        support = hierarchical.path_support(
            [(row.outcome_event_id, row.truth.key, 1.0)], expanded,
        )
        counts["unique_arrivals"] += int(support["unique_outcomes"]["n"])
        counts["exact_support_total"] += int(math.fsum(support["exact"].values()))
        counts["derived_path_support_total"] += int(math.fsum(support["path"].values()))
        ancestors = list(hierarchical.unique_ancestor_paths(
            expanded, row.truth.key,
        ).items())
        counts["arrivals_with_structural_ancestor"] += bool(ancestors)
        executable = [(action, path) for action, path in ancestors if action in tree.executable]
        counts["arrivals_with_executable_ancestor"] += bool(executable)
        counts["executable_ancestor_support_total"] += len(executable)
        for _, path in ancestors:
            distances[str(len(path))] += 1
            edge_counts.update(path)
    return {
        **dict(counts),
        "ancestor_distance_counts": dict(sorted(distances.items(), key=lambda item: int(item[0]))),
        "edge_occurrence_counts": dict(sorted(edge_counts.items())),
        "source": "outer_train_independent_confirmed_arrivals_only",
        "exact_arrivals_and_derived_path_support_kept_separate": True,
        "path_support_used_as_additional_arrivals": False,
        "persisted_target_identifiers": False,
    }


def percentage(value: float | None) -> str:
    return "—" if value is None else f"{100.0 * value:.2f}%"


def build_report(
    cohorts: Mapping[str, Sequence[dict[str, Any]]], canonical: Mapping[str, Any],
    experimental: Mapping[str, Any], paired: Mapping[str, Any],
) -> str:
    lines = [
        "# 分层回退统计模型实验",
        "",
        "本报告只保存汇总结果。目标编号、URL、路径、候选清单、逐条预测和逐条得分都没有写入磁盘。",
        "统一评分继续使用当前 `evaluation_contract.py`；多级父页面和父目录的分值只在试验表中出现。",
        "",
        "## 样本",
        "",
        f"- 全部合法样本：{len(cohorts['all_legal'])}",
        f"- 预测前已有可用依据：{len(cohorts['prior_supported'])}",
        f"- 真实目标有可尝试的上级路径或等价服务首页：{len(cohorts['hierarchy_applicable'])}（路线拆分见审计）",
        "- 两个固定模型：无情景单步统计、二阶路径计数；测试集没有参与选模型或选曲线。",
        "",
        "## 统一评分",
        "",
        "`original` 是原模型顺序，`filtered` 屏蔽当前语义动作并按模型顺序合并等价首页、补足名额，三条曲线再做父子入口竞争。",
        "接受率把合法服务首页视为接受命中；首页的0.6部分收益只进入综合位置分、行动收益与损失，精确率另列。",
    ]
    variants = (ORIGINAL, CURRENT_FILTERED, *(policy.name for policy in POLICIES))
    for cohort in ("all_legal", "prior_supported", "hierarchy_applicable"):
        lines += [
            "", f"### {cohort}（n={len(cohorts[cohort])}）", "",
            "| 模型 | 顺序 | 接受Top1 | 接受Top2 | 接受Top4 | 精确Top1 | 精确Top2 | 精确Top4 | 综合位置分 |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for model in MODEL_ORDER:
            for variant in variants:
                score = canonical["cohorts"][cohort][model][variant]
                exact = score["exact"]
                lines.append(
                    f"| {model} | {variant} | {percentage(score['top1'])} | "
                    f"{percentage(score['top2'])} | {percentage(score['top4'])} | "
                    f"{percentage(exact['top1'])} | {percentage(exact['top2'])} | "
                    f"{percentage(exact['top4'])} | {percentage(score['rank_score']['mean'])} |"
                )

    lines += [
        "", "## 路径成功命中与部分收益", "",
        "本实验的 TopK 成功命中包含真实目标及预测前可用、与真实目标有可靠路径关系的中间网址或父目录；它们只得部分收益。下表只比较 `filtered` 与分层菜单，完全同一目标的成绩仍在上表单列。",
        "",
        "| 曲线 | 模型 | 样本组 | 成功Top1 原→新 | 成功Top2 原→新 | 成功Top4 原→新 | 部分收益Top1 原→新 | 部分收益Top2 原→新 | 部分收益Top4 原→新 | Top4收益变化 | 帮助/伤害条数 |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for policy in POLICIES:
        for model in MODEL_ORDER:
            for cohort in ("all_legal", "prior_supported", "hierarchy_applicable"):
                scores = experimental["cohorts"][cohort][model][policy.name]
                before, after = scores[CURRENT_FILTERED], scores[policy.name]
                pair = paired["cohorts"][cohort][model][policy.name]["parent_aggregation"]
                top4_pair = pair["experimental_multilevel"]["top4"]
                delta = None if after["top4"] is None else after["top4"] - before["top4"]
                lines.append(
                    f"| {policy.name} | {model} | {cohort} | "
                    f"{before['top1_success_n']}/{before['n']}→{after['top1_success_n']}/{after['n']} | "
                    f"{before['top2_success_n']}/{before['n']}→{after['top2_success_n']}/{after['n']} | "
                    f"{before['top4_success_n']}/{before['n']}→{after['top4_success_n']}/{after['n']} | "
                    f"{percentage(before['top1'])}→{percentage(after['top1'])} | "
                    f"{percentage(before['top2'])}→{percentage(after['top2'])} | "
                    f"{percentage(before['top4'])}→{percentage(after['top4'])} | "
                    f"{percentage(delta)} | {top4_pair['helped']}/{top4_pair['hurt']} |"
                )
    lines += [
        "", "## 解释边界", "",
        "- 当前地点、当前所属应用及当前首页的等价记录先被去掉；非当前等价首页也先合并并补足名额，父级汇总的变化因此不会混入重复首页清理。",
        "- 父级入口汇总的是多个互斥具体目标的概率，同一次需求只保留一个最高收益，不把父级和子级重复相加。",
        "- 本轮没有可靠的未知目标概率，因此只在预测前已知、可打开的入口内重新归一化，并在审计中保留被去掉的总概率。",
        "- `hierarchy_applicable` 需要看真实下一步，只是事后诊断组，不能当作线上置信保证。",
        "- 三条父级分值曲线是预先固定的研究方案；共享的首页 0.6 评分没有被改写。",
        "",
    ]
    return "\n".join(lines)


def environment() -> dict[str, Any]:
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "modules": {
            "numpy": __import__("numpy").__version__,
            "scipy": __import__("scipy").__version__,
        },
    }


def copy_sources(output: Path) -> dict[str, str]:
    names = (
        Path(__file__).name,
        "hierarchical_fallback_policy.py",
        "evaluation_contract.py",
        "compare_multistep_statistical.py",
        "multistep_statistical_models.py",
        "context_statistics.py",
        "run_offline_batch_rankers.py",
        "navigation_hierarchy.py",
        "known_inactivity.py",
        "semantic_annotations.py",
    )
    result: dict[str, str] = {}
    for name in names:
        source = SCRIPT_DIR / name
        target = output / ("source-" + name)
        shutil.copyfile(source, target)
        target.chmod(0o600)
        result[name] = offline.sha256(source)
    return result


def artifact_hashes(output: Path) -> dict[str, str]:
    return {
        str(path.relative_to(output)): offline.sha256(path)
        for path in sorted(output.rglob("*"))
        if path.is_file() and path.name not in {"artifact-hashes.json", "run.json"}
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-legal-n", type=int, default=139)
    args = parser.parse_args()

    snapshot = args.snapshot.expanduser().resolve()
    split_dir = args.split.expanduser().resolve()
    output = args.output.expanduser().resolve()
    private_root = PRIVATE_ROOT.resolve()
    if args.expected_legal_n < 0:
        parser.error("--expected-legal-n cannot be negative")
    if output == private_root or not output.is_relative_to(private_root):
        parser.error("Output must be a new private SmartApp training directory")
    if output.exists():
        parser.error("Output directory must not already exist")

    train, test, history, input_audit, sequence_audit = statistical.load_dataset(
        snapshot, split_dir,
    )
    progress("inputs_verified", train_members=len(train), test_members=len(test))

    os.umask(0o077)
    output.mkdir(parents=True, mode=0o700)
    output.chmod(0o700)

    plan = {
        "version": VERSION,
        "scope": "private_local_offline_research_only",
        "snapshot": str(snapshot),
        "split": str(split_dir),
        "foreground_only": True,
        "runtime_or_online_model_modified": False,
        "fixed_models": {
            MODEL_BASELINE: statistical.BASELINE_CONFIG,
            MODEL_MARKOV2: MARKOV2_CONFIG,
        },
        "test_used_for_model_or_policy_selection": False,
        "policies": [policy_description(policy) for policy in POLICIES],
        "ablation": [ORIGINAL, CURRENT_FILTERED, "parent_aggregation_for_each_fixed_curve"],
        "current_filtered_definition": (
            "Remove the current semantic destination and current owner app, then keep one "
            "model-ranked ID per equivalent confirmed site homepage and backfill the reference menu."
        ),
        "canonical_evaluation": evaluation.contract(),
        "canonical_service_metadata": (
            "Only service-home actions actually emitted by a variant are passed transiently to the scorer."
        ),
        "allowed_hierarchy": [
            "explicit_parentID", "structurally_derived_same_origin_url_ancestor",
            "structurally_derived_file_directory_ancestor", "approved_service_home", "owner_app",
        ],
        "forbidden_hierarchy_inference": [
            "same_site_without_path_ancestry", "arbitrary_url_path_truncation",
            "filesystem_scan", "background_inventory",
        ],
        "persistence": {
            "aggregate_metrics_only": True,
            "target_ids": False,
            "locators": False,
            "candidate_lists": False,
            "raw_predictions": False,
            "per_row_scores": False,
            "model_parameters": False,
        },
    }
    offline.write_private_json(output / "plan.json", plan)
    offline.write_private_json(output / "input-audit.json", input_audit)
    offline.write_private_json(output / "sequence-input-audit.json", sequence_audit)
    offline.write_private_json(output / "environment.json", environment())
    offline.write_private_json(
        output / "path-support-audit.json", training_path_support_audit(train, history),
    )

    outputs, contexts, prediction_audit = replay(train, test, history)
    progress("heldout_predictions_built", test_members=len(test), models=len(outputs))
    homes = emitted_service_homes(outputs, contexts)
    audited = offline.evaluation_rows(
        [row.observation for row in test], homes,
    )
    cohorts, cohort_audit = build_cohorts(audited, contexts, args.expected_legal_n)
    canonical = canonical_metrics(cohorts, outputs)
    experimental = experimental_metrics(cohorts, outputs, contexts)
    paired = paired_metrics(cohorts, outputs, contexts)

    prediction_audit["emitted_service_home_metadata_n"] = len(homes)
    prediction_audit["emitted_scoring_metadata_persisted"] = False
    offline.write_private_json(output / "prediction-audit.json", prediction_audit)
    offline.write_private_json(output / "cohort-audit.json", cohort_audit)
    offline.write_private_json(output / "canonical-metrics.json", canonical)
    offline.write_private_json(output / "experimental-multilevel-metrics.json", experimental)
    offline.write_private_json(output / "paired-help-harm.json", paired)
    offline.write_private_text(
        output / "report-private.md",
        build_report(cohorts, canonical, experimental, paired),
    )

    source_hashes = copy_sources(output)
    hashes = artifact_hashes(output)
    offline.write_private_json(output / "artifact-hashes.json", hashes)
    offline.write_private_json(output / "run.json", {
        "version": VERSION,
        "complete": True,
        "snapshot": str(snapshot),
        "split": str(split_dir),
        "input_snapshot_hashes": input_audit["snapshot_hashes"],
        "evaluation_rules_sha256": evaluation.contract()["rules_sha256"],
        "source_hashes": source_hashes,
        "artifact_hashes_sha256": offline.sha256(output / "artifact-hashes.json"),
        "test_members": len(test),
        "all_legal_n": len(cohorts["all_legal"]),
        "prior_supported_n": len(cohorts["prior_supported"]),
        "hierarchy_applicable_n": len(cohorts["hierarchy_applicable"]),
        "fixed_models": list(MODEL_ORDER),
        "fixed_policies": [policy.name for policy in POLICIES],
        "test_inference_passes": 1,
        "test_used_for_model_or_policy_selection": False,
        "candidate_lists_persisted": False,
        "raw_predictions_persisted": False,
        "target_identifiers_or_locators_persisted": False,
        "per_row_scores_persisted": False,
        "model_parameters_persisted": False,
    })
    progress(
        "complete",
        all_legal=len(cohorts["all_legal"]),
        prior_supported=len(cohorts["prior_supported"]),
        hierarchy_applicable=len(cohorts["hierarchy_applicable"]),
    )


if __name__ == "__main__":
    main()
