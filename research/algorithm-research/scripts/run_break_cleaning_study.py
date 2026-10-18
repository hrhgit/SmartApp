#!/usr/bin/env python3
"""Compare foreground-gap cleaning on one sealed SmartApp test membership.

Raw observations stay immutable.  Only training rows whose first destination was
confirmed before the outer boundary are added.  Six predeclared arms isolate
first-step recovery, qualified gap joins, and half-weight inferred joins.  All
models and arms use the same frozen test members and shared scoring contract.
Only aggregate scores and opaque per-row help/harm facts enter a private run.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

import cleaned_foreground_data as cleaning
import compare_multistep_statistical as statistical
import evaluation_contract as evaluation
import hierarchical_fallback_policy as path_policy
import run_hierarchical_batch_experiment as prior_batch
import run_hierarchical_statistical_experiment as hierarchy_report
import run_offline_batch_rankers as offline
import run_parallel_destination_experiment as parallel
import run_offline_sequence_shadow as sequence_shadow
import sequence_asof_features as asof
import online_model_suite as online_suite
import parallel_destination_data as parallel_data
from navigation_hierarchy import disposition


VERSION = "break-cleaning-study-v1"
PRIVATE_ROOT = offline.PRIVATE_ROOT
ARM_PARTS = {
    "base": (False, False, 0.0),
    "first_step": (True, False, 0.0),
    "short_join": (False, True, 1.0),
    "both": (True, True, 1.0),
    "short_join_half": (False, True, 0.5),
    "both_half": (True, True, 0.5),
}
BOARD_NAMES = ("pretest_route_supported", "route_supported", "all_legal")
PATH_MENU_RANK_LIMIT = 64
RICH_COLUMNS = (
    "time_hour_sin", "time_hour_cos", "time_weekday_sin", "time_weekday_cos",
    "last_confirmed_dwell_log_s", "has_confirmed_dwell",
    "quality_ready_fraction", "quality_syncing_fraction",
    "quality_unavailable_fraction", "quality_failed_fraction",
    "quality_stale_fraction", "quality_unknown_fraction", "quality_confirmed",
    "has_same_target_inferred_active", "same_target_inferred_active_log_s",
    "has_legacy_short_gap", "has_excluded_scope_gap",
    "known_gap_log_s", "known_gap_duration", "hierarchy_layer_count_log",
    "path_count_5m_log", "path_count_1h_log", "path_count_24h_log", "path_count_all_log",
    "direct_count_5m_log", "direct_count_1h_log", "direct_count_24h_log", "direct_count_all_log",
    "rich_input_missing",
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _anonymous(value: str) -> str:
    return offline.token(value)


def _progress(stage: str, **counts: int) -> None:
    print(json.dumps({"stage": stage, **counts}, sort_keys=True), flush=True)


def _write_private_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
    with path.open("x", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
    path.chmod(0o600)


def _prepared(row: offline.Observation) -> statistical.PreparedRow:
    """The unordered baseline needs only its confirmed current destination."""
    sequence = ((row.current.key,) if row.current is not None else ())
    return statistical.PreparedRow(row, sequence, None)


def _usable(row: offline.Observation) -> bool:
    return statistical.navigation_training_row(_prepared(row))


def training_arm(base: Iterable[offline.Observation], derived: Any, arm: str,
                 *, boundary_index: int | None = None
                 ) -> tuple[list[offline.Observation], dict[str, float], dict[str, int]]:
    """Keep one training relation per confirmed outcome, preferring original data.

    A join and an observed-first-step sample may refer to the same arrival.
    Calling either one an additional visit would inflate the frequency model.
    """
    if arm not in ARM_PARTS:
        raise ValueError("Unknown cleaning arm")
    add_first, add_join, join_weight = ARM_PARTS[arm]
    originals = list(base)
    candidates: list[tuple[str, offline.Observation, float]] = []
    if add_first:
        candidates.extend(("first_step", row, 1.0) for row in derived.salvaged_train)
    if add_join:
        candidates.extend(("qualified_join", row, join_weight) for row in derived.stitched_train)
    result: list[offline.Observation] = list(originals)
    weights: dict[str, float] = {row.sample_id: 1.0 for row in originals}
    audit: collections.Counter[str] = collections.Counter()
    audit["original_members_retained"] = len(originals)
    seen_sample: set[str] = {row.sample_id for row in originals}
    seen_outcome: set[str] = {row.outcome_event_id for row in originals if _usable(row)}
    if len(seen_sample) != len(originals):
        raise ValueError("Original training membership contains duplicate sample IDs")
    if boundary_index is not None and any(row.completed_event_index >= boundary_index for row in originals):
        raise ValueError("Original training membership matures at or after test boundary")
    for source, row, weight in candidates:
        if boundary_index is not None and row.completed_event_index >= boundary_index:
            audit[source + "_not_mature_before_boundary"] += 1
            continue
        if row.sample_id in seen_sample:
            audit[source + "_duplicate_sample"] += 1
            continue
        if not _usable(row):
            audit[source + "_not_navigation"] += 1
            continue
        if row.outcome_event_id in seen_outcome:
            audit[source + "_duplicate_confirmed_arrival"] += 1
            continue
        seen_sample.add(row.sample_id)
        seen_outcome.add(row.outcome_event_id)
        result.append(row)
        weights[row.sample_id] = weight
        audit[source + "_admitted"] += 1
    result.sort(key=lambda row: (row.input_event_index, row.sample_id))
    if len({row.sample_id for row in result}) != len(result):
        raise AssertionError("A training sample was counted twice")
    return result, weights, dict(sorted(audit.items()))


def replay_statistics(training: list[offline.Observation],
                      queries: list[offline.Observation], history: offline.TargetHistory,
                      weights: Mapping[str, float], *, learn_test: bool = False
                      ) -> tuple[dict[int, dict[str, Any]], dict[str, int]]:
    """Replay an unordered baseline, optionally with mature test feedback.

    `learn_test=True` is only the separately named online simulation.  A label
    enters after its actual completed event and before the next prediction.
    """
    model = statistical.StatisticsNoneModel()
    stream = sorted([*training, *(queries if learn_test else ())],
                    key=lambda row: (row.completed_event_index, row.input_event_index, row.sample_id))
    cursor = 0
    learned_outcomes: set[str] = set()
    audit: collections.Counter[str] = collections.Counter()
    outputs: dict[int, dict[str, Any]] = {}
    for query in sorted(queries, key=lambda row: (row.input_event_index, row.position)):
        while cursor < len(stream) and stream[cursor].completed_event_index < query.input_event_index:
            row = stream[cursor]
            cursor += 1
            if row.outcome_event_id in learned_outcomes or not _usable(row):
                audit["duplicate_or_unusable_mature_row"] += 1
                continue
            if row.input_event_index >= query.input_event_index or row.outcome_timestamp_ms >= query.input_timestamp_ms:
                raise ValueError("Future behavior reached an online prediction")
            record = model.record(_prepared(row))
            record["weight"] = float(weights.get(row.sample_id, 1.0))
            if not 0.0 < record["weight"] <= 1.0:
                raise ValueError("Training weight must be in (0, 1]")
            before = len(model.memory.training)
            model.memory.learn(model.query(_prepared(row)), record, row.completed_event_index)
            learned_outcomes.add(row.outcome_event_id)
            if len(model.memory.training) > before:
                audit["learned_unique_arrival"] += 1
                if row.split == "test":
                    audit["learned_mature_test_feedback"] += 1
            else:
                audit["semantic_duplicate_not_learned"] += 1
        outputs[query.position] = statistical.prediction_for("baseline", model, _prepared(query), history)
    audit["queries"] = len(queries)
    return outputs, dict(sorted(audit.items()))


def mlp_input(row: offline.Observation, index: asof.RichIndex,
              history: offline.TargetHistory, events: list[dict[str, Any]],
              mode: str) -> dict[str, float]:
    """Only the anchor's known foreground state and prior confirmed arrivals."""
    if row.current is None:
        raise ValueError("The online model needs a current foreground action")
    values = {"origin:" + _anonymous(row.current.key): 1.0,
              "owner:" + _anonymous(row.current.bundle or ""): 1.0,
              "kind:" + row.current.kind: 1.0}
    if mode == "target_only":
        return values
    if mode != "rich":
        raise ValueError("Unknown online input mode")
    prefix = index.by_event_index.get(row.input_event_index)
    if prefix is None:
        values["rich_input_missing"] = 1.0
        return values
    for name, value in asof.cutoff_features(prefix, events).items():
        values["context:" + name] = float(value)
    path, direct = asof.path_frequency(index, row.input_event_index, row.input_timestamp_ms)
    origin_layer = parallel_data.semantic_key(row.current.meta)
    for slot, period in enumerate(("5m", "1h", "24h", "all")):
        values["origin_path_count_" + period] = math.log1p(path.get(origin_layer, (0, 0, 0, 0))[slot])
        values["origin_direct_count_" + period] = math.log1p(direct.get(origin_layer, (0, 0, 0, 0))[slot])
    return values


def replay_online_mlp(training: list[offline.Observation],
                      queries: list[offline.Observation], history: offline.TargetHistory,
                      events: list[dict[str, Any]], index: asof.RichIndex,
                      *, mode: str, hidden: int, learning_rate: float,
                      learn_query_feedback: bool,
                      input_cache: dict[tuple[int, str], dict[str, float]] | None = None,
                      ) -> tuple[dict[int, dict[str, Any]], dict[str, int]]:
    """One incremental MLP update per matured independent arrival."""
    model = online_suite.CandidateMLP(hidden, learning_rate, negatives=8, seed=20260929)
    stream = sorted([*training, *(queries if learn_query_feedback else ())],
                    key=lambda row: (row.completed_event_index, row.input_event_index, row.sample_id))
    seen_outcomes: set[str] = set()
    cursor = 0
    output: dict[int, dict[str, Any]] = {}
    audit: collections.Counter[str] = collections.Counter()
    cache = input_cache if input_cache is not None else {}

    def features(row: offline.Observation) -> dict[str, float]:
        key = row.input_event_index, mode
        if key not in cache:
            cache[key] = mlp_input(row, index, history, events, mode)
        return cache[key]

    for query in sorted(queries, key=lambda row: (row.input_event_index, row.position)):
        while cursor < len(stream) and stream[cursor].completed_event_index < query.input_event_index:
            row = stream[cursor]
            cursor += 1
            if row.outcome_event_id in seen_outcomes or not _usable(row):
                audit["duplicate_or_unusable_mature_row"] += 1
                continue
            if row.input_event_index >= query.input_event_index or row.outcome_timestamp_ms >= query.input_timestamp_ms:
                raise ValueError("Future label reached online MLP")
            assert row.truth is not None
            # Feedback can arrive later than this example's input. Alternatives
            # must have existed at the original prediction cutoff.
            prior_catalog = history.catalog_at(row.input_event_index)
            allowed_negatives = frozenset(
                key for key in prior_catalog
                if disposition(key, row.current.meta) == "navigation")
            model.learn({"origin": row.current.key, "label": row.truth.key,
                         "x": features(row),
                         "allowed_negatives": allowed_negatives})
            seen_outcomes.add(row.outcome_event_id)
            audit["mature_unique_updates"] += 1
            if row.split == "test":
                audit["mature_test_feedback_updates"] += 1
        if query.current is None:
            output[query.position] = {"predictions": [], "rankable": []}
            continue
        catalog = history.catalog_at(query.input_event_index)
        labels = [key for key in sorted(model.seen_labels & set(catalog))
                  if disposition(key, query.current.meta) == "navigation"]
        scores = model.scores({
            "origin": query.current.key, "labels": labels,
            "x": features(query),
        })
        ranked = sorted(scores, key=lambda label: (-scores[label], label))
        output[query.position] = {"predictions": ranked[:4], "rankable": ranked}
        audit["queries"] += 1
        audit["queries_with_known_label"] += bool(ranked)
    return output, dict(sorted(audit.items()))


class WeightedHistoryStats(offline.HistoryStats):
    """As-of frequency features with the same half weight as training loss."""

    def __init__(self) -> None:
        super().__init__()
        self.weighted_records: list[tuple[int, str, float]] = []

    def add_weighted(self, row: offline.Observation, weight: float) -> bool:
        if row.outcome_event_id in self.seen_outcomes or not _usable(row):
            return False
        if not 0.0 < weight <= 1.0:
            raise ValueError("Invalid training weight")
        assert row.current is not None and row.truth is not None
        self.seen_outcomes.add(row.outcome_event_id)
        self.global_counts[row.truth.key] += weight
        self.conditional_counts[row.current.key, row.truth.key] += weight
        self.origin_counts[row.current.key] += weight
        self.weighted_records.append((row.outcome_timestamp_ms, row.truth.key, weight))
        self.last_seen[row.truth.key] = max(self.last_seen.get(row.truth.key, -1), row.outcome_timestamp_ms)
        self._summary_cache.clear()
        return True

    def summary(self, current_key: str, timestamp_ms: int) -> dict[str, Any]:
        key = current_key, timestamp_ms
        if key in self._summary_cache:
            return self._summary_cache[key]
        recent_1h: collections.Counter[str] = collections.Counter()
        recent_24h: collections.Counter[str] = collections.Counter()
        for seen_at, target, weight in self.weighted_records:
            if seen_at > timestamp_ms:
                raise ValueError("Future behavior reached frequency input")
            age = timestamp_ms - seen_at
            if age <= 3_600_000:
                recent_1h[target] += weight
            if age <= 86_400_000:
                recent_24h[target] += weight
        value = {"recent_1h": recent_1h, "recent_24h": recent_24h}
        self._summary_cache[key] = value
        return value


def weighted_training_groups(rows: list[offline.Observation], history: offline.TargetHistory,
                             weights: Mapping[str, float]) -> tuple[list[offline.QueryGroup], WeightedHistoryStats, dict[str, int]]:
    stats = WeightedHistoryStats()
    maturity = sorted(rows, key=lambda row: (row.completed_event_index, row.input_event_index, row.sample_id))
    groups: list[offline.QueryGroup] = []
    audit: collections.Counter[str] = collections.Counter()
    cursor = 0
    for row in sorted(rows, key=lambda row: (row.input_event_index, row.sample_id)):
        while cursor < len(maturity) and maturity[cursor].completed_event_index < row.input_event_index:
            mature = maturity[cursor]
            stats.add_weighted(mature, weights[mature.sample_id])
            cursor += 1
        if not _usable(row):
            audit["non_navigation_training_member"] += 1
            continue
        group = offline.make_group(row, stats, history, training=True)
        if group.truth_index is not None and len(group.actions) > 1:
            groups.append(group)
            audit["rankable_training_query"] += 1
        else:
            audit["truth_not_in_asof_catalog_or_one_action"] += 1
    frozen_stats = WeightedHistoryStats()
    for row in maturity:
        frozen_stats.add_weighted(row, weights[row.sample_id])
    return groups, frozen_stats, dict(sorted(audit.items()))


def rich_features(group: offline.QueryGroup, index: asof.RichIndex,
                  history: offline.TargetHistory, events: list[dict[str, Any]]
                  ) -> list[dict[str, float]]:
    prefix = index.by_event_index.get(group.row.input_event_index)
    if prefix is None:
        return [{name: (1.0 if name == "rich_input_missing" else 0.0)
                 for name in RICH_COLUMNS} for _ in group.actions]
    rows = asof.candidate_feature_rows(group, index, history, events, prefix)
    if len(rows) != len(group.actions):
        raise AssertionError("Rich candidate feature order changed")
    result = []
    for row in rows:
        if set(row) != set(RICH_COLUMNS) - {"rich_input_missing"}:
            raise ValueError("As-of batch features changed without a declared schema update")
        result.append({**row, "rich_input_missing": 0.0})
    return result


class RichCatBoostResidual:
    def __init__(self, model: Any, index: asof.RichIndex,
                 history: offline.TargetHistory, events: list[dict[str, Any]]):
        self.model, self.index, self.history, self.events = model, index, history, events

    def correction(self, group: offline.QueryGroup) -> np.ndarray:
        import pandas as pd
        if not group.actions:
            return np.array([], dtype=float)
        rich = rich_features(group, self.index, self.history, self.events)
        frame = pd.DataFrame([
            {**{name: feature[name] for name in offline.CAT_COLUMNS + offline.NUMERIC_COLUMNS},
             **rich[index]}
            for index, feature in enumerate(group.features)
        ], columns=offline.CAT_COLUMNS + offline.NUMERIC_COLUMNS + RICH_COLUMNS)
        return np.asarray(self.model.predict(frame), dtype=float)


def train_weighted_catboost(groups: list[offline.QueryGroup], weights: Mapping[str, float],
                            config: dict[str, Any], *, rich_index: asof.RichIndex | None = None,
                            history: offline.TargetHistory | None = None,
                            events: list[dict[str, Any]] | None = None
                            ) -> offline.CatBoostResidual | RichCatBoostResidual:
    from catboost import CatBoostRanker, Pool
    import pandas as pd

    frame, labels, group_ids, baselines = offline.cat_frame(groups)
    if len(set(group_ids)) < 2:
        raise ValueError("CatBoost needs at least two rankable groups")
    numeric_columns = offline.NUMERIC_COLUMNS
    if rich_index is not None:
        if history is None or events is None:
            raise ValueError("Rich input needs an as-of index, metadata and original events")
        rich_rows = [feature for group in groups
                     for feature in rich_features(group, rich_index, history, events)]
        if len(rich_rows) != len(frame):
            raise AssertionError("Rich features do not align with CatBoost candidates")
        frame = pd.concat([frame.reset_index(drop=True),
                           pd.DataFrame(rich_rows, columns=RICH_COLUMNS)], axis=1)
        numeric_columns += RICH_COLUMNS
    group_weights = [float(weights[group.row.sample_id]) for group in groups
                     for _ in group.actions]
    if len(group_weights) != len(labels):
        raise AssertionError("A group lost its candidate weights")
    pool = Pool(frame, label=labels, group_id=group_ids, baseline=baselines,
                group_weight=group_weights, cat_features=list(offline.CAT_COLUMNS),
                feature_names=list(offline.CAT_COLUMNS + numeric_columns))
    model = CatBoostRanker(loss_function="YetiRankPairwise", eval_metric="NDCG:top=4",
                           iterations=config["iterations"], depth=config["depth"],
                           learning_rate=config["learning_rate"], l2_leaf_reg=config["l2_leaf_reg"],
                           random_seed=config["seed"], thread_count=1,
                           allow_writing_files=False, verbose=False)
    model.fit(pool)
    if rich_index is None:
        return offline.CatBoostResidual(model, dict(config))
    assert history is not None and events is not None
    return RichCatBoostResidual(model, rich_index, history, events)


def replay_catboost(training: list[offline.Observation], test: list[offline.Observation],
                    history: offline.TargetHistory, weights: Mapping[str, float],
                    config: dict[str, Any], *, rich_index: asof.RichIndex | None = None,
                    events: list[dict[str, Any]] | None = None
                    ) -> tuple[dict[int, dict[str, Any]], dict[str, int]]:
    groups, stats, audit = weighted_training_groups(training, history, weights)
    model = train_weighted_catboost(groups, weights, config, rich_index=rich_index,
                                    history=history, events=events)
    queries = offline.build_static_groups(test, stats, history)
    return offline.predictions(queries, model.correction), {
        **audit, "frozen_training_rows": len(training), "test_queries": len(test),
        "input_mode": "rich" if rich_index is not None else "target_only",
        "rich_test_prefixes": (sum(row.input_event_index in rich_index.by_event_index for row in test)
                               if rich_index is not None else None),
    }


def evaluation_rows(rows: list[offline.Observation], history: offline.TargetHistory
                    ) -> list[dict[str, Any]]:
    """Supply the shared scorer only each cutoff's already known actions.

    The per-cutoff action metadata stays in memory.  Later test-period target
    updates cannot make a parent executable at an earlier prediction.
    """
    observations = []
    labels = []
    for row in rows:
        known = [action.meta for action in history.catalog_at(row.input_event_index).values()]
        observations.append({
            "position": row.position, "index": row.input_event_index,
            "ready_index": row.completed_event_index,
            "target": row.current.meta if row.current else {},
            "parent_action": row.current_parent.meta if row.current_parent else None,
            "known_actions": known,
        })
        labels.append({
            "position": row.position, "sample_id": row.sample_id,
            "truth_action": row.truth.key if row.truth else None,
            "truth_metadata": row.truth.meta if row.truth else {},
            "split": row.split, "evidence_grade": "offline_batch_all_legal_members",
            "truth_parent_metadata": row.truth_parent.meta if row.truth_parent else None,
            "prior_evidence": {
                "exact_target": row.has_prior_exact_action,
                "parent_action": row.has_prior_parent_action,
            },
        })
    return evaluation.prepare_evaluation({"observations": observations, "evaluation": labels})


def compact_score(score: dict[str, Any]) -> dict[str, Any]:
    result = {
        "n": score["n"],
        "success": {f"top{k}": score["success"][f"top{k}_hits"] for k in (1, 2, 4)},
        "rankable_useful_action_n": score["success"].get("rankable_useful_action_n"),
        "rankable_useful_action_coverage": score["success"].get("rankable_useful_action_coverage"),
        "service": {f"top{k}": score[f"top{k}_hits"] for k in (1, 2, 4)},
        "exact": {f"top{k}": score["exact"][f"top{k}_hits"] for k in (1, 2, 4)},
        "rank_score": score["rank_score"]["mean"],
        "top4_mean_gain": score["actionable_topk"]["top4_gain"],
        "prediction_coverage": score["prediction_coverage"],
        "comparison_key": score["comparison_key"],
        "audit": score["audit"],
    }
    if "path_utility" in score:
        result["path_utility"] = score["path_utility"]
    return result


def score_boards(rows: Mapping[str, list[dict[str, Any]]],
                 outputs: Mapping[str, Mapping[int, dict[str, Any]]]
                 ) -> dict[str, dict[str, Any]]:
    scores = {}
    for board_name in BOARD_NAMES:
        part = rows[board_name]
        board = {name: evaluation.measure(part, result) for name, result in outputs.items()}
        evaluation.require_comparable(*board.values())
        scores[board_name] = {name: compact_score(score) for name, score in board.items()}
    return scores


def path_menu_variants(test: list[offline.Observation], history: offline.TargetHistory,
                       outputs: Mapping[str, Mapping[int, dict[str, Any]]], selected_arm: str,
                       *, include_derived: bool = True
                       ) -> tuple[dict[str, dict[int, dict[str, Any]]], dict[str, Any]]:
    """Separate inference-scope experiment: rank proxy plus as-of parent menu.

    This intentionally does not claim posterior calibration or change any
    training arm.  It can emit a structurally derived homepage/directory even
    if nobody stopped at that ancestor, solely from an earlier known locator.
    """
    policy = path_policy.FallbackPolicy("product0.6", "product", content_retention=0.6,
                                        app_retention=evaluation.FALLBACK_GAIN)
    chosen = {name: result for name, result in outputs.items()
              if (name == f"statistical/{selected_arm}"
                  or name == f"catboost/{selected_arm}"
                  or name == f"catboost_rich/{selected_arm}"
                  or (name.startswith("sequence_") and name.endswith("/" + selected_arm)))}
    contexts = {row.position: hierarchy_report.build_context(_prepared(row), history)
                for row in test}
    variants: dict[str, dict[int, dict[str, Any]]] = {}
    audit: dict[str, Any] = {}
    for name, output in chosen.items():
        variant: dict[int, dict[str, Any]] = {}
        counts: collections.Counter[str] = collections.Counter()
        for row in test:
            base = output[row.position]
            context = contexts[row.position]
            tree = context.hierarchy
            ordered = [action for action in base.get("rankable", base["predictions"])
                       if action in tree.executable][:PATH_MENU_RANK_LIMIT]
            ordered = list(dict.fromkeys(ordered))
            if not ordered:
                variant[row.position] = {"predictions": [], "rankable": []}
                counts["empty_base_ranking"] += 1
                continue
            ranking_values = {action: float(len(ordered) - index)
                              for index, action in enumerate(ordered)}
            proxy = path_policy.rank_proxy_posterior(ranking_values)
            allowed = set(ordered)
            for action in ordered:
                allowed.update(tree.paths(action))
            allowed &= tree.executable
            if not include_derived:
                allowed -= tree.derived_parent_ids
            menu_tree = path_policy.Hierarchy(tree.metadata, tree.parents, allowed,
                                               _skip_validation=True)
            menu, _ = path_policy.greedy_menu(
                proxy, menu_tree, policy, size=4,
                current_action=context.current_action, current_app=context.current_app,
                candidate_order=ordered)
            variant[row.position] = {
                "predictions": menu,
                "rankable": list(dict.fromkeys([*menu, *ordered, *sorted(allowed)])),
            }
            counts["queries"] += 1
            counts["emitted_derived_parent_slots"] += sum(action in tree.derived_parent_ids
                                                          for action in menu)
            counts["queries_with_derived_parent"] += any(action in tree.derived_parent_ids
                                                         for action in menu)
        prefix = "path_menu__" if include_derived else "path_menu_recorded_only__"
        variants[f"{prefix}{name}"] = variant
        audit[name] = dict(sorted(counts.items()))
    return variants, {
        "policy": "product0.6 predeclared experimental curve",
        "ranking_input": f"scale-invariant rank proxy from top {PATH_MENU_RANK_LIMIT} original positions, not calibrated probabilities",
        "candidate_scope": ("as-of recorded actions plus structurally derived executable parents"
                            if include_derived else "as-of recorded actions only"),
        "derived_parents_allowed": include_derived,
        "model_audits": audit,
    }


def paired_rows(rows: list[dict[str, Any]],
                outputs: Mapping[str, Mapping[int, dict[str, Any]]],
                reference: str | Mapping[str, str]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Persist opaque individual changes, never raw targets or ranked actions."""
    details: list[dict[str, Any]] = []
    aggregate = {}
    def path_prefixes(row: dict[str, Any], output: dict[str, Any], policy: str) -> dict[str, float]:
        actions = output["predictions"]
        return {f"top{k}": max((evaluation.path_action_gain(row, action, policy)
                                 for action in actions[:k]), default=0.0)
                for k in (1, 4)}

    for model_name, predictions in outputs.items():
        if isinstance(reference, Mapping) and model_name not in reference:
            continue
        comparison = (reference[model_name] if isinstance(reference, Mapping)
                      else reference)
        if model_name == comparison:
            continue
        baseline = outputs[comparison]
        aggregate[model_name] = evaluation.paired_topk(rows, baseline, predictions)
        aggregate[model_name]["reference_model"] = comparison
        for row in rows:
            if not evaluation.eligible(row):
                continue
            position = row["position"]
            before_output = evaluation.prediction_at(baseline, position)
            after_output = evaluation.prediction_at(predictions, position)
            first = evaluation.hits(row, before_output)
            second = evaluation.hits(row, after_output)
            first["success"] = evaluation.success_hits(row, before_output)
            second["success"] = evaluation.success_hits(row, after_output)
            path_before = {name: path_prefixes(row, before_output, name)
                           for name in ("product0.6", "product0.8", "saturating0.8")}
            path_after = {name: path_prefixes(row, after_output, name)
                          for name in ("product0.6", "product0.8", "saturating0.8")}
            hits_same = all(first[level][k] == second[level][k]
                            for level in ("success", "accepted", "exact") for k in (1, 2, 4))
            gains_same = all(math.isclose(path_before[name][f"top{k}"],
                                          path_after[name][f"top{k}"], abs_tol=1e-12)
                             for name in path_before for k in (1, 4))
            if hits_same and gains_same:
                continue
            details.append({
                "sample_token": _anonymous(row["sample_id"]),
                "model": model_name, "reference_model": comparison,
                "before": {level: {f"top{k}": bool(first[level][k]) for k in (1, 2, 4)}
                           for level in ("success", "accepted", "exact")},
                "after": {level: {f"top{k}": bool(second[level][k]) for k in (1, 2, 4)}
                          for level in ("success", "accepted", "exact")},
                "before_rank_gain": evaluation.ranked_gain(row, before_output)["score"],
                "after_rank_gain": evaluation.ranked_gain(row, after_output)["score"],
                "path_gain_before": path_before,
                "path_gain_after": path_after,
                "path_gain_definition": "predeclared experimental curves, not an agreed fixed partial value",
            })
    return details, aggregate


def _report(metrics: Mapping[str, Any], training_counts: Mapping[str, Any],
            selected: str, expanded_n: int,
            menu_pairs: Mapping[str, Any]) -> str:
    lines = ["# 前台断点清洗：固定测试对照", "",
             "原始记录不变。已确认第一步与跨合格短断点推定分别进入训练；后者另试半权重。",
             "所有方案共用固定 598 条原训练成员和 149 条原测试成员；主要成绩先看测试开始前已有可得分路径的固定样本。",
             "短断点两端不同目标只表示‘相邻可推荐目标’，不声称观察到了断点内的操作。",
             f"训练内部选中的清洗方案：`{selected}`。测试期另外 {expanded_n} 条已确认第一步仅作独立诊断。", "",
             "| 方案 | 新增第一步 | 新增推定接续 | 测试可评分数 | 成功 Top1 | Top2 | Top4 | 完全一致 Top1 | Top2 | Top4 | 服务接受 Top1 | Top2 | Top4 | 路径收益 0.6 / 0.8 / 饱和 0.8（实验） | 固定排位分 | 可得分入口覆盖 | 输出覆盖 |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    primary = metrics["pretest_route_supported"]
    for name, score in primary.items():
        arm = name.rsplit("/", 1)[-1]
        counts = training_counts.get(arm, {})
        admitted = counts.get("audit", {})
        n = score["n"]
        success, exact, service = score["success"], score["exact"], score["service"]
        curves = score["path_utility"]
        curve_text = " / ".join(f"{curves[name]['top4_gain']:.3f}"
                              for name in ("product0.6", "product0.8", "saturating0.8"))
        lines.append(
            f"| {name} | {admitted.get('first_step_admitted', 0)} | {admitted.get('qualified_join_admitted', 0)} | {n} | "
            f"{success['top1']}/{n} | {success['top2']}/{n} | {success['top4']}/{n} | "
            f"{exact['top1']}/{n} | {exact['top2']}/{n} | {exact['top4']}/{n} | "
            f"{service['top1']}/{n} | {service['top2']}/{n} | {service['top4']}/{n} | "
            f"{curve_text} | {score['rank_score']:.3f} | "
            f"{score['rankable_useful_action_n']}/{n} | "
            f"{score['prediction_coverage']:.1%} |"
        )
    lines += ["", "路径收益三列是尚未约定的分层部分分值实验，不作为已定评分；固定排位分沿用已约定的首页／应用回退收益。",
              "普通模型使用旧的已记录动作候选范围；下列路径菜单实验另从预测前已知网址或路径临时生成可打开的上级。",
              "路径菜单用原排序位置构造未经校准的排序代理，不表示模型已经学会上级概率。",
              "", "| 路径菜单对照（同一模型、同一清洗） | 成功 Top4 前→后 | 新增成功／失去成功 | 实验路径收益 Top4 前→后 |",
              "|---|---:|---:|---:|"]
    for name, paired in menu_pairs.items():
        reference = paired["reference_model"]
        before = primary[reference]
        after = primary[name]
        hit_change = paired["success"]["top4"]
        gain_before = before["path_utility"]["product0.6"]["top4_gain"]
        gain_after = after["path_utility"]["product0.6"]["top4_gain"]
        lines.append(f"| {reference} → {name} | {before['success']['top4']}→{after['success']['top4']} | "
                     f"{hit_change['helped']}/{hit_change['hurt']} | {gain_before:.3f}→{gain_after:.3f} |")
    lines += ["",
              "‘online_simulation’ 是测试期间只在反馈成熟后更新的无情景统计基准，不代表加入时间与停留的情景模型。",
              "‘online_mlp’ 是独立的轻量在线模型，分别试仅目标与时间、停留、观察质量等丰富输入；测试期只在真实反馈形成后逐条更新。",
              "丰富输入的 CatBoost、RNN、LSTM、Transformer 只在训练期选中的清洗臂比较；半权重序列实验的共享旧统计先验仍用整数次数，降权只进入神经部分。",
              "选参只看原训练时期内部按时间顺序划出的验证段。测试结果不用于改阈值或模型配置。",
              "所有完整 URL 和目录路径留在用户私有快照中；报告只包含汇总计数。", ""]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-selection", type=Path, required=True,
                        help="Completed, train-only offline batch configuration run")
    parser.add_argument("--skip-catboost", action="store_true", help="Diagnostic only")
    parser.add_argument("--skip-sequence", action="store_true", help="Diagnostic only")
    parser.add_argument("--skip-online-mlp", action="store_true", help="Diagnostic only")
    parser.add_argument("--sequence-modes", default="target_only,temporal,hierarchy,full",
                        help="Comma-separated predeclared input ablations for the selected cleaning arm")
    parser.add_argument("--sequence-epochs", type=int, default=12)
    parser.add_argument("--sequence-screen-epochs", type=int, default=4)
    args = parser.parse_args()
    snapshot, split_dir, output = (value.expanduser().resolve()
                                   for value in (args.snapshot, args.split, args.output))
    private_root = PRIVATE_ROOT.expanduser().resolve()
    if output == private_root or not output.is_relative_to(private_root) or output.exists():
        parser.error("Output must be a new private SmartApp training directory")

    train, test, history, load_audit = offline.load_observations(snapshot, split_dir)
    if len(train) != 598 or len(test) != 149:
        raise ValueError("The agreed fixed 598/149 split changed")
    split_report = offline.read_json(split_dir / "split-report.json")
    boundary = int(split_report["boundary"]["eventIndex"])
    events = offline.read_jsonl(snapshot / "events.jsonl")
    samples = offline.read_jsonl(snapshot / "navigation-samples.jsonl")
    derived = cleaning.derive_cleaning(events, samples, history,
                                        test_boundary_index=boundary,
                                        existing_train_rows=train, existing_test_rows=test)
    _progress("cleaning_derived", original_train=len(train), original_test=len(test),
              salvaged_train=len(derived.salvaged_train),
              stitched_train=len(derived.stitched_train),
              separate_test_diagnostic=len(derived.salvaged_test_diagnostic))
    if any(row.completed_event_index >= boundary for row in (*derived.salvaged_train, *derived.stitched_train)):
        raise ValueError("Cleaning supplied an unmatured training label")

    selected_configs, selection_audit = prior_batch.load_selected_configs(
        args.batch_selection.expanduser().resolve(), snapshot, split_dir)
    cat_config = selected_configs["catboost"]["config"]
    sequence_modes = tuple(part.strip() for part in args.sequence_modes.split(",") if part.strip())
    if (not sequence_modes or len(set(sequence_modes)) != len(sequence_modes)
            or not set(sequence_modes) <= {"target_only", "temporal", "hierarchy", "full"}
            or args.sequence_epochs < 1 or args.sequence_screen_epochs < 1):
        parser.error("Invalid sequence input modes or epoch count")

    arm_training: dict[str, list[offline.Observation]] = {}
    arm_weights: dict[str, dict[str, float]] = {}
    training_counts = {}
    for arm in ARM_PARTS:
        rows, weights, audit = training_arm(train, derived, arm, boundary_index=boundary)
        arm_training[arm], arm_weights[arm] = rows, weights
        training_counts[arm] = {"rows": len(rows), "audit": audit,
                                "inferred_weight": ARM_PARTS[arm][2]}
    _progress("arms_prepared", count=len(ARM_PARTS))

    # The inner validation members are original, completed-horizon rows only.
    # Each arm's training material must already be mature before their cutoffs.
    _, _, validation, inner = offline.causal_three_way(train)
    inner_boundary = int(inner["selection_boundary_event_index"])
    inner_outputs = {}
    for arm in ARM_PARTS:
        fit_rows = [row for row in arm_training[arm] if row.completed_event_index < inner_boundary]
        inner_outputs[arm], _ = replay_statistics(fit_rows, validation, history, arm_weights[arm])
    inner_audited = evaluation_rows(validation, history)
    inner_scores = {arm: evaluation.measure(inner_audited, output)
                    for arm, output in inner_outputs.items()}
    evaluation.require_comparable(*inner_scores.values())
    selected_arm = min(ARM_PARTS, key=lambda arm: (
        -inner_scores[arm]["success"]["top4_hits"],
        -inner_scores[arm]["exact"]["top4_hits"],
        -inner_scores[arm]["rank_score"]["mean"], arm,
    ))
    _progress("inner_selection_complete", validation_members=len(validation))

    outputs: dict[str, dict[int, dict[str, Any]]] = {}
    model_audits = {}
    for arm in ARM_PARTS:
        outputs[f"statistical/{arm}"], model_audits[f"statistical/{arm}"] = replay_statistics(
            arm_training[arm], test, history, arm_weights[arm])
        outputs[f"online_simulation/{arm}"], model_audits[f"online_simulation/{arm}"] = replay_statistics(
            arm_training[arm], test, history, arm_weights[arm], learn_test=True)
        _progress("statistical_arm_complete", arms_completed=list(ARM_PARTS).index(arm) + 1)
        if not args.skip_catboost:
            outputs[f"catboost/{arm}"], model_audits[f"catboost/{arm}"] = replay_catboost(
                arm_training[arm], test, history, arm_weights[arm], cat_config)
            _progress("catboost_arm_complete", arms_completed=list(ARM_PARTS).index(arm) + 1)

    catboost_input_validation = None
    selected_bridges = derived.bridges if ARM_PARTS[selected_arm][1] else ()
    if not args.skip_catboost:
        rich_index = asof.build_rich_index(events, history, 16, selected_bridges)
        fit_rows = [row for row in arm_training[selected_arm]
                    if row.completed_event_index < inner_boundary]
        old_validation_outputs, _ = replay_catboost(
            fit_rows, validation, history, arm_weights[selected_arm], cat_config)
        rich_validation_outputs, _ = replay_catboost(
            fit_rows, validation, history, arm_weights[selected_arm], cat_config,
            rich_index=rich_index, events=events)
        checked = evaluation_rows(validation, history)
        old_validation_score = evaluation.measure(checked, old_validation_outputs)
        rich_validation_score = evaluation.measure(checked, rich_validation_outputs)
        evaluation.require_comparable(old_validation_score, rich_validation_score)
        key = lambda score: (score["success"]["top4_hits"],
                             score["exact"]["top4_hits"], score["rank_score"]["mean"])
        selected_catboost_input = ("rich" if key(rich_validation_score) > key(old_validation_score)
                                   else "target_only")
        catboost_input_validation = {
            "selected": selected_catboost_input,
            "target_only": compact_score(old_validation_score),
            "rich": compact_score(rich_validation_score),
            "selection_rule": "train-period validation path Top4, exact Top4, fixed rank score; ties prefer target_only",
            "rich_feature_columns": list(RICH_COLUMNS),
            "rich_index_audit": rich_index.audit,
        }
        outputs[f"catboost_rich/{selected_arm}"], model_audits[f"catboost_rich/{selected_arm}"] = replay_catboost(
            arm_training[selected_arm], test, history, arm_weights[selected_arm], cat_config,
            rich_index=rich_index, events=events)
        _progress("catboost_rich_complete", validation_members=len(validation))

    online_mlp_selection = None
    if not args.skip_online_mlp:
        mlp_arm = {"short_join_half": "short_join", "both_half": "both"}.get(
            selected_arm, selected_arm)
        selected_training = arm_training[mlp_arm]
        mlp_bridges = derived.bridges if ARM_PARTS[mlp_arm][1] else ()
        prepared_online = asof.build_rich_index(events, history, 16, mlp_bridges)
        online_feature_cache: dict[tuple[int, str], dict[str, float]] = {}
        fit_rows = [row for row in selected_training
                    if row.completed_event_index < inner_boundary]
        audited_validation = evaluation_rows(validation, history)
        grid = [(mode, hidden, 0.001) for mode in ("target_only", "rich")
                for hidden in (16, 32)]
        evaluated = {}
        for mode, hidden, rate in grid:
            result, replay_audit = replay_online_mlp(
                fit_rows, validation, history, events, prepared_online,
                mode=mode, hidden=hidden, learning_rate=rate,
                learn_query_feedback=True, input_cache=online_feature_cache)
            score = evaluation.measure(audited_validation, result)
            evaluated[f"{mode}/h{hidden}"] = {
                "config": {"mode": mode, "hidden": hidden, "learning_rate": rate},
                "score": compact_score(score), "replay_audit": replay_audit,
            }
            _progress("online_mlp_validation_complete", configurations_completed=len(evaluated))
        choice = min(evaluated, key=lambda name: (
            -evaluated[name]["score"]["success"]["top4"],
            -evaluated[name]["score"]["exact"]["top4"],
            -evaluated[name]["score"]["rank_score"], name,
        ))
        config = evaluated[choice]["config"]
        outputs[f"online_mlp/{mlp_arm}"], mlp_audit = replay_online_mlp(
            selected_training, test, history, events, prepared_online,
            mode=config["mode"], hidden=config["hidden"],
            learning_rate=config["learning_rate"],
            learn_query_feedback=True, input_cache=online_feature_cache)
        model_audits[f"online_mlp/{mlp_arm}"] = mlp_audit
        online_mlp_selection = {
            "status": "complete", "selected": choice,
            "cleaning_arm": mlp_arm,
            "selected_arm_weight_mapping": (
                "selected half-weight arm mapped to same full-weight data because CandidateMLP uses one update per arrival"
                if mlp_arm != selected_arm else None),
            "selection_rule": "train-period online validation path Top4, exact Top4, fixed rank score",
            "validation": evaluated,
            "test_updates_only_after_maturity": True,
            "negative_pairs_are_sampled_unselected_asof_alternatives_not_explicit_rejections": True,
        }
        _progress("online_mlp_fixed_test_complete", queries=len(test))

    sequence_selection = None
    if not args.skip_sequence:
        selected_training = arm_training[selected_arm]
        inferred_ids = frozenset(row.sample_id for row in derived.stitched_train)
        inferred_ids &= frozenset(row.sample_id for row in selected_training)
        inferred_weight = ARM_PARTS[selected_arm][2] if selected_bridges else 1.0
        prepared_index = asof.build_rich_index(events, history, 16, selected_bridges)
        screening: dict[str, dict[str, Any]] = {}
        for mode in sequence_modes:
            _, _, _, sequence_audit = sequence_shadow.replay_cleaning_sequence(
                selected_training, test, history, events, selected_bridges,
                input_mode=mode, inferred_sample_ids=inferred_ids,
                inferred_weight=inferred_weight, epochs=args.sequence_screen_epochs,
                score_test=False, prepared_index=prepared_index)
            screening[mode] = sequence_audit
            _progress("sequence_screen_mode_complete", modes_completed=sequence_modes.index(mode) + 1)

        def sequence_key(score: dict[str, Any], mode: str) -> tuple[float, float, float, str]:
            return (float(score["success_hits"]["top4"]),
                    float(score["exact_hits"]["top4"]),
                    float(score["rank_score"]), mode)

        selected_by_family = {}
        for family in ("rnn", "lstm", "transformer"):
            selected_mode = max(sequence_modes, key=lambda mode: sequence_key(
                screening[mode]["models"][family]["inner_selection_score"], mode))
            selected_by_family[family] = selected_mode
            _, sequence_outputs, _, sequence_audit = sequence_shadow.replay_cleaning_sequence(
                selected_training, test, history, events, selected_bridges,
                input_mode=selected_mode, inferred_sample_ids=inferred_ids,
                inferred_weight=inferred_weight, epochs=args.sequence_epochs,
                families=(family,), prepared_index=prepared_index)
            for name, result in sequence_outputs.items():
                if name == "no_context_statistical":
                    continue
                outputs[f"sequence_{name}/{selected_mode}/{selected_arm}"] = result
            model_audits[f"sequence/{family}/{selected_mode}/{selected_arm}"] = {
                **sequence_audit,
                "prior_count_weight_limitation": (
                    "inferred residual loss uses requested weight; shared frozen prior count is integer"
                    if inferred_weight == 0.5 else None),
            }
            _progress("sequence_family_complete", families_completed=len(selected_by_family))
        sequence_selection = {
            "screen_epochs": args.sequence_screen_epochs,
            "final_epochs": args.sequence_epochs,
            "selected_input_by_family": selected_by_family,
            "train_only_screen": {mode: {
                family: screening[mode]["models"][family]["inner_selection_score"]
                for family in ("rnn", "lstm", "transformer")}
                for mode in sequence_modes},
            "selection_rule": "train-period inner score only; fixed test opened once per family",
        }

    menu_outputs, path_menu_audit = path_menu_variants(test, history, outputs, selected_arm)
    outputs.update(menu_outputs)
    audited = evaluation_rows(test, history)
    contexts = {row.position: hierarchy_report.build_context(_prepared(row), history) for row in test}
    cohorts, cohort_audit = hierarchy_report.build_cohorts(audited, contexts, 139)
    old = {row["position"] for row in cohorts["prior_supported"]}
    route = {row["position"] for row in cohorts["hierarchy_applicable"]}
    cohorts["route_supported"] = [row for row in cohorts["all_legal"]
                                  if row["position"] in old | route]
    cohorts["pretest_route_supported"] = parallel.frozen_training_route_rows(
        cohorts["route_supported"], contexts, history, boundary, parallel.POLICIES[0])
    expected = {"all_legal": 139, "route_supported": 129, "pretest_route_supported": 120}
    for board_name, count in expected.items():
        if len(cohorts[board_name]) != count:
            raise ValueError(f"Fixed {board_name} membership changed: {len(cohorts[board_name])}")
        cohort_audit["cohorts"][board_name] = {
            "n": count, "membership_sha256": hierarchy_report.membership_hash(cohorts[board_name]),
        }
    metrics = score_boards(cohorts, outputs)
    paired, paired_aggregate = paired_rows(cohorts["pretest_route_supported"], outputs, "statistical/base")
    menu_pairs, menu_pair_aggregate = paired_rows(
        cohorts["pretest_route_supported"],
        {**outputs},
        {name: name.removeprefix("path_menu__") for name in menu_outputs},
    ) if menu_outputs else ([], {})

    # An expanded one-step cohort is intentionally not mixed into fixed scores.
    expanded_rows = list(derived.salvaged_test_diagnostic)
    expanded_report: dict[str, Any] = {"candidate_rows": len(expanded_rows), "separate_from_fixed_test": True}
    if expanded_rows:
        expanded_outputs = {}
        for arm in ARM_PARTS:
            expanded_outputs[f"statistical/{arm}"], _ = replay_statistics(
                arm_training[arm], expanded_rows, history, arm_weights[arm])
        expanded_audited = evaluation_rows(expanded_rows, history)
        expanded_report["scores"] = {
            name: compact_score(evaluation.measure(expanded_audited, result))
            for name, result in expanded_outputs.items()
        }

    source_files = [__file__, cleaning.__file__, statistical.__file__, offline.__file__,
                    evaluation.__file__, asof.__file__, sequence_shadow.__file__,
                    online_suite.__file__, path_policy.__file__]
    run = {
        "version": VERSION, "complete": True,
        "snapshot": str(snapshot), "split": str(split_dir),
        "snapshot_hashes": load_audit["snapshot_hashes"],
        "fixed_train_members": 598, "fixed_test_members": 149,
        "fixed_cohorts": expected,
        "scoring_contract": evaluation.contract(),
        "source_hashes": {Path(path).name: _sha256(Path(path)) for path in source_files},
        "train_only_selection": True, "selected_cleaning_arm": selected_arm,
        "batch_config_source": selection_audit,
        "candidate_scope": "ordinary models use as-of recorded actions; separate path menu derives executable parents from as-of locators with uncalibrated rank proxy",
        "online_simulation_name": "incrementally_updated_no_context_statistical_baseline",
        "raw_snapshot_modified": False, "running_model_modified": False,
        "private_derived_target_ledger_persisted": True,
        "ranked_prediction_lists_persisted": False,
        "test_used_for_selection": False,
    }
    report = _report(metrics, training_counts, selected_arm, len(expanded_rows),
                     menu_pair_aggregate)
    os.umask(0o077)
    output.mkdir(parents=True, mode=0o700)
    output.chmod(0o700)
    offline.write_private_json(output / "run.json", run)
    offline.write_private_json(output / "cleaning-audit.json", derived.audit)
    _write_private_jsonl(output / "derived-ledger-private.jsonl",
                         cleaning.private_ledger_records(derived))
    offline.write_private_json(output / "training-counts.json", training_counts)
    offline.write_private_json(output / "model-audits.json", model_audits)
    offline.write_private_json(output / "path-menu-audit.json", path_menu_audit)
    offline.write_private_json(output / "inner-validation.json", {
        "split": inner, "scores": {name: compact_score(score) for name, score in inner_scores.items()},
        "selected_arm": selected_arm,
        "catboost_input": catboost_input_validation,
        "online_mlp": online_mlp_selection,
        "sequence_input": sequence_selection,
    })
    offline.write_private_json(output / "cohort-audit.json", cohort_audit)
    offline.write_private_json(output / "fixed-metrics.json", metrics)
    offline.write_private_json(output / "paired-aggregate.json", paired_aggregate)
    offline.write_private_json(output / "paired-private.json", paired)
    offline.write_private_json(output / "path-menu-paired-aggregate.json", menu_pair_aggregate)
    offline.write_private_json(output / "path-menu-paired-private.json", menu_pairs)
    offline.write_private_json(output / "expanded-first-step-diagnostic.json", expanded_report)
    offline.write_private_text(output / "report-private.md", report)
    hashes = {path.name: _sha256(path) for path in output.iterdir() if path.is_file()}
    offline.write_private_json(output / "artifact-hashes.json", hashes)
    print(json.dumps({"stage": "complete", "fixed_cohorts": expected,
                      "selected_arm": selected_arm,
                      "expanded_first_step": len(expanded_rows)}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
