#!/usr/bin/env python3
"""Compare a causal parallel-layer learner with fixed statistical baselines.

The same confirmed arrival supplies one exact label plus derived path labels.
All raw layers, predictions, and target locators stay in memory.  Only aggregate
scores and provenance are saved in the private training directory.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import compare_multistep_statistical as statistical
import evaluation_contract as evaluation
import hierarchical_fallback_policy as hierarchy
import parallel_destination_data as data
from parallel_destination_model import ParallelDestinationModel
import run_hierarchical_statistical_experiment as reference


VERSION = "parallel-destination-experiment-v1"
POLICIES = (
    hierarchy.FallbackPolicy("product0.6", "product", content_retention=.6),
    hierarchy.FallbackPolicy("product0.8", "product", content_retention=.8),
)
MODEL = "parallel_path_count"
RECENT_WINDOW_NAMES = {0: "recent5m", 1: "recent1h", 2: "recent24h"}


def _catalog(history: Any, index: int, *extra: Any) -> list[Mapping]:
    result = {action.key: action.meta for action in history.catalog_at(index).values()}
    for action in extra:
        if action is not None:
            result[action.key] = action.meta
    return list(result.values())


def replay_parallel(
    training_rows: Sequence[statistical.PreparedRow],
    query_rows: Sequence[statistical.PreparedRow], history: Any,
    *, recent_windows: tuple[int, ...] = (),
) -> tuple[dict[str, dict[int, dict[str, Any]]], dict[str, Any]]:
    """Only completed outer-training outcomes may update the shared layer model."""
    model = ParallelDestinationModel(shrinkage=5.0)
    stream = sorted(
        (row for row in training_rows if statistical.navigation_training_row(row)),
        key=lambda row: (row.observation.completed_event_index,
                         row.observation.input_event_index, row.observation.position),
    )
    outputs: dict[str, dict[int, dict[str, Any]]] = {
        policy.name: {} for policy in POLICIES
    }
    for window in recent_windows:
        if window not in RECENT_WINDOW_NAMES:
            raise ValueError("Unknown recent frequency experiment window")
        for policy in POLICIES:
            outputs[f"{RECENT_WINDOW_NAMES[window]}__{policy.name}"] = {}
    frequency_history = data.FrequencyHistory()
    cursor = 0
    input_layer_counts: Counter[int] = Counter()
    output_layer_counts: Counter[int] = Counter()
    recent_active_rows: Counter[str] = Counter()
    for row in sorted(query_rows, key=lambda item: (
            item.observation.input_event_index, item.observation.position)):
        observation = row.observation
        cutoff = int(observation.input_event_index)
        now = float(observation.input_timestamp_ms)
        while cursor < len(stream) and stream[cursor].observation.completed_event_index < cutoff:
            mature = stream[cursor].observation
            if mature.input_event_index >= cutoff or mature.outcome_timestamp_ms >= now:
                raise ValueError("Future behavior entered parallel destination training")
            assert mature.current is not None and mature.truth is not None
            arrival = data.make_arrival(
                outcome_id=mature.outcome_event_id,
                input_event_index=int(mature.input_event_index),
                mature_event_index=int(mature.completed_event_index),
                current=mature.current.meta, destination=mature.truth.meta,
                input_catalog=_catalog(history, int(mature.input_event_index),
                                       mature.current, mature.current_parent),
                mature_catalog=_catalog(history, int(mature.completed_event_index),
                                        mature.truth, mature.truth_parent),
            )
            if model.learn(arrival):
                frequency_history.observe(arrival, float(mature.outcome_timestamp_ms))
                input_layer_counts[len(arrival.current)] += 1
                output_layer_counts[len(arrival.destination)] += 1
            cursor += 1

        context = reference.build_context(row, history)
        position = int(observation.position)
        if observation.current is None:
            for variant in outputs.values():
                variant[position] = {"predictions": [], "rankable": []}
            continue
        input_tree = data.focused_hierarchy(
            _catalog(history, cutoff, observation.current, observation.current_parent),
            observation.current.meta,
        )
        current_layers = data.layers_for(input_tree, observation.current.key)
        frequency_input = (frequency_history.snapshot(now) if recent_windows else None)
        if frequency_input is not None:
            for window in recent_windows:
                if frequency_input.total[window] > 0:
                    recent_active_rows[RECENT_WINDOW_NAMES[window]] += 1
        for policy in POLICIES:
            prediction = model.predict(
                current_layers, context.hierarchy, policy,
                current_action=context.current_action, current_app=context.current_app,
            )
            outputs[policy.name][position] = {
                "predictions": list(prediction.ranked[:4]),
                "rankable": list(prediction.ranked),
            }
            for window in recent_windows:
                recent_prediction = model.predict(
                    current_layers, context.hierarchy, policy,
                    current_action=context.current_action,
                    current_app=context.current_app,
                    frequency=frequency_input, recent_window=window,
                )
                outputs[f"{RECENT_WINDOW_NAMES[window]}__{policy.name}"][position] = {
                    "predictions": list(recent_prediction.ranked[:4]),
                    "rankable": list(recent_prediction.ranked),
                }
    return outputs, {
        "training_stream_rows": len(stream),
        "learned_independent_arrivals": model.global_n,
        "input_layer_counts_per_arrival": dict(sorted(input_layer_counts.items())),
        "output_layer_counts_per_arrival": dict(sorted(output_layer_counts.items())),
        "shrinkage": model.shrinkage,
        "prediction_rows": len(query_rows),
        "recent_frequency_windows": [RECENT_WINDOW_NAMES[window] for window in recent_windows],
        "query_rows_with_nonempty_recent_history": {
            RECENT_WINDOW_NAMES[window]: recent_active_rows[RECENT_WINDOW_NAMES[window]]
            for window in recent_windows
        },
    }


def _aggregate(rows: Sequence[Mapping], output: Mapping[int, Mapping],
               contexts: Mapping[int, reference.TransientContext],
               policy: hierarchy.FallbackPolicy) -> dict[str, Any]:
    totals = Counter()
    success = Counter()
    for row in rows:
        position = row["position"]
        context = contexts[position]
        score = hierarchy.experimental_rank_score(
            output[position]["predictions"], reference.normalized_truth(context),
            context.hierarchy, policy,
            current_action=context.current_action, current_app=context.current_app,
        )
        for name, value in score.items():
            totals[name] += value
            if name in {"top1", "top2", "top4"}:
                success[name] += value > 0
    n = len(rows)
    return {
        "n": n,
        **{f"{name}_success_n": success[name] for name in ("top1", "top2", "top4")},
        **{f"{name}_gain": totals[name] / n if n else None
           for name in ("top1", "top2", "top4", "rank_score")},
    }


def _paired(rows: Sequence[Mapping], before: Mapping[int, Mapping],
            after: Mapping[int, Mapping],
            contexts: Mapping[int, reference.TransientContext],
            policy: hierarchy.FallbackPolicy) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for row in rows:
        position = row["position"]
        context = contexts[position]
        common = (reference.normalized_truth(context), context.hierarchy, policy)
        options = {"current_action": context.current_action,
                   "current_app": context.current_app}
        left = hierarchy.experimental_rank_score(before[position]["predictions"],
                                                 *common, **options)["top4"]
        right = hierarchy.experimental_rank_score(after[position]["predictions"],
                                                  *common, **options)["top4"]
        counts["helped" if right > left + 1e-12 else
               "hurt" if right < left - 1e-12 else "unchanged"] += 1
    return dict(counts)


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def frozen_training_route_rows(
    rows: Sequence[Mapping], contexts: Mapping[int, reference.TransientContext],
    history: Any, boundary_index: int, policy: hierarchy.FallbackPolicy,
) -> list[Mapping]:
    """Keep outcomes whose useful route existed before the frozen test period.

    The realized outcome is used only to form this offline comparison group.
    It never enters prediction inputs or changes the full legal leaderboard.
    """
    if boundary_index <= 0:
        raise ValueError("A positive fixed-test boundary is required")
    catalog = history.catalog_at(boundary_index - 1)
    pretest_tree = hierarchy.build_hierarchy(
        (action.meta for action in catalog.values()), executable=catalog.keys(),
        include_owner_apps=True,
    )
    selected = []
    for row in rows:
        context = contexts[row["position"]]
        truth = reference.normalized_truth(context)
        if not truth.get("id"):
            continue
        tree = hierarchy.add_outcome(pretest_tree, truth)
        if any(hierarchy.action_utility(
            tree, action, str(truth["id"]), policy,
            current_action=context.current_action, current_app=context.current_app,
        ) > 0 for action in pretest_tree.executable):
            selected.append(row)
    return selected


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    snapshot = args.snapshot.expanduser().resolve()
    split = args.split.expanduser().resolve()
    output = args.output.expanduser().resolve()
    private_root = reference.PRIVATE_ROOT.expanduser().resolve()
    if output == private_root or not output.is_relative_to(private_root):
        raise ValueError("Research output must stay under the private training root")
    if output.exists():
        raise ValueError("Research runs are immutable; use a fresh output directory")

    train, test, history, input_audit, sequence_audit = statistical.load_dataset(
        snapshot, split,
    )
    os.umask(0o077)
    output.mkdir(parents=True, mode=0o700)
    output.chmod(0o700)
    print(json.dumps({"stage": "inputs_verified", "train": len(train),
                      "test": len(test)}), flush=True)

    baseline, contexts, baseline_audit = reference.replay(train, test, history)
    parallel, parallel_audit = replay_parallel(train, test, history)
    print(json.dumps({"stage": "predictions_built", "models": len(baseline) + 1}),
          flush=True)

    all_outputs = {
        **{f"{name}__{policy.name}": variants[policy.name]
           for name, variants in baseline.items() for policy in POLICIES},
        **{f"{MODEL}__{name}": records for name, records in parallel.items()},
    }
    emitted_homes = {}
    for records in all_outputs.values():
        for position, result in records.items():
            tree = contexts[position].hierarchy
            for action in result["predictions"]:
                metadata = tree.metadata.get(action)
                if metadata and hierarchy.is_service_home(metadata):
                    emitted_homes[action] = metadata
    audited = reference.offline.evaluation_rows(
        [row.observation for row in test], emitted_homes,
    )
    cohorts, cohort_audit = reference.build_cohorts(audited, contexts, 139)
    old_ids = {row["position"] for row in cohorts["prior_supported"]}
    route_ids = {row["position"] for row in cohorts["hierarchy_applicable"]}
    cohorts["route_supported"] = [row for row in cohorts["all_legal"]
                                  if row["position"] in old_ids | route_ids]
    boundary_index = int(json.loads((split / "split-report.json").read_text())
                         ["boundary"]["eventIndex"])
    cohorts["pretest_route_supported"] = frozen_training_route_rows(
        cohorts["route_supported"], contexts, history, boundary_index, POLICIES[0],
    )
    cohort_audit["cohorts"]["route_supported"] = {
        "n": len(cohorts["route_supported"]),
        "membership_sha256": reference.membership_hash(cohorts["route_supported"]),
    }
    cohort_audit["membership"]["route_supported"] = (
        "Prediction-before-known exact destination, usable app, or structurally "
        "derived URL/directory ancestor of the realized outcome; posthoc cohort "
        "membership does not depend on the model prediction."
    )
    cohort_audit["cohorts"]["pretest_route_supported"] = {
        "n": len(cohorts["pretest_route_supported"]),
        "membership_sha256": reference.membership_hash(cohorts["pretest_route_supported"]),
    }
    cohort_audit["membership"]["pretest_route_supported"] = (
        "For this frozen offline test, the realized outcome had an as-of executable "
        "exact or derived ancestor route already present before the test boundary; "
        "posthoc membership, independent of model predictions."
    )

    metrics: dict[str, Any] = {}
    for cohort_name in ("all_legal", "prior_supported", "route_supported",
                        "pretest_route_supported"):
        rows = cohorts[cohort_name]
        metrics[cohort_name] = {}
        for policy in POLICIES:
            for name in (reference.MODEL_BASELINE, reference.MODEL_MARKOV2, MODEL):
                records = (parallel[policy.name] if name == MODEL
                           else baseline[name][policy.name])
                metrics[cohort_name][f"{name}__{policy.name}"] = {
                    "path": _aggregate(rows, records, contexts, policy),
                    "canonical": evaluation.measure(rows, records),
                }
            metrics[cohort_name][f"{MODEL}__{policy.name}"]["paired_vs_no_context"] = _paired(
                rows, baseline[reference.MODEL_BASELINE][policy.name],
                parallel[policy.name], contexts, policy,
            )

    lines = ["# 并行目的地层级实验", "",
             "一次真实到达同时提供当前和结果的具体目标及路径层级；中间层不是额外到达。",
             "原统计模型在同一训练/测试切分上重放；模型与菜单结果只保留在内存。",
             "路径成功含合法中间层，完全一致目标与部分收益分列。", "",
             "| 样本组 | 曲线 | 模型 | 成功Top1 | 成功Top2 | 成功Top4 | 完全一致Top1 | 完全一致Top2 | 完全一致Top4 | 服务接受Top1 | 服务接受Top2 | 服务接受Top4 | 部分收益Top4 | 较无情景统计帮助/伤害 |",
             "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for cohort_name in ("pretest_route_supported", "route_supported",
                        "prior_supported", "all_legal"):
        for policy in POLICIES:
            for name in (reference.MODEL_BASELINE, reference.MODEL_MARKOV2, MODEL):
                score = metrics[cohort_name][f"{name}__{policy.name}"]
                path, canonical = score["path"], score["canonical"]
                pair = score.get("paired_vs_no_context", {})
                pair_text = (f"{pair.get('helped', 0)}/{pair.get('hurt', 0)}"
                             if pair else "—")
                lines.append(
                    f"| {cohort_name} ({path['n']}) | {policy.name} | {name} | "
                    f"{path['top1_success_n']}/{path['n']} | "
                    f"{path['top2_success_n']}/{path['n']} | "
                    f"{path['top4_success_n']}/{path['n']} | "
                    f"{canonical['exact']['top1_hits']}/{path['n']} | "
                    f"{canonical['exact']['top2_hits']}/{path['n']} | "
                    f"{canonical['exact']['top4_hits']}/{path['n']} | "
                    f"{canonical['top1_hits']}/{path['n']} | "
                    f"{canonical['top2_hits']}/{path['n']} | "
                    f"{canonical['top4_hits']}/{path['n']} | "
                    f"{path['top4_gain']:.2%} | {pair_text} |"
                )
    lines += ["", "冻结模型主比较榜 pretest_route_supported 要求测试开始前已有可用的具体目标或路径上级；测试期间首次出现的全部可用入口不纳入该榜。该条件不保证目标曾作为训练结果。",
              "route_supported 保留预测前可用入口的旧口径；all_legal 保留所有合法样本。",
              "普通中间层的 0.6/0.8 是预先固定的试验曲线；服务首页始终使用已约定的 0.6。",
              "`route_supported` 是事后看真实目标路径建立的可用依据组，不能用作预测时的置信保证。", ""]

    write_json = reference.offline.write_private_json
    write_text = reference.offline.write_private_text
    write_json(output / "input-audit.json", input_audit)
    write_json(output / "sequence-input-audit.json", sequence_audit)
    write_json(output / "cohort-audit.json", cohort_audit)
    write_json(output / "training-audit.json", {"baseline": baseline_audit,
                                                 "parallel": parallel_audit})
    write_json(output / "aggregate-metrics.json", metrics)
    write_text(output / "report-private.md", "\n".join(lines))
    sources = {}
    for name in ("parallel_destination_data.py", "parallel_destination_model.py",
                 "run_parallel_destination_experiment.py", "hierarchical_fallback_policy.py",
                 "evaluation_contract.py"):
        sources[name] = _hash(Path(__file__).with_name(name))
    artifacts = {path.name: _hash(path) for path in output.iterdir() if path.is_file()}
    write_json(output / "artifact-hashes.json", artifacts)
    write_json(output / "run.json", {
        "version": VERSION, "complete": True, "snapshot": str(snapshot),
        "split": str(split), "train_members": len(train), "test_members": len(test),
        "all_legal_n": len(cohorts["all_legal"]),
        "prior_supported_n": len(cohorts["prior_supported"]),
        "route_supported_n": len(cohorts["route_supported"]),
        "pretest_route_supported_n": len(cohorts["pretest_route_supported"]),
        "source_hashes": sources,
        "artifact_hashes_sha256": _hash(output / "artifact-hashes.json"),
        "raw_predictions_persisted": False,
        "target_identifiers_or_locators_persisted": False,
        "candidate_lists_persisted": False,
        "model_parameters_persisted": False,
        "test_used_for_selection": False,
    })
    print(json.dumps({"stage": "complete", "all_legal": len(cohorts["all_legal"]),
                      "route_supported": len(cohorts["route_supported"])}), flush=True)


if __name__ == "__main__":
    main()
