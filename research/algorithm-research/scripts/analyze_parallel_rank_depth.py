#!/usr/bin/env python3
"""Aggregate private TopK depth curves without persisting target-level outputs."""

from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path

import compare_multistep_statistical as statistical
import hierarchical_fallback_policy as hierarchy
import run_hierarchical_statistical_experiment as reference
import run_parallel_destination_experiment as parallel


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Aggregate private TopK depth curves without persisting target-level outputs."
    )
    parser.add_argument("--snapshot", type=Path, required=True,
                        help="Private SmartApp v2 export directory")
    parser.add_argument("--split", type=Path, required=True,
                        help="Matching fixed offline split directory")
    args = parser.parse_args()
    train, test, history, *_ = statistical.load_dataset(
        args.snapshot.expanduser().resolve(), args.split.expanduser().resolve()
    )
    outputs, audit = parallel.replay_parallel(train, test, history)
    records = outputs["product0.6"]
    policy = parallel.POLICIES[0]
    contexts = {row.observation.position: reference.build_context(row, history) for row in test}
    emitted_homes = {}
    for position, record in records.items():
        tree = contexts[position].hierarchy
        for action in record["predictions"]:
            metadata = tree.metadata.get(action)
            if metadata and hierarchy.is_service_home(metadata):
                emitted_homes[action] = metadata
    audited = reference.offline.evaluation_rows(
        [row.observation for row in test], emitted_homes,
    )
    cohorts, _ = reference.build_cohorts(audited, contexts, 139)
    old_ids = {row["position"] for row in cohorts["prior_supported"]}
    route_ids = {row["position"] for row in cohorts["hierarchy_applicable"]}
    cohorts["route_supported"] = [
        row for row in cohorts["all_legal"] if row["position"] in old_ids | route_ids
    ]
    report = {}
    for name in ("route_supported", "prior_supported", "all_legal"):
        values = []
        max_rank = 0
        for row in cohorts[name]:
            position = row["position"]
            context = contexts[position]
            truth = reference.normalized_truth(context)
            truth_id = str(truth["id"])
            tree = hierarchy.add_outcome(context.hierarchy, truth)
            ranked = records[position]["rankable"]
            first_path = first_exact = first_full = None
            best = 0.0
            prefix = []
            for rank, action in enumerate(ranked, 1):
                gain = hierarchy.action_utility(
                    tree, action, truth_id, policy,
                    current_action=context.current_action, current_app=context.current_app,
                )
                best = max(best, gain)
                prefix.append(best)
                if gain > 0 and first_path is None:
                    first_path = rank
                if gain >= 1 - 1e-12 and first_full is None:
                    first_full = rank
                if action == truth_id and first_exact is None:
                    first_exact = rank
            oracle = max((hierarchy.action_utility(
                tree, action, truth_id, policy,
                current_action=context.current_action, current_app=context.current_app,
            ) for action in context.hierarchy.executable), default=0.0)
            values.append((first_path, first_exact, first_full, prefix, len(ranked), oracle,
                           truth_id in context.hierarchy.executable))
            max_rank = max(max_rank, len(ranked))
        ks = sorted(set((1, 2, 3, 4, 5, 6, 8, 10, 12, 16, 20, 24, 32, 40, 45, 48,
                         50, 64, 80, 100, 233, max_rank)))
        curve = []
        previous = (0, 0, 0.0)
        for k in ks:
            path = sum(item[0] is not None and item[0] <= k for item in values)
            exact = sum(item[1] is not None and item[1] <= k for item in values)
            full = sum(item[2] is not None and item[2] <= k for item in values)
            gain = sum(item[3][min(k, len(item[3])) - 1] if item[3] else 0.0
                       for item in values) / len(values)
            curve.append({
                "k": k, "path": path, "exact": exact, "full_gain": full,
                "avg_best_gain": round(gain, 5), "new_path": path - previous[0],
                "new_exact": exact - previous[1],
                "gain_delta": round(gain - previous[2], 5),
            })
            previous = path, exact, gain
        thresholds = {}
        for fraction in (0.8, 0.9, 0.95, 1.0):
            needed = math.ceil(fraction * len(values) - 1e-10)
            ranks = sorted(item[0] for item in values if item[0] is not None)
            thresholds[str(fraction)] = {
                "required_n": needed,
                "k": ranks[needed - 1] if len(ranks) >= needed else None,
            }
        report[name] = {
            "n": len(values), "max_ranked": max_rank,
            "ranked_length_quantiles": {
                "min": min(item[4] for item in values),
                "median": sorted(item[4] for item in values)[len(values) // 2],
                "max": max(item[4] for item in values),
            },
            "path_positive_ceiling": sum(item[0] is not None for item in values),
            "exact_positive_ceiling": sum(item[1] is not None for item in values),
            "oracle_path_ceiling": sum(item[5] > 0 for item in values),
            "oracle_full_gain_ceiling": sum(item[5] >= 1 - 1e-12 for item in values),
            "oracle_exact_available": sum(item[6] for item in values),
            "thresholds": thresholds, "curve": curve,
        }
    print(json.dumps({"training_audit": audit, "report": report},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
