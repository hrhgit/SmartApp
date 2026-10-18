#!/usr/bin/env python3
"""Test as-of foreground frequency as another input to parallel destinations.

Each mature independent arrival contributes once to its exact destination and
derived path layers. The same arrivals are viewed at 5m, 1h, and 24h cutoffs;
the views are never new behavior records. Only aggregates are persisted.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import compare_multistep_statistical as statistical
import evaluation_contract as evaluation
import hierarchical_fallback_policy as hierarchy
import run_hierarchical_statistical_experiment as reference
import run_parallel_destination_experiment as parallel


VERSION = "parallel-recent-frequency-v1"
VARIANTS = ("base", "recent5m", "recent1h", "recent24h")


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot", required=True, type=Path)
    parser.add_argument("--split", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    snapshot = args.snapshot.expanduser().resolve()
    split = args.split.expanduser().resolve()
    output = args.output.expanduser().resolve()
    private_root = reference.PRIVATE_ROOT.expanduser().resolve()
    if output == private_root or not output.is_relative_to(private_root):
        raise ValueError("Research output must remain in the private training root")
    if output.exists():
        raise ValueError("Research runs are immutable; choose a new output directory")

    train, test, history, input_audit, sequence_audit = statistical.load_dataset(
        snapshot, split,
    )
    print(json.dumps({"stage": "inputs_verified", "train": len(train), "test": len(test)}),
          flush=True)
    outputs, training_audit = parallel.replay_parallel(
        train, test, history, recent_windows=(0, 1, 2),
    )
    contexts = {row.observation.position: reference.build_context(row, history)
                for row in test}
    emitted_homes = {}
    for records in outputs.values():
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
    old = {row["position"] for row in cohorts["prior_supported"]}
    route = {row["position"] for row in cohorts["hierarchy_applicable"]}
    cohorts["route_supported"] = [row for row in cohorts["all_legal"]
                                  if row["position"] in old | route]
    boundary = int(json.loads((split / "split-report.json").read_text())
                   ["boundary"]["eventIndex"])
    cohorts["pretest_route_supported"] = parallel.frozen_training_route_rows(
        cohorts["route_supported"], contexts, history, boundary, parallel.POLICIES[0],
    )
    expected = {"all_legal": 139, "route_supported": 129,
                "pretest_route_supported": 120}
    for name, count in expected.items():
        if len(cohorts[name]) != count:
            raise ValueError(f"Frozen cohort {name} changed membership size")
        cohort_audit["cohorts"][name] = {
            "n": count, "membership_sha256": reference.membership_hash(cohorts[name]),
        }

    metrics = {}
    for cohort_name in ("pretest_route_supported", "route_supported", "all_legal"):
        rows = cohorts[cohort_name]
        metrics[cohort_name] = {}
        for policy in parallel.POLICIES:
            base = outputs[policy.name]
            for variant in VARIANTS:
                records = (base if variant == "base"
                           else outputs[f"{variant}__{policy.name}"])
                score = {
                    "path": parallel._aggregate(rows, records, contexts, policy),
                    "canonical": evaluation.measure(rows, records),
                }
                if variant != "base":
                    score["paired_vs_base"] = parallel._paired(
                        rows, base, records, contexts, policy,
                    )
                metrics[cohort_name][f"{variant}__{policy.name}"] = score

    lines = ["# 当前层级加近期频率：固定数据实验", "",
             "一次成熟前台到达在具体目标及其路径上级各计一次支持；时间窗是同一行为的不同视图。",
             "输入仅使用预测前已成熟的训练到达；测试期间不更新模型或频率历史。",
             "近期频率只在预测时作为候选目标的历史计数输入，以既有强度向全局频率收缩。",
             "训练尚未学习频率和当前层级的联合关系；本试验只检验加入这种输入是否值得继续研究。",
             "各时间窗有非空历史的测试行数：" + ", ".join(
                 f"{name}={count}" for name, count in training_audit[
                     "query_rows_with_nonempty_recent_history"].items()
             ) + f" / {len(test)}。",
             "", "| 样本组 | 部分收益规则 | 输入 | 路径成功 Top1 | Top2 | Top4 | 完全一致 Top4 | Top4 平均收益 | 相对原版帮助/伤害 |",
             "|---|---|---|---:|---:|---:|---:|---:|---:|"]
    for cohort_name in ("pretest_route_supported", "route_supported", "all_legal"):
        for policy in parallel.POLICIES:
            for variant in VARIANTS:
                score = metrics[cohort_name][f"{variant}__{policy.name}"]
                path, exact = score["path"], score["canonical"]["exact"]
                paired = score.get("paired_vs_base", {})
                pair_text = (f"{paired.get('helped', 0)}/{paired.get('hurt', 0)}"
                             if paired else "—")
                lines.append(
                    f"| {cohort_name} ({path['n']}) | {policy.name} | {variant} | "
                    f"{path['top1_success_n']}/{path['n']} | "
                    f"{path['top2_success_n']}/{path['n']} | "
                    f"{path['top4_success_n']}/{path['n']} | "
                    f"{exact['top4_hits']}/{path['n']} | "
                    f"{path['top4_gain']:.2%} | {pair_text} |"
                )
    lines += ["", "三个时间窗是预先固定的并列试验，不根据测试结果事后挑选赢家。",
              "本试验的频率历史只覆盖已成熟且进入训练的独立到达，尚非所有自然前台到达。", ""]

    os.umask(0o077)
    output.mkdir(parents=True, mode=0o700)
    output.chmod(0o700)
    write_json = reference.offline.write_private_json
    write_text = reference.offline.write_private_text
    write_json(output / "input-audit.json", input_audit)
    write_json(output / "sequence-input-audit.json", sequence_audit)
    write_json(output / "training-audit.json", training_audit)
    write_json(output / "cohort-audit.json", cohort_audit)
    write_json(output / "aggregate-metrics.json", metrics)
    write_text(output / "report-private.md", "\n".join(lines))
    sources = {name: digest(Path(__file__).with_name(name)) for name in (
        "parallel_destination_data.py", "parallel_destination_model.py",
        "run_parallel_destination_experiment.py",
        "run_parallel_recent_frequency_experiment.py",
        "hierarchical_fallback_policy.py", "evaluation_contract.py",
    )}
    artifacts = {path.name: digest(path) for path in output.iterdir() if path.is_file()}
    write_json(output / "artifact-hashes.json", artifacts)
    write_json(output / "run.json", {
        "version": VERSION, "complete": True, "snapshot": str(snapshot),
        "split": str(split), "train_members": len(train), "test_members": len(test),
        "cohort_sizes": expected, "source_hashes": sources,
        "artifact_hashes_sha256": digest(output / "artifact-hashes.json"),
        "raw_predictions_persisted": False,
        "target_identifiers_or_locators_persisted": False,
        "candidate_lists_persisted": False,
        "model_parameters_persisted": False,
        "test_used_for_selection": False,
    })
    print(json.dumps({"stage": "complete", "cohorts": expected}), flush=True)


if __name__ == "__main__":
    main()
