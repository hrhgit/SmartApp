#!/usr/bin/env python3
"""Private, fixed-membership ablation of menu reranking versus derived parents."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import cleaned_foreground_data as cleaning
import evaluation_contract as evaluation
import hierarchical_fallback_policy as path_policy
import run_break_cleaning_study as study
import run_hierarchical_statistical_experiment as hierarchy_report
import run_offline_batch_rankers as offline
import run_parallel_destination_experiment as parallel
import run_hierarchical_batch_experiment as prior_batch


def cohorts_for(test, history, boundary):
    audited = study.evaluation_rows(test, history)
    contexts = {row.position: hierarchy_report.build_context(study._prepared(row), history)
                for row in test}
    cohorts, audit = hierarchy_report.build_cohorts(audited, contexts, 139)
    old = {row["position"] for row in cohorts["prior_supported"]}
    route = {row["position"] for row in cohorts["hierarchy_applicable"]}
    cohorts["route_supported"] = [row for row in cohorts["all_legal"]
                                  if row["position"] in old | route]
    cohorts["pretest_route_supported"] = parallel.frozen_training_route_rows(
        cohorts["route_supported"], contexts, history, boundary, parallel.POLICIES[0])
    for name, count in (("all_legal", 139), ("route_supported", 129),
                        ("pretest_route_supported", 120)):
        if len(cohorts[name]) != count:
            raise ValueError(f"Fixed {name} changed to {len(cohorts[name])}")
        audit["cohorts"][name] = {
            "n": count, "membership_sha256": hierarchy_report.membership_hash(cohorts[name]),
        }
    return cohorts, audit


def score_line(name, score):
    n = score["n"]
    success, exact, service = score["success"], score["exact"], score["service"]
    success_text = "/".join(str(success[f"top{k}"]) for k in (1, 2, 4))
    exact_text = "/".join(str(exact[f"top{k}"]) for k in (1, 2, 4))
    service_text = "/".join(str(service[f"top{k}"]) for k in (1, 2, 4))
    gain = score["path_utility"]["product0.6"]["top4_gain"]
    return (f"| {name} | {n} | {success_text} | {exact_text} | {service_text} | "
            f"{gain:.3f} | {score['rank_score']:.3f} |")


def report(metrics, paired, training_counts):
    primary = metrics["pretest_route_supported"]
    lines = ["# 路径菜单的独立对照", "",
             "固定原 598 条训练、149 条测试成员；主榜是测试开始前已有可得分路径的 120 条，另保留全部 139 条合法评分样本。",
             "三档依次是原模型名单、仅重排预测前已记录动作、相同重排再允许从预测前已知网址或目录推导可打开的上级。",
             "排序位置代理未经概率校准；新增上级和重排效果分开算。网页中间层的 0.6 收益只是预声明的实验曲线，尚非约定分值。", "",
             "| 模型或推理方式 | 主榜条数 | 成功 Top1/2/4 | 完全一致 Top1/2/4 | 服务接受 Top1/2/4 | 实验部分收益 Top4 | 已定固定排位分 |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    ordered = ["statistical/base", "statistical/first_step", "statistical/short_join",
               "statistical/both", "path_menu_recorded_only__statistical/both",
               "path_menu__statistical/both", "online_simulation/base",
               "online_simulation/first_step", "online_simulation/short_join",
               "online_simulation/both", "catboost/base", "catboost/both",
               "path_menu_recorded_only__catboost/both", "path_menu__catboost/both"]
    lines.extend(score_line(name, primary[name]) for name in ordered)
    lines += ["", "| 同模型比较 | 成功 Top4 新增/失去 | 净变化 |",
              "|---|---:|---:|"]
    for name, summary in paired.items():
        changes = summary["success"]["top4"]
        lines.append(f"| {summary['reference_model']} → {name} | "
                     f"{changes['helped']}/{changes['hurt']} | {changes['net']} |")
    lines += ["",
              f"清洗 both 实际保留 {training_counts['both']['audit']['first_step_admitted']} 条额外已确认第一步和 "
              f"{training_counts['both']['audit']['qualified_join_admitted']} 条跨合格短断点推定切换。",
              "上一个 final-01 报告把序列模型和路径菜单行的新增训练列误写成 0：它们使用 both，正确数字是 240 和 41；模型得分及成员没有受这项展示错误影响。",
              "测试期间另外 68 条已确认第一步只在 final-01 的独立诊断中，不进入这些固定分母。", ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", required=True, type=Path)
    parser.add_argument("--split", required=True, type=Path)
    parser.add_argument("--batch-selection", required=True, type=Path)
    parser.add_argument("--base-run", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    snapshot, split, batch_selection, base_run, output = (
        item.expanduser().resolve() for item in
        (args.snapshot, args.split, args.batch_selection, args.base_run, args.output))
    private_root = study.PRIVATE_ROOT.expanduser().resolve()
    if not output.is_relative_to(private_root) or output.exists():
        parser.error("Output must be a new private SmartApp training directory")
    original = offline.read_json(base_run / "run.json")
    if original["selected_cleaning_arm"] != "both":
        raise ValueError("This ablation is declared for the selected both arm")
    train, test, history, load_audit = offline.load_observations(snapshot, split)
    if len(train) != 598 or len(test) != 149:
        raise ValueError("Fixed split changed")
    boundary = int(offline.read_json(split / "split-report.json")["boundary"]["eventIndex"])
    events = offline.read_jsonl(snapshot / "events.jsonl")
    samples = offline.read_jsonl(snapshot / "navigation-samples.jsonl")
    derived = cleaning.derive_cleaning(events, samples, history,
                                        test_boundary_index=boundary,
                                        existing_train_rows=train, existing_test_rows=test)
    selected, _ = prior_batch.load_selected_configs(batch_selection, snapshot, split)
    cat_config = selected["catboost"]["config"]
    outputs = {}
    counts = {}
    for arm in ("base", "first_step", "short_join", "both"):
        training, weights, audit = study.training_arm(train, derived, arm,
                                                      boundary_index=boundary)
        counts[arm] = {"rows": len(training), "audit": audit}
        outputs[f"statistical/{arm}"], _ = study.replay_statistics(
            training, test, history, weights)
        outputs[f"online_simulation/{arm}"], _ = study.replay_statistics(
            training, test, history, weights, learn_test=True)
        if arm in ("base", "both"):
            outputs[f"catboost/{arm}"], _ = study.replay_catboost(
                training, test, history, weights, cat_config)
        print(json.dumps({"stage": "models", "arm": arm}), flush=True)

    bases = {name: outputs[name] for name in ("statistical/both", "catboost/both")}
    recorded, recorded_audit = study.path_menu_variants(
        test, history, bases, "both", include_derived=False)
    derived_menu, derived_audit = study.path_menu_variants(
        test, history, bases, "both", include_derived=True)
    outputs.update(recorded)
    outputs.update(derived_menu)
    print(json.dumps({"stage": "menus"}), flush=True)

    cohorts, cohort_audit = cohorts_for(test, history, boundary)
    original_cohort = offline.read_json(base_run / "cohort-audit.json")["cohorts"]
    for name in ("all_legal", "route_supported", "pretest_route_supported"):
        if cohort_audit["cohorts"][name]["membership_sha256"] != original_cohort[name]["membership_sha256"]:
            raise ValueError(f"Fixed {name} members differ from sealed base run")
    metrics = study.score_boards(cohorts, outputs)
    for board_name in study.BOARD_NAMES:
        base_metrics = offline.read_json(base_run / "fixed-metrics.json")[board_name]
        for name in outputs:
            if name in base_metrics and metrics[board_name][name] != base_metrics[name]:
                raise ValueError(f"Original {name} result drifted on {board_name}")
    references = {
        "statistical/first_step": "statistical/base",
        "statistical/short_join": "statistical/base",
        "statistical/both": "statistical/base",
        "online_simulation/first_step": "online_simulation/base",
        "online_simulation/short_join": "online_simulation/base",
        "online_simulation/both": "online_simulation/base",
        "catboost/both": "catboost/base",
        "path_menu_recorded_only__statistical/both": "statistical/both",
        "path_menu__statistical/both": "path_menu_recorded_only__statistical/both",
        "path_menu_recorded_only__catboost/both": "catboost/both",
        "path_menu__catboost/both": "path_menu_recorded_only__catboost/both",
    }
    individual, paired = study.paired_rows(cohorts["pretest_route_supported"],
                                           outputs, references)
    direct, direct_pairs = study.paired_rows(
        cohorts["pretest_route_supported"], outputs,
        {"path_menu__statistical/both": "statistical/both",
         "path_menu__catboost/both": "catboost/both"})
    os.umask(0o077)
    output.mkdir(mode=0o700, parents=True)
    output.chmod(0o700)
    source_files = [Path(__file__), Path(study.__file__), Path(evaluation.__file__),
                    Path(path_policy.__file__), Path(cleaning.__file__),
                    Path(offline.__file__)]
    run = {
        "version": "path-menu-ablation-v1", "complete": True,
        "base_run": str(base_run), "snapshot_hashes": load_audit["snapshot_hashes"],
        "fixed_cohorts": {name: cohort_audit["cohorts"][name] for name in study.BOARD_NAMES},
        "scoring_contract": evaluation.contract(),
        "source_hashes": {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                          for path in source_files},
        "rank_proxy": "same original rank positions and product0.6 greedy menu for both menu arms",
        "recorded_only": "prohibit all structurally derived ancestors, retain recorded actions",
        "derived_allowed": "also allow as-of URL and directory ancestors",
        "original_model_scores_verified_equal_to_base_run": True,
        "ranked_prediction_lists_persisted": False,
    }
    for name, value in (
        ("run.json", run), ("fixed-metrics.json", metrics),
        ("cohort-audit.json", cohort_audit), ("training-counts.json", counts),
        ("paired-aggregate.json", paired), ("paired-direct-aggregate.json", direct_pairs),
        ("path-menu-audit.json", {"recorded_only": recorded_audit,
                                  "derived_allowed": derived_audit}),
    ):
        offline.write_private_json(output / name, value)
    study._write_private_jsonl(output / "paired-private.jsonl", individual)
    study._write_private_jsonl(output / "paired-direct-private.jsonl", direct)
    offline.write_private_text(output / "report-private.md", report(metrics, paired, counts))
    files = {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
             for path in output.iterdir() if path.is_file()}
    offline.write_private_json(output / "artifact-hashes.json", files)
    print(json.dumps({"stage": "complete", "output": str(output),
                      "main_n": metrics["pretest_route_supported"]["statistical/both"]["n"]}), flush=True)


if __name__ == "__main__":
    main()
