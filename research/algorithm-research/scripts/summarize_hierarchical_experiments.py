#!/usr/bin/env python3
"""汇总一轮批量模型和一轮序列模型的分层回退实验。

输入必须是已完成的 ``hierarchical-batch-experiment-v3`` 和
``hierarchical-sequence-experiment-v3`` 私有研究目录。本脚本只读取其中的聚合
JSON，核对来源文件、统一评分规则和固定样本成员，然后生成一个新的私有汇总
目录。它不会读取或写出目标编号、URL、本地目标路径、逐条预测或候选清单。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Iterable, Mapping


VERSION = "hierarchical-experiment-summary-v3"
PRIVATE_ROOT = Path.home() / "Library/Application Support/SmartApp/training"
BATCH_VERSION = "hierarchical-batch-experiment-v3"
SEQUENCE_VERSION = "hierarchical-sequence-experiment-v3"
FIXED_COUNTS = {
    "test_members": 149,
    "all_legal": 139,
}
COHORTS = ("all_legal", "prior_supported", "hierarchy_applicable")
REPORT_COHORTS = ("all_legal", "prior_supported")
POLICIES = ("product0.6", "product0.8", "saturating0.8")
BATCH_MODELS = (
    "no_context_statistical",
    "conditional_logit",
    "catboost_yetirank",
    "lightfm_hybrid",
)
SEQUENCE_MODELS = (
    "no_context_statistical",
    "rnn_residual",
    "lstm_residual",
    "transformer_residual",
)
EXPECTED_BATCH_SEMANTICS = {
    "no_context_statistical": ("logits_softmax",),
    "conditional_logit": ("logits_softmax",),
    "catboost_yetirank": ("logits_softmax", "rank_proxy"),
    "lightfm_hybrid": ("logits_softmax", "rank_proxy"),
}


class ValidationError(ValueError):
    """输入研究目录无法支持可比较汇总。"""


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_digest(value: Any) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValidationError(f"无法读取聚合文件：{path.name}") from error
    if not isinstance(value, dict):
        raise ValidationError(f"聚合文件必须是 JSON 对象：{path.name}")
    return value


def finite_ratio(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValidationError(f"{label} 不是数字")
    result = float(value)
    if not math.isfinite(result) or result < -1e-12 or result > 1.0 + 1e-12:
        raise ValidationError(f"{label} 超出 0 到 1")
    return min(1.0, max(0.0, result))


def integer(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValidationError(f"{label} 不是整数")
    return value


def require_private_source(path: Path, label: str) -> Path:
    root = PRIVATE_ROOT.expanduser().resolve()
    resolved = path.expanduser().resolve()
    if not resolved.is_dir() or resolved == root or not resolved.is_relative_to(root):
        raise ValidationError(f"{label} 必须是 SmartApp training 下的研究目录")
    return resolved


def verify_artifacts(run_dir: Path, required: Iterable[str]) -> dict[str, Any]:
    """核对来源目录的清单；只读取文件散列，不读取非聚合内容。"""
    manifest_path = run_dir / "artifact-hashes.json"
    manifest = read_json(manifest_path)
    for name in required:
        if name not in manifest:
            raise ValidationError(f"来源清单缺少聚合文件：{name}")
    for relative, expected in manifest.items():
        if not isinstance(relative, str) or not isinstance(expected, str):
            raise ValidationError("来源文件清单格式错误")
        candidate = Path(relative)
        if candidate.is_absolute() or ".." in candidate.parts:
            raise ValidationError("来源文件清单含有越界名称")
        path = (run_dir / candidate).resolve()
        if not path.is_relative_to(run_dir) or not path.is_file():
            raise ValidationError(f"来源文件不存在：{candidate.name}")
        if sha256(path) != expected:
            raise ValidationError(f"来源文件散列不一致：{candidate.name}")
    return {
        "manifest_sha256": sha256(manifest_path),
        "verified_files": len(manifest),
    }


def require_complete_sources(batch_dir: Path, sequence_dir: Path) -> dict[str, Any]:
    batch_run = read_json(batch_dir / "run.json")
    sequence_run = read_json(sequence_dir / "run.json")
    sequence_summary = read_json(sequence_dir / "summary.json")
    if batch_run.get("version") != BATCH_VERSION or batch_run.get("complete") is not True:
        raise ValidationError("批量模型来源不是已完成的指定版本")
    if sequence_run.get("version") != SEQUENCE_VERSION or sequence_run.get("complete") is not True:
        raise ValidationError("序列模型来源不是已完成的指定版本")
    if sequence_summary.get("version") != SEQUENCE_VERSION or sequence_summary.get("complete") is not True:
        raise ValidationError("序列模型汇总没有完成")
    if batch_run.get("test_inference_passes") != 1 or batch_run.get("test_used_for_training_or_selection") is not False:
        raise ValidationError("批量模型没有保留固定的一次测试推理边界")
    if sequence_run.get("test_scored_once") is not True or sequence_summary.get("test_scored_once") is not True:
        raise ValidationError("序列模型没有保留固定的一次测试评分边界")
    forbidden_true = (
        (batch_run, "candidate_lists_persisted"),
        (batch_run, "raw_predictions_persisted"),
        (batch_run, "target_identifiers_or_locators_persisted"),
        (sequence_run, "raw_predictions_or_candidate_lists_written"),
        (sequence_summary, "raw_predictions_or_candidate_lists_written"),
    )
    if any(record.get(key) is not False for record, key in forbidden_true):
        raise ValidationError("来源实验没有通过聚合结果隐私检查")
    return {
        "batch_run": batch_run,
        "sequence_run": sequence_run,
        "sequence_summary": sequence_summary,
    }


def score_view(score: Mapping[str, Any], expected_n: int, label: str) -> dict[str, Any]:
    if integer(score.get("n"), f"{label}.n") != expected_n:
        raise ValidationError(f"{label} 的样本数不一致")
    exact = score.get("exact")
    rank = score.get("rank_score")
    if not isinstance(exact, Mapping) or not isinstance(rank, Mapping):
        raise ValidationError(f"{label} 缺少精确分或综合位置分")
    return {
        "n": expected_n,
        "accepted": {
            f"top{k}": finite_ratio(score.get(f"top{k}"), f"{label}.accepted.top{k}")
            for k in (1, 2, 4)
        },
        "exact": {
            f"top{k}": finite_ratio(exact.get(f"top{k}"), f"{label}.exact.top{k}")
            for k in (1, 2, 4)
        },
        "rank_score": finite_ratio(rank.get("mean"), f"{label}.rank_score"),
    }


def transition(before: Mapping[str, Any], after: Mapping[str, Any]) -> dict[str, Any]:
    if before["n"] != after["n"]:
        raise ValidationError("成对比较的样本数不一致")
    return {
        "n": before["n"],
        "accepted": {
            key: {
                "before": before["accepted"][key],
                "after": after["accepted"][key],
                "delta": after["accepted"][key] - before["accepted"][key],
            }
            for key in ("top1", "top2", "top4")
        },
        "exact": {
            key: {
                "before": before["exact"][key],
                "after": after["exact"][key],
                "delta": after["exact"][key] - before["exact"][key],
            }
            for key in ("top1", "top2", "top4")
        },
        "rank_score": {
            "before": before["rank_score"],
            "after": after["rank_score"],
            "delta": after["rank_score"] - before["rank_score"],
        },
    }


def help_harm_item(value: Mapping[str, Any], expected_n: int, label: str) -> dict[str, int]:
    n = integer(value.get("n", expected_n), f"{label}.n")
    helped = integer(value.get("helped"), f"{label}.helped")
    hurt = integer(value.get("hurt"), f"{label}.hurt")
    unchanged = integer(value.get("unchanged", n - helped - hurt), f"{label}.unchanged")
    if n != expected_n or min(helped, hurt, unchanged) < 0 or helped + hurt + unchanged != n:
        raise ValidationError(f"{label} 的帮助/伤害条数不一致")
    return {"n": n, "helped": helped, "hurt": hurt, "unchanged": unchanged}


def help_harm_block(
    canonical: Mapping[str, Any], expected_n: int,
    experimental: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    accepted = canonical.get("accepted", canonical.get("canonical_accepted"))
    exact = canonical.get("exact", canonical.get("canonical_exact"))
    if not isinstance(accepted, Mapping) or not isinstance(exact, Mapping):
        raise ValidationError("帮助/伤害汇总缺少统一评分")
    result: dict[str, Any] = {
        "n": expected_n,
        "canonical_accepted": {
            f"top{k}": help_harm_item(
                accepted[f"top{k}"], expected_n, f"accepted.top{k}",
            )
            for k in (1, 2, 4)
        },
        "canonical_exact": {
            f"top{k}": help_harm_item(exact[f"top{k}"], expected_n, f"exact.top{k}")
            for k in (1, 2, 4)
        },
    }
    rank = canonical.get("canonical_rank_score")
    if isinstance(rank, Mapping):
        result["canonical_rank_score"] = help_harm_item(
            rank, expected_n, "canonical_rank_score",
        )
    experimental_block = experimental
    if experimental_block is None:
        possible = canonical.get("experimental_multilevel")
        experimental_block = possible if isinstance(possible, Mapping) else None
    if isinstance(experimental_block, Mapping):
        result["experimental_multilevel"] = {
            metric: help_harm_item(
                experimental_block[metric], expected_n, f"experimental.{metric}",
            )
            for metric in ("top1", "top2", "top4", "rank_score")
        }
    return result


def normalized_policy_definitions(
    batch_experimental: Mapping[str, Any], sequence_experimental: Mapping[str, Any],
) -> list[dict[str, Any]]:
    batch: dict[str, dict[str, Any]] = {}
    for name in POLICIES:
        item = batch_experimental.get(name)
        if not isinstance(item, Mapping) or not isinstance(item.get("utility"), Mapping):
            raise ValidationError(f"批量模型缺少固定曲线：{name}")
        utility = item["utility"]
        batch[name] = {
            "name": name,
            "mode": utility.get("mode"),
            "content_retention": utility.get("content_retention"),
            "content_retention_scope": utility.get("content_retention_scope"),
            "service_home_retention": utility.get("service_home_retention"),
            "app_retention": utility.get("app_retention"),
            "floor": utility.get("floor") if utility.get("mode") == "saturating" else None,
        }
    sequence_items = sequence_experimental.get("policies")
    if not isinstance(sequence_items, list):
        raise ValidationError("序列模型缺少固定曲线说明")
    sequence: dict[str, dict[str, Any]] = {}
    for raw in sequence_items:
        if not isinstance(raw, Mapping) or raw.get("name") not in POLICIES:
            raise ValidationError("序列模型含有未约定的曲线")
        name = str(raw["name"])
        sequence[name] = {
            "name": name,
            "mode": raw.get("mode"),
            "content_retention": raw.get("content_retention"),
            "content_retention_scope": raw.get("content_retention_scope"),
            "service_home_retention": raw.get("service_home_retention"),
            "app_retention": raw.get("app_retention"),
            "floor": raw.get("floor") if raw.get("mode") == "saturating" else None,
        }
    if set(sequence) != set(POLICIES) or any(batch[name] != sequence[name] for name in POLICIES):
        raise ValidationError("两类模型使用的三条固定曲线不一致")
    if any(
        batch[name]["content_retention_scope"] != "explicit_and_structurally_derived_url_or_directory_parent_levels"
        or batch[name]["service_home_retention"] != 0.6
        for name in POLICIES
    ):
        raise ValidationError("普通父级曲线错误地改写了固定的首页0.6")
    return [batch[name] for name in POLICIES]


def comparison_keys(
    batch_canonical: Mapping[str, Any], sequence_canonical: Mapping[str, Any],
) -> dict[str, str]:
    result: dict[str, str] = {}
    for cohort in COHORTS:
        keys: set[str] = set()
        batch_scores = batch_canonical.get("cohorts", {}).get(cohort)
        sequence_scores = sequence_canonical.get("cohorts", {}).get(cohort)
        if not isinstance(batch_scores, Mapping) or not isinstance(sequence_scores, Mapping):
            raise ValidationError(f"缺少固定样本组：{cohort}")
        for score in batch_scores.values():
            if not isinstance(score, Mapping) or not isinstance(score.get("comparison_key"), str):
                raise ValidationError(f"批量模型 {cohort} 缺少成员核对码")
            keys.add(str(score["comparison_key"]))
        for variants in sequence_scores.values():
            if not isinstance(variants, Mapping):
                raise ValidationError(f"序列模型 {cohort} 格式错误")
            for score in variants.values():
                if not isinstance(score, Mapping) or not isinstance(score.get("comparison_key"), str):
                    raise ValidationError(f"序列模型 {cohort} 缺少成员核对码")
                keys.add(str(score["comparison_key"]))
        if len(keys) != 1:
            raise ValidationError(f"两类模型的 {cohort} 成员不一致")
        result[cohort] = next(iter(keys))
    return result


def validate_shared_inputs(
    batch_run: Mapping[str, Any], sequence_run: Mapping[str, Any],
    sequence_summary: Mapping[str, Any], batch_input: Mapping[str, Any],
    sequence_input: Mapping[str, Any], batch_canonical: Mapping[str, Any],
    sequence_canonical: Mapping[str, Any],
) -> dict[str, Any]:
    snapshot_values = (
        batch_run.get("input_snapshot_hashes"),
        sequence_run.get("input_snapshot_hashes"),
        batch_input.get("snapshot_hashes"),
        sequence_input.get("snapshot_hashes"),
    )
    if not all(isinstance(value, Mapping) for value in snapshot_values):
        raise ValidationError("来源实验缺少快照散列")
    if any(value != snapshot_values[0] for value in snapshot_values[1:]):
        raise ValidationError("两类模型使用的快照不同")
    for source, audit in (("批量", batch_input), ("序列", sequence_input)):
        if audit.get("train_members") != 598 or audit.get("test_members") != FIXED_COUNTS["test_members"]:
            raise ValidationError(f"{source}模型没有使用固定的 598/149 切分")
    batch_counts = {
        "test_members": batch_run.get("test_members"),
        "all_legal": batch_run.get("all_legal_n"),
        "prior_supported": batch_run.get("prior_supported_n"),
        "hierarchy_applicable": batch_run.get("hierarchy_applicable_n"),
    }
    sequence_counts = {
        "test_members": sequence_summary.get("test_members"),
        "all_legal": sequence_summary.get("legal_members"),
        "prior_supported": sequence_summary.get("prior_supported_members"),
        "hierarchy_applicable": sequence_summary.get("hierarchy_applicable_members"),
    }
    if batch_counts != sequence_counts:
        raise ValidationError("两类模型的合法及预测前依据成员数不一致")
    if any(batch_counts.get(key) != value for key, value in FIXED_COUNTS.items()):
        raise ValidationError("模型没有使用固定的 149 个测试成员和 139 个合法成员")
    for cohort in ("prior_supported", "hierarchy_applicable"):
        value = batch_counts.get(cohort)
        if (isinstance(value, bool) or not isinstance(value, int)
                or value < 0 or value > batch_counts["all_legal"]):
            raise ValidationError(f"{cohort} 成员数无效")
    keys = comparison_keys(batch_canonical, sequence_canonical)
    return {
        "same_snapshot": True,
        "same_fixed_split": True,
        "snapshot_signature_sha256": json_digest(snapshot_values[0]),
        "cohort_comparison_keys": keys,
        "counts": dict(batch_counts),
        "split_evidence": (
            "两边的快照散列、598/149 切分人数及所有预测前依据成员核对码一致"
        ),
    }


def validate_contract(
    batch_run: Mapping[str, Any], sequence_run: Mapping[str, Any],
    batch_canonical: Mapping[str, Any], sequence_canonical: Mapping[str, Any],
) -> dict[str, str]:
    batch_contract = batch_canonical.get("contract")
    sequence_contract = sequence_canonical.get("scoring_contract")
    if not isinstance(batch_contract, Mapping) or not isinstance(sequence_contract, Mapping):
        raise ValidationError("来源实验缺少统一评分说明")
    values = {
        batch_run.get("evaluation_rules_sha256"),
        sequence_run.get("scoring_contract_sha256"),
        batch_contract.get("rules_sha256"),
        sequence_contract.get("rules_sha256"),
    }
    if len(values) != 1 or None in values:
        raise ValidationError("两类模型使用的评分规则不同")
    batch_policy = batch_run.get("source_hashes", {}).get("hierarchical_fallback_policy.py")
    sequence_policy = sequence_run.get("hierarchical_policy_source_sha256")
    if not isinstance(batch_policy, str) or batch_policy != sequence_policy:
        raise ValidationError("两类模型使用的分层回退实现不同")
    return {
        "scoring_contract_sha256": str(next(iter(values))),
        "hierarchical_policy_source_sha256": batch_policy,
    }


def batch_comparisons(
    canonical: Mapping[str, Any], paired: Mapping[str, Any],
    cohort_counts: Mapping[str, int],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    variants = canonical.get("variants")
    cohorts = canonical.get("cohorts")
    canonical_pairs = paired.get("canonical")
    experimental_pairs = paired.get("experimental_multilevel")
    if not all(isinstance(item, Mapping) for item in (variants, cohorts, canonical_pairs, experimental_pairs)):
        raise ValidationError("批量模型聚合结果格式错误")
    models = {str(meta.get("model")) for meta in variants.values() if isinstance(meta, Mapping)}
    if models != set(BATCH_MODELS):
        raise ValidationError("批量模型名单不完整")
    suppression: list[dict[str, Any]] = []
    hierarchy: list[dict[str, Any]] = []
    suppression_help: list[dict[str, Any]] = []
    hierarchy_help: list[dict[str, Any]] = []
    for model in BATCH_MODELS:
        original = [name for name, meta in variants.items()
                    if meta.get("model") == model and meta.get("kind") == "original"]
        filtered = [name for name, meta in variants.items()
                    if meta.get("model") == model and meta.get("kind") == "current_filtered_exact"]
        if len(original) != 1 or len(filtered) != 1 or variants[filtered[0]].get("reference") != original[0]:
            raise ValidationError(f"{model} 缺少 original→current_filtered 对照")
        entry = {"source_family": "batch", "model": model, "cohorts": {}}
        for cohort in REPORT_COHORTS:
            n = cohort_counts[cohort]
            before = score_view(cohorts[cohort][original[0]], n, f"batch.{model}.{cohort}.original")
            after = score_view(cohorts[cohort][filtered[0]], n, f"batch.{model}.{cohort}.filtered")
            entry["cohorts"][cohort] = transition(before, after)
        suppression.append(entry)
        pair = canonical_pairs.get(filtered[0])
        if not isinstance(pair, Mapping) or pair.get("reference") != original[0]:
            raise ValidationError(f"{model} 缺少 current_filtered 成对统计")
        suppression_help.append({
            "source_family": "batch",
            "model": model,
            **help_harm_block(
                pair["cohorts"]["hierarchy_applicable"],
                cohort_counts["hierarchy_applicable"],
            ),
        })

        hierarchy_names = [name for name, meta in variants.items()
                           if meta.get("model") == model and meta.get("kind") == "hierarchical_rerank"]
        observed = {(variants[name].get("policy"), variants[name].get("probability_semantics"))
                    for name in hierarchy_names}
        expected = {(policy, semantic) for policy in POLICIES
                    for semantic in EXPECTED_BATCH_SEMANTICS[model]}
        if observed != expected:
            raise ValidationError(f"{model} 没有完整保留三条曲线及固定分数解释")
        for policy in POLICIES:
            for semantic in EXPECTED_BATCH_SEMANTICS[model]:
                name = next(name for name in hierarchy_names
                            if variants[name].get("policy") == policy
                            and variants[name].get("probability_semantics") == semantic)
                if variants[name].get("reference") != filtered[0]:
                    raise ValidationError(f"{model}/{policy} 的分层参照错误")
                item = {
                    "source_family": "batch",
                    "model": model,
                    "policy": policy,
                    "probability_semantics": semantic,
                    "cohorts": {},
                }
                for cohort in REPORT_COHORTS:
                    n = cohort_counts[cohort]
                    before = score_view(cohorts[cohort][filtered[0]], n, f"batch.{model}.{cohort}.filtered")
                    after = score_view(cohorts[cohort][name], n, f"batch.{model}.{cohort}.{policy}.{semantic}")
                    item["cohorts"][cohort] = transition(before, after)
                hierarchy.append(item)
                canonical_pair = canonical_pairs.get(name)
                experimental_pair = experimental_pairs.get(name)
                if not isinstance(canonical_pair, Mapping) or not isinstance(experimental_pair, Mapping):
                    raise ValidationError(f"{model}/{policy}/{semantic} 缺少帮助/伤害统计")
                hierarchy_help.append({
                    "source_family": "batch",
                    "model": model,
                    "policy": policy,
                    "probability_semantics": semantic,
                    **help_harm_block(
                        canonical_pair["cohorts"]["hierarchy_applicable"],
                        cohort_counts["hierarchy_applicable"],
                        experimental_pair["cohorts"]["hierarchy_applicable"],
                    ),
                })
    return suppression, hierarchy, suppression_help, hierarchy_help


def sequence_comparisons(
    canonical: Mapping[str, Any], paired: Mapping[str, Any],
    cohort_counts: Mapping[str, int],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    cohorts = canonical.get("cohorts")
    paired_cohorts = paired.get("cohorts")
    if not isinstance(cohorts, Mapping) or not isinstance(paired_cohorts, Mapping):
        raise ValidationError("序列模型聚合结果格式错误")
    models = set(cohorts.get("all_legal", {}))
    if models != set(SEQUENCE_MODELS):
        raise ValidationError("序列模型名单不完整")
    suppression: list[dict[str, Any]] = []
    hierarchy: list[dict[str, Any]] = []
    suppression_help: list[dict[str, Any]] = []
    hierarchy_help: list[dict[str, Any]] = []
    for model in SEQUENCE_MODELS:
        for cohort in COHORTS:
            observed = set(cohorts[cohort][model])
            expected = {"original_order", "current_filtered_exact_order", *POLICIES}
            if observed != expected:
                raise ValidationError(f"{model} 没有完整保留三条曲线")
        entry = {"source_family": "sequence", "model": model, "cohorts": {}}
        for cohort in REPORT_COHORTS:
            n = cohort_counts[cohort]
            before = score_view(
                cohorts[cohort][model]["original_order"], n,
                f"sequence.{model}.{cohort}.original",
            )
            after = score_view(
                cohorts[cohort][model]["current_filtered_exact_order"], n,
                f"sequence.{model}.{cohort}.filtered",
            )
            entry["cohorts"][cohort] = transition(before, after)
        suppression.append(entry)
        suppression_pair = paired_cohorts["hierarchy_applicable"][model][POLICIES[0]]["current_suppression"]
        suppression_help.append({
            "source_family": "sequence",
            "model": model,
            **help_harm_block(
                suppression_pair, cohort_counts["hierarchy_applicable"],
            ),
        })
        for policy in POLICIES:
            item = {
                "source_family": "sequence",
                "model": model,
                "policy": policy,
                "probability_semantics": "logits_softmax",
                "cohorts": {},
            }
            for cohort in REPORT_COHORTS:
                n = cohort_counts[cohort]
                before = score_view(
                    cohorts[cohort][model]["current_filtered_exact_order"], n,
                    f"sequence.{model}.{cohort}.filtered",
                )
                after = score_view(
                    cohorts[cohort][model][policy], n,
                    f"sequence.{model}.{cohort}.{policy}",
                )
                item["cohorts"][cohort] = transition(before, after)
            hierarchy.append(item)
            pair = paired_cohorts["hierarchy_applicable"][model][policy]["parent_aggregation"]
            hierarchy_help.append({
                "source_family": "sequence",
                "model": model,
                "policy": policy,
                "probability_semantics": "logits_softmax",
                **help_harm_block(pair, cohort_counts["hierarchy_applicable"]),
            })
    return suppression, hierarchy, suppression_help, hierarchy_help


def pct(value: float) -> str:
    return f"{100.0 * value:.2f}%"


def arrow(metric: Mapping[str, float]) -> str:
    delta = 100.0 * metric["delta"]
    sign = "+" if delta > 0 else ""
    return f"{pct(metric['before'])}→{pct(metric['after'])} ({sign}{delta:.2f}点)"


def family_name(value: str) -> str:
    return "批量" if value == "batch" else "序列"


def policy_name(value: str) -> str:
    return {
        "product0.6": "普通父级逐级×0.6；首页固定0.6",
        "product0.8": "普通父级逐级×0.8；首页固定0.6",
        "saturating0.8": "普通父级逐级×0.8，最低0.4；首页固定0.6",
    }[value]


def semantic_name(value: str) -> str:
    return {"logits_softmax": "模型分数归一", "rank_proxy": "只按名次"}[value]


def metric_row(prefix: list[str], item: Mapping[str, Any], cohort: str) -> str:
    score = item["cohorts"][cohort]
    values = [
        arrow(score["accepted"][f"top{k}"]) for k in (1, 2, 4)
    ] + [
        arrow(score["exact"][f"top{k}"]) for k in (1, 2, 4)
    ] + [arrow(score["rank_score"])]
    return "| " + " | ".join([*prefix, cohort, *values]) + " |"


def compact_help(block: Mapping[str, Any]) -> str:
    return "；".join(
        f"T{k} {block[f'top{k}']['helped']}/{block[f'top{k}']['hurt']}"
        for k in (1, 2, 4)
    )


def one_help(value: Mapping[str, Any] | None) -> str:
    if value is None:
        return "—"
    return f"{value['helped']}/{value['hurt']}"


def build_report(summary: Mapping[str, Any]) -> str:
    applicable_n = int(summary["hierarchy_applicable_help_harm"]["n"])
    validation = summary["validation"]
    suppression = summary["comparisons"]["original_to_current_filtered"]
    hierarchy = summary["comparisons"]["current_filtered_to_hierarchy"]
    help_harm = summary["hierarchy_applicable_help_harm"]
    lines = [
        "# 批量与神经序列模型的分层回退汇总",
        "",
        "本报告只汇总批量模型和神经序列模型两轮实验已经写出的聚合 JSON。无情景统计与二阶路径计数另有独立报告。这里没有读取或写出目标编号、URL、本地目标路径、逐条预测或候选清单。",
        "",
        "## 核对结果",
        "",
        f"- 固定切分：训练 {validation['shared_inputs']['train_members']} 条，测试 {validation['shared_inputs']['counts']['test_members']} 条。",
        f"- 全部合法样本 {validation['shared_inputs']['counts']['all_legal']} 条；预测前已有依据 {validation['shared_inputs']['counts']['prior_supported']} 条；有已知可打开父级或等价服务首页 {validation['shared_inputs']['counts']['hierarchy_applicable']} 条（路线拆分见来源审计）。",
        "- 两轮实验的快照散列、切分成员核对码、统一评分规则和分层回退实现全部一致。",
        "- 两个来源目录中的文件散列全部通过。",
        "",
        "三条曲线只改变普通父页面/父目录：逐级×0.6、逐级×0.8、逐级×0.8且最低保留0.4；服务首页始终固定为0.6。这里全部保留，不按这次测试结果挑一条作为部署方案。",
        "",
        "## 第一步：original → current_filtered",
        "",
        "这一步屏蔽当前地点、当前所属应用与当前首页的等价记录，并把非当前的等价首页按模型顺序合成一个后补足名额。每格是“之前→之后（百分点变化）”。",
        "",
        "| 模型类 | 模型 | 样本组 | 接受T1 | 接受T2 | 接受T4 | 精确T1 | 精确T2 | 精确T4 | 综合位置分 |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for item in suppression:
        for cohort in REPORT_COHORTS:
            lines.append(metric_row(
                [family_name(item["source_family"]), item["model"]], item, cohort,
            ))
    lines += [
        "",
        "## 第二步：current_filtered → hierarchy",
        "",
        "这一步在当前语义动作已屏蔽、等价首页已合并的结果上，让已确认的父页面或父目录与具体位置争夺推荐位。三条曲线都列出。",
        "",
        "| 模型类 | 模型 | 曲线 | 分数用法 | 样本组 | 接受T1 | 接受T2 | 接受T4 | 精确T1 | 精确T2 | 精确T4 | 综合位置分 |",
        "|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for item in hierarchy:
        prefix = [
            family_name(item["source_family"]), item["model"],
            policy_name(item["policy"]), semantic_name(item["probability_semantics"]),
        ]
        for cohort in REPORT_COHORTS:
            lines.append(metric_row(prefix, item, cohort))
    lines += [
        "",
        f"## 有父级或等价首页的 {applicable_n} 条样本：帮助/伤害",
        "",
        f"下表的“帮助/伤害”是成对比较的条数。这个 {applicable_n} 条样本组是看到真实下一步后做的分析组，只能解释变化，不能当成线上提前知道的保证。",
        "",
        "### original → current_filtered",
        "",
        "| 模型类 | 模型 | 接受T1/T2/T4 帮助/伤害 | 精确T1/T2/T4 帮助/伤害 | 综合位置分 帮助/伤害 |",
        "|---|---|---|---|---:|",
    ]
    for item in help_harm["original_to_current_filtered"]:
        lines.append(
            f"| {family_name(item['source_family'])} | {item['model']} | "
            f"{compact_help(item['canonical_accepted'])} | "
            f"{compact_help(item['canonical_exact'])} | "
            f"{one_help(item.get('canonical_rank_score'))} |"
        )
    lines += [
        "",
        "### current_filtered → hierarchy",
        "",
        "“上级试验T4”和“上级试验综合分”会给已确认的父页面/父目录部分分；它们不冒充精确命中率。",
        "",
        "| 模型类 | 模型 | 曲线 | 分数用法 | 接受T1/T2/T4 帮助/伤害 | 精确T1/T2/T4 帮助/伤害 | 统一综合分 帮助/伤害 | 上级试验T4 帮助/伤害 | 上级试验综合分 帮助/伤害 |",
        "|---|---|---|---|---|---|---:|---:|---:|",
    ]
    for item in help_harm["current_filtered_to_hierarchy"]:
        experimental = item.get("experimental_multilevel", {})
        lines.append(
            f"| {family_name(item['source_family'])} | {item['model']} | "
            f"{policy_name(item['policy'])} | {semantic_name(item['probability_semantics'])} | "
            f"{compact_help(item['canonical_accepted'])} | "
            f"{compact_help(item['canonical_exact'])} | "
            f"{one_help(item.get('canonical_rank_score'))} | "
            f"{one_help(experimental.get('top4'))} | "
            f"{one_help(experimental.get('rank_score'))} |"
        )
    lines += [
        "",
        "## 使用边界",
        "",
        "- “接受率”把合法服务首页视为一次接受命中；首页的0.6部分收益只进入统一综合位置分、行动收益与损失，精确率另列。普通父页面或父目录的试验分也单独列出。",
        f"- `hierarchy_applicable` 只有{applicable_n}条，任何帮助或伤害都应同时看条数，不能只看百分比。",
        "- `rank_proxy` 只按名次分配权重；它用于检查结果是否依赖模型分数大小，不代表可靠概率。",
        "- 本汇总没有选部署策略。这个固定测试切分已经用于多轮查错，结果只能作同分母探索；选定策略后仍需新的时间后移数据。",
        "",
    ]
    return "\n".join(lines)


def write_private_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    path.chmod(0o600)


def write_private_text(path: Path, value: str) -> None:
    path.write_text(value, encoding="utf-8")
    path.chmod(0o600)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-run", type=Path, required=True)
    parser.add_argument("--sequence-run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    try:
        batch_dir = require_private_source(args.batch_run, "--batch-run")
        sequence_dir = require_private_source(args.sequence_run, "--sequence-run")
        if batch_dir == sequence_dir:
            raise ValidationError("批量模型和序列模型不能指向同一目录")
        output = args.output.expanduser().resolve()
        private_root = PRIVATE_ROOT.expanduser().resolve()
        if output == private_root or not output.is_relative_to(private_root):
            raise ValidationError("--output 必须是 SmartApp training 下的新目录")
        if output.exists():
            raise ValidationError("--output 必须不存在")

        sources = require_complete_sources(batch_dir, sequence_dir)
        batch_artifacts = verify_artifacts(batch_dir, (
            "canonical-metrics.json", "experimental-multilevel-metrics.json",
            "paired-help-harm.json", "input-audit.json", "cohort-audit.json",
        ))
        sequence_artifacts = verify_artifacts(sequence_dir, (
            "canonical-metrics.json", "experimental-multilevel-metrics.json",
            "paired-help-harm.json", "input-audit.json", "hierarchy-audit.json",
            "summary.json",
        ))
        if sources["batch_run"].get("artifact_hashes_sha256") != batch_artifacts["manifest_sha256"]:
            raise ValidationError("批量模型的来源清单自身散列不一致")

        batch_input = read_json(batch_dir / "input-audit.json")
        sequence_input = read_json(sequence_dir / "input-audit.json")
        batch_canonical = read_json(batch_dir / "canonical-metrics.json")
        sequence_canonical = read_json(sequence_dir / "canonical-metrics.json")
        batch_experimental = read_json(batch_dir / "experimental-multilevel-metrics.json")
        sequence_experimental = read_json(sequence_dir / "experimental-multilevel-metrics.json")
        batch_paired = read_json(batch_dir / "paired-help-harm.json")
        sequence_paired = read_json(sequence_dir / "paired-help-harm.json")

        shared_inputs = validate_shared_inputs(
            sources["batch_run"], sources["sequence_run"], sources["sequence_summary"],
            batch_input, sequence_input, batch_canonical, sequence_canonical,
        )
        shared_inputs["train_members"] = 598
        cohort_counts = shared_inputs["counts"]
        contract = validate_contract(
            sources["batch_run"], sources["sequence_run"],
            batch_canonical, sequence_canonical,
        )
        policies = normalized_policy_definitions(batch_experimental, sequence_experimental)

        batch_parts = batch_comparisons(batch_canonical, batch_paired, cohort_counts)
        sequence_parts = sequence_comparisons(sequence_canonical, sequence_paired, cohort_counts)
        suppression = [*batch_parts[0], *sequence_parts[0]]
        hierarchy = [*batch_parts[1], *sequence_parts[1]]
        suppression_help = [*batch_parts[2], *sequence_parts[2]]
        hierarchy_help = [*batch_parts[3], *sequence_parts[3]]

        summary = {
            "version": VERSION,
            "complete": True,
            "validation": {
                "source_runs_complete": True,
                "source_artifact_hashes_verified": True,
                "batch_verified_artifact_n": batch_artifacts["verified_files"],
                "sequence_verified_artifact_n": sequence_artifacts["verified_files"],
                "shared_inputs": shared_inputs,
                "shared_contract": contract,
                "source_run_sha256": {
                    "batch": sha256(batch_dir / "run.json"),
                    "sequence": sha256(sequence_dir / "run.json"),
                },
                "source_artifact_manifest_sha256": {
                    "batch": batch_artifacts["manifest_sha256"],
                    "sequence": sequence_artifacts["manifest_sha256"],
                },
            },
            "fixed_policies": policies,
            "policy_selection": {
                "all_three_preserved": True,
                "selected_for_deployment": None,
                "test_metrics_used_to_select_policy": False,
            },
            "comparisons": {
                "original_to_current_filtered": suppression,
                "current_filtered_to_hierarchy": hierarchy,
            },
            "hierarchy_applicable_help_harm": {
                "n": cohort_counts["hierarchy_applicable"],
                "posthoc_diagnostic_only": True,
                "original_to_current_filtered": suppression_help,
                "current_filtered_to_hierarchy": hierarchy_help,
            },
            "persistence": {
                "aggregate_json_only": True,
                "target_identifiers_written": False,
                "urls_written": False,
                "local_target_paths_written": False,
                "raw_predictions_written": False,
                "candidate_lists_written": False,
            },
        }
        report = build_report(summary)

        os.umask(0o077)
        output.mkdir(mode=0o700)
        output.chmod(0o700)
        write_private_json(output / "summary.json", summary)
        write_private_text(output / "report-private.md", report)
        output_artifacts = {
            "report-private.md": sha256(output / "report-private.md"),
            "summary.json": sha256(output / "summary.json"),
        }
        write_private_json(output / "artifact-hashes.json", output_artifacts)
        write_private_json(output / "run.json", {
            "version": VERSION,
            "complete": True,
            "artifact_hashes_sha256": sha256(output / "artifact-hashes.json"),
            "summarizer_source_sha256": sha256(Path(__file__).resolve()),
            "source_runs_complete": True,
            "source_artifact_hashes_verified": True,
            "same_snapshot_and_fixed_split": True,
            "same_scoring_contract": True,
            "counts": dict(cohort_counts),
            "all_three_fixed_policies_preserved": True,
            "test_metrics_used_to_select_policy": False,
            "target_identifiers_urls_paths_predictions_or_candidates_written": False,
        })
        expected_files = {
            "summary.json", "report-private.md", "artifact-hashes.json", "run.json",
        }
        if {path.name for path in output.iterdir()} != expected_files:
            raise AssertionError("汇总目录含有未约定的文件")
        if (output.stat().st_mode & 0o777) != 0o700:
            raise AssertionError("汇总目录权限不是 700")
        for path in output.iterdir():
            if (path.stat().st_mode & 0o777) != 0o600:
                raise AssertionError(f"{path.name} 权限不是 600")
    except (KeyError, TypeError, ValidationError, OSError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
