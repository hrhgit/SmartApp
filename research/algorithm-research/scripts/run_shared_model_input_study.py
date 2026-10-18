#!/usr/bin/env python3
"""Replay one fixed SmartApp split through the shared, causal model input.

This is research-only.  It neither changes the collector nor the running
recommender, and it never writes candidate IDs, URLs, paths or per-row model
outputs.  The original 598/149 membership and scorer remain unchanged.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import cleaned_foreground_data as cleaning
import evaluation_contract as evaluation
import hierarchical_fallback_policy as hierarchy_policy
import online_model_suite as online_suite
import parallel_destination_data as parallel_data
import prepare_offline_batch_split as split
import run_break_cleaning_study as old_study
import run_hierarchical_statistical_experiment as hierarchy_report
import run_offline_batch_rankers as offline
import run_parallel_destination_experiment as parallel_experiment
import sequence_asof_features as asof
import shared_model_input as shared
import shared_model_adapters as adapters


VERSION = "shared-model-input-study-v1"
EXPECTED_MEMBERS = {"all_legal": 139, "route_supported": 129,
                    "pretest_route_supported": 120}
EMPTY_AFTER_SUPERVISION = hierarchy_policy.Hierarchy({}, {}, (), _skip_validation=True)


def _model_cutoff(cutoff: shared.CutoffInput) -> shared.CutoffInput:
    """Release the full private graph after deriving labels and candidates."""
    return replace(cutoff, hierarchy=EMPTY_AFTER_SUPERVISION)


@dataclass(frozen=True)
class SharedTrainCase:
    position: int
    cutoff: shared.CutoffInput
    outcome: shared.MaturedOutcome
    weight: float


@dataclass(frozen=True)
class SharedPredictionCase:
    position: int
    cutoff: shared.CutoffInput


@dataclass(frozen=True)
class SharedFeedbackCase:
    position: int
    cutoff: shared.CutoffInput
    outcome: shared.MaturedOutcome


@dataclass(frozen=True)
class SharedBatch:
    train: tuple[SharedTrainCase, ...]
    test: tuple[SharedPredictionCase, ...]
    test_feedback: tuple[SharedFeedbackCase, ...]
    audit: Mapping[str, int]


def feedback_exclusion_reason(cutoff: shared.CutoffInput,
                              outcome: shared.MaturedOutcome) -> str | None:
    """Reject a non-switch or a raw ID that now names another concrete place.

    Both cases remain in the original fixed snapshot and test scoring.  They
    cannot be represented as an unambiguous new learning action under the
    shared action-ID contract.
    """
    destination = outcome.confirmed_destination
    if destination.semantic_key == cutoff.current_semantic_key:
        return "same_semantic_not_switch"
    for candidate in cutoff.candidates:
        if (candidate.action_id == destination.action_id
                and (candidate.semantic_key != destination.semantic_key
                     or candidate.kind != destination.kind
                     or candidate.owner_app_id != destination.owner_app_id)):
            return "reused_id_action_conflict"
    return None


def build_batch(train: Sequence[offline.Observation], test: Sequence[offline.Observation],
                builder: shared.SharedInputBuilder, *, boundary_index: int,
                weights: Mapping[str, float]) -> SharedBatch:
    """Materialize one versioned input per cutoff and a separate feedback list."""
    train_cases: list[SharedTrainCase] = []
    test_cases: list[SharedPredictionCase] = []
    feedback: list[SharedFeedbackCase] = []
    train_ledger, test_ledger = shared.OutcomeLedger(), shared.OutcomeLedger()
    counts: dict[str, int] = {}

    def add(name: str, amount: int = 1) -> None:
        counts[name] = counts.get(name, 0) + amount

    for row in sorted(train, key=lambda item: (item.input_event_index, item.position)):
        if not old_study._usable(row):
            add("train_not_navigation")
            continue
        if row.completed_event_index >= boundary_index:
            raise ValueError("Frozen training feedback crossed the fixed test boundary")
        cutoff = builder.at(row)
        outcome = builder.matured(cutoff, row, learning_event_index=boundary_index)
        if not train_ledger.admit(outcome):
            add("train_duplicate_confirmed_arrival")
            continue
        add("train_matured_navigation")
        excluded = feedback_exclusion_reason(cutoff, outcome)
        if excluded is not None:
            add("train_excluded_" + excluded)
            continue
        weight = float(weights.get(row.sample_id, 1.0))
        if not 0 < weight <= 1:
            raise ValueError("Invalid shared training weight")
        train_cases.append(SharedTrainCase(row.position, _model_cutoff(cutoff), outcome, weight))
        add("train_cases")
        add("train_exact_action_available", int(outcome.exact_candidate_id is not None))
        add("train_ancestor_only", int(outcome.exact_candidate_id is None
                                       and bool(outcome.useful_action_ids)))
        add("train_no_useful_action", int(not outcome.useful_action_ids))
        add("train_no_useful_then_confirmed_exact", int(
            not outcome.useful_action_ids
            and outcome.confirmed_destination.action_id == outcome.exact_action_id))
        if not outcome.useful_action_ids:
            eligible_after_feedback = adapters.confirmed_destination_is_eligible(outcome)
            add("train_no_useful_after_feedback_eligible", int(eligible_after_feedback))
            add("train_no_useful_after_feedback_ineligible", int(not eligible_after_feedback))
        add("train_inferred_half_weight", int(weight < 1))
    if (len(train_cases)
            + counts.get("train_excluded_same_semantic_not_switch", 0)
            + counts.get("train_excluded_reused_id_action_conflict", 0)
            != train_ledger.unique_arrivals):
        raise AssertionError("Training arrival count differs from outcome ledger")

    # `test_cases` deliberately has no truth/feedback field.  An online model
    # receives the separate list and must check maturity before each query.
    for test_number, row in enumerate(sorted(
            test, key=lambda item: (item.input_event_index, item.position))):
        cutoff = builder.at(row)
        add("test_cutoffs")
        if test_number == 0:
            add("first_test_recorded_actions", sum(
                item.source == "recorded_action" for item in cutoff.candidates))
            add("first_test_derived_parents", sum(
                item.source == "derived_from_known_locator" for item in cutoff.candidates))
        add("test_has_derived_parent", int(any(candidate.source == "derived_from_known_locator"
                                               for candidate in cutoff.candidates)))
        if not old_study._usable(row):
            test_cases.append(SharedPredictionCase(row.position, _model_cutoff(cutoff)))
            add("test_feedback_not_navigation")
            continue
        outcome = builder.matured(cutoff, row, learning_event_index=len(builder.events))
        model_cutoff = _model_cutoff(cutoff)
        test_cases.append(SharedPredictionCase(row.position, model_cutoff))
        if not test_ledger.admit(outcome):
            add("test_duplicate_confirmed_arrival")
            continue
        add("test_matured_navigation")
        add("test_exact_action_available", int(outcome.exact_candidate_id is not None))
        add("test_ancestor_only", int(outcome.exact_candidate_id is None
                                      and bool(outcome.useful_action_ids)))
        add("test_no_useful_action", int(not outcome.useful_action_ids))
        add("test_no_useful_then_confirmed_exact", int(
            not outcome.useful_action_ids
            and outcome.confirmed_destination.action_id == outcome.exact_action_id))
        if not outcome.useful_action_ids:
            eligible_after_feedback = adapters.confirmed_destination_is_eligible(outcome)
            add("test_no_useful_after_feedback_eligible", int(eligible_after_feedback))
            add("test_no_useful_after_feedback_ineligible", int(not eligible_after_feedback))
        excluded = feedback_exclusion_reason(cutoff, outcome)
        if excluded is not None:
            add("test_feedback_excluded_" + excluded)
            continue
        feedback.append(SharedFeedbackCase(row.position, model_cutoff, outcome))
        add("test_feedback_cases")
    if len(test_cases) != len(test):
        raise AssertionError("A fixed test prediction member was lost")
    if len({case.position for case in test_cases}) != len(test):
        raise AssertionError("Test prediction positions are not unique")
    return SharedBatch(tuple(train_cases), tuple(test_cases), tuple(feedback),
                       dict(sorted(counts.items())))


def checked_outputs(cases: Sequence[SharedPredictionCase],
                    outputs: Mapping[int, Mapping[str, Any]]) -> dict[int, dict[str, Any]]:
    """Require every shared model to score exactly the same as-of actions."""
    expected = {case.position for case in cases}
    if set(outputs) != expected:
        raise ValueError("Model omitted or invented fixed test positions")
    result: dict[int, dict[str, Any]] = {}
    for case in cases:
        output = outputs[case.position]
        predictions = list(output.get("predictions", ()))
        rankable = list(output.get("rankable", predictions))
        allowed = set(case.cutoff.action_ids)
        if (len(predictions) > 4 or len(predictions) != len(set(predictions))
                or len(rankable) != len(set(rankable))
                or len(rankable) != len(allowed) or set(rankable) != allowed
                or predictions != rankable[:4]):
            raise ValueError("Model output departed from the common cutoff candidate set")
        result[case.position] = {"predictions": predictions, "rankable": rankable}
    return result


def fixed_cohorts(test: Sequence[offline.Observation], history: offline.TargetHistory,
                  boundary_index: int) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    audited = old_study.evaluation_rows(list(test), history)
    contexts = {row.position: hierarchy_report.build_context(old_study._prepared(row), history)
                for row in test}
    cohorts, audit = hierarchy_report.build_cohorts(audited, contexts, 139)
    prior = {row["position"] for row in cohorts["prior_supported"]}
    route = {row["position"] for row in cohorts["hierarchy_applicable"]}
    cohorts["route_supported"] = [row for row in cohorts["all_legal"]
                                  if row["position"] in prior | route]
    cohorts["pretest_route_supported"] = parallel_experiment.frozen_training_route_rows(
        cohorts["route_supported"], contexts, history, boundary_index,
        parallel_experiment.POLICIES[0])
    for name, expected in EXPECTED_MEMBERS.items():
        if len(cohorts[name]) != expected:
            raise ValueError(f"Fixed {name} membership changed: {len(cohorts[name])}")
        audit["cohorts"][name] = {
            "n": expected, "membership_sha256": hierarchy_report.membership_hash(cohorts[name]),
        }
    return cohorts, audit


def score_all(cohorts: Mapping[str, list[dict[str, Any]]],
              outputs: Mapping[str, Mapping[int, dict[str, Any]]]) -> dict[str, Any]:
    return old_study.score_boards(cohorts, outputs)


def paired_against_shared(rows: Sequence[dict[str, Any]],
                          outputs: Mapping[str, Mapping[int, dict[str, Any]]]
                          ) -> dict[str, Any]:
    """Aggregate paired gains on one fixed cohort, without saving row identities."""
    reference_name = "shared_path_statistical"
    reference = outputs[reference_name]
    valid = [row for row in rows if evaluation.eligible(row)]
    result: dict[str, Any] = {}
    for name, alternative in outputs.items():
        if name == reference_name or not name.startswith("shared_"):
            continue
        pair = evaluation.paired_topk(valid, reference, alternative)
        before = [evaluation.ranked_gain(
            row, evaluation.prediction_at(reference, row["position"]))["score"]
            for row in valid]
        after = [evaluation.ranked_gain(
            row, evaluation.prediction_at(alternative, row["position"]))["score"]
            for row in valid]
        pair["rank_score"] = hierarchy_report.paired_values(before, after)
        pair["reference_model"] = reference_name
        result[name] = pair
    return result


def _render_report(metrics: Mapping[str, Any], counts: Mapping[str, int],
                   model_audits: Mapping[str, Any], cohort_audit: Mapping[str, Any],
                   paired: Mapping[str, Any]) -> str:
    main = metrics["pretest_route_supported"]
    lines = ["# 共用输入层离线研究", "",
             "固定原 598/149 成员；清洗采用 both（已观察第一步与严格短断点），评分仍由原统一入口完成。",
             f"清洗后有 {counts.get('train_matured_navigation', 0)} 条已确认训练目标变化；"
             f"其中同一语义位置 {counts.get('train_excluded_same_semantic_not_switch', 0)} 条、"
             f"同一目标 ID 换位置或类别 {counts.get('train_excluded_reused_id_action_conflict', 0)} 条"
             "无法按共用动作含义学习，"
             f"故共用模型训练使用 {counts.get('train_cases', 0)} 条。原固定 598/149 成员和测试评分分母不变。",
             f"共用训练的 {counts.get('train_cases', 0)} 条：预测时具体入口已可用 "
             f"{counts.get('train_exact_action_available', 0)} 条、只有可得分上级 "
             f"{counts.get('train_ancestor_only', 0)} 条、两者都没有 "
             f"{counts.get('train_no_useful_action', 0)} 条。最后一类在反馈后有 "
             f"{counts.get('train_no_useful_after_feedback_eligible', 0)} 条可成为以后学习的动作，"
             f"{counts.get('train_no_useful_after_feedback_ineligible', 0)} 条仍不合格；"
             "均不倒计为原预测覆盖。",
             f"测试中确认 {counts.get('test_matured_navigation', 0)} 条后续目标，"
             f"其中 {counts.get('test_feedback_cases', 0)} 条能供在线学习：预测时具体入口已可用 "
             f"{counts.get('test_exact_action_available', 0)} 条、只有可得分上级 "
             f"{counts.get('test_ancestor_only', 0)} 条、两者都没有 "
             f"{counts.get('test_no_useful_action', 0)} 条。最后一类在反馈后有 "
             f"{counts.get('test_no_useful_after_feedback_eligible', 0)} 条可供后续学习、"
             f"{counts.get('test_no_useful_after_feedback_ineligible', 0)} 条仍不合格；"
             "均不计回原预测覆盖。",
             f"首个测试预测截点有此前台已确认入口 "
             f"{counts.get('first_test_recorded_actions', 0)} 个、从其已知网址或路径临时推导的上级 "
             f"{counts.get('first_test_derived_parents', 0)} 个。前者也包括同次到达的补全确认，"
             "不全是独立到达；仅保存目标信息但从未前台确认的入口不参与候选。",
             "下面的 TopK 是可用具体目标或合法上级进入前 K 位的次数；完全一致和服务接受另列。", "",
             "| 模型 | 成功 Top1/Top2/Top4 | 完全一致 Top1/Top2/Top4 | 服务接受 Top1/Top2/Top4 | 综合分 |", "|---|---:|---:|---:|---:|"]
    for name, item in main.items():
        triple = lambda part: "/".join(str(part[f"top{k}"]) for k in (1, 2, 4))
        lines.append(f"| {name} | {triple(item['success'])} | {triple(item['exact'])} | "
                     f"{triple(item['service'])} | {item['rank_score']:.3f} |")
    lines += ["", "在同一 120 条主榜上，与共享路径统计逐条相比（提升/退步）：", "",
              "| 模型 | 成功 Top4 提升/退步 | 综合分提升/退步 | 综合分平均变化 |",
              "|---|---:|---:|---:|"]
    for name, item in paired.items():
        top4 = item["success"]["top4"]
        ranked = item["rank_score"]
        lines.append(f"| {name} | {top4['helped']}/{top4['hurt']} | "
                     f"{ranked['helped']}/{ranked['hurt']} | {ranked['mean_delta']:+.3f} |")
    lines += ["", "父级部分收益与输出覆盖率见 `fixed-metrics.json`；榜单同时保留全部 139 条合法样本。",
              "训练与测试使用同一版预测输入和候选；模型实际读取的字段以各自适配器声明与模型审计为准，"
              "输入层提供字段不代表模型已学会使用它。测试反馈仅供在线模型在真实形成后更新。",
              "共享路径统计实际读取当前目标、候选入口、全部历史的路径/直接次数与已成熟的转移计数；"
              "它不读取父级身份、短期次数、时间、停留、观察质量或近期顺序。"
              "在线 MLP 读取共用时间/停留/质量、各档次数、目标类别及哈希后的层级身份；"
              "CatBoost 读取同组表格字段；完整序列模型读取已确认到达序列、时间/质量数值、"
              "候选次数及层级身份，但不读取表格专用的来源类别等字段。",
              "这里的路径集合训练主要检验上级或具体入口能否进入 TopK，未按约定的各级部分分值训练最优取舍，"
              "也未证明模型能把精确入口与上级入口的分数校准成可靠概率。",
              "在线模型和序列模型按一次真实到达的可得分入口集合学习；CatBoost 的候选行彼此相关，"
              "其原始分数不能直接解释为校准后的概率。旧无情景统计参考使用不同的历史候选范围，"
              "且训练成员未做本次四条语义冲突排除，仅看 TopK 不能据此断定架构优劣。",
              "此报告不保存目标 ID、完整网址/路径或逐条推荐名单。", "",
              "## 范围核对", "",
              f"主要榜单 {EXPECTED_MEMBERS['pretest_route_supported']} 条；可用路径榜单 "
              f"{EXPECTED_MEMBERS['route_supported']} 条；全部合法榜单 "
              f"{EXPECTED_MEMBERS['all_legal']} 条。", "",
              "模型、清洗、候选和评分代码哈希记录在 `run.json`。", ""]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-selection", type=Path, required=True)
    parser.add_argument("--sequence-epochs", type=int, default=12)
    parser.add_argument("--skip-sequence", action="store_true")
    args = parser.parse_args()
    snapshot, split_dir, output = (item.expanduser().resolve()
                                   for item in (args.snapshot, args.split, args.output))
    if (output.exists() or output == offline.PRIVATE_ROOT.resolve()
            or not output.is_relative_to(offline.PRIVATE_ROOT.resolve())):
        parser.error("Output must be a new private SmartApp training directory")
    if args.sequence_epochs < 1:
        parser.error("Sequence epochs must be positive")

    import shared_tabular_runner as tabular
    import shared_online_path_mlp as online_path_mlp
    import run_offline_sequence_shadow as sequence_core
    sequences = None
    if not args.skip_sequence:
        import shared_sequence_runner as sequences
    source_files = [__file__, shared.__file__, cleaning.__file__, old_study.__file__,
                    offline.__file__, evaluation.__file__, asof.__file__,
                    hierarchy_policy.__file__, online_suite.__file__, parallel_data.__file__,
                    hierarchy_report.__file__, parallel_experiment.__file__, split.__file__,
                    adapters.__file__, tabular.__file__, online_path_mlp.__file__,
                    sequence_core.__file__,
                    Path(__file__).resolve().parents[3] / "AGENTS.md"]
    if sequences is not None:
        source_files.append(sequences.__file__)
    starting_source_hashes = {Path(path).name: offline.sha256(Path(path))
                              for path in source_files}

    train, test, history, load_audit = offline.load_observations(snapshot, split_dir)
    if len(train) != 598 or len(test) != 149:
        raise ValueError("The fixed 598/149 split changed")
    split_report = offline.read_json(split_dir / "split-report.json")
    boundary = int(split_report["boundary"]["eventIndex"])
    events = offline.read_jsonl(snapshot / "events.jsonl")
    samples = offline.read_jsonl(snapshot / "navigation-samples.jsonl")
    derived = cleaning.derive_cleaning(events, samples, history,
                                        test_boundary_index=boundary,
                                        existing_train_rows=train,
                                        existing_test_rows=test)
    cleaned, weights, cleaning_audit = old_study.training_arm(
        train, derived, "both", boundary_index=boundary)
    index = asof.build_rich_index(events, history, 16, derived.bridges)
    batch = build_batch(cleaned, test, shared.SharedInputBuilder(events, history, index),
                        boundary_index=boundary, weights=weights)
    print(json.dumps({"stage": "shared_batch", **batch.audit}, sort_keys=True), flush=True)

    selected, selection_audit = old_study.prior_batch.load_selected_configs(
        args.batch_selection.expanduser().resolve(), snapshot, split_dir)
    cat_config = selected["catboost"]["config"]
    outputs: dict[str, dict[int, dict[str, Any]]] = {}
    model_audits: dict[str, Any] = {}
    # Historical reference is frozen and deliberately uses its original
    # action scope.  All new models below use the common shared input.
    old_output, old_audit = old_study.replay_statistics(cleaned, test, history, weights)
    outputs["previous_frozen_statistical"] = old_output
    model_audits["previous_frozen_statistical"] = old_audit
    model_calls = [
        ("shared_path_statistical", lambda: tabular.run_path_statistics(batch.train, batch.test)),
        ("shared_path_statistical_online", lambda: tabular.run_path_statistics(
            batch.train, batch.test, online_feedback=batch.test_feedback)),
        ("shared_online_mlp", lambda: tabular.run_shared_online_mlp(
            batch.train, batch.test, batch.test_feedback, hidden=16, learning_rate=0.001)),
        ("shared_catboost", lambda: tabular.run_shared_catboost(
            batch.train, batch.test, config=cat_config)),
    ]
    if not args.skip_sequence:
        assert sequences is not None
        model_calls.extend((
            f"shared_{family}",
            lambda family=family: sequences.run_shared_sequence(
                batch.train, batch.test, family=family, mode="full",
                epochs=args.sequence_epochs))
            for family in ("rnn", "lstm", "transformer"))
    for name, call in model_calls:
        result, audit = call()
        outputs[name] = checked_outputs(batch.test, result)
        model_audits[name] = audit
        print(json.dumps({"stage": "model_complete", "model": name}, sort_keys=True), flush=True)

    cohorts, cohort_audit = fixed_cohorts(test, history, boundary)
    metrics = score_all(cohorts, outputs)
    paired = paired_against_shared(cohorts["pretest_route_supported"], outputs)
    after, _ = split.snapshot_hashes(snapshot)
    if after != load_audit["snapshot_hashes"]:
        raise ValueError("Raw snapshot changed during the research run")
    final_source_hashes = {Path(path).name: offline.sha256(Path(path))
                           for path in source_files}
    if final_source_hashes != starting_source_hashes:
        raise ValueError("Research source or AGENTS.md changed during the run")
    output.mkdir(mode=0o700, parents=False)
    os.chmod(output, 0o700)
    offline.write_private_json(output / "fixed-metrics.json", metrics)
    offline.write_private_json(output / "paired-shared-aggregate.json", paired)
    offline.write_private_json(output / "model-audits.json", model_audits)
    offline.write_private_json(output / "shared-batch-audit.json", {
        "counts": batch.audit, "cleaning": cleaning_audit,
        "rich_index": index.audit, "cohorts": cohort_audit,
        "shared_schema_version": shared.SCHEMA_VERSION,
        "train_candidate_rows_are_transient": True,
        "test_feedback_separate_from_prediction_inputs": True,
    })
    offline.write_private_text(output / "report-private.md",
                               _render_report(metrics, batch.audit, model_audits,
                                              cohort_audit, paired))
    manifest = {path.name: offline.sha256(path) for path in output.iterdir()
                if path.is_file() and path.name != "artifact-hashes.json"}
    offline.write_private_json(output / "artifact-hashes.json", manifest)
    run = {
        "version": VERSION, "complete": True, "snapshot": str(snapshot),
        "split": str(split_dir), "snapshot_hashes": load_audit["snapshot_hashes"],
        "fixed_train_members": 598, "fixed_test_members": 149,
        "fixed_cohorts": EXPECTED_MEMBERS,
        "scoring_contract": evaluation.contract(),
        "source_hashes": starting_source_hashes,
        "train_only_configuration_source": selection_audit,
        "shared_input_schema": shared.SCHEMA_VERSION,
        "test_used_for_selection": False,
        "raw_snapshot_modified": False, "running_model_modified": False,
        "candidate_lists_persisted": False,
        "artifact_manifest_sha256": offline.sha256(output / "artifact-hashes.json"),
    }
    offline.write_private_json(output / "run.json", run)
    print(json.dumps({"stage": "complete", "output": str(output)}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
