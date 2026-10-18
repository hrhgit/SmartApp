"""Research-only replay of shared SmartApp inputs for non-sequence models.

All three families receive the same prediction-time actions and one matured
arrival with all useful path actions.  A useful parent is an alternative
action for that arrival, never a second arrival or a numeric reward.  Existing
sealed studies and the running recommender remain untouched.
"""

from __future__ import annotations

from collections import Counter
import hashlib
import math
from typing import Any, Iterable

import numpy as np

import shared_model_adapters as adapters
import shared_model_input as shared
import shared_online_path_mlp as online_path


def _ordered_cases(cases: Iterable[Any]) -> list[Any]:
    return sorted(cases, key=lambda case: (case.cutoff.cutoff_event_index, case.position))


def _rank(action_ids: tuple[str, ...], scores: dict[str, float]) -> dict[str, list[str]]:
    if set(scores) != set(action_ids) or any(not math.isfinite(value) for value in scores.values()):
        raise ValueError("A model must score every shared cutoff action")
    ranked = sorted(action_ids, key=lambda action: (-scores[action], action))
    return {"predictions": ranked[:4], "rankable": ranked}


def _attach(case: Any, family: str, gate: adapters.MaturedFeedbackGate,
            learning_index: int) -> adapters.TrainingExample:
    return adapters.adapt_feedback_training(
        case.cutoff, case.outcome, family=family,
        learning_event_index=learning_index, gate=gate)


class SharedPathStatistics:
    """Simple path-frequency reference with one event per confirmed arrival.

    Conditional and global counts learn only matured training outcomes.  The
    prediction-time path/direct counts describe already observed foreground
    arrivals, and a small specificity factor lets a nearly equally frequent
    child beat its parent.  These are ranking counts, not calibrated chances.
    """

    def __init__(self) -> None:
        self.conditional: Counter[tuple[str, str]] = Counter()
        self.global_actions: Counter[str] = Counter()
        self.origin_events: Counter[str] = Counter()
        self.event_count = 0.0

    def learn(self, example: adapters.TrainingExample, weight: float) -> bool:
        if not 0.0 < weight <= 1.0:
            raise ValueError("Invalid independent-arrival weight")
        prediction = example.prediction
        if (not prediction.after_feedback_only
                or prediction.original_candidate_count is None):
            raise ValueError("Statistics needs one matured common training example")
        if not example.labels.useful_indices:
            return False
        # The common adapter owns both the old useful parents and a newly
        # confirmed exact leaf.  One result increments each alternative once.
        actions = {prediction.action_ids[index]
                   for index in example.labels.useful_indices}
        origin = prediction.cutoff.current_action_id
        for action in actions:
            self.conditional[origin, action] += weight
            self.global_actions[action] += weight
        self.origin_events[origin] += weight
        self.event_count += weight
        return True

    def scores(self, prediction: adapters.PredictionInput) -> dict[str, float]:
        origin = prediction.cutoff.current_action_id
        local_total = self.origin_events[origin]
        global_total = max(self.event_count, 1.0)
        score: dict[str, float] = {}
        for candidate in prediction.cutoff.candidates:
            action = candidate.action_id
            global_rate = self.global_actions[action] / global_total
            conditional_rate = ((self.conditional[origin, action] + 3.0 * global_rate)
                                / (local_total + 3.0))
            observed_path = candidate.path_counts[-1]
            observed_direct = candidate.direct_counts[-1]
            # A parent remains competitive when visits spread among children;
            # a child wins clearly when it explains almost every parent visit.
            specificity = 0.65 + 0.35 * (observed_direct / max(observed_path, 1))
            score[action] = conditional_rate * specificity
        return score


def run_path_statistics(train_cases: Iterable[Any], test_cases: Iterable[Any],
                        *, online_feedback: Iterable[Any] = ()
                        ) -> tuple[dict[int, dict[str, list[str]]], dict[str, Any]]:
    """Predict with a shared-input path-frequency reference.

    Non-empty online_feedback enables a separate progressive replay.  Feedback
    enters only after its completed event, never at its prediction cutoff.
    """
    tests = _ordered_cases(test_cases)
    feedback = sorted(online_feedback, key=lambda case: case.outcome.completed_event_index)
    first_index = tests[0].cutoff.cutoff_event_index if tests else 2**63 - 1
    model = SharedPathStatistics()
    gate = adapters.MaturedFeedbackGate()
    audit: Counter[str] = Counter()

    def learn(case: Any, learning_index: int, weight: float, scope: str) -> None:
        outcome = case.outcome
        example = _attach(case, "statistical", gate, learning_index)
        audit[scope + "_feedback_seen"] += 1
        if not model.learn(example, weight):
            audit[scope + "_no_eligible_training_action"] += 1
            return
        audit[scope + "_unique_arrivals"] += 1
        audit[scope + "_exact_known_at_input"] += outcome.exact_candidate_id is not None
        audit[scope + "_new_exact_after_feedback"] += outcome.exact_candidate_id is None
        audit[scope + "_parent_only_at_input"] += bool(outcome.useful_action_ids and
                                                       outcome.exact_candidate_id is None)

    for case in sorted(train_cases, key=lambda item: item.outcome.completed_event_index):
        if case.outcome.completed_event_index >= first_index:
            raise ValueError("Batch training received feedback from the test period")
        learn(case, first_index, float(case.weight), "train")
    outputs: dict[int, dict[str, list[str]]] = {}
    cursor = 0
    for case in tests:
        index = case.cutoff.cutoff_event_index
        while cursor < len(feedback) and feedback[cursor].outcome.completed_event_index < index:
            ready = feedback[cursor]
            cursor += 1
            learn(ready, index, 1.0, "mature_test")
        if not case.cutoff.action_ids:
            outputs[case.position] = {"predictions": [], "rankable": []}
            audit["queries_without_asof_actions"] += 1
        else:
            prediction = adapters.adapt(case.cutoff, family="statistical")
            outputs[case.position] = _rank(prediction.action_ids, model.scores(prediction))
        audit["queries"] += 1
    audit["test_feedback_not_yet_mature"] = len(feedback) - cursor
    return outputs, dict(sorted(audit.items()))


def run_shared_online_mlp(train_cases: Iterable[Any], test_cases: Iterable[Any],
                          test_feedback: Iterable[Any], *, hidden: int = 16,
                          learning_rate: float = 0.001,
                          negatives: int = 8
                          ) -> tuple[dict[int, dict[str, list[str]]], dict[str, Any]]:
    """One incremental set-loss update per matured arrival and its path."""
    tests = _ordered_cases(test_cases)
    feedback = sorted(test_feedback, key=lambda case: case.outcome.completed_event_index)
    first_index = tests[0].cutoff.cutoff_event_index if tests else 2**63 - 1
    model = online_path.SharedOnlineSetMLP(hidden, learning_rate,
                                          negatives=negatives, seed=20260929)
    gate = adapters.MaturedFeedbackGate()
    audit: Counter[str] = Counter()

    def learn(case: Any, learning_index: int, weight: float) -> None:
        example = _attach(case, "candidate_mlp", gate, learning_index)
        if not example.labels.useful_indices:
            audit["mature_no_eligible_training_action"] += 1
            return
        update = model.learn(example.prediction, example.labels.useful_indices,
                             weight=weight)
        audit["mature_unique_arrivals"] += 1
        audit["optimizer_updates"] += bool(update["updated"])
        audit["new_exact_after_feedback"] += case.outcome.exact_candidate_id is None
        audit["parent_only_at_input"] += bool(case.outcome.useful_action_ids and
                                              case.outcome.exact_candidate_id is None)
        audit["no_asof_actions"] += not bool(case.cutoff.action_ids)
        if not update["updated"]:
            audit["no_optimizer_step_" + update["reason"]] += 1

    for case in sorted(train_cases, key=lambda item: item.outcome.completed_event_index):
        if case.outcome.completed_event_index >= first_index:
            raise ValueError("Batch training received feedback from the test period")
        learn(case, first_index, float(case.weight))
    outputs: dict[int, dict[str, list[str]]] = {}
    cursor = 0
    for case in tests:
        index = case.cutoff.cutoff_event_index
        while cursor < len(feedback) and feedback[cursor].outcome.completed_event_index < index:
            learn(feedback[cursor], index, 1.0)
            audit["mature_test_feedback"] += 1
            cursor += 1
        if not case.cutoff.action_ids:
            outputs[case.position] = {"predictions": [], "rankable": []}
            audit["queries_without_asof_actions"] += 1
        else:
            prediction = adapters.adapt(case.cutoff, family="candidate_mlp")
            scores = model.scores(prediction)
            outputs[case.position] = _rank(prediction.action_ids, scores)
        audit["queries"] += 1
    audit["test_feedback_not_yet_mature"] = len(feedback) - cursor
    return outputs, {**dict(sorted(audit.items())), "set_loss_model": model.audit()}


def _sample_training_indices(example: adapters.TrainingExample, max_actions: int
                             ) -> tuple[int, ...]:
    positives = set(example.labels.useful_indices)
    if not positives:
        return ()
    rows = example.prediction.payload["rows"]
    if len(rows) <= max_actions:
        return tuple(range(len(rows)))
    if len(positives) >= max_actions:
        return tuple(sorted(positives))
    # Keep the strongest already-observed alternatives, then deterministic
    # diverse negatives.  Never truncate an as-of useful parent.
    negatives = [index for index in range(len(rows)) if index not in positives]
    negatives.sort(key=lambda index: (
        -float(rows[index]["path_count_all_log"]),
        hashlib.sha256(example.prediction.action_ids[index].encode()).hexdigest(),
    ))
    selected = positives | set(negatives[:max_actions - len(positives)])
    return tuple(sorted(selected))


def run_shared_catboost(train_cases: Iterable[Any], test_cases: Iterable[Any],
                        *, config: dict[str, Any], max_training_actions: int = 64
                        ) -> tuple[dict[int, dict[str, list[str]]], dict[str, Any]]:
    """Train a frozen batch ranker with the common path-positive labels."""
    from catboost import CatBoostRanker, Pool
    import pandas as pd

    tests = _ordered_cases(test_cases)
    if max_training_actions < 2:
        raise ValueError("CatBoost needs room for a useful and an alternative action")
    first_index = tests[0].cutoff.cutoff_event_index if tests else 2**63 - 1
    gate = adapters.MaturedFeedbackGate()
    audit: Counter[str] = Counter()
    records: list[dict[str, Any]] = []
    labels: list[int] = []
    group_ids: list[int] = []
    group_weights: list[float] = []
    for case in _ordered_cases(train_cases):
        if case.outcome.completed_event_index >= first_index:
            raise ValueError("Batch training received feedback from the test period")
        example = _attach(case, "catboost", gate, first_index)
        audit["train_unique_arrivals"] += 1
        audit["train_new_exact_after_feedback"] += case.outcome.exact_candidate_id is None
        audit["train_no_asof_actions"] += not bool(case.cutoff.action_ids)
        if not example.labels.useful_indices:
            audit["train_no_eligible_training_action"] += 1
            continue
        selected = _sample_training_indices(example, max_training_actions)
        if len(selected) < 2 or len(selected) == len(example.labels.useful_indices):
            audit["train_without_comparison"] += 1
            continue
        group_id = audit["train_groups"]
        positive = set(example.labels.useful_indices)
        # YetiRankPairwise forms comparisons from every useful path action.
        # Divide the group's weight across those alternatives so a deep path
        # does not turn one confirmed arrival into several full-weight labels.
        one_arrival_weight = float(case.weight) / len(positive)
        for index in selected:
            records.append(example.prediction.payload["rows"][index])
            labels.append(int(index in positive))
            group_ids.append(group_id)
            group_weights.append(one_arrival_weight)
        audit["train_groups"] += 1
        audit["train_parent_only_at_input"] += bool(case.outcome.useful_action_ids and
                                                     case.outcome.exact_candidate_id is None)
        audit["train_candidate_rows"] += len(selected)
    if len(set(group_ids)) < 2:
        raise ValueError("Shared CatBoost needs at least two rankable groups")
    columns = (shared.COMMON_CONTEXT_COLUMNS + shared.CANDIDATE_NUMERIC_COLUMNS
               + shared.CANDIDATE_CATEGORICAL_COLUMNS)
    frame = pd.DataFrame(records, columns=columns)
    categorical = list(shared.CANDIDATE_CATEGORICAL_COLUMNS)
    pool = Pool(frame, label=labels, group_id=group_ids,
                group_weight=group_weights, cat_features=categorical,
                feature_names=list(columns))
    model = CatBoostRanker(
        loss_function="YetiRankPairwise", eval_metric="NDCG:top=4",
        iterations=int(config["iterations"]), depth=int(config["depth"]),
        learning_rate=float(config["learning_rate"]),
        l2_leaf_reg=float(config["l2_leaf_reg"]),
        random_seed=int(config.get("seed", 20260929)), thread_count=1,
        allow_writing_files=False, verbose=False,
    )
    model.fit(pool)
    outputs: dict[int, dict[str, list[str]]] = {}
    for case in tests:
        if not case.cutoff.action_ids:
            outputs[case.position] = {"predictions": [], "rankable": []}
            audit["queries_without_asof_actions"] += 1
            audit["queries"] += 1
            continue
        prediction = adapters.adapt(case.cutoff, family="catboost")
        query = pd.DataFrame(prediction.payload["rows"], columns=columns)
        values = np.asarray(model.predict(query), dtype=float)
        if values.shape != (len(prediction.action_ids),):
            raise ValueError("CatBoost changed shared action order or count")
        outputs[case.position] = _rank(
            prediction.action_ids,
            dict(zip(prediction.action_ids, (float(value) for value in values))),
        )
        audit["queries"] += 1
    audit["test_feedback_learned"] = 0
    return outputs, {
        **dict(sorted(audit.items())),
        "group_path_weight_rule": "one arrival weight divided by useful action count",
        "training_objective": "pairwise path relevance, not calibrated set probability",
    }
