"""Research-only RNN/LSTM/Transformer replay on the common cutoff input.

The caller supplies training examples whose feedback matured before the fixed
test boundary, and prediction-only test cutoffs.  This runner never rebuilds
an action catalog or reads a test outcome.  Every model family receives the
same shared action order, and one training arrival may have several useful
path actions through the normalized set-likelihood objective.
"""

from __future__ import annotations

import collections
import math
from typing import Any, Callable, Iterable, Mapping

import numpy as np

import run_offline_sequence_shadow as sequence
import shared_model_adapters as adapters
import shared_model_input as shared


DEFAULT_SEED = 23
MIN_ARRIVALS = 2
FAMILIES = frozenset({"rnn", "lstm", "transformer"})
MODES = frozenset({"target_only", "temporal", "hierarchy", "full"})


def _fixed_config(family: str, mode: str, epochs: int,
                  supplied: dict[str, Any] | None) -> dict[str, Any]:
    if family not in FAMILIES or mode not in MODES or epochs < 1:
        raise ValueError("Choose a declared sequence family, input mode and positive epoch count")
    selected = sequence.model_configurations(
        DEFAULT_SEED, sequence.DEFAULT_SEQUENCE_LENGTH, sequence.TOKEN_BUCKETS)[family][0]
    config = {**selected, **(supplied or {})}
    config["epochs"] = epochs
    config["input_mode"] = mode
    config["use_common_context"] = mode in {"temporal", "full"}
    config["common_context_width"] = len(shared.COMMON_CONTEXT_COLUMNS)
    if int(config["sequence_length"]) < MIN_ARRIVALS:
        raise ValueError("A sequence window must retain the current and one prior arrival")
    if int(config["bucket_count"]) < 2:
        raise ValueError("Sequence token bucket count is invalid")
    return config


def _adapt(cutoff: shared.CutoffInput, family: str, mode: str,
           config: dict[str, Any]) -> adapters.PredictionInput:
    return adapters.adapt(
        cutoff, family=family, mode=mode,
        bucket=lambda key: sequence.action_bucket(key, int(config["bucket_count"])),
        max_sequence_steps=int(config["sequence_length"]),
    )


def _model_scores(model: Any, prediction: adapters.PredictionInput) -> np.ndarray:
    """Read exactly the arrays forwarded by the declared prediction adapter."""
    torch, _, _ = sequence.require_torch()
    payload = prediction.payload

    def tensor(name: str, dtype: Any, *, batch: bool = False) -> Any:
        value = payload.get(name)
        if value is None:
            return None
        result = torch.tensor(value, dtype=dtype)
        return result.unsqueeze(0) if batch else result

    model.eval()
    with torch.no_grad():
        scores = model.raw_scores(
            tensor("sequence_ids", torch.long, batch=True),
            tensor("candidate_ids", torch.long),
            sequence_numeric=tensor("sequence_numeric", torch.float32, batch=True),
            sequence_layers=tensor("sequence_layers", torch.long, batch=True),
            candidate_numeric=tensor("candidate_numeric", torch.float32),
            candidate_layers=tensor("candidate_layers", torch.long),
            context_numeric=tensor("context_numeric", torch.float32, batch=True),
        ).detach().cpu().numpy().astype(float, copy=False)
    if scores.shape != (len(prediction.action_ids),) or not np.isfinite(scores).all():
        raise ValueError("Sequence scores do not match the shared candidate actions")
    return scores


def _prior(cutoff: shared.CutoffInput,
           prior_for_cutoff: Callable[[shared.CutoffInput], Mapping[str, float]] | None
           ) -> np.ndarray:
    if prior_for_cutoff is None:
        return np.zeros(len(cutoff.action_ids), dtype=np.float32)
    keyed = prior_for_cutoff(cutoff)
    if set(keyed) != set(cutoff.action_ids):
        raise ValueError("Prior scores must cover exactly the shared prediction-time actions")
    values = np.asarray([keyed[action] for action in cutoff.action_ids], dtype=np.float32)
    if not np.isfinite(values).all():
        raise ValueError("Prior scores must be finite")
    return values


def run_shared_sequence(train_cases: Iterable[Any], test_cases: Iterable[Any], *,
                        family: str, mode: str = "full", epochs: int = 12,
                        config: dict[str, Any] | None = None,
                        prior_for_cutoff: Callable[[shared.CutoffInput], Mapping[str, float]] | None = None
                        ) -> tuple[dict[int, dict[str, list[str]]], dict[str, Any]]:
    """Fit once on matured training behavior; predict the untouched fixed test.

    `train_cases` have position/cutoff/outcome/weight.  `test_cases` have only
    position/cutoff.  A caller may pass an explicitly preselected config; the
    default is the first predeclared architecture, with no held-out search.
    Without a prior, scores come from the neural model alone.  An optional
    cutoff-only, action-keyed prior produces a residual study; both fitting
    and inference then use that same shared action order.  Neither kind of
    score is an uncalibrated posterior chance.
    """
    config_used = _fixed_config(family, mode, epochs, config)
    train = tuple(train_cases)
    test = tuple(test_cases)
    if not test:
        raise ValueError("A fixed prediction cohort is required")
    if len({case.position for case in test}) != len(test):
        raise ValueError("Test positions must be unique")
    first_test_cutoff = min(case.cutoff.cutoff_event_index for case in test)
    audit: collections.Counter[str] = collections.Counter()
    examples: list[sequence.SequenceExample] = []
    seen_outcomes: set[str] = set()
    for case in train:
        outcome = case.outcome
        if outcome.outcome_event_id in seen_outcomes:
            raise ValueError("One confirmed arrival cannot train a sequence model twice")
        seen_outcomes.add(outcome.outcome_event_id)
        if not case.cutoff.cutoff_event_index < outcome.completed_event_index < first_test_cutoff:
            raise ValueError("Training feedback crosses the fixed test boundary")
        weight = float(case.weight)
        if not math.isfinite(weight) or not 0 <= weight <= 1:
            raise ValueError("Invalid derived training weight")
        if weight == 0:
            audit["zero_weight_training_skipped"] += 1
            continue
        if len(case.cutoff.sequence_steps) < MIN_ARRIVALS:
            audit["short_training_sequence"] += 1
            continue
        # Only a matured feedback view may append a newly confirmed exact
        # leaf.  Its original cutoff candidate prefix remains unchanged, and
        # this extended view is never used for test prediction.
        attached = adapters.adapt_feedback_training(
            case.cutoff, outcome, family=family, mode=mode,
            bucket=lambda key: sequence.action_bucket(key, int(config_used["bucket_count"])),
            max_sequence_steps=int(config_used["sequence_length"]),
            learning_event_index=first_test_cutoff,
        )
        prediction = attached.prediction
        if not attached.labels.useful_indices:
            audit["no_useful_asof_training_action"] += 1
            continue
        original_count = prediction.original_candidate_count
        if original_count != len(case.cutoff.action_ids):
            raise ValueError("Feedback training changed the original cutoff action prefix")
        prior = _prior(case.cutoff, prior_for_cutoff)
        if len(prediction.action_ids) == original_count + 1:
            if original_count not in attached.labels.useful_indices:
                raise ValueError("A feedback-only exact leaf must be positive")
            prior = np.append(prior, np.float32(0.0))
            audit["feedback_only_exact_leaf_appended"] += 1
        elif len(prediction.action_ids) != original_count:
            raise ValueError("Feedback training added more than one action")
        examples.append(adapters.to_sequence_example(attached, prior, weight=weight))
        audit["trained_original_arrivals"] += 1
        if (attached.labels.exact_index is None
                or attached.labels.exact_index >= original_count):
            audit["trained_without_original_exact"] += 1
            if any(index < original_count for index in attached.labels.useful_indices):
                audit["trained_via_parent_without_exact"] += 1
        if original_count == 0:
            audit["no_original_contrast_action"] += 1
        if len(attached.labels.useful_indices) > 1:
            audit["arrivals_with_multiple_useful_actions"] += 1

    model = training = None
    if examples:
        model, training = sequence.train_sequence_model(family, config_used, examples)
    outputs: dict[int, dict[str, list[str]]] = {}
    for case in test:
        cutoff = case.cutoff
        if not cutoff.action_ids:
            ranked: list[str] = []
            audit["no_prediction_actions"] += 1
        else:
            prior = _prior(cutoff, prior_for_cutoff)
            if len(cutoff.sequence_steps) < MIN_ARRIVALS or model is None:
                scores = prior
                audit["prediction_prior_only"] += 1
            else:
                prediction = _adapt(cutoff, family, mode, config_used)
                scores = prior + _model_scores(model, prediction)
                audit["model_scored_predictions"] += 1
            order = sorted(range(len(cutoff.action_ids)),
                           key=lambda index: (-scores[index], cutoff.action_ids[index]))
            ranked = [cutoff.action_ids[index] for index in order]
        outputs[int(case.position)] = {"predictions": ranked[:4], "rankable": ranked}

    manifest = adapters.manifest_for(family, mode)
    return outputs, {
        "status": "trained" if model is not None else "no_eligible_training_examples",
        "family": family, "input_mode": mode,
        "schema_version": shared.SCHEMA_VERSION,
        "forwarded_fields": sorted(manifest.forwarded_fields),
        "intentionally_unused_fields": sorted(manifest.intentionally_unused_fields),
        "config": {key: value for key, value in config_used.items()
                   if isinstance(value, (str, int, float, bool))},
        "config_selection": "supplied_training_only" if config is not None else "first_predeclared_no_test_selection",
        "prior_source": "caller_cutoff_action_scores" if prior_for_cutoff is not None else "zero",
        "feedback_only_exact_leaf_prior": "zero_when_appended",
        "test_used_for_selection": False,
        "test_truth_accessed": False,
        "train_cases": len(train), "unique_training_arrivals": len(seen_outcomes),
        "test_cases": len(test), "first_test_cutoff_event_index": first_test_cutoff,
        "counts": dict(sorted(audit.items())),
        "training": training,
    }
