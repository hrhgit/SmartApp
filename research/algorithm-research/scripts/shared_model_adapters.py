"""Declared, causal projections of one SmartApp research sample to model families.

The shared input builder owns action eligibility, order, hierarchy and history.
An adapter may change a *representation*, but must never rebuild those facts.
PredictionInput cannot contain the future outcome; MaturedFeedbackGate attaches
one separately, only after the feedback event has completed.

These projections are in-memory research inputs.  They do not persist a menu or
change the running recommender.  ``useful`` is a Boolean path relation, not a
numeric reward or a second arrival.
"""

from __future__ import annotations

from dataclasses import dataclass, fields, replace
import math
from typing import Any, Callable, Mapping

import numpy as np

import run_offline_batch_rankers as offline
import sequence_asof_features as asof
import shared_model_input as shared


# Intentional, reviewed mirrors of the common schema.  In particular, do not
# replace these with ``shared.COMMON_CONTEXT_COLUMNS``: then a new common field
# could silently become an unused model input without an explicit decision.
EXPECTED_CUTOFF_FIELDS = frozenset({
    "schema_version", "sample_id", "cutoff_event_index", "cutoff_timestamp_ms",
    "current_action_id", "current_kind", "current_owner_app_id",
    "current_semantic_key", "current_layers", "candidates", "context",
    "sequence_steps", "sequence_numeric", "hierarchy",
})
EXPECTED_CANDIDATE_FIELDS = frozenset({
    "action_id", "kind", "owner_app_id", "semantic_key", "layers",
    "path_counts", "direct_counts", "source", "metadata",
})
EXPECTED_OUTCOME_FIELDS = frozenset({
    "schema_version", "sample_id", "outcome_event_id", "completed_event_index",
    "exact_action_id", "candidate_paths", "exact_candidate_id",
    "confirmed_destination",
})
EXPECTED_STEP_FIELDS = frozenset({
    "key", "event_index", "timestamp_ms", "monotonic_ms", "layers",
    "source_quality", "marker_kind", "marker_duration_ms",
    "marker_same_exact", "marker_active_elapsed_ms",
    "marker_observed_pre_gap_ms", "marker_begin_index", "marker_end_index",
    "source_id", "source_status",
})
EXPECTED_CONTEXT_COLUMNS = (
    "time_hour_sin", "time_hour_cos", "time_weekday_sin", "time_weekday_cos",
    "last_confirmed_dwell_log_s", "has_confirmed_dwell",
    "quality_ready_fraction", "quality_syncing_fraction",
    "quality_unavailable_fraction", "quality_failed_fraction",
    "quality_stale_fraction", "quality_unknown_fraction", "quality_confirmed",
    "has_same_target_inferred_active", "same_target_inferred_active_log_s",
    "has_legacy_short_gap", "has_excluded_scope_gap", "known_gap_log_s",
    "known_gap_duration", "rich_input_missing",
)
EXPECTED_CANDIDATE_NUMERIC_COLUMNS = (
    "hierarchy_layer_count_log", "is_derived_parent", "same_owner_app",
    "path_count_5m_log", "path_count_1h_log", "path_count_24h_log",
    "path_count_all_log", "direct_count_5m_log", "direct_count_1h_log",
    "direct_count_24h_log", "direct_count_all_log",
)
EXPECTED_CANDIDATE_CATEGORICAL_COLUMNS = (
    "origin_token", "candidate_token", "origin_candidate_token",
    "origin_kind", "candidate_kind", "origin_bundle_token",
    "candidate_bundle_token", "candidate_source",
    "candidate_parent_token_1", "candidate_parent_token_2",
    "candidate_parent_token_3", "candidate_parent_token_4",
    "current_parent_token_1", "current_parent_token_2",
)

STATISTICAL_VIEW_FIELDS = frozenset({
    "schema_version", "origin", "current_layers", "action_ids",
    "candidate_layers", "path_counts", "direct_counts",
})
MLP_VIEW_FIELDS = frozenset({
    "schema_version", "origin", "labels", "x", "candidate_features",
})
SEQUENCE_VIEW_FIELDS = frozenset({
    "schema_version", "action_ids", "context_numeric", "sequence_ids", "sequence_numeric",
    "sequence_layers", "candidate_ids", "candidate_numeric", "candidate_layers",
})
SEQUENCE_VIEW_ORDER = (
    "schema_version", "action_ids", "context_numeric", "sequence_ids",
    "sequence_numeric", "sequence_layers", "candidate_ids",
    "candidate_numeric", "candidate_layers",
)


@dataclass(frozen=True)
class AdapterManifest:
    """Exactly which common view fields a model projection passes through.

    This describes the adapter boundary, not a claim that an existing model
    architecture has learned a useful weight for every forwarded feature.
    """

    family: str
    input_mode: str
    view: str
    forwarded_fields: frozenset[str]
    intentionally_unused_fields: frozenset[str]


def _manifest(family: str, mode: str, view: str,
              universe: frozenset[str], forwarded: frozenset[str],
              ignored: frozenset[str]) -> AdapterManifest:
    if forwarded & ignored or forwarded | ignored != universe:
        raise ValueError(f"Incomplete adapter field decision: {family}/{mode}")
    return AdapterManifest(family, mode, view, forwarded, ignored)


# Each projection explicitly accounts for every key in the common view.  The
# sequence ablations do not quietly pick up extra timing or hierarchy arrays.
REGISTRY: Mapping[tuple[str, str], AdapterManifest] = {
    ("statistical", "full"): _manifest(
        "statistical", "full", "statistical", STATISTICAL_VIEW_FIELDS,
        STATISTICAL_VIEW_FIELDS, frozenset()),
    ("candidate_mlp", "full"): _manifest(
        "candidate_mlp", "full", "mlp", MLP_VIEW_FIELDS,
        MLP_VIEW_FIELDS, frozenset()),
    ("catboost", "full"): _manifest(
        "catboost", "full", "catboost", frozenset({"rows", "action_ids", "context"}),
        frozenset({"rows", "action_ids", "context"}), frozenset()),
    **{
        (family, mode): _manifest(
            family, mode, "sequence", SEQUENCE_VIEW_FIELDS,
            frozenset({"schema_version", "action_ids", "sequence_ids", "candidate_ids"})
            | (frozenset({"context_numeric", "sequence_numeric"})
               if mode in {"temporal", "full"} else frozenset())
            | (frozenset({"sequence_layers", "candidate_numeric", "candidate_layers"})
               if mode in {"hierarchy", "full"} else frozenset()),
            (frozenset({"context_numeric", "sequence_numeric"})
             if mode in {"target_only", "hierarchy"} else frozenset())
            | (frozenset({"sequence_layers", "candidate_numeric", "candidate_layers"})
               if mode in {"target_only", "temporal"} else frozenset()),
        )
        for family in ("rnn", "lstm", "transformer")
        for mode in ("target_only", "temporal", "hierarchy", "full")
    },
}


def validate_schema() -> None:
    """Fail closed when a shared field is added without adapter review."""
    expected = (
        ("CutoffInput", frozenset(item.name for item in fields(shared.CutoffInput)), EXPECTED_CUTOFF_FIELDS),
        ("CandidateInput", frozenset(item.name for item in fields(shared.CandidateInput)), EXPECTED_CANDIDATE_FIELDS),
        ("MaturedOutcome", frozenset(item.name for item in fields(shared.MaturedOutcome)), EXPECTED_OUTCOME_FIELDS),
        ("Step", frozenset(item.name for item in fields(asof.Step)), EXPECTED_STEP_FIELDS),
        ("context columns", tuple(shared.COMMON_CONTEXT_COLUMNS), EXPECTED_CONTEXT_COLUMNS),
        ("candidate numeric columns", tuple(shared.CANDIDATE_NUMERIC_COLUMNS), EXPECTED_CANDIDATE_NUMERIC_COLUMNS),
        ("candidate categorical columns", tuple(shared.CANDIDATE_CATEGORICAL_COLUMNS), EXPECTED_CANDIDATE_CATEGORICAL_COLUMNS),
    )
    for name, observed, declared in expected:
        if observed != declared:
            raise ValueError(f"{name} changed; review every model adapter's consume/ignore declaration")
    for manifest in REGISTRY.values():
        if manifest.forwarded_fields & manifest.intentionally_unused_fields:
            raise ValueError(f"Overlapping fields in {manifest.family}/{manifest.input_mode}")
        universe = (STATISTICAL_VIEW_FIELDS if manifest.view == "statistical" else
                    MLP_VIEW_FIELDS if manifest.view == "mlp" else
                    SEQUENCE_VIEW_FIELDS if manifest.view == "sequence" else
                    frozenset({"rows", "action_ids", "context"}))
        if manifest.forwarded_fields | manifest.intentionally_unused_fields != universe:
            raise ValueError(f"Unaccounted fields in {manifest.family}/{manifest.input_mode}")


@dataclass(frozen=True)
class PredictionInput:
    """The same pre-outcome action order and context for one model family."""

    manifest: AdapterManifest
    schema_version: str
    sample_id: str
    cutoff_event_index: int
    action_ids: tuple[str, ...]
    common_context: Mapping[str, float]
    payload: Mapping[str, Any]
    cutoff: shared.CutoffInput
    # Only a separate, post-feedback training view may extend the cutoff.
    after_feedback_only: bool = False
    original_candidate_count: int | None = None


@dataclass(frozen=True)
class LabelRelation:
    """One matured arrival, with exact and useful-path labels kept separate."""

    outcome_event_id: str
    completed_event_index: int
    exact_action_id: str
    exact_index: int | None
    exact: tuple[bool, ...]
    useful: tuple[bool, ...]
    typed_paths: tuple[tuple[str, ...] | None, ...]

    @property
    def useful_indices(self) -> tuple[int, ...]:
        return tuple(index for index, matched in enumerate(self.useful) if matched)


@dataclass(frozen=True)
class TrainingExample:
    prediction: PredictionInput
    labels: LabelRelation


class FeedbackTrainingUnavailable(ValueError):
    """A known but unusable feedback case, with a stable audit reason."""

    def __init__(self, reason: str, message: str) -> None:
        self.reason = reason
        super().__init__(message)


def manifest_for(family: str, mode: str = "full") -> AdapterManifest:
    validate_schema()
    try:
        manifest = REGISTRY[family, mode]
    except KeyError as exc:
        raise ValueError(f"Undeclared model projection: {family}/{mode}") from exc
    # The shared contract also checks the actual consumed/ignored common field
    # groups; this function is defined by shared_model_input, not by a model.
    shared.validate_adapter_field_usage(family)
    return manifest


def adapt(cutoff: shared.CutoffInput, *, family: str, mode: str = "full",
          bucket: Callable[[str], int] | None = None,
          max_sequence_steps: int | None = None) -> PredictionInput:
    """Project only the cutoff state.  There is no outcome argument by design."""
    manifest = manifest_for(family, mode)
    if cutoff.schema_version != shared.SCHEMA_VERSION:
        raise ValueError("Unrecognized shared input version")
    if len(cutoff.action_ids) != len(set(cutoff.action_ids)):
        raise ValueError("Duplicate shared action catalog")
    if set(cutoff.context) != set(EXPECTED_CONTEXT_COLUMNS):
        raise ValueError("Common context fields changed")
    if any(not math.isfinite(float(value)) for value in cutoff.context.values()):
        raise ValueError("Non-finite common context")

    if manifest.view == "statistical":
        view = cutoff.statistical_view()
        _assert_view(view, STATISTICAL_VIEW_FIELDS)
        payload = view
    elif manifest.view == "mlp":
        view = cutoff.mlp_view()
        _assert_view(view, MLP_VIEW_FIELDS)
        payload = view
    elif manifest.view == "catboost":
        rows = cutoff.catboost_rows()
        expected = (EXPECTED_CONTEXT_COLUMNS + EXPECTED_CANDIDATE_NUMERIC_COLUMNS
                    + EXPECTED_CANDIDATE_CATEGORICAL_COLUMNS)
        if any(tuple(row) != expected for row in rows):
            raise ValueError("CatBoost row columns changed or were reordered")
        payload = {"rows": rows, "action_ids": cutoff.action_ids, "context": dict(cutoff.context)}
    else:
        if bucket is None:
            raise ValueError("Sequence adapter needs its declared token bucket")
        view = cutoff.sequence_view(bucket)
        _assert_view(view, SEQUENCE_VIEW_FIELDS)
        payload = {name: view[name] for name in SEQUENCE_VIEW_ORDER
                   if name in manifest.forwarded_fields}
        if max_sequence_steps is not None:
            if max_sequence_steps < 1:
                raise ValueError("max_sequence_steps must be positive")
            for name in ("sequence_ids", "sequence_numeric", "sequence_layers"):
                if name in payload:
                    payload[name] = payload[name][-max_sequence_steps:]
        if len(payload["sequence_ids"]) == 0:
            raise ValueError("Sequence has no confirmed current arrival")

    if _payload_actions(manifest, payload) != cutoff.action_ids:
        raise ValueError("Model adapter changed shared action order")
    return PredictionInput(manifest, cutoff.schema_version, cutoff.sample_id,
                           cutoff.cutoff_event_index, cutoff.action_ids,
                           {name: float(cutoff.context[name]) for name in EXPECTED_CONTEXT_COLUMNS},
                           payload, cutoff)


def _assert_view(view: Mapping[str, Any], expected: frozenset[str]) -> None:
    if set(view) != expected:
        raise ValueError("Shared model view changed without adapter review")


def _payload_actions(manifest: AdapterManifest, payload: Mapping[str, Any]) -> tuple[str, ...]:
    if manifest.view == "mlp":
        return tuple(payload["labels"])
    return tuple(payload["action_ids"])


def attach_matured(prediction: PredictionInput, outcome: shared.MaturedOutcome,
                   *, learning_event_index: int) -> TrainingExample:
    """Attach supervision only after outcome feedback has actually matured."""
    if prediction.after_feedback_only:
        raise ValueError("Attach the outcome to the original prediction cutoff")
    if (prediction.schema_version != outcome.schema_version
            or prediction.sample_id != outcome.sample_id):
        raise ValueError("Outcome does not belong to this prediction input")
    if not prediction.cutoff_event_index < outcome.completed_event_index < learning_event_index:
        raise ValueError("Future or not-yet-matured outcome cannot train a model")
    raw_labels = outcome.labels_in_order(prediction.cutoff)
    exact = tuple(bool(item[0]) for item in raw_labels)
    useful = tuple(bool(item[1]) for item in raw_labels)
    paths = tuple(item[2] for item in raw_labels)
    if len(raw_labels) != len(prediction.action_ids):
        raise ValueError("Outcome relation order differs from shared actions")
    exact_indices = tuple(i for i, yes in enumerate(exact) if yes)
    if len(exact_indices) > 1 or (outcome.exact_candidate_id is None) != (not exact_indices):
        raise ValueError("Exact target relation is inconsistent")
    if any(is_exact and not is_useful for is_exact, is_useful in zip(exact, useful)):
        raise ValueError("Exact destination must also be a useful action")
    if any(is_useful != (path is not None) for is_useful, path in zip(useful, paths)):
        raise ValueError("Useful-path relation is inconsistent")
    label = LabelRelation(outcome.outcome_event_id, outcome.completed_event_index,
                          outcome.exact_action_id,
                          exact_indices[0] if exact_indices else None,
                          exact, useful, paths)
    return TrainingExample(prediction, label)


class MaturedFeedbackGate:
    """Reject duplicate training arrivals across all model projections."""

    def __init__(self) -> None:
        self._seen_outcomes: set[str] = set()

    def attach_once(self, prediction: PredictionInput, outcome: shared.MaturedOutcome,
                    *, learning_event_index: int) -> TrainingExample:
        example = attach_matured(prediction, outcome,
                                 learning_event_index=learning_event_index)
        if example.labels.outcome_event_id in self._seen_outcomes:
            raise ValueError("One confirmed arrival cannot be learned twice")
        self._seen_outcomes.add(example.labels.outcome_event_id)
        return example


def confirmed_destination_is_eligible(outcome: shared.MaturedOutcome) -> bool:
    """Whether feedback may add the confirmed exact action to later learning.

    An ordinary ineligible action returns False.  A malformed or unverified
    window fallback raises, so the caller can distinguish a contract failure
    from a legitimate target that cannot become a recommendation action.
    """
    candidate = outcome.confirmed_destination
    if (candidate.source != "after_feedback_only"
            or candidate.action_id != outcome.exact_action_id
            or not candidate.layers
            or candidate.layers[0].key != candidate.semantic_key
            or candidate.path_counts != (0, 0, 0, 0)
            or candidate.direct_counts != (0, 0, 0, 0)):
        raise ValueError("Malformed after-feedback confirmed destination")
    meta = dict(candidate.metadata)
    action = offline.action_from_meta(meta, catalog_only=True)
    if action is None and candidate.kind == "app":
        admission = meta.get("admission")
        if isinstance(admission, dict) and admission.get("basis") == "window_action_fallback":
            bundle = meta.get("bundleID")
            proof = meta.get("_verified_window_fallback")
            if (not isinstance(bundle, str) or not bundle
                    or set(meta) != {"id", "kind", "bundleID", "admission",
                                     "_verified_window_fallback"}
                    or meta.get("id") != "app:" + bundle
                    or admission != {"basis": "window_action_fallback",
                                     "policyVersion": offline.VERSION}
                    or not isinstance(proof, Mapping)
                    or set(proof) != {"outcome_event_id", "bundleID", "raw_kind",
                                      "raw_admission_basis"}
                    or proof.get("outcome_event_id") != outcome.outcome_event_id
                    or proof.get("bundleID") != bundle
                    or proof.get("raw_kind") != "window"
                    or not offline.admitted({
                        "kind": "window", "bundleID": bundle,
                        "admission": {"basis": proof.get("raw_admission_basis")},
                    })):
                raise ValueError("Unverified after-feedback window action fallback")
            # The raw admitted foreground window was checked when the common
            # outcome was built.  Do not make this synthetic app globally
            # admissible as a prediction-time catalog entry.
            action = offline.Action(candidate.action_id, meta, "app", bundle)
    if action is None:
        return False
    semantic = shared.parallel.semantic_key(action.meta)
    owner = "app:" + action.bundle if action.bundle else None
    if (action.key != candidate.action_id or action.kind != candidate.kind
            or semantic != candidate.semantic_key or owner != candidate.owner_app_id):
        raise ValueError("After-feedback destination metadata disagrees with its action")
    return True


def _accepted_home_equivalent(old: shared.CandidateInput,
                              outcome: shared.MaturedOutcome) -> bool:
    """Use the common outcome's score-compatible homepage marker.

    The private hierarchy is deliberately released before batch training.
    The common builder names this score-accepted relation explicitly.  The
    candidate must also be a recognizable root page.  A non-home identical
    URL/path is not exact.
    """
    if (old.kind != "page"
            or outcome.candidate_paths.get(old.action_id) != ("equivalent_service_home",)):
        return False
    location = shared.hierarchy_policy._web_location(old.metadata)
    return location is not None and not location[1]


def _feedback_candidate(cutoff: shared.CutoffInput,
                        outcome: shared.MaturedOutcome) -> tuple[shared.CandidateInput | None,
                                                                   int | None]:
    """Return a new leaf or an equivalent old action, preserving cutoff order."""
    candidate = outcome.confirmed_destination
    if not confirmed_destination_is_eligible(outcome):
        return None, None
    if candidate.semantic_key == cutoff.current_semantic_key:
        raise FeedbackTrainingUnavailable("same_as_current",
                                          "Feedback destination is the current destination")
    for index, old in enumerate(cutoff.candidates):
        if old.action_id == candidate.action_id:
            if (old.semantic_key != candidate.semantic_key or old.kind != candidate.kind
                    or old.owner_app_id != candidate.owner_app_id):
                raise FeedbackTrainingUnavailable(
                    "reused_id_changed_locator",
                    "A reused target ID changed its concrete destination")
            return None, index
        if (old.semantic_key == candidate.semantic_key and old.kind == candidate.kind
                and old.owner_app_id == candidate.owner_app_id):
            # A different ID with the same URL/path is not generally the
            # scorer's exact action.  Only an already score-accepted homepage
            # can stand in for another recorded ID at the same home.
            if _accepted_home_equivalent(old, outcome):
                return None, index
    # The supervision object retains its explicit after-feedback source.  The
    # learning-only feature row uses the source it will have once a confirmed
    # arrival enters a later prediction-time catalog.  PredictionInput's flag
    # is the isolation boundary; the model must never score this view.
    return replace(candidate, source="recorded_action"), None


def adapt_online_feedback(cutoff: shared.CutoffInput,
                          outcome: shared.MaturedOutcome, *,
                          family: str = "candidate_mlp", mode: str = "full",
                          bucket: Callable[[str], int] | None = None,
                          max_sequence_steps: int | None = None,
                          learning_event_index: int | None = None
                          ) -> tuple[PredictionInput, tuple[int, ...]]:
    """Build an in-memory learning view after a confirmed arrival has matured.

    The returned view is forbidden for prediction.  Its prefix is exactly the
    original cutoff's action list, so only that prefix may supply negatives.
    The newly confirmed leaf, when eligible, is a positive at the final index.
    Callers must supply a real learning event index or have already gated the
    matured outcome before invoking this lower-level conversion.
    """
    validate_schema()
    if (cutoff.schema_version != outcome.schema_version
            or cutoff.sample_id != outcome.sample_id
            or outcome.completed_event_index <= cutoff.cutoff_event_index):
        raise ValueError("Outcome does not belong after this prediction cutoff")
    if (learning_event_index is not None
            and outcome.completed_event_index >= learning_event_index):
        raise ValueError("Future or not-yet-matured outcome cannot train a model")
    appended, equivalent_index = _feedback_candidate(cutoff, outcome)
    old_count = len(cutoff.action_ids)
    augmented = (replace(cutoff, candidates=(*cutoff.candidates, appended))
                 if appended is not None else cutoff)
    prediction = adapt(augmented, family=family, mode=mode, bucket=bucket,
                       max_sequence_steps=max_sequence_steps)
    if prediction.action_ids[:old_count] != cutoff.action_ids:
        raise ValueError("Feedback view changed prediction-time action order")
    old_positive = {index for index, action_id in enumerate(cutoff.action_ids)
                    if action_id in outcome.candidate_paths}
    # Preserve the fixed scorer's identity rule even if a malformed outcome
    # relation marked a same-locator, different-ID non-home as useful.
    confirmed = outcome.confirmed_destination
    for index, old in enumerate(cutoff.candidates):
        if (old.action_id != confirmed.action_id
                and old.semantic_key == confirmed.semantic_key
                and old.kind == confirmed.kind
                and old.owner_app_id == confirmed.owner_app_id
                and not _accepted_home_equivalent(old, outcome)):
            old_positive.discard(index)
    if equivalent_index is not None:
        old_positive.add(equivalent_index)
    if appended is not None:
        old_positive.add(old_count)
    prediction = replace(prediction, after_feedback_only=True,
                         original_candidate_count=old_count)
    return prediction, tuple(sorted(old_positive))


def adapt_feedback_training(cutoff: shared.CutoffInput,
                            outcome: shared.MaturedOutcome, *,
                            family: str, learning_event_index: int,
                            mode: str = "full",
                            bucket: Callable[[str], int] | None = None,
                            max_sequence_steps: int | None = None,
                            gate: MaturedFeedbackGate | None = None) -> TrainingExample:
    """Train every family on one arrival's old useful paths and new exact leaf.

    The original prediction and its exact/path labels are checked first.
    No new action enters the prediction cutoff or its negative-action prefix.
    A gate, when provided, admits the independent arrival only once.
    """
    prediction, positive_indices = adapt_online_feedback(
        cutoff, outcome, family=family, mode=mode, bucket=bucket,
        max_sequence_steps=max_sequence_steps,
        learning_event_index=learning_event_index)
    original = adapt(cutoff, family=family, mode=mode, bucket=bucket,
                     max_sequence_steps=max_sequence_steps)
    base = (gate.attach_once(original, outcome, learning_event_index=learning_event_index)
            if gate is not None else attach_matured(
                original, outcome, learning_event_index=learning_event_index))
    old_count = len(cutoff.action_ids)
    useful = list(base.labels.useful)
    paths = list(base.labels.typed_paths)
    exact = list(base.labels.exact)
    positive_set = set(positive_indices)
    for index in range(old_count):
        if index not in positive_set:
            useful[index] = False
            paths[index] = None
        elif not useful[index]:
            useful[index] = True
            paths[index] = ()
    exact_index = base.labels.exact_index
    if len(prediction.action_ids) == old_count + 1:
        exact.append(True)
        useful.append(True)
        paths.append(())
        exact_index = old_count
    label = replace(base.labels, exact_index=exact_index,
                    exact=tuple(exact), useful=tuple(useful),
                    typed_paths=tuple(paths))
    if label.useful_indices != positive_indices:
        raise ValueError("Feedback positives disagree with shared path relations")
    return TrainingExample(prediction, label)


def to_query_group(example: TrainingExample, row: offline.Observation,
                   prior: np.ndarray) -> offline.QueryGroup:
    """Bridge shared actions to QueryGroup without reconstructing its catalog.

    The old QueryGroup stores only an exact single-label index.  The caller
    must retain ``example.labels`` for useful-parent supervision; an unseen
    exact target remains ``truth_index=None`` instead of selecting a parent.
    The shared feature rows use the new schema, so old fixed CatBoost columns
    are not implicitly reused.
    """
    prediction = example.prediction
    if (row.sample_id != prediction.sample_id
            or row.input_event_index != prediction.cutoff_event_index):
        raise ValueError("QueryGroup row differs from the shared cutoff")
    prior_array = _checked_prior(prior, len(prediction.action_ids))
    actions = [offline.Action(candidate.action_id, dict(candidate.metadata),
                              candidate.kind,
                              candidate.owner_app_id[4:] if candidate.owner_app_id else None)
               for candidate in prediction.cutoff.candidates]
    features = list(prediction.cutoff.feature_rows())
    if tuple(action.key for action in actions) != prediction.action_ids:
        raise ValueError("QueryGroup action order changed")
    return offline.QueryGroup(row, actions, prior_array, features,
                              example.labels.exact_index)


def to_sequence_example(example: TrainingExample, prior: np.ndarray,
                        *, weight: float = 1.0) -> Any:
    """Bridge shared sequence arrays to the offline sequence trainer."""
    if example.prediction.manifest.view != "sequence":
        raise ValueError("Expected an RNN, LSTM or Transformer projection")
    if not 0.0 < weight <= 1.0:
        raise ValueError("Invalid training weight")
    label = example.labels
    if not label.useful_indices:
        raise ValueError("No useful candidate in the prediction-time catalog")
    payload = example.prediction.payload
    prior_array = _checked_prior(prior, len(example.prediction.action_ids))
    import run_offline_sequence_shadow as sequence

    return sequence.SequenceExample(
        sequence_ids=tuple(int(value) for value in payload["sequence_ids"]),
        candidate_ids=tuple(int(value) for value in payload["candidate_ids"]),
        prior=prior_array,
        truth_index=label.exact_index,
        sequence_numeric=payload.get("sequence_numeric"),
        sequence_layers=payload.get("sequence_layers"),
        candidate_numeric=payload.get("candidate_numeric"),
        candidate_layers=payload.get("candidate_layers"),
        context_numeric=payload.get("context_numeric"),
        weight=weight,
        positive_indices=label.useful_indices,
    )


def _checked_prior(prior: np.ndarray, count: int) -> np.ndarray:
    values = np.asarray(prior, dtype=np.float32)
    if values.shape != (count,) or not np.isfinite(values).all():
        raise ValueError("Prior must have one finite value per shared action")
    return values.copy()


def mlp_pair_inputs(prediction: PredictionInput) -> tuple[dict[str, Any], ...]:
    """Candidate-specific numeric records for the existing pairwise MLP API.

    The old MLP scores one label with one numeric feature map.  A caller can
    pass each returned query separately to ``scores`` and use the same map for
    its matured positive/negative pair update.  No outcome enters this method.
    """
    if prediction.manifest.view != "mlp":
        raise ValueError("Expected the CandidateMLP projection")
    view = prediction.payload
    records: list[dict[str, Any]] = []
    for action_id in prediction.action_ids:
        row = view["candidate_features"][action_id]
        numeric = {"context:" + key: float(value) for key, value in view["x"].items()}
        for key in EXPECTED_CANDIDATE_NUMERIC_COLUMNS:
            numeric["candidate:" + key] = float(row[key])
        for key in EXPECTED_CANDIDATE_CATEGORICAL_COLUMNS:
            numeric["candidate:" + key + "=" + str(row[key])] = 1.0
        for layer in row["layer_keys"]:
            numeric["candidate_layer:" + offline.token(layer)] = 1.0
        records.append({"origin": view["origin"], "labels": (action_id,), "x": numeric})
    return tuple(records)
