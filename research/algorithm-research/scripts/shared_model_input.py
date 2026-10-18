"""One causal, in-memory input contract for SmartApp offline model research.

The cutoff object never reads a future destination.  The outcome object is
constructed separately, after that destination is confirmed and its feedback
has matured.  All model families can adapt the same action IDs, hierarchy,
foreground sequence and numeric fields without collecting or saving a
candidate list.  A derived parent is an executable *action*, not a visit.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from datetime import datetime
import bisect
import hashlib
import math
from types import MappingProxyType
from typing import Any, Callable, Mapping
from zoneinfo import ZoneInfo

import numpy as np

import evaluation_contract as evaluation
import hierarchical_fallback_policy as hierarchy_policy
import parallel_destination_data as parallel
import run_offline_batch_rankers as offline
import sequence_asof_features as asof


SCHEMA_VERSION = "shared-model-input-v3"
WINDOW_NAMES = ("5m", "1h", "24h", "all")
EXPECTED_WINDOW_NAMES = ("5m", "1h", "24h", "all")
COMMON_CONTEXT_COLUMNS = (
    "time_hour_sin", "time_hour_cos", "time_weekday_sin", "time_weekday_cos",
    "last_confirmed_dwell_log_s", "has_confirmed_dwell",
    "quality_ready_fraction", "quality_syncing_fraction",
    "quality_unavailable_fraction", "quality_failed_fraction",
    "quality_stale_fraction", "quality_unknown_fraction", "quality_confirmed",
    "has_same_target_inferred_active", "same_target_inferred_active_log_s",
    "has_legacy_short_gap", "has_excluded_scope_gap",
    "known_gap_log_s", "known_gap_duration", "rich_input_missing",
)
CANDIDATE_NUMERIC_COLUMNS = (
    "hierarchy_layer_count_log", "is_derived_parent", "same_owner_app",
    *(f"path_count_{name}_log" for name in WINDOW_NAMES),
    *(f"direct_count_{name}_log" for name in WINDOW_NAMES),
)
CANDIDATE_CATEGORICAL_COLUMNS = (
    "origin_token", "candidate_token", "origin_candidate_token",
    "origin_kind", "candidate_kind", "origin_bundle_token",
    "candidate_bundle_token", "candidate_source",
    "candidate_parent_token_1", "candidate_parent_token_2",
    "candidate_parent_token_3", "candidate_parent_token_4",
    "current_parent_token_1", "current_parent_token_2",
)

# Every public field belongs to exactly one group.  The names below are
# intentionally enumerated: adding a field to a dataclass or a feature list
# fails validation until its consumers and intentional non-consumers are
# declared.  This is a guard against another model silently missing inputs.
FIELD_GROUPS: dict[str, frozenset[str]] = {
    "identity": frozenset({
        "cutoff.schema_version", "cutoff.sample_id", "cutoff.cutoff_event_index",
    }),
    "current": frozenset({
        "cutoff.current_action_id", "cutoff.current_layers",
    }),
    "current_metadata": frozenset({
        "cutoff.current_kind", "cutoff.current_owner_app_id",
        "cutoff.current_semantic_key",
    }),
    "candidate_scope": frozenset({
        "cutoff.candidates", "cutoff.hierarchy", "candidate.action_id",
    }),
    "candidate_metadata": frozenset({
        "candidate.kind", "candidate.owner_app_id", "candidate.source", "candidate.metadata",
    }),
    "candidate_layers": frozenset({
        "candidate.semantic_key", "candidate.layers",
    }),
    "tabular_candidate_flags": frozenset({
        "feature.hierarchy_layer_count_log", "feature.is_derived_parent",
        "feature.same_owner_app",
    }),
    "history_counts": frozenset({
        "candidate.path_counts", "candidate.direct_counts",
        *("feature.path_count_" + name + "_log" for name in WINDOW_NAMES),
        *("feature.direct_count_" + name + "_log" for name in WINDOW_NAMES),
    }),
    "time": frozenset({
        "cutoff.cutoff_timestamp_ms", "context.time_hour_sin", "context.time_hour_cos",
        "context.time_weekday_sin", "context.time_weekday_cos",
    }),
    "dwell": frozenset({
        "context.last_confirmed_dwell_log_s", "context.has_confirmed_dwell",
        "context.has_same_target_inferred_active",
        "context.same_target_inferred_active_log_s",
    }),
    "quality": frozenset({
        "cutoff.context", "context.quality_ready_fraction",
        "context.quality_syncing_fraction", "context.quality_unavailable_fraction",
        "context.quality_failed_fraction", "context.quality_stale_fraction",
        "context.quality_unknown_fraction", "context.quality_confirmed",
        "context.rich_input_missing",
    }),
    "gap": frozenset({
        "context.has_legacy_short_gap", "context.has_excluded_scope_gap",
        "context.known_gap_log_s", "context.known_gap_duration",
    }),
    "sequence": frozenset({
        "cutoff.sequence_steps", "cutoff.sequence_numeric",
    }),
    "categorical_tokens": frozenset({
        "feature.origin_token", "feature.candidate_token",
        "feature.origin_candidate_token", "feature.origin_kind",
        "feature.candidate_kind", "feature.origin_bundle_token",
        "feature.candidate_bundle_token", "feature.candidate_source",
        "feature.candidate_parent_token_1", "feature.candidate_parent_token_2",
        "feature.candidate_parent_token_3", "feature.candidate_parent_token_4",
        "feature.current_parent_token_1", "feature.current_parent_token_2",
    }),
}

_ALL_GROUPS = frozenset(FIELD_GROUPS)
ADAPTER_FIELD_USAGE: dict[str, dict[str, frozenset[str]]] = {
    "statistical": {
        "consume": frozenset({"current", "candidate_scope", "history_counts"}),
        "intentional_ignore": frozenset({"identity", "current_metadata", "candidate_metadata",
                                          "candidate_layers", "tabular_candidate_flags", "time", "dwell",
                                          "quality", "gap", "sequence", "categorical_tokens"}),
    },
    "mlp": {
        "consume": frozenset({"current", "candidate_scope", "candidate_layers",
                              "current_metadata", "candidate_metadata", "tabular_candidate_flags",
                              "history_counts", "time", "dwell", "quality", "gap",
                              "categorical_tokens"}),
        "intentional_ignore": frozenset({"identity", "sequence"}),
    },
    "catboost": {
        "consume": frozenset({"current", "candidate_scope", "candidate_layers",
                              "current_metadata", "candidate_metadata", "tabular_candidate_flags",
                              "history_counts", "time", "dwell", "quality", "gap",
                              "categorical_tokens"}),
        "intentional_ignore": frozenset({"identity", "sequence"}),
    },
    "sequence": {
        "consume": frozenset({"current", "candidate_scope", "candidate_layers",
                              "history_counts", "time", "dwell", "quality", "gap",
                              "sequence"}),
        "intentional_ignore": frozenset({"identity", "current_metadata",
                                          "candidate_metadata", "tabular_candidate_flags",
                                          "categorical_tokens"}),
    },
}
ADAPTER_FIELD_USAGE["candidate_mlp"] = ADAPTER_FIELD_USAGE["mlp"]
for _family in ("rnn", "lstm", "transformer"):
    ADAPTER_FIELD_USAGE[_family] = ADAPTER_FIELD_USAGE["sequence"]

EXPECTED_SEQUENCE_NUMERIC_WIDTH = 23
EXPECTED_CANDIDATE_NUMERIC_WIDTH = 8
EXPECTED_OUTCOME_FIELDS = frozenset({
    "schema_version", "sample_id", "outcome_event_id", "completed_event_index",
    "exact_action_id", "candidate_paths", "exact_candidate_id",
    "confirmed_destination",
})


def validate_adapter_field_usage(name: str) -> dict[str, frozenset[str]]:
    """Fail closed if any canonical field is unclassified for any adapter."""
    if name not in ADAPTER_FIELD_USAGE:
        raise ValueError("Adapter has no declared field-use contract: " + name)
    actual = {
        *("cutoff." + item.name for item in fields(CutoffInput)),
        *("candidate." + item.name for item in fields(CandidateInput)),
        *("context." + item for item in COMMON_CONTEXT_COLUMNS),
        *("feature." + item for item in CANDIDATE_NUMERIC_COLUMNS),
        *("feature." + item for item in CANDIDATE_CATEGORICAL_COLUMNS),
    }
    declared = [item for group in FIELD_GROUPS.values() for item in group]
    if len(declared) != len(set(declared)) or set(declared) != actual:
        raise ValueError("Canonical input schema changed without a complete field-use declaration")
    if (WINDOW_NAMES != EXPECTED_WINDOW_NAMES
            or {item.name for item in fields(MaturedOutcome)} != EXPECTED_OUTCOME_FIELDS):
        raise ValueError("Canonical history or outcome schema changed without a version update")
    if (asof.NUMERIC_WIDTH != EXPECTED_SEQUENCE_NUMERIC_WIDTH
            or asof.CANDIDATE_WIDTH != EXPECTED_CANDIDATE_NUMERIC_WIDTH):
        raise ValueError("Sequence numeric width changed without a schema update")
    for adapter, usage in ADAPTER_FIELD_USAGE.items():
        consumed = usage.get("consume", frozenset())
        ignored = usage.get("intentional_ignore", frozenset())
        if consumed & ignored or consumed | ignored != _ALL_GROUPS:
            raise ValueError("Adapter field-use declaration incomplete: " + adapter)
    return ADAPTER_FIELD_USAGE[name]


@dataclass(frozen=True)
class CandidateInput:
    """A prediction-time action and the parallel layers supporting it."""

    action_id: str
    kind: str
    owner_app_id: str | None
    semantic_key: str
    layers: tuple[parallel.Layer, ...]
    path_counts: tuple[int, int, int, int]
    direct_counts: tuple[int, int, int, int]
    source: str
    metadata: Mapping[str, Any] = field(repr=False, compare=False)

    @property
    def layer_keys(self) -> tuple[str, ...]:
        return tuple(layer.key for layer in self.layers)


@dataclass(frozen=True)
class CutoffInput:
    """Only foreground facts available at one prediction cutoff.

    `hierarchy` and metadata contain private locators solely in memory.
    Candidate order is stable and shared by every adapter.  The current
    destination and its owner application are never output candidates.
    """

    schema_version: str
    sample_id: str
    cutoff_event_index: int
    cutoff_timestamp_ms: int
    current_action_id: str
    current_kind: str
    current_owner_app_id: str | None
    current_semantic_key: str
    current_layers: tuple[parallel.Layer, ...]
    candidates: tuple[CandidateInput, ...]
    context: Mapping[str, float]
    sequence_steps: tuple[asof.Step, ...]
    sequence_numeric: np.ndarray = field(repr=False, compare=False)
    hierarchy: hierarchy_policy.Hierarchy = field(repr=False, compare=False)

    @property
    def action_ids(self) -> tuple[str, ...]:
        return tuple(item.action_id for item in self.candidates)

    @property
    def sequence_keys(self) -> tuple[str, ...]:
        return tuple(step.key for step in self.sequence_steps)

    def candidate(self, action_id: str) -> CandidateInput:
        for item in self.candidates:
            if item.action_id == action_id:
                return item
        raise KeyError(action_id)

    def feature_rows(self) -> tuple[dict[str, Any], ...]:
        """Shared candidate features for table models and pairwise online models.

        Tokens are opaque; URL and directory locators never become feature
        values.  The returned rows retain action IDs only for in-memory joins.
        """
        origin_token = offline.token(self.current_action_id)
        bundle = self.current_owner_app_id[4:] if self.current_owner_app_id else None
        current_parents = tuple(layer.key for layer in self.current_layers[1:])
        rows: list[dict[str, Any]] = []
        for item in self.candidates:
            candidate_bundle = item.owner_app_id[4:] if item.owner_app_id else None
            candidate_parents = tuple(layer.key for layer in item.layers[1:])
            record: dict[str, Any] = {
                "action_id": item.action_id,
                "semantic_key": item.semantic_key,
                "layer_keys": item.layer_keys,
                "origin_token": origin_token,
                "candidate_token": offline.token(item.action_id),
                "origin_candidate_token": offline.token(self.current_action_id + "\0" + item.action_id),
                "origin_kind": self.current_kind,
                "candidate_kind": item.kind,
                "origin_bundle_token": offline.token(bundle),
                "candidate_bundle_token": offline.token(candidate_bundle),
                "candidate_source": item.source,
                "hierarchy_layer_count_log": math.log1p(len(item.layers)),
                "is_derived_parent": float(item.source == "derived_from_known_locator"),
                "same_owner_app": float(bool(bundle and bundle == candidate_bundle)),
                **self.context,
            }
            for slot in range(4):
                record[f"candidate_parent_token_{slot + 1}"] = offline.token(
                    candidate_parents[slot] if slot < len(candidate_parents) else None)
            for slot in range(2):
                record[f"current_parent_token_{slot + 1}"] = offline.token(
                    current_parents[slot] if slot < len(current_parents) else None)
            for slot, name in enumerate(WINDOW_NAMES):
                record[f"path_count_{name}_log"] = math.log1p(item.path_counts[slot])
                record[f"direct_count_{name}_log"] = math.log1p(item.direct_counts[slot])
            expected = ({"action_id", "semantic_key", "layer_keys"}
                        | set(COMMON_CONTEXT_COLUMNS) | set(CANDIDATE_NUMERIC_COLUMNS)
                        | set(CANDIDATE_CATEGORICAL_COLUMNS))
            if set(record) != expected:
                raise ValueError("Candidate feature schema changed without a version update")
            rows.append(record)
        return tuple(rows)

    def statistical_view(self) -> dict[str, Any]:
        """Action/level/count view; a baseline may deliberately ignore fields."""
        validate_adapter_field_usage("statistical")
        return {"schema_version": self.schema_version,
                "origin": self.current_action_id,
                "current_layers": tuple(layer.key for layer in self.current_layers),
                "action_ids": self.action_ids,
                "candidate_layers": {item.action_id: item.layer_keys for item in self.candidates},
                "path_counts": {item.action_id: item.path_counts for item in self.candidates},
                "direct_counts": {item.action_id: item.direct_counts for item in self.candidates}}

    def mlp_view(self) -> dict[str, Any]:
        """Shared actions and pair features for an incremental pair scorer."""
        validate_adapter_field_usage("candidate_mlp")
        return {"schema_version": self.schema_version,
                "origin": self.current_action_id,
                "labels": self.action_ids,
                "x": dict(self.context),
                "candidate_features": {row["action_id"]: row for row in self.feature_rows()}}

    def catboost_rows(self) -> tuple[dict[str, Any], ...]:
        """One fixed-schema row per action, in the common action order."""
        validate_adapter_field_usage("catboost")
        columns = COMMON_CONTEXT_COLUMNS + CANDIDATE_NUMERIC_COLUMNS + CANDIDATE_CATEGORICAL_COLUMNS
        return tuple({name: row[name] for name in columns} for row in self.feature_rows())

    def sequence_view(self, bucket: Callable[[str], int], *, max_layers: int = asof.MAX_LAYERS
                      ) -> dict[str, Any]:
        """Arrays for RNN/LSTM/Transformer; caller supplies its own token map."""
        validate_adapter_field_usage("sequence")
        if max_layers < 1:
            raise ValueError("max_layers must be positive")
        step_layers = np.zeros((len(self.sequence_steps), max_layers), dtype=np.int64)
        candidate_layers = np.zeros((len(self.candidates), max_layers), dtype=np.int64)
        candidate_numeric = np.zeros((len(self.candidates), asof.CANDIDATE_WIDTH), dtype=np.float32)
        for i, step in enumerate(self.sequence_steps):
            for j, key in enumerate(step.layers[:max_layers]):
                step_layers[i, j] = bucket(key)
        for i, item in enumerate(self.candidates):
            for j, key in enumerate(item.layer_keys[:max_layers]):
                candidate_layers[i, j] = bucket(key)
            candidate_numeric[i, :4] = np.log1p(item.path_counts)
            candidate_numeric[i, 4:] = np.log1p(item.direct_counts)
        return {"schema_version": self.schema_version,
                "action_ids": self.action_ids,
                "context_numeric": np.asarray(
                    [self.context[name] for name in COMMON_CONTEXT_COLUMNS], dtype=np.float32),
                "sequence_ids": np.asarray([bucket(key) for key in self.sequence_keys], dtype=np.int64),
                "sequence_numeric": self.sequence_numeric.copy(),
                "sequence_layers": step_layers,
                "candidate_ids": np.asarray([bucket(item.action_id) for item in self.candidates], dtype=np.int64),
                "candidate_numeric": candidate_numeric,
                "candidate_layers": candidate_layers}


@dataclass(frozen=True)
class MaturedOutcome:
    """One actual destination and its relation to cutoff-time actions.

    This is training supervision, never a prediction input.  No numeric
    parent reward is embedded here; the caller chooses its declared policy.
    """

    schema_version: str
    sample_id: str
    outcome_event_id: str
    completed_event_index: int
    exact_action_id: str
    candidate_paths: Mapping[str, tuple[str, ...]]
    exact_candidate_id: str | None
    confirmed_destination: CandidateInput

    @property
    def useful_action_ids(self) -> frozenset[str]:
        return frozenset(self.candidate_paths)

    def labels_in_order(self, cutoff: CutoffInput) -> tuple[tuple[bool, bool, tuple[str, ...] | None], ...]:
        if self.schema_version != cutoff.schema_version or self.sample_id != cutoff.sample_id:
            raise ValueError("Outcome and cutoff input differ")
        return tuple((candidate.action_id == self.exact_candidate_id,
                      candidate.action_id in self.candidate_paths,
                      self.candidate_paths.get(candidate.action_id))
                     for candidate in cutoff.candidates)


class OutcomeLedger:
    """Admit one independent confirmed arrival once per model replay."""

    def __init__(self) -> None:
        self._seen: dict[str, MaturedOutcome] = {}

    def admit(self, outcome: MaturedOutcome) -> bool:
        previous = self._seen.get(outcome.outcome_event_id)
        if previous is not None:
            if previous != outcome:
                raise ValueError("One confirmed arrival has conflicting training labels")
            return False
        self._seen[outcome.outcome_event_id] = outcome
        return True

    @property
    def unique_arrivals(self) -> int:
        return len(self._seen)


class SharedInputBuilder:
    """Build shared views without writing candidates or reading future labels."""

    def __init__(self, events: list[dict[str, Any]], history: offline.TargetHistory,
                 rich_index: asof.RichIndex, *, timezone: str = "America/Los_Angeles") -> None:
        self.events = events
        self.history = history
        self.rich_index = rich_index
        self.timezone = timezone
        self._event_index_by_id = {event.get("id"): index for index, event in enumerate(events)
                                   if isinstance(event.get("id"), str)}
        if len(self._event_index_by_id) != sum(isinstance(event.get("id"), str) for event in events):
            raise ValueError("Duplicate raw event ID")
        confirmed_foreground: list[tuple[int, tuple[offline.Action, ...]]] = []
        for event_index, event in enumerate(events):
            if event.get("kind") != "context.change":
                continue
            data = event.get("data") or {}
            coverage = data.get("inputCoverage") or {}
            if (not isinstance(coverage, dict)
                    or coverage.get("version") != offline.FOREGROUND_COVERAGE_VERSION
                    or coverage.get("status") != "foreground_only"):
                continue
            meta = history.metadata_at(data.get("targetID"), event_index)
            action = offline.action_from_meta(meta, catalog_only=True)
            if action is None or action.kind not in offline.CONCRETE_KINDS | {"app"}:
                continue
            # Resolution updates and recovery confirmations can establish an
            # executable target, while independent visit counts remain solely
            # in RichIndex.records.  They are not extra training behaviors.
            proven = [action]
            if action.bundle and action.kind != "app":
                # The foreground child proves its owner was used.  An app
                # becomes executable only if an admitted app record already
                # existed at this same confirmation, never from a synthetic
                # bundle name or a later unconfirmed update.
                owner_meta = history.metadata_at("app:" + action.bundle, event_index)
                owner = offline.action_from_meta(owner_meta, catalog_only=True)
                if owner is not None and owner.kind == "app":
                    proven.append(owner)
            confirmed_foreground.append((event_index, tuple(proven)))
        self._foreground_actions = tuple(confirmed_foreground)
        self._foreground_indices = tuple(item[0] for item in confirmed_foreground)
        self._foreground_cursor = 0
        self._confirmed_actions: dict[str, offline.Action] = {}
        self._tree_cache: tuple[int, dict[str, offline.Action],
                                hierarchy_policy.Hierarchy,
                                tuple[tuple[Any, ...], ...]] | None = None
        self._current_layer_cache: dict[tuple[int, str, str],
                                        tuple[parallel.Layer, ...]] = {}

    def _observed_before_or_at(self, index: int) -> tuple[int, dict[str, offline.Action]]:
        count = bisect.bisect_right(self._foreground_indices, index)
        if count < self._foreground_cursor:
            self._foreground_cursor = 0
            self._confirmed_actions.clear()
        while self._foreground_cursor < count:
            _, actions = self._foreground_actions[self._foreground_cursor]
            for action in actions:
                self._confirmed_actions[action.key] = action
            self._foreground_cursor += 1
        return count, self._confirmed_actions

    def _asof_tree(self, index: int) -> tuple[int, dict[str, offline.Action],
                                            hierarchy_policy.Hierarchy,
                                            tuple[tuple[Any, ...], ...]]:
        # Only foreground confirmations change the available action graph.
        # Target updates alone and future labels must not promote an action.
        version, confirmed_actions = self._observed_before_or_at(index)
        cached = self._tree_cache
        if cached is None or cached[0] != version:
            # Each action uses the metadata captured at its most recent
            # foreground confirmation.  Later target.update events do not
            # silently change its URL, path, admission, or parent relation.
            catalog = dict(confirmed_actions)
            tree = hierarchy_policy.build_hierarchy(
                (action.meta for action in catalog.values()), executable=catalog.keys(),
                include_owner_apps=True)
            derived = tree.derived_parent_ids
            templates: list[tuple[Any, ...]] = []
            for action_id in sorted(tree.executable):
                meta = tree.metadata[action_id]
                kind = str(meta.get("kind", ""))
                if kind not in offline.CONCRETE_KINDS | {"app"}:
                    continue
                semantic = parallel.semantic_key(meta)
                owner = meta.get("bundleID")
                owner_id = "app:" + owner if isinstance(owner, str) and owner else None
                templates.append((action_id, kind, owner_id, semantic,
                                  parallel.layers_for(tree, action_id),
                                  "derived_from_known_locator" if action_id in derived
                                  else "recorded_action", MappingProxyType(meta)))
            cached = (version, catalog, tree, tuple(templates))
            self._tree_cache = cached
            self._current_layer_cache.clear()
        return cached

    def at(self, row: offline.Observation) -> CutoffInput:
        validate_adapter_field_usage("statistical")
        index = row.input_event_index
        if row.current is None or not 0 <= index < len(self.events):
            raise ValueError("Prediction needs a confirmed current foreground target")
        event = self.events[index]
        if event.get("kind") != "context.change" or event.get("timestampMs") != row.input_timestamp_ms:
            raise ValueError("Prediction cutoff does not match its foreground event")
        version, catalog, tree, templates = self._asof_tree(index)
        current_app_id = (row.current_parent.key if row.current_parent is not None
                          else "app:" + row.current.bundle if row.current.bundle else None)
        blocked = hierarchy_policy.blocked_current_actions(
            tree, current_action=row.current.key, current_app=current_app_id)
        current_semantic = parallel.semantic_key(row.current.meta)
        # The current observation can prove an input layer, but cannot promote
        # a synthetic application layer to a launchable catalog action.
        current_layer_key = (version, row.current.key, current_semantic)
        current_layers = self._current_layer_cache.get(current_layer_key)
        if current_layers is None:
            input_tree = parallel.focused_hierarchy(
                [action.meta for action in catalog.values()], row.current.meta)
            current_layers = parallel.layers_for(input_tree, row.current.key)
            self._current_layer_cache[current_layer_key] = current_layers
        path, direct = asof.path_frequency(self.rich_index, index, row.input_timestamp_ms)
        candidates: list[CandidateInput] = []
        for action_id, kind, owner_id, semantic, layers, source, meta in templates:
            if action_id in blocked or semantic == current_semantic:
                continue
            candidates.append(CandidateInput(
                action_id, kind, owner_id, semantic, layers,
                path.get(semantic, (0, 0, 0, 0)), direct.get(semantic, (0, 0, 0, 0)),
                source, meta))
        prefix = self.rich_index.by_event_index.get(index)
        if prefix is not None and prefix.current_key != row.current.key:
            raise ValueError("Foreground sequence and current action disagree")
        context, sequence_numeric = self._context(prefix, event, row.input_timestamp_ms)
        sequence_numeric.flags.writeable = False
        return CutoffInput(
            SCHEMA_VERSION, row.sample_id, index, row.input_timestamp_ms,
            row.current.key, row.current.kind, current_app_id, current_semantic,
            current_layers, tuple(candidates), MappingProxyType(context),
            prefix.steps if prefix is not None else (), sequence_numeric, tree)

    def matured(self, cutoff: CutoffInput, row: offline.Observation, *, learning_event_index: int
                ) -> MaturedOutcome:
        """Call only once feedback has formed; retain one outcome identity."""
        if cutoff.schema_version != SCHEMA_VERSION or cutoff.sample_id != row.sample_id:
            raise ValueError("Outcome does not match its original cutoff")
        if row.input_event_index != cutoff.cutoff_event_index:
            raise ValueError("Outcome anchor changed")
        if (row.truth is None or not row.outcome_event_id
                or not cutoff.cutoff_event_index < row.completed_event_index < learning_event_index):
            raise ValueError("Outcome has not matured before the learning cutoff")
        outcome_index = self._event_index_by_id.get(row.outcome_event_id)
        if (outcome_index is None or not cutoff.cutoff_event_index < outcome_index
                or outcome_index > row.completed_event_index
                or self.events[outcome_index].get("timestampMs") != row.outcome_timestamp_ms):
            raise ValueError("Confirmed outcome event does not match its feedback")
        outcome_event = self.events[outcome_index]
        outcome_data = outcome_event.get("data")
        coverage = outcome_data.get("inputCoverage") if isinstance(outcome_data, dict) else None
        if (outcome_event.get("kind") != "context.change"
                or not isinstance(coverage, dict)
                or coverage.get("version") != offline.FOREGROUND_COVERAGE_VERSION
                or coverage.get("status") != "foreground_only"):
            raise ValueError("Outcome is not a confirmed foreground arrival")
        confirmed_meta = dict(row.truth.meta)
        admission = confirmed_meta.get("admission")
        if (row.truth.kind == "app" and isinstance(admission, dict)
                and admission.get("basis") == "window_action_fallback"):
            raw_target_id = outcome_data.get("targetID")
            raw_meta = (self.history.metadata_at(raw_target_id, outcome_index)
                        if isinstance(raw_target_id, str) else None)
            if (not isinstance(raw_meta, dict)
                    or raw_meta.get("id") != raw_target_id
                    or raw_meta.get("kind") != "window"
                    or not offline.admitted(raw_meta)
                    or offline.action_from_meta(raw_meta, catalog_only=True) != row.truth):
                raise ValueError("Window fallback lacks its admitted foreground window")
            confirmed_meta["_verified_window_fallback"] = {
                "outcome_event_id": row.outcome_event_id,
                "bundleID": row.truth.bundle,
                "raw_kind": "window",
                "raw_admission_basis": raw_meta["admission"]["basis"],
            }
        prior_meta = cutoff.hierarchy.metadata.get(row.truth.key)
        same_concrete_place = (prior_meta is None or
                               parallel.semantic_key(prior_meta) == parallel.semantic_key(row.truth.meta))
        if same_concrete_place:
            outcome_meta = row.truth.meta
            outcome_node = row.truth.key
        else:
            # A reused target ID with a changed locator is a new concrete
            # place.  Give its score-only node a private temporary identity,
            # keeping the original prediction-time action and its old locator.
            alias = hashlib.sha256(row.outcome_event_id.encode("utf-8")).hexdigest()[:24]
            outcome_node = "matured-outcome:" + alias
            outcome_meta = {**row.truth.meta, "id": outcome_node}
        expanded = hierarchy_policy.add_outcome(cutoff.hierarchy, outcome_meta)
        exact_id = (row.truth.key if same_concrete_place and row.truth.key in cutoff.action_ids
                    else None)
        # A structural parent relation alone does not guarantee a score.  For
        # example, an app reached through both a directory parent and an
        # owner-app edge receives zero utility under the common scorer.  Keep
        # only positive-utility fallbacks, including equivalent homepages.
        fallback_paths = hierarchy_policy.useful_fallback_actions(
            expanded, outcome_node, evaluation._path_policy("product0.8"),
            current_action=cutoff.current_action_id,
            current_app=cutoff.current_owner_app_id)
        useful = {action_id: fallback_paths[action_id] for action_id in cutoff.action_ids
                  if action_id in fallback_paths}
        if exact_id is not None:
            useful[exact_id] = ()
        parent_catalog = [row.truth.meta]
        pointer = row.truth.meta
        visited = {row.truth.key}
        while isinstance(pointer.get("parentID"), str):
            parent_id = pointer["parentID"]
            if parent_id in visited:
                break
            visited.add(parent_id)
            parent = self.history.metadata_at(parent_id, outcome_index)
            if not isinstance(parent, dict):
                break
            parent_catalog.append(parent)
            pointer = parent
        outcome_tree = parallel.focused_hierarchy(parent_catalog, row.truth.meta)
        outcome_layers = parallel.layers_for(outcome_tree, row.truth.key)
        owner = row.truth.bundle
        confirmed_destination = CandidateInput(
            action_id=row.truth.key, kind=row.truth.kind,
            owner_app_id="app:" + owner if owner else None,
            semantic_key=parallel.semantic_key(row.truth.meta),
            layers=outcome_layers, path_counts=(0, 0, 0, 0),
            direct_counts=(0, 0, 0, 0), source="after_feedback_only",
            metadata=MappingProxyType(confirmed_meta))
        return MaturedOutcome(SCHEMA_VERSION, row.sample_id, row.outcome_event_id,
                              row.completed_event_index, row.truth.key, useful, exact_id,
                              confirmed_destination)

    def _context(self, prefix: asof.RichPrefix | None, event: Mapping[str, Any],
                 timestamp_ms: int) -> tuple[dict[str, float], np.ndarray]:
        if prefix is not None:
            context = asof.cutoff_features(prefix, self.events, self.timezone)
            numeric = asof.sequence_numeric(prefix, self.events, self.timezone)
            context["rich_input_missing"] = 0.0
        else:
            local = datetime.fromtimestamp(timestamp_ms / 1000, tz=ZoneInfo(self.timezone))
            hour = local.hour + local.minute / 60.0
            weekday = local.weekday()
            context = {name: 0.0 for name in COMMON_CONTEXT_COLUMNS}
            context.update({
                "time_hour_sin": math.sin(2 * math.pi * hour / 24),
                "time_hour_cos": math.cos(2 * math.pi * hour / 24),
                "time_weekday_sin": math.sin(2 * math.pi * weekday / 7),
                "time_weekday_cos": math.cos(2 * math.pi * weekday / 7),
                "rich_input_missing": 1.0,
            })
            numeric = np.zeros((0, asof.NUMERIC_WIDTH), dtype=np.float32)
        if set(context) != set(COMMON_CONTEXT_COLUMNS):
            raise ValueError("Shared context schema changed without a version update")
        return context, numeric
