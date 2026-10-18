#!/usr/bin/env python3
"""Run sealed, foreground-only local sequence-model shadow experiments.

This experiment compares small RNN, LSTM, and Transformer residual rankers
against the frozen single-step, no-context statistical prior already used by
the offline batch ranker.  It does not touch the application's runtime model.

Only confirmed ``context.change`` foreground arrivals are eligible destination
tokens. A confirmed excluded-foreground interval contributes one generic mask
token, never an excluded destination. Resolution/recovery records are not tokens;
session changes and unknown capture gaps remain hard boundaries. Target/action keys are hashed
into fixed buckets in memory; checkpoints contain tensor weights and
architecture metadata only, never a target vocabulary, candidate list, raw
prediction, locator, or per-sample output.

The held-out membership is scored exactly once, after all inner temporal
selection and final retraining.  Candidate catalogs are reconstructed only in
memory at each prediction cutoff by the shared offline ranker utilities.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import math
import os
import platform
import random
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from scipy.optimize import minimize_scalar


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import evaluation_contract as evaluation
import run_offline_batch_rankers as runner
import sequence_asof_features as asof


VERSION = "offline-sequence-shadow-v1"
PRIVATE_ROOT = Path.home() / "Library/Application Support/SmartApp/training"
HASH_PERSON = b"SeqShadowV1"
DEFAULT_SEQUENCE_LENGTH = 16
MIN_SEQUENCE_LENGTH = 2
TOKEN_BUCKETS = 2048
HARD_BOUNDARY_KINDS = {"session.start", "session.end", "capture.gap"}
# These records may describe a recovered/browser-history observation, not a
# new independently confirmed arrival.  Keeping the set conservative avoids
# treating a recovery snapshot as behavior.
NON_ARRIVAL_REASONS = {"resolution_update", "browser_history", "browser_resync"}


def require_torch() -> tuple[Any, Any, Any]:
    """Import the private experiment runtime only when a neural model is needed."""
    try:
        import torch
        import torch.nn as nn
        import torch.nn.functional as functional
    except ImportError as error:  # pragma: no cover - depends on local runtime
        raise RuntimeError(
            "PyTorch is required only for this local sequence shadow experiment; "
            "install it in the private SmartApp training runtime first."
        ) from error
    return torch, nn, functional


def action_bucket(action_key: str, bucket_count: int = TOKEN_BUCKETS) -> int:
    """Map a transient action key to a nonzero fixed hash bucket.

    Bucket zero is intentionally reserved for padding.  No inverse map is
    created or written, so a model checkpoint cannot become a candidate list.
    """
    if bucket_count < 2:
        raise ValueError("Need at least one non-padding sequence token bucket")
    digest = hashlib.blake2b(action_key.encode("utf-8"), digest_size=8, person=HASH_PERSON).digest()
    return int.from_bytes(digest, "little") % (bucket_count - 1) + 1


@dataclass(frozen=True)
class PrefixIndex:
    """In-memory bounded prefixes keyed by their confirmed arrival event index."""

    by_event_index: dict[int, tuple[str, ...]]
    action_by_event_index: dict[int, str]
    audit: dict[str, int]


@dataclass(frozen=True)
class SequenceAttachments:
    """Rows whose anchor matches a permitted prefix, plus model-eligible rows."""

    anchored: dict[int, tuple[str, ...]]
    eligible: dict[int, tuple[str, ...]]
    audit: dict[str, int]


@dataclass(frozen=True)
class SequenceExample:
    """One arrival and its transient, prediction-time action order.

    ``positive_indices`` opts into path supervision.  All useful actions are
    alternatives for *one* arrival, not separate visits or separately weighted
    rewards.  When the exact action is absent at the cutoff, ``truth_index``
    may be None as long as a useful parent action is present.  The default
    remains the original exact-only cross-entropy objective.
    """

    sequence_ids: tuple[int, ...]
    candidate_ids: tuple[int, ...]
    prior: np.ndarray
    truth_index: int | None
    sequence_numeric: np.ndarray | None = None
    sequence_layers: np.ndarray | None = None
    candidate_numeric: np.ndarray | None = None
    candidate_layers: np.ndarray | None = None
    weight: float = 1.0
    positive_indices: tuple[int, ...] | None = None
    context_numeric: np.ndarray | None = None


def attach_rich_sequences(rows: Iterable[runner.Observation], index: asof.RichIndex,
                          minimum_length: int = MIN_SEQUENCE_LENGTH
                          ) -> tuple[dict[int, asof.RichPrefix], dict[str, int]]:
    """Attach an observed anchor only; masks never count as extra arrivals."""
    attached: dict[int, asof.RichPrefix] = {}
    audit: collections.Counter[str] = collections.Counter()
    for row in rows:
        prefix = index.by_event_index.get(row.input_event_index)
        if prefix is None or row.current is None:
            audit["missing_asof_prefix_or_current"] += 1
        elif prefix.current_key != row.current.key:
            audit["anchor_action_mismatch"] += 1
        elif prefix.arrival_count < minimum_length:
            audit["too_few_independent_arrivals"] += 1
        else:
            attached[row.position] = prefix
            audit["eligible"] += 1
    return attached, dict(sorted(audit.items()))


def _rich_arrays(group: runner.QueryGroup, prefix: asof.RichPrefix,
                 index: asof.RichIndex, history: runner.TargetHistory,
                 events: list[dict[str, Any]], mode: str, bucket_count: int
                 ) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None, np.ndarray | None]:
    if mode not in {"target_only", "temporal", "hierarchy", "full"}:
        raise ValueError(f"Unknown sequence input mode: {mode}")
    bucket = lambda key: action_bucket(key, bucket_count)
    temporal = mode in {"temporal", "full"}
    hierarchy = mode in {"hierarchy", "full"}
    sequence_numeric = asof.sequence_numeric(prefix, events) if temporal else None
    sequence_layers = asof.step_layer_ids(prefix, bucket) if hierarchy else None
    candidate_layers = candidate_numeric = None
    if hierarchy:
        candidate_layers, candidate_numeric = asof.candidate_inputs(
            group, index, history, bucket, bucket_count=bucket_count)
    return sequence_numeric, sequence_layers, candidate_numeric, candidate_layers


def foreground_coverage(data: dict[str, Any]) -> bool:
    coverage = data.get("inputCoverage")
    return (isinstance(coverage, dict)
            and coverage.get("version") == runner.FOREGROUND_COVERAGE_VERSION
            and coverage.get("status") == "foreground_only")


def build_prefix_index(events: list[dict[str, Any]], history: runner.TargetHistory,
                       sequence_length: int) -> PrefixIndex:
    """Build only as-of, confirmed foreground-arrival prefixes.

    A capture gap is deliberately treated as a hard boundary even if a future
    collector learns more about the gap.  That conservative rule prevents an
    unknown observation period from silently becoming an ordered transition.
    """
    if sequence_length < MIN_SEQUENCE_LENGTH:
        raise ValueError("Sequence length must retain at least the current and one prior arrival")
    by_event: dict[int, tuple[str, ...]] = {}
    action_by_event: dict[int, str] = {}
    audit: collections.Counter[str] = collections.Counter()
    active: list[str] = []
    excluded_scope_id: str | None = None
    current_session: str | None = None

    def cut(reason: str) -> None:
        nonlocal active, excluded_scope_id
        active = []
        excluded_scope_id = None
        audit["hard_boundary_" + reason] += 1

    for index, event in enumerate(events):
        session = event.get("sessionID")
        if not isinstance(session, str) or not session:
            # Do not carry an ordered prefix through an unidentified session.
            if current_session is not None or active:
                cut("missing_session")
            current_session = None
        elif current_session is None:
            current_session = session
        elif session != current_session:
            cut("session_change")
            current_session = session

        kind = event.get("kind")
        if kind in HARD_BOUNDARY_KINDS:
            cut(str(kind).replace(".", "_"))
            if kind == "session.end":
                current_session = None
            continue
        if kind == "observation.scope":
            data = event.get("data") or {}
            phase, scope_id = data.get("phase"), data.get("scopeID")
            if data.get("scopeVersion") != "excluded-foreground-v1" or not isinstance(scope_id, str):
                cut("invalid_excluded_scope")
            elif phase == "begin" and excluded_scope_id is None:
                excluded_scope_id = scope_id
                active.append("observation:excluded")
                audit["masked_excluded_foreground"] += 1
            elif phase == "end" and excluded_scope_id == scope_id:
                excluded_scope_id = None
            else:
                cut("unpaired_excluded_scope")
            continue
        if excluded_scope_id is not None:
            if kind == "context.change":
                cut("context_during_excluded_scope")
            continue
        if kind != "context.change":
            continue
        audit["context_events_seen"] += 1
        data = event.get("data")
        if not isinstance(data, dict) or not foreground_coverage(data):
            audit["context_not_foreground_covered"] += 1
            continue
        reason = data.get("reason")
        if not isinstance(reason, str):
            audit["context_missing_reason"] += 1
            continue
        if reason in NON_ARRIVAL_REASONS:
            audit["context_non_arrival_" + reason] += 1
            continue
        target_id = data.get("targetID")
        meta = history.metadata_at(target_id, index)
        action = runner.action_from_meta(meta, catalog_only=False)
        if action is None or action.kind not in runner.ACTIONABLE_KINDS:
            audit["context_without_actionable_asof_target"] += 1
            continue

        # Several observer messages can describe one unchanged foreground
        # destination.  Repeated copies do not become new behavior tokens.
        if active and active[-1] == action.key:
            audit["consecutive_duplicate_arrival"] += 1
        else:
            active.append(action.key)
            audit["independent_foreground_arrival"] += 1
        prefix = tuple(active[-sequence_length:])
        by_event[index] = prefix
        action_by_event[index] = action.key

    return PrefixIndex(dict(by_event), dict(action_by_event), dict(sorted(audit.items())))


def attach_sequences(rows: Iterable[runner.Observation], prefixes: PrefixIndex,
                     minimum_length: int = MIN_SEQUENCE_LENGTH) -> SequenceAttachments:
    """Attach only an anchor's own bounded prefix; never reconstruct a missing one."""
    anchored: dict[int, tuple[str, ...]] = {}
    eligible: dict[int, tuple[str, ...]] = {}
    audit: collections.Counter[str] = collections.Counter()
    for row in rows:
        if row.current is None:
            audit["row_missing_current_action"] += 1
            continue
        sequence = prefixes.by_event_index.get(row.input_event_index)
        if not sequence:
            audit["anchor_without_confirmed_prefix"] += 1
            continue
        if sequence[-1] != row.current.key:
            # Never append a synthetic current token to repair an inconsistency.
            audit["anchor_action_mismatch"] += 1
            continue
        anchored[row.position] = sequence
        if len(sequence) >= minimum_length:
            eligible[row.position] = sequence
            audit["model_eligible_prefix"] += 1
        else:
            audit["single_arrival_prefix_falls_back_to_baseline"] += 1
    return SequenceAttachments(anchored, eligible, dict(sorted(audit.items())))


def examples_from_groups(groups: Iterable[runner.QueryGroup], sequences: dict[int, tuple[str, ...]],
                         bucket_count: int, *, rich_sequences: dict[int, asof.RichPrefix] | None = None,
                         rich_index: asof.RichIndex | None = None,
                         history: runner.TargetHistory | None = None,
                         events: list[dict[str, Any]] | None = None,
                         input_mode: str = "target_only",
                         inferred_sample_ids: frozenset[str] = frozenset(),
                         inferred_weight: float = 1.0) -> tuple[list[SequenceExample], dict[str, int]]:
    """Convert temporary query groups to hash-only training tensors in memory."""
    result: list[SequenceExample] = []
    audit: collections.Counter[str] = collections.Counter()
    if not 0 <= inferred_weight <= 1:
        raise ValueError("Inferred training weight must be between zero and one")
    if input_mode != "target_only" and (rich_sequences is None or rich_index is None
                                        or history is None or events is None):
        raise ValueError("Rich sequence input requires observed prefixes, history, and events")
    for group in groups:
        sequence = sequences.get(group.row.position)
        if sequence is None:
            audit["no_eligible_sequence"] += 1
            continue
        if group.truth_index is None:
            audit["truth_not_in_asof_catalog"] += 1
            continue
        if len(group.actions) < 2:
            audit["fewer_than_two_candidates"] += 1
            continue
        weight = inferred_weight if group.row.sample_id in inferred_sample_ids else 1.0
        if weight == 0:
            audit["zero_weight_inferred_skipped"] += 1
            continue
        rich = rich_sequences.get(group.row.position) if rich_sequences is not None else None
        if input_mode != "target_only" and (rich is None or rich.keys != sequence):
            audit["missing_or_mismatched_rich_prefix"] += 1
            continue
        arrays = (_rich_arrays(group, rich, rich_index, history, events, input_mode, bucket_count)
                  if input_mode != "target_only" else (None, None, None, None))
        result.append(SequenceExample(
            sequence_ids=tuple(action_bucket(key, bucket_count) for key in sequence),
            candidate_ids=tuple(action_bucket(action.key, bucket_count) for action in group.actions),
            prior=np.asarray(group.prior, dtype=np.float32),
            truth_index=group.truth_index,
            sequence_numeric=arrays[0], sequence_layers=arrays[1],
            candidate_numeric=arrays[2], candidate_layers=arrays[3], weight=weight,
        ))
        audit["ranking_examples"] += 1
        if weight < 1:
            audit["downweighted_inferred_examples"] += 1
    return result, dict(sorted(audit.items()))


def build_sequence_model(family: str, config: dict[str, Any]) -> Any:
    """Create a deliberately small sequence encoder with a hashed candidate head."""
    torch, nn, _ = require_torch()
    bucket_count = int(config["bucket_count"])
    embedding_dim = int(config["embedding_dim"])
    hidden_dim = int(config["hidden_dim"])
    layers = int(config["layers"])
    dropout = float(config["dropout"])
    max_length = int(config["sequence_length"])
    input_mode = str(config.get("input_mode", "target_only"))
    use_common_context = bool(config.get("use_common_context", False))
    common_context_width = int(config.get("common_context_width", 0))
    if family not in {"rnn", "lstm", "transformer"}:
        raise ValueError(f"Unsupported sequence family: {family}")
    if input_mode not in {"target_only", "temporal", "hierarchy", "full"}:
        raise ValueError(f"Unsupported sequence input mode: {input_mode}")
    if bucket_count < 2 or embedding_dim < 2 or hidden_dim < 2 or layers < 1:
        raise ValueError("Invalid small sequence model dimensions")
    if use_common_context and common_context_width < 1:
        raise ValueError("Common cutoff context needs an explicit positive width")

    class CandidateSequenceModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.family = family
            self.embedding = nn.Embedding(bucket_count, embedding_dim, padding_idx=0)
            self.candidate_embedding = nn.Embedding(bucket_count, hidden_dim, padding_idx=0)
            self.candidate_bias = nn.Embedding(bucket_count, 1, padding_idx=0)
            self.temporal_projection = (nn.Linear(asof.NUMERIC_WIDTH, embedding_dim)
                                        if input_mode in {"temporal", "full"} else None)
            self.step_layer_embedding = (nn.Embedding(bucket_count, embedding_dim, padding_idx=0)
                                         if input_mode in {"hierarchy", "full"} else None)
            self.candidate_layer_embedding = (nn.Embedding(bucket_count, hidden_dim, padding_idx=0)
                                              if input_mode in {"hierarchy", "full"} else None)
            self.frequency_projection = (nn.Linear(asof.CANDIDATE_WIDTH, hidden_dim)
                                         if input_mode in {"hierarchy", "full"} else None)
            self.common_context_projection = (nn.Linear(common_context_width, hidden_dim)
                                              if use_common_context else None)
            if family == "rnn":
                self.encoder = nn.RNN(embedding_dim, hidden_dim, num_layers=layers,
                                      batch_first=True, nonlinearity="tanh",
                                      dropout=dropout if layers > 1 else 0.0)
                self.position = None
                self.projection = None
            elif family == "lstm":
                self.encoder = nn.LSTM(embedding_dim, hidden_dim, num_layers=layers,
                                       batch_first=True, dropout=dropout if layers > 1 else 0.0)
                self.position = None
                self.projection = None
            else:
                heads = int(config["heads"])
                if embedding_dim % heads:
                    raise ValueError("Transformer embedding dimension must divide evenly by head count")
                self.position = nn.Embedding(max_length, embedding_dim)
                layer = nn.TransformerEncoderLayer(
                    d_model=embedding_dim, nhead=heads, dim_feedforward=int(config["feedforward_dim"]),
                    dropout=dropout, activation="gelu", batch_first=True, norm_first=False,
                )
                self.encoder = nn.TransformerEncoder(layer, num_layers=layers, enable_nested_tensor=False)
                self.projection = nn.Linear(embedding_dim, hidden_dim)

        def encode(self, sequence_ids: Any, *, sequence_numeric: Any = None,
                   sequence_layers: Any = None) -> Any:
            if sequence_ids.ndim != 2 or sequence_ids.shape[0] != 1:
                raise ValueError("Sequence encoder expects one nonempty unpadded sequence")
            if sequence_ids.shape[1] < 1 or sequence_ids.shape[1] > max_length:
                raise ValueError("Sequence length is outside the configured bounded window")
            embedded = self.embedding(sequence_ids)
            if self.temporal_projection is not None:
                if sequence_numeric is None or sequence_numeric.shape != (
                        1, sequence_ids.shape[1], asof.NUMERIC_WIDTH):
                    raise ValueError("Temporal input must match the observed sequence")
                embedded = embedded + self.temporal_projection(sequence_numeric)
            if self.step_layer_embedding is not None:
                if sequence_layers is None or sequence_layers.shape != (
                        1, sequence_ids.shape[1], asof.MAX_LAYERS):
                    raise ValueError("Parallel layers must match the observed sequence")
                layer_vectors = self.step_layer_embedding(sequence_layers)
                count = (sequence_layers != 0).sum(dim=-1, keepdim=True).clamp(min=1)
                embedded = embedded + layer_vectors.sum(dim=-2) / count
            if family == "rnn":
                _, hidden = self.encoder(embedded)
                return hidden[-1]
            if family == "lstm":
                _, (hidden, _) = self.encoder(embedded)
                return hidden[-1]
            positions = torch.arange(sequence_ids.shape[1], device=sequence_ids.device).unsqueeze(0)
            encoded = self.encoder(embedded + self.position(positions))
            return self.projection(encoded[:, -1, :])

        def raw_scores(self, sequence_ids: Any, candidate_ids: Any, *,
                       sequence_numeric: Any = None, sequence_layers: Any = None,
                       candidate_numeric: Any = None, candidate_layers: Any = None,
                       context_numeric: Any = None) -> Any:
            if candidate_ids.ndim != 1 or candidate_ids.numel() < 1:
                raise ValueError("Candidate scorer expects at least one transient as-of action")
            context = self.encode(sequence_ids, sequence_numeric=sequence_numeric,
                                  sequence_layers=sequence_layers)
            if self.common_context_projection is not None:
                if context_numeric is None or context_numeric.shape != (1, common_context_width):
                    raise ValueError("Common context must match the shared cutoff schema")
                context = context + self.common_context_projection(context_numeric)
            candidate = self.candidate_embedding(candidate_ids)
            if self.candidate_layer_embedding is not None:
                if candidate_layers is None or candidate_layers.shape != (
                        candidate_ids.numel(), asof.MAX_LAYERS):
                    raise ValueError("Candidate parallel layers have the wrong shape")
                if candidate_numeric is None or candidate_numeric.shape != (
                        candidate_ids.numel(), asof.CANDIDATE_WIDTH):
                    raise ValueError("Candidate frequency input has the wrong shape")
                layer_vectors = self.candidate_layer_embedding(candidate_layers)
                count = (candidate_layers != 0).sum(dim=-1, keepdim=True).clamp(min=1)
                candidate = candidate + layer_vectors.sum(dim=-2) / count
                candidate = candidate + self.frequency_projection(candidate_numeric)
            bias = self.candidate_bias(candidate_ids).squeeze(-1)
            return (candidate @ context.squeeze(0)) / math.sqrt(hidden_dim) + bias

    return CandidateSequenceModel()


def configure_determinism(seed: int) -> None:
    torch, _, _ = require_torch()
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    # CPU-only execution keeps the training path reproducible on the local Mac.
    torch.use_deterministic_algorithms(True, warn_only=True)


def sequence_training_loss(logits: Any, example: SequenceExample, functional: Any) -> Any:
    """One normalized likelihood term for one confirmed destination arrival.

    A path example maximizes the *combined* probability of its eligible
    prediction-time actions.  This does not duplicate the arrival for each
    ancestor or assign an unagreed numeric reward to a parent.  The default
    single exact label retains the previous cross-entropy computation.
    """
    if logits.ndim != 1 or logits.numel() != len(example.candidate_ids):
        raise ValueError("Sequence logits must match the shared action order")
    if example.positive_indices is None:
        if example.truth_index is None:
            raise ValueError("Exact-only training needs an exact candidate")
        positives = (example.truth_index,)
    else:
        positives = example.positive_indices
        if not positives:
            raise ValueError("Path training needs at least one useful action")
        if len(set(positives)) != len(positives):
            raise ValueError("Useful actions cannot be counted twice")
        if example.truth_index is not None and example.truth_index not in positives:
            raise ValueError("An available exact action must also be useful")
    if any(type(index) is not int or not 0 <= index < logits.numel() for index in positives):
        raise ValueError("Training label is outside the shared action order")
    if example.positive_indices is None:
        target = logits.new_tensor([example.truth_index]).long()
        return functional.cross_entropy(logits.unsqueeze(0), target)
    return logits.logsumexp(dim=0) - logits[list(positives)].logsumexp(dim=0)


def train_sequence_model(family: str, config: dict[str, Any], examples: list[SequenceExample]) -> tuple[Any, dict[str, Any]]:
    """Fit a residual ranker using only the temporary causal training groups."""
    if not examples:
        raise ValueError("No causally valid sequence training examples")
    torch, _, functional = require_torch()
    configure_determinism(int(config["seed"]))
    model = build_sequence_model(family, config).to("cpu")
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(config["learning_rate"]),
                                  weight_decay=float(config["weight_decay"]))
    generator = np.random.default_rng(int(config["seed"]))
    losses: list[float] = []
    for _ in range(int(config["epochs"])):
        model.train()
        epoch_losses: list[float] = []
        for example_index in generator.permutation(len(examples)):
            example = examples[int(example_index)]
            sequence_ids = torch.tensor(example.sequence_ids, dtype=torch.long).unsqueeze(0)
            candidate_ids = torch.tensor(example.candidate_ids, dtype=torch.long)
            prior = torch.from_numpy(example.prior)
            sequence_numeric = (torch.from_numpy(example.sequence_numeric).unsqueeze(0)
                                if example.sequence_numeric is not None else None)
            sequence_layers = (torch.from_numpy(example.sequence_layers).unsqueeze(0)
                               if example.sequence_layers is not None else None)
            candidate_numeric = (torch.from_numpy(example.candidate_numeric)
                                 if example.candidate_numeric is not None else None)
            candidate_layers = (torch.from_numpy(example.candidate_layers)
                                if example.candidate_layers is not None else None)
            context_numeric = (torch.from_numpy(example.context_numeric).unsqueeze(0)
                               if example.context_numeric is not None else None)
            optimizer.zero_grad(set_to_none=True)
            logits = prior + model.raw_scores(
                sequence_ids, candidate_ids, sequence_numeric=sequence_numeric,
                sequence_layers=sequence_layers, candidate_numeric=candidate_numeric,
                candidate_layers=candidate_layers, context_numeric=context_numeric)
            loss = sequence_training_loss(logits, example, functional) * example.weight
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite sequence training loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            epoch_losses.append(float(loss.detach().cpu()))
        losses.append(float(np.mean(epoch_losses)))
    model.eval()
    path_examples = sum(example.positive_indices is not None for example in examples)
    training = {
        "examples": len(examples), "epochs": int(config["epochs"]),
        "initial_mean_training_loss": losses[0], "final_mean_training_loss": losses[-1],
        "path_supervised_examples": path_examples,
        "path_examples_without_exact_candidate": sum(
            example.positive_indices is not None and example.truth_index is None for example in examples),
        "loss_objective": "set_likelihood" if path_examples else "exact_cross_entropy",
        "device": "cpu", "deterministic_algorithms": True,
    }
    if not path_examples:
        training["initial_mean_cross_entropy"] = losses[0]
        training["final_mean_cross_entropy"] = losses[-1]
    return model, training


def raw_sequence_scores(model: Any, group: runner.QueryGroup, sequences: dict[int, tuple[str, ...]],
                        bucket_count: int, *, rich_sequences: dict[int, asof.RichPrefix] | None = None,
                        rich_index: asof.RichIndex | None = None,
                        history: runner.TargetHistory | None = None,
                        events: list[dict[str, Any]] | None = None,
                        input_mode: str = "target_only") -> np.ndarray:
    """Compute a neural correction without inspecting the held-out truth label."""
    sequence = sequences.get(group.row.position)
    if sequence is None or not group.actions:
        return np.zeros(len(group.actions), dtype=float)
    torch, _, _ = require_torch()
    device = next(model.parameters()).device
    sequence_ids = torch.tensor(
        [action_bucket(key, bucket_count) for key in sequence], dtype=torch.long, device=device
    ).unsqueeze(0)
    candidate_ids = torch.tensor(
        [action_bucket(action.key, bucket_count) for action in group.actions], dtype=torch.long, device=device
    )
    rich = rich_sequences.get(group.row.position) if rich_sequences is not None else None
    if input_mode != "target_only" and (rich is None or rich_index is None or history is None
                                        or events is None or rich.keys != sequence):
        raise ValueError("Missing as-of rich input for an eligible sequence")
    arrays = (_rich_arrays(group, rich, rich_index, history, events, input_mode, bucket_count)
              if input_mode != "target_only" else (None, None, None, None))
    tensor = lambda value, dtype: (torch.tensor(value, dtype=dtype, device=device)
                                   if value is not None else None)
    model.eval()
    with torch.no_grad():
        values = model.raw_scores(
            sequence_ids, candidate_ids,
            sequence_numeric=tensor(arrays[0], torch.float32).unsqueeze(0) if arrays[0] is not None else None,
            sequence_layers=tensor(arrays[1], torch.long).unsqueeze(0) if arrays[1] is not None else None,
            candidate_numeric=tensor(arrays[2], torch.float32),
            candidate_layers=tensor(arrays[3], torch.long),
        ).detach().cpu().numpy().astype(float, copy=False)
    if len(values) != len(group.actions) or not np.isfinite(values).all():
        raise ValueError("Sequence model returned invalid transient candidate scores")
    return values


@dataclass
class SequenceResidual:
    model: Any
    sequences: dict[int, tuple[str, ...]]
    bucket_count: int
    scale: float
    rich_sequences: dict[int, asof.RichPrefix] | None = None
    rich_index: asof.RichIndex | None = None
    history: runner.TargetHistory | None = None
    events: list[dict[str, Any]] | None = None
    input_mode: str = "target_only"

    def correction(self, group: runner.QueryGroup) -> np.ndarray:
        return self.scale * raw_sequence_scores(
            self.model, group, self.sequences, self.bucket_count,
            rich_sequences=self.rich_sequences, rich_index=self.rich_index,
            history=self.history, events=self.events, input_mode=self.input_mode)


def fit_residual_scale(model: Any, groups: Iterable[runner.QueryGroup], sequences: dict[int, tuple[str, ...]],
                       bucket_count: int, *, rich_sequences: dict[int, asof.RichPrefix] | None = None,
                       rich_index: asof.RichIndex | None = None,
                       history: runner.TargetHistory | None = None,
                       events: list[dict[str, Any]] | None = None,
                       input_mode: str = "target_only") -> tuple[float, dict[str, Any]]:
    """Learn, rather than hand-set, a nonnegative statistical-residual scale."""
    usable: list[tuple[runner.QueryGroup, np.ndarray]] = []
    for group in groups:
        if group.truth_index is None or group.row.position not in sequences:
            continue
        raw = raw_sequence_scores(model, group, sequences, bucket_count,
                                  rich_sequences=rich_sequences, rich_index=rich_index,
                                  history=history, events=events, input_mode=input_mode)
        if len(raw):
            usable.append((group, raw))
    if not usable:
        return 0.0, {"calibration_groups": 0, "status": "no_sequence_eligible_groups", "scale": 0.0}

    def objective(scale: float) -> float:
        total = 0.0
        for group, raw in usable:
            assert group.truth_index is not None
            logits = group.prior + scale * raw
            total += float(np.logaddexp.reduce(logits) - logits[group.truth_index])
        return total / len(usable)

    result = minimize_scalar(objective, bounds=(0.0, 32.0), method="bounded", options={"xatol": 1e-5})
    selected = float(result.x) if result.success and math.isfinite(result.x) else 0.0
    return selected, {
        "calibration_groups": len(usable), "status": "optimized" if result.success else "optimizer_failed",
        "objective": float(objective(selected)), "scale": selected,
    }


def metric_key(score: dict[str, Any], config_id: str) -> tuple[Any, ...]:
    """Top4 remains primary; Top2 is a required tie-breaking reference."""
    exact = score["exact"]
    return (-score["top4_hits"], -exact["top4_hits"], -score["top2_hits"], -exact["top2_hits"],
            -score["rank_score"]["mean"], -score["top1_hits"], -exact["top1_hits"], config_id)


def model_configurations(seed: int, sequence_length: int, bucket_count: int) -> dict[str, list[dict[str, Any]]]:
    """A compact, predeclared grid selected without opening the held-out test."""
    common = {"bucket_count": bucket_count, "sequence_length": sequence_length, "layers": 1,
              "dropout": 0.0, "weight_decay": 5e-4}
    return {
        "rnn": [
            {**common, "id": "rnn-24", "embedding_dim": 24, "hidden_dim": 24, "epochs": 24,
             "learning_rate": 0.025, "seed": seed + 11},
            {**common, "id": "rnn-32", "embedding_dim": 32, "hidden_dim": 32, "epochs": 28,
             "learning_rate": 0.020, "seed": seed + 12},
        ],
        "lstm": [
            {**common, "id": "lstm-24", "embedding_dim": 24, "hidden_dim": 24, "epochs": 24,
             "learning_rate": 0.020, "seed": seed + 21},
            {**common, "id": "lstm-32", "embedding_dim": 32, "hidden_dim": 32, "epochs": 28,
             "learning_rate": 0.016, "seed": seed + 22},
        ],
        "transformer": [
            {**common, "id": "transformer-24-h2", "embedding_dim": 24, "hidden_dim": 24, "heads": 2,
             "feedforward_dim": 48, "epochs": 24, "learning_rate": 0.016, "seed": seed + 31},
            {**common, "id": "transformer-32-h4", "embedding_dim": 32, "hidden_dim": 32, "heads": 4,
             "feedforward_dim": 64, "epochs": 28, "learning_rate": 0.012, "seed": seed + 32},
        ],
    }


def run_inner_selection(fit: list[runner.Observation], calibration: list[runner.Observation],
                        selection: list[runner.Observation], history: runner.TargetHistory,
                        train_sequences: dict[int, tuple[str, ...]], sequence_length: int,
                        bucket_count: int, seed: int) -> tuple[dict[str, Any], dict[str, Any]]:
    """Select one small architecture per family without scoring the held-out split."""
    fit_groups, fit_group_audit = runner.build_causal_training_groups(fit, history)
    frozen_fit_stats = runner.HistoryStats.from_rows(fit)
    calibration_groups = runner.build_static_groups(calibration, frozen_fit_stats, history)
    selection_groups = runner.build_static_groups(selection, frozen_fit_stats, history)
    fit_examples, fit_example_audit = examples_from_groups(fit_groups, train_sequences, bucket_count)
    baseline_scores, baseline_outputs, baseline_audited = runner.score_groups(selection, selection_groups, {
        "no_context_statistical": runner.baseline_correction,
    })
    baseline_leaderboards = runner.score_leaderboards(baseline_audited, baseline_outputs)
    baseline_selection_board, baseline_selection_metrics = evaluation.primary_leaderboard_metric(
        baseline_leaderboards, "no_context_statistical")
    results: dict[str, list[dict[str, Any]]] = {"rnn": [], "lstm": [], "transformer": []}
    for family, configs in model_configurations(seed, sequence_length, bucket_count).items():
        for config in configs:
            model, training = train_sequence_model(family, config, fit_examples)
            scale, calibration_audit = fit_residual_scale(model, calibration_groups, train_sequences, bucket_count)
            residual = SequenceResidual(model, train_sequences, bucket_count, scale)
            scores, outputs, audited = runner.score_groups(selection, selection_groups, {"candidate": residual.correction})
            leaderboards = runner.score_leaderboards(audited, outputs)
            selection_board, selection_metrics = evaluation.primary_leaderboard_metric(leaderboards, "candidate")
            results[family].append({
                "id": config["id"], "config": config, "training": training,
                "frozen_inner_calibration": calibration_audit, "score": scores["candidate"],
                "leaderboards": leaderboards, "selection_leaderboard": selection_board,
                "selection_metrics": selection_metrics,
            })
            del residual
            del model
    selected = {family: min(candidates, key=lambda item: metric_key(item["selection_metrics"], item["id"]))
                for family, candidates in results.items()}
    summary = {
        "version": VERSION, "test_accessed": False,
        "selection_rule": (
            "maximize prior-supported (or all-legal if empty) service-accepted Top4, exact Top4, service-accepted Top2, exact Top2, "
            "RankScore, service-accepted Top1, exact Top1; stable ID breaks ties"
        ),
        "baseline_on_selection": baseline_scores["no_context_statistical"],
        "baseline_selection_leaderboards": baseline_leaderboards,
        "baseline_selection_leaderboard": baseline_selection_board,
        "baseline_selection_metrics": baseline_selection_metrics,
        "fit_training_groups": fit_group_audit,
        "fit_sequence_examples": fit_example_audit,
        "candidate_results": results,
        "selected": selected,
    }
    state = {"fit_groups": fit_groups, "frozen_fit_stats": frozen_fit_stats,
             "calibration_groups": calibration_groups, "selection_groups": selection_groups}
    return summary, state


def save_models(output: Path, fitted: dict[str, tuple[Any, dict[str, Any], float]]) -> dict[str, str]:
    """Write private tensor checkpoints without a vocab/candidate mapping."""
    torch, _, _ = require_torch()
    model_dir = output / "models"
    model_dir.mkdir(mode=0o700)
    model_dir.chmod(0o700)
    hashes: dict[str, str] = {}
    for family, (model, config, scale) in fitted.items():
        path = model_dir / (family + "-residual.pt")
        payload = {
            "version": VERSION,
            "family": family,
            "config": config,
            "frozen_inner_calibration_scale": scale,
            "tokenization": {
                "algorithm": "blake2b fixed bucket", "bucket_count": config["bucket_count"],
                "padding_bucket": 0, "target_vocabulary_persisted": False,
            },
            "state_dict": model.state_dict(),
        }
        torch.save(payload, path)
        path.chmod(0o600)
        hashes[str(path.relative_to(output))] = runner.sha256(path)
    return hashes


def source_artifacts(output: Path) -> None:
    for source in (Path(__file__), SCRIPT_DIR / "evaluation_contract.py",
                   SCRIPT_DIR / "prepare_offline_batch_split.py", SCRIPT_DIR / "run_offline_batch_rankers.py",
                   SCRIPT_DIR / "sequence_asof_features.py"):
        target = output / ("source-" + source.name)
        if target.exists():
            raise ValueError("Duplicate source artifact")
        shutil.copyfile(source, target)
        target.chmod(0o600)


def environment() -> dict[str, Any]:
    torch, _, _ = require_torch()
    freeze = subprocess.run([sys.executable, "-m", "pip", "freeze"], check=False,
                            capture_output=True, text=True)
    return {
        "python": sys.version, "executable": sys.executable, "platform": platform.platform(),
        "modules": {"numpy": np.__version__, "scipy": __import__("scipy").__version__,
                    "torch": torch.__version__},
        "torch_device": "cpu", "pip_freeze": freeze.stdout.splitlines() if freeze.returncode == 0 else [],
    }


def artifact_hashes(output: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    for path in sorted(output.rglob("*")):
        if path.is_file() and path.name not in {"artifact-hashes.json", "run.json"}:
            result[str(path.relative_to(output))] = runner.sha256(path)
    return result


def progress(stage: str, **details: Any) -> None:
    """Safe progress reporting: counts and config IDs only, never target data."""
    print(__import__("json").dumps({"stage": stage, **details}, ensure_ascii=False, sort_keys=True), flush=True)


def compact_score(score: dict[str, Any]) -> dict[str, Any]:
    """Keep every required TopK field in the user-facing private summary."""
    exact = score["exact"]
    return {
        "n": score["n"],
        "success": {f"top{k}": score["success"][f"top{k}"] for k in (1, 2, 4)},
        "success_hits": {f"top{k}": score["success"][f"top{k}_hits"] for k in (1, 2, 4)},
        "accepted": {f"top{k}": score[f"top{k}"] for k in (1, 2, 4)},
        "accepted_hits": {f"top{k}": score[f"top{k}_hits"] for k in (1, 2, 4)},
        "exact": {f"top{k}": exact[f"top{k}"] for k in (1, 2, 4)},
        "exact_hits": {f"top{k}": exact[f"top{k}_hits"] for k in (1, 2, 4)},
        "rank_score": score["rank_score"]["mean"],
        "prediction_coverage": score["prediction_coverage"],
        "rankable_truth_coverage": score["rankable_truth_coverage"],
    }


def evaluation_rows_with_known_actions(rows: list[runner.Observation],
                                       history: runner.TargetHistory) -> list[dict[str, Any]]:
    """Use only cutoff-visible actions in the one shared path-success scorer."""
    observations: list[dict[str, Any]] = []
    labels: list[dict[str, Any]] = []
    for row in rows:
        known = [action.meta for action in history.catalog_at(row.input_event_index).values()]
        observations.append({
            "position": row.position, "index": row.input_event_index,
            "ready_index": row.completed_event_index,
            "target": row.current.meta if row.current else {},
            "parent_action": row.current_parent.meta if row.current_parent else None,
            "known_actions": known,
        })
        labels.append({
            "position": row.position, "sample_id": row.sample_id,
            "truth_action": row.truth.key if row.truth else None,
            "truth_metadata": row.truth.meta if row.truth else {},
            "split": row.split, "evidence_grade": "offline_batch_all_legal_members",
            "truth_parent_metadata": row.truth_parent.meta if row.truth_parent else None,
            "prior_evidence": {
                "exact_target": row.has_prior_exact_action,
                "parent_action": row.has_prior_parent_action,
            },
        })
    return evaluation.prepare_evaluation({"observations": observations, "evaluation": labels})


def score_with_known_actions(rows: list[runner.Observation],
                             groups: list[runner.QueryGroup],
                             scorers: dict[str, Any], history: runner.TargetHistory
                             ) -> tuple[dict[str, Any], dict[str, dict[int, dict[str, Any]]],
                                        list[dict[str, Any]]]:
    if {group.row.position for group in groups} != {row.position for row in rows}:
        raise ValueError("Prediction groups do not match the scoring membership")
    audited = evaluation_rows_with_known_actions(rows, history)
    outputs = {name: runner.predictions(groups, scorer) for name, scorer in scorers.items()}
    scores = {name: evaluation.measure(audited, result) for name, result in outputs.items()}
    evaluation.require_comparable(*scores.values())
    return scores, outputs, audited


def replay_cleaning_sequence(
    train: list[runner.Observation], test: list[runner.Observation],
    history: runner.TargetHistory, events: list[dict[str, Any]], bridges: Iterable[Any] = (), *,
    input_mode: str = "full", inferred_sample_ids: frozenset[str] = frozenset(),
    inferred_weight: float = 1.0, families: tuple[str, ...] = ("rnn", "lstm", "transformer"),
    sequence_length: int = DEFAULT_SEQUENCE_LENGTH, bucket_count: int = TOKEN_BUCKETS,
    seed: int = 20260928, epochs: int = 12, score_test: bool = True,
    prepared_index: asof.RichIndex | None = None,
) -> tuple[dict[str, Any], dict[str, dict[int, dict[str, Any]]], list[dict[str, Any]], dict[str, Any]]:
    """Fit one preregistered small configuration per family and score fixed test.

    This in-memory entrypoint lets cleaning arms share the exact membership and
    evaluator.  Architecture, epoch count, and input mode are chosen before the
    held-out rows are scored.  Only sample IDs explicitly identified as
    inferred links receive a reduced training weight.
    """
    if input_mode not in {"target_only", "temporal", "hierarchy", "full"}:
        raise ValueError("Unknown input mode")
    bridges = tuple(bridges)
    if epochs < 1 or sequence_length < MIN_SEQUENCE_LENGTH or bucket_count < 2:
        raise ValueError("Invalid fixed sequence configuration")
    if not 0 <= inferred_weight <= 1:
        raise ValueError("Inferred weight must be between zero and one")
    if not set(families) <= {"rnn", "lstm", "transformer"} or not families:
        raise ValueError("Unknown sequence family")
    train_ids = {row.sample_id for row in train}
    if not inferred_sample_ids <= train_ids:
        raise ValueError("Inferred sample IDs must be actual training members")
    if train and test and max(row.completed_event_index for row in train) >= min(
            row.input_event_index for row in test):
        raise ValueError("Training labels overlap the fixed held-out period")

    index = prepared_index or asof.build_rich_index(events, history, sequence_length, bridges)
    train_rich, train_attachment_audit = attach_rich_sequences(train, index)
    test_rich, test_attachment_audit = (attach_rich_sequences(test, index)
                                        if score_test else ({}, {"heldout_not_opened": len(test)}))
    train_sequences = {position: prefix.keys for position, prefix in train_rich.items()}
    test_sequences = {position: prefix.keys for position, prefix in test_rich.items()}
    fit, calibration, selection, split_audit = runner.causal_three_way(train)
    fit_groups, fit_group_audit = runner.build_causal_training_groups(fit, history)
    fit_examples, fit_example_audit = examples_from_groups(
        fit_groups, train_sequences, bucket_count, rich_sequences=train_rich,
        rich_index=index, history=history, events=events, input_mode=input_mode,
        inferred_sample_ids=inferred_sample_ids, inferred_weight=inferred_weight)
    if not fit_examples:
        raise ValueError("No eligible causal sequence training examples")
    calibration_groups = runner.build_static_groups(
        calibration, runner.HistoryStats.from_rows(fit), history)
    selection_groups = runner.build_static_groups(
        selection, runner.HistoryStats.from_rows(fit), history)
    final_groups: list[runner.QueryGroup] = []
    final_group_audit: dict[str, int] = {}
    final_examples: list[SequenceExample] = []
    final_example_audit: dict[str, int] = {}
    if score_test:
        final_groups, final_group_audit = runner.build_causal_training_groups(train, history)
        final_examples, final_example_audit = examples_from_groups(
            final_groups, train_sequences, bucket_count, rich_sequences=train_rich,
            rich_index=index, history=history, events=events, input_mode=input_mode,
            inferred_sample_ids=inferred_sample_ids, inferred_weight=inferred_weight)
        if not final_examples:
            raise ValueError("No eligible final sequence training examples")
    configs = model_configurations(seed, sequence_length, bucket_count)
    test_groups = (runner.build_static_groups(test, runner.HistoryStats.from_rows(train), history)
                   if score_test else [])
    scorers: dict[str, Any] = {"no_context_statistical": runner.baseline_correction}
    model_audit: dict[str, Any] = {}
    for family in families:
        config = {**configs[family][0], "input_mode": input_mode, "epochs": epochs}
        inner_model, inner_training = train_sequence_model(family, config, fit_examples)
        scale, scale_audit = fit_residual_scale(
            inner_model, calibration_groups, train_sequences, bucket_count,
            rich_sequences=train_rich, rich_index=index, history=history,
            events=events, input_mode=input_mode)
        inner_residual = SequenceResidual(
            inner_model, train_sequences, bucket_count, scale,
            train_rich, index, history, events, input_mode)
        selection_scores, selection_outputs, selection_audited = score_with_known_actions(
            selection, selection_groups, {"candidate": inner_residual.correction}, history)
        selection_metric = evaluation.primary_leaderboard_metric(
            runner.score_leaderboards(selection_audited, selection_outputs), "candidate")[1]
        final_training: dict[str, Any] | None = None
        if score_test:
            final_model, final_training = train_sequence_model(family, config, final_examples)
            test_residual = SequenceResidual(
                final_model, test_sequences, bucket_count, scale,
                test_rich, index, history, events, input_mode)
            scorers[family + "_residual"] = test_residual.correction
        model_audit[family] = {
            "config": config, "inner_training": inner_training,
            "inner_scale": scale_audit, "inner_selection_metric": selection_metric,
            "inner_selection_score": compact_score(selection_scores["candidate"]),
            "final_training": final_training,
        }
    scores, outputs, audited = (score_with_known_actions(test, test_groups, scorers, history)
                                if score_test else ({}, {}, []))
    audit = {
        "input_mode": input_mode, "bridge_count": len(bridges),
        "inferred_sample_ids_count": len(inferred_sample_ids),
        "inferred_weight": inferred_weight,
        "prefix": index.audit, "train_attachment": train_attachment_audit,
        "test_attachment": test_attachment_audit,
        "inner_split": split_audit, "fit_groups": fit_group_audit,
        "fit_examples": fit_example_audit,
        "final_groups": final_group_audit,
        "final_examples": final_example_audit,
        "models": model_audit,
        "test_members": len(test), "test_scored_once": score_test,
        "test_labels_used_for_training": False,
    }
    return scores, outputs, audited, audit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sequence-length", type=int, default=DEFAULT_SEQUENCE_LENGTH)
    parser.add_argument("--token-buckets", type=int, default=TOKEN_BUCKETS)
    parser.add_argument("--seed", type=int, default=20260928)
    args = parser.parse_args()
    snapshot = args.snapshot.expanduser().resolve()
    split_dir = args.split.expanduser().resolve()
    output = args.output.expanduser().resolve()
    private = PRIVATE_ROOT.resolve()
    if args.sequence_length < MIN_SEQUENCE_LENGTH:
        parser.error("--sequence-length must be at least 2")
    if args.token_buckets < 2:
        parser.error("--token-buckets must be at least 2")
    if output == private or not output.is_relative_to(private):
        parser.error("Output must be a new private SmartApp training directory")
    if output.exists():
        parser.error("Output directory must not already exist")
    # Fail before writing any artifact if the requested offline runtime is absent.
    try:
        require_torch()
    except RuntimeError as error:
        parser.error(str(error))

    os.umask(0o077)
    output.mkdir(parents=True, mode=0o700)
    output.chmod(0o700)
    train, test, history, input_audit = runner.load_observations(snapshot, split_dir)
    events = runner.read_jsonl(snapshot / "events.jsonl")
    prefixes = build_prefix_index(events, history, args.sequence_length)
    train_attachments = attach_sequences(train, prefixes)
    fit, calibration, selection, inner_split = runner.causal_three_way(train)
    progress("input_verified", train_members=len(train), test_members=len(test), **inner_split)

    plan = {
        "version": VERSION,
        "scope": "local_offline_research_only",
        "snapshot": str(snapshot), "split": str(split_dir),
        "foreground_only": True, "candidate_lists_persisted": False,
        "runtime_or_online_model_modified": False,
        "sequence_input": {
            "maximum_confirmed_arrivals": args.sequence_length,
            "minimum_confirmed_arrivals_for_neural_correction": MIN_SEQUENCE_LENGTH,
            "source": "foreground-covered context.change only",
            "excluded_recovery_or_completion_reasons": sorted(NON_ARRIVAL_REASONS),
            "hard_boundaries": sorted(HARD_BOUNDARY_KINDS),
            "consecutive_duplicate_arrivals": "deduplicated without inventing a new arrival",
        },
        "tokenization": {"algorithm": "blake2b fixed buckets", "bucket_count": args.token_buckets,
                         "target_vocabulary_persisted": False},
        "test_protocol": "inner temporal fit/calibration/selection first; test scored exactly once after final retraining",
        "statistical_baseline": "frozen no-context global frequency prior with Laplace smoothing",
        "combination": "each neural encoder learns a residual; its nonnegative scale is fit on the causal inner calibration interval",
        "offline_update_policy": "no weight or count update during test; as-of catalog is reconstructed transiently from observed foreground target history",
        "models": ["no_context_statistical", "rnn_residual", "lstm_residual", "transformer_residual"],
        "action_semantics": "app/window→app action; page/directory/file exact action; service and prediction-time-known parent-app fallbacks are scored by the shared contract without becoming exact hits",
        "leaderboard_protocol": "keep all legal rows and use the prior-supported board as the main model-comparison board when it is nonempty",
    }
    runner.write_private_json(output / "plan.json", plan)
    runner.write_private_json(output / "input-audit.json", input_audit)
    runner.write_private_json(output / "sequence-input-audit.json", {
        "prefix_construction": prefixes.audit,
        "train_attachment": train_attachments.audit,
        "raw_target_or_locator_persisted": False,
    })
    runner.write_private_json(output / "inner-split.json", inner_split)
    source_artifacts(output)
    runner.write_private_json(output / "environment.json", environment())

    inner, _ = run_inner_selection(
        fit, calibration, selection, history, train_attachments.eligible,
        args.sequence_length, args.token_buckets, args.seed,
    )
    runner.write_private_json(output / "inner-selection.json", inner)
    progress("inner_selection_complete", selected={family: value["id"] for family, value in inner["selected"].items()})

    # Refit exactly once using all mature outer-train labels.  The calibration
    # scale remains the value selected strictly before this final fit.
    final_groups, final_group_audit = runner.build_causal_training_groups(train, history)
    final_examples, final_example_audit = examples_from_groups(
        final_groups, train_attachments.eligible, args.token_buckets
    )
    final_models: dict[str, tuple[Any, dict[str, Any], float]] = {}
    final_training: dict[str, Any] = {}
    for family, selected in inner["selected"].items():
        config = selected["config"]
        model, training = train_sequence_model(family, config, final_examples)
        scale = float(selected["frozen_inner_calibration"]["scale"])
        final_models[family] = (model, config, scale)
        final_training[family] = {"config": config, "training": training,
                                  "frozen_inner_calibration_scale": scale}
    model_hashes = save_models(output, final_models)
    runner.write_private_json(output / "final-training.json", {
        "ranker_groups": len(final_groups), "ranker_group_audit": final_group_audit,
        "sequence_examples": final_example_audit, "models": final_training,
        "model_hashes": model_hashes,
    })
    progress("final_models_fit", sequence_training_examples=len(final_examples))

    # This is intentionally the first held-out scoring path.  The test catalog
    # and ranked outputs remain in memory for scoring and paired accounting.
    progress("heldout_scoring_started", test_members=len(test))
    final_stats = runner.HistoryStats.from_rows(train)
    test_groups = runner.build_static_groups(test, final_stats, history)
    test_attachments = attach_sequences(test, prefixes)
    scorers: dict[str, Any] = {"no_context_statistical": runner.baseline_correction}
    for family, (model, _, scale) in final_models.items():
        residual = SequenceResidual(model, test_attachments.eligible, args.token_buckets, scale)
        scorers[family + "_residual"] = residual.correction
    test_scores, test_outputs, audited_test = runner.score_groups(test, test_groups, scorers)
    test_leaderboards = runner.score_leaderboards(audited_test, test_outputs)
    test_leaderboard_audit = evaluation.leaderboard_audit(audited_test)
    paired = {
        name: evaluation.paired_topk(audited_test, test_outputs["no_context_statistical"], model_output)
        for name, model_output in test_outputs.items() if name != "no_context_statistical"
    }
    candidate_audit = {
        "test_members": len(test), "test_groups_with_actions": sum(bool(group.actions) for group in test_groups),
        "test_truth_in_asof_action_catalog": sum(group.truth_index is not None for group in test_groups),
        "test_catalog_action_count": {
            "min": min((len(group.actions) for group in test_groups), default=0),
            "max": max((len(group.actions) for group in test_groups), default=0),
            "mean": sum(len(group.actions) for group in test_groups) / len(test_groups) if test_groups else 0.0,
        },
        "sequence_attachment": test_attachments.audit,
        "all_predictions_transient": True,
        "raw_predictions_or_candidate_lists_written": False,
    }
    runner.write_private_json(output / "test-metrics.json", test_scores)
    runner.write_private_json(output / "score-leaderboards.json", {
        "primary": evaluation.PRIMARY_LEADERBOARD,
        "audit": test_leaderboard_audit,
        "metrics": test_leaderboards,
    })
    runner.write_private_json(output / "paired-help-harm.json", paired)
    runner.write_private_json(output / "test-audit.json", candidate_audit)
    summary_models = {name: compact_score(score) for name, score in test_scores.items()}
    for name, pair in paired.items():
        summary_models[name]["paired_vs_no_context"] = pair
    summary = {
        "version": VERSION, "complete": True, "test_scored_once": True,
        "foreground_only": True, "candidate_lists_persisted": False,
        "raw_predictions_or_candidate_lists_written": False,
        "runtime_or_online_model_modified": False,
        "test_members_requested": len(test),
        "test_members_after_label_audit": test_scores["no_context_statistical"]["audit"],
        "comparison_key": test_scores["no_context_statistical"]["comparison_key"],
        "primary_scoreboard": evaluation.PRIMARY_LEADERBOARD,
        "scoreboard_audit": test_leaderboard_audit,
        "scoreboards": test_leaderboards,
        "models": summary_models,
        "selected_configurations": inner["selected"],
    }
    runner.write_private_json(output / "summary.json", summary)
    runner.write_private_json(output / "artifact-hashes.json", artifact_hashes(output))
    runner.write_private_json(output / "run.json", {
        "version": VERSION, "complete": True, "snapshot": str(snapshot), "split": str(split_dir),
        "input_snapshot_hashes": input_audit["snapshot_hashes"], "test_scored_once": True,
        "foreground_only": True, "candidate_lists_persisted": False,
        "raw_predictions_or_candidate_lists_written": False,
        "runtime_or_online_model_modified": False,
    })
    print(__import__("json").dumps({"status": "complete", "output": str(output), "test_members": len(test),
                                    "models": list(test_scores)}, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
