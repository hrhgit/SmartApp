"""One-update online path MLP over the common prediction-time candidate rows.

Every useful parent and the exact destination, when available, are alternative
actions for one confirmed arrival.  Their combined probability is optimized
once.  All features and sampled negatives come from the same shared cutoff;
no background inventory or future destination enters prediction inputs.
"""

from __future__ import annotations

from collections import Counter
import hashlib
import math
from typing import Any, Iterable

import numpy as np

import run_offline_sequence_shadow as sequence
import shared_model_adapters as adapters
import shared_model_input as shared


NUMERIC_FIELDS = (
    *("context:" + name for name in shared.COMMON_CONTEXT_COLUMNS),
    *("candidate:" + name for name in shared.CANDIDATE_NUMERIC_COLUMNS),
)
HASH_PERSON = b"SharedMLPPath1"


class SharedOnlineSetMLP:
    """Tiny incremental scorer with one normalized path-set loss per arrival.

    `scores` returns one value per shared action ID.  `learn` accepts indices
    into that *same* order, samples only other actions in that cutoff, and
    performs at most one optimizer step.  Scores are ranks, not probabilities.
    """

    def __init__(self, hidden: int, lr: float, negatives: int = 8,
                 seed: int = 20260929, *, categorical_buckets: int = 512) -> None:
        if hidden < 2 or not math.isfinite(lr) or lr <= 0 or negatives < 1:
            raise ValueError("Invalid online path MLP configuration")
        if categorical_buckets < 16:
            raise ValueError("Categorical hash space is too small")
        self.torch, nn, _ = sequence.require_torch()
        self.torch.set_num_threads(1)
        with self.torch.random.fork_rng(devices=[]):
            self.torch.manual_seed(seed)
            self.model = nn.Sequential(
                nn.Linear(len(NUMERIC_FIELDS) + categorical_buckets, hidden),
                nn.ReLU(),
                nn.Linear(hidden, 1),
            ).to("cpu")
        self.optimizer = self.torch.optim.Adam(self.model.parameters(), lr=lr)
        self.hidden = hidden
        self.learning_rate = lr
        self.seed = seed
        self.negatives = negatives
        self.categorical_buckets = categorical_buckets
        self.random = np.random.default_rng(seed)
        self.counts: Counter[str] = Counter()
        self.training_weight_sum = 0.0
        self._order_hash = hashlib.sha256()

    def _record_order(self, prediction: adapters.PredictionInput) -> str:
        if prediction.manifest.view != "mlp":
            raise ValueError("Expected a common candidate MLP input")
        joined = "\0".join(prediction.action_ids).encode("utf-8")
        digest = hashlib.sha256(joined).hexdigest()
        self._order_hash.update(bytes.fromhex(digest))
        self.counts["shared_action_orders_checked"] += 1
        return digest

    def _candidate_matrix(self, prediction: adapters.PredictionInput) -> Any:
        pairs = adapters.mlp_pair_inputs(prediction)
        if len(pairs) != len(prediction.action_ids):
            raise ValueError("MLP pair rows differ from shared candidate order")
        matrix = np.zeros((len(pairs), len(NUMERIC_FIELDS) + self.categorical_buckets),
                          dtype=np.float32)
        numeric_names = set(NUMERIC_FIELDS)
        for index, (action_id, pair) in enumerate(zip(prediction.action_ids, pairs)):
            if tuple(pair["labels"]) != (action_id,):
                raise ValueError("MLP pair action differs from shared candidate order")
            features = pair["x"]
            if numeric_names - features.keys():
                raise ValueError("A common numeric input was silently omitted")
            for column, name in enumerate(NUMERIC_FIELDS):
                matrix[index, column] = float(features[name])
            for name, value in features.items():
                if name in numeric_names:
                    continue
                if not ((name.startswith("candidate:") and "=" in name)
                        or name.startswith("candidate_layer:")):
                    raise ValueError("Undeclared MLP input field: " + name)
                number = float(value)
                if not math.isfinite(number):
                    raise ValueError("Nonfinite candidate feature")
                digest = hashlib.blake2b(name.encode("utf-8"), digest_size=8,
                                         person=HASH_PERSON).digest()
                bucket = int.from_bytes(digest, "little") % self.categorical_buckets
                sign = 1.0 if digest[0] & 1 else -1.0
                matrix[index, len(NUMERIC_FIELDS) + bucket] += sign * number
        if not np.isfinite(matrix).all():
            raise ValueError("Nonfinite shared MLP feature")
        return self.torch.from_numpy(matrix)

    def scores(self, prediction: adapters.PredictionInput) -> dict[str, float]:
        """Score every prediction-time action without reading a future label."""
        if getattr(prediction, "after_feedback_only", False):
            raise ValueError("A feedback-only exact action cannot enter prediction")
        self._record_order(prediction)
        features = self._candidate_matrix(prediction)
        self.model.eval()
        with self.torch.no_grad():
            values = self.model(features).squeeze(-1).cpu().numpy()
        if values.shape != (len(prediction.action_ids),) or not np.isfinite(values).all():
            raise ValueError("Online path MLP did not score every shared action")
        self.counts["prediction_calls"] += 1
        self.counts["prediction_actions_scored"] += len(prediction.action_ids)
        return {action: float(value) for action, value in zip(prediction.action_ids, values)}

    def learn(self, prediction: adapters.PredictionInput,
              useful_indices: Iterable[int], *, weight: float = 1.0) -> dict[str, Any]:
        """One set-likelihood update, with no duplicated parent-positive visit."""
        order_hash = self._record_order(prediction)
        positive = tuple(useful_indices)
        count = len(prediction.action_ids)
        feedback_only = bool(getattr(prediction, "after_feedback_only", False))
        original_count = getattr(prediction, "original_candidate_count", None)
        if feedback_only:
            if (type(original_count) is not int or not 0 <= original_count <= count
                    or count - original_count not in (0, 1)):
                raise ValueError("Feedback-only training has an invalid original action boundary")
            if count > original_count and count - 1 not in positive:
                raise ValueError("Appended exact leaf must be a useful training action")
        elif original_count is not None and original_count != count:
            raise ValueError("Prediction-time training cannot contain extra candidates")
        negative_boundary = original_count if feedback_only else count
        if not positive or len(set(positive)) != len(positive):
            raise ValueError("One arrival needs distinct useful action indices")
        if any(type(index) is not int or not 0 <= index < count for index in positive):
            raise ValueError("Useful action index is outside this shared cutoff")
        if not math.isfinite(weight) or not 0 <= weight <= 1:
            raise ValueError("Invalid independent-arrival weight")
        if weight == 0:
            self.counts["zero_weight_arrivals_skipped"] += 1
            return {"updated": False, "reason": "zero_weight",
                    "useful_action_count": len(positive),
                    "candidate_order_sha256": order_hash}
        # The appended exact leaf is confirmed only after the decision.  It
        # may be a positive training target, never an original-cutoff negative.
        negative_pool = [index for index in range(negative_boundary)
                         if index not in positive]
        if not negative_pool:
            self.counts["arrivals_without_negative_action"] += 1
            return {"updated": False, "reason": "no_negative_action",
                    "useful_action_count": len(positive),
                    "candidate_order_sha256": order_hash}
        sampled_negative = self.random.choice(
            negative_pool, size=min(self.negatives, len(negative_pool)), replace=False)
        selected = tuple(sorted((*positive, *(int(index) for index in sampled_negative))))
        positive_local = [selected.index(index) for index in positive]
        features = self._candidate_matrix(prediction)[list(selected)]
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        logits = self.model(features).squeeze(-1)
        loss = (logits.logsumexp(dim=0)
                - logits[positive_local].logsumexp(dim=0)) * weight
        if not self.torch.isfinite(loss):
            raise ValueError("Nonfinite online path MLP loss")
        loss.backward()
        self.torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
        self.optimizer.step()
        self.counts["optimizer_updates"] += 1
        self.counts["confirmed_arrivals_trained"] += 1
        self.training_weight_sum += weight
        self.counts["useful_action_labels_seen"] += len(positive)
        self.counts["negative_actions_sampled"] += len(sampled_negative)
        self.counts["training_actions_scored"] += len(selected)
        if feedback_only:
            self.counts["after_feedback_exact_updates"] += 1
        return {
            "updated": True,
            "after_feedback_only": feedback_only,
            "original_candidate_count": negative_boundary,
            "useful_action_count": len(positive),
            "sampled_negative_count": len(sampled_negative),
            "sampled_negative_indices": tuple(int(index) for index in sampled_negative),
            "sampled_action_count": len(selected),
            "candidate_action_count": count,
            "candidate_order_sha256": order_hash,
            "loss": float(loss.detach().cpu()),
        }

    def audit(self) -> dict[str, Any]:
        """Counts and candidate-order digest, without persisting action IDs."""
        return {
            "objective": "one_set_likelihood_per_confirmed_arrival",
            "feature_source": "shared_candidate_mlp_view",
            "numeric_feature_count": len(NUMERIC_FIELDS),
            "categorical_hash_buckets": self.categorical_buckets,
            "hidden": self.hidden,
            "learning_rate": self.learning_rate,
            "seed": self.seed,
            "negative_limit_per_update": self.negatives,
            "training_weight_sum": self.training_weight_sum,
            "candidate_order_stream_sha256": self._order_hash.hexdigest(),
            **dict(sorted(self.counts.items())),
        }
