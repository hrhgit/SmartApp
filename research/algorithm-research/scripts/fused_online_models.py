"""Online reciprocal-rank fusion of two independently updated scorers.

The wrapper adds no features or evidence and does not inspect an observed answer.
Both components see each mature transition immediately.  The replay still owns
label maturity, actual-arrival deduplication, candidate resolution, and hierarchy
suppression after fusion.

For parity with a concrete-candidate replay, query['candidate_key_order'] may map
each semantic label to its currently resolved concrete candidate key.  Exact
component-score ties then use those keys; otherwise semantic labels are used.
The mapping is current prediction-time state, not a future window identifier.
"""

import math


def _rank_scores(scores, tie_keys):
    if any(not math.isfinite(value) for value in scores.values()):
        raise ValueError('component returned a non-finite score')
    # Zero and negative component scores are valid ranks (e.g. linear margins).
    # Omission alone means no vote; never turn a returned zero into abstention.
    return {label: rank for rank, label in enumerate(sorted(
        scores, key=lambda label: (-scores[label], tie_keys.get(label, label), label)), start=1)}


class FusedModel:
    """Weighted RRF; score=(1-w)/(c+rank_left)+w/(c+rank_right)."""

    def __init__(self, left, right, right_weight, rank_offset=0):
        right_weight = float(right_weight)
        rank_offset = float(rank_offset)
        if not math.isfinite(right_weight) or not 0 <= right_weight <= 1:
            raise ValueError('right_weight must be finite and between 0 and 1')
        if not math.isfinite(rank_offset) or rank_offset < 0:
            raise ValueError('rank_offset must be finite and nonnegative')
        if left is right:
            raise ValueError('fusion requires two independent component instances')
        self.left = left
        self.right = right
        self.right_weight = right_weight
        self.rank_offset = rank_offset

    def learn(self, record):
        return self.left.learn(record), self.right.learn(record)

    def scores(self, query):
        tie_keys = query.get('candidate_key_order', {})
        left = _rank_scores(self.left.scores(query), tie_keys)
        right = _rank_scores(self.right.scores(query), tie_keys)
        result = {}
        for label in sorted(left.keys() | right.keys()):
            value = ((1 - self.right_weight) / (self.rank_offset + left[label])
                     if label in left else 0.0)
            value += (self.right_weight / (self.rank_offset + right[label])
                      if label in right else 0.0)
            # Retain all returned keys even at endpoint weights 0 or 1, exactly
            # matching rank-list union fusion. A zero final weight ranks last.
            result[label] = value
        return result
