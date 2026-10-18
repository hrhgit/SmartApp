"""Prospective, common evidence cohorts for the foreground-only replay.

The cohort is fixed from the no-sequence statistical baseline's first emitted
action and mature same-origin history available at the prediction cutoff.  It
never reads the query's realized truth or whether any model was correct.  The
cohort membership is therefore shared by all models in an evaluation arm.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
import hashlib
import json
from typing import Any

from navigation_hierarchy import SERVICE, disposition, service


ACTIONABLE_KINDS = {"app", "page", "directory", "file"}
GRADES = ("strong", "repeated", "single", "none")


def grade(support: int, total: int) -> str:
    """Match the established evidence thresholds used by the project."""
    if support < 0 or total < support:
        raise ValueError("Invalid evidence counts")
    if support >= 3 and support / total >= 0.7:
        return "strong"
    if support >= 2:
        return "repeated"
    return "single" if support else "none"


def _origin_key(row: Any) -> str | None:
    current = row.observation.current
    if current is None:
        return None
    return service(current.meta) or current.key


def _label_key(row: Any) -> str | None:
    truth = row.observation.truth
    if truth is None:
        return None
    if disposition(truth.key, row.observation.current.meta) != "navigation":
        return None
    return service(truth.meta) or truth.key


def _is_navigation_training_row(row: Any) -> bool:
    observation = row.observation
    if observation.current is None or observation.truth is None:
        return False
    if observation.current.kind not in ACTIONABLE_KINDS or observation.truth.kind not in ACTIONABLE_KINDS:
        return False
    return disposition(observation.truth.key, observation.current.meta) == "navigation"


def _membership_digest(values: dict[int, str]) -> str:
    payload = json.dumps(sorted((int(position), grade_name) for position, grade_name in values.items()),
                         separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True)
class EvidenceCohort:
    grade_by_position: dict[int, str]
    support_by_position: dict[int, dict[str, int | float | None]]
    audit: dict[str, Any]


def build(rows: list[Any], training_rows: list[Any], baseline_outputs: dict[int, dict]) -> EvidenceCohort:
    """Build a common predictive evidence cohort for one replay arm.

    ``training_rows`` is the exact stream used by the replay arm: train-only
    for the frozen test, and train plus already mature test rows for the
    prequential diagnostic.
    """
    stream = [row for row in training_rows if _is_navigation_training_row(row)]
    stream.sort(key=lambda row: (row.observation.completed_event_index,
                                 row.observation.input_event_index,
                                 row.observation.position))
    queries = sorted(rows, key=lambda row: (row.observation.input_event_index,
                                            row.observation.position))
    cursor = 0
    history: dict[str, dict[str, tuple[str, str]]] = defaultdict(dict)
    grades: dict[int, str] = {}
    supports: dict[int, dict[str, int | float | None]] = {}
    for row in queries:
        cutoff = int(row.observation.input_event_index)
        while cursor < len(stream) and stream[cursor].observation.completed_event_index < cutoff:
            mature = stream[cursor]
            if mature.observation.input_event_index >= cutoff:
                raise ValueError("Future input entered prospective evidence history")
            origin = _origin_key(mature)
            label = _label_key(mature)
            if origin and label:
                outcome_id = str(mature.observation.outcome_event_id)
                history[origin].setdefault(outcome_id, (label, origin))
            cursor += 1

        position = int(row.observation.position)
        origin = _origin_key(row)
        records = list(history.get(origin, {}).values()) if origin else []
        total = len(records)
        first = (baseline_outputs.get(position) or {}).get("predictions") or []
        anchor = first[0] if first else None
        anchor_key = SERVICE if anchor == SERVICE else anchor
        support = sum(label == anchor_key for label, _ in records) if anchor_key else 0
        current_grade = grade(support, total)
        grades[position] = current_grade
        supports[position] = {
            "support": support,
            "total": total,
            "share": support / total if total else None,
            "has_origin_history": bool(total),
            "has_baseline_prediction": bool(first),
        }

    counts = Counter(grades.values())
    audit = {
        "definition": "baseline first emitted action has mature same-origin historical outcome support at the prediction cutoff",
        "grades": {name: counts.get(name, 0) for name in GRADES},
        "rows": len(rows),
        "rows_with_origin_history": sum(bool(value["total"]) for value in supports.values()),
        "rows_with_baseline_prediction": sum(bool(value["has_baseline_prediction"]) for value in supports.values()),
        "mature_training_rows_considered": len(stream),
        "membership_sha256": _membership_digest(grades),
        "uses_realized_truth_for_membership": False,
        "uses_model_hit_for_membership": False,
        "anchor": "statistics/none first emitted action",
        "thresholds": {
            "strong": "support >= 3 and support / all mature same-origin outcomes >= 0.70",
            "repeated": "support >= 2 and not strong",
            "single": "support == 1",
            "none": "support == 0",
        },
    }
    return EvidenceCohort(grades, supports, audit)
