from types import SimpleNamespace
import unittest

import evidence_cohort


def action(key, kind="app"):
    bundle = key.removeprefix("app:")
    return SimpleNamespace(key=key, kind=kind, meta={"id": key, "kind": kind, "bundleID": bundle})


def row(position, input_index, completed_index, outcome_id=None, destination="app:B"):
    current = action("app:A")
    truth = action(destination) if outcome_id else None
    observation = SimpleNamespace(
        position=position,
        input_event_index=input_index,
        completed_event_index=completed_index,
        outcome_event_id=outcome_id,
        current=current,
        truth=truth,
    )
    return SimpleNamespace(observation=observation)


class EvidenceCohortTests(unittest.TestCase):
    def test_uses_mature_history_and_established_thresholds(self):
        training = [row(i, i, i + 1, f"outcome-{i}") for i in range(3)]
        query = [row(10, 10, 11)]
        cohort = evidence_cohort.build(query, training, {10: {"predictions": ["app:B"]}})
        self.assertEqual(cohort.grade_by_position[10], "strong")
        self.assertEqual(cohort.support_by_position[10]["support"], 3)
        self.assertEqual(cohort.support_by_position[10]["total"], 3)
        self.assertFalse(cohort.audit["uses_realized_truth_for_membership"])

    def test_query_truth_does_not_change_membership(self):
        training = [row(1, 1, 2, "outcome-1")]
        query_a = row(10, 10, 11, destination="app:wrong")
        query_b = row(10, 10, 11, destination="app:B")
        output = {10: {"predictions": ["app:B"]}}
        cohort_a = evidence_cohort.build([query_a], training, output)
        cohort_b = evidence_cohort.build([query_b], training, output)
        self.assertEqual(cohort_a.grade_by_position, cohort_b.grade_by_position)
        self.assertEqual(cohort_a.audit["membership_sha256"], cohort_b.audit["membership_sha256"])

    def test_maturity_boundary_is_strict(self):
        training = [row(1, 1, 10, "outcome-1"), row(2, 2, 9, "outcome-2")]
        query = [row(10, 10, 11)]
        cohort = evidence_cohort.build(query, training, {10: {"predictions": ["app:B"]}})
        self.assertEqual(cohort.support_by_position[10]["support"], 1)
        self.assertEqual(cohort.support_by_position[10]["total"], 1)


if __name__ == "__main__":
    unittest.main()
