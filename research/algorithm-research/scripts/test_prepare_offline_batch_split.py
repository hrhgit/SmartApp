#!/usr/bin/env python3
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import prepare_offline_batch_split as split


def event(event_id, session, sequence, timestamp, kind="context.change", target=None):
    data = {} if target is None else {"targetID": target}
    return {"id": event_id, "sessionID": session, "sequence": sequence,
            "timestampMs": timestamp, "monotonicMs": float(timestamp), "kind": kind, "data": data}


def sample(sample_id, anchor, session, anchor_sequence, anchor_timestamp, completion_sequence, next_target):
    return {
        "id": sample_id,
        "schemaVersion": 4,
        "labelRuleVersion": split.LABEL_RULE_VERSION,
        "sessionID": session,
        "anchorEventID": anchor,
        "inputCutoffMs": float(anchor_timestamp),
        "inputThroughSequence": anchor_sequence,
        "completedThroughSequence": completion_sequence,
        "status": "observed_horizon",
        "inputCoverage": {"version": split.FOREGROUND_COVERAGE_VERSION, "status": "foreground_only"},
        "outcomeCoverage": "observed",
        "inventoryEventIDs": [],
        "origin": "natural_observed",
        "outcomes": [{"eventID": f"outcome-{sample_id}", "targetID": next_target}],
    }


class OfflineBatchSplitTests(unittest.TestCase):
    def snapshot(self, root: Path) -> Path:
        snapshot = root / "snapshot"; snapshot.mkdir()
        events = []
        samples = []
        for index in range(8):
            anchor_id = f"anchor-{index}"
            events.append(event(anchor_id, "s", index * 10 + 1, index * 10 + 1, target=f"input-{index}"))
            events.append(event(f"done-{index}", "s", index * 10 + 2, index * 10 + 2, kind="heartbeat"))
            samples.append(sample(f"sample-{index}", anchor_id, "s", index * 10 + 1,
                                  index * 10 + 1, index * 10 + 2, f"next-{index}"))
        files = {
            "events.jsonl": "".join(json.dumps(value) + "\n" for value in events),
            "target-history.jsonl": "",
            "navigation-samples.jsonl": "".join(json.dumps(value) + "\n" for value in samples),
            "manifest.json": json.dumps({"schemaVersion": 2, "sampleSchemaVersion": 4,
                                           "stateCoverageVersion": split.FOREGROUND_COVERAGE_VERSION,
                                           "events": len(events), "samples": len(samples)}),
        }
        checksums = []
        for name, content in files.items():
            path = snapshot / name; path.write_text(content)
            checksums.append(f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {name}\n")
        (snapshot / "SHA256SUMS").write_text("".join(checksums))
        return snapshot

    def test_causal_split_has_no_maturity_leakage(self):
        with tempfile.TemporaryDirectory() as directory:
            snapshot = self.snapshot(Path(directory))
            members, rejected, _, _ = split.eligible_members(snapshot)
            train, test, guard, _ = split.split_members(members, .75)
            self.assertFalse(rejected)
            self.assertTrue(train)
            self.assertTrue(test)
            self.assertLess(max(item["completedEventIndex"] for item in train),
                            min(item["inputEventIndex"] for item in test))
            self.assertEqual({item["sampleID"] for item in train + test + guard},
                             {item["sampleID"] for item in members})

    def test_rejects_non_foreground_and_self_destination_samples(self):
        with tempfile.TemporaryDirectory() as directory:
            snapshot = self.snapshot(Path(directory))
            path = snapshot / "navigation-samples.jsonl"
            rows = [json.loads(line) for line in path.read_text().splitlines()]
            rows[0]["inputCoverage"]["status"] = "complete"
            rows[1]["outcomes"][0]["targetID"] = "input-1"
            path.write_text("".join(json.dumps(row) + "\n" for row in rows))
            sums = snapshot / "SHA256SUMS"
            values = []
            for line in sums.read_text().splitlines():
                _, name = line.split(maxsplit=1)
                target = snapshot / name
                values.append(f"{hashlib.sha256(target.read_bytes()).hexdigest()}  {name}\n")
            sums.write_text("".join(values))
            _, rejected, _, _ = split.eligible_members(snapshot)
            self.assertEqual(rejected["input_not_foreground_only"], 1)
            self.assertEqual(rejected["same_destination"], 1)

    def test_boundary_crossing_horizon_is_excluded_from_both_sets(self):
        with tempfile.TemporaryDirectory() as directory:
            snapshot = self.snapshot(Path(directory))
            members, _, _, _ = split.eligible_members(snapshot)
            crossing = next(item for item in members if item["sampleID"] == "sample-5")
            crossing["completedEventIndex"] = members[6]["inputEventIndex"]
            train, test, guard, boundary = split.split_members(members, .75)
            self.assertIn(crossing, guard)
            self.assertNotIn(crossing, train)
            self.assertNotIn(crossing, test)
            self.assertGreater(boundary, crossing["inputEventIndex"])


if __name__ == "__main__":
    unittest.main()
