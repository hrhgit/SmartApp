#!/usr/bin/env python3
"""Back up the live WAL database and materialize verified, as-of v3 training samples.

This changes neither historical events nor immutable exports. Unknown observations stay unknown.
"""
import argparse
import collections
import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import sqlite3
import subprocess
import uuid


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def lines(path):
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def write_json(path, value):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    path.chmod(0o600)


def verify(database, output, exports):
    connection = sqlite3.connect(database.as_uri() + "?mode=ro", uri=True)
    try:
        assert connection.execute("PRAGMA quick_check").fetchone()[0] == "ok"
        original = [json.loads(row[0]) for row in connection.execute("SELECT json FROM events ORDER BY seq")]
    finally:
        connection.close()
    raw = list(lines(output / "events.jsonl"))
    assert raw == original, "Migration changed or omitted historical raw events"
    events = {event["id"]: event for event in raw}
    assert len(events) == len(raw), "Duplicate event IDs"
    assert list(lines(output / "target-history.jsonl")) == [e for e in raw if e["kind"] == "target.update"]
    previous_exports = []
    for path in sorted(exports.glob("*/events.jsonl")):
        checksums = path.parent / "SHA256SUMS"
        if checksums.exists():
            for row in checksums.read_text().splitlines():
                checksum, filename = row.split(maxsplit=1)
                target = path.parent / filename.lstrip("*")
                assert target.parent == path.parent and digest(target) == checksum, "Prior export checksum mismatch"
        count = 0
        for old in lines(path):
            assert events.get(old["id"]) == old, "Prior export contains history absent from backup; preserve and reconcile before publishing"
            count += 1
        previous_exports.append({"path": str(path.parent), "events": count, "sha256": digest(path)})

    samples = list(lines(output / "navigation-samples.jsonl"))
    inputs, outcomes = collections.Counter(), collections.Counter()
    empty_inputs = 0
    resolution_count = 0
    hierarchy_counts = collections.Counter()
    for sample in samples:
        assert sample["schemaVersion"] == 3
        anchor = events[sample["anchorEventID"]]
        assert anchor["data"].get("reason") != "resolution_update"
        assert sample["inputThroughSequence"] == anchor["sequence"]
        assert sample["inputCutoffMs"] == anchor["monotonicMs"]
        assert sample["sessionID"] == anchor["sessionID"]
        coverage = sample["inputCoverage"]
        assert coverage["cutoffSequence"] == anchor["sequence"]
        assert coverage["cutoffMs"] == anchor["monotonicMs"]
        references = sample["recentContextEventIDs"] + sample["activityContextEventIDs"] + sample["inventoryEventIDs"]
        if coverage.get("registryEventID"):
            references.append(coverage["registryEventID"])
        for source in coverage["sources"]:
            for key in ("snapshotEventID", "statusEventID"):
                if source.get(key):
                    references.append(source[key])
        for reference in references:
            event = events[reference]
            assert event["sessionID"] == sample["sessionID"]
            assert event["sequence"] <= sample["inputThroughSequence"], "Future reference leaked into inputs"
            assert event["monotonicMs"] <= sample["inputCutoffMs"]
        for outcome in sample["outcomes"]:
            event = events[outcome["eventID"]]
            assert event["sessionID"] == sample["sessionID"] and event["sequence"] > anchor["sequence"]
            assert event["data"].get("reason") != "resolution_update", "Resolution counted as independent behavior"
            assert 0 <= outcome["offsetMs"] < 30_000
        for reference in sample.get("resolutionEventIDs", []):
            event = events[reference]
            assert event["sessionID"] == sample["sessionID"] and event["sequence"] > anchor["sequence"]
            assert event["data"].get("reason") == "resolution_update"
            resolution_count += 1
        inputs[coverage["status"]] += 1
        outcomes[sample.get("outcomeCoverage", "unknown")] += 1
        if not sample["inventoryEventIDs"]:
            empty_inputs += 1
            assert coverage["status"] != "complete"
        hierarchy = sample.get("hierarchy", {})
        for key in ("appTargetID", "containerTargetID", "locator"):
            if hierarchy.get(key):
                hierarchy_counts[key] += 1
    return {"events": len(raw), "sessions": len({e["sessionID"] for e in raw}), "samples": len(samples),
            "inputCoverage": dict(inputs), "outcomeCoverage": dict(outcomes), "noInventorySamples": empty_inputs,
            "resolutionReferencesSeparated": resolution_count, "hierarchyCoverage": dict(hierarchy_counts),
            "priorExportsVerified": previous_exports, "rawEventsUnchanged": True,
            "allReferencesBeforeCutoff": True, "resolutionExcludedFromActions": True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cli", type=Path, default=Path(__file__).resolve().parents[3] / "build/smartapp-cli")
    parser.add_argument("--data-root", type=Path, default=Path.home() / "Library/Application Support/SmartApp")
    args = parser.parse_args()
    os.umask(0o077)
    root = args.data_root.resolve()
    database = root / "context-v2.sqlite"
    assert database.is_file(), "No context-v2 database"
    tag = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + str(uuid.uuid4())
    backup = root / "backups" / ("state-v3-" + tag)
    output = root / "training" / ("state-v3-migration-" + tag)
    backup.mkdir(parents=True, mode=0o700, exist_ok=False)
    subprocess.run([str(args.cli), "backup", "--database", str(database), "--output", str(backup / "context-v2.sqlite")], check=True, capture_output=True)
    for filename in ("identity-v2.key", "registered-targets.json"):
        source = root / filename
        if source.exists():
            shutil.copy2(source, backup / filename)
            (backup / filename).chmod(0o600)
            assert digest(source) == digest(backup / filename)
    subprocess.run([str(args.cli), "migrate-state", "--database", str(backup / "context-v2.sqlite"), "--output", str(output)], check=True, capture_output=True)
    metrics = verify(backup / "context-v2.sqlite", output, root / "exports/context-v2")
    manifest = {"migrationVersion": "state-v3-v1", "sourceDatabase": str(database), "backupDirectory": str(backup),
                "backupSHA256": digest(backup / "context-v2.sqlite"), "cliSHA256": digest(args.cli),
                "rawHistoryRewritten": False, "missingObservationsBackfilled": False,
                "verification": metrics}
    write_json(output / "migration.json", manifest)
    with (output / "migration-report.md").open("x", encoding="utf-8") as stream:
        stream.write("# 状态数据迁移验证\n\n")
        stream.write(f"已验证迁移 {metrics['events']:,} 条原始事件、{metrics['sessions']} 个会话，生成 {metrics['samples']:,} 条 v3 样本。\n\n")
        stream.write("原始事件逐条比对一致；既有导出已校验且全部包含在迁移数据内。应用仍使用原数据库，未来导出统一使用新版回放规则。\n\n")
        stream.write(f"无库存引用的样本：{metrics['noInventorySamples']}，均未标为完整输入。\n\n")
        stream.write(f"输入覆盖：`{json.dumps(metrics['inputCoverage'], ensure_ascii=False)}`。旧会话没有逐来源完整性证明，因此保留为 unknown；这不等于所有已采状态不可用。\n\n")
        stream.write(f"后续覆盖：`{json.dumps(metrics['outcomeCoverage'], ensure_ascii=False)}`。补全观测已移出独立行为标签，保留其事件引用。\n\n")
        stream.write("输入引用全部早于或等于各自截点；应用、窗口和具体目的地关系按当时的记录还原。历史未采到的状态与文件地址没有编造或回填。\n\n")
        stream.write(f"可恢复备份：`{backup}`。数据库使用 SQLite 在线备份并通过完整性检查，附身份密钥及存在时的登记文件。\n")
    filenames = ["events.jsonl", "target-history.jsonl", "navigation-samples.jsonl", "manifest.json", "migration.json", "migration-report.md"]
    with (output / "SHA256SUMS").open("x") as stream:
        for name in filenames:
            stream.write(f"{digest(output / name)}  {name}\n")
    pointer = root / "training/current-state-dataset.json"
    if pointer.exists():
        shutil.copy2(pointer, output / "previous-dataset-pointer.json")
    temporary = pointer.with_name(".current-state-dataset-" + tag + ".json")
    write_json(temporary, {"schemaVersion": 3, "migrationVersion": "state-v3-v1", "directory": str(output), "verified": True})
    os.replace(temporary, pointer)
    print(json.dumps({"output": str(output), "backup": str(backup), "verification": metrics}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
