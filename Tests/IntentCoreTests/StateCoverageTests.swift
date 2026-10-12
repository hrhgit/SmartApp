import XCTest
@testable import IntentCore

final class StateCoverageTests: XCTestCase {
    private func event(_ kind: EventKind, _ n: Int, source: String = "test", data: [String: JSONValue] = [:]) -> BehaviorEvent {
        var e = BehaviorEvent(id: "e\(n)", sessionID: "s", timestampMs: Int64(n), monotonicMs: Double(n), kind: kind, source: source, data: data)
        e.schemaVersion = 2; e.sequence = n; return e
    }
    private func registry(_ keys: [String], at n: Int = 1) -> BehaviorEvent {
        event(.sourceStatus, n, source: "source_registry", data: ["coverageVersion": .string("state-coverage-v1"),
            "expectedSources": .array(keys.map { .object(["source": .string($0), "maxAgeMs": .number(100)]) })])
    }
    private func inventory(_ source: String, at n: Int, legacy: Bool = false) -> BehaviorEvent {
        var data: [String: JSONValue] = ["entrances": .array([])]
        if !legacy { data["observationStatus"] = .string("ready") }
        return event(.inventoryChanged, n, source: source, data: data)
    }
    func testMissingAndSuccessfullyObservedEmptyAreDifferentAndInputsAreFrozen() throws {
        var builder = NavigationSampleBuilder()
        _ = builder.consume(registry(["running_apps", "ax:1"]))
        _ = builder.consume(inventory("running_apps", at: 2))
        _ = builder.consume(event(.contextChanged, 3, data: ["targetID": .string("app:a")]))
        _ = builder.consume(inventory("ax:1", at: 4))
        _ = builder.consume(event(.contextChanged, 5, data: ["targetID": .string("app:b")]))
        let samples = builder.finish()
        XCTAssertEqual(samples[0].inputCoverage?.status, "partial")
        XCTAssertEqual(samples[0].inputCoverage?.missingSources, ["ax:1"])
        XCTAssertFalse(samples[0].inventoryEventIDs.contains("e4"))
        XCTAssertEqual(samples[1].inputCoverage?.status, "complete")
        XCTAssertEqual(samples[1].inputCoverage?.sources.first { $0.source == "ax:1" }?.snapshotEventID, "e4")
    }
    func testLocalFailureKeepsHealthySourceAndInvalidatesOutcomeCoverage() {
        var builder = NavigationSampleBuilder()
        _ = builder.consume(registry(["running_apps", "browser"]))
        _ = builder.consume(inventory("running_apps", at: 2)); _ = builder.consume(inventory("browser", at: 3))
        _ = builder.consume(event(.contextChanged, 4, data: ["targetID": .string("app:a")]))
        _ = builder.consume(event(.sourceStatus, 5, source: "browser", data: ["coverageVersion": .string("state-coverage-v1"),
            "observationStatus": .string("unavailable"), "reason": .string("disconnected")]))
        _ = builder.consume(event(.contextChanged, 6, data: ["targetID": .string("app:b")]))
        let samples = builder.finish()
        XCTAssertEqual(samples[0].inputCoverage?.status, "complete")
        XCTAssertEqual(samples[1].inventoryEventIDs, ["e2"])
        XCTAssertEqual(samples[1].inputCoverage?.missingSources, ["browser"])
        XCTAssertEqual(samples[1].recentContextEventIDs, ["e4", "e6"])
    }
    func testOldEmptyInventoryAndAbsentRegistryStayUnknown() {
        var coverage = StateCoverageTracker()
        coverage.consume(inventory("running_apps", at: 1, legacy: true))
        let snapshot = coverage.snapshot(at: 2, sequence: 2)
        XCTAssertEqual(snapshot.status, "unknown")
        XCTAssertEqual(snapshot.sources.first?.status, "unknown")
        XCTAssertEqual(snapshot.sources.first?.reason, "legacy_empty_unverified")
    }
    func testUnchangedCheckFreshnessGlobalGapAndNoFutureRecovery() {
        var coverage = StateCoverageTracker()
        coverage.consume(registry(["running_apps"])); coverage.consume(inventory("running_apps", at: 2))
        XCTAssertEqual(coverage.snapshot(at: 103, sequence: 3).sources.first?.status, "stale")
        coverage.consume(event(.sourceStatus, 104, source: "running_apps", data: ["coverageVersion": .string("state-coverage-v1"),
            "observationStatus": .string("ready"), "snapshotEventID": .string("e2")]))
        XCTAssertEqual(coverage.snapshot(at: 105, sequence: 105).status, "complete")
        coverage.consume(event(.captureGap, 106, data: ["reason": .string("sleep")]))
        let beforeRecovery = coverage.snapshot(at: 107, sequence: 107)
        coverage.consume(inventory("running_apps", at: 108))
        XCTAssertEqual(beforeRecovery.status, "partial")
        XCTAssertNil(beforeRecovery.sources.first?.snapshotEventID)
        XCTAssertEqual(coverage.snapshot(at: 109, sequence: 109).status, "complete")
    }
    func testHierarchyIsResolvedAtEventTimeAndRetainsWindowWithFile() throws {
        var coverage = StateCoverageTracker()
        let app = Destination(id: "app:editor", kind: .app, name: "Editor", bundleID: "editor")
        let file = Destination(id: "file:one", kind: .file, name: "One", locator: "file:///synthetic/one.txt")
        let window = Destination(id: "window:one", kind: .window, name: "Window", bundleID: "editor")
        for (i, target) in [app, file, window].enumerated() {
            coverage.consume(event(.targetUpdated, i + 1, data: ["target": try .encoded(target)]))
        }
        let entrance = TargetEntrance(id: "entry", targetID: file.id, bundleID: "editor", windowID: window.id, kind: "ax_window", containerID: window.id)
        coverage.consume(event(.inventoryChanged, 4, data: ["entrances": try .encoded([entrance]), "observationStatus": .string("ready")]))
        let arrival = event(.contextChanged, 5, data: ["targetID": .string(file.id), "entranceID": .string(entrance.id)])
        let hierarchy = try XCTUnwrap(coverage.hierarchy(for: arrival))
        XCTAssertEqual(hierarchy.appTargetID, app.id); XCTAssertEqual(hierarchy.containerTargetID, window.id)
        XCTAssertEqual(hierarchy.locator, file.locator)
        var changed = file; changed.locator = "file:///synthetic/two.txt"
        coverage.consume(event(.targetUpdated, 6, data: ["target": try .encoded(changed)]))
        XCTAssertEqual(hierarchy.locator, "file:///synthetic/one.txt")
    }
    func testInvalidatedEntranceCannotBeRevivedByStaleReadyCheck() throws {
        var gate = TargetEventGate()
        let app = Destination(id: "app:a", kind: .app, name: "A", bundleID: "a", admission: TargetAdmission(basis: "standalone_application"))
        XCTAssertTrue(gate.accept(event(.targetUpdated, 1, data: ["target": try .encoded(app)])))
        XCTAssertTrue(gate.accept(inventory("source", at: 2)))
        XCTAssertTrue(gate.accept(event(.sourceStatus, 3, source: "source", data: ["coverageVersion": .string("state-coverage-v1"), "observationStatus": .string("unavailable")])))
        XCTAssertFalse(gate.accept(event(.sourceStatus, 4, source: "source", data: ["coverageVersion": .string("state-coverage-v1"), "observationStatus": .string("ready"), "snapshotEventID": .string("e2")])))
    }
    func testOnlineBackupIncludesWALAndExportMigrationPreservesRawEvents() throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: root) }
        let source = try EventStore(url: root.appendingPathComponent("live.sqlite"))
        let app = Destination(id: "app:a", kind: .app, name: "A", bundleID: "a", admission: TargetAdmission(basis: "standalone_application", policyVersion: "target-eligibility-v3"))
        let events = [event(.sessionStart, 0, data: ["targetEligibilityPolicyVersion": .string("target-eligibility-v3")]),
            event(.targetUpdated, 1, data: ["target": try .encoded(app)]),
            event(.contextChanged, 2, data: ["targetID": .string(app.id)])]
        try source.append(events)
        let backupURL = root.appendingPathComponent("backup.sqlite")
        try source.backup(to: backupURL)
        XCTAssertThrowsError(try source.backup(to: backupURL))
        let backup = try EventStore(url: backupURL, readOnly: true)
        XCTAssertEqual(try backup.recent().reversed().map(\.id), events.map(\.id))
        let out = root.appendingPathComponent("migration")
        let report = try backup.export(to: out)
        XCTAssertEqual(report.events, events.count); XCTAssertEqual(report.sampleSchemaVersion, 4)
        XCTAssertEqual(report.inputCoverageCounts, ["unknown": 1])
        let raw = try String(contentsOf: out.appendingPathComponent("events.jsonl"), encoding: .utf8).split(separator: "\n")
            .map { try JSONDecoder().decode(BehaviorEvent.self, from: Data($0.utf8)) }
        XCTAssertEqual(raw, events)
        XCTAssertEqual((try FileManager.default.attributesOfItem(atPath: backupURL.path)[.posixPermissions] as? NSNumber)?.intValue, 0o600)
    }
}
