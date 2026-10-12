import XCTest
@testable import IntentCore

final class ForegroundOnlyDatabaseRebuilderTests: XCTestCase {
    private func event(_ kind: EventKind, session: String, sequence: Int, source: String = "fixture",
                       app: AppContext? = nil, data: [String: JSONValue] = [:]) -> BehaviorEvent {
        var value = BehaviorEvent(id: "\(session)-\(sequence)", sessionID: session,
                                  timestampMs: Int64(sequence), monotonicMs: Double(sequence),
                                  kind: kind, source: source, app: app, data: data)
        value.schemaVersion = 2; value.sequence = sequence
        return value
    }

    private func foregroundStart(_ session: String, clipboardPolicy: Bool = false) -> BehaviorEvent {
        var data: [String: JSONValue] = [
            "stateCoverageVersion": .string(StateCoverage.foregroundOnlyVersion),
            "targetEligibilityPolicyVersion": .string(TargetEligibilityPolicy.version),
            "observationPolicy": .string("foreground_destinations_only_no_background_state"),
            "backgroundStateCollection": .bool(false)
        ]
        if clipboardPolicy { data["clipboardPolicy"] = .string("types_only") }
        return event(.sessionStart, session: session, sequence: 1, source: "lifecycle", data: data)
    }

    private func app(_ bundle: String = "fixture.editor") -> Destination {
        Destination(id: "app:" + bundle, kind: .app, name: "Fixture Editor", bundleID: bundle,
                    admission: TargetAdmission(basis: "standalone_application"))
    }

    func testRebuildKeepsOnlyCompleteForegroundSessionsAndCanonicalFields() throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: root) }
        let sourceURL = root.appendingPathComponent("source.sqlite")
        let source = try EventStore(url: sourceURL)
        let target = app()

        let legacySession = "legacy"
        let legacy = [
            event(.sessionStart, session: legacySession, sequence: 1, source: "lifecycle",
                  data: ["stateCoverageVersion": .string(StateCoverage.inventoryVersion)]),
            event(.inventoryChanged, session: legacySession, sequence: 2,
                  data: ["entrances": .array([]), "browserState": .object(["windows": .array([])])]),
            event(.sessionEnd, session: legacySession, sequence: 3, source: "lifecycle")
        ]

        let retainedSession = "foreground-complete"
        let foreground = [
            foregroundStart(retainedSession, clipboardPolicy: true),
            event(.clipboardChanged, session: retainedSession, sequence: 2, source: "pasteboard_metadata",
                  data: ["types": .array([.string("text")])]),
            event(.targetUpdated, session: retainedSession, sequence: 3, source: "target_catalog",
                  data: ["target": try .encoded(target)]),
            event(.contextChanged, session: retainedSession, sequence: 4, source: "workspace",
                  app: .init(bundleID: target.bundleID!, name: target.name),
                  data: ["targetID": .string(target.id), "reason": .string("foreground_change")]),
            event(.sessionEnd, session: retainedSession, sequence: 5, source: "lifecycle")
        ]

        let incompleteSession = "foreground-incomplete"
        let incomplete = [
            foregroundStart(incompleteSession),
            event(.targetUpdated, session: incompleteSession, sequence: 2, source: "target_catalog",
                  data: ["target": try .encoded(target)]),
            event(.contextChanged, session: incompleteSession, sequence: 3, source: "workspace",
                  app: .init(bundleID: target.bundleID!, name: target.name),
                  data: ["targetID": .string(target.id), "reason": .string("foreground_change")])
        ]
        try source.append(legacy + foreground + incomplete)
        let sourceIDs = try source.recommendationHistory(seedThroughEventID: nil).map(\.id)

        let outputURL = root.appendingPathComponent("foreground.sqlite")
        let report = try ForegroundOnlyDatabaseRebuilder.rebuild(source: sourceURL, destination: outputURL)
        XCTAssertEqual(report.sourceEvents, legacy.count + foreground.count + incomplete.count)
        XCTAssertEqual(report.sourceSessions, 3)
        XCTAssertEqual(report.retainedSessions, 1)
        XCTAssertEqual(report.retainedEvents, 4)
        XCTAssertEqual(report.discardedSessions, 2)
        XCTAssertEqual(report.discardedByKind[EventKind.clipboardChanged.rawValue], 1)
        XCTAssertEqual(report.removedFields["clipboardPolicy"], 1)
        XCTAssertGreaterThan(report.databaseBytes, 0)

        let output = try EventStore(url: outputURL, readOnly: true)
        let events = try output.recommendationHistory(seedThroughEventID: nil)
        XCTAssertEqual(events.map(\.sessionID), Array(repeating: retainedSession, count: 4))
        XCTAssertEqual(events.map(\.kind), [.sessionStart, .targetUpdated, .contextChanged, .sessionEnd])
        XCTAssertNil(events.first?.data["clipboardPolicy"])
        XCTAssertFalse(events.contains { $0.kind == .inventoryChanged || $0.kind == .clipboardChanged })
        XCTAssertEqual(try EventStore(url: sourceURL, readOnly: true).recommendationHistory(seedThroughEventID: nil).map(\.id), sourceIDs)
    }

    func testRebuildRefusesAForegroundSessionThatContainsBackgroundState() throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: root) }
        let sourceURL = root.appendingPathComponent("source.sqlite")
        let source = try EventStore(url: sourceURL)
        let session = "invalid-foreground"
        try source.append([
            foregroundStart(session),
            event(.sourceStatus, session: session, sequence: 2, source: "fixture",
                  data: ["browserState": .object(["windows": .array([])])]),
            event(.sessionEnd, session: session, sequence: 3, source: "lifecycle")
        ])

        let outputURL = root.appendingPathComponent("foreground.sqlite")
        XCTAssertThrowsError(try ForegroundOnlyDatabaseRebuilder.rebuild(source: sourceURL, destination: outputURL))
        XCTAssertFalse(FileManager.default.fileExists(atPath: outputURL.path))
    }
}
