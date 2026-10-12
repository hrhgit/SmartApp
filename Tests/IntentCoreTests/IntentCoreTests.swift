import XCTest
@testable import IntentCore

final class IntentCoreTests: XCTestCase {
    private let appA = AppContext(bundleID: "test.editor", name: "Editor")
    private let appB = AppContext(bundleID: "test.browser", name: "Browser")

    private func event(_ kind: EventKind, _ time: Double, session: String = "session-a",
                       app: AppContext? = nil, data: [String: JSONValue] = [:]) -> BehaviorEvent {
        BehaviorEvent(id: "\(session)-\(kind.rawValue)-\(time)", sessionID: session,
                      timestampMs: 1_700_000_000_000 + Int64(time), monotonicMs: time,
                      kind: kind, source: "test", app: app, data: data)
    }

    func testTypedCharactersAndOptionCompositionsNeverRetainPhysicalKeys() throws {
        for key in [0, 1, 2, 3, 12, 18, 49, 50] {
            for option in [false, true] {
                let fields = KeyPrivacy.fields(keyCode: key, command: false, control: false,
                                               option: option, shift: true, repeated: false)
                XCTAssertEqual(fields["category"], .string("typing"))
                XCTAssertNil(fields["keyCode"]); XCTAssertNil(fields["key"])
            }
        }
    }

    func testExclusionsProtectSystemPromptsAndRespectBundleBoundaries() {
        XCTAssertTrue(CapturePolicy.isExcluded("com.apple.UserNotificationCenter", rules: []))
        XCTAssertTrue(CapturePolicy.isExcluded(CapturePolicy.ownBundleID, rules: []))
        XCTAssertTrue(CapturePolicy.isExcluded("com.private.editor", rules: ["com.private.*"]))
        XCTAssertFalse(CapturePolicy.isExcluded("com.privateExtra.editor", rules: ["com.private.*"]))
        XCTAssertFalse(CapturePolicy.isExcluded("com.apple.finder", rules: ["com.apple.Passwords"]))
    }

    func testScrollFeaturesKeepPointAndLineUnitsSeparate() throws {
        var builder = SampleBuilder()
        _ = builder.consume(event(.scroll, 100, app: appA, data: ["deltaY": .number(100), "deltaUnit": .string("points")]))
        _ = builder.consume(event(.scroll, 300, app: appA, data: ["deltaY": .number(-3), "deltaUnit": .string("lines")]))
        _ = builder.consume(event(.mouseDown, 1_000, app: appA))
        let samples = builder.consume(event(.appActivated, 1_100, app: appB))
        let sample = try XCTUnwrap(samples.last)
        XCTAssertEqual(sample.features["scroll_points_abs_y"], 100)
        XCTAssertEqual(sample.features["scroll_lines_abs_y"], 3)
    }

    func testShortcutAndNavigationRemainUsefulWithoutText() {
        let shortcut = KeyPrivacy.fields(keyCode: 48, command: true, control: false,
                                         option: false, shift: false, repeated: false)
        XCTAssertEqual(shortcut["category"], .string("shortcut"))
        XCTAssertEqual(shortcut["keyCode"], .number(48))
        let navigation = KeyPrivacy.fields(keyCode: 123, command: false, control: false,
                                           option: false, shift: false, repeated: true)
        XCTAssertEqual(navigation["key"], .string("left"))
        XCTAssertNil(navigation["keyCode"])
    }

    func testEventRoundTripPreservesClockAndPayloadTypes() throws {
        let original = event(.scroll, 100.125, app: appA, data: [
            "deltaY": .number(-1.5), "continuous": .bool(true), "device": .string("unknown"), "optional": .null
        ])
        XCTAssertEqual(try JSONDecoder().decode(BehaviorEvent.self, from: JSONEncoder().encode(original)), original)
    }

    func testNextApplicationIsLabelAndNeverFutureFeature() throws {
        var builder = SampleBuilder()
        _ = builder.consume(event(.sessionStart, 0))
        let pointer = event(.pointerMoved, 100, app: appA, data: ["x": .number(0.2), "y": .number(0.4)])
        _ = builder.consume(pointer)
        let trigger = event(.mouseDown, 200, app: appA)
        XCTAssertTrue(builder.consume(trigger).isEmpty)
        let outcome = event(.appActivated, 700, app: appB)
        let sample = try XCTUnwrap(builder.consume(outcome).first)
        XCTAssertEqual(sample.sourceApp, appA.bundleID); XCTAssertEqual(sample.targetApp, appB.bundleID)
        XCTAssertEqual(sample.outcomeEventID, outcome.id); XCTAssertEqual(sample.outcomeDelayMs, 500)
        XCTAssertEqual(sample.inputEventIDs, [pointer.id, trigger.id])
        XCTAssertEqual(sample.features["input_events"], 2)
        XCTAssertFalse(sample.inputEventIDs.contains(outcome.id))
    }

    func testNegativeLabelRequiresEntireFutureHorizon() throws {
        var builder = SampleBuilder()
        _ = builder.consume(event(.mouseDown, 100, app: appA))
        XCTAssertTrue(builder.consume(event(.heartbeat, 2_099, app: appA)).isEmpty)
        let sample = try XCTUnwrap(builder.consume(event(.heartbeat, 2_100, app: appA)).first)
        XCTAssertEqual(sample.targetApp, "__stay__")
        XCTAssertNil(sample.outcomeEventID)
    }

    func testExactHorizonActivationWinsOverNegative() throws {
        var builder = SampleBuilder()
        _ = builder.consume(event(.mouseDown, 100, app: appA))
        let sample = try XCTUnwrap(builder.consume(event(.appActivated, 2_100, app: appB)).first)
        XCTAssertEqual(sample.targetApp, appB.bundleID)
    }

    func testActivationAfterHorizonCannotBecomePositive() throws {
        var builder = SampleBuilder()
        _ = builder.consume(event(.mouseDown, 100, app: appA))
        let sample = try XCTUnwrap(builder.consume(event(.appActivated, 2_101, app: appB)).first)
        XCTAssertEqual(sample.targetApp, "__stay__")
    }

    func testGapsDoNotBecomeFalseNegativesOrCrossBoundaryFeatures() throws {
        var builder = SampleBuilder()
        let old = event(.mouseDown, 100, app: appA)
        _ = builder.consume(old)
        XCTAssertTrue(builder.consume(event(.captureGap, 200)).isEmpty)
        XCTAssertTrue(builder.consume(event(.heartbeat, 2_200, app: appA)).isEmpty)
        let fresh = event(.keyDown, 2_500, app: appA, data: ["category": .string("typing")])
        _ = builder.consume(fresh)
        let sample = try XCTUnwrap(builder.consume(event(.appActivated, 2_900, app: appB)).first)
        XCTAssertEqual(sample.inputEventIDs, [fresh.id])
        XCTAssertEqual(sample.features["typing_keys"], 1)
    }

    func testPausedSessionAndDifferentSessionsNeverConnect() {
        var builder = SampleBuilder()
        _ = builder.consume(event(.mouseDown, 100, app: appA))
        XCTAssertTrue(builder.consume(event(.sessionEnd, 200)).isEmpty)
        XCTAssertTrue(builder.consume(event(.appActivated, 900, app: appB)).isEmpty)
        _ = builder.consume(event(.mouseDown, 1_000, app: appA))
        XCTAssertTrue(builder.consume(event(.appActivated, 1_300, session: "session-b", app: appB)).isEmpty)
    }

    func testLongUnobservedIntervalIsDiscarded() {
        var builder = SampleBuilder()
        _ = builder.consume(event(.mouseDown, 100, app: appA))
        XCTAssertTrue(builder.consume(event(.heartbeat, 10_000, app: appA)).isEmpty)
    }

    func testLookbackAndApplicationFilterExcludeOldOrForeignInputs() throws {
        var builder = SampleBuilder()
        _ = builder.consume(event(.pointerMoved, 0, app: appA))
        _ = builder.consume(event(.pointerMoved, 2_000, app: appB))
        let current = event(.mouseDown, 3_500, app: appA)
        _ = builder.consume(current)
        let sample = try XCTUnwrap(builder.consume(event(.appActivated, 3_700, app: appB)).first)
        XCTAssertEqual(sample.inputEventIDs, [current.id])
    }

    func testTriggerThrottlingAndFirstOutcome() {
        var builder = SampleBuilder()
        _ = builder.consume(event(.scroll, 0, app: appA))
        _ = builder.consume(event(.scroll, 100, app: appA))
        _ = builder.consume(event(.scroll, 750, app: appA))
        XCTAssertEqual(builder.consume(event(.appActivated, 1_000, app: appB)).count, 2)
        XCTAssertTrue(builder.consume(event(.appActivated, 1_100, app: appA)).isEmpty)
    }

    func testPhysicalDisplayChangesDoNotCreateImpossiblePointerDistance() throws {
        var builder = SampleBuilder()
        _ = builder.consume(event(.pointerMoved, 0, app: appA, data: ["x": .number(0), "y": .number(0), "displayID": .number(1)]))
        _ = builder.consume(event(.pointerMoved, 50, app: appA, data: ["x": .number(1), "y": .number(1), "displayID": .number(2)]))
        _ = builder.consume(event(.mouseDown, 100, app: appA))
        let sample = try XCTUnwrap(builder.consume(event(.appActivated, 200, app: appB)).first)
        XCTAssertEqual(sample.features["pointer_distance"], 0)
    }

    func testStoragePersistencePermissionsAndSnapshotExport() throws {
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: directory) }
        let store = try EventStore(url: directory.appendingPathComponent("events.sqlite"))
        let events = [event(.sessionStart, 0), event(.mouseDown, 100, app: appA),
                      event(.appActivated, 200, app: appB), event(.mouseDown, 1_000, app: appB),
                      event(.heartbeat, 2_000, app: appB), event(.heartbeat, 3_000, app: appB),
                      event(.annotation, 3_100, data: ["intent": .string("阅读资料")]), event(.sessionEnd, 3_200)]
        try store.append(events)
        let stats = try store.stats()
        XCTAssertEqual(stats.events, events.count); XCTAssertEqual(stats.sessions, 1)
        XCTAssertEqual(stats.applications, 2); XCTAssertEqual(stats.annotations, 1)
        let attributes = try FileManager.default.attributesOfItem(atPath: store.url.path)
        XCTAssertEqual((attributes[.posixPermissions] as? NSNumber)?.intValue, 0o600)
        let export = directory.appendingPathComponent("export")
        let report = try store.export(to: export)
        XCTAssertEqual(report.events, events.count); XCTAssertEqual(report.samples, 2)
        XCTAssertEqual(report.switchSamples, 1); XCTAssertEqual(report.staySamples, 1)
        let samples = try readLines(TrainingSample.self, at: export.appendingPathComponent("samples.jsonl"))
        XCTAssertEqual(samples.map(\.targetApp), [appB.bundleID, "__stay__"])
        XCTAssertEqual(try readLines(BehaviorEvent.self, at: export.appendingPathComponent("events.jsonl")), events)
        XCTAssertEqual(try readLines(BehaviorEvent.self, at: export.appendingPathComponent("annotations.jsonl")).count, 1)
        let reader = try EventStore(url: store.url, readOnly: true)
        XCTAssertEqual(try reader.recent(limit: 1).first, events.last)
        XCTAssertThrowsError(try store.export(to: export))
        XCTAssertTrue(FileManager.default.fileExists(atPath: export.appendingPathComponent("manifest.json").path))
    }

    func testFailedBatchRollsBackAllEvents() throws {
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: directory) }
        let store = try EventStore(url: directory.appendingPathComponent("events.sqlite"))
        let original = event(.mouseDown, 100, app: appA)
        try store.append([original])
        XCTAssertThrowsError(try store.append([event(.heartbeat, 200), original]))
        XCTAssertEqual(try store.stats().events, 1)
    }

    func testCommittedPositionsMatchStreamingHistoryAndIgnoreFailedBatch() throws {
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: directory) }
        let store = try EventStore(url: directory.appendingPathComponent("events.sqlite"))
        let first = event(.sessionStart, 0)
        let second = event(.heartbeat, 1)
        let positions = try store.appendWithSequences([first, second])
        XCTAssertEqual(positions.count, 2)
        XCTAssertLessThan(positions[0], positions[1])
        XCTAssertThrowsError(try store.append([event(.heartbeat, 2), first]))
        var replayed: [(Int64, String)] = []
        try store.replayRecommendationHistory(seedThroughEventID: nil) { position, value in
            replayed.append((position, value.id))
        }
        XCTAssertEqual(replayed.map(\.0), positions)
        XCTAssertEqual(replayed.map(\.1), [first.id, second.id])
        struct StopReplay: Error {}
        XCTAssertThrowsError(try store.replayRecommendationHistory(seedThroughEventID: nil) { _, _ in
            throw StopReplay()
        })
        XCTAssertEqual(try store.stats().events, 2)
    }

    func testCommittedRangeUsesInclusiveUpperBoundWhileWriterCanAdvance() throws {
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: directory) }
        let store = try EventStore(url: directory.appendingPathComponent("events.sqlite"))
        let positions = try store.appendWithSequences([event(.sessionStart, 0), event(.heartbeat, 1)])
        let reader = try EventStore(url: store.url, readOnly: true)
        var replayed: [Int64] = []
        var third: Int64 = 0
        try reader.replayCommitted(after: positions[0], through: positions[1]) { sequence, _ in
            replayed.append(sequence)
            third = try XCTUnwrap(store.appendWithSequences([event(.heartbeat, 2)]).last)
        }
        XCTAssertEqual(replayed, [positions[1]])
        try reader.replayCommitted(after: positions[1], through: third) { sequence, _ in
            replayed.append(sequence)
        }
        XCTAssertEqual(replayed, [positions[1], third])
        try reader.replayCommitted(after: third, through: positions[0]) { _, _ in XCTFail("Empty range") }
    }

    func testExportDropsIncompleteTrailingSamples() throws {
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: directory) }
        let store = try EventStore(url: directory.appendingPathComponent("events.sqlite"))
        try store.append([event(.mouseDown, 100, app: appA), event(.heartbeat, 500, app: appA)])
        XCTAssertEqual(try store.export(to: directory.appendingPathComponent("export")).samples, 0)
    }

    private func readLines<T: Decodable>(_ type: T.Type, at url: URL) throws -> [T] {
        try String(contentsOf: url, encoding: .utf8).split(separator: "\n").map {
            try JSONDecoder().decode(type, from: Data($0.utf8))
        }
    }
}
