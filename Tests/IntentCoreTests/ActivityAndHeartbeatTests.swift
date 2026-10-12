import XCTest
@testable import IntentCore

final class ActivityAndHeartbeatTests: XCTestCase {
    private func event(_ kind: EventKind, _ time: Double, _ data: [String: JSONValue] = [:]) -> BehaviorEvent {
        var event = BehaviorEvent(sessionID: "s", monotonicMs: time, kind: kind, source: "fixture", data: data)
        event.schemaVersion = 2
        return event
    }
    private func fixture() throws -> [BehaviorEvent] {
        var events = [event(.sessionStart, 0, ["targetEligibilityPolicyVersion": .string(TargetEligibilityPolicy.version)])]
        for name in ["a", "b", "c"] {
            let target = Destination(id: "app:" + name, kind: .app, name: name, bundleID: name,
                admission: TargetAdmission(basis: "standalone_application"))
            events.append(event(.targetUpdated, 1, ["target": try .encoded(target)]))
        }
        return events
    }
    private func arrival(_ target: String, _ time: Double, resolution: Bool = false) -> BehaviorEvent {
        event(.contextChanged, time, ["targetID": .string("app:" + target),
            "reason": .string(resolution ? "resolution_update" : "foreground_change")])
    }
    func testOnlyIndependentObservedTransitionsCount() throws {
        var counter = UserActivityCounter()
        for value in try fixture() { XCTAssertFalse(counter.consume(value)) }
        XCTAssertFalse(counter.consume(arrival("a", 10))) // baseline
        XCTAssertFalse(counter.consume(event(.heartbeat, 20)))
        XCTAssertFalse(counter.consume(event(.clipboardChanged, 21)))
        XCTAssertFalse(counter.consume(event(.appActivated, 22)))
        XCTAssertFalse(counter.consume(arrival("a", 25))) // same destination
        XCTAssertTrue(counter.consume(arrival("b", 30)))
        XCTAssertFalse(counter.consume(arrival("c", 31, resolution: true)))
        XCTAssertFalse(counter.consume(arrival("c", 32))) // refinement becomes current state
        XCTAssertFalse(counter.consume(event(.captureGap, 40, ["reason": .string("sleep")])))
        XCTAssertFalse(counter.consume(arrival("a", 50))) // recovery baseline
        XCTAssertTrue(counter.consume(arrival("b", 60)))
        XCTAssertEqual(counter.count, 2)
    }
    func testSourceFailureAndUntrustedSessionsCannotCreateOperations() throws {
        var counter = UserActivityCounter()
        for value in try fixture() { counter.consume(value) }
        counter.consume(arrival("a", 10))
        counter.consume(event(.sourceStatus, 20, ["coverageVersion": .string("state-coverage-v1"), "observationStatus": .string("unavailable")]))
        XCTAssertFalse(counter.consume(arrival("b", 30)))
        var unknown = event(.sessionStart, 40); unknown.sessionID = "unsupported"
        counter.consume(unknown)
        XCTAssertFalse(counter.consume(arrival("c", 50)))
        XCTAssertEqual(counter.count, 0)
    }
    func testStoreUsesSameCounterAcrossIncrementalReadsAndReopen() throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: root) }
        let store = try EventStore(url: root.appendingPathComponent("test.sqlite"))
        try store.append(try fixture() + [arrival("a", 10), event(.heartbeat, 20), arrival("b", 30), arrival("c", 31, resolution: true)])
        let reader = try EventStore(url: store.url, readOnly: true)
        XCTAssertEqual(try reader.stats().userActions, 1)
        XCTAssertEqual(try reader.stats().userActions, 1)
        try store.append([arrival("a", 40)])
        XCTAssertEqual(try reader.stats().userActions, 2)
        XCTAssertEqual(try EventStore(url: store.url, readOnly: true).stats().userActions, 2)
        XCTAssertGreaterThan(try reader.stats().events, 2)
    }
    func testSummaryPreservesObservedBoundsAndCannotLeakIdentityAtGap() throws {
        var summary = HeartbeatSummary()
        for time in [500.0, 1000, 1500, 2000] { summary.observe(at: time) }
        let data = try XCTUnwrap(summary.take())
        XCTAssertEqual(data["checkCount"]?.number, 4)
        XCTAssertEqual(data["firstObservedMonotonicMs"]?.number, 500)
        XCTAssertEqual(data["lastObservedMonotonicMs"]?.number, 2000)
        XCTAssertEqual(data["maximumCheckGapMs"]?.number, 500)
        XCTAssertNil(summary.take())
        var gate = TargetEventGate()
        var gap = event(.captureGap, 40_000, ["reason": .string("sleep"), "collectorHeartbeatSummary": .object(data)])
        XCTAssertTrue(gate.accept(gap))
        var builder = NavigationSampleBuilder()
        _ = builder.consume(arrival("a", 1))
        // A late boundary must not be preceded by a synthetic successful heartbeat.
        XCTAssertEqual(builder.consume(gap).first?.status, "incomplete")
        var unsafe = data; unsafe["targetID"] = .string("secret")
        gap.data["collectorHeartbeatSummary"] = .object(unsafe)
        XCTAssertFalse(gate.accept(gap))
    }
}
