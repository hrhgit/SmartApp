import XCTest
@testable import IntentCore

final class KnownInactivityTests: XCTestCase {
    private let pauseID = "5B8FAB08-C213-4105-B28B-48BB173D07B4"
    private func event(_ kind: EventKind, _ time: Double, id: String = UUID().uuidString,
                       data: [String: JSONValue] = [:]) -> BehaviorEvent {
        BehaviorEvent(id: id, sessionID: "s", timestampMs: Int64(time), monotonicMs: time,
                      kind: kind, source: "fixture", data: data)
    }
    private func pause(_ phase: String, _ time: Double) -> BehaviorEvent {
        event(.captureGap, time, data: ["reason": .string("known_inactivity_" + phase),
            "pauseVersion": .string("known-inactivity-v1"), "pauseID": .string(pauseID),
            "pausePhase": .string(phase), "pauseReasons": .array([.string("system_sleep")])])
    }
    private func visit(_ id: String, _ target: String, _ time: Double, resume: String? = nil) -> BehaviorEvent {
        var data: [String: JSONValue] = ["targetID": .string("app:" + target)]
        if let resume {
            data["reason"] = .string("resolution_update"); data["resolvesContextEventID"] = .string(resume)
            data["continuityPauseID"] = .string(pauseID)
        }
        return event(.contextChanged, time, id: id, data: data)
    }
    func testPairedSleepExcludesTimeWithoutChangingRawCutoffsOrAddingArrivals() {
        var builder = NavigationSampleBuilder()
        _ = builder.consume(visit("a", "a", 1_000))
        _ = builder.consume(visit("b", "b", 2_000))
        XCTAssertTrue(builder.consume(pause("begin", 2_500)).isEmpty)
        XCTAssertTrue(builder.consume(event(.heartbeat, 3_000_000)).isEmpty)
        XCTAssertTrue(builder.consume(pause("end", 3_602_500)).isEmpty)
        _ = builder.consume(visit("resume", "b", 3_603_000, resume: "b"))
        _ = builder.consume(visit("c", "c", 3_603_500))
        XCTAssertTrue(builder.consume(event(.heartbeat, 3_630_999)).isEmpty)
        let sample = builder.consume(event(.heartbeat, 3_631_000)).first
        XCTAssertEqual(sample?.anchorEventID, "a")
        XCTAssertEqual(sample?.inputCutoffMs, 1_000)
        XCTAssertEqual(sample?.activeInputCutoffMs, 1_000)
        XCTAssertEqual(sample?.outcomes.map(\.eventID), ["b", "c"])
        XCTAssertEqual(sample?.outcomes.last?.offsetMs, 3_602_500)
        XCTAssertEqual(sample?.outcomes.last?.activeOffsetMs, 2_500)
        XCTAssertEqual(sample?.completedAtMonotonicMs, 3_631_000)
        XCTAssertEqual(sample?.resolutionEventIDs, ["resume"])
    }
    func testMissingEndChangedTargetAndPrivacyGapNeverBridge() {
        for scenario in ["missing_end", "different_target", "privacy"] {
            var clock = KnownInactivityTracker()
            _ = clock.consume(visit("a", "a", 0))
            XCTAssertEqual(clock.consume(pause("begin", 100)), .suspended)
            if scenario != "missing_end" { _ = clock.consume(pause("end", 10_100)) }
            if scenario == "privacy" {
                XCTAssertEqual(clock.consume(event(.captureGap, 10_150, data: ["reason": .string("privacy")])), .boundary)
                XCTAssertEqual(clock.consume(visit("resume", "a", 10_200, resume: "a")), .normal)
            } else {
                XCTAssertEqual(clock.consume(visit("resume", scenario == "different_target" ? "b" : "a", 10_200, resume: "a")), .boundary)
            }
        }
    }
    func testOnlineLearningDoesNotTreatSleepAsLongDwell() throws {
        var learner = StreamingSemanticLearner()
        _ = learner.consume(event(.sessionStart, 0, data: ["targetEligibilityPolicyVersion": .string(TargetEligibilityPolicy.version)]))
        for name in ["a", "b", "c"] {
            let target = Destination(id: "app:" + name, kind: .app, name: name, bundleID: name,
                                     admission: TargetAdmission(basis: "standalone_application"))
            _ = learner.consume(event(.targetUpdated, 1, data: ["target": try .encoded(target)]))
        }
        for e in [visit("a", "a", 1_000), visit("b", "b", 2_000), pause("begin", 2_500),
                  pause("end", 3_602_500), visit("resume", "b", 3_603_000, resume: "b"),
                  visit("c", "c", 3_603_500)] { XCTAssertTrue(learner.consume(e).isEmpty) }
        let learned = learner.consume(event(.heartbeat, 3_631_000))
        XCTAssertEqual(learned.map(\.origin), ["app:a"])
        XCTAssertEqual(learned.map(\.label), ["app:c"])
    }
    func testIncompletePauseRemainsIncompleteAndMarkersCannotContainIdentity() {
        var builder = NavigationSampleBuilder()
        _ = builder.consume(visit("a", "a", 0)); _ = builder.consume(pause("begin", 100))
        XCTAssertTrue(builder.consume(event(.heartbeat, 999_999)).isEmpty)
        XCTAssertEqual(builder.finish().first?.status, "incomplete")
        var gate = TargetEventGate()
        XCTAssertTrue(gate.accept(pause("begin", 100)))
        var invalid = pause("begin", 100)
        invalid.data["targetID"] = .string("private")
        XCTAssertFalse(gate.accept(invalid))
        invalid = pause("begin", 100); invalid.data["pauseReasons"] = .array([.string("private-app")])
        XCTAssertFalse(gate.accept(invalid))
    }
    func testExcludedScopeMarkerCannotCarryTargetIdentity() {
        var gate = TargetEventGate()
        let id = UUID().uuidString
        let marker = event(.observationScope, 100, data: [
            "scopeVersion": .string("excluded-foreground-v1"), "scopeID": .string(id),
            "phase": .string("begin"), "category": .string("excluded_app")])
        var allowed = marker; allowed.source = "lifecycle"
        XCTAssertTrue(gate.accept(allowed))
        var invalid = allowed; invalid.data["targetID"] = .string("private")
        XCTAssertFalse(gate.accept(invalid))
        invalid = allowed; invalid.data["category"] = .string("private-app-name")
        XCTAssertFalse(gate.accept(invalid))
    }
}
