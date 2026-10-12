import XCTest
import Carbon
import IntentCore
@testable import SmartApp

final class ContinuityAndLoginTests: XCTestCase {
    func testDisplayWakeDoesNotCancelSleepOrLock() {
        var state = SystemObservationState()
        _ = state.update(.displaySleep, active: true)
        XCTAssertFalse(state.blocksObservation)
        _ = state.update(.systemSleep, active: true)
        _ = state.update(.screenLocked, active: true)
        _ = state.update(.displaySleep, active: false)
        XCTAssertTrue(state.blocksObservation)
        _ = state.update(.systemSleep, active: false)
        XCTAssertTrue(state.blocksObservation)
        XCTAssertTrue(state.permitsContinuity)
        _ = state.update(.screenLocked, active: false)
        XCTAssertFalse(state.blocksObservation)
        _ = state.update(.sessionInactive, active: true)
        XCTAssertFalse(state.permitsContinuity)
    }
    func testLoginConsentRequiresAllConditionsAndExplicitCLIStillWorks() {
        for login in [false, true] { for enabled in [false, true] { for consent in [false, true] {
            XCTAssertEqual(LaunchContext.shouldCollect(isLogin: login, loginEnabled: enabled, consent: consent,
                                                       explicitlyCollect: false), login && enabled && consent)
        } } }
        XCTAssertTrue(LaunchContext.shouldCollect(isLogin: false, loginEnabled: false, consent: false, explicitlyCollect: true))
        XCTAssertFalse(LaunchContext.isLogin(nil))
        let event = NSAppleEventDescriptor(eventClass: kCoreEventClass, eventID: kAEOpenApplication,
            targetDescriptor: nil, returnID: AEReturnID(kAutoGenerateReturnID), transactionID: AETransactionID(kAnyTransactionID))
        XCTAssertFalse(LaunchContext.isLogin(event))
        event.setParam(NSAppleEventDescriptor(enumCode: keyAELaunchedAsLogInItem), forKeyword: keyAEPropData)
        XCTAssertTrue(LaunchContext.isLogin(event))
    }
    @MainActor func testRecorderPersistsPairAndResumesWithoutExtraSwitch() throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        let suite = "test." + UUID().uuidString
        let defaults = try XCTUnwrap(UserDefaults(suiteName: suite))
        defer { defaults.removePersistentDomain(forName: suite); try? FileManager.default.removeItem(at: root) }
        let recorder = RecorderModel(dataDirectory: root, defaults: defaults, liveObservation: false)
        XCTAssertFalse(recorder.collectOnLogin)
        recorder.collectOnLogin = true
        XCTAssertTrue(defaults.bool(forKey: "collectOnLoginV1"))
        recorder.startAtLaunch(isLogin: false)
        XCTAssertFalse(recorder.isCollecting)
        recorder.start()
        let a = Destination(id: "app:fixture.a", kind: .app, name: "A", bundleID: "fixture.a",
                            admission: TargetAdmission(basis: "standalone_application"))
        let b = Destination(id: "app:fixture.b", kind: .app, name: "B", bundleID: "fixture.b",
                            admission: TargetAdmission(basis: "standalone_application"))
        XCTAssertTrue(recorder.upsert(a)); XCTAssertTrue(recorder.upsert(b))
        recorder.emitContext(targetID: a.id, entranceID: nil, source: "fixture", reason: "foreground_change")
        recorder.systemObservationChanged(.displaySleep, active: true)
        XCTAssertNil(recorder.suspension)
        recorder.systemObservationChanged(.systemSleep, active: true)
        XCTAssertNotNil(recorder.suspension)
        recorder.systemObservationChanged(.screenLocked, active: true)
        recorder.systemObservationChanged(.displaySleep, active: false)
        XCTAssertNotNil(recorder.suspension)
        recorder.systemObservationChanged(.systemSleep, active: false)
        XCTAssertNotNil(recorder.suspension)
        recorder.systemObservationChanged(.screenLocked, active: false)
        XCTAssertNil(recorder.suspension)
        recorder.emitContext(targetID: a.id, entranceID: nil, source: "fixture", reason: "foreground_change")
        XCTAssertEqual(recorder.sessionSwitches, 0)
        recorder.emitContext(targetID: b.id, entranceID: nil, source: "fixture", reason: "foreground_change")
        XCTAssertEqual(recorder.sessionSwitches, 1)
        recorder.shutdown()
        let store = try EventStore(url: root.appendingPathComponent("context-v2.sqlite"))
        let output = root.appendingPathComponent("export")
        _ = try store.export(to: output)
        let events = try String(contentsOf: output.appendingPathComponent("events.jsonl"), encoding: .utf8)
            .split(separator: "\n").map { try JSONDecoder().decode(BehaviorEvent.self, from: Data($0.utf8)) }
        XCTAssertEqual(events.filter { $0.kind == .captureGap }.compactMap { $0.data["pausePhase"]?.string }, ["begin", "end"])
        let resumed = events.first { $0.data["continuityPauseID"] != nil }
        XCTAssertEqual(resumed?.data["reason"]?.string, "resolution_update")
        XCTAssertEqual(resumed?.data["targetID"]?.string, a.id)
    }
    @MainActor func testUnknownSystemInterruptionRecordsChangingCausesAsPairedSegments() throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: root) }
        let recorder = RecorderModel(dataDirectory: root, liveObservation: false)
        recorder.start()
        recorder.systemObservationChanged(.systemSleep, active: true)
        recorder.systemObservationChanged(.screenLocked, active: true)
        recorder.shutdown()
        let events = try EventStore(url: root.appendingPathComponent("context-v2.sqlite"), readOnly: true)
            .recent(limit: 100).reversed().filter { $0.kind == .captureGap }
        XCTAssertEqual(events.compactMap { $0.data["interruptionPhase"]?.string }, ["begin", "end", "begin"])
        XCTAssertEqual(events[0].data["interruptionID"]?.string, events[1].data["interruptionID"]?.string)
        XCTAssertEqual(events[0].data["causeSignals"]?.array?.compactMap(\.string), ["system_sleep"])
        XCTAssertEqual(events[2].data["causeSignals"]?.array?.compactMap(\.string), ["screen_locked", "system_sleep"])
        XCTAssertNotEqual(events[1].data["interruptionID"]?.string, events[2].data["interruptionID"]?.string)
    }

    @MainActor func testKnownTargetStillRecordsArrivalInNextSession() throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: root) }
        let target = Destination(id: "app:known", kind: .app, name: "Known", bundleID: "known",
                                 admission: TargetAdmission(basis: "standalone_application"))
        for _ in 0..<2 {
            let recorder = RecorderModel(dataDirectory: root, liveObservation: false)
            recorder.start()
            XCTAssertTrue(recorder.upsert(target))
            recorder.emitContext(targetID: target.id, entranceID: nil, source: "fixture", reason: "foreground_change")
            recorder.shutdown()
        }
        let store = try EventStore(url: root.appendingPathComponent("context-v2.sqlite"))
        let events = try store.recommendationHistory(seedThroughEventID: nil)
        let sessions = Dictionary(grouping: events, by: \.sessionID).values
        XCTAssertEqual(sessions.count, 2)
        for session in sessions {
            XCTAssertEqual(session.filter { $0.kind == .targetUpdated }.count, 1)
            XCTAssertEqual(session.filter { $0.kind == .contextChanged }.count, 1)
        }
    }
}
