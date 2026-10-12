import XCTest
import AppKit
import IntentCore
@testable import SmartApp

final class TargetCaptureTests: XCTestCase {
    @MainActor func testHeartbeatCompressionAndDisplayedActivitySharePersistedSemantics() throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        let suite = "smartapp.activity-test." + UUID().uuidString
        let defaults = UserDefaults(suiteName: suite)!
        defer { defaults.removePersistentDomain(forName: suite); try? FileManager.default.removeItem(at: root) }
        let recorder = RecorderModel(dataDirectory: root, defaults: defaults, liveObservation: false)
        recorder.start()
        for name in ["a", "b", "c"] {
            XCTAssertTrue(recorder.upsert(Destination(id: "app:" + name, kind: .app, name: name, bundleID: name,
                admission: TargetAdmission(basis: "standalone_application"))))
        }
        recorder.emitContext(targetID: "app:a", entranceID: nil, source: "fixture", reason: "foreground_change")
        recorder.emitContext(targetID: "app:b", entranceID: nil, source: "fixture", reason: "foreground_change")
        recorder.emitContext(targetID: "app:c", entranceID: nil, source: "fixture", reason: "resolution_update")
        let now = ProcessInfo.processInfo.systemUptime * 1000
        for index in 0...60 { recorder.observeHeartbeat(at: now - 30_000 + Double(index) * 500) }
        recorder.observeHeartbeat(at: now + 1)
        recorder.gap("fixture_boundary")
        recorder.emitContext(targetID: "app:a", entranceID: nil, source: "fixture", reason: "foreground_change")
        recorder.emitContext(targetID: "app:b", entranceID: nil, source: "fixture", reason: "foreground_change")
        XCTAssertEqual(recorder.sessionSwitches, 2)
        XCTAssertEqual(recorder.recent.count, 2)
        recorder.shutdown()
        let store = try EventStore(url: root.appendingPathComponent("context-v2.sqlite"), readOnly: true)
        XCTAssertEqual(try store.stats().userActions, recorder.sessionSwitches)
        let events = try store.recent(limit: 100)
        let heartbeats = events.filter { $0.kind == .heartbeat }
        XCTAssertEqual(heartbeats.count, 1)
        XCTAssertEqual(heartbeats.first?.data["checkCount"]?.number, 61)
        let gap = try XCTUnwrap(events.first { $0.kind == .captureGap && $0.data["reason"]?.string == "fixture_boundary" })
        XCTAssertEqual(gap.data["collectorHeartbeatSummary"]?.object?["checkCount"]?.number, 1)
        _ = try store.export(to: root.appendingPathComponent("export"))
    }
    @MainActor func testFailureReasonChangesAreNotCompressedAway() throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        let suite = "smartapp.reason-test." + UUID().uuidString
        let defaults = UserDefaults(suiteName: suite)!
        defer { defaults.removePersistentDomain(forName: suite); try? FileManager.default.removeItem(at: root) }
        let recorder = RecorderModel(dataDirectory: root, defaults: defaults, liveObservation: false)
        recorder.start()
        recorder.setSourceStatus("fixture", "unavailable", reason: "timeout")
        recorder.setSourceStatus("fixture", "unavailable", reason: "timeout")
        recorder.setSourceStatus("fixture", "unavailable", reason: "denied")
        recorder.shutdown()
        let events = try EventStore(url: root.appendingPathComponent("context-v2.sqlite"), readOnly: true).recent(limit: 100)
        XCTAssertEqual(events.filter { $0.source == "fixture" && $0.kind == .sourceStatus }.count, 2)
    }
    @MainActor func testRecorderPersistsPointInterruptionCause() throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: root) }
        let recorder = RecorderModel(dataDirectory: root, liveObservation: false)
        recorder.start()
        recorder.gap("scheduler_or_clock_gap")
        recorder.shutdown()
        let events = try EventStore(url: root.appendingPathComponent("context-v2.sqlite"), readOnly: true).recent(limit: 100)
        let gap = try XCTUnwrap(events.first { $0.kind == .captureGap })
        XCTAssertEqual(gap.data["reason"]?.string, "scheduler_or_clock_gap")
        XCTAssertEqual(gap.data["causeCategory"]?.string, "scheduler_or_clock_gap")
        XCTAssertNil(gap.data["targetID"])
    }

    @MainActor func testSourceRecoveryIsAnObservationRefinementNotAnotherVisit() throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        let suite = "smartapp.recovery-test." + UUID().uuidString
        let defaults = UserDefaults(suiteName: suite)!
        defer { defaults.removePersistentDomain(forName: suite); try? FileManager.default.removeItem(at: root) }
        let recorder = RecorderModel(dataDirectory: root, defaults: defaults, liveObservation: false)
        recorder.start()
        let app = Destination(id: "app:a", kind: .app, name: "A", bundleID: "a", admission: TargetAdmission(basis: "standalone_application"))
        let window = Destination(id: "window:one", kind: .window, name: "Window", bundleID: "a", admission: TargetAdmission(basis: "standard_work_window"))
        XCTAssertTrue(recorder.upsert(app)); XCTAssertTrue(recorder.upsert(window))
        let entry = TargetEntrance(id: "entry", targetID: window.id, bundleID: "a", kind: "ax_window", containerID: window.id)
        recorder.setForegroundEntrance("fixture", entry)
        recorder.emitContext(targetID: window.id, entranceID: entry.id, source: "fixture", reason: "foreground_change")
        recorder.setSourceStatus("fixture", "unavailable", reason: "disconnected")
        recorder.setForegroundEntrance("fixture", entry)
        recorder.emitContext(targetID: window.id, entranceID: entry.id, source: "fixture", reason: "foreground_change")
        recorder.shutdown()
        let store = try EventStore(url: root.appendingPathComponent("context-v2.sqlite"), readOnly: true)
        let contexts = try store.recent(limit: 100).reversed().filter { $0.kind == .contextChanged }
        XCTAssertEqual(contexts.count, 2)
        XCTAssertEqual(contexts.last?.data["reason"]?.string, "resolution_update")
        XCTAssertEqual(contexts.last?.data["resolvesContextEventID"]?.string, contexts.first?.id)
        XCTAssertEqual(try store.export(to: root.appendingPathComponent("export")).samples, 1)
    }
    @MainActor func testSourceStatusRecoveryDoesNotPersistAnInventory() throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        let suite = "smartapp.coverage-test." + UUID().uuidString
        let defaults = UserDefaults(suiteName: suite)!
        defer { defaults.removePersistentDomain(forName: suite); try? FileManager.default.removeItem(at: root) }
        let recorder = RecorderModel(dataDirectory: root, defaults: defaults, liveObservation: false)
        recorder.start()
        recorder.setSourceStatus("source-a", "ready")
        recorder.setSourceStatus("source-b", "ready")
        recorder.setSourceStatus("source-a", "unavailable", reason: "fixture_disconnect")
        recorder.setSourceStatus("source-a", "ready")
        recorder.shutdown()
        let events = try EventStore(url: root.appendingPathComponent("context-v2.sqlite"), readOnly: true).recent(limit: 100).reversed()
        XCTAssertFalse(events.contains { $0.kind == .inventoryChanged })
        XCTAssertFalse(events.contains { $0.kind == .captureGap })
        let failure = try XCTUnwrap(events.first {
            $0.kind == .sourceStatus && $0.source == "source-a" && $0.data["observationStatus"]?.string == "unavailable"
        })
        XCTAssertEqual(failure.data["observationStatus"]?.string, "unavailable")
        XCTAssertNil(failure.data["entrances"])
    }
    @MainActor func testObservedFileNeedsRealFileAndAdmittedOwnerButNotRegistration() throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        let suite = "smartapp.document-test." + UUID().uuidString
        let defaults = UserDefaults(suiteName: suite)!
        defer { defaults.removePersistentDomain(forName: suite); try? FileManager.default.removeItem(at: root) }
        let recorder = RecorderModel(dataDirectory: root, defaults: defaults, liveObservation: false)
        recorder.start()
        let file = root.appendingPathComponent("synthetic.txt")
        try Data("Synthetic test document".utf8).write(to: file)
        XCTAssertNil(recorder.observedDocument(file, bundle: "test.editor"))
        XCTAssertTrue(recorder.upsert(Destination(id: "app:test.editor", kind: .app, name: "Editor", bundleID: "test.editor", admission: TargetAdmission(basis: "standalone_application"))))
        let target = try XCTUnwrap(recorder.observedDocument(file, bundle: "test.editor"))
        XCTAssertEqual(target.kind, .file); XCTAssertEqual(target.locator, file.absoluteString)
        XCTAssertTrue(recorder.registeredTargets.isEmpty)
        XCTAssertTrue(recorder.upsert(target))
        XCTAssertNil(recorder.observedDocument(root, bundle: "test.editor"))
        XCTAssertNil(recorder.observedDocument(root.appendingPathComponent("missing.txt"), bundle: "test.editor"))
        XCTAssertNil(recorder.observedDocument(file, bundle: "com.apple.SecurityAgentHelper"))
        recorder.shutdown()
        let store = try EventStore(url: root.appendingPathComponent("context-v2.sqlite"), readOnly: true)
        XCTAssertTrue(try store.knownTargets().contains(target))
        XCTAssertFalse(String(decoding: try JSONEncoder().encode(store.recent(limit: 100)), as: UTF8.self).contains("Synthetic test document"))
        let reloaded = RecorderModel(dataDirectory: root, defaults: defaults, liveObservation: false)
        XCTAssertEqual(reloaded.reopenURL(target), file)
    }
    @MainActor func testSystemFinderSpecialBundleTypeIsAdmitted() throws {
        let url = try XCTUnwrap(NSWorkspace.shared.urlForApplication(withBundleIdentifier: "com.apple.finder"))
        let facts = ApplicationClassifier.evidence(bundleID: "com.apple.finder", url: url, activation: .regular)
        XCTAssertTrue(facts.standaloneApplication)
        XCTAssertTrue(TargetEligibilityPolicy.application(facts).isAllowed)
        XCTAssertFalse(TargetEligibilityPolicy.application(facts, rules: ["com.apple.finder"]).isAllowed)
    }
    @MainActor func testRecorderRejectsHelperAcrossCatalogInventoryContextAndLifecycle() throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        let suite = "smartapp.capture-test." + UUID().uuidString
        let defaults = UserDefaults(suiteName: suite)!
        defer { defaults.removePersistentDomain(forName: suite); try? FileManager.default.removeItem(at: root) }
        let recorder = RecorderModel(dataDirectory: root, defaults: defaults, liveObservation: false)
        recorder.start()
        let helper = Destination(id: "app:com.apple.SecurityAgentHelper.arm64", kind: .app, name: "REJECTED_SENTINEL",
            bundleID: "com.apple.SecurityAgentHelper.arm64", admission: TargetAdmission(basis: "standalone_application"))
        XCTAssertFalse(recorder.upsert(helper))
        recorder.setForegroundEntrance("test-source", TargetEntrance(id: "REJECTED_ENTRANCE", targetID: helper.id, bundleID: helper.bundleID!, kind: "running_app"))
        recorder.emitContext(targetID: helper.id, entranceID: nil, source: "test", reason: "foreground_change")
        for kind: EventKind in [.appActivated, .appLaunched, .appTerminated] {
            XCTAssertNil(recorder.record(kind, source: "test", app: AppContext(bundleID: helper.bundleID!, name: helper.name)))
        }
        let allowed = Destination(id: "app:test.normal", kind: .app, name: "Normal", bundleID: "test.normal", admission: TargetAdmission(basis: "standalone_application"))
        XCTAssertTrue(recorder.upsert(allowed))
        recorder.emitContext(targetID: allowed.id, entranceID: nil, source: "test", reason: "foreground_change")
        recorder.shutdown()
        let store = try EventStore(url: root.appendingPathComponent("context-v2.sqlite"), readOnly: true)
        let events = try store.recent(limit: 100)
        XCTAssertTrue(events.contains { $0.kind == .contextChanged })
        let serialized = String(decoding: try JSONEncoder().encode(events), as: UTF8.self)
        for forbidden in [helper.id, helper.name, "REJECTED_ENTRANCE"] { XCTAssertFalse(serialized.contains(forbidden)) }
        let export = root.appendingPathComponent("export")
        XCTAssertEqual(try store.export(to: export).samples, 1)
        let samples = try String(contentsOf: export.appendingPathComponent("navigation-samples.jsonl"), encoding: .utf8)
        XCTAssertFalse(samples.contains(helper.id))
    }
    @MainActor func testOldCatalogIsNotRepublishedAndClosingMenuDoesNotLinkNaturalNavigation() throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        let suite = "smartapp.capture-test." + UUID().uuidString
        let defaults = UserDefaults(suiteName: suite)!
        defer { defaults.removePersistentDomain(forName: suite); try? FileManager.default.removeItem(at: root) }
        let db = root.appendingPathComponent("context-v2.sqlite")
        let old = Destination(id: "app:old", kind: .app, name: "Old", bundleID: "old")
        do {
            let store = try EventStore(url: db)
            var event = BehaviorEvent(sessionID: "old", kind: .targetUpdated, source: "test", data: ["target": try .encoded(old)])
            event.schemaVersion = 2; try store.append([event])
        }
        let recorder = RecorderModel(dataDirectory: root, defaults: defaults, liveObservation: false)
        recorder.start()
        var shown: PickerSession?
        recorder.onShowPicker = { shown = $0 }
        recorder.invokePicker()
        let picker = try XCTUnwrap(shown)
        XCTAssertTrue(picker.choices.isEmpty)
        recorder.pickerShown(picker); recorder.pickerClosed(picker)
        let good = Destination(id: "app:good", kind: .app, name: "Good", bundleID: "good", admission: TargetAdmission(basis: "standalone_application"))
        XCTAssertTrue(recorder.upsert(good))
        recorder.emitContext(targetID: good.id, entranceID: nil, source: "test", reason: "foreground_change")
        recorder.shutdown()
        let events = try EventStore(url: db, readOnly: true).recent(limit: 100).filter { $0.sessionID != "old" }
        XCTAssertFalse(events.contains { [.menuInvoked, .recommendationShown, .recommendationFeedback].contains($0.kind) })
        XCTAssertNil(events.first(where: { $0.kind == .contextChanged })?.decisionID)
        XCTAssertFalse(String(decoding: try JSONEncoder().encode(events), as: UTF8.self).contains("app:old"))
    }
    @MainActor func testBundleStructureDistinguishesIndependentAppsFromEmbeddedComponents() throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: root) }
        func bundle(_ path: String) throws -> URL {
            let url = root.appendingPathComponent(path)
            let executable = url.appendingPathComponent("Contents/MacOS/Test")
            try FileManager.default.createDirectory(at: executable.deletingLastPathComponent(), withIntermediateDirectories: true)
            try Data("fixture".utf8).write(to: executable)
            try FileManager.default.setAttributes([.posixPermissions: 0o700], ofItemAtPath: executable.path)
            let plist: [String: Any] = ["CFBundleIdentifier": "test.helper", "CFBundlePackageType": "APPL", "CFBundleExecutable": "Test"]
            try PropertyListSerialization.data(fromPropertyList: plist, format: .xml, options: 0).write(to: url.appendingPathComponent("Contents/Info.plist"))
            return url
        }
        let normal = try bundle("RealHelper.app")
        XCTAssertTrue(TargetEligibilityPolicy.application(ApplicationClassifier.evidence(bundleID: "test.helper", url: normal, activation: .regular)).isAllowed)
        for path in ["Host.app/Contents/Library/LoginItems/Helper.app", "Service.xpc", "Plugin.appex"] {
            let url = try bundle(path)
            XCTAssertFalse(TargetEligibilityPolicy.application(ApplicationClassifier.evidence(bundleID: "test.helper", url: url, activation: .regular)).isAllowed)
        }
    }
}
