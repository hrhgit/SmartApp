import XCTest
import AppKit
import IntentCore
import FileRecallCore
@testable import SmartApp

private actor RecordingDirectoryDecider: DirectoryDeciding {
    private(set) var calls = 0
    private(set) var text = ""
    func decide(context: PageContext, refinement: String, options: [DecisionOption]) async throws -> ChoiceDecision {
        calls += 1; text = context.text
        let chosen = options.first { $0.id != "uncertain" }!.id
        return .init(choice: chosen, confidence: 1,
                     probabilities: Dictionary(uniqueKeysWithValues: options.map { ($0.id, $0.id == chosen ? 1 : 0) }))
    }
}

final class DirectoryLookupIntegrationTests: XCTestCase {
    private func history(_ root: URL, extraUnvisited: Bool = false) throws -> [BehaviorEvent] {
        let identity = TargetIdentity(key: Data(repeating: 9, count: 32))
        let bundle = "com.apple.finder"
        let app = Destination(id: "app:" + bundle, kind: .app, name: "Finder", bundleID: bundle,
                              admission: .init(basis: "standalone_application"))
        var directory = identity.local(root.appendingPathComponent("project-alpha"), directory: true)
        directory.bundleID = bundle
        directory.admission = TargetEligibilityPolicy.finderDirectory(validFileURL: true, isDirectory: true,
            owner: .init(.allowed, "standalone_application")).admission
        var file = identity.local(root.appendingPathComponent("project-alpha/notes.txt"), directory: false)
        file.bundleID = "editor"
        file.admission = TargetEligibilityPolicy.document(validFileURL: true, isRegularFile: true,
            owner: .init(.allowed, "standalone_application")).admission
        let editor = Destination(id: "app:editor", kind: .app, name: "Editor", bundleID: "editor", admission: .init(basis: "standalone_application"))
        let start = BehaviorEvent(sessionID: "fixture", timestampMs: 1, kind: .sessionStart, source: "fixture", data: [
            "stateCoverageVersion": .string(StateCoverage.foregroundOnlyVersion),
            "targetEligibilityPolicyVersion": .string(TargetEligibilityPolicy.version)])
        var events = [start]
        for target in [app, editor, directory, file] {
            events.append(.init(sessionID: "fixture", timestampMs: 2, kind: .targetUpdated, source: "fixture", data: ["target": try .encoded(target)]))
        }
        events.append(.init(id: "arrival-dir", sessionID: "fixture", timestampMs: 10, kind: .contextChanged, source: "fixture",
            app: .init(bundleID: bundle, name: "Finder"), data: ["targetID": .string(directory.id), "reason": .string("foreground_change")]))
        events.append(.init(id: "arrival-file", sessionID: "fixture", timestampMs: 20, kind: .contextChanged, source: "fixture",
            app: .init(bundleID: "editor", name: "Editor"), data: ["targetID": .string(file.id), "reason": .string("foreground_change")]))
        events.append(.init(id: "same-file", sessionID: "fixture", timestampMs: 21, kind: .contextChanged, source: "fixture",
            app: .init(bundleID: "editor", name: "Editor"), data: ["targetID": .string(file.id), "reason": .string("resolution_update")]))
        if extraUnvisited {
            var other = identity.local(root.appendingPathComponent("project-alpha-unvisited"), directory: true)
            other.bundleID = bundle; other.admission = directory.admission
            events.append(.init(sessionID: "fixture", timestampMs: 30, kind: .targetUpdated, source: "fixture", data: ["target": try .encoded(other)]))
        }
        return events
    }
    func testRecallUsesOnlyConfirmedArrivalsAndDoesNotCountResolutionTwice() throws {
        let root = URL(fileURLWithPath: "/tmp/recall-fixture")
        var replay = try DirectoryHistoryReplay(context: .init(application: "Fixture", text: "project-alpha"), refinement: "", nowMs: 100)
        var events = try history(root, extraUnvisited: true)
        let fileID = try XCTUnwrap(events.first { $0.id == "same-file" }?.data["targetID"]?.string)
        events.append(.init(sessionID: "fixture", timestampMs: 40, kind: .captureGap, source: "fixture", data: ["reason": .string("pause")]))
        events.append(.init(id: "resume-file", sessionID: "fixture", timestampMs: 41, kind: .contextChanged, source: "fixture",
            app: .init(bundleID: "editor", name: "Editor"), data: ["targetID": .string(fileID), "reason": .string("resolution_update"),
                "resolvesContextEventID": .string("same-file"), "resolutionEvidence": .string("same_target_after_known_inactivity")]))
        for event in events { try replay.consume(event) }
        let result = try replay.result()
        XCTAssertEqual(result.candidates.count, 1)
        XCTAssertEqual(result.candidates[0].supportingOpenCount, 2)
        XCTAssertEqual(result.observedFiles.count, 1)
        XCTAssertFalse(result.candidates.contains { $0.directory.locator.contains("unvisited") })
    }
    func testCutoffExcludesLaterArrivalsAndOldInventorySessions() throws {
        let root = URL(fileURLWithPath: "/tmp/recall-fixture")
        var replay = try DirectoryHistoryReplay(context: .init(application: "Fixture", text: "project-alpha"), refinement: "", nowMs: 15)
        for event in try history(root) { try replay.consume(event) }
        let result = try replay.result()
        XCTAssertEqual(result.candidates[0].supportingOpenCount, 1)
        XCTAssertTrue(result.observedFiles.isEmpty)
        var legacy = try history(root)
        legacy[0].data["stateCoverageVersion"] = .string(StateCoverage.inventoryVersion)
        var old = try DirectoryHistoryReplay(context: .init(application: "Fixture", text: "project-alpha"), refinement: "", nowMs: 100)
        for event in legacy { try old.consume(event) }
        XCTAssertTrue(try old.result().candidates.isEmpty)
    }
    @MainActor func testClipboardIsPassedToCandidateAndFinalJudgmentAndClearedOnClose() async throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: root) }
        let store = try EventStore(url: root.appendingPathComponent("context-v2.sqlite"))
        try store.append(history(root))
        let decider = RecordingDirectoryDecider()
        let model = DirectoryLookupController(databaseURL: store.url, decider: decider, prepare: {})
        let entry = ClipboardEntry(origin: .observedChange, items: [.init(representations: [], text: "project-alpha")])
        model.begin(context: .init(application: "Browser", text: "Please work on this issue"),
                    clipboard: .init(entry: entry, explicitlySelected: false), at: Date())
        await model.waitForCurrentRequest()
        XCTAssertFalse(model.isLoading); XCTAssertNil(model.error)
        XCTAssertEqual(model.rows.count, 1); XCTAssertEqual(model.rows[0].files.count, 1)
        let text = await decider.text; let calls = await decider.calls
        XCTAssertEqual(calls, 1); XCTAssertTrue(text.contains("project-alpha")); XCTAssertTrue(text.contains("does not prove"))
        try IntegrationPreviewRenderer.render(DirectoryLookupView(model: model, close: {}, back: {}), name: "filerecall", size: CGSize(width: 760, height: 540))
        model.close(); XCTAssertTrue(model.rows.isEmpty); XCTAssertTrue(model.refinement.isEmpty)
        XCTAssertNil(model.clipboardDescription)
    }
    @MainActor func testUnreadablePageStopsBeforeModelEvenWhenClipboardHasUsefulText() async throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: root) }
        let decider = RecordingDirectoryDecider()
        let model = DirectoryLookupController(databaseURL: root.appendingPathComponent("context-v2.sqlite"), decider: decider, prepare: {})
        let clip = ClipboardEntry(origin: .observedChange, items: [.init(representations: [], text: "project-alpha")])
        model.begin(context: .init(application: "Fixture", text: ""), clipboard: .init(entry: clip, explicitlySelected: false))
        await model.waitForCurrentRequest()
        XCTAssertNotNil(model.error); let calls = await decider.calls; XCTAssertEqual(calls, 0)
    }
    @MainActor func testLiveLocalKevWithSyntheticContextWhenRequested() async throws {
        guard ProcessInfo.processInfo.environment["SMARTAPP_TEST_KEV"] == "1" else {
            throw XCTSkip("Explicit local-model smoke check only")
        }
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: root) }
        let store = try EventStore(url: root.appendingPathComponent("context-v2.sqlite"))
        try store.append(history(root))
        let model = DirectoryLookupController(databaseURL: store.url)
        let entry = ClipboardEntry(origin: .observedChange, items: [.init(representations: [], text: "project-alpha notes.txt")])
        model.begin(context: .init(application: "Synthetic fixture", text: "Find the directory for the project-alpha notes."),
                    clipboard: .init(entry: entry, explicitlySelected: false))
        await model.waitForCurrentRequest()
        defer { model.shutdown() }
        XCTAssertNil(model.error)
        XCTAssertFalse(model.rows.isEmpty)
        XCTAssertTrue(model.rows.allSatisfy { $0.probability.isFinite && (0...1).contains($0.probability) })
        model.close()
        XCTAssertTrue(model.rows.isEmpty); XCTAssertFalse(model.canRefine)
        model.begin(context: .init(application: "Second synthetic fixture", text: "Find project-alpha after closing the prior request."),
                    clipboard: .init(entry: entry, explicitlySelected: true))
        await model.waitForCurrentRequest()
        XCTAssertNil(model.error); XCTAssertFalse(model.rows.isEmpty)
    }
    @MainActor func testDefaultMenuExcludesLocalTargetsButHistoryRetainsThem() throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        let suite = "smartapp.menu-scope-test." + UUID().uuidString
        let defaults = UserDefaults(suiteName: suite)!
        defer { defaults.removePersistentDomain(forName: suite); try? FileManager.default.removeItem(at: root) }
        let model = RecorderModel(dataDirectory: root, defaults: defaults, liveObservation: false)
        model.start()
        let app = Destination(id: "app:fixture", kind: .app, name: "Fixture", bundleID: "fixture", admission: .init(basis: "standalone_application"))
        XCTAssertTrue(model.upsert(app))
        let folder = root.appendingPathComponent("project-alpha")
        try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
        let fileURL = folder.appendingPathComponent("notes.txt"); try Data("fixture".utf8).write(to: fileURL)
        let identity = try TargetIdentity(keyURL: root.appendingPathComponent("identity-v2.key"))
        for (url, directory) in [(folder, true), (fileURL, false)] {
            var target = identity.local(url, directory: directory)
            target.admission = TargetEligibilityPolicy.local(registered: true, validFileURL: true).admission
            XCTAssertTrue(model.upsert(target))
        }
        var session: PickerSession?
        model.onShowPicker = { session = $0 }
        model.invokePicker()
        XCTAssertFalse(try XCTUnwrap(session).choices.contains { [.directory, .file].contains($0.target.kind) })
        model.leavePickerForTools(); model.shutdown()
        let targets = try EventStore(url: model.databaseURL, readOnly: true).knownTargets()
        XCTAssertEqual(targets.filter { [.directory, .file].contains($0.kind) }.count, 2)
    }
}
