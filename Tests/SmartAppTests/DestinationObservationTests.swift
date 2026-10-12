import XCTest
import AppKit
import IntentCore
@testable import SmartApp

private final class LockedBox<Value>: @unchecked Sendable {
    private let lock = NSLock()
    private var stored: Value

    init(_ value: Value) { stored = value }

    func read() -> Value {
        lock.lock(); defer { lock.unlock() }
        return stored
    }

    func update(_ body: (inout Value) -> Void) {
        lock.lock(); defer { lock.unlock() }
        body(&stored)
    }
}

final class DestinationObservationTests: XCTestCase {
    private let identity = TargetIdentity(key: Data(repeating: 31, count: 32))

    private var nativeHostURL: URL {
        URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent()
            .deletingLastPathComponent().appendingPathComponent(".build/debug/smartapp-directory-host")
    }

    func testNativeDirectoryHostRefusesNonForegroundRequest() throws {
        XCTAssertTrue(FileManager.default.isExecutableFile(atPath: nativeHostURL.path))
        let session = DirectoryNativeSession(executableURL: nativeHostURL)
        defer { session.close() }
        XCTAssertEqual(session.read(source: "finder", pid: Int32.max)?.status, "stale")
        XCTAssertEqual(session.read(source: "qspace", pid: Int32.max)?.status, "stale")
    }

    func testNativeDirectoryHostTimeoutReturnsControl() throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }
        let host = root.appendingPathComponent("slow-reader")
        try "#!/usr/bin/env python3\nimport time\ntime.sleep(10)\n".write(to: host, atomically: true, encoding: .utf8)
        try FileManager.default.setAttributes([.posixPermissions: 0o700], ofItemAtPath: host.path)
        let session = DirectoryNativeSession(executableURL: host)
        defer { session.close() }
        let start = ProcessInfo.processInfo.systemUptime
        XCTAssertNil(session.read(source: "finder", pid: 1))
        XCTAssertLessThan(ProcessInfo.processInfo.systemUptime - start, 4)
        XCTAssertNil(session.read(source: "finder", pid: 1)) // The failed helper cools down.
    }

    func testDirectoryScriptReusesInterpreterAndRecoversAfterFailure() {
        defer { AutomationScript.closeDirectorySession() }
        let processID = "ObjC.import('Foundation'); String($.NSProcessInfo.processInfo.processIdentifier)"
        let first = AutomationScript.readDirectory(processID)
        let second = AutomationScript.readDirectory(processID)
        XCTAssertEqual(first.code, 0)
        XCTAssertEqual(second.code, 0)
        XCTAssertEqual(first.output, second.output)
        XCTAssertFalse(first.output.isEmpty)

        let fixture = """
        const path = "file:///tmp/课程%20目录/%23页?x=1&y=2";
        JSON.stringify({windows:[{id:"7",front:true,url:path}]});
        """
        let original = AutomationScript.run(fixture)
        let reused = AutomationScript.readDirectory(fixture)
        XCTAssertEqual(original.code, 0)
        XCTAssertEqual(reused.code, 0)
        XCTAssertEqual(original.output.trimmingCharacters(in: .whitespacesAndNewlines),
                       reused.output.trimmingCharacters(in: .whitespacesAndNewlines))

        let failure = AutomationScript.readDirectory("throw new Error('test failure')")
        XCTAssertNotEqual(failure.code, 0)
        let recovered = AutomationScript.readDirectory(processID)
        XCTAssertEqual(recovered.code, 0)
        XCTAssertNotEqual(recovered.output, first.output)
    }

    func testDirectoryScriptBenchmarkWhenRequested() {
        guard ProcessInfo.processInfo.environment["SMARTAPP_DIRECTORY_BENCHMARK"] == "1" else { return }
        defer { AutomationScript.closeDirectorySession() }
        let script = "JSON.stringify({windows:[{id:'7',front:true,url:'file:///tmp/example'}]})"
        let count = 24
        let oldStart = ProcessInfo.processInfo.systemUptime
        let old = (0..<count).map { _ in AutomationScript.run(script) }
        let oldSeconds = ProcessInfo.processInfo.systemUptime - oldStart
        AutomationScript.closeDirectorySession()
        let newStart = ProcessInfo.processInfo.systemUptime
        let new = (0..<count).map { _ in AutomationScript.readDirectory(script) }
        let newSeconds = ProcessInfo.processInfo.systemUptime - newStart
        XCTAssertEqual(old.map(\.code), new.map(\.code))
        XCTAssertEqual(old.map { $0.output.trimmingCharacters(in: .whitespacesAndNewlines) },
                       new.map { $0.output.trimmingCharacters(in: .whitespacesAndNewlines) })
        print("Directory script transport, \(count) reads: one-shot \(oldSeconds)s, reusable \(newSeconds)s")
    }

    func testLiveDirectoryTransportWhenRequested() throws {
        guard let source = ProcessInfo.processInfo.environment["SMARTAPP_LIVE_DIRECTORY_CHECK"] else { return }
        defer { AutomationScript.closeDirectorySession() }
        let bundle = source == "finder" ? FinderReader.bundleID : QSpaceReader.bundleID
        let script = source == "finder" ? FinderReader.observationScript : QSpaceReader.observationScript
        guard NSWorkspace.shared.frontmostApplication?.bundleIdentifier == bundle else {
            throw XCTSkip("Requested file manager is not foreground")
        }
        let old = AutomationScript.run(script)
        let new = AutomationScript.readDirectory(script)
        XCTAssertEqual(old.code, new.code)
        if old.code == -1743 { throw XCTSkip("Test host lacks automation permission") }
        XCTAssertEqual(old.code, 0)
        XCTAssertTrue(old.output.trimmingCharacters(in: .whitespacesAndNewlines) ==
                      new.output.trimmingCharacters(in: .whitespacesAndNewlines),
                      "Foreground snapshots differ between transports")
        let native = DirectoryNativeSession(executableURL: nativeHostURL)
        defer { native.close() }
        let pid = try XCTUnwrap(NSWorkspace.shared.frontmostApplication?.processIdentifier)
        let reply = try XCTUnwrap(native.read(source: source, pid: pid))
        XCTAssertEqual(reply.status, "ready")
        let nativeSnapshot = try XCTUnwrap(reply.snapshot)
        let nativeObject = try XCTUnwrap(JSONSerialization.jsonObject(with: nativeSnapshot) as? NSDictionary)
        let scriptObject = try XCTUnwrap(JSONSerialization.jsonObject(with: Data(new.output.utf8)) as? NSDictionary)
        XCTAssertTrue(nativeObject.isEqual(scriptObject), "Native and script snapshots differ")
        let hasConcreteDirectory: Bool
        if source == "finder" {
            hasConcreteDirectory = (try? JSONDecoder().decode(FinderSnapshot.self, from: Data(new.output.utf8)))?
                .windows.first(where: \.front)?.url != nil
        } else {
            hasConcreteDirectory = (try? JSONDecoder().decode(QSpaceSnapshot.self, from: Data(new.output.utf8)))?
                .windows.first(where: \.front)?.panes.first(where: \.active)?.url != nil
        }
        print("Live \(source) transport: status \(old.code), snapshots equal, directory visible \(hasConcreteDirectory)")
        if ProcessInfo.processInfo.environment["SMARTAPP_LIVE_DIRECTORY_BENCHMARK"] == "1" {
            let count = 8
            let oldStart = ProcessInfo.processInfo.systemUptime
            let oldReads = (0..<count).map { _ in AutomationScript.run(script) }
            let oldSeconds = ProcessInfo.processInfo.systemUptime - oldStart
            let newStart = ProcessInfo.processInfo.systemUptime
            let newReads = (0..<count).map { _ in AutomationScript.readDirectory(script) }
            let newSeconds = ProcessInfo.processInfo.systemUptime - newStart
            XCTAssertTrue(oldReads.allSatisfy { $0.code == 0 })
            XCTAssertTrue(newReads.allSatisfy { $0.code == 0 })
            print("Live \(source), \(count) reads: one-shot \(oldSeconds)s, reusable \(newSeconds)s")
        }
    }

    private func browser(_ url: String, active: String = "11", privateFront: Bool = false) -> BrowserSnapshot {
        BrowserSnapshot(privateFront: privateFront, windows: [
            BrowserWindow(id: "1", front: true, activeID: active, tabs: [
                BrowserTab(id: active, url: url)])
        ])
    }

    func testBrowserNavigationAndTabSwitchPreserveExactLocationsAndStableEntrance() throws {
        let first = browser("https://example.com/course/1?mode=study#chapter-2").resolve(
            identity: identity, bundle: "com.google.Chrome", pid: 7, epoch: "s", excluded: { _ in false })
        let navigated = browser("https://example.com/course/2?mode=study#chapter-3").resolve(
            identity: identity, bundle: "com.google.Chrome", pid: 7, epoch: "s", excluded: { _ in false })
        XCTAssertEqual(first.foreground?.id, navigated.foreground?.id)
        XCTAssertNotEqual(first.foreground?.targetID, navigated.foreground?.targetID)
        XCTAssertEqual(first.targets.first(where: { $0.id == first.foreground?.targetID })?.locator,
                       "https://example.com/course/1?mode=study#chapter-2")
        let switched = browser("https://example.com/course/1", active: "12").resolve(
            identity: identity, bundle: "com.google.Chrome", pid: 7, epoch: "s", excluded: { _ in false })
        XCTAssertNotEqual(first.foreground?.id, switched.foreground?.id)
        XCTAssertEqual(first.entrances.count, 1)
    }

    func testBrowserExclusionPrivateAndUnsupportedPagesCannotBecomeVisits() {
        for url in ["chrome://settings", "https://excluded.example/path?q=keep-out"] {
            let resolved = browser(url).resolve(identity: identity, bundle: "com.google.Chrome", pid: 7, epoch: "s",
                excluded: { $0 == "https://excluded.example" })
            XCTAssertTrue(resolved.excludedFront); XCTAssertNil(resolved.foreground)
            XCTAssertFalse(resolved.targets.contains { $0.locator == url })
        }
        let privateResult = browser("https://example.com", privateFront: true).resolve(
            identity: identity, bundle: "com.google.Chrome", pid: 7, epoch: "s", excluded: { _ in false })
        XCTAssertTrue(privateResult.excludedFront); XCTAssertNil(privateResult.foreground)
        XCTAssertTrue(browser("https://example.com").resolve(identity: identity, bundle: "unknown", pid: 7, epoch: "s",
            excluded: { _ in false }).targets.isEmpty)
    }

    func testFinderDirectoryWithoutRegistrationAndVirtualWindowFallback() throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: root) }
        let folder = root.appendingPathComponent("课程 空格 # &", isDirectory: true)
        try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
        let file = root.appendingPathComponent("test.txt")
        try Data("fixture".utf8).write(to: file)
        let observed = FinderSnapshot(windows: [FinderWindow(id: "1", front: true, url: folder.absoluteString),
            FinderWindow(id: "2", front: false, url: nil), FinderWindow(id: "3", front: false, url: file.absoluteString)])
            .resolve(identity: identity, pid: 2, epoch: "s")
        let directories = observed.targets.filter { $0.kind == .directory }
        XCTAssertEqual(directories.count, 1)
        let directory = try XCTUnwrap(directories.first)
        XCTAssertEqual(URL(string: directory.locator!)?.path, folder.path)
        XCTAssertEqual(directory.parentID, identity.local(root, directory: true).id)
        XCTAssertEqual(directory.admission?.basis, "observed_finder_directory")
        XCTAssertTrue(TargetEligibilityPolicy.validates(directory))
        XCTAssertEqual(observed.foreground?.targetID, directory.id)
        XCTAssertEqual(observed.entrances.count, 1)
        XCTAssertFalse(observed.targets.contains { $0.locator == file.absoluteString })
    }

    func testQSpaceDirectoryUsesOnlyRealPaneLocationsAndOnlyTheActiveFrontPaneVisits() throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: root) }
        let activeFolder = root.appendingPathComponent("active folder # &", isDirectory: true)
        let backgroundFolder = root.appendingPathComponent("background", isDirectory: true)
        let file = root.appendingPathComponent("not-a-directory.txt")
        try FileManager.default.createDirectory(at: activeFolder, withIntermediateDirectories: true)
        try FileManager.default.createDirectory(at: backgroundFolder, withIntermediateDirectories: true)
        try Data("fixture".utf8).write(to: file)

        let observed = QSpaceSnapshot(windows: [
            QSpaceWindow(id: "front", front: true, panes: [
                QSpacePane(id: "active", active: true, url: activeFolder.absoluteString),
                QSpacePane(id: "file", active: false, url: file.absoluteString),
                QSpacePane(id: "virtual", active: false, url: nil)
            ]),
            QSpaceWindow(id: "background", front: false, panes: [
                QSpacePane(id: "other", active: true, url: backgroundFolder.absoluteString)
            ])
        ]).resolve(identity: identity, pid: 9, epoch: "s")

        let directories = observed.targets.filter { $0.kind == .directory }
        XCTAssertEqual(directories.count, 1)
        let activeDirectory = try XCTUnwrap(directories.first { URL(string: $0.locator ?? "")?.path == activeFolder.path })
        XCTAssertEqual(activeDirectory.bundleID, QSpaceReader.bundleID)
        XCTAssertEqual(activeDirectory.admission?.basis, "observed_qspace_directory")
        XCTAssertEqual(activeDirectory.parentID, identity.local(root, directory: true).id)
        XCTAssertTrue(TargetEligibilityPolicy.validates(activeDirectory))

        let activeEntrance = try XCTUnwrap(observed.foreground)
        XCTAssertEqual(activeEntrance.kind, "qspace_pane")
        XCTAssertEqual(activeEntrance.windowID, "front")
        XCTAssertEqual(activeEntrance.paneID, "active")
        XCTAssertEqual(activeEntrance.targetID, activeDirectory.id)
        XCTAssertTrue(activeEntrance.id.hasPrefix("qspace-pane:s:9:front:active"))
        XCTAssertEqual(observed.entrances.count, 1)
        XCTAssertNil(observed.entrances.first { $0.paneID == "file" })
        XCTAssertNil(observed.entrances.first { $0.paneID == "virtual" })
        XCTAssertNil(observed.entrances.first { $0.paneID == "other" })
        XCTAssertFalse(observed.targets.contains { URL(string: $0.locator ?? "")?.path == backgroundFolder.path })
    }

    @MainActor func testObservedTargetsPersistExportAndReopenAfterEntrancesDisappearAndRestart() throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        let suite = "smartapp.destinations." + UUID().uuidString
        let defaults = UserDefaults(suiteName: suite)!
        defer { defaults.removePersistentDomain(forName: suite); try? FileManager.default.removeItem(at: root) }
        let folder = root.appendingPathComponent("folder", isDirectory: true)
        try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
        let recorder = RecorderModel(dataDirectory: root, defaults: defaults, liveObservation: false)
        let localIdentity = try TargetIdentity(keyURL: root.appendingPathComponent("identity-v2.key"))
        recorder.start()
        for bundle in ["com.google.Chrome", FinderReader.bundleID] {
            XCTAssertTrue(recorder.upsert(Destination(id: "app:" + bundle, kind: .app, name: bundle,
                bundleID: bundle, admission: TargetAdmission(basis: "standalone_application"))))
        }
        let web = browser("https://example.com/project/parent/child?q=semantic#route").resolve(
            identity: localIdentity, bundle: "com.google.Chrome", pid: 7, epoch: "s", excluded: { _ in false })
        let finder = FinderSnapshot(windows: [FinderWindow(id: "1", front: true, url: folder.absoluteString)])
            .resolve(identity: localIdentity, pid: 2, epoch: "s")
        recorder.acceptObservation(web, source: "browser:com.google.Chrome", foreground: false,
            fallbackID: "app:com.google.Chrome", reason: "page_or_tab_change", started: 1)
        XCTAssertFalse(recorder.recent.contains { $0.kind == .contextChanged })
        for _ in 0..<2 {
            recorder.acceptObservation(web, source: "browser:com.google.Chrome", foreground: true,
                fallbackID: "app:com.google.Chrome", reason: "page_or_tab_change", started: 2)
        }
        recorder.acceptObservation(finder, source: "finder:" + FinderReader.bundleID, foreground: true,
            fallbackID: "app:" + FinderReader.bundleID, reason: "directory_or_window_change", started: 3)
        XCTAssertEqual(recorder.recent.filter { $0.kind == .contextChanged }.count, 1)
        XCTAssertTrue(recorder.registeredTargets.isEmpty)
        let page = try XCTUnwrap(web.targets.first { $0.id == web.foreground?.targetID })
        let directory = try XCTUnwrap(finder.targets.first { $0.kind == .directory })
        recorder.setSourceStatus("browser:com.google.Chrome", "unavailable", reason: "fixture_entrance_gone")
        recorder.setSourceStatus("finder:" + FinderReader.bundleID, "unavailable", reason: "fixture_entrance_gone")
        XCTAssertTrue(recorder.eligibleNow(page)); XCTAssertTrue(recorder.eligibleNow(directory))
        XCTAssertEqual(recorder.reopenURL(page)?.absoluteString, page.locator)
        XCTAssertEqual(recorder.reopenURL(directory)?.path, folder.path)
        recorder.shutdown()
        let store = try EventStore(url: root.appendingPathComponent("context-v2.sqlite"), readOnly: true)
        XCTAssertTrue(try store.knownTargets().contains(page)); XCTAssertTrue(try store.knownTargets().contains(directory))
        let export = root.appendingPathComponent("export")
        let report = try store.export(to: export)
        XCTAssertEqual(report.samples, 2)
        let lines = try String(contentsOf: export.appendingPathComponent("target-history.jsonl"), encoding: .utf8)
        XCTAssertTrue(lines.contains("semantic")); XCTAssertTrue(lines.contains("observed_finder_directory"))
        let restarted = RecorderModel(dataDirectory: root, defaults: defaults, liveObservation: false)
        XCTAssertTrue(restarted.eligibleNow(page)); XCTAssertTrue(restarted.eligibleNow(directory))
        restarted.excludedSitesText = "example.com"; restarted.saveSettings()
        XCTAssertFalse(restarted.eligibleNow(page)); XCTAssertNil(restarted.reopenURL(page))
        restarted.shutdown()
    }

    func testAutomationHelperTimeoutDoesNotHangCollector() {
        let started = Date()
        let result = AutomationScript.run("delay(10); 'late';", timeout: 0.2)
        XCTAssertNotEqual(result.code, 0)
        XCTAssertLessThan(Date().timeIntervalSince(started), 3)
    }

    func testAutomationHelperCanRunAppleScript() {
        let result = AutomationScript.run("return \"automation-ready\"", language: .appleScript)
        XCTAssertEqual(result.code, 0)
        XCTAssertEqual(result.output.trimmingCharacters(in: .whitespacesAndNewlines), "automation-ready")
    }

    @MainActor func testSelectingNavigatedBrowserEntranceReopensSavedURL() async throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        let suite = "smartapp.open-test." + UUID().uuidString
        let defaults = UserDefaults(suiteName: suite)!
        defer { defaults.removePersistentDomain(forName: suite); try? FileManager.default.removeItem(at: root) }
        var opened: [URL] = []
        let changedSnapshot = browser("https://example.com/already-navigated")
        var actions = DestinationActions()
        actions.readBrowser = { _ in BrowserResult(snapshot: changedSnapshot, status: "ready") }
        actions.activateBrowser = { _, _, _, _ in XCTFail("Must not raise a tab that now points elsewhere"); return false }
        actions.openURL = { opened.append($0); return true }
        let recorder = RecorderModel(dataDirectory: root, defaults: defaults, liveObservation: false, destinationActions: actions)
        let localIdentity = try TargetIdentity(keyURL: root.appendingPathComponent("identity-v2.key"))
        recorder.start()
        let bundle = "com.google.Chrome"
        _ = recorder.upsert(Destination(id: "app:" + bundle, kind: .app, name: "Chrome", bundleID: bundle,
            admission: TargetAdmission(basis: "standalone_application")))
        let observation = browser("https://example.com/saved?q=1#child").resolve(identity: localIdentity,
            bundle: bundle, pid: 7, epoch: "s", excluded: { _ in false })
        for target in observation.foregroundOnly().targets { XCTAssertTrue(recorder.upsert(target)) }
        var picker: PickerSession?
        recorder.onShowPicker = { picker = $0 }
        recorder.invokePicker()
        let session = try XCTUnwrap(picker)
        let choice = try XCTUnwrap(session.choices.first { $0.target.locator == "https://example.com/saved?q=1#child" })
        // Carry the actual observed entrance even on test hosts without Chrome installed.
        let observedChoice = NavigationChoice(target: choice.target, entrance: observation.foreground)
        recorder.choose(observedChoice, in: session)
        for _ in 0..<100 where opened.isEmpty { try await Task.sleep(for: .milliseconds(10)) }
        XCTAssertEqual(opened.map(\.absoluteString), ["https://example.com/saved?q=1#child"])
        XCTAssertFalse(recorder.recent.contains { $0.kind == .recommendationFeedback })
        recorder.shutdown()
        let events = try EventStore(url: root.appendingPathComponent("context-v2.sqlite"), readOnly: true).recent(limit: 100)
        XCTAssertTrue(events.contains { $0.kind == .recommendationFeedback && $0.data["action"]?.string == "execution_result" && $0.data["success"] == .bool(true) })
        XCTAssertFalse(events.contains { $0.kind == .recommendationShown })
        XCTAssertFalse(events.contains { $0.kind == .recommendationFeedback &&
            ($0.data["candidateTargetIDs"] != nil || $0.data["items"] != nil) })
    }

    @MainActor func testSelectingClosedFinderWindowOpensSavedDirectory() async throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        let suite = "smartapp.finder-open-test." + UUID().uuidString
        let defaults = UserDefaults(suiteName: suite)!
        defer { defaults.removePersistentDomain(forName: suite); try? FileManager.default.removeItem(at: root) }
        var opened: [URL] = []
        var actions = DestinationActions()
        actions.activateFinder = { _, _ in false }
        actions.openURL = { opened.append($0); return true }
        let recorder = RecorderModel(dataDirectory: root, defaults: defaults, liveObservation: false, destinationActions: actions)
        let localIdentity = try TargetIdentity(keyURL: root.appendingPathComponent("identity-v2.key"))
        recorder.start()
        _ = recorder.upsert(Destination(id: "app:" + FinderReader.bundleID, kind: .app, name: "Finder", bundleID: FinderReader.bundleID,
            admission: TargetAdmission(basis: "standalone_application")))
        let observation = FinderSnapshot(windows: [FinderWindow(id: "1", front: true, url: root.absoluteString)])
            .resolve(identity: localIdentity, pid: 2, epoch: "s")
        for target in observation.foregroundOnly().targets { XCTAssertTrue(recorder.upsert(target)) }
        let target = try XCTUnwrap(observation.foregroundOnly().targets.first { $0.kind == .directory })
        let success = await recorder.openRecallTarget(target, decisionID: "recall-fixture", preferredEntrance: observation.foreground)
        XCTAssertTrue(success)
        for _ in 0..<100 where opened.isEmpty { try await Task.sleep(for: .milliseconds(10)) }
        XCTAssertEqual(opened.map(\.path), [root.path])
        recorder.shutdown()
    }

    @MainActor func testSelectingExistingQSpacePaneActivatesThatPaneBeforeFallback() async throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        let suite = "smartapp.qspace-activate-test." + UUID().uuidString
        let defaults = UserDefaults(suiteName: suite)!
        defer { defaults.removePersistentDomain(forName: suite); try? FileManager.default.removeItem(at: root) }
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        let activation = LockedBox<(window: String, pane: String, url: String?)?>(nil)
        let fallbackCalled = LockedBox(false)
        var actions = DestinationActions()
        actions.activateQSpace = { window, pane, url in
            activation.update { $0 = (window, pane, url) }
            return true
        }
        actions.openQSpace = { _ in fallbackCalled.update { $0 = true }; return false }
        actions.openURL = { _ in fallbackCalled.update { $0 = true }; return false }
        let recorder = RecorderModel(dataDirectory: root, defaults: defaults, liveObservation: false, destinationActions: actions)
        let localIdentity = try TargetIdentity(keyURL: root.appendingPathComponent("identity-v2.key"))
        recorder.start()
        _ = recorder.upsert(Destination(id: "app:" + QSpaceReader.bundleID, kind: .app, name: "QSpace Pro",
            bundleID: QSpaceReader.bundleID, admission: TargetAdmission(basis: "standalone_application")))
        let observation = QSpaceSnapshot(windows: [QSpaceWindow(id: "window-1", front: true, panes: [
            QSpacePane(id: "pane-2", active: true, url: root.absoluteString)
        ])]).resolve(identity: localIdentity, pid: 9, epoch: "s")
        for target in observation.foregroundOnly().targets { XCTAssertTrue(recorder.upsert(target)) }
        let target = try XCTUnwrap(observation.foregroundOnly().targets.first { $0.kind == .directory })
        let success = await recorder.openRecallTarget(target, decisionID: "recall-fixture", preferredEntrance: observation.foreground)
        XCTAssertTrue(success)
        for _ in 0..<100 where activation.read() == nil { try await Task.sleep(for: .milliseconds(10)) }
        XCTAssertEqual(activation.read()?.window, "window-1")
        XCTAssertEqual(activation.read()?.pane, "pane-2")
        XCTAssertEqual(activation.read()?.url, root.absoluteString)
        XCTAssertFalse(fallbackCalled.read())
        recorder.shutdown()
    }

    @MainActor func testSelectingClosedQSpacePaneReopensSavedDirectoryInQSpace() async throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        let suite = "smartapp.qspace-open-test." + UUID().uuidString
        let defaults = UserDefaults(suiteName: suite)!
        defer { defaults.removePersistentDomain(forName: suite); try? FileManager.default.removeItem(at: root) }
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        let qspaceOpened = LockedBox<[URL]>([])
        let genericOpened = LockedBox<[URL]>([])
        var actions = DestinationActions()
        actions.activateQSpace = { _, _, _ in false }
        actions.openQSpace = { url in qspaceOpened.update { $0.append(url) }; return true }
        actions.openURL = { url in genericOpened.update { $0.append(url) }; return true }
        let recorder = RecorderModel(dataDirectory: root, defaults: defaults, liveObservation: false, destinationActions: actions)
        let localIdentity = try TargetIdentity(keyURL: root.appendingPathComponent("identity-v2.key"))
        recorder.start()
        _ = recorder.upsert(Destination(id: "app:" + QSpaceReader.bundleID, kind: .app, name: "QSpace Pro",
            bundleID: QSpaceReader.bundleID, admission: TargetAdmission(basis: "standalone_application")))
        let observation = QSpaceSnapshot(windows: [QSpaceWindow(id: "window-1", front: true, panes: [
            QSpacePane(id: "pane-2", active: true, url: root.absoluteString)
        ])]).resolve(identity: localIdentity, pid: 9, epoch: "s")
        for target in observation.foregroundOnly().targets { XCTAssertTrue(recorder.upsert(target)) }
        let target = try XCTUnwrap(observation.foregroundOnly().targets.first { $0.kind == .directory })
        let success = await recorder.openRecallTarget(target, decisionID: "recall-fixture", preferredEntrance: observation.foreground)
        XCTAssertTrue(success)
        for _ in 0..<100 where qspaceOpened.read().isEmpty { try await Task.sleep(for: .milliseconds(10)) }
        XCTAssertEqual(qspaceOpened.read().map(\.absoluteString), [root.absoluteString])
        XCTAssertTrue(genericOpened.read().isEmpty)
        recorder.shutdown()
    }
}
