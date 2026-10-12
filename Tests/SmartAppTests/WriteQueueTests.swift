import XCTest
import IntentCore
@testable import SmartApp

final class WriteQueueTests: XCTestCase {
    @MainActor
    func testRestoringKnownTargetsDoesNotReenqueueUnchangedTargetUpdates() throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        let suite = "smartapp.write-queue." + UUID().uuidString
        let defaults = try XCTUnwrap(UserDefaults(suiteName: suite))
        defer {
            defaults.removePersistentDomain(forName: suite)
            try? FileManager.default.removeItem(at: root)
        }

        let store = try EventStore(url: root.appendingPathComponent("context-v2.sqlite"))
        let targets = (0..<1_050).map { index in
            Destination(id: "app:fixture.\(index)", kind: .app, name: "Fixture \(index)",
                        bundleID: "fixture.\(index)",
                        admission: TargetAdmission(basis: "standalone_application"))
        }
        let seed = targets.map { target in
            BehaviorEvent(sessionID: "seed", kind: .targetUpdated, source: "target_catalog",
                          data: ["target": try! JSONValue.encoded(target)])
        }
        try store.append(seed)

        let recorder = RecorderModel(dataDirectory: root, defaults: defaults, liveObservation: false)
        recorder.start()
        // Starting a session must not republish the historical target catalog.
        // A target is repeated only when it is observed in this new session.
        recorder.shutdown()

        let stats = try EventStore(url: root.appendingPathComponent("context-v2.sqlite"), readOnly: true).stats()
        XCTAssertEqual(stats.targets, targets.count)
    }

    @MainActor
    func testStartingAfterWriteBackpressureWarningClearsStaleMessage() throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        let suite = "smartapp.write-warning." + UUID().uuidString
        let defaults = try XCTUnwrap(UserDefaults(suiteName: suite))
        defer {
            defaults.removePersistentDomain(forName: suite)
            try? FileManager.default.removeItem(at: root)
        }

        let recorder = RecorderModel(dataDirectory: root, defaults: defaults, liveObservation: false)
        recorder.errorMessage = "写入积压超过安全上限，已暂停采集并提交会话结束；请检查磁盘或导出后重试。"
        recorder.start()
        XCTAssertNil(recorder.errorMessage)
        recorder.shutdown()
    }

    @MainActor
    func testBackpressureDrainsAndResumesWithoutManualRetry() async throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        let suite = "smartapp.automatic-write-recovery." + UUID().uuidString
        let defaults = try XCTUnwrap(UserDefaults(suiteName: suite))
        defer {
            defaults.removePersistentDomain(forName: suite)
            try? FileManager.default.removeItem(at: root)
        }

        let recorder = RecorderModel(dataDirectory: root, defaults: defaults, liveObservation: false)
        recorder.start()
        for index in 0..<1_300 {
            let target = Destination(id: "app:burst.\(index)", kind: .app, name: "Burst \(index)",
                                     bundleID: "burst.\(index)",
                                     admission: TargetAdmission(basis: "standalone_application"))
            XCTAssertTrue(recorder.upsert(target))
            if recorder.isWritingBacklog { break }
        }

        XCTAssertTrue(recorder.isWritingBacklog)
        XCTAssertFalse(recorder.isCollecting)
        for _ in 0..<300 {
            if recorder.isCollecting && !recorder.isWritingBacklog { break }
            try await Task.sleep(for: .milliseconds(10))
        }
        XCTAssertTrue(recorder.isCollecting)
        XCTAssertFalse(recorder.isWritingBacklog)
        XCTAssertNil(recorder.errorMessage)

        recorder.shutdown()
        let store = try EventStore(url: root.appendingPathComponent("context-v2.sqlite"), readOnly: true)
        let events = try store.recent(limit: 2_000)
        XCTAssertEqual(events.filter { $0.kind == .sessionEnd && $0.data["reason"]?.string == "write_backpressure" }.count, 1)
        XCTAssertGreaterThanOrEqual(try store.stats().sessions, 2)
    }
}
