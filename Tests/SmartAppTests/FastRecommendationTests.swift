import XCTest
import CryptoKit
import IntentCore
@testable import SmartApp

final class FastRecommendationTests: XCTestCase {
    private func app(_ id: String) -> Destination {
        Destination(id: "app:" + id, kind: .app, name: id, bundleID: id,
                    admission: TargetAdmission(basis: "standalone_application"))
    }
    private func status(_ root: URL) throws -> [String: JSONValue] {
        try JSONDecoder().decode([String: JSONValue].self,
            from: Data(contentsOf: root.appendingPathComponent("runtime/recommendation-status.json")))
    }
    func testNoWeightsDoesNotCreateOrTrainRetiredModel() throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: root) }
        let store = try EventStore(url: root.appendingPathComponent("context-v2.sqlite"))
        _ = try TargetIdentity(keyURL: root.appendingPathComponent("identity-v2.key"))
        let service = FastRecommendationService(dataDirectory: root, store: store)
        service.flush()
        let a = app("source"), b = app("destination")
        func event(_ kind: EventKind, _ time: Double, _ data: [String: JSONValue] = [:]) -> BehaviorEvent {
            BehaviorEvent(sessionID: "s", timestampMs: Int64(time), monotonicMs: time,
                          kind: kind, source: "fixture", data: data)
        }
        let events = [event(.sessionStart, 0, ["targetEligibilityPolicyVersion": .string(TargetEligibilityPolicy.version)]),
                      event(.targetUpdated, 1, ["target": try .encoded(a)]),
                      event(.targetUpdated, 2, ["target": try .encoded(b)]),
                      event(.contextChanged, 1_000, ["targetID": .string(a.id)]),
                      event(.contextChanged, 2_000, ["targetID": .string(b.id)]), event(.heartbeat, 31_000)]
        let sequences = try store.appendWithSequences(events)
        service.consumeCommitted(throughDatabaseSequence: sequences.last!)
        service.flush()
        XCTAssertNil(service.onlineRanked(current: a))
        XCTAssertFalse(service.requiresCommittedCutoff())
        XCTAssertFalse(FileManager.default.fileExists(atPath: root.appendingPathComponent("models").path))
        XCTAssertEqual(try status(root)["activeLearningModels"], .number(0))
        XCTAssertEqual(try status(root)["ready"], .bool(false))
        XCTAssertNotNil(try status(root)["error"]?.string)
    }
    func testWrongIdentityKeyDoesNotOverwriteOnlineWeightsOrUseOldModel() throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: root) }
        let store = try EventStore(url: root.appendingPathComponent("context-v2.sqlite"))
        _ = try TargetIdentity(keyURL: root.appendingPathComponent("identity-v2.key"))
        let directory = root.appendingPathComponent("models/" + OnlinePathMLPScorer.modelVersion)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        let bytes = try JSONEncoder().encode(OnlinePathTestFixture.weights(identity: "wrong-key", seed: "seed"))
        let file = directory.appendingPathComponent("state.json")
        try bytes.write(to: file)
        let service = FastRecommendationService(dataDirectory: root, store: store)
        service.flush()
        XCTAssertNil(service.onlineRanked(current: app("source")))
        XCTAssertEqual(try Data(contentsOf: file), bytes)
        XCTAssertEqual(try status(root)["activeLearningModels"], .number(0))
        XCTAssertFalse(FileManager.default.fileExists(atPath:
            root.appendingPathComponent("models/" + SemanticModelState.version).path))
    }
    func testMissingHistoryBoundaryDoesNotRelearnOrOverwriteOnlineWeights() throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: root) }
        let store = try EventStore(url: root.appendingPathComponent("context-v2.sqlite"))
        _ = try TargetIdentity(keyURL: root.appendingPathComponent("identity-v2.key"))
        let fingerprint = SHA256.hash(data: try Data(contentsOf: root.appendingPathComponent("identity-v2.key")))
            .map { String(format: "%02x", $0) }.joined()
        let directory = root.appendingPathComponent("models/" + OnlinePathMLPScorer.modelVersion)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        let bytes = try JSONEncoder().encode(OnlinePathTestFixture.weights(identity: fingerprint, seed: "absent-event"))
        let file = directory.appendingPathComponent("weights.json")
        try bytes.write(to: file)
        let service = FastRecommendationService(dataDirectory: root, store: store)
        service.flush()
        XCTAssertNil(service.onlineRanked(current: app("source")))
        XCTAssertEqual(try Data(contentsOf: file), bytes)
        XCTAssertFalse(FileManager.default.fileExists(atPath: directory.appendingPathComponent("state.json").path))
        XCTAssertEqual(try status(root)["activeLearningModels"], .number(0))
    }
    @MainActor func testRetiredSeedIsIgnoredAndHistoricalMenuStillReopensSavedTarget() throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        let suite = UUID().uuidString, defaults = try XCTUnwrap(UserDefaults(suiteName: suite))
        defer { defaults.removePersistentDomain(forName: suite); try? FileManager.default.removeItem(at: root) }
        let identity = try TargetIdentity(keyURL: root.appendingPathComponent("identity-v2.key"))
        let pages = try ["https://example.test/current", "https://example.test/saved"].map { address -> Destination in
            var page = try XCTUnwrap(identity.page(address))
            page.admission = TargetEligibilityPolicy.page(origin: page.origin, supportedSource: true, excluded: false).admission
            return page
        }
        let store = try EventStore(url: root.appendingPathComponent("context-v2.sqlite"))
        try store.append(try pages.map { page in
            BehaviorEvent(sessionID: "previous-session", kind: .targetUpdated, source: "fixture",
                          data: ["target": try .encoded(page)])
        })
        let directory = root.appendingPathComponent("models/" + SemanticModelState.version)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        let bytes = Data("retired-seed-must-not-be-decoded".utf8)
        let file = directory.appendingPathComponent("seed.json")
        try bytes.write(to: file)
        let recorder = RecorderModel(dataDirectory: root, defaults: defaults, liveObservation: false)
        recorder.start()
        XCTAssertTrue(recorder.upsert(pages[0]))
        recorder.emitContext(targetID: pages[0].id, entranceID: nil, source: "fixture", reason: "foreground_change")
        var shown: PickerSession?
        recorder.onShowPicker = { shown = $0 }
        recorder.invokePicker()
        let picker = try XCTUnwrap(shown)
        XCTAssertEqual(picker.choices.map { $0.target.id }, [pages[1].id])
        XCTAssertEqual(picker.modelVersion, "recency-fallback-v1")
        XCTAssertNil(picker.choices.first?.entrance)
        recorder.pickerClosed(picker)
        recorder.shutdown()
        XCTAssertEqual(try Data(contentsOf: file), bytes)
        XCTAssertFalse(FileManager.default.fileExists(atPath: directory.appendingPathComponent("state.json").path))
        let events = try EventStore(url: root.appendingPathComponent("context-v2.sqlite"), readOnly: true).recent(limit: 100)
        XCTAssertFalse(events.contains { [.recommendationFeedback, .recommendationShown, .menuInvoked].contains($0.kind) })
    }
}
