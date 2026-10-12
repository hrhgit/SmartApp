import XCTest
import CryptoKit
@testable import IntentCore
@testable import SmartApp

final class DynamicRecommendationIntegrationTests: XCTestCase {
    private func seededMargins() throws -> ScoreMarginConformalState {
        let context = try ScoreMarginContext(modelVersion: OnlinePathMLPScorer.modelVersion,
            featureSchema: OnlinePathMLPScorer.featureSchema, matchingRule: OnlinePathInputHistory.usefulActionRuleVersion + ":" + MainMenuTargetScope.version)
        var controller = try AdaptiveScoreMarginConformal(context: context, targetMissRate: 0.1,
                                                          learningRate: 0.005, startSequence: 0)
        for index in 0..<10 {
            let p = try ScoreMarginPrediction(predictionID: "calibration-\(index)", context: context,
                modelRevision: "fixture", trainedThroughSequence: 0, cutoffSequence: Int64(index * 2 + 1),
                candidates: [.init(actionID: "wrong", score: 2), .init(actionID: "truth", score: 1.5)])
            let decision = try controller.select(p, maximumCount: 6)
            try controller.observe(decision, outcomeID: "arrival-\(index)", outcomeSequence: Int64(index * 2 + 2),
                                   acceptableActionIDs: ["truth"])
        }
        XCTAssertEqual(controller.threshold, .finite(0.5))
        return try controller.checkpoint(maximumCount: 6)
    }

    private func fixture(at root: URL, malformed: Bool = false) throws -> (EventStore, Destination, Destination) {
        let identity = try TargetIdentity(keyURL: root.appendingPathComponent("identity-v2.key"))
        let pages = try (0..<9).map { index -> Destination in
            var page = try XCTUnwrap(identity.page("https://example-\(index).test/project"))
            page.admission = TargetEligibilityPolicy.page(origin: page.origin, supportedSource: true, excluded: false).admission
            return page
        }
        let coverage: JSONValue = .object(["version": .string(StateCoverage.foregroundOnlyVersion),
            "status": .string("foreground_only"), "sources": .array([
                .object(["source": .string("fixture"), "status": .string("ready")])])])
        let start: [String: JSONValue] = ["targetEligibilityPolicyVersion": .string(TargetEligibilityPolicy.version),
                                          "stateCoverageVersion": .string(StateCoverage.foregroundOnlyVersion)]
        var events: [BehaviorEvent] = []
        func append(_ kind: EventKind, _ data: [String: JSONValue] = [:], session: String = "history") {
            let count = events.count
            var event = BehaviorEvent(sessionID: session, timestampMs: Int64(count * 200),
                monotonicMs: Double(count * 200), kind: kind, source: "fixture", data: data)
            event.sequence = count + 1
            events.append(event)
        }
        func arrive(_ page: Destination, session: String = "history") {
            append(.contextChanged, ["targetID": .string(page.id), "inputCoverage": coverage,
                                      "reason": .string("foreground_change")], session: session)
        }
        append(.sessionStart, start)
        for page in pages { append(.targetUpdated, ["target": try .encoded(page)]) }
        for _ in 0..<16 { arrive(pages[1]); arrive(pages[0]) }
        for page in pages.dropFirst(2) { arrive(page); arrive(pages[0]) }
        append(.sessionEnd)
        let seed = try XCTUnwrap(events.last)
        append(.sessionStart, start, session: "current")
        arrive(pages[0], session: "current")
        let store = try EventStore(url: root.appendingPathComponent("context-v2.sqlite"))
        try store.append(events)
        let fingerprint = SHA256.hash(data: try Data(contentsOf: root.appendingPathComponent("identity-v2.key")))
            .map { String(format: "%02x", $0) }.joined()
        var weights = OnlinePathTestFixture.weights(identity: fingerprint, seed: seed.id)
        let direct = try XCTUnwrap(OnlinePathMLPScorer.numericFields.firstIndex(of: "candidate:direct_count_all_log"))
        weights.layer1Weights[0][direct] = 1
        weights.scoreMarginCalibration = try seededMargins()
        if malformed { weights.scoreMarginCalibration?.margins = [-1] }
        let directory = root.appendingPathComponent("models/" + OnlinePathMLPScorer.modelVersion)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        try JSONEncoder().encode(weights).write(to: directory.appendingPathComponent("weights.json"))
        return (store, pages[0], pages[1])
    }

    func testScoredModelUsesRestoredMarginAndRepeatedMenuReadsNeverLearn() throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: root) }
        let (store, current, frequent) = try fixture(at: root)
        let service = FastRecommendationService(dataDirectory: root, store: store)
        service.flush()
        let ranked = try XCTUnwrap(service.onlineMenuRanking(current: current))
        XCTAssertGreaterThan(ranked.ranked.count, 6)
        XCTAssertEqual(ranked.preview.selected.map(\.actionID), [frequent.id])
        for _ in 0..<4 {
            XCTAssertEqual(try XCTUnwrap(service.onlineMenuRanking(current: current)).preview.selected.count, 1)
        }
        service.flush()
        XCTAssertFalse(FileManager.default.fileExists(atPath:
            root.appendingPathComponent("models/" + OnlinePathMLPScorer.modelVersion + "/state.json").path))
        let status = try JSONDecoder().decode([String: JSONValue].self,
            from: Data(contentsOf: root.appendingPathComponent("runtime/recommendation-status.json")))
        XCTAssertEqual(status["marginCalibrationSamples"], .number(10))
        XCTAssertEqual(status["marginFeedbackCount"], .number(10))
        XCTAssertEqual(status["dynamicMenuMaximumItems"], .number(6))
        XCTAssertEqual(status["marginCalibrationError"], .null)
    }

    @MainActor func testRecorderShowsOneCalibratedChoiceWithoutRefillingFourAndCancelWritesNoMenuData() throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        let suite = UUID().uuidString
        let defaults = try XCTUnwrap(UserDefaults(suiteName: suite))
        defer { defaults.removePersistentDomain(forName: suite); try? FileManager.default.removeItem(at: root) }
        let (_, current, frequent) = try fixture(at: root)
        let recorder = RecorderModel(dataDirectory: root, defaults: defaults, liveObservation: false)
        recorder.start()
        XCTAssertTrue(recorder.upsert(current))
        recorder.emitContext(targetID: current.id, entranceID: nil, source: "fixture", reason: "foreground_change")
        var shown: PickerSession?
        recorder.onShowPicker = { shown = $0 }
        recorder.invokePicker()
        let picker = try XCTUnwrap(shown)
        XCTAssertEqual(picker.modelVersion, OnlinePathMLPScorer.modelVersion)
        XCTAssertEqual(picker.choices.map { $0.target.id }, [frequent.id])
        recorder.pickerShown(picker)
        recorder.pickerClosed(picker)
        recorder.shutdown()
        let events = try EventStore(url: root.appendingPathComponent("context-v2.sqlite"), readOnly: true).recent(limit: 1_000)
        XCTAssertFalse(events.contains { [.menuInvoked, .recommendationShown, .recommendationFeedback].contains($0.kind) })
    }

    func testMalformedMarginStatePreservesModelAndReportsCalibrationReset() throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: root) }
        let (store, current, _) = try fixture(at: root, malformed: true)
        let service = FastRecommendationService(dataDirectory: root, store: store)
        service.flush()
        let ranked = try XCTUnwrap(service.onlineMenuRanking(current: current))
        XCTAssertEqual(ranked.preview.selected.count, 6)
        let status = try JSONDecoder().decode([String: JSONValue].self,
            from: Data(contentsOf: root.appendingPathComponent("runtime/recommendation-status.json")))
        XCTAssertEqual(status["activeLearningModels"], .number(1))
        XCTAssertEqual(status["marginCalibrationSamples"], .number(0))
        XCTAssertNotNil(status["marginCalibrationError"]?.string)
    }
}
