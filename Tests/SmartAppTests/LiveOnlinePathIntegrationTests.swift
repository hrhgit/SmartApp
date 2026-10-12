import XCTest
import CryptoKit
@testable import IntentCore
@testable import SmartApp

final class LiveOnlinePathIntegrationTests: XCTestCase {
    private func page(_ address: String, using identity: TargetIdentity) throws -> Destination {
        var result = try XCTUnwrap(identity.page(address))
        result.admission = TargetEligibilityPolicy.page(
            origin: result.origin, supportedSource: true, excluded: false).admission
        return result
    }

    private var coverage: JSONValue {
        .object(["version": .string("foreground-coverage-v2"),
                 "status": .string("foreground_only"),
                 "sources": .array([.object(["source": .string("fixture"),
                                               "status": .string("ready")])])])
    }

    @MainActor func testPrivateWeightsRankDerivedParentsAndContinueLearningOnce() throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: root) }
        let store = try EventStore(url: root.appendingPathComponent("context-v2.sqlite"))
        let identity = try TargetIdentity(keyURL: root.appendingPathComponent("identity-v2.key"))
        let key = try Data(contentsOf: root.appendingPathComponent("identity-v2.key"))
        let fingerprint = SHA256.hash(data: key).map { String(format: "%02x", $0) }.joined()
        let current = try page("https://example.test/course/lesson", using: identity)
        let next = try page("https://other.test/project", using: identity)
        func event(_ kind: EventKind, _ time: Int64, _ data: [String: JSONValue] = [:],
                   session: String = "fixture-session") -> BehaviorEvent {
            BehaviorEvent(sessionID: session, timestampMs: time,
                          monotonicMs: Double(time), kind: kind, source: "fixture", data: data)
        }
        let startData: [String: JSONValue] =
                         ["targetEligibilityPolicyVersion": .string(TargetEligibilityPolicy.version),
                          "stateCoverageVersion": .string("foreground-coverage-v2")]
        let seed = event(.sessionStart, 300, startData)
        let beginning = [event(.sessionStart, 0, startData, session: "history"),
                         event(.targetUpdated, 1, ["target": try .encoded(current)], session: "history"),
                         event(.targetUpdated, 2, ["target": try .encoded(next)], session: "history"),
                         event(.contextChanged, 100,
                               ["targetID": .string(next.id), "inputCoverage": coverage,
                                "reason": .string("foreground_change")], session: "history"),
                         event(.sessionEnd, 200, session: "history"), seed,
                         event(.targetUpdated, 301, ["target": try .encoded(current)]),
                         event(.targetUpdated, 302, ["target": try .encoded(next)]),
                         event(.contextChanged, 1_000,
                               ["targetID": .string(current.id), "inputCoverage": coverage,
                                "reason": .string("foreground_change")])].enumerated().map { index, raw in
            var value = raw
            value.sequence = raw.sessionID == "history" ? index + 1 : index - 4
            return value
        }
        _ = try store.appendWithSequences(beginning)
        let directory = root.appendingPathComponent("models/" + OnlinePathMLPScorer.modelVersion)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        try JSONEncoder().encode(OnlinePathTestFixture.weights(identity: fingerprint, seed: seed.id))
            .write(to: directory.appendingPathComponent("weights.json"))
        // Corrupt retired models must neither block the only active model nor
        // be repaired, migrated or rewritten by startup/continued learning.
        let retiredFiles = try [SemanticModelState.version, SemanticModelState.previousVersion,
                                SemanticModelState.legacyVersion].map { version -> URL in
            let oldDirectory = root.appendingPathComponent("models/" + version)
            try FileManager.default.createDirectory(at: oldDirectory, withIntermediateDirectories: true)
            let file = oldDirectory.appendingPathComponent("state.json")
            try Data("retired-invalid-model".utf8).write(to: file)
            return file
        }
        let service = FastRecommendationService(dataDirectory: root, store: store)
        service.flush()
        let ranked = try XCTUnwrap(service.onlineRanked(current: current))
        XCTAssertTrue(ranked.contains { $0.recordedTarget?.id == next.id })
        XCTAssertTrue(ranked.contains { $0.source == "derived_from_known_locator" &&
            $0.executableLocator == "https://example.test/course/" })
        XCTAssertFalse(FileManager.default.fileExists(atPath: directory.appendingPathComponent("state.json").path))

        let later = [event(.contextChanged, 2_000,
                           ["targetID": .string(next.id), "inputCoverage": coverage,
                            "reason": .string("foreground_change")]),
                     event(.heartbeat, 31_500)].enumerated().map { index, raw in
            var value = raw
            value.sequence = beginning.last!.sequence! + index + 1
            return value
        }
        let sequences = try store.appendWithSequences(later)
        service.consumeCommitted(throughDatabaseSequence: sequences.last!)
        service.consumeCommitted(throughDatabaseSequence: sequences.last!)
        service.flush()
        let saved = try JSONDecoder().decode(OnlinePathMLPWeights.self,
            from: Data(contentsOf: directory.appendingPathComponent("state.json")))
        XCTAssertEqual(saved.optimizerState?.layer1Weights.step, 1)
        XCTAssertEqual(saved.scoreMarginCalibration?.feedbackCount, 1)
        XCTAssertEqual(saved.scoreMarginCalibration?.targetMissRate, 0.1)
        XCTAssertEqual(saved.scoreMarginCalibration?.policyMaximumCount, 6)
        XCTAssertEqual(saved.trainedThroughEventID, later.last?.id)
        let restarted = FastRecommendationService(dataDirectory: root, store: store)
        restarted.flush()
        restarted.consumeCommitted(throughDatabaseSequence: sequences.last!)
        restarted.flush()
        XCTAssertNotNil(restarted.onlineRanked(current: next))
        let resumed = try JSONDecoder().decode(OnlinePathMLPWeights.self,
            from: Data(contentsOf: directory.appendingPathComponent("state.json")))
        XCTAssertEqual(resumed.optimizerState?.layer1Weights.step, 1)
        XCTAssertEqual(resumed.scoreMarginCalibration?.feedbackCount, 1)

        // A confirmed first step remains useful when a later unknown gap
        // prevents the original 30-second sample from completing.
        let firstStep = [event(.contextChanged, 32_000,
                               ["targetID": .string(current.id), "inputCoverage": coverage,
                                "reason": .string("foreground_change")]),
                         event(.contextChanged, 32_500,
                               ["targetID": .string(next.id), "inputCoverage": coverage,
                                "reason": .string("foreground_change")]),
                         event(.captureGap, 33_000,
                               ["reason": .string("unknown_failure")])].enumerated().map { index, raw in
            var value = raw
            value.sequence = later.last!.sequence! + index + 1
            return value
        }
        let firstStepSequences = try store.appendWithSequences(firstStep)
        restarted.consumeCommitted(throughDatabaseSequence: firstStepSequences.last!)
        restarted.flush()
        let afterFirstStep = try JSONDecoder().decode(OnlinePathMLPWeights.self,
            from: Data(contentsOf: directory.appendingPathComponent("state.json")))
        XCTAssertEqual(afterFirstStep.optimizerState?.layer1Weights.step, 2)
        XCTAssertEqual(afterFirstStep.trainedThroughEventID, firstStep.last?.id)

        let suite = UUID().uuidString
        let defaults = try XCTUnwrap(UserDefaults(suiteName: suite))
        defer { defaults.removePersistentDomain(forName: suite) }
        let recorder = RecorderModel(dataDirectory: root, defaults: defaults, liveObservation: false)
        recorder.start()
        XCTAssertTrue(recorder.upsert(next))
        recorder.emitContext(targetID: next.id, entranceID: nil,
                             source: "fixture", reason: "foreground_change")
        var shown: PickerSession?
        recorder.onShowPicker = { shown = $0 }
        recorder.invokePicker()
        XCTAssertEqual(shown?.modelVersion, OnlinePathMLPScorer.modelVersion)
        XCTAssertTrue(shown?.choices.contains(where: { $0.derivedActionID != nil }) == true)
        if let shown { recorder.pickerClosed(shown) }
        recorder.shutdown()
        for file in retiredFiles {
            XCTAssertEqual(try Data(contentsOf: file), Data("retired-invalid-model".utf8))
        }
        let status = try JSONDecoder().decode([String: JSONValue].self,
            from: Data(contentsOf: root.appendingPathComponent("runtime/recommendation-status.json")))
        XCTAssertEqual(status["version"]?.string, OnlinePathMLPScorer.modelVersion)
        XCTAssertEqual(status["activeLearningModels"], .number(1))
        XCTAssertNil(status["records"])
    }

    func testLatestCommittedPositionDrainsBurstWithoutDroppingPendingInputs() throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: root) }
        let store = try EventStore(url: root.appendingPathComponent("context-v2.sqlite"))
        _ = try TargetIdentity(keyURL: root.appendingPathComponent("identity-v2.key"))
        let key = try Data(contentsOf: root.appendingPathComponent("identity-v2.key"))
        let fingerprint = SHA256.hash(data: key).map { String(format: "%02x", $0) }.joined()
        let a = Destination(id: "app:a", kind: .app, name: "A", bundleID: "a",
                            admission: TargetAdmission(basis: "standalone_application"))
        let b = Destination(id: "app:b", kind: .app, name: "B", bundleID: "b",
                            admission: TargetAdmission(basis: "standalone_application"))
        let c = Destination(id: "app:c", kind: .app, name: "C", bundleID: "c",
                            admission: TargetAdmission(basis: "standalone_application"))
        func event(_ kind: EventKind, _ number: Int, _ data: [String: JSONValue] = [:]) -> BehaviorEvent {
            var result = BehaviorEvent(sessionID: "burst", timestampMs: Int64(number), monotonicMs: Double(number),
                                       kind: kind, source: "fixture", data: data)
            result.sequence = number + 1
            return result
        }
        let startData: [String: JSONValue] =
            ["targetEligibilityPolicyVersion": .string(TargetEligibilityPolicy.version),
             "stateCoverageVersion": .string(StateCoverage.foregroundOnlyVersion)]
        let seed = event(.sessionEnd, 7)
        try store.append([event(.sessionStart, 0, startData),
                          event(.targetUpdated, 1, ["target": try .encoded(a)]),
                          event(.targetUpdated, 2, ["target": try .encoded(b)]),
                          event(.targetUpdated, 3, ["target": try .encoded(c)])] +
                         [a, b, c].enumerated().map { index, target in
                             event(.contextChanged, index + 4,
                                   ["targetID": .string(target.id), "inputCoverage": coverage,
                                    "reason": .string("foreground_change")])
                         } + [seed])
        let directory = root.appendingPathComponent("models/" + OnlinePathMLPScorer.modelVersion)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        try JSONEncoder().encode(OnlinePathTestFixture.weights(identity: fingerprint, seed: seed.id))
            .write(to: directory.appendingPathComponent("weights.json"))
        let service = FastRecommendationService(dataDirectory: root, store: store)
        service.flush()
        let burst = [event(.sessionStart, 8, startData)] + (0..<520).map { index in
            event(.contextChanged, index + 9,
                  ["targetID": .string(index % 2 == 0 ? a.id : b.id), "inputCoverage": coverage,
                   "reason": .string("foreground_change")])
        } + [event(.heartbeat, 31_000)]
        let positions = try store.appendWithSequences(burst)
        // No notification for intermediate rows: the newest committed mark
        // must catch up in write order. Older/duplicate marks add no updates.
        for _ in 0..<2_000 {
            service.consumeCommitted(throughDatabaseSequence: positions.last!)
            service.consumeCommitted(throughDatabaseSequence: positions.first!)
        }
        service.flush()
        let status = try JSONDecoder().decode([String: JSONValue].self,
            from: Data(contentsOf: root.appendingPathComponent("runtime/recommendation-status.json")))
        XCTAssertEqual(status["ready"], .bool(true), "\(status)")
        XCTAssertEqual(status["error"], .null, "\(status)")
        XCTAssertEqual(status["onlinePathRunUpdates"], .number(519), "\(status)")
        let checkpoint = try JSONDecoder().decode(OnlinePathMLPWeights.self,
            from: Data(contentsOf: directory.appendingPathComponent("state.json")))
        XCTAssertEqual(checkpoint.optimizerState?.layer1Weights.step, 519)
        XCTAssertEqual(checkpoint.trainedThroughEventID, burst.last?.id)
        XCTAssertNotNil(service.onlineRanked(current: b))
        service.consumeCommitted(throughDatabaseSequence: positions.last!)
        service.flush()
        let repeated = try Data(contentsOf: directory.appendingPathComponent("state.json"))
        XCTAssertEqual(try JSONDecoder().decode(OnlinePathMLPWeights.self, from: repeated).optimizerState?.layer1Weights.step, 519)
    }
}
