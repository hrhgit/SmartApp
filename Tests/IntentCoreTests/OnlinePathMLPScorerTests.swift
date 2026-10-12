import XCTest
import CryptoKit
import Darwin
@testable import IntentCore

final class OnlinePathMLPScorerTests: XCTestCase {
    private func weights() -> OnlinePathMLPWeights {
        OnlinePathMLPWeights(
            format: "online-path-mlp-f32-v1",
            modelVersion: OnlinePathMLPScorer.modelVersion,
            featureSchema: OnlinePathMLPScorer.featureSchema,
            numericFields: OnlinePathMLPScorer.numericFields,
            categoricalFields: OnlinePathMLPScorer.categoricalFields,
            categoricalBuckets: 512, tokenHashPerson: "OffRankV1",
            hashPerson: "SharedMLPPath1", hidden: 16,
            layer1Weights: Array(repeating: Array(repeating: 0, count: 543), count: 16),
            layer1Bias: Array(repeating: 0, count: 16),
            layer2Weights: Array(repeating: 0, count: 16), layer2Bias: 0,
            identityKeySHA256: "identity", trainedThroughEventID: nil)
    }

    func testPythonHashParityAndVectorLayout() throws {
        XCTAssertEqual(OnlinePathMLPScorer.stableToken("app:com.google.Chrome"),
                       "0d3de8a54a7e59eaaa585552")
        XCTAssertEqual(OnlinePathMLPScorer.stableToken("page:123"),
                       "656caec20bd6858089656c52")
        XCTAssertEqual(OnlinePathMLPScorer.stableToken(String(repeating: "x", count: 128)),
                       "1a50b6f3e70d988337ff0eaf")
        XCTAssertEqual(OnlinePathMLPScorer.stableToken(String(repeating: "x", count: 129)),
                       "1e246be3c74816080177c1f9")
        XCTAssertEqual(OnlinePathMLPScorer.stableToken(nil), "none")
        let scorer = try OnlinePathMLPScorer(weights: weights(), identityKeySHA256: "identity")
        var numeric = Dictionary(uniqueKeysWithValues: OnlinePathMLPScorer.numericFields.map { ($0, 0.0) })
        numeric["context:time_hour_sin"] = 0.5
        var categorical = Dictionary(uniqueKeysWithValues: OnlinePathMLPScorer.categoricalFields.map { ($0, "none") })
        categorical["origin_kind"] = "page"
        categorical["candidate_kind"] = "directory"
        let vector = try scorer.featureVector(for: OnlinePathMLPRow(
            numeric: numeric, categorical: categorical, layerKeys: ["page:123"]))
        XCTAssertEqual(vector.count, 543)
        XCTAssertEqual(vector[0], 0.5)
        let expected: [Int: Float] = [36: -1, 41: 1, 51: 1, 81: 1, 92: -1,
                                       170: -1, 193: 1, 238: -1, 243: 1,
                                       265: 1, 335: 1, 337: 1, 392: -1,
                                       425: 1, 480: -1]
        for bucket in 0..<512 {
            XCTAssertEqual(vector[31 + bucket], expected[bucket] ?? 0, "bucket \(bucket)")
        }
    }

    func testIdentityAndSchemaMismatchFailClosed() throws {
        var value = weights()
        XCTAssertThrowsError(try OnlinePathMLPScorer(weights: value, identityKeySHA256: "other"))
        value.numericFields.removeLast()
        XCTAssertThrowsError(try OnlinePathMLPScorer(weights: value, identityKeySHA256: "identity"))
        let scorer = try OnlinePathMLPScorer(weights: weights(), identityKeySHA256: "identity")
        XCTAssertThrowsError(try scorer.score(OnlinePathMLPRow(numeric: [:], categorical: [:], layerKeys: [])))
    }

    func testForwardPassUsesReLUAndFiniteScores() throws {
        var value = weights()
        value.layer1Weights[0][0] = 2
        value.layer1Bias[0] = -0.5
        value.layer2Weights[0] = 3
        value.layer2Bias = -1
        let scorer = try OnlinePathMLPScorer(weights: value, identityKeySHA256: "identity")
        var numeric = Dictionary(uniqueKeysWithValues: OnlinePathMLPScorer.numericFields.map { ($0, 0.0) })
        let categorical = Dictionary(uniqueKeysWithValues: OnlinePathMLPScorer.categoricalFields.map { ($0, "none") })
        numeric["context:time_hour_sin"] = 1
        XCTAssertEqual(try scorer.score(OnlinePathMLPRow(numeric: numeric, categorical: categorical, layerKeys: [])), 3.5)
        numeric["context:time_hour_sin"] = 0
        XCTAssertEqual(try scorer.score(OnlinePathMLPRow(numeric: numeric, categorical: categorical, layerKeys: [])), -1)
    }

    func testOneArrivalUpdatesOnceAndCheckpointRestores() throws {
        var value = weights()
        value.layer1Weights[0][0] = 1
        value.layer2Weights[0] = 1
        value.learningRate = 0.001
        value.negativeLimit = 8
        value.seed = 20260929
        value.beta1 = 0.9
        value.beta2 = 0.999
        value.epsilon = 1e-8
        let zeros = Array(repeating: [Float](repeating: 0, count: 543), count: 16)
        let short = [Float](repeating: 0, count: 16)
        value.optimizerState = OnlinePathMLPOptimizerState(
            layer1Weights: OnlinePathMLPMatrixMoments(expAvg: zeros, expAvgSq: zeros, step: 0),
            layer1Bias: OnlinePathMLPVectorMoments(expAvg: short, expAvgSq: short, step: 0),
            layer2Weights: OnlinePathMLPVectorMoments(expAvg: short, expAvgSq: short, step: 0),
            layer2Bias: OnlinePathMLPScalarMoments(expAvg: 0, expAvgSq: 0, step: 0))
        var learner = try OnlinePathMLPScorer(weights: value, identityKeySHA256: "identity")
        XCTAssertTrue(learner.canContinueLearning)
        let categorical = Dictionary(uniqueKeysWithValues: OnlinePathMLPScorer.categoricalFields.map { ($0, "none") })
        func row(_ first: Double) -> OnlinePathMLPRow {
            var numeric = Dictionary(uniqueKeysWithValues: OnlinePathMLPScorer.numericFields.map { ($0, 0.0) })
            numeric["context:time_hour_sin"] = first
            return OnlinePathMLPRow(numeric: numeric, categorical: categorical, layerKeys: [])
        }
        let rows = [row(1), row(0)]
        let before = try learner.score(rows[0]) - learner.score(rows[1])
        let result = try learner.learn(rows: rows, positiveIndices: [0],
                                       originalCandidateCount: 2, arrivalID: "one-arrival")
        XCTAssertTrue(result.updated)
        XCTAssertEqual(result.sampledNegativeIndices, [1])
        XCTAssertLessThan(result.loss ?? .infinity, 0.32)
        XCTAssertGreaterThan(try learner.score(rows[0]) - learner.score(rows[1]), before)
        XCTAssertEqual(learner.checkpoint.optimizerState?.layer1Weights.step, 1)
        XCTAssertThrowsError(try learner.learn(rows: rows, positiveIndices: [0],
                                               originalCandidateCount: 2, arrivalID: "one-arrival"))
        let restored = try OnlinePathMLPScorer(weights: learner.checkpoint, identityKeySHA256: "identity")
        XCTAssertEqual(try restored.score(rows[0]), try learner.score(rows[0]))
    }

    func testPrivatePythonWeightParityWhenFixtureIsProvided() throws {
        guard let directory = ProcessInfo.processInfo.environment["SMARTAPP_PRIVATE_MODEL_PARITY_DIR"] else { return }
        struct Probe: Decodable { var featureVector: [Float]; var expectedScore: Double }
        struct Fixture: Decodable { var probes: [Probe] }
        let root = URL(fileURLWithPath: directory)
        let modelURL = root.appendingPathComponent("weights.json")
        let modelWeights = try JSONDecoder().decode(OnlinePathMLPWeights.self, from: Data(contentsOf: modelURL))
        let scorer = try OnlinePathMLPScorer(contentsOf: modelURL,
                                             identityKeySHA256: modelWeights.identityKeySHA256)
        XCTAssertTrue(scorer.canContinueLearning)
        let fixture = try JSONDecoder().decode(Fixture.self,
            from: Data(contentsOf: root.appendingPathComponent("native-parity-private.json")))
        XCTAssertEqual(fixture.probes.count, 3)
        for probe in fixture.probes {
            XCTAssertEqual(try scorer.scoreFeatureVector(probe.featureVector),
                           probe.expectedScore, accuracy: 0.0001)
        }
    }

    func testPrivateRealCutoffParityWhenFixtureIsProvided() throws {
        guard let directory = ProcessInfo.processInfo.environment["SMARTAPP_PRIVATE_MODEL_PARITY_DIR"] else { return }
        struct ExpectedAction: Decodable {
            struct LayerShape: Decodable { var kind: String; var path: [String] }
            var actionSHA256: String
            var vector: [Float]
            var score: Double
            var numeric: [Double]
            var categorical: [String]
            var layerTokens: [String]
            var kind: String
            var source: String
            var hasExplicitParent: Bool
            var hasOwner: Bool
            var layerShapes: [LayerShape]
        }
        struct ExpectedCase: Decodable {
            var eventIndex: Int
            var eventID: String
            var actions: [ExpectedAction]
        }
        struct ExpectedFixture: Decodable {
            var snapshot: String
            var cases: [ExpectedCase]
        }
        let root = URL(fileURLWithPath: directory)
        let fixtureName = ProcessInfo.processInfo.environment["SMARTAPP_PRIVATE_PARITY_FIXTURE"]
            ?? "real-cutoff-parity-private-v5.json"
        let fixture = try JSONDecoder().decode(ExpectedFixture.self,
            from: Data(contentsOf: root.appendingPathComponent(fixtureName)))
        let key = try Data(contentsOf: root.deletingLastPathComponent().deletingLastPathComponent()
            .appendingPathComponent("identity-v2.key"))
        let fingerprint = SHA256.hash(data: key).map { String(format: "%02x", $0) }.joined()
        let scorer = try OnlinePathMLPScorer(contentsOf: root.appendingPathComponent("weights.json"),
                                             identityKeySHA256: fingerprint)
        let raw = try Data(contentsOf: URL(fileURLWithPath: fixture.snapshot)
            .appendingPathComponent("events.jsonl"))
        let eventLines = String(decoding: raw, as: UTF8.self).split(separator: "\n")
        let casesByIndex = Dictionary(uniqueKeysWithValues: fixture.cases.map { ($0.eventIndex, $0) })
        var history = OnlinePathInputHistory()
        var metadata: [String: Destination] = [:]
        var comparedCases = 0, unsupportedCases = 0
        var expectedActions = 0, observedActions = 0, commonActions = 0
        var differingVectors = 0, differingVectorElements = 0, differingScores = 0, matchingTop4 = 0
        var maxVectorDifference = 0.0, maxScoreDifference = 0.0
        var firstBuildMs: [Double] = [], cachedBuildMs: [Double] = [], scoreMs: [Double] = []
        var numericFieldMismatches: [String: Int] = [:]
        var categoricalFieldMismatches: [String: Int] = [:]
        var layerTokenMismatches = 0, layerSameSetMismatches = 0
        var extraCandidateKinds: [String: Int] = [:]
        var layerMismatchShapes: [String: Int] = [:]
        var directFrequencyDirections: [String: Int] = [:]
        var pathFrequencyDirections: [String: Int] = [:]
        var extraDirectoryShapes: [String: Int] = [:]
        var firstLayerExamples: [String: String] = [:]
        var perCase: [String] = []
        for (index, line) in eventLines.enumerated() {
            if index > (fixture.cases.map(\.eventIndex).max() ?? -1) { break }
            let event = try JSONDecoder().decode(BehaviorEvent.self, from: Data(line.utf8))
            if event.kind == .targetUpdated, let encoded = event.data["target"],
               let target = try? encoded.decoded(Destination.self) {
                metadata[target.id] = target
            }
            history.consume(event)
            guard let expected = casesByIndex[index] else { continue }
            XCTAssertEqual(event.id, expected.eventID)
            guard let targetID = event.data["targetID"]?.string,
                  let current = metadata[targetID] else {
                unsupportedCases += 1; continue
            }
            let started = ProcessInfo.processInfo.systemUptime
            guard let candidates = try? history.candidates(current: current) else {
                unsupportedCases += 1; continue
            }
            firstBuildMs.append((ProcessInfo.processInfo.systemUptime - started) * 1_000)
            let cachedStarted = ProcessInfo.processInfo.systemUptime
            _ = try history.candidates(current: current)
            cachedBuildMs.append((ProcessInfo.processInfo.systemUptime - cachedStarted) * 1_000)
            comparedCases += 1
            expectedActions += expected.actions.count
            observedActions += candidates.count
            let expectedByHash = Dictionary(uniqueKeysWithValues:
                expected.actions.map { ($0.actionSHA256, $0) })
            var actualScores: [(String, Double)] = []
            var caseCommon = 0, caseVectors = 0, caseScores = 0
            let scoreStarted = ProcessInfo.processInfo.systemUptime
            for candidate in candidates {
                let hash = SHA256.hash(data: Data(candidate.actionID.utf8))
                    .map { String(format: "%02x", $0) }.joined()
                let vector = try scorer.featureVector(for: candidate.row)
                let score = try scorer.scoreFeatureVector(vector)
                actualScores.append((hash, score))
                guard let other = expectedByHash[hash] else {
                    let kind = candidate.recordedTarget?.kind.rawValue ?? candidate.derivedKind?.rawValue ?? "unknown"
                    extraCandidateKinds[kind, default: 0] += 1
                    if kind == "directory", let locator = candidate.executableLocator,
                       let parts = URLComponents(string: locator) {
                        let encoded = parts.percentEncodedPath
                        let shape = [
                            "source=\(candidate.source)",
                            "root=\(encoded == "/")", "trailing=\(encoded.hasSuffix("/"))",
                            "localhost=\(parts.host?.lowercased() == "localhost")",
                            "otherhost=\(parts.host != nil && parts.host != "" && parts.host?.lowercased() != "localhost")",
                            "doubleslash=\(encoded.contains("//"))",
                            "encodedspace=\(encoded.lowercased().contains("%20"))",
                        ].joined(separator: ",")
                        extraDirectoryShapes[shape, default: 0] += 1
                    }
                    continue
                }
                commonActions += 1
                caseCommon += 1
                for (column, name) in OnlinePathMLPScorer.numericFields.enumerated() {
                    let actual = Float(candidate.row.numeric[name] ?? .nan)
                    if column >= other.numeric.count ||
                        abs(Double(actual) - Double(Float(other.numeric[column]))) > 0.00001 {
                        numericFieldMismatches[name, default: 0] += 1
                        if name.contains("direct_count"), column < other.numeric.count {
                            let direction = actual > Float(other.numeric[column]) ? "high" : "low"
                            directFrequencyDirections["\(name):\(other.kind):\(direction)", default: 0] += 1
                        }
                        if name.contains("path_count"), column < other.numeric.count {
                            let direction = actual > Float(other.numeric[column]) ? "high" : "low"
                            pathFrequencyDirections["\(name):\(other.kind):\(direction)", default: 0] += 1
                        }
                    }
                }
                for (column, name) in OnlinePathMLPScorer.categoricalFields.enumerated() {
                    if column >= other.categorical.count ||
                        candidate.row.categorical[name] != other.categorical[column] {
                        categoricalFieldMismatches[name, default: 0] += 1
                    }
                }
                let actualLayerTokens = candidate.row.layerKeys.map { OnlinePathMLPScorer.stableToken($0) }
                if actualLayerTokens != other.layerTokens {
                    layerTokenMismatches += 1
                    if Set(actualLayerTokens) == Set(other.layerTokens) { layerSameSetMismatches += 1 }
                    let delta = candidate.row.layerKeys.count - other.layerTokens.count
                    let shape = "kind=\(other.kind),source=\(other.source),delta=\(delta)," +
                        "explicit=\(other.hasExplicitParent),owner=\(other.hasOwner)"
                    layerMismatchShapes[shape, default: 0] += 1
                    if firstLayerExamples[other.kind] == nil {
                        let expectedShapes = other.layerShapes.map { "\($0.kind):\($0.path.joined(separator: ">"))" }
                        let shapesByToken = Dictionary(uniqueKeysWithValues:
                            zip(other.layerTokens, expectedShapes))
                        let actualShapes = actualLayerTokens.map { shapesByToken[$0] ?? "other" }
                        firstLayerExamples[other.kind] = "cutoff=\(index),action_sha=\(hash.prefix(12))," +
                            "swift=\(actualShapes),python=\(expectedShapes)," +
                            "swift_tokens=\(actualLayerTokens.map { String($0.prefix(12)) })," +
                            "python_tokens=\(other.layerTokens.map { String($0.prefix(12)) })"
                    }
                }
                if vector.count != other.vector.count {
                    differingVectors += 1
                    differingVectorElements += max(vector.count, other.vector.count)
                    continue
                }
                var bad = false
                for column in vector.indices {
                    let difference = abs(Double(vector[column]) - Double(other.vector[column]))
                    maxVectorDifference = max(maxVectorDifference, difference)
                    if difference > 0.00001 { bad = true; differingVectorElements += 1 }
                }
                if bad { differingVectors += 1; caseVectors += 1 }
                let scoreDifference = abs(score - other.score)
                maxScoreDifference = max(maxScoreDifference, scoreDifference)
                if scoreDifference > 0.001 { differingScores += 1; caseScores += 1 }
            }
            scoreMs.append((ProcessInfo.processInfo.systemUptime - scoreStarted) * 1_000)
            let expectedTop4 = expected.actions.sorted { $0.score > $1.score }.prefix(4).map(\.actionSHA256)
            let actualTop4 = actualScores.sorted { $0.1 > $1.1 }.prefix(4).map(\.0)
            if expectedTop4 == actualTop4 { matchingTop4 += 1 }
            perCase.append("\(index):expected=\(expected.actions.count),observed=\(candidates.count)," +
                           "common=\(caseCommon),vectors=\(caseVectors),scores=\(caseScores)," +
                           "top4=\(expectedTop4 == actualTop4)," +
                           "dwell_swift=\(candidates.first?.row.numeric["context:last_confirmed_dwell_log_s"] ?? -1)," +
                           "dwell_python=\(expected.actions.first?.numeric[4] ?? -1)")
        }
        print("REAL_CUTOFF_PARITY cases=\(comparedCases) unsupported=\(unsupportedCases) " +
              "expected_actions=\(expectedActions) observed_actions=\(observedActions) " +
              "common_actions=\(commonActions) differing_vectors=\(differingVectors) " +
              "differing_elements=\(differingVectorElements) differing_scores=\(differingScores) " +
              "matching_top4=\(matchingTop4) max_vector_diff=\(maxVectorDifference) " +
              "max_score_diff=\(maxScoreDifference) first_build_ms=\(firstBuildMs) " +
              "cached_build_ms=\(cachedBuildMs) score_ms=\(scoreMs) " +
              "numeric_fields=\(numericFieldMismatches) category_fields=\(categoricalFieldMismatches) " +
              "layer_token_mismatches=\(layerTokenMismatches) layer_same_set=\(layerSameSetMismatches) " +
              "extra_kinds=\(extraCandidateKinds) " +
              "layer_shapes=\(layerMismatchShapes) direct_directions=\(directFrequencyDirections) " +
              "path_directions=\(pathFrequencyDirections) " +
              "extra_directory_shapes=\(extraDirectoryShapes) first_layer_examples=\(firstLayerExamples) " +
              "per_case=\(perCase)")
        XCTAssertEqual(comparedCases + unsupportedCases, fixture.cases.count)
    }

    /// Optional local-only startup measurement. The snapshot stays in the
    /// user's private data directory; output contains counts and timing only.
    func testPrivateStartupReplayBenchmarkWhenRequested() throws {
        guard ProcessInfo.processInfo.environment["SMARTAPP_PRIVATE_REPLAY_BENCH"] == "1",
              let directory = ProcessInfo.processInfo.environment["SMARTAPP_PRIVATE_MODEL_PARITY_DIR"] else { return }
        let run = try JSONSerialization.jsonObject(with: Data(contentsOf:
            URL(fileURLWithPath: directory).appendingPathComponent("run.json")))
        guard let manifest = run as? [String: Any], let snapshot = manifest["snapshot"] as? String else {
            XCTFail("Private run manifest has no snapshot"); return
        }
        let path = URL(fileURLWithPath: snapshot).appendingPathComponent("events.jsonl").path
        guard let handle = fopen(path, "r") else { XCTFail("Private event snapshot could not open"); return }
        defer { fclose(handle) }
        var line: UnsafeMutablePointer<CChar>?
        var capacity = 0
        defer { free(line) }
        var history = OnlinePathInputHistory()
        var learner = OnlinePathFeedbackLearner()
        let decoder = JSONDecoder()
        let component = ProcessInfo.processInfo.environment["SMARTAPP_PRIVATE_REPLAY_COMPONENT"] ?? "both"
        let usePool = ProcessInfo.processInfo.environment["SMARTAPP_PRIVATE_REPLAY_POOL"] == "1"
        var eventCount = 0, feedbackCount = 0
        var sampledPeakRSS: [Int64] = []
        var baseline = rusage()
        XCTAssertEqual(getrusage(RUSAGE_SELF, &baseline), 0)
        let started = ProcessInfo.processInfo.systemUptime
        while true {
            let count = getline(&line, &capacity, handle)
            if count < 0 { break }
            let process = {
                let event = try decoder.decode(BehaviorEvent.self, from: Data(bytes: line!, count: Int(count)))
                if component != "learner" { history.consume(event) }
                if component != "history" { feedbackCount += learner.consume(event).count }
            }
            if usePool { try autoreleasepool(invoking: process) } else { try process() }
            eventCount += 1
            if eventCount % 5_000 == 0 {
                var intermediate = rusage()
                XCTAssertEqual(getrusage(RUSAGE_SELF, &intermediate), 0)
                sampledPeakRSS.append(Int64(intermediate.ru_maxrss))
            }
        }
        let elapsed = (ProcessInfo.processInfo.systemUptime - started) * 1_000
        var finished = rusage()
        XCTAssertEqual(getrusage(RUSAGE_SELF, &finished), 0)
        print("PRIVATE_REPLAY_BENCH component=\(component) pool=\(usePool) events=\(eventCount) feedback=\(feedbackCount) " +
              "elapsed_ms=\(elapsed) baseline_peak_rss_bytes=\(baseline.ru_maxrss) " +
              "process_peak_rss_bytes=\(finished.ru_maxrss) sampled_peak_rss_bytes=\(sampledPeakRSS)")
        XCTAssertGreaterThan(eventCount, 0)
    }
}
