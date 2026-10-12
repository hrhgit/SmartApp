import Foundation

/// One prediction-time candidate row. The caller constructs these fields from
/// confirmed foreground history only; this type contains no outcome or menu data.
public struct OnlinePathMLPRow: Sendable {
    // Keep valid rows in schema order. A frozen input no longer owns two hash
    // tables (and repeats 45 field-name strings) for every candidate. Public
    // access remains compatible, including mutation and invalid-row checks.
    private var numericValues: [Double]?
    private var categoricalValues: [String]?
    private var invalidNumeric: [String: Double]?
    private var invalidCategorical: [String: String]?
    public var numeric: [String: Double] {
        get {
            guard let numericValues else { return invalidNumeric ?? [:] }
            return Dictionary(uniqueKeysWithValues: zip(OnlinePathMLPScorer.numericFields, numericValues))
        }
        set {
            if newValue.count == OnlinePathMLPScorer.numericFields.count,
               OnlinePathMLPScorer.numericFields.allSatisfy({ newValue[$0] != nil }) {
                numericValues = OnlinePathMLPScorer.numericFields.map { newValue[$0]! }
                invalidNumeric = nil
            } else { numericValues = nil; invalidNumeric = newValue }
        }
    }
    public var categorical: [String: String] {
        get {
            guard let categoricalValues else { return invalidCategorical ?? [:] }
            return Dictionary(uniqueKeysWithValues: zip(OnlinePathMLPScorer.categoricalFields, categoricalValues))
        }
        set {
            if newValue.count == OnlinePathMLPScorer.categoricalFields.count,
               OnlinePathMLPScorer.categoricalFields.allSatisfy({ newValue[$0] != nil }) {
                categoricalValues = OnlinePathMLPScorer.categoricalFields.map { newValue[$0]! }
                invalidCategorical = nil
            } else { categoricalValues = nil; invalidCategorical = newValue }
        }
    }
    public var layerKeys: [String]

    public init(numeric: [String: Double], categorical: [String: String], layerKeys: [String]) {
        self.layerKeys = layerKeys
        self.numeric = numeric
        self.categorical = categorical
    }

    init(numericValues: [Double], categoricalValues: [String], layerKeys: [String]) {
        precondition(numericValues.count == OnlinePathMLPScorer.numericFields.count &&
                     categoricalValues.count == OnlinePathMLPScorer.categoricalFields.count)
        self.numericValues = numericValues
        self.categoricalValues = categoricalValues
        self.layerKeys = layerKeys
    }

    fileprivate func orderedValues() throws -> (numeric: [Double], categorical: [String]) {
        guard let numericValues, let categoricalValues,
              numericValues.allSatisfy(\.isFinite) else { throw OnlinePathMLPError.invalidFeatureRow }
        return (numericValues, categoricalValues)
    }
}

/// Private, user-specific model parameters exported by shared_online_path_mlp.py.
/// The file is stored in the personal data directory, never in the app bundle.
public struct OnlinePathMLPWeights: Codable, Sendable {
    public var format: String
    public var modelVersion: String
    public var featureSchema: String
    public var numericFields: [String]
    public var categoricalFields: [String]
    public var categoricalBuckets: Int
    public var tokenHashPerson: String
    public var hashPerson: String
    public var hidden: Int
    public var layer1Weights: [[Float]]
    public var layer1Bias: [Float]
    public var layer2Weights: [Float]
    public var layer2Bias: Float
    public var identityKeySHA256: String
    public var trainedThroughEventID: String?
    public var learningRate: Float? = nil
    public var negativeLimit: Int? = nil
    public var seed: Int? = nil
    public var beta1: Float? = nil
    public var beta2: Float? = nil
    public var epsilon: Float? = nil
    public var optimizerState: OnlinePathMLPOptimizerState? = nil
    public var scoreMarginCalibration: ScoreMarginConformalState? = nil
}

public struct OnlinePathMLPMatrixMoments: Codable, Sendable {
    public var expAvg: [[Float]]
    public var expAvgSq: [[Float]]
    public var step: Int
}

public struct OnlinePathMLPVectorMoments: Codable, Sendable {
    public var expAvg: [Float]
    public var expAvgSq: [Float]
    public var step: Int
}

public struct OnlinePathMLPScalarMoments: Codable, Sendable {
    public var expAvg: Float
    public var expAvgSq: Float
    public var step: Int
}

public struct OnlinePathMLPOptimizerState: Codable, Sendable {
    public var layer1Weights: OnlinePathMLPMatrixMoments
    public var layer1Bias: OnlinePathMLPVectorMoments
    public var layer2Weights: OnlinePathMLPVectorMoments
    public var layer2Bias: OnlinePathMLPScalarMoments
}

public struct OnlinePathMLPUpdateResult: Sendable {
    public var updated: Bool
    public var loss: Double?
    public var sampledNegativeIndices: [Int]
}

public enum OnlinePathMLPError: Error, Equatable {
    case incompatibleWeights
    case invalidFeatureRow
    case nonfiniteScore
    case unavailableOptimizer
    case invalidTrainingExample
}

/// The exact two-linear-layer, ReLU inference path used by the research MLP.
/// Scores are ranking values, not probabilities.
public struct OnlinePathMLPScorer: Sendable {
    public static let modelVersion = "shared-online-path-mlp-v1"
    public static let featureSchema = "shared-model-input-v3"
    public static let numericFields = [
        "context:time_hour_sin", "context:time_hour_cos",
        "context:time_weekday_sin", "context:time_weekday_cos",
        "context:last_confirmed_dwell_log_s", "context:has_confirmed_dwell",
        "context:quality_ready_fraction", "context:quality_syncing_fraction",
        "context:quality_unavailable_fraction", "context:quality_failed_fraction",
        "context:quality_stale_fraction", "context:quality_unknown_fraction",
        "context:quality_confirmed", "context:has_same_target_inferred_active",
        "context:same_target_inferred_active_log_s", "context:has_legacy_short_gap",
        "context:has_excluded_scope_gap", "context:known_gap_log_s",
        "context:known_gap_duration", "context:rich_input_missing",
        "candidate:hierarchy_layer_count_log", "candidate:is_derived_parent",
        "candidate:same_owner_app", "candidate:path_count_5m_log",
        "candidate:path_count_1h_log", "candidate:path_count_24h_log",
        "candidate:path_count_all_log", "candidate:direct_count_5m_log",
        "candidate:direct_count_1h_log", "candidate:direct_count_24h_log",
        "candidate:direct_count_all_log"
    ]
    public static let categoricalFields = [
        "origin_token", "candidate_token", "origin_candidate_token",
        "origin_kind", "candidate_kind", "origin_bundle_token",
        "candidate_bundle_token", "candidate_source",
        "candidate_parent_token_1", "candidate_parent_token_2",
        "candidate_parent_token_3", "candidate_parent_token_4",
        "current_parent_token_1", "current_parent_token_2"
    ]

    private var weights: OnlinePathMLPWeights
    private var seenArrivalIDs = Set<String>()
    private var seenArrivalOrder: [String] = []

    public init(weights: OnlinePathMLPWeights, identityKeySHA256: String) throws {
        let width = Self.numericFields.count + 512
        guard weights.format == "online-path-mlp-f32-v1",
              weights.modelVersion == Self.modelVersion,
              weights.featureSchema == Self.featureSchema,
              weights.numericFields == Self.numericFields,
              weights.categoricalFields == Self.categoricalFields,
              weights.categoricalBuckets == 512,
              weights.tokenHashPerson == "OffRankV1",
              weights.hashPerson == "SharedMLPPath1",
              weights.hidden == 16,
              weights.identityKeySHA256 == identityKeySHA256,
              weights.layer1Weights.count == weights.hidden,
              weights.layer1Weights.allSatisfy({ $0.count == width && $0.allSatisfy(\.isFinite) }),
              weights.layer1Bias.count == weights.hidden,
              weights.layer1Bias.allSatisfy(\.isFinite),
              weights.layer2Weights.count == weights.hidden,
              weights.layer2Weights.allSatisfy(\.isFinite),
              weights.layer2Bias.isFinite else { throw OnlinePathMLPError.incompatibleWeights }
        self.weights = weights
    }

    public init(contentsOf url: URL, identityKeySHA256: String) throws {
        try self.init(weights: JSONDecoder().decode(OnlinePathMLPWeights.self, from: Data(contentsOf: url)),
                      identityKeySHA256: identityKeySHA256)
    }

    public var checkpoint: OnlinePathMLPWeights { weights }

    public var canContinueLearning: Bool { Self.validOptimizer(weights) }

    private static func validOptimizer(_ weights: OnlinePathMLPWeights) -> Bool {
        guard let state = weights.optimizerState, let rate = weights.learningRate,
              let negativeLimit = weights.negativeLimit, weights.seed != nil,
              let beta1 = weights.beta1, let beta2 = weights.beta2,
              let epsilon = weights.epsilon,
              rate.isFinite, rate > 0, negativeLimit == 8,
              beta1 == 0.9, beta2 == 0.999, epsilon == 1e-8 else { return false }
        let width = Self.numericFields.count + 512
        let step = state.layer1Weights.step
        return step >= 0 && state.layer1Bias.step == step &&
            state.layer2Weights.step == step && state.layer2Bias.step == step &&
            state.layer1Weights.expAvg.count == 16 &&
            state.layer1Weights.expAvgSq.count == 16 &&
            state.layer1Weights.expAvg.allSatisfy({ $0.count == width && $0.allSatisfy(\.isFinite) }) &&
            state.layer1Weights.expAvgSq.allSatisfy({ $0.count == width && $0.allSatisfy({ $0.isFinite && $0 >= 0 }) }) &&
            state.layer1Bias.expAvg.count == 16 && state.layer1Bias.expAvgSq.count == 16 &&
            state.layer1Bias.expAvg.allSatisfy(\.isFinite) &&
            state.layer1Bias.expAvgSq.allSatisfy({ $0.isFinite && $0 >= 0 }) &&
            state.layer2Weights.expAvg.count == 16 && state.layer2Weights.expAvgSq.count == 16 &&
            state.layer2Weights.expAvg.allSatisfy(\.isFinite) &&
            state.layer2Weights.expAvgSq.allSatisfy({ $0.isFinite && $0 >= 0 }) &&
            state.layer2Bias.expAvg.isFinite && state.layer2Bias.expAvgSq.isFinite &&
            state.layer2Bias.expAvgSq >= 0
    }

    /// Python offline.token(value): BLAKE2b-12 with person=OffRankV1.
    public static func stableToken(_ value: String?) -> String {
        guard let value, !value.isEmpty else { return "none" }
        return lowercaseHex(Blake2b.digest(Data(value.utf8), outputLength: 12, person: "OffRankV1"))
    }

    /// Hex formatting without one autoreleased NSString per digest byte.
    static func lowercaseHex<S: Sequence>(_ bytes: S) -> String where S.Element == UInt8 {
        let alphabet = Array("0123456789abcdef".utf8)
        var result: [UInt8] = []
        result.reserveCapacity(bytes.underestimatedCount * 2)
        for byte in bytes { result.append(alphabet[Int(byte >> 4)]); result.append(alphabet[Int(byte & 15)]) }
        return String(decoding: result, as: UTF8.self)
    }

    /// Public for parity tests and the common runtime feature builder.
    public func featureVector(for row: OnlinePathMLPRow) throws -> [Float] {
        let ordered = try row.orderedValues()
        var values = [Float](repeating: 0, count: Self.numericFields.count + weights.categoricalBuckets)
        for (column, number) in ordered.numeric.enumerated() {
            let value = Float(number)
            guard value.isFinite else { throw OnlinePathMLPError.invalidFeatureRow }
            values[column] = value
        }
        for (name, value) in zip(Self.categoricalFields, ordered.categorical) {
            addHashed("candidate:\(name)=\(value)", into: &values)
        }
        for layer in row.layerKeys {
            addHashed("candidate_layer:\(Self.stableToken(layer))", into: &values)
        }
        guard values.allSatisfy(\.isFinite) else { throw OnlinePathMLPError.invalidFeatureRow }
        return values
    }

    public func score(_ row: OnlinePathMLPRow) throws -> Double {
        let input = try featureVector(for: row)
        return try scoreFeatureVector(input)
    }

    /// Also permits private cross-language parity checks on an already built
    /// input vector. App callers should use `score(_:)` with a checked row.
    public func scoreFeatureVector(_ input: [Float]) throws -> Double {
        guard input.count == Self.numericFields.count + weights.categoricalBuckets,
              input.allSatisfy(\.isFinite) else { throw OnlinePathMLPError.invalidFeatureRow }
        let result = forward(input)
        guard result.score.isFinite else { throw OnlinePathMLPError.nonfiniteScore }
        return Double(result.score)
    }

    private func forward(_ input: [Float]) -> (preReLU: [Float], postReLU: [Float], score: Float) {
        var preReLU = [Float](repeating: 0, count: weights.hidden)
        var postReLU = [Float](repeating: 0, count: weights.hidden)
        var output = weights.layer2Bias
        for hiddenIndex in 0..<weights.hidden {
            var activation = weights.layer1Bias[hiddenIndex]
            for column in input.indices {
                activation += weights.layer1Weights[hiddenIndex][column] * input[column]
            }
            preReLU[hiddenIndex] = activation
            postReLU[hiddenIndex] = max(0, activation)
            output += weights.layer2Weights[hiddenIndex] * postReLU[hiddenIndex]
        }
        return (preReLU, postReLU, output)
    }

    /// One confirmed arrival is one update, even when several parent actions
    /// are useful. An after-feedback exact leaf may be appended at the end of
    /// rows, but it is never eligible as a prediction-time negative.
    @discardableResult
    public mutating func learn(rows: [OnlinePathMLPRow], positiveIndices: [Int],
                               originalCandidateCount: Int, arrivalID: String,
                               weight: Float = 1) throws -> OnlinePathMLPUpdateResult {
        guard Self.validOptimizer(weights), var state = weights.optimizerState,
              let learningRate = weights.learningRate, let seed = weights.seed,
              let beta1 = weights.beta1, let beta2 = weights.beta2,
              let epsilon = weights.epsilon else { throw OnlinePathMLPError.unavailableOptimizer }
        let positive = Set(positiveIndices)
        guard !arrivalID.isEmpty, !seenArrivalIDs.contains(arrivalID),
              !positive.isEmpty, positive.count == positiveIndices.count,
              positive.allSatisfy({ $0 >= 0 && $0 < rows.count }),
              originalCandidateCount >= 0 && originalCandidateCount <= rows.count,
              rows.count - originalCandidateCount <= 1,
              rows.count == originalCandidateCount || positive.contains(rows.count - 1),
              weight.isFinite && weight >= 0 && weight <= 1 else {
            throw OnlinePathMLPError.invalidTrainingExample
        }
        if weight == 0 {
            return OnlinePathMLPUpdateResult(updated: false, loss: nil, sampledNegativeIndices: [])
        }
        var negativePool = (0..<originalCandidateCount).filter { !positive.contains($0) }
        guard !negativePool.isEmpty else {
            return OnlinePathMLPUpdateResult(updated: false, loss: nil, sampledNegativeIndices: [])
        }
        var generator = SplitMix64(state: UInt64(bitPattern: Int64(seed)) &+ UInt64(state.layer1Weights.step))
        let limit = min(weights.negativeLimit ?? 8, negativePool.count)
        for i in 0..<limit {
            let chosen = i + Int(generator.next() % UInt64(negativePool.count - i))
            negativePool.swapAt(i, chosen)
        }
        let negatives = Array(negativePool.prefix(limit))
        let selected = (positiveIndices + negatives).sorted()
        let vectors = try selected.map { try featureVector(for: rows[$0]) }
        let outputs = vectors.map(forward)
        guard outputs.allSatisfy({ $0.score.isFinite && $0.preReLU.allSatisfy(\.isFinite) }) else {
            throw OnlinePathMLPError.nonfiniteScore
        }
        let maxAll = outputs.map(\.score).max() ?? 0
        let maxPositive = zip(selected, outputs).filter { positive.contains($0.0) }
            .map { $0.1.score }.max() ?? 0
        let exponentials = outputs.map { Float(Foundation.exp(Double($0.score - maxAll))) }
        let positiveExponentials = zip(selected, outputs).map { index, output in
            positive.contains(index) ? Float(Foundation.exp(Double(output.score - maxPositive))) : 0
        }
        let allSum = exponentials.reduce(0, +)
        let positiveSum = positiveExponentials.reduce(0, +)
        guard allSum.isFinite, positiveSum.isFinite, allSum > 0, positiveSum > 0 else {
            throw OnlinePathMLPError.nonfiniteScore
        }
        let loss = (Double(maxAll) + log(Double(allSum)) -
                    Double(maxPositive) - log(Double(positiveSum))) * Double(weight)
        let width = Self.numericFields.count + weights.categoricalBuckets
        var gradW1 = Array(repeating: [Float](repeating: 0, count: width), count: weights.hidden)
        var gradB1 = [Float](repeating: 0, count: weights.hidden)
        var gradW2 = [Float](repeating: 0, count: weights.hidden)
        var gradB2: Float = 0
        for rowIndex in selected.indices {
            let dScore = weight * (exponentials[rowIndex] / allSum -
                positiveExponentials[rowIndex] / positiveSum)
            gradB2 += dScore
            for hiddenIndex in 0..<weights.hidden {
                gradW2[hiddenIndex] += dScore * outputs[rowIndex].postReLU[hiddenIndex]
                if outputs[rowIndex].preReLU[hiddenIndex] > 0 {
                    let dActivation = dScore * weights.layer2Weights[hiddenIndex]
                    gradB1[hiddenIndex] += dActivation
                    for column in 0..<width {
                        gradW1[hiddenIndex][column] += dActivation * vectors[rowIndex][column]
                    }
                }
            }
        }
        var squaredNorm = Double(gradB2) * Double(gradB2)
        for index in 0..<weights.hidden {
            squaredNorm += Double(gradB1[index]) * Double(gradB1[index])
            squaredNorm += Double(gradW2[index]) * Double(gradW2[index])
            for column in 0..<width {
                squaredNorm += Double(gradW1[index][column]) * Double(gradW1[index][column])
            }
        }
        guard squaredNorm.isFinite else { throw OnlinePathMLPError.nonfiniteScore }
        let scale = Float(min(1, 5 / (sqrt(squaredNorm) + 1e-6)))
        let step = state.layer1Weights.step + 1
        let correction1 = 1 - pow(beta1, Float(step))
        let correction2 = 1 - pow(beta2, Float(step))
        guard correction1 > 0, correction2 > 0 else { throw OnlinePathMLPError.unavailableOptimizer }
        var updatedWeights = weights
        for index in 0..<weights.hidden {
            for column in 0..<width {
                Self.adam(&updatedWeights.layer1Weights[index][column],
                          &state.layer1Weights.expAvg[index][column],
                          &state.layer1Weights.expAvgSq[index][column],
                          gradient: gradW1[index][column] * scale, rate: learningRate,
                          beta1: beta1, beta2: beta2, epsilon: epsilon,
                          correction1: correction1, correction2: correction2)
            }
            Self.adam(&updatedWeights.layer1Bias[index], &state.layer1Bias.expAvg[index],
                      &state.layer1Bias.expAvgSq[index], gradient: gradB1[index] * scale,
                      rate: learningRate, beta1: beta1, beta2: beta2, epsilon: epsilon,
                      correction1: correction1, correction2: correction2)
            Self.adam(&updatedWeights.layer2Weights[index], &state.layer2Weights.expAvg[index],
                      &state.layer2Weights.expAvgSq[index], gradient: gradW2[index] * scale,
                      rate: learningRate, beta1: beta1, beta2: beta2, epsilon: epsilon,
                      correction1: correction1, correction2: correction2)
        }
        Self.adam(&updatedWeights.layer2Bias, &state.layer2Bias.expAvg, &state.layer2Bias.expAvgSq,
                  gradient: gradB2 * scale, rate: learningRate, beta1: beta1,
                  beta2: beta2, epsilon: epsilon,
                  correction1: correction1, correction2: correction2)
        state.layer1Weights.step = step
        state.layer1Bias.step = step
        state.layer2Weights.step = step
        state.layer2Bias.step = step
        updatedWeights.optimizerState = state
        guard updatedWeights.layer1Weights.allSatisfy({ $0.allSatisfy(\.isFinite) }),
              updatedWeights.layer1Bias.allSatisfy(\.isFinite),
              updatedWeights.layer2Weights.allSatisfy(\.isFinite), updatedWeights.layer2Bias.isFinite else {
            throw OnlinePathMLPError.nonfiniteScore
        }
        weights = updatedWeights
        seenArrivalIDs.insert(arrivalID)
        seenArrivalOrder.append(arrivalID)
        if seenArrivalOrder.count > 8_192 {
            let expired = Array(seenArrivalOrder.prefix(1_024))
            seenArrivalOrder.removeFirst(expired.count)
            for id in expired { seenArrivalIDs.remove(id) }
        }
        return OnlinePathMLPUpdateResult(updated: true, loss: loss,
                                         sampledNegativeIndices: negatives)
    }

    private static func adam(_ parameter: inout Float, _ average: inout Float,
                             _ squaredAverage: inout Float, gradient: Float,
                             rate: Float, beta1: Float, beta2: Float, epsilon: Float,
                             correction1: Float, correction2: Float) {
        average = beta1 * average + (1 - beta1) * gradient
        squaredAverage = beta2 * squaredAverage + (1 - beta2) * gradient * gradient
        parameter -= (rate / correction1) * average /
            (sqrt(squaredAverage) / sqrt(correction2) + epsilon)
    }

    private func addHashed(_ name: String, into vector: inout [Float]) {
        let bytes = Blake2b.digest(Data(name.utf8), outputLength: 8, person: "SharedMLPPath1")
        var number: UInt64 = 0
        for (offset, byte) in bytes.enumerated() {
            number |= UInt64(byte) << UInt64(8 * offset)
        }
        let bucket = Int(number % UInt64(weights.categoricalBuckets))
        vector[Self.numericFields.count + bucket] += bytes[0] & 1 == 1 ? 1 : -1
    }
}

private struct SplitMix64 {
    var state: UInt64
    mutating func next() -> UInt64 {
        state &+= 0x9e3779b97f4a7c15
        var value = state
        value = (value ^ (value >> 30)) &* 0xbf58476d1ce4e5b9
        value = (value ^ (value >> 27)) &* 0x94d049bb133111eb
        return value ^ (value >> 31)
    }
}

/// Minimal BLAKE2b implementation for exact parity with Python's hashlib.
/// Only unkeyed hashing and a personalization string are needed here.
private enum Blake2b {
    private static let iv: [UInt64] = [
        0x6a09e667f3bcc908, 0xbb67ae8584caa73b, 0x3c6ef372fe94f82b,
        0xa54ff53a5f1d36f1, 0x510e527fade682d1, 0x9b05688c2b3e6c1f,
        0x1f83d9abfb41bd6b, 0x5be0cd19137e2179
    ]
    private static let sigma: [[Int]] = [
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
        [14, 10, 4, 8, 9, 15, 13, 6, 1, 12, 0, 2, 11, 7, 5, 3],
        [11, 8, 12, 0, 5, 2, 15, 13, 10, 14, 3, 6, 7, 1, 9, 4],
        [7, 9, 3, 1, 13, 12, 11, 14, 2, 6, 5, 10, 4, 0, 15, 8],
        [9, 0, 5, 7, 2, 4, 10, 15, 14, 1, 11, 12, 6, 8, 3, 13],
        [2, 12, 6, 10, 0, 11, 8, 3, 4, 13, 7, 5, 15, 14, 1, 9],
        [12, 5, 1, 15, 14, 13, 4, 10, 0, 7, 6, 3, 9, 2, 8, 11],
        [13, 11, 7, 14, 12, 1, 3, 9, 5, 0, 15, 4, 8, 6, 2, 10],
        [6, 15, 14, 9, 11, 3, 0, 8, 12, 2, 13, 7, 1, 4, 10, 5],
        [10, 2, 8, 4, 7, 6, 1, 5, 15, 11, 9, 14, 3, 12, 13, 0],
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
        [14, 10, 4, 8, 9, 15, 13, 6, 1, 12, 0, 2, 11, 7, 5, 3]
    ]

    static func digest(_ input: Data, outputLength: Int, person: String) -> [UInt8] {
        precondition((1...64).contains(outputLength) && person.utf8.count <= 16)
        var h = iv
        h[0] ^= UInt64(0x01010000 | outputLength)
        let personBytes = Array(person.utf8)
        for offset in personBytes.indices {
            h[6 + offset / 8] ^= UInt64(personBytes[offset]) << UInt64((offset % 8) * 8)
        }
        let bytes = Array(input)
        let blockCount = max(1, (bytes.count + 127) / 128)
        for blockIndex in 0..<blockCount {
            let start = blockIndex * 128
            let count = min(128, max(0, bytes.count - start))
            var block = [UInt8](repeating: 0, count: 128)
            if count > 0 { block.replaceSubrange(0..<count, with: bytes[start..<(start + count)]) }
            let total = UInt64(start + count)
            compress(&h, block: block, total: total, final: blockIndex == blockCount - 1)
        }
        var result: [UInt8] = []
        for word in h {
            for offset in 0..<8 { result.append(UInt8(truncatingIfNeeded: word >> UInt64(offset * 8))) }
        }
        return Array(result.prefix(outputLength))
    }

    private static func compress(_ h: inout [UInt64], block: [UInt8], total: UInt64, final: Bool) {
        var m = [UInt64](repeating: 0, count: 16)
        for index in 0..<16 {
            for offset in 0..<8 {
                m[index] |= UInt64(block[index * 8 + offset]) << UInt64(offset * 8)
            }
        }
        var v = h + iv
        v[12] ^= total
        if final { v[14] = ~v[14] }
        for round in 0..<12 {
            let s = sigma[round]
            mix(&v, 0, 4, 8, 12, m[s[0]], m[s[1]])
            mix(&v, 1, 5, 9, 13, m[s[2]], m[s[3]])
            mix(&v, 2, 6, 10, 14, m[s[4]], m[s[5]])
            mix(&v, 3, 7, 11, 15, m[s[6]], m[s[7]])
            mix(&v, 0, 5, 10, 15, m[s[8]], m[s[9]])
            mix(&v, 1, 6, 11, 12, m[s[10]], m[s[11]])
            mix(&v, 2, 7, 8, 13, m[s[12]], m[s[13]])
            mix(&v, 3, 4, 9, 14, m[s[14]], m[s[15]])
        }
        for index in 0..<8 { h[index] ^= v[index] ^ v[index + 8] }
    }

    private static func mix(_ v: inout [UInt64], _ a: Int, _ b: Int, _ c: Int, _ d: Int,
                            _ x: UInt64, _ y: UInt64) {
        v[a] = v[a] &+ v[b] &+ x
        v[d] = (v[d] ^ v[a]).rotatedRight(32)
        v[c] = v[c] &+ v[d]
        v[b] = (v[b] ^ v[c]).rotatedRight(24)
        v[a] = v[a] &+ v[b] &+ y
        v[d] = (v[d] ^ v[a]).rotatedRight(16)
        v[c] = v[c] &+ v[d]
        v[b] = (v[b] ^ v[c]).rotatedRight(63)
    }
}

private extension UInt64 {
    func rotatedRight(_ bits: UInt64) -> UInt64 { (self >> bits) | (self << (64 - bits)) }
}
