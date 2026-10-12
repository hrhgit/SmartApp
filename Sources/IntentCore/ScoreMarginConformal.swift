import Foundation

public enum ScoreMarginConformalError: Error, Equatable {
    case invalidIdentifier
    case invalidMissRate
    case invalidLearningRate
    case invalidPredictionSequence
    case invalidOutcomeSequence
    case duplicateCandidate
    case nonfiniteScore
    case marginOverflow
    case emptyAcceptableActions
    case emptyCalibration
    case duplicatePrediction
    case duplicateOutcome
    case incompatibleContext
    case incompatibleModelRevision
    case predictionBeforeCalibration
    case feedbackOutOfOrder
    case foreignDecision
    case nonfiniteUpdate
    case invalidMaximumCount
    case incompatibleState
}

/// The matching rule is supplied by the caller's authoritative scorer. Exact
/// target coverage and coverage via a useful ancestor require different contexts.
public struct ScoreMarginContext: Sendable, Equatable {
    public let modelVersion: String
    public let featureSchema: String
    public let matchingRule: String

    public init(modelVersion: String, featureSchema: String, matchingRule: String) throws {
        guard [modelVersion, featureSchema, matchingRule].allSatisfy(Self.validID) else {
            throw ScoreMarginConformalError.invalidIdentifier
        }
        self.modelVersion = modelVersion
        self.featureSchema = featureSchema
        self.matchingRule = matchingRule
    }

    fileprivate static func validID(_ value: String) -> Bool {
        !value.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
    }
}

/// Transient scores only. These types deliberately have no persistence API.
public struct ScoreMarginCandidate: Sendable, Equatable {
    public let actionID: String
    public let score: Double

    public init(actionID: String, score: Double) {
        self.actionID = actionID
        self.score = score
    }
}

/// An immutable prediction made before its outcome. Sequences refer to a single
/// caller-owned chronological timeline, not timestamps reconstructed afterward.
public struct ScoreMarginPrediction: Sendable, Equatable {
    public let predictionID: String
    public let context: ScoreMarginContext
    public let modelRevision: String
    public let trainedThroughSequence: Int64
    public let cutoffSequence: Int64
    public let candidates: [ScoreMarginCandidate]
    fileprivate let token = UUID()

    public init(predictionID: String, context: ScoreMarginContext, modelRevision: String,
                trainedThroughSequence: Int64, cutoffSequence: Int64,
                candidates: [ScoreMarginCandidate]) throws {
        guard ScoreMarginContext.validID(predictionID), ScoreMarginContext.validID(modelRevision) else {
            throw ScoreMarginConformalError.invalidIdentifier
        }
        guard trainedThroughSequence >= 0, cutoffSequence >= trainedThroughSequence else {
            throw ScoreMarginConformalError.invalidPredictionSequence
        }
        var ids = Set<String>()
        for candidate in candidates {
            guard ScoreMarginContext.validID(candidate.actionID) else {
                throw ScoreMarginConformalError.invalidIdentifier
            }
            guard candidate.score.isFinite else { throw ScoreMarginConformalError.nonfiniteScore }
            guard ids.insert(candidate.actionID).inserted else {
                throw ScoreMarginConformalError.duplicateCandidate
            }
        }
        let ranked = candidates.sorted {
            $0.score == $1.score ? $0.actionID < $1.actionID : $0.score > $1.score
        }
        if let top = ranked.first?.score {
            guard ranked.allSatisfy({ (top - $0.score).isFinite }) else {
                throw ScoreMarginConformalError.marginOverflow
            }
        }
        self.predictionID = predictionID
        self.context = context
        self.modelRevision = modelRevision
        self.trainedThroughSequence = trainedThroughSequence
        self.cutoffSequence = cutoffSequence
        self.candidates = ranked
    }

    /// Call only for an independently confirmed, legal outcome. Include its
    /// exact action ID even if absent; approved ancestors may be added by the
    /// existing matcher. Several acceptable actions still produce one sample.
    public func observe(outcomeID: String, outcomeSequence: Int64,
                        acceptableActionIDs: Set<String>) throws -> ScoreMarginObservation {
        guard ScoreMarginContext.validID(outcomeID),
              acceptableActionIDs.allSatisfy(ScoreMarginContext.validID) else {
            throw ScoreMarginConformalError.invalidIdentifier
        }
        guard outcomeSequence > cutoffSequence else {
            throw ScoreMarginConformalError.invalidOutcomeSequence
        }
        guard !acceptableActionIDs.isEmpty else {
            throw ScoreMarginConformalError.emptyAcceptableActions
        }
        let margin: Double
        let acceptedRank: Int?
        if let top = candidates.first?.score,
           let index = candidates.firstIndex(where: { acceptableActionIDs.contains($0.actionID) }) {
            margin = top - candidates[index].score
            acceptedRank = index + 1
        } else {
            // A missing correct action is an unavoidable miss, even if every
            // available action is shown. Keep it in the calibration denominator.
            margin = .infinity
            acceptedRank = nil
        }
        return ScoreMarginObservation(predictionID: predictionID, context: context,
            modelRevision: modelRevision, trainedThroughSequence: trainedThroughSequence,
            cutoffSequence: cutoffSequence, outcomeID: outcomeID,
            outcomeSequence: outcomeSequence, margin: margin, acceptedRank: acceptedRank, predictionToken: token)
    }

    /// Explicitly confirmed legal outcome with no matching prediction-time
    /// action. This is distinct from an unknown outcome or an empty label.
    public func observeUnavailableOutcome(outcomeID: String, outcomeSequence: Int64) throws -> ScoreMarginObservation {
        guard ScoreMarginContext.validID(outcomeID) else { throw ScoreMarginConformalError.invalidIdentifier }
        guard outcomeSequence > cutoffSequence else { throw ScoreMarginConformalError.invalidOutcomeSequence }
        return ScoreMarginObservation(predictionID: predictionID, context: context,
            modelRevision: modelRevision, trainedThroughSequence: trainedThroughSequence,
            cutoffSequence: cutoffSequence, outcomeID: outcomeID, outcomeSequence: outcomeSequence,
            margin: .infinity, acceptedRank: nil, predictionToken: token)
    }
}

/// No candidate identities or scores are retained in calibration observations.
public struct ScoreMarginObservation: Sendable, Equatable {
    public let predictionID: String
    public let context: ScoreMarginContext
    public let modelRevision: String
    public let trainedThroughSequence: Int64
    public let cutoffSequence: Int64
    public let outcomeID: String
    public let outcomeSequence: Int64
    public let margin: Double
    public let acceptedRank: Int?
    fileprivate let predictionToken: UUID
    public var hasAcceptableCandidate: Bool { margin.isFinite }
    fileprivate init(predictionID: String, context: ScoreMarginContext, modelRevision: String,
                     trainedThroughSequence: Int64, cutoffSequence: Int64, outcomeID: String,
                     outcomeSequence: Int64, margin: Double, acceptedRank: Int?, predictionToken: UUID) {
        self.predictionID = predictionID
        self.context = context
        self.modelRevision = modelRevision
        self.trainedThroughSequence = trainedThroughSequence
        self.cutoffSequence = cutoffSequence
        self.outcomeID = outcomeID
        self.outcomeSequence = outcomeSequence
        self.margin = margin
        self.acceptedRank = acceptedRank
        self.predictionToken = predictionToken
    }
}

public enum ScoreMarginThreshold: Sendable, Equatable {
    case finite(Double)
    case allCandidates
    case noCandidates

    fileprivate func includes(_ margin: Double) -> Bool {
        // Infinity denotes an absent acceptable action, not a very low score.
        guard margin.isFinite else { return false }
        switch self {
        case .finite(let value): return margin <= value
        case .allCandidates: return true
        case .noCandidates: return false
        }
    }
}

/// A frozen threshold and prediction; later feedback cannot alter this choice.
/// Keeping the decision in memory permits delayed outcome matching without
/// persisting a candidate list or recomputing with future model weights.
public struct ScoreMarginDecision: Sendable, Equatable {
    public let prediction: ScoreMarginPrediction
    public let threshold: ScoreMarginThreshold
    /// Quantile argument used for this choice, not a measured error rate,
    /// calibrated probability or per-prediction confidence.
    public let effectiveMissRate: Double
    public let maximumCount: Int?
    fileprivate let owner: UUID?
    fileprivate let ordinal: Int?

    public var selected: [ScoreMarginCandidate] {
        guard let top = prediction.candidates.first?.score else { return [] }
        let eligible = prediction.candidates.filter { threshold.includes(top - $0.score) }
        return maximumCount.map { Array(eligible.prefix($0)) } ?? eligible
    }

    /// Execution-time qualification for a read-only menu preview. Learning
    /// decisions must keep their original candidate universe unchanged.
    public func restrictingPreview(toActionIDs actionIDs: Set<String>) throws -> ScoreMarginDecision {
        guard owner == nil else { throw ScoreMarginConformalError.foreignDecision }
        let filtered = try ScoreMarginPrediction(predictionID: prediction.predictionID,
            context: prediction.context, modelRevision: prediction.modelRevision,
            trainedThroughSequence: prediction.trainedThroughSequence,
            cutoffSequence: prediction.cutoffSequence,
            candidates: prediction.candidates.filter { actionIDs.contains($0.actionID) })
        return ScoreMarginDecision(prediction: filtered, threshold: threshold,
            effectiveMissRate: effectiveMissRate, maximumCount: maximumCount, owner: nil, ordinal: nil)
    }

    public func covers(_ observation: ScoreMarginObservation) throws -> Bool {
        guard observation.context == prediction.context else {
            throw ScoreMarginConformalError.incompatibleContext
        }
        guard observation.predictionID == prediction.predictionID,
              observation.modelRevision == prediction.modelRevision,
              observation.cutoffSequence == prediction.cutoffSequence,
              observation.trainedThroughSequence == prediction.trainedThroughSequence,
              observation.predictionToken == prediction.token else {
            throw ScoreMarginConformalError.foreignDecision
        }
        return threshold.includes(observation.margin) &&
            (maximumCount.map { (observation.acceptedRank ?? Int.max) <= $0 } ?? true)
    }
}

/// Split conformal calibration with rank ceil((n + 1) * (1 - alpha)).
/// Actual coverage still depends on valid held-out observations, matching rules,
/// the candidate universe and the relevant exchangeability assumptions.
public struct ScoreMarginCalibration: Sendable {
    public let context: ScoreMarginContext
    public let modelRevision: String
    public let trainedThroughSequence: Int64
    public let calibratedThroughSequence: Int64
    public let targetMissRate: Double
    fileprivate let sortedMargins: [Double]
    fileprivate let predictionIDs: Set<String>
    fileprivate let outcomeIDs: Set<String>

    public var sampleCount: Int { sortedMargins.count }
    public var unavailableOutcomeCount: Int { sortedMargins.filter { !$0.isFinite }.count }
    public var quantileRank: Int { Int(ceil(Double(sampleCount + 1) * (1 - targetMissRate))) }
    public var threshold: ScoreMarginThreshold {
        Self.threshold(sortedMargins: sortedMargins, missRate: targetMissRate)
    }

    public init(observations: [ScoreMarginObservation], targetMissRate: Double) throws {
        guard targetMissRate.isFinite, targetMissRate > 0, targetMissRate < 1 else {
            throw ScoreMarginConformalError.invalidMissRate
        }
        guard let first = observations.first else { throw ScoreMarginConformalError.emptyCalibration }
        var predictionIDs = Set<String>(), outcomeIDs = Set<String>()
        for item in observations {
            guard item.context == first.context else { throw ScoreMarginConformalError.incompatibleContext }
            guard item.modelRevision == first.modelRevision,
                  item.trainedThroughSequence == first.trainedThroughSequence else {
                throw ScoreMarginConformalError.incompatibleModelRevision
            }
            guard predictionIDs.insert(item.predictionID).inserted else {
                throw ScoreMarginConformalError.duplicatePrediction
            }
            guard outcomeIDs.insert(item.outcomeID).inserted else {
                throw ScoreMarginConformalError.duplicateOutcome
            }
        }
        context = first.context
        modelRevision = first.modelRevision
        trainedThroughSequence = first.trainedThroughSequence
        calibratedThroughSequence = observations.map(\.outcomeSequence).max()!
        self.targetMissRate = targetMissRate
        sortedMargins = observations.map(\.margin).sorted()
        self.predictionIDs = predictionIDs
        self.outcomeIDs = outcomeIDs
    }

    public func select(_ prediction: ScoreMarginPrediction, maximumCount: Int? = nil) throws -> ScoreMarginDecision {
        try Self.validateMaximumCount(maximumCount)
        guard prediction.context == context else { throw ScoreMarginConformalError.incompatibleContext }
        guard prediction.modelRevision == modelRevision,
              prediction.trainedThroughSequence == trainedThroughSequence else {
            throw ScoreMarginConformalError.incompatibleModelRevision
        }
        guard prediction.cutoffSequence >= calibratedThroughSequence else {
            throw ScoreMarginConformalError.predictionBeforeCalibration
        }
        guard !predictionIDs.contains(prediction.predictionID) else {
            throw ScoreMarginConformalError.duplicatePrediction
        }
        return ScoreMarginDecision(prediction: prediction, threshold: threshold,
            effectiveMissRate: targetMissRate, maximumCount: maximumCount, owner: nil, ordinal: nil)
    }

    fileprivate static func validateMaximumCount(_ count: Int?) throws {
        if let count, count <= 0 { throw ScoreMarginConformalError.invalidMaximumCount }
    }

    fileprivate static func threshold(sortedMargins: [Double], missRate: Double) -> ScoreMarginThreshold {
        if missRate <= 0 { return .allCandidates }
        if missRate >= 1 { return .noCandidates }
        let rank = ceil(Double(sortedMargins.count + 1) * (1 - missRate))
        guard rank <= Double(sortedMargins.count) else { return .allCandidates }
        let value = sortedMargins[Int(rank) - 1]
        return value.isFinite ? .finite(value) : .allCandidates
    }
}

/// Persistable derived state contains scalar margins and aggregate counts only.
/// Prediction/action/outcome IDs, pending decisions and locators are excluded.
public struct ScoreMarginConformalState: Codable, Sendable {
    public var format: String
    public var modelVersion: String
    public var featureSchema: String
    public var matchingRule: String
    public var targetMissRate: Double
    public var learningRate: Double
    public var effectiveMissRate: Double
    public var margins: [Double?]
    public var feedbackCount: Int
    public var coveredFeedbackCount: Int
    public var thresholdCoveredFeedbackCount: Int
    public var policyMaximumCount: Int?
}

/// Independent ACI controller: alpha_next = alpha + gamma * (target - miss).
/// Callers resolve or discard issued decisions in order. Feedback uses the
/// issued threshold; only then are its margin and alpha added to future state.
/// This is a value type: a copied controller is a fork of that series and must
/// not be merged back. No model, menu, collector, clock or file I/O is involved.
public struct AdaptiveScoreMarginConformal: Sendable {
    public let context: ScoreMarginContext
    public let targetMissRate: Double
    public let learningRate: Double
    /// Internal adaptive quantile argument; deliberately may leave [0, 1].
    /// Actual observed misses are feedbackCount - coveredFeedbackCount.
    public private(set) var effectiveMissRate: Double
    public private(set) var feedbackCount = 0
    public private(set) var coveredFeedbackCount = 0
    public private(set) var thresholdCoveredFeedbackCount = 0
    public var capLimitedFeedbackCount: Int { thresholdCoveredFeedbackCount - coveredFeedbackCount }
    private var sortedMargins: [Double]
    private var predictionIDs: Set<String>
    private var outcomeIDs: Set<String>
    private let owner = UUID()
    private var pending: [Int: String] = [:]
    private var issued = 0
    private var nextFeedback = 0
    private var lastPredictionSequence: Int64
    private var lastOutcomeSequence: Int64

    public var sampleCount: Int { sortedMargins.count }
    public var unavailableOutcomeCount: Int { sortedMargins.filter { !$0.isFinite }.count }
    public var pendingCount: Int { pending.count }
    public var threshold: ScoreMarginThreshold {
        ScoreMarginCalibration.threshold(sortedMargins: sortedMargins, missRate: effectiveMissRate)
    }

    public init(calibration: ScoreMarginCalibration, learningRate: Double) throws {
        guard learningRate.isFinite, learningRate > 0 else {
            throw ScoreMarginConformalError.invalidLearningRate
        }
        context = calibration.context
        targetMissRate = calibration.targetMissRate
        effectiveMissRate = targetMissRate
        self.learningRate = learningRate
        sortedMargins = calibration.sortedMargins
        predictionIDs = calibration.predictionIDs
        outcomeIDs = calibration.outcomeIDs
        lastPredictionSequence = calibration.calibratedThroughSequence
        lastOutcomeSequence = calibration.calibratedThroughSequence
    }

    /// Start without reusing behaviors already used to fit model weights. An
    /// empty calibration distribution explicitly selects all scored actions.
    public init(context: ScoreMarginContext, targetMissRate: Double, learningRate: Double,
                startSequence: Int64) throws {
        guard targetMissRate.isFinite, targetMissRate > 0, targetMissRate < 1 else {
            throw ScoreMarginConformalError.invalidMissRate
        }
        guard learningRate.isFinite, learningRate > 0 else { throw ScoreMarginConformalError.invalidLearningRate }
        guard startSequence >= 0 else { throw ScoreMarginConformalError.invalidPredictionSequence }
        self.context = context
        self.targetMissRate = targetMissRate
        self.learningRate = learningRate
        effectiveMissRate = targetMissRate
        sortedMargins = []
        predictionIDs = []
        outcomeIDs = []
        lastPredictionSequence = startSequence
        lastOutcomeSequence = startSequence
    }

    public init(state: ScoreMarginConformalState, startSequence: Int64) throws {
        let context = try ScoreMarginContext(modelVersion: state.modelVersion,
            featureSchema: state.featureSchema, matchingRule: state.matchingRule)
        try self.init(context: context, targetMissRate: state.targetMissRate,
                      learningRate: state.learningRate, startSequence: startSequence)
        try ScoreMarginCalibration.validateMaximumCount(state.policyMaximumCount)
        let margins = state.margins.map { $0 ?? .infinity }
        guard state.format == "score-margin-state-v1", state.effectiveMissRate.isFinite,
              state.feedbackCount >= 0, state.feedbackCount <= margins.count,
              state.coveredFeedbackCount >= 0,
              state.coveredFeedbackCount <= state.thresholdCoveredFeedbackCount,
              state.thresholdCoveredFeedbackCount <= state.feedbackCount,
              margins.allSatisfy({ $0 >= 0 && !$0.isNaN }),
              state.margins.compactMap({ $0 }).allSatisfy(\.isFinite),
              zip(margins, margins.dropFirst()).allSatisfy({ $0.0 <= $0.1 }) else {
            throw ScoreMarginConformalError.incompatibleState
        }
        sortedMargins = margins
        effectiveMissRate = state.effectiveMissRate
        feedbackCount = state.feedbackCount
        coveredFeedbackCount = state.coveredFeedbackCount
        thresholdCoveredFeedbackCount = state.thresholdCoveredFeedbackCount
    }

    public func checkpoint(maximumCount: Int? = nil) throws -> ScoreMarginConformalState {
        try ScoreMarginCalibration.validateMaximumCount(maximumCount)
        return ScoreMarginConformalState(format: "score-margin-state-v1", modelVersion: context.modelVersion,
            featureSchema: context.featureSchema, matchingRule: context.matchingRule,
            targetMissRate: targetMissRate, learningRate: learningRate, effectiveMissRate: effectiveMissRate,
            margins: sortedMargins.map { $0.isFinite ? $0 : nil }, feedbackCount: feedbackCount,
            coveredFeedbackCount: coveredFeedbackCount, thresholdCoveredFeedbackCount: thresholdCoveredFeedbackCount,
            policyMaximumCount: maximumCount)
    }

    /// Menu presentation is a pure read; opening or cancelling it cannot
    /// enqueue a prediction or teach the calibration controller.
    public func preview(_ prediction: ScoreMarginPrediction, maximumCount: Int? = nil) throws -> ScoreMarginDecision {
        try ScoreMarginCalibration.validateMaximumCount(maximumCount)
        guard prediction.context == context else { throw ScoreMarginConformalError.incompatibleContext }
        guard prediction.cutoffSequence >= max(lastPredictionSequence, lastOutcomeSequence) else {
            throw ScoreMarginConformalError.predictionBeforeCalibration
        }
        return ScoreMarginDecision(prediction: prediction, threshold: threshold,
            effectiveMissRate: effectiveMissRate, maximumCount: maximumCount, owner: nil, ordinal: nil)
    }

    /// Weight revisions may evolve within this declared model/input/matching
    /// series. A changed context requires fresh calibration, not silent reuse.
    public mutating func select(_ prediction: ScoreMarginPrediction, maximumCount: Int? = nil) throws -> ScoreMarginDecision {
        try ScoreMarginCalibration.validateMaximumCount(maximumCount)
        guard prediction.context == context else { throw ScoreMarginConformalError.incompatibleContext }
        guard prediction.cutoffSequence >= max(lastPredictionSequence, lastOutcomeSequence) else {
            throw ScoreMarginConformalError.predictionBeforeCalibration
        }
        guard !predictionIDs.contains(prediction.predictionID) else {
            throw ScoreMarginConformalError.duplicatePrediction
        }
        let decision = ScoreMarginDecision(prediction: prediction, threshold: threshold,
            effectiveMissRate: effectiveMissRate, maximumCount: maximumCount, owner: owner, ordinal: issued)
        predictionIDs.insert(prediction.predictionID)
        pending[issued] = prediction.predictionID
        issued += 1
        lastPredictionSequence = prediction.cutoffSequence
        return decision
    }

    @discardableResult
    public mutating func observe(_ decision: ScoreMarginDecision, outcomeID: String,
                                 outcomeSequence: Int64, acceptableActionIDs: Set<String>) throws -> Bool {
        try validatePending(decision)
        let observation = try decision.prediction.observe(outcomeID: outcomeID,
            outcomeSequence: outcomeSequence, acceptableActionIDs: acceptableActionIDs)
        return try observe(decision, observation: observation)
    }

    @discardableResult
    public mutating func observe(_ decision: ScoreMarginDecision, observation: ScoreMarginObservation) throws -> Bool {
        try validatePending(decision)
        guard observation.outcomeSequence >= lastOutcomeSequence else {
            throw ScoreMarginConformalError.invalidOutcomeSequence
        }
        guard !outcomeIDs.contains(observation.outcomeID) else { throw ScoreMarginConformalError.duplicateOutcome }
        let covered = try decision.covers(observation)
        let updated = effectiveMissRate + learningRate * (targetMissRate - (covered ? 0 : 1))
        guard updated.isFinite else { throw ScoreMarginConformalError.nonfiniteUpdate }
        // All validation precedes mutation. Alpha is not clipped; its threshold
        // selects all or no actions before an explicitly requested count limit.
        effectiveMissRate = updated
        var low = 0, high = sortedMargins.count
        while low < high {
            let mid = low + (high - low) / 2
            if sortedMargins[mid] < observation.margin { low = mid + 1 } else { high = mid }
        }
        sortedMargins.insert(observation.margin, at: low)
        outcomeIDs.insert(observation.outcomeID)
        lastOutcomeSequence = observation.outcomeSequence
        feedbackCount += 1
        if decision.threshold.includes(observation.margin) { thresholdCoveredFeedbackCount += 1 }
        if covered { coveredFeedbackCount += 1 }
        pending.removeValue(forKey: nextFeedback)
        nextFeedback += 1
        return covered
    }

    /// Unconfirmed outcomes, interrupted observations and cancelled menu
    /// sessions do not provide a hit or miss. Discarding changes no calibration.
    public mutating func discard(_ decision: ScoreMarginDecision) throws {
        try validatePending(decision)
        pending.removeValue(forKey: nextFeedback)
        nextFeedback += 1
    }

    private func validatePending(_ decision: ScoreMarginDecision) throws {
        guard decision.owner == owner, let ordinal = decision.ordinal,
              pending[ordinal] == decision.prediction.predictionID else {
            throw ScoreMarginConformalError.foreignDecision
        }
        guard ordinal == nextFeedback else { throw ScoreMarginConformalError.feedbackOutOfOrder }
    }
}
