import Foundation
import CryptoKit
import IntentCore

struct DynamicRecommendationConfiguration: Sendable {
    // User-approved coverage goal and temporary presentation limit.
    let targetMissRate: Double = 0.1
    let learningRate: Double = 0.005
    let maximumItems: Int = 6
    var context: ScoreMarginContext {
        get throws {
            try ScoreMarginContext(modelVersion: OnlinePathMLPScorer.modelVersion,
                featureSchema: OnlinePathMLPScorer.featureSchema,
                matchingRule: OnlinePathInputHistory.usefulActionRuleVersion + ":" + MainMenuTargetScope.version)
        }
    }
}

struct ScoredOnlinePathCandidate: Sendable {
    let candidate: OnlinePathCandidate
    let score: Double
}

struct OnlineMenuRanking: Sendable {
    let ranked: [ScoredOnlinePathCandidate]
    let preview: ScoreMarginDecision
}

/// Personal model files stay outside the application bundle and source tree.
final class FastRecommendationService: @unchecked Sendable {
    private let queue = DispatchQueue(label: "local.smartapp.semantic-model", qos: .utility)
    private let readinessLock = NSLock()
    private let commitLock = NSLock()
    private var committedThroughSequence: Int64 = 0
    private var commitDrainScheduled = false
    private var historyReader: EventStore?
    private let startupGroup = DispatchGroup()
    private var readyForRequests = false
    private var startupWaitUsed = false
    private let pathDirectory: URL
    private var pathModel: OnlinePathMLPScorer?
    private var pathHistory = OnlinePathInputHistory()
    private var pathLearner = OnlinePathFeedbackLearner()
    private var pathSeedEventID: String?
    private var pathSeedReached = false
    private var pathLastUpdateEventID: String?
    private var pathDirty = false
    private var pathUpdates = 0
    private let dynamicConfiguration = DynamicRecommendationConfiguration()
    private var marginController: AdaptiveScoreMarginConformal?
    private var marginRestore: ScoreMarginConformalState?
    private var marginFailure: String?
    private var trainedThroughDatabaseSequence: Int64 = 0
    private var confirmedSequences: [String: Int64] = [:]
    private var frozenPathOrder: [String] = []
    private var readyFeedback: [String: OnlinePathFeedback] = [:]
    private struct FrozenPathInput {
        let current: Destination
        let candidates: [OnlinePathCandidate]
        let marginDecision: ScoreMarginDecision?
    }
    private var frozenPathInputs: [String: FrozenPathInput] = [:]
    private var lastProcessedDatabaseSequence: Int64 = 0
    private var ready = false
    private var failure: String?

    init(dataDirectory: URL, store: EventStore) {
        pathDirectory = dataDirectory.appendingPathComponent("models/" + OnlinePathMLPScorer.modelVersion)
        startupGroup.enter()
        queue.async { [self] in
            defer { startupGroup.leave() }
            do {
                let reader = try EventStore(url: store.url, readOnly: true)
                historyReader = reader
                let key = try Data(contentsOf: dataDirectory.appendingPathComponent("identity-v2.key"))
                let fingerprint = SHA256.hash(data: key).map { String(format: "%02x", $0) }.joined()
                let pathState = pathDirectory.appendingPathComponent("state.json")
                let pathWeights = pathDirectory.appendingPathComponent("weights.json")
                guard FileManager.default.fileExists(atPath: pathState.path) ||
                      FileManager.default.fileExists(atPath: pathWeights.path) else {
                    throw StoreError("Online model weights are absent")
                }
                let pathURL = FileManager.default.fileExists(atPath: pathState.path) ? pathState : pathWeights
                let loaded = try OnlinePathMLPScorer(contentsOf: pathURL,
                                                     identityKeySHA256: fingerprint)
                guard loaded.canContinueLearning, let seed = loaded.checkpoint.trainedThroughEventID,
                      !seed.isEmpty else { throw OnlinePathMLPError.incompatibleWeights }
                pathModel = loaded
                marginRestore = loaded.checkpoint.scoreMarginCalibration
                pathSeedEventID = seed
                try reader.replayRecommendationHistory(seedThroughEventID: nil) { sequence, event in
                    processPath(event, databaseSequence: sequence)
                    lastProcessedDatabaseSequence = sequence
                }
                guard pathSeedReached else {
                    pathModel = nil
                    frozenPathInputs.removeAll()
                    throw StoreError("Trained model event boundary is absent from the local history")
                }
                ready = true
                try save()
                writeStatus()
                readinessLock.lock(); readyForRequests = true; readinessLock.unlock()
            } catch { failure = error.localizedDescription; writeStatus() }
        }
    }
    private func canAnswerNow() -> Bool {
        readinessLock.lock()
        if readyForRequests { readinessLock.unlock(); return true }
        guard !startupWaitUsed else { readinessLock.unlock(); return false }
        startupWaitUsed = true
        readinessLock.unlock()
        // A small seed may load before the user opens the first menu. A full
        // history replay is allowed to continue without holding the UI.
        _ = startupGroup.wait(timeout: .now() + .milliseconds(150))
        readinessLock.lock(); defer { readinessLock.unlock() }
        return readyForRequests
    }
    func consumeCommitted(throughDatabaseSequence sequence: Int64) {
        guard sequence > 0 else { return }
        commitLock.lock()
        committedThroughSequence = max(committedThroughSequence, sequence)
        guard !commitDrainScheduled else { commitLock.unlock(); return }
        commitDrainScheduled = true
        commitLock.unlock()
        queue.async { [self] in
            do { try drainCommittedHistory() }
            catch {
                commitLock.lock(); commitDrainScheduled = false; commitLock.unlock()
                failure = error.localizedDescription; writeStatus()
            }
        }
    }
    private func drainCommittedHistory() throws {
        while true {
            commitLock.lock()
            let through = committedThroughSequence
            guard ready, let historyReader, through > lastProcessedDatabaseSequence else {
                commitDrainScheduled = false; commitLock.unlock(); return
            }
            commitLock.unlock()
            try historyReader.replayCommitted(after: lastProcessedDatabaseSequence, through: through) { sequence, event in
                processPath(event, databaseSequence: sequence)
                lastProcessedDatabaseSequence = sequence
            }
            guard lastProcessedDatabaseSequence == through else {
                throw StoreError("Committed learning position is absent from the local history")
            }
            try autoreleasepool { try save() }
        }
    }
    func requiresCommittedCutoff() -> Bool {
        guard canAnswerNow() else { return false }
        return queue.sync { ready && pathModel != nil && pathSeedReached }
    }
    /// The pretrained path model returns its original action identities and
    /// transient parent locators. None of these candidates is persisted.
    func onlineRanked(current: Destination) -> [OnlinePathCandidate]? {
        onlineMenuRanking(current: current)?.ranked.map(\.candidate)
    }
    func onlineMenuRanking(current: Destination) -> OnlineMenuRanking? {
        guard canAnswerNow() else { return nil }
        return queue.sync {
            autoreleasepool {
            guard ready, pathSeedReached, let pathModel, let marginController else { return nil }
            do {
                let candidates = try pathHistory.candidates(current: current).filter {
                    MainMenuTargetScope.includes($0.recordedTarget?.kind ?? $0.derivedKind)
                }
                let scored = try candidates.map { candidate in
                    ScoredOnlinePathCandidate(candidate: candidate, score: try pathModel.score(candidate.row))
                }
                let prediction = try scorePrediction(id: "menu:" + UUID().uuidString,
                    sequence: lastProcessedDatabaseSequence, scored: scored)
                let preview = try marginController.preview(prediction, maximumCount: dynamicConfiguration.maximumItems)
                let ranked = scored.sorted {
                    $0.score == $1.score ? $0.candidate.actionID < $1.candidate.actionID : $0.score > $1.score
                }
                return OnlineMenuRanking(ranked: ranked, preview: preview)
            } catch {
                failure = error.localizedDescription
                writeStatus()
                return nil
            }
            }
        }
    }
    func flush() {
        queue.sync {
            do { try drainCommittedHistory(); if ready { try autoreleasepool { try save() } } }
            catch { failure = error.localizedDescription; writeStatus() }
        }
    }
    private func processPath(_ event: BehaviorEvent, databaseSequence: Int64) {
        guard pathModel != nil else { return }
        pathHistory.consume(event)
        let matured = pathLearner.consume(event)
        let canLearn = pathSeedReached
        if pathHistory.confirmedDestination(eventID: event.id) != nil {
            confirmedSequences[event.id] = databaseSequence
        }
        if canLearn, event.kind == .contextChanged,
           pathLearner.pendingInputEventIDs.contains(event.id),
           let current = pathHistory.confirmedDestination(eventID: event.id),
           let candidates = try? pathHistory.candidates(current: current) {
            var decision: ScoreMarginDecision?
            if var controller = marginController, let pathModel {
                do {
                    let scores = try candidates.filter { MainMenuTargetScope.includes($0.recordedTarget?.kind ?? $0.derivedKind) }.map {
                        ScoredOnlinePathCandidate(candidate: $0, score: try pathModel.score($0.row))
                    }
                    let prediction = try scorePrediction(id: event.id, sequence: databaseSequence, scored: scores)
                    decision = try controller.select(prediction, maximumCount: dynamicConfiguration.maximumItems)
                    marginController = controller
                } catch { marginFailure = error.localizedDescription }
            }
            frozenPathInputs[event.id] = FrozenPathInput(current: current, candidates: candidates,
                                                        marginDecision: decision)
            frozenPathOrder.append(event.id)
        }
        if canLearn {
            for feedback in matured where frozenPathInputs[feedback.inputEventID] != nil {
                readyFeedback[feedback.inputEventID] = feedback
            }
            let pending = pathLearner.pendingInputEventIDs
            while let first = frozenPathOrder.first, let frozen = frozenPathInputs[first] {
                if let feedback = readyFeedback.removeValue(forKey: first) {
                    learnPath(feedback, frozen: frozen, completedBy: event.id, databaseSequence: databaseSequence)
                } else if !pending.contains(first) {
                    discardMarginDecision(frozen.marginDecision)
                } else { break }
                frozenPathInputs[first] = nil
                frozenPathOrder.removeFirst()
            }
        }
        let kept = pathLearner.requiredConfirmedEventIDs.union([event.id])
            .union(frozenPathInputs.keys).union(readyFeedback.values.map(\.outcomeEventID))
        pathHistory.releaseConfirmedEvents(keeping: kept)
        confirmedSequences = confirmedSequences.filter { kept.contains($0.key) }
        if event.id == pathSeedEventID {
            pathSeedReached = true
            trainedThroughDatabaseSequence = databaseSequence
            do {
                if let state = marginRestore {
                    guard state.modelVersion == OnlinePathMLPScorer.modelVersion,
                          state.featureSchema == OnlinePathMLPScorer.featureSchema,
                          state.matchingRule == OnlinePathInputHistory.usefulActionRuleVersion + ":" + MainMenuTargetScope.version,
                          state.targetMissRate == dynamicConfiguration.targetMissRate,
                          state.learningRate == dynamicConfiguration.learningRate,
                          state.policyMaximumCount == dynamicConfiguration.maximumItems else {
                        throw ScoreMarginConformalError.incompatibleState
                    }
                    marginController = try AdaptiveScoreMarginConformal(state: state, startSequence: databaseSequence)
                } else { try startMarginController(at: databaseSequence) }
            } catch {
                marginFailure = error.localizedDescription
                try? startMarginController(at: databaseSequence)
            }
            marginRestore = nil
        }
    }
    private func startMarginController(at sequence: Int64) throws {
        marginController = try AdaptiveScoreMarginConformal(context: dynamicConfiguration.context,
            targetMissRate: dynamicConfiguration.targetMissRate, learningRate: dynamicConfiguration.learningRate,
            startSequence: sequence)
    }
    private func scorePrediction(id: String, sequence: Int64,
                                 scored: [ScoredOnlinePathCandidate]) throws -> ScoreMarginPrediction {
        let revision = (pathSeedEventID ?? "unseeded") + ":" + String(pathUpdates)
        return try ScoreMarginPrediction(predictionID: id, context: dynamicConfiguration.context,
            modelRevision: revision, trainedThroughSequence: trainedThroughDatabaseSequence,
            cutoffSequence: sequence,
            candidates: scored.map { ScoreMarginCandidate(actionID: $0.candidate.actionID, score: $0.score) })
    }
    private func learnPath(_ feedback: OnlinePathFeedback, frozen: FrozenPathInput,
                           completedBy eventID: String, databaseSequence: Int64) {
        guard let outcome = pathHistory.confirmedDestination(eventID: feedback.outcomeEventID),
              var pathModel else {
            discardMarginDecision(frozen.marginDecision)
            return
        }
        if !MainMenuTargetScope.includes(outcome.kind) { discardMarginDecision(frozen.marginDecision) }
        if MainMenuTargetScope.includes(outcome.kind), let decision = frozen.marginDecision, var controller = marginController {
            do {
                guard let sequence = confirmedSequences[feedback.outcomeEventID] else {
                    throw ScoreMarginConformalError.invalidOutcomeSequence
                }
                let indices = OnlinePathInputHistory.usefulActionIndices(for: outcome, current: frozen.current,
                                                                          candidates: frozen.candidates)
                let accepted = Set(indices.filter {
                    MainMenuTargetScope.includes(frozen.candidates[$0].recordedTarget?.kind ?? frozen.candidates[$0].derivedKind)
                }.map { frozen.candidates[$0].actionID })
                let observation = try accepted.isEmpty ?
                    decision.prediction.observeUnavailableOutcome(outcomeID: feedback.outcomeEventID, outcomeSequence: sequence) :
                    decision.prediction.observe(outcomeID: feedback.outcomeEventID, outcomeSequence: sequence,
                                                acceptableActionIDs: accepted)
                try controller.observe(decision, observation: observation)
                marginController = controller
                pathDirty = true
                pathLastUpdateEventID = eventID
                trainedThroughDatabaseSequence = databaseSequence
            } catch {
                marginFailure = error.localizedDescription
                // Invalid/unknown feedback is not a miss. Release the frozen
                // decision so one rejected sample cannot block later feedback.
                do { try controller.discard(decision); marginController = controller }
                catch { marginFailure = error.localizedDescription }
            }
        }
        let training = OnlinePathInputHistory.trainingRows(
            for: outcome, current: frozen.current, frozenCandidates: frozen.candidates,
            confirmedParents: pathHistory.confirmedParentChain(eventID: feedback.outcomeEventID) ?? [])
        guard !training.positiveIndices.isEmpty else { return }
        do {
            let result = try pathModel.learn(rows: training.rows,
                                             positiveIndices: training.positiveIndices,
                                             originalCandidateCount: training.originalCandidateCount,
                                             arrivalID: feedback.outcomeEventID)
            if result.updated {
                self.pathModel = pathModel
                pathLastUpdateEventID = eventID
                pathDirty = true
                pathUpdates += 1
                trainedThroughDatabaseSequence = databaseSequence
            }
        } catch { failure = error.localizedDescription; writeStatus() }
    }
    private func discardMarginDecision(_ decision: ScoreMarginDecision?) {
        guard let decision, var controller = marginController else { return }
        do { try controller.discard(decision); marginController = controller }
        catch { marginFailure = error.localizedDescription }
    }
    private func save() throws {
        guard pathDirty else { return }
        let encoder = JSONEncoder(); encoder.outputFormatting = [.sortedKeys, .withoutEscapingSlashes]
        if pathDirty, let pathModel {
            try FileManager.default.createDirectory(at: pathDirectory, withIntermediateDirectories: true,
                                                    attributes: [.posixPermissions: 0o700])
            var checkpoint = pathModel.checkpoint
            checkpoint.trainedThroughEventID = pathLastUpdateEventID
            checkpoint.scoreMarginCalibration = try marginController?.checkpoint(maximumCount: dynamicConfiguration.maximumItems)
            let pathURL = pathDirectory.appendingPathComponent("state.json")
            try encoder.encode(checkpoint).write(to: pathURL, options: .atomic)
            try FileManager.default.setAttributes([.posixPermissions: 0o600], ofItemAtPath: pathURL.path)
            pathDirty = false
        }
        writeStatus()
    }
    private func writeStatus() {
        // Aggregate health only, never input, prediction, inventory or menu lists.
        let url = pathDirectory.deletingLastPathComponent().deletingLastPathComponent().appendingPathComponent("runtime/recommendation-status.json")
        let value: [String: JSONValue] = ["version": .string(OnlinePathMLPScorer.modelVersion),
            "menuTargetScope": .string(MainMenuTargetScope.version),
            "localArrivalsRetainedForLearning": .bool(true),
            "ready": .bool(ready),
            "activeLearningModels": .number(ready && pathModel != nil ? 1 : 0),
            "onlinePathReady": .bool(pathModel != nil && pathSeedReached),
            "onlinePathVersion": .string(OnlinePathMLPScorer.modelVersion),
            "onlinePathUpdates": .number(Double(pathUpdates)),
            "onlinePathRunUpdates": .number(Double(pathUpdates)),
            "onlinePathTotalUpdates": .number(Double(pathModel?.checkpoint.optimizerState?.layer1Weights.step ?? 0)),
            "onlinePathError": failure.map(JSONValue.string) ?? .null,
            "dynamicMenuMaximumItems": .number(Double(dynamicConfiguration.maximumItems)),
            "dynamicMenuTargetMissRate": .number(dynamicConfiguration.targetMissRate),
            "marginCalibrationSamples": .number(Double(marginController?.sampleCount ?? 0)),
            "marginUnavailableOutcomes": .number(Double(marginController?.unavailableOutcomeCount ?? 0)),
            "marginFeedbackCount": .number(Double(marginController?.feedbackCount ?? 0)),
            "marginFeedbackCovered": .number(Double(marginController?.coveredFeedbackCount ?? 0)),
            "marginFeedbackThresholdCovered": .number(Double(marginController?.thresholdCoveredFeedbackCount ?? 0)),
            "marginFeedbackCapLimited": .number(Double(marginController?.capLimitedFeedbackCount ?? 0)),
            "marginCalibrationError": marginFailure.map(JSONValue.string) ?? .null,
            "error": failure.map(JSONValue.string) ?? .null]
        do {
            try FileManager.default.createDirectory(at: url.deletingLastPathComponent(), withIntermediateDirectories: true, attributes: [.posixPermissions: 0o700])
            try JSONEncoder().encode(value).write(to: url, options: .atomic)
            try FileManager.default.setAttributes([.posixPermissions: 0o600], ofItemAtPath: url.path)
        } catch { /* Diagnostics must not interrupt the collector. */ }
    }
}
