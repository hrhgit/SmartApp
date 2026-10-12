import XCTest
@testable import IntentCore

final class ScoreMarginConformalTests: XCTestCase {
    func testExplicitSixItemLimitCountsASeventhCorrectTargetAsCapMiss() throws {
        var adaptive = try AdaptiveScoreMarginConformal(context: context(), targetMissRate: 0.1,
                                                        learningRate: 0.005, startSequence: 0)
        let p = try prediction(scores: Array(repeating: 1, count: 9))
        let decision = try adaptive.select(p, maximumCount: 6)
        XCTAssertEqual(decision.selected.count, 6)
        let observation = try p.observe(outcomeID: "seventh", outcomeSequence: 101,
                                        acceptableActionIDs: ["action-6"])
        XCTAssertEqual(observation.acceptedRank, 7)
        XCTAssertFalse(try adaptive.observe(decision, observation: observation))
        XCTAssertEqual(adaptive.thresholdCoveredFeedbackCount, 1)
        XCTAssertEqual(adaptive.capLimitedFeedbackCount, 1)
        XCTAssertEqual(adaptive.unavailableOutcomeCount, 0)
    }

    func testColdPreviewIsPureAndQualificationOccursBeforeSixItemLimit() throws {
        var adaptive = try AdaptiveScoreMarginConformal(context: context(), targetMissRate: 0.1,
                                                        learningRate: 0.005, startSequence: 0)
        let p = try prediction(scores: [9, 8, 7, 6, 5, 4, 3, 2])
        let preview = try adaptive.preview(p, maximumCount: 6)
        let qualified = try preview.restrictingPreview(toActionIDs: ["action-2", "action-3", "action-4",
                                                                     "action-5", "action-6", "action-7"])
        XCTAssertEqual(qualified.selected.map(\.actionID), (2..<8).map { "action-\($0)" })
        XCTAssertEqual(adaptive.pendingCount, 0)
        XCTAssertEqual(adaptive.sampleCount, 0)
        XCTAssertEqual(adaptive.effectiveMissRate, 0.1)
        let learning = try adaptive.select(p, maximumCount: 6)
        assertError(.foreignDecision, try learning.restrictingPreview(toActionIDs: ["action-0"]))
    }

    func testScalarCheckpointRestoresWithoutPersistingPredictionIdentities() throws {
        var adaptive = try AdaptiveScoreMarginConformal(context: context(), targetMissRate: 0.1,
                                                        learningRate: 0.005, startSequence: 0)
        let p = try prediction("private-prediction-id", scores: [2], ids: ["private-action-id"])
        let decision = try adaptive.select(p, maximumCount: 6)
        let observed = try p.observeUnavailableOutcome(outcomeID: "private-outcome-id", outcomeSequence: 101)
        XCTAssertFalse(try adaptive.observe(decision, observation: observed))
        let bytes = try JSONEncoder().encode(adaptive.checkpoint(maximumCount: 6))
        let text = String(decoding: bytes, as: UTF8.self)
        XCTAssertFalse(text.contains("private-prediction-id"))
        XCTAssertFalse(text.contains("private-action-id"))
        XCTAssertFalse(text.contains("private-outcome-id"))
        let state = try JSONDecoder().decode(ScoreMarginConformalState.self, from: bytes)
        let restored = try AdaptiveScoreMarginConformal(state: state, startSequence: 101)
        XCTAssertEqual(restored.sampleCount, 1)
        XCTAssertEqual(restored.unavailableOutcomeCount, 1)
        XCTAssertEqual(restored.feedbackCount, 1)
        XCTAssertEqual(restored.effectiveMissRate, adaptive.effectiveMissRate)
        XCTAssertEqual(restored.threshold, adaptive.threshold)
        XCTAssertEqual(restored.pendingCount, 0)
        assertError(.predictionBeforeCalibration, try restored.preview(
            prediction("earlier-preview", scores: [1], sequence: 100), maximumCount: 6))
        assertError(.predictionBeforeCalibration, try { () -> ScoreMarginDecision in
            var copy = restored
            return try copy.select(prediction("earlier", scores: [1], sequence: 100))
        }())
        var malformed = state
        malformed.margins = [-1]
        assertError(.incompatibleState, try AdaptiveScoreMarginConformal(state: malformed, startSequence: 101))
    }
    private func context(rule: String = "exact-test-v1") throws -> ScoreMarginContext {
        try ScoreMarginContext(modelVersion: "test-ranker-v1", featureSchema: "test-input-v1",
                               matchingRule: rule)
    }

    private func prediction(_ id: String = "next", scores: [Double], ids: [String]? = nil,
                            sequence: Int64 = 100, revision: String = "weights-1",
                            trainedThrough: Int64 = 0, rule: String = "exact-test-v1") throws -> ScoreMarginPrediction {
        let actionIDs = ids ?? scores.indices.map { "action-\($0)" }
        return try ScoreMarginPrediction(predictionID: id, context: context(rule: rule),
            modelRevision: revision, trainedThroughSequence: trainedThrough, cutoffSequence: sequence,
            candidates: zip(actionIDs, scores).map { .init(actionID: $0.0, score: $0.1) })
    }

    private func observations(_ gaps: [Double]) throws -> [ScoreMarginObservation] {
        try gaps.enumerated().map { index, gap in
            let cut = Int64(2 * index + 1)
            let p = try prediction("seed-\(index)", scores: gap.isFinite ? [100, 100 - gap] : [100],
                                   ids: gap.isFinite ? ["top", "truth"] : ["top"], sequence: cut)
            return try p.observe(outcomeID: "outcome-\(index)", outcomeSequence: cut + 1,
                                 acceptableActionIDs: ["truth"])
        }
    }

    private func calibration(_ gaps: [Double] = [0, 0, 1, 1], alpha: Double = 0.4) throws -> ScoreMarginCalibration {
        try ScoreMarginCalibration(observations: observations(gaps), targetMissRate: alpha)
    }

    private func assertError<T>(_ expected: ScoreMarginConformalError,
                                _ operation: @autoclosure () throws -> T,
                                file: StaticString = #filePath, line: UInt = #line) {
        XCTAssertThrowsError(try operation(), file: file, line: line) {
            XCTAssertEqual($0 as? ScoreMarginConformalError, expected, file: file, line: line)
        }
    }

    func testCorrectedQuantileUsesNPlusOneAndIncludesBoundary() throws {
        let fitted = try calibration([0, 1, 2, 3], alpha: 0.3)
        XCTAssertEqual(fitted.quantileRank, 4) // ordinary n-based percentile would select 2
        XCTAssertEqual(fitted.threshold, .finite(3))
        let p = try prediction(scores: [10, 7, 6.5])
        XCTAssertEqual(try fitted.select(p).selected.map(\.actionID), ["action-0", "action-1"])
    }

    func testVariableSizesIncludingMoreThanFourHaveNoCapOrRefill() throws {
        let fitted = try calibration()
        let one = try fitted.select(prediction("one", scores: [9, 6, 5, 4]))
        let three = try fitted.select(prediction("three", scores: [9, 8.75, 8.5, 5]))
        let many = try fitted.select(prediction("many", scores: (0..<37).map { 10 - Double($0) / 64 }))
        XCTAssertEqual(one.selected.count, 1)
        XCTAssertEqual(three.selected.count, 3)
        XCTAssertEqual(many.selected.count, 37)
    }

    func testUnsortedNegativeScoresTiesAndOffsetsHaveStableSelection() throws {
        let fitted = try calibration()
        let scores = [-7.5, -7, -8, -10, -7]
        let ids = ["middle", "z", "boundary", "outside", "a"]
        let p = try prediction(scores: scores, ids: ids)
        let shifted = try prediction("shifted", scores: scores.map { $0 + 1_024 }, ids: ids)
        XCTAssertEqual(try fitted.select(p).selected.map(\.actionID), ["a", "z", "middle", "boundary"])
        XCTAssertEqual(try fitted.select(p).selected.map(\.actionID),
                       try fitted.select(shifted).selected.map(\.actionID))
    }

    func testSmallCalibrationReturnsAllWithoutInventingFiniteThreshold() throws {
        let fitted = try calibration([0], alpha: 0.01)
        XCTAssertEqual(fitted.quantileRank, 2)
        XCTAssertEqual(fitted.threshold, .allCandidates)
        let p = try prediction(scores: [10, -1_000, -2_000])
        XCTAssertEqual(try fitted.select(p).selected.count, 3)
    }

    func testAbsentAcceptableActionsStayInDenominatorAndNeverBecomeHits() throws {
        let fitted = try calibration([0, 1, .infinity], alpha: 0.4)
        XCTAssertEqual(fitted.sampleCount, 3)
        XCTAssertEqual(fitted.unavailableOutcomeCount, 1)
        XCTAssertEqual(fitted.threshold, .allCandidates)
        let p = try prediction(scores: [10, 8], ids: ["one", "two"])
        let decision = try fitted.select(p)
        let absent = try p.observe(outcomeID: "new-outcome", outcomeSequence: 101,
                                   acceptableActionIDs: ["new-target"])
        XCTAssertFalse(absent.hasAcceptableCandidate)
        XCTAssertFalse(try decision.covers(absent))
        let empty = try prediction("empty", scores: [])
        XCTAssertTrue(try fitted.select(empty).selected.isEmpty)
        XCTAssertFalse(try empty.observe(outcomeID: "another", outcomeSequence: 101,
                                         acceptableActionIDs: ["truth"]).hasAcceptableCandidate)
    }

    func testSeveralLegalActionsAreOneOutcomeAndMatchingDefinitionsCannotMix() throws {
        let p = try prediction(scores: [10, 9.5, 8], ids: ["wrong", "parent", "exact"], rule: "path-test-v1")
        let observed = try p.observe(outcomeID: "arrival", outcomeSequence: 101,
                                     acceptableActionIDs: ["parent", "exact"])
        XCTAssertEqual(observed.margin, 0.5)
        let fitted = try ScoreMarginCalibration(observations: [observed], targetMissRate: 0.6)
        XCTAssertEqual(fitted.sampleCount, 1)
        XCTAssertEqual(fitted.threshold, .finite(0.5))
        let exact = try prediction("exact-only", scores: [10, 8], ids: ["wrong", "exact"],
                                   sequence: 102)
        let exactObservation = try exact.observe(outcomeID: "arrival-2", outcomeSequence: 103,
                                                 acceptableActionIDs: ["exact"])
        assertError(.incompatibleContext, try ScoreMarginCalibration(
            observations: [observed, exactObservation], targetMissRate: 0.6))
    }

    func testCalibrationRejectsDuplicatePredictionsOutcomesAndMixedWeights() throws {
        let samples = try observations([0, 1])
        assertError(.duplicatePrediction, try ScoreMarginCalibration(observations: [samples[0], samples[0]],
                                                                     targetMissRate: 0.2))
        let p = try prediction("different", scores: [2, 1], ids: ["wrong", "truth"])
        let sameOutcome = try p.observe(outcomeID: samples[0].outcomeID, outcomeSequence: 101,
                                        acceptableActionIDs: ["truth"])
        assertError(.duplicateOutcome, try ScoreMarginCalibration(observations: [samples[0], sameOutcome],
                                                                  targetMissRate: 0.2))
        let changed = try prediction("changed", scores: [2, 1], ids: ["wrong", "truth"], revision: "weights-2")
        let changedObservation = try changed.observe(outcomeID: "other", outcomeSequence: 101,
                                                     acceptableActionIDs: ["truth"])
        assertError(.incompatibleModelRevision, try ScoreMarginCalibration(
            observations: [samples[0], changedObservation], targetMissRate: 0.2))
    }

    func testFutureTrainingOutcomesAndCalibrationCannotEnterEarlierPredictions() throws {
        assertError(.invalidPredictionSequence, try prediction(scores: [1], sequence: 2, trainedThrough: 3))
        let p = try prediction(scores: [1])
        assertError(.invalidOutcomeSequence, try p.observe(outcomeID: "outcome", outcomeSequence: 100,
                                                           acceptableActionIDs: ["action-0"]))
        let fitted = try calibration()
        assertError(.predictionBeforeCalibration, try fitted.select(prediction(scores: [1], sequence: 7)))
        assertError(.incompatibleModelRevision, try fitted.select(prediction(scores: [1], revision: "weights-2")))
        assertError(.incompatibleModelRevision, try fitted.select(prediction(scores: [1], trainedThrough: 1)))
        assertError(.duplicatePrediction, try fitted.select(prediction("seed-0", scores: [1])))
    }

    func testRecomputedScoresCannotReplaceFrozenPredictionForFeedback() throws {
        let fitted = try calibration()
        let original = try prediction(scores: [10, 6], ids: ["wrong", "truth"])
        let decision = try fitted.select(original)
        let recomputed = try prediction(scores: [10, 10], ids: ["wrong", "truth"])
        let wrongSnapshot = try recomputed.observe(outcomeID: "outcome", outcomeSequence: 101,
                                                   acceptableActionIDs: ["truth"])
        assertError(.foreignDecision, try decision.covers(wrongSnapshot))
        let actual = try original.observe(outcomeID: "outcome", outcomeSequence: 101,
                                          acceptableActionIDs: ["truth"])
        XCTAssertFalse(try decision.covers(actual))
    }

    func testInputValidationDoesNotSilentlyDropBadScoresOrMissingLabels() throws {
        for score in [Double.nan, .infinity, -.infinity] {
            assertError(.nonfiniteScore, try prediction(scores: [score]))
        }
        assertError(.marginOverflow, try prediction(scores: [.greatestFiniteMagnitude, -.greatestFiniteMagnitude]))
        assertError(.duplicateCandidate, try prediction(scores: [1, 2], ids: ["same", "same"]))
        assertError(.invalidIdentifier, try prediction(scores: [1], ids: [" "]))
        let p = try prediction(scores: [1])
        assertError(.emptyAcceptableActions, try p.observe(outcomeID: "observed", outcomeSequence: 101,
                                                          acceptableActionIDs: []))
        assertError(.emptyCalibration, try ScoreMarginCalibration(observations: [], targetMissRate: 0.2))
        for alpha in [0.0, 1, -0.2, Double.nan, .infinity] {
            assertError(.invalidMissRate, try calibration(alpha: alpha))
        }
    }

    func testFiniteSampleCoverageByEnumeratingEveryHeldOutRank() throws {
        // Five distinct, exchangeable margins: each rank is held out once.
        // Corrected rank 4 among the remaining 4 must cover exactly 4/5.
        var hits = 0
        for heldOut in 0..<5 {
            let fitted = try calibration((0..<5).filter { $0 != heldOut }.map(Double.init), alpha: 0.2)
            let p = try prediction("holdout", scores: [10, 10 - Double(heldOut)], ids: ["top", "truth"])
            let decision = try fitted.select(p)
            let observed = try p.observe(outcomeID: "holdout-outcome", outcomeSequence: 101,
                                         acceptableActionIDs: ["truth"])
            if try decision.covers(observed) { hits += 1 }
        }
        XCTAssertEqual(hits, 4)
    }

    func testLowerDesiredMissRateCannotShrinkSelectedSet() throws {
        let p = try prediction(scores: [10, 9, 8, 7, 6])
        let smaller = try calibration([0, 1, 2, 3], alpha: 0.6).select(p)
        let larger = try calibration([0, 1, 2, 3], alpha: 0.2).select(p)
        XCTAssertTrue(Set(smaller.selected.map(\.actionID)).isSubset(of: Set(larger.selected.map(\.actionID))))
        XCTAssertGreaterThan(larger.selected.count, smaller.selected.count)
    }

    func testAdaptiveMissUsesFrozenSetBeforeUpdatingThreshold() throws {
        let fitted = try calibration([0, 1, 2, 3], alpha: 0.3)
        var adaptive = try AdaptiveScoreMarginConformal(calibration: fitted, learningRate: 0.2)
        let p = try prediction(scores: [10, 6], ids: ["wrong", "truth"])
        let decision = try adaptive.select(p)
        XCTAssertEqual(decision.threshold, .finite(3))
        XCTAssertFalse(try adaptive.observe(decision, outcomeID: "arrival", outcomeSequence: 101,
                                           acceptableActionIDs: ["truth"]))
        XCTAssertEqual(adaptive.effectiveMissRate, 0.16, accuracy: 1e-12)
        XCTAssertEqual(adaptive.sampleCount, 5)
        XCTAssertEqual(adaptive.feedbackCount, 1)
        XCTAssertEqual(adaptive.coveredFeedbackCount, 0)
        XCTAssertEqual(adaptive.threshold, .allCandidates)
        XCTAssertEqual(decision.threshold, .finite(3))
        XCTAssertEqual(decision.selected.map(\.actionID), ["wrong"])
    }

    func testAdaptiveHitNarrowsAndWeightRevisionsMayEvolveWithinContext() throws {
        var adaptive = try AdaptiveScoreMarginConformal(calibration: calibration([0, 1, 2, 3], alpha: 0.3),
                                                        learningRate: 0.2)
        let p = try prediction(scores: [10, 9], ids: ["wrong", "truth"], revision: "weights-2", trainedThrough: 99)
        let decision = try adaptive.select(p)
        XCTAssertTrue(try adaptive.observe(decision, outcomeID: "arrival", outcomeSequence: 101,
                                          acceptableActionIDs: ["truth"]))
        XCTAssertEqual(adaptive.effectiveMissRate, 0.36, accuracy: 1e-12)
        XCTAssertEqual(adaptive.threshold, .finite(2))
        XCTAssertEqual(adaptive.coveredFeedbackCount, 1)
        assertError(.incompatibleContext, try adaptive.select(prediction("changed-rule", scores: [1],
                                                                       sequence: 102, rule: "path-test-v1")))
    }

    func testAdaptiveOutOfRangeAlphaExplicitlyProducesAllOrNoneWithoutClipping() throws {
        var wider = try AdaptiveScoreMarginConformal(calibration: calibration([0, 1, 2, 3], alpha: 0.3),
                                                     learningRate: 1)
        let miss = try wider.select(prediction(scores: [10, 6], ids: ["wrong", "truth"]))
        _ = try wider.observe(miss, outcomeID: "miss", outcomeSequence: 101, acceptableActionIDs: ["truth"])
        XCTAssertEqual(wider.effectiveMissRate, -0.4, accuracy: 1e-12)
        XCTAssertEqual(wider.threshold, .allCandidates)
        var narrower = try AdaptiveScoreMarginConformal(calibration: calibration(), learningRate: 2)
        let hit = try narrower.select(prediction(scores: [1], ids: ["truth"]))
        _ = try narrower.observe(hit, outcomeID: "hit", outcomeSequence: 101, acceptableActionIDs: ["truth"])
        XCTAssertEqual(narrower.effectiveMissRate, 1.2, accuracy: 1e-12)
        XCTAssertEqual(narrower.threshold, .noCandidates)
        let next = try narrower.select(prediction("next-empty", scores: [1], sequence: 102))
        XCTAssertTrue(next.selected.isEmpty)
    }

    func testFeedbackOrderFailuresAreAtomicAndDuplicateOutcomesAreRejected() throws {
        var adaptive = try AdaptiveScoreMarginConformal(calibration: calibration(), learningRate: 0.1)
        let first = try adaptive.select(prediction("first", scores: [1], ids: ["truth"]))
        let second = try adaptive.select(prediction("second", scores: [1], ids: ["truth"], sequence: 101))
        assertError(.feedbackOutOfOrder, try adaptive.observe(second, outcomeID: "arrival-2", outcomeSequence: 103,
                                                              acceptableActionIDs: ["truth"]))
        assertError(.duplicateOutcome, try adaptive.observe(first, outcomeID: "outcome-0", outcomeSequence: 102,
                                                            acceptableActionIDs: ["truth"]))
        XCTAssertEqual(adaptive.feedbackCount, 0)
        XCTAssertEqual(adaptive.sampleCount, 4)
        XCTAssertEqual(adaptive.effectiveMissRate, 0.4)
        XCTAssertEqual(adaptive.pendingCount, 2)
        _ = try adaptive.observe(first, outcomeID: "arrival-1", outcomeSequence: 102, acceptableActionIDs: ["truth"])
        assertError(.foreignDecision, try adaptive.observe(first, outcomeID: "again", outcomeSequence: 103,
                                                           acceptableActionIDs: ["truth"]))
        _ = try adaptive.observe(second, outcomeID: "arrival-2", outcomeSequence: 103, acceptableActionIDs: ["truth"])
        XCTAssertEqual(adaptive.feedbackCount, 2)
        XCTAssertEqual(adaptive.pendingCount, 0)
    }

    func testDiscardedOrUnknownOutcomesNeverChangeCalibration() throws {
        var adaptive = try AdaptiveScoreMarginConformal(calibration: calibration(), learningRate: 0.1)
        let before = adaptive.threshold
        let discarded = try adaptive.select(prediction("unconfirmed", scores: [10, 5]))
        try adaptive.discard(discarded)
        XCTAssertEqual(adaptive.threshold, before)
        XCTAssertEqual(adaptive.effectiveMissRate, 0.4)
        XCTAssertEqual(adaptive.sampleCount, 4)
        XCTAssertEqual(adaptive.feedbackCount, 0)
        XCTAssertEqual(adaptive.pendingCount, 0)
        assertError(.duplicatePrediction, try adaptive.select(prediction("unconfirmed", scores: [10, 5])))
        let next = try adaptive.select(prediction("confirmed", scores: [1], ids: ["truth"], sequence: 101))
        XCTAssertTrue(try adaptive.observe(next, outcomeID: "arrival", outcomeSequence: 102,
                                           acceptableActionIDs: ["truth"]))
    }

    func testAbsentOutcomeIsCountedAsAdaptiveMissEvenForAllCandidates() throws {
        var adaptive = try AdaptiveScoreMarginConformal(calibration: calibration([0], alpha: 0.1),
                                                        learningRate: 0.1)
        let decision = try adaptive.select(prediction(scores: [1], ids: ["known"]))
        XCTAssertEqual(decision.threshold, .allCandidates)
        XCTAssertFalse(try adaptive.observe(decision, outcomeID: "new-arrival", outcomeSequence: 101,
                                           acceptableActionIDs: ["unknown"]))
        XCTAssertEqual(adaptive.sampleCount, 2)
        XCTAssertEqual(adaptive.unavailableOutcomeCount, 1)
        XCTAssertEqual(adaptive.coveredFeedbackCount, 0)
    }

    func testInvalidAdaptiveConfigurationAndForeignDecisionsAreRejected() throws {
        let fitted = try calibration()
        for rate in [0.0, -1, Double.nan, .infinity] {
            assertError(.invalidLearningRate, try AdaptiveScoreMarginConformal(calibration: fitted, learningRate: rate))
        }
        var a = try AdaptiveScoreMarginConformal(calibration: fitted, learningRate: 0.1)
        var b = try AdaptiveScoreMarginConformal(calibration: fitted, learningRate: 0.1)
        let decision = try a.select(prediction(scores: [1]))
        assertError(.foreignDecision, try b.discard(decision))
        assertError(.duplicatePrediction, try a.select(prediction(scores: [1])))
        assertError(.predictionBeforeCalibration, try a.select(prediction("past", scores: [1], sequence: 99)))
        XCTAssertEqual(b.pendingCount, 0)
        XCTAssertEqual(b.sampleCount, 4)
    }
}
