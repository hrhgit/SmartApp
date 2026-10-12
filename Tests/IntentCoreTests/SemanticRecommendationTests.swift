import XCTest
@testable import IntentCore

final class SemanticRecommendationTests: XCTestCase {
    private func app(_ name: String) -> Destination {
        Destination(id: "app:" + name, kind: .app, name: name, bundleID: name, admission: TargetAdmission(basis: "standalone_application"))
    }
    private func event(_ kind: EventKind, _ time: Double, id: String = UUID().uuidString,
                       data: [String: JSONValue] = [:], session: String = "test") -> BehaviorEvent {
        BehaviorEvent(id: id, sessionID: session, timestampMs: Int64(time), monotonicMs: time, kind: kind, source: "fixture", data: data)
    }
    private func setup(_ learner: inout StreamingSemanticLearner, _ targets: [Destination]) throws {
        _ = learner.consume(event(.sessionStart, 0, data: ["targetEligibilityPolicyVersion": .string(TargetEligibilityPolicy.version)]))
        for target in targets { _ = learner.consume(event(.targetUpdated, 1, data: ["target": try .encoded(target)])) }
    }
    private func visit(_ id: String, _ time: Double, _ target: Destination) -> BehaviorEvent {
        event(.contextChanged, time, id: id, data: ["targetID": .string(target.id)])
    }
    func testSemanticScopeAndExactLocatorStayDistinct() {
        func page(_ value: String) -> Destination { Destination(id: value, kind: .page, name: value, locator: value) }
        let a = SemanticMetadata.annotate(page("https://school.example/courses/123/a.pdf?version=1#p2"))
        let b = SemanticMetadata.annotate(page("https://school.example/courses/123/b.pdf"))
        let c = SemanticMetadata.annotate(page("https://school.example/courses/456/b.pdf"))
        XCTAssertGreaterThan(a.similarity(to: b), 0.7)
        XCTAssertLessThan(a.similarity(to: c), 0.7)
        XCTAssertEqual(a.identity.locator, "https://school.example/courses/123/a.pdf?version=1#p2")
        let other = SemanticMetadata.annotate(page("https://other.example/courses/123/a.pdf"))
        XCTAssertEqual(a.similarity(to: other), 0)
    }
    func testExactFrequencySemanticSharingAndDeduplication() {
        let source = app("a"), b = app("b"), c = app("c")
        let profile = SemanticMetadata.annotate(source)
        let record = SemanticLearningRecord(origin: source.id, annotation: profile, label: b.id, outcomeID: "one", outcomeTimestamp: 1)
        var model = SemanticRecommendationModel(records: [record, record])
        XCTAssertEqual(model.records.count, 1)
        model.learn(.init(origin: source.id, annotation: profile, label: c.id, outcomeID: "two", outcomeTimestamp: 2))
        XCTAssertEqual(Set(model.ranked(origin: source.id, annotation: profile)), [b.id, c.id])
        XCTAssertEqual(model.scores(origin: source.id, annotation: profile).values.reduce(0,+), 1, accuracy: 1e-12)
    }
    func testWeightedCrossEntropyMovesProbabilityTowardObservedClass() {
        let source = app("source"), b = app("b"), c = app("c")
        let profile = SemanticMetadata.annotate(source)
        var model = SemanticRecommendationModel()
        // Introducing c gives the dynamic softmax two classes; its observed
        // update should make c more likely than b at this same input.
        model.learn(.init(origin: source.id, annotation: profile, label: b.id, outcomeID: "b-0", outcomeTimestamp: 1))
        model.learn(.init(origin: source.id, annotation: profile, label: c.id, outcomeID: "c-0", outcomeTimestamp: 2))
        let afterC = model.scores(origin: source.id, annotation: profile)
        XCTAssertGreaterThan(afterC[c.id]!, afterC[b.id]!)
        // A later weighted b outcome applies the softmax cross-entropy gradient
        // to both output rows, reversing the preference without a count blend.
        model.learn(.init(origin: source.id, annotation: profile, label: b.id, outcomeID: "b-1", outcomeTimestamp: 3, weight: 1.5))
        let afterB = model.scores(origin: source.id, annotation: profile)
        XCTAssertGreaterThan(afterB[b.id]!, afterB[c.id]!)
        XCTAssertEqual(afterB.values.reduce(0, +), 1, accuracy: 1e-12)
    }
    func testShortConfirmedVisitSkippedButUnknownDwellRetainedAtMaturity() throws {
        var learner = StreamingSemanticLearner()
        let a = app("a"), b = app("b"), c = app("c")
        try setup(&learner, [a,b,c])
        XCTAssertTrue(learner.consume(visit("a", 1_000, a)).isEmpty)
        XCTAssertTrue(learner.consume(visit("b", 2_000, b)).isEmpty)
        XCTAssertTrue(learner.consume(visit("c", 3_000, c)).isEmpty)
        XCTAssertTrue(learner.consume(event(.heartbeat, 30_999)).isEmpty)
        let records = learner.consume(event(.heartbeat, 31_000))
        XCTAssertEqual(records.count, 1)
        XCTAssertEqual(records.first?.origin, a.id)
        XCTAssertEqual(records.first?.label, c.id)
        XCTAssertEqual(records.first?.outcomeID, "c")
    }
    func testResolutionIsNotAnotherArrivalAndWindowsFallBackToApp() throws {
        var learner = StreamingSemanticLearner()
        let a = app("a"), b = app("b")
        let window = Destination(id: "window:b", kind: .window, name: "Window", bundleID: "b", admission: TargetAdmission(basis: "standard_work_window"))
        try setup(&learner, [a,b,window])
        _ = learner.consume(visit("a", 1_000, a)); _ = learner.consume(visit("b", 2_000, b))
        _ = learner.consume(event(.contextChanged, 2_100, id: "resolution", data: ["targetID": .string(window.id), "reason": .string("resolution_update"), "resolvesContextEventID": .string("b")]))
        let result = learner.consume(event(.heartbeat, 31_000))
        XCTAssertEqual(result.map(\.label), [b.id]); XCTAssertEqual(result.map(\.outcomeID), ["b"])
    }
    func testCaptureGapAndGranularityLossDoNotBridgeTrajectories() throws {
        for gap in [true,false] {
            var learner = StreamingSemanticLearner()
            let a = app("a"), b = app("b")
            let page = Destination(id: "page:a", kind: .page, name: "Page", bundleID: "a", locator: "https://example.test/", admission: TargetAdmission(basis: "observed_browser_page"))
            // Use the collector's actual web admission contract.
            var valid = page
            valid.origin = "https://example.test"
            valid.admission = TargetEligibilityPolicy.page(origin: valid.origin, supportedSource: true, excluded: false).admission
            try setup(&learner, [a,b,valid])
            _ = learner.consume(visit("page", 1_000, valid))
            if gap { _ = learner.consume(event(.captureGap, 2_000)) }
            else { _ = learner.consume(visit("coarse", 2_000, a)) }
            _ = learner.consume(visit("b", 3_000, b))
            let result = learner.consume(event(.heartbeat, 31_000))
            XCTAssertFalse(result.contains { $0.origin == valid.id })
        }
    }
    func testExcludedForegroundDoesNotTrainDirectTransitionButLaterVisitsStillTrain() throws {
        var learner = StreamingSemanticLearner()
        let a = app("a"), b = app("b"), c = app("c")
        try setup(&learner, [a, b, c])
        let scopeID = UUID().uuidString
        _ = learner.consume(visit("a", 1_000, a))
        _ = learner.consume(event(.observationScope, 2_000, data: [
            "scopeVersion": .string("excluded-foreground-v1"), "scopeID": .string(scopeID),
            "phase": .string("begin"), "category": .string("excluded_app")]))
        _ = learner.consume(event(.heartbeat, 10_000))
        _ = learner.consume(event(.observationScope, 20_000, data: [
            "scopeVersion": .string("excluded-foreground-v1"), "scopeID": .string(scopeID),
            "phase": .string("end"), "category": .string("excluded_app")]))
        _ = learner.consume(visit("b", 21_000, b))
        _ = learner.consume(visit("c", 27_000, c))
        let records = learner.consume(event(.heartbeat, 51_000))
        XCTAssertEqual(records.map(\.origin), [b.id])
        XCTAssertEqual(records.map(\.label), [c.id])
    }
    func testLearningDisabledBeforeSeedBoundaryAndRecommendationObservedOnce() throws {
        var learner = StreamingSemanticLearner(); learner.allowNewAnchors = false
        let a = app("a"), b = app("b")
        try setup(&learner, [a,b]); _ = learner.consume(visit("before", 1_000, a))
        learner.allowNewAnchors = true
        var selected = visit("selected", 2_000, b); selected.decisionID = "decision"
        _ = learner.consume(selected)
        _ = learner.consume(event(.recommendationFeedback, 2_500, data: ["targetID": .string(b.id), "action": .string("select")]))
        _ = learner.consume(visit("return", 8_000, a))
        let records = learner.consume(event(.heartbeat, 32_000))
        XCTAssertEqual(records.count, 1); XCTAssertEqual(records.first?.origin, b.id)
        XCTAssertEqual(records.first?.source, "recommendation_observed")
    }
}
