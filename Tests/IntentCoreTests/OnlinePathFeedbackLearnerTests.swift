import XCTest
@testable import IntentCore

final class OnlinePathFeedbackLearnerTests: XCTestCase {
    private let session = "feedback-fixture"
    private var coverage: JSONValue {
        .object(["version": .string(StateCoverage.foregroundOnlyVersion),
                 "status": .string("foreground_only"), "sources": .array([])])
    }

    private func target(_ bundle: String) -> Destination {
        Destination(id: "app:" + bundle, kind: .app, name: bundle, bundleID: bundle,
                    admission: TargetAdmission(basis: "standalone_application"))
    }

    private func event(_ kind: EventKind, _ ms: Int64, _ sequence: Int,
                       _ data: [String: JSONValue] = [:], session: String? = nil) -> BehaviorEvent {
        var value = BehaviorEvent(sessionID: session ?? self.session, timestampMs: ms,
                                  monotonicMs: Double(ms), kind: kind,
                                  source: kind == .observationScope ? "lifecycle" : "fixture",
                                  data: data)
        value.sequence = sequence
        return value
    }

    private func start() -> BehaviorEvent {
        event(.sessionStart, 0, 1,
              ["targetEligibilityPolicyVersion": .string(TargetEligibilityPolicy.version),
               "stateCoverageVersion": .string(StateCoverage.foregroundOnlyVersion)])
    }

    private func update(_ target: Destination, _ ms: Int64, _ sequence: Int) throws -> BehaviorEvent {
        try event(.targetUpdated, ms, sequence, ["target": .encoded(target)])
    }

    private func arrival(_ target: Destination, _ ms: Int64, _ sequence: Int,
                         reason: String = "foreground_change") -> BehaviorEvent {
        event(.contextChanged, ms, sequence,
              ["targetID": .string(target.id), "reason": .string(reason),
               "inputCoverage": coverage])
    }

    func testCompleteObservationUsesConfirmedFirstStep() throws {
        let a = target("a"), b = target("b"), c = target("c")
        let first = arrival(a, 100, 5)
        let second = arrival(b, 1_000, 6)
        let third = arrival(c, 1_100, 7)
        var learner = OnlinePathFeedbackLearner()
        var results: [OnlinePathFeedback] = []
        for item in [start(), try update(a, 1, 2), try update(b, 2, 3),
                     try update(c, 3, 4), first, second, third,
                     event(.heartbeat, 30_101, 8)] {
            results += learner.consume(item)
        }
        XCTAssertEqual(results.first { $0.inputEventID == first.id },
                       OnlinePathFeedback(inputEventID: first.id,
                                          outcomeEventID: second.id,
                                          evidence: .completeObservation))
        XCTAssertFalse(results.contains { $0.inputEventID == first.id && $0.outcomeEventID == third.id })
        XCTAssertTrue(learner.discardableInputEventIDs.contains(first.id))
    }

    func testIncompleteSampleKeepsOnlyAlreadyConfirmedFirstStep() throws {
        let a = target("a"), b = target("b")
        let before = arrival(a, 100, 4)
        let after = arrival(b, 1_000, 5)
        let gap = event(.captureGap, 1_500, 6, ["reason": .string("unknown_failure")])
        var learner = OnlinePathFeedbackLearner()
        var results: [OnlinePathFeedback] = []
        for item in [start(), try update(a, 1, 2), try update(b, 2, 3),
                     before, after, gap] {
            results += learner.consume(item)
        }
        XCTAssertEqual(results, [OnlinePathFeedback(inputEventID: before.id,
                                                    outcomeEventID: after.id,
                                                    evidence: .confirmedFirstStep)])
        XCTAssertTrue(learner.discardableInputEventIDs.contains(before.id))
    }

    func testUnknownBoundaryNeverTurnsLaterArrivalIntoFirstStep() throws {
        let a = target("a"), b = target("b")
        let before = arrival(a, 100, 4)
        let after = arrival(b, 500, 6)
        let events = try [start(), update(a, 1, 2), update(b, 2, 3), before,
                          event(.captureGap, 200, 5, ["reason": .string("unknown_failure")]),
                          after, event(.sessionEnd, 1_000, 7)]
        var learner = OnlinePathFeedbackLearner()
        XCTAssertTrue(events.flatMap { learner.consume($0) }.isEmpty)
    }

    func testSessionChangeFlushesOnlyAlreadyConfirmedStep() throws {
        let a = target("a"), b = target("b")
        let before = arrival(a, 100, 4)
        let after = arrival(b, 500, 5)
        var next = start()
        next.id = UUID().uuidString
        next.sessionID = "next-session"
        next.timestampMs = 1_000
        next.monotonicMs = 1_000
        let events = try [start(), update(a, 1, 2), update(b, 2, 3),
                          before, after, next]
        var learner = OnlinePathFeedbackLearner()
        XCTAssertEqual(events.flatMap { learner.consume($0) },
                       [OnlinePathFeedback(inputEventID: before.id,
                                           outcomeEventID: after.id,
                                           evidence: .confirmedFirstStep)])
    }

    func testPageToItsAlreadyForegroundOwnerAppIsNotAChange() throws {
        let chrome = target("chrome")
        let page = Destination(id: "page:one", kind: .page, name: "Page",
                               origin: "https://example.test", locator: "https://example.test/one",
                               admission: TargetAdmission(basis: "supported_browser_page"))
        var before = arrival(page, 100, 4)
        before.app = AppContext(bundleID: "chrome", name: "Chrome")
        var after = arrival(chrome, 500, 5)
        after.app = AppContext(bundleID: "chrome", name: "Chrome")
        let events = try [start(), update(chrome, 1, 2), update(page, 2, 3),
                          before, after,
                          event(.captureGap, 1_000, 6, ["reason": .string("unknown_failure")])]
        var learner = OnlinePathFeedbackLearner()
        XCTAssertTrue(events.flatMap { learner.consume($0) }.isEmpty)
    }

    private func bridgeFeedback(gapMs: Int64, delayMs: Int64, afterSame: Bool = false,
                                nested: Bool = false, newSession: Bool = false)
        throws -> [OnlinePathFeedback] {
        let a = target("a"), b = target("b")
        var learner = OnlinePathFeedbackLearner()
        let before = arrival(a, 100, 4)
        let begin = event(.captureGap, 200, 5, ["reason": .string("observation_suspended")])
        let end = event(.captureGap, 200 + gapMs, nested ? 7 : 6,
                        ["reason": .string("observation_resumed")])
        let after = event(.contextChanged, 200 + gapMs + delayMs, nested ? 8 : 7,
                          ["targetID": .string(afterSame ? a.id : b.id),
                           "reason": .string("foreground_change"),
                           "inputCoverage": coverage],
                          session: newSession ? "other-session" : nil)
        var stream = [start(), try update(a, 1, 2), try update(b, 2, 3), before, begin]
        if nested {
            stream.append(event(.captureGap, 250, 6,
                                ["reason": .string("observation_suspended")]))
        }
        stream += [end, after]
        return stream.flatMap { learner.consume($0) }
    }

    func testStrictLegacyShortGapAndRejectedBoundaries() throws {
        let accepted = try bridgeFeedback(gapMs: 1_000, delayMs: 1_000)
        XCTAssertEqual(accepted.map(\.evidence), [.legacyShortGap])
        XCTAssertTrue(try bridgeFeedback(gapMs: 1_001, delayMs: 100).isEmpty)
        XCTAssertTrue(try bridgeFeedback(gapMs: 100, delayMs: 1_001).isEmpty)
        XCTAssertTrue(try bridgeFeedback(gapMs: 100, delayMs: 100, afterSame: true).isEmpty)
        XCTAssertTrue(try bridgeFeedback(gapMs: 100, delayMs: 100, nested: true).isEmpty)
        XCTAssertTrue(try bridgeFeedback(gapMs: 100, delayMs: 100, newSession: true).isEmpty)
    }

    func testLongPairedExclusionCountsNoUnknownIntervalAsDwell() throws {
        let a = target("a"), b = target("b")
        let id = UUID().uuidString
        let before = arrival(a, 100, 4)
        let marker: [String: JSONValue] =
            ["scopeVersion": .string("excluded-foreground-v1"),
             "scopeID": .string(id), "category": .string("excluded_page")]
        let begin = event(.observationScope, 200, 5, marker.merging(["phase": .string("begin")]) { _, new in new })
        let end = event(.observationScope, 600_200, 6, marker.merging(["phase": .string("end")]) { _, new in new })
        let after = arrival(b, 600_300, 7)
        var learner = OnlinePathFeedbackLearner()
        let results = try [start(), update(a, 1, 2), update(b, 2, 3), before, begin, end, after]
            .flatMap { learner.consume($0) }
        XCTAssertEqual(results, [OnlinePathFeedback(inputEventID: before.id,
                                                    outcomeEventID: after.id,
                                                    evidence: .excludedScope)])
    }

    func testOutcomeDeduplicationRetainsOnlyRecentIDs() throws {
        let a = target("a"), b = target("b")
        var learner = OnlinePathFeedbackLearner()
        for index in 0..<8_193 {
            let id = "session-\(index)"
            let base = Int64(index) * 10_000
            var beginning = start()
            beginning.sessionID = id
            beginning.timestampMs = base
            beginning.monotonicMs = Double(base)
            _ = learner.consume(beginning)
            if index == 0 {
                var first = try update(a, base + 1, 2)
                first.sessionID = id
                var second = try update(b, base + 2, 3)
                second.sessionID = id
                _ = learner.consume(first)
                _ = learner.consume(second)
            }
            for (offset, destination) in [(4, a), (5, b)] {
                var context = arrival(destination, base + Int64(offset) * 100, offset)
                context.sessionID = id
                _ = learner.consume(context)
            }
            var gap = event(.captureGap, base + 600, 6,
                            ["reason": .string("unknown_failure")])
            gap.sessionID = id
            XCTAssertEqual(learner.consume(gap).count, 1)
        }
        XCTAssertEqual(learner.recentOutcomeCount, 8_192)
        XCTAssertEqual(learner.retainedContextCount, 0)
        XCTAssertTrue(learner.pendingInputEventIDs.isEmpty)
    }
}
