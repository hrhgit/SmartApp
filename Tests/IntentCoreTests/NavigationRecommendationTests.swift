import XCTest
@testable import IntentCore

final class NavigationRecommendationTests: XCTestCase {
    private func app(_ owner: String) -> Destination {
        Destination(id: "app:" + owner, kind: .app, name: owner, bundleID: owner,
                    admission: TargetAdmission(basis: "standalone_application"))
    }
    private func page(_ number: Int, suffix: String = "") -> Destination {
        let token = String(format: "%012d", number)
        return Destination(id: "page:\(number)\(suffix)", kind: .page, name: "Page", bundleID: "browser",
                           locator: "https://chatgpt.com/c/00000000-0000-0000-0000-\(token)" + suffix)
    }
    private func record(_ input: Destination, _ target: Destination, _ n: Int) -> SemanticLearningRecord {
        SemanticLearningRecord(origin: SemanticRecommendationModel.origin(input), annotation: SemanticMetadata.annotate(input),
            label: SemanticRecommendationModel.action(target)!, outcomeID: "arrival:\(n)", outcomeTimestamp: Double(n),
            destination: SemanticMetadata.annotate(target).identity)
    }
    func testSelfReturnsRemainAuditableWithoutTrainingNavigation() {
        let a = app("a"), b = app("b")
        var model = NavigationRecommendationModel(records: [record(a, a, 1), record(a, a, 2)])
        XCTAssertEqual(model.records.count, 2)
        XCTAssertTrue(model.records.allSatisfy { $0.navigationDisposition == .currentDestination })
        XCTAssertTrue(model.ranked(current: a).isEmpty)
        model.learn(record(a, b, 3))
        XCTAssertEqual(model.ranked(current: a), [b.id])
    }
    func testSavedParametersRestoreAndContinueLearningWithoutReplay() throws {
        let a = app("a"), b = app("b"), c = app("c")
        let history = [record(a, b, 1), record(a, c, 2), record(a, b, 3)]
        let trained = NavigationRecommendationModel(records: history)
        let saved = try trained.checkpoint()
        var restored = try XCTUnwrap(NavigationRecommendationModel.restored(records: trained.records, checkpoint: saved))
        // Score calculations reduce dictionary values; reconstruction can
        // change addition order without changing weights or recommendation.
        func assertSameScores(_ left: NavigationRecommendationModel, _ right: NavigationRecommendationModel) throws {
            let actual = left.scores(current: a), expected = right.scores(current: a)
            XCTAssertEqual(Set(actual.keys), Set(expected.keys))
            for (key, value) in expected {
                XCTAssertEqual(try XCTUnwrap(actual[key]), value, accuracy: 1e-12)
            }
            XCTAssertEqual(left.ranked(current: a), right.ranked(current: a))
        }
        try assertSameScores(restored, trained)
        restored.learn(history[0])
        XCTAssertEqual(restored.records.count, history.count)
        let next = record(a, c, 4)
        restored.learn(next)
        var uninterrupted = trained
        uninterrupted.learn(next)
        try assertSameScores(restored, uninterrupted)
        XCTAssertEqual(restored.records, uninterrupted.records)

        var mismatched = saved
        mismatched.recordsSHA256 = "wrong"
        XCTAssertNil(NavigationRecommendationModel.restored(records: trained.records, checkpoint: mismatched))
    }
    func testActiveApplicationCannotReturnThroughGlobalPrior() {
        let a = app("a"), b = app("b"), c = app("c")
        var records = (0..<10).map { record(b, a, $0) }
        records.append(record(a, c, 11))
        let window = Destination(id: "window:a", kind: .window, name: "Window", bundleID: "a")
        let model = NavigationRecommendationModel(records: records)
        XCTAssertFalse(model.ranked(current: window).contains(a.id))
        XCTAssertTrue(model.ranked(current: window).contains(c.id))
    }
    func testSameApplicationResourceRemainsARealNavigation() {
        let a = page(1), b = page(2)
        let model = NavigationRecommendationModel(records: (0..<4).map { record(a, b, $0) })
        XCTAssertEqual(model.ranked(current: a), [b.id])
        XCTAssertTrue(NavigationSemantics.changesDestination(b, current: a))
        XCTAssertFalse(NavigationSemantics.changesDestination(app("browser"), current: a))
    }
    func testDiverseConversationsPoolOneServiceAndDefaultToEntry() {
        let a = app("a")
        let records = (0..<8).map { record(a, page(1), $0) } + (8..<16).map { record(a, page($0), $0) }
        let model = NavigationRecommendationModel(records: records)
        XCTAssertEqual(model.ranked(current: a), [NavigationSemantics.chatService])
        XCTAssertEqual(model.records.map(\.label), records.map(\.label))
        XCTAssertEqual(model.records.map(\.destination), records.map(\.destination))
    }
    func testSpecificityNeedsIndependentExactContextEvidence() {
        let a = app("a"), b = app("b"), child = page(1)
        let first = record(a, child, 1)
        let duplicate = NavigationRecommendationModel(records: Array(repeating: first, count: 20))
        XCTAssertEqual(duplicate.ranked(current: a), [NavigationSemantics.chatService])
        let strong = NavigationRecommendationModel(records: (0..<4).map { record(a, child, $0) })
        XCTAssertEqual(strong.ranked(current: a), [child.id])
        XCTAssertEqual(strong.ranked(current: b), [NavigationSemantics.chatService])
    }
    func testSameServiceCurrentHomeIsNotRecommendedAgain() {
        let a = app("a")
        let home = Destination(id: "home", kind: .page, name: "Home", bundleID: "browser", locator: NavigationSemantics.chatEntry)
        let model = NavigationRecommendationModel(records: (0..<8).map { record(a, page($0), $0) })
        XCTAssertTrue(model.ranked(current: home).isEmpty)
    }
    func testScopesAndUnknownWebsitesDoNotCollapse() {
        for locator in ["https://other.example/c/00000000-0000-0000-0000-000000000001",
                        "https://chatgpt.com/g/project/c/00000000-0000-0000-0000-000000000001",
                        "https://chatgpt.com/c/00000000-0000-0000-0000-000000000001?workspace=secret",
                        "https://chatgpt.com/c/00000000-0000-0000-0000-000000000001#/project/one",
                        "https://chatgpt.com:8443/", "https://chatgpt.com/auth/login"] {
            let target = Destination(id: "target", kind: .page, name: "Page", locator: locator)
            XCTAssertNil(NavigationSemantics.service(SemanticMetadata.annotate(target).identity))
        }
        let scoped = page(1, suffix: "?messageId=message#reference")
        let annotation = SemanticMetadata.annotate(scoped)
        XCTAssertEqual(NavigationSemantics.service(annotation.identity), NavigationSemantics.chatService)
        XCTAssertEqual(annotation.identity.locator, scoped.locator)
    }
    func testServiceResolutionUsesKnownEntryWithoutInventingArrival() {
        let home = Destination(id: "home", kind: .page, name: "Home", locator: NavigationSemantics.chatEntry)
        let catalog = [home.id: home]
        XCTAssertEqual(NavigationSemantics.resolve(NavigationSemantics.chatService, catalog: catalog), home)
        XCTAssertNil(NavigationSemantics.resolve(NavigationSemantics.chatService, catalog: [:]))
    }
    func testShortDetourReturnIsLabeledAndDoesNotSkipToAnotherFutureDestination() throws {
        var learner = StreamingSemanticLearner()
        let a = app("a"), b = app("b"), c = app("c")
        func event(_ kind: EventKind, _ time: Double, _ data: [String: JSONValue] = [:]) -> BehaviorEvent {
            BehaviorEvent(id: "\(kind.rawValue):\(time)", sessionID: "s", timestampMs: Int64(time), monotonicMs: time,
                          kind: kind, source: "fixture", data: data)
        }
        _ = learner.consume(event(.sessionStart, 0, ["targetEligibilityPolicyVersion": .string(TargetEligibilityPolicy.version)]))
        for (index, target) in [a,b,c].enumerated() {
            _ = learner.consume(event(.targetUpdated, Double(index + 1), ["target": try .encoded(target)]))
        }
        _ = learner.consume(event(.contextChanged, 1_000, ["targetID": .string(a.id)]))
        _ = learner.consume(event(.contextChanged, 2_000, ["targetID": .string(b.id)]))
        _ = learner.consume(event(.contextChanged, 3_000, ["targetID": .string(a.id)]))
        _ = learner.consume(event(.contextChanged, 12_000, ["targetID": .string(c.id)]))
        let records = learner.consume(event(.heartbeat, 31_000))
        XCTAssertEqual(records.count, 1)
        XCTAssertEqual(records.first?.label, a.id)
        XCTAssertEqual(records.first?.observedStep, 2)
        XCTAssertEqual(records.first?.navigationDisposition, .currentDestination)
        XCTAssertTrue(NavigationRecommendationModel(records: records).ranked(current: a).isEmpty)
    }
}
