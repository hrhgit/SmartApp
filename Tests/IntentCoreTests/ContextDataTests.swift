import XCTest
@testable import IntentCore

final class ContextDataTests: XCTestCase {
    private func event(_ kind: EventKind, _ time: Double, sequence: Int, source: String = "test",
                       data: [String: JSONValue] = [:], session: String = "s") -> BehaviorEvent {
        var value = BehaviorEvent(id: "e\(sequence)", sessionID: session, timestampMs: Int64(100_000 + time),
                                  monotonicMs: time, kind: kind, source: source, data: data)
        value.schemaVersion = 2; value.sequence = sequence
        return value
    }
    func testPageIdentityPreservesReopenAddressAndSemanticPathsQueriesAndRoutes() throws {
        let identity = TargetIdentity(key: Data(repeating: 42, count: 32))
        let a = try XCTUnwrap(identity.page("https://user:password@example.com/home?secret=abc#route-a"))
        let b = try XCTUnwrap(identity.page("https://example.com/home?secret=abc#route-b"))
        XCTAssertNotEqual(a.id, b.id)
        XCTAssertEqual(a.locator, "https://example.com/home?secret=abc#route-a")
        XCTAssertEqual(a.origin, "https://example.com")
        XCTAssertEqual(a.id, identity.page("https://example.com:443/home?secret=abc#route-a")?.id)
        XCTAssertNotEqual(identity.page("http://localhost:3000")?.id, identity.page("http://localhost:8000")?.id)
        XCTAssertNotEqual(identity.page("https://example.com/home")?.id, identity.page("https://example.com/other")?.id)
        let json = String(decoding: try JSONEncoder().encode(a), as: UTF8.self)
        XCTAssertFalse(json.contains("password"))
        XCTAssertEqual(try JSONDecoder().decode(Destination.self, from: Data(json.utf8)).locator, a.locator)
        XCTAssertEqual(identity.page("https://example.com/a%2Fb?x=%2B&x=2#/%E4%B8%AD")?.locator,
                       "https://example.com/a%2Fb?x=%2B&x=2#/%E4%B8%AD")
        XCTAssertNil(identity.page("file:///Users/private/file.pdf"))
        XCTAssertNil(identity.page("chrome://settings"))
        XCTAssertNil(identity.page("https:///missing-host"))
    }
    func testIdentityKeySurvivesRestart() throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: root) }
        let url = root.appendingPathComponent("identity.key")
        let first = try TargetIdentity(keyURL: url), second = try TargetIdentity(keyURL: url)
        XCTAssertEqual(first.page("https://example.com/task")?.id, second.page("https://example.com/task")?.id)
        let mode = try FileManager.default.attributesOfItem(atPath: url.path)[.posixPermissions] as? NSNumber
        XCTAssertEqual(mode?.intValue, 0o600)
    }
    func testHistoryAndInventoryAreFrozenBeforeFutureMultiStepOutcomes() throws {
        var builder = NavigationSampleBuilder()
        _ = builder.consume(event(.inventoryChanged, 0, sequence: 1))
        _ = builder.consume(event(.contextChanged, 100, sequence: 2, data: ["targetID": .string("editor")]))
        _ = builder.consume(event(.inventoryChanged, 500, sequence: 3))
        _ = builder.consume(event(.contextChanged, 1_000, sequence: 4, data: ["targetID": .string("browser")]))
        _ = builder.consume(event(.contextChanged, 2_000, sequence: 5, data: ["targetID": .string("issue")]))
        let sample = try XCTUnwrap(builder.consume(event(.heartbeat, 30_101, sequence: 6)).first)
        XCTAssertEqual(sample.anchorEventID, "e2")
        XCTAssertEqual(sample.recentContextEventIDs, ["e2"])
        XCTAssertEqual(sample.inventoryEventIDs, ["e1"])
        XCTAssertEqual(sample.outcomes.map(\.targetID), ["browser", "issue"])
        XCTAssertEqual(sample.status, "observed_horizon")
        XCTAssertEqual(sample.inputThroughSequence, 2)
    }
    func testForegroundOnlySessionExportsNoInventoryInput() throws {
        var builder = NavigationSampleBuilder()
        _ = builder.consume(event(.sessionStart, 0, sequence: 1,
            data: ["stateCoverageVersion": .string(StateCoverage.foregroundOnlyVersion)]))
        _ = builder.consume(event(.sourceStatus, 1, sequence: 2, source: "browser-extension:com.google.Chrome:profile",
            data: ["coverageVersion": .string(StateCoverage.foregroundOnlyVersion), "observationStatus": .string("ready")]))
        _ = builder.consume(event(.contextChanged, 100, sequence: 3, data: ["targetID": .string("page:a")]))
        _ = builder.consume(event(.contextChanged, 1_000, sequence: 4, data: ["targetID": .string("page:b")]))
        let sample = try XCTUnwrap(builder.consume(event(.heartbeat, 30_100, sequence: 5)).first)
        XCTAssertEqual(sample.schemaVersion, 4)
        XCTAssertEqual(sample.labelRuleVersion, "observed-foreground-sequence-30s-v4")
        XCTAssertEqual(sample.inputCoverage?.version, StateCoverage.foregroundOnlyVersion)
        XCTAssertEqual(sample.inputCoverage?.status, "foreground_only")
        XCTAssertTrue(sample.inventoryEventIDs.isEmpty)
        XCTAssertEqual(sample.outcomeCoverage, "observed")
    }
    func testGapsAndExportCutoffsKeepUnknownResultsWithoutStayLabels() throws {
        var builder = NavigationSampleBuilder()
        _ = builder.consume(event(.contextChanged, 0, sequence: 1, data: ["targetID": .string("a")]))
        let result = builder.consume(event(.captureGap, 100, sequence: 2, data: ["reason": .string("lock")]))
        XCTAssertEqual(result.first?.status, "incomplete")
        XCTAssertEqual(result.first?.endReason, "lock")
        _ = builder.consume(event(.contextChanged, 200, sequence: 3, data: ["targetID": .string("b")]))
        let trailing = builder.finish()
        XCTAssertEqual(trailing.first?.recentContextEventIDs, ["e3"])
        XCTAssertEqual(trailing.first?.endReason, "export_cutoff")
        XCTAssertEqual(trailing.first?.outcomes, [])
    }
    func testInterruptionCausesAreIdentityFreeAndConsistent() {
        var gate = TargetEventGate()
        let id = UUID().uuidString
        let begin = event(.captureGap, 0, sequence: 1, source: "lifecycle", data: [
            "reason": .string("observation_suspended"),
            "interruptionVersion": .string("interruption-cause-v1"),
            "interruptionID": .string(id), "interruptionPhase": .string("begin"),
            "causeCategory": .string("foreground_unconfirmed")])
        XCTAssertTrue(gate.accept(begin))
        var end = begin
        end.data["reason"] = .string("observation_resumed")
        end.data["interruptionPhase"] = .string("end")
        XCTAssertTrue(gate.accept(end))
        var mismatched = end
        mismatched.data["interruptionPhase"] = .string("begin")
        XCTAssertFalse(gate.accept(mismatched))
        var identified = begin
        identified.data["targetID"] = .string("app:private")
        XCTAssertFalse(gate.accept(identified))
        var invalid = begin
        invalid.data["causeCategory"] = .string("unknown_private_app")
        XCTAssertFalse(gate.accept(invalid))
        var missingID = begin
        missingID.data.removeValue(forKey: "interruptionID")
        XCTAssertFalse(gate.accept(missingID))
        var system = begin
        system.data["causeCategory"] = .string("system_unavailable")
        system.data["causeSignals"] = .array([.string("screen_locked"), .string("system_sleep")])
        XCTAssertTrue(gate.accept(system))
        system.data["causeSignals"] = .array([.string("display_sleep")])
        XCTAssertFalse(gate.accept(system))
        system.data["causeSignals"] = .array([])
        XCTAssertFalse(gate.accept(system))
        XCTAssertTrue(gate.accept(event(.captureGap, 1, sequence: 2, source: "lifecycle", data: [
            "reason": .string("scheduler_or_clock_gap"), "causeCategory": .string("scheduler_or_clock_gap")])))
        XCTAssertFalse(gate.accept(event(.captureGap, 2, sequence: 3, source: "lifecycle", data: [
            "reason": .string("settings_changed"), "causeCategory": .string("scheduler_or_clock_gap")])))
        XCTAssertTrue(gate.accept(event(.captureGap, 3, sequence: 4, data: ["reason": .string("legacy_gap")])))
    }
    func testExcludedForegroundCensorsOutcomeButKeepsEarlierHistoryWithMask() throws {
        var builder = NavigationSampleBuilder()
        _ = builder.consume(event(.contextChanged, 0, sequence: 1, data: ["targetID": .string("a")]))
        let begin = event(.observationScope, 100, sequence: 2, source: "lifecycle", data: [
            "scopeVersion": .string("excluded-foreground-v1"), "scopeID": .string(UUID().uuidString),
            "phase": .string("begin"), "category": .string("excluded_app")])
        let incomplete = try XCTUnwrap(builder.consume(begin).first)
        XCTAssertEqual(incomplete.status, "incomplete")
        XCTAssertEqual(incomplete.endReason, "excluded_foreground")
        XCTAssertTrue(incomplete.outcomes.isEmpty)
        _ = builder.consume(event(.observationScope, 500, sequence: 3, source: "lifecycle", data: [
            "scopeVersion": .string("excluded-foreground-v1"), "scopeID": begin.data["scopeID"]!,
            "phase": .string("end"), "category": .string("excluded_app")]))
        _ = builder.consume(event(.contextChanged, 600, sequence: 4, data: ["targetID": .string("b")]))
        _ = builder.consume(event(.contextChanged, 700, sequence: 5, data: ["targetID": .string("c")]))
        let sample = try XCTUnwrap(builder.consume(event(.heartbeat, 30_600, sequence: 6)).first)
        XCTAssertEqual(sample.anchorEventID, "e4")
        XCTAssertEqual(sample.outcomes.map(\.eventID), ["e5"])
        XCTAssertEqual(sample.activityContextEventIDs, ["e1", "e4"])
        XCTAssertEqual(sample.excludedScopeEventIDs, ["e2", "e3"])
    }
    func testUnpairedExcludedScopeCannotCarryHistoryIntoNextArrival() throws {
        var builder = NavigationSampleBuilder()
        _ = builder.consume(event(.contextChanged, 0, sequence: 1, data: ["targetID": .string("a")]))
        _ = builder.consume(event(.observationScope, 100, sequence: 2, source: "lifecycle", data: [
            "scopeVersion": .string("excluded-foreground-v1"), "scopeID": .string(UUID().uuidString),
            "phase": .string("begin"), "category": .string("excluded_app")]))
        _ = builder.consume(event(.contextChanged, 200, sequence: 3, data: ["targetID": .string("b")]))
        let sample = try XCTUnwrap(builder.finish().first)
        XCTAssertEqual(sample.activityContextEventIDs, ["e3"])
        XCTAssertNil(sample.excludedScopeEventIDs)
    }
    func testLongerActivityHistorySurvivesShortLookbackAndRecommendationOriginIsPreserved() throws {
        var builder = NavigationSampleBuilder()
        _ = builder.consume(event(.contextChanged, 0, sequence: 1, data: ["targetID": .string("homework")]))
        _ = builder.consume(event(.heartbeat, 40_000, sequence: 2))
        var later = event(.contextChanged, 600_000, sequence: 3, data: ["targetID": .string("gpt")])
        later.decisionID = "d1"
        _ = builder.consume(later)
        let sample = try XCTUnwrap(builder.finish().first)
        XCTAssertEqual(sample.recentContextEventIDs, ["e3"])
        XCTAssertEqual(sample.activityContextEventIDs, ["e1", "e3"])
        XCTAssertEqual(sample.origin, "recommendation_related")
        XCTAssertEqual(sample.decisionID, "d1")
    }
    func testContainerResolutionDoesNotCreateDuplicateAnchor() {
        var builder = NavigationSampleBuilder()
        _ = builder.consume(event(.contextChanged, 0, sequence: 1, data: ["targetID": .string("browser")]))
        _ = builder.consume(event(.contextChanged, 100, sequence: 2,
            data: ["targetID": .string("page"), "reason": .string("resolution_update")]))
        let samples = builder.finish()
        XCTAssertEqual(samples.count, 1); XCTAssertEqual(samples.first?.outcomes, [])
        XCTAssertEqual(samples.first?.resolutionEventIDs, ["e2"])
    }
    func testShakeRequiresRepeatedReversalAndIgnoresDraggingAndCooldown() {
        var detector = ShakeDetector()
        for i in 0..<10 { XCTAssertFalse(detector.move(x: Double(i * 40), time: Double(i) * 0.04, dragging: false)) }
        detector.reset()
        let results = [0.0, 50, 0, 50, 0, 50, 0].enumerated().map {
            detector.move(x: $0.element, time: 2 + Double($0.offset) * 0.06, dragging: false)
        }
        XCTAssertEqual(results.filter { $0 }.count, 1)
        for i in 0..<10 { XCTAssertFalse(detector.move(x: i.isMultiple(of: 2) ? 0 : 80, time: 2.5 + Double(i) * 0.04, dragging: false)) }
        for i in 0..<10 { XCTAssertFalse(detector.move(x: i.isMultiple(of: 2) ? 0 : 80, time: 10 + Double(i) * 0.04, dragging: true)) }
    }
    func testV2StoreExportsReplayableEventsTargetsAndSequenceSamples() throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: root) }
        let store = try EventStore(url: root.appendingPathComponent("v2.sqlite"))
        let target = Destination(id: "app:test", kind: .app, name: "Test", bundleID: "test", admission: TargetAdmission(basis: "standalone_application"))
        let other = Destination(id: "app:other", kind: .app, name: "Other", bundleID: "other", admission: TargetAdmission(basis: "standalone_application"))
        try store.append([
            event(.sessionStart, 0, sequence: 1, data: ["targetEligibilityPolicyVersion": .string(TargetEligibilityPolicy.version)]),
            event(.targetUpdated, 1, sequence: 2, data: ["target": try .encoded(target)]),
            event(.targetUpdated, 1.5, sequence: 3, data: ["target": try .encoded(other)]),
            event(.contextChanged, 2, sequence: 4, data: ["targetID": .string(target.id)]),
            event(.contextChanged, 3_000, sequence: 5, data: ["targetID": .string(other.id)]),
            event(.heartbeat, 31_000, sequence: 6)
        ])
        XCTAssertEqual(try store.knownTargets(), [other, target])
        let directory = root.appendingPathComponent("export")
        let report = try store.export(to: directory)
        XCTAssertEqual(report.schemaVersion, 2)
        XCTAssertEqual(report.completeSamples, 1); XCTAssertEqual(report.partialSamples, 1)
        XCTAssertEqual(report.targetRecords, 2); XCTAssertEqual(report.staySamples, 0)
        XCTAssertEqual(report.targetEligibilityPolicyVersion, TargetEligibilityPolicy.version)
        XCTAssertTrue(FileManager.default.fileExists(atPath: directory.appendingPathComponent("navigation-samples.jsonl").path))
        XCTAssertFalse(FileManager.default.fileExists(atPath: directory.appendingPathComponent("samples.jsonl").path))
    }
}
