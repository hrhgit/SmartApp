import XCTest
@testable import IntentCore

final class TargetEligibilityTests: XCTestCase {
    private func app(_ bundle: String = "test.editor", admission: Bool = true) -> Destination {
        Destination(id: "app:" + bundle, kind: .app, name: "Synthetic Helper", bundleID: bundle,
                    admission: admission ? TargetAdmission(basis: "standalone_application") : nil)
    }
    private func event(_ kind: EventKind, _ n: Int, data: [String: JSONValue] = [:], app: AppContext? = nil) -> BehaviorEvent {
        var e = BehaviorEvent(id: "e\(n)", sessionID: "s", timestampMs: Int64(n), monotonicMs: Double(n), kind: kind, source: "test", app: app, data: data)
        e.schemaVersion = 2; e.sequence = n; return e
    }
    func testApplicationEvidenceAndHardExclusionPrecedence() {
        for bundle in ["test.editor", "com.apple.finder", "test.RealHelper"] {
            XCTAssertTrue(TargetEligibilityPolicy.application(.init(bundleID: bundle, standaloneApplication: true, activation: .regular)).isAllowed)
        }
        for bundle in ["com.apple.SecurityAgent", "com.apple.SecurityAgentHelper", "com.apple.SecurityAgentHelper.arm64", "com.apple.SecurityAgentHelper.future"] {
            let facts = ApplicationEvidence(bundleID: bundle, standaloneApplication: true, activation: .regular, independentWindow: true)
            XCTAssertEqual(TargetEligibilityPolicy.application(facts).disposition, .excluded)
            XCTAssertFalse(TargetEligibilityPolicy.validates(app(bundle)))
        }
        XCTAssertFalse(CapturePolicy.isExcluded("com.apple.SecurityAgentHelperUnrelated", rules: []))
        XCTAssertEqual(TargetEligibilityPolicy.application(.init(bundleID: "test.editor", standaloneApplication: true, activation: .regular), rules: ["test.*"]).disposition, .excluded)
        XCTAssertFalse(TargetEligibilityPolicy.application(.init(bundleID: "test.worker", standaloneApplication: true, backgroundOnly: true, activation: .regular)).isAllowed)
        XCTAssertFalse(TargetEligibilityPolicy.application(.init(bundleID: "test.plugin", standaloneApplication: false, activation: .regular)).isAllowed)
        XCTAssertFalse(TargetEligibilityPolicy.application(.init(bundleID: "test.worker", standaloneApplication: true, activation: .prohibited)).isAllowed)
        XCTAssertEqual(TargetEligibilityPolicy.application(.init(bundleID: nil, standaloneApplication: true, activation: .regular)).disposition, .unknown)
    }
    func testAccessoryRequiresIndependentWindowIncludingUIElementWithRegularPolicy() {
        for activation: ApplicationEvidence.Activation in [.regular, .accessory] {
            var facts = ApplicationEvidence(bundleID: "test.utility", standaloneApplication: true, uiElement: true, activation: activation)
            XCTAssertEqual(TargetEligibilityPolicy.application(facts).disposition, .unknown)
            facts.independentWindow = true
            XCTAssertEqual(TargetEligibilityPolicy.application(facts).reason, "independent_work_window")
        }
    }
    func testWindowEvidenceDoesNotPromoteDialogsUnknownRolesOrUnapprovedOwners() {
        let owner = AdmissionDecision(.allowed, "standalone_application")
        let standard = WindowEvidence(role: "AXWindow", subrole: "AXStandardWindow", modal: false, canRaise: true, main: true)
        XCTAssertTrue(TargetEligibilityPolicy.window(standard, owner: owner).isAllowed)
        XCTAssertFalse(TargetEligibilityPolicy.window(standard, owner: .init(.excluded, "excluded_source")).isAllowed)
        for subrole in ["AXDialog", "AXSystemDialog", "AXFloatingWindow", nil] {
            XCTAssertFalse(TargetEligibilityPolicy.window(.init(role: "AXWindow", subrole: subrole, modal: false, canRaise: true), owner: owner).isAllowed)
        }
        for modal: Bool? in [true, nil] {
            XCTAssertFalse(TargetEligibilityPolicy.window(.init(role: "AXWindow", subrole: "AXStandardWindow", modal: modal, canRaise: true), owner: owner).isAllowed)
        }
        XCTAssertFalse(TargetEligibilityPolicy.window(.init(role: "AXWindow", subrole: "AXStandardWindow", modal: false, canRaise: false), owner: owner).isAllowed)
    }
    func testPagesAndLocalObjectsNeedExplicitSourceEvidence() {
        XCTAssertTrue(TargetEligibilityPolicy.page(origin: "https://example.com", supportedSource: true, excluded: false).isAllowed)
        XCTAssertFalse(TargetEligibilityPolicy.page(origin: "chrome://settings", supportedSource: true, excluded: false).isAllowed)
        XCTAssertFalse(TargetEligibilityPolicy.page(origin: "https://example.com", supportedSource: false, excluded: false).isAllowed)
        XCTAssertFalse(TargetEligibilityPolicy.page(origin: "https://example.com", supportedSource: true, excluded: true).isAllowed)
        XCTAssertTrue(TargetEligibilityPolicy.local(registered: true, validFileURL: true).isAllowed)
        XCTAssertFalse(TargetEligibilityPolicy.local(registered: false, validFileURL: true).isAllowed)
        XCTAssertFalse(TargetEligibilityPolicy.local(registered: true, validFileURL: false).isAllowed)
        XCTAssertTrue(TargetEligibilityPolicy.finderDirectory(validFileURL: true, isDirectory: true,
            owner: .init(.allowed, "standalone_application")).isAllowed)
        XCTAssertFalse(TargetEligibilityPolicy.finderDirectory(validFileURL: true, isDirectory: false,
            owner: .init(.allowed, "standalone_application")).isAllowed)
        XCTAssertFalse(TargetEligibilityPolicy.finderDirectory(validFileURL: true, isDirectory: true,
            owner: .init(.excluded, "excluded_source")).isAllowed)
        XCTAssertFalse(TargetEligibilityPolicy.validates(app(admission: false)))
    }
    func testObservedQSpaceDirectoriesRequireItsCurrentPolicyAndBundle() {
        var qspace = Destination(id: "directory:qspace", kind: .directory, name: "QSpace folder",
            bundleID: TargetEligibilityPolicy.qspaceBundleID, locator: "file:///tmp/qspace-folder/",
            admission: TargetAdmission(basis: "observed_qspace_directory"))
        XCTAssertTrue(TargetEligibilityPolicy.qspaceDirectory(validFileURL: true, isDirectory: true,
            owner: .init(.allowed, "standalone_application")).isAllowed)
        XCTAssertTrue(TargetEligibilityPolicy.validates(qspace))

        qspace.admission?.policyVersion = "target-eligibility-v2"
        XCTAssertFalse(TargetEligibilityPolicy.validates(qspace))
        qspace.admission?.policyVersion = TargetEligibilityPolicy.version
        qspace.bundleID = TargetEligibilityPolicy.finderBundleID
        XCTAssertFalse(TargetEligibilityPolicy.validates(qspace))

        let legacyFinder = Destination(id: "directory:finder", kind: .directory, name: "Finder folder",
            bundleID: TargetEligibilityPolicy.finderBundleID, locator: "file:///tmp/finder-folder/",
            admission: TargetAdmission(basis: "observed_finder_directory", policyVersion: "target-eligibility-v2"))
        XCTAssertTrue(TargetEligibilityPolicy.validates(legacyFinder))
        let qspaceWindow = Destination(id: "qspace-window:s:1:w", kind: .window, name: "QSpace window",
            bundleID: TargetEligibilityPolicy.qspaceBundleID,
            admission: TargetAdmission(basis: "supported_qspace_window"))
        XCTAssertTrue(TargetEligibilityPolicy.validates(qspaceWindow))
    }
    func testGenerationRejectsCallbacksEvenAfterObservationResumes() {
        var generation = ObservationGeneration()
        let beforePause = generation.token
        XCTAssertTrue(generation.accepts(beforePause))
        generation.invalidate(); let paused = generation.token
        generation.invalidate()
        XCTAssertFalse(generation.accepts(beforePause)); XCTAssertFalse(generation.accepts(paused))
        XCTAssertTrue(generation.accepts(generation.token))
    }
    func testReplayKeepsVersionOfEachSupportedSession() throws {
        var replay = EligibleNavigationReplay()
        _ = replay.consume(event(.sessionStart, 0, data: ["targetEligibilityPolicyVersion": .string("target-eligibility-v1")]))
        var old = app(); old.admission = TargetAdmission(basis: "standalone_application", policyVersion: "target-eligibility-v1")
        _ = replay.consume(event(.targetUpdated, 1, data: ["target": try .encoded(old)]))
        _ = replay.consume(event(.contextChanged, 2, data: ["targetID": .string(old.id)]))
        var start = event(.sessionStart, 3, data: ["targetEligibilityPolicyVersion": .string(TargetEligibilityPolicy.version)])
        start.sessionID = "next"
        let previous = replay.consume(start)
        XCTAssertEqual(previous.first?.targetEligibilityPolicyVersion, "target-eligibility-v1")
        var directory = Destination(id: "directory:test", kind: .directory, name: "Test", bundleID: "com.apple.finder",
            locator: "file:///tmp/test/", admission: TargetAdmission(basis: "observed_finder_directory"))
        XCTAssertTrue(TargetEligibilityPolicy.validates(directory))
        directory.admission?.policyVersion = "target-eligibility-v1"
        XCTAssertFalse(TargetEligibilityPolicy.validates(directory))
    }
    func testFinalBoundaryRejectsIdentityAndReferencesAcrossEventShapes() throws {
        var gate = TargetEventGate()
        let helper = app("com.apple.SecurityAgentHelper.arm64")
        let update = event(.targetUpdated, 1, data: ["target": try .encoded(helper)])
        XCTAssertFalse(gate.accept(update))
        XCTAssertFalse(gate.accept(event(.appActivated, 2, app: .init(bundleID: helper.bundleID!, name: helper.name))))
        XCTAssertFalse(gate.accept(event(.contextChanged, 3, data: ["targetID": .string(helper.id)])))
        let entrance = TargetEntrance(id: "helper-window", targetID: helper.id, bundleID: helper.bundleID!, kind: "running_app")
        XCTAssertFalse(gate.accept(event(.inventoryChanged, 4, data: ["entrances": try .encoded([entrance])])) )
        XCTAssertFalse(gate.accept(event(.recommendationShown, 5, data: ["candidateTargetIDs": .array([.string(helper.id)])])))
        XCTAssertFalse(gate.accept(event(.recommendationFeedback, 6, data: ["targetID": .string(helper.id)])))
        XCTAssertTrue(gate.targets.isEmpty)
        let good = app()
        XCTAssertTrue(gate.accept(event(.targetUpdated, 7, data: ["target": try .encoded(good)])))
        XCTAssertTrue(gate.accept(event(.contextChanged, 8, data: ["targetID": .string(good.id)])))
    }
    func testReplaySkipsLegacySessionsAndBreaksAcrossRejectedObjects() throws {
        var replay = EligibleNavigationReplay()
        XCTAssertTrue(replay.consume(event(.sessionStart, 0)).isEmpty)
        _ = replay.consume(event(.contextChanged, 1, data: ["targetID": .string("old-helper")]))
        XCTAssertTrue(replay.finish().isEmpty)
        _ = replay.consume(event(.sessionStart, 2, data: ["targetEligibilityPolicyVersion": .string(TargetEligibilityPolicy.version)]))
        let good = app()
        _ = replay.consume(event(.targetUpdated, 3, data: ["target": try .encoded(good)]))
        _ = replay.consume(event(.contextChanged, 4, data: ["targetID": .string(good.id)]))
        let ended = replay.consume(event(.contextChanged, 5, data: ["targetID": .string("old-helper")]))
        XCTAssertEqual(ended.first?.endReason, "target_eligibility_boundary")
        _ = replay.consume(event(.contextChanged, 6, data: ["targetID": .string(good.id)]))
        let resumed = try XCTUnwrap(replay.finish().first)
        XCTAssertEqual(resumed.recentContextEventIDs, ["e6"])
        XCTAssertEqual(resumed.targetEligibilityPolicyVersion, TargetEligibilityPolicy.version)
        XCTAssertFalse(String(decoding: try JSONEncoder().encode(ended + [resumed]), as: UTF8.self).contains("old-helper"))
    }
}
