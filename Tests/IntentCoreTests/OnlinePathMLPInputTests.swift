import XCTest
import CryptoKit
@testable import IntentCore

final class OnlinePathMLPInputTests: XCTestCase {
    private func page(_ id: String, _ locator: String) -> Destination {
        Destination(id: id, kind: .page, name: locator, bundleID: "browser",
                    origin: "https://example.test", locator: locator,
                    admission: TargetAdmission(basis: "supported_browser_page"))
    }

    private func app(_ bundle: String) -> Destination {
        Destination(id: "app:" + bundle, kind: .app, name: bundle, bundleID: bundle,
                    admission: TargetAdmission(basis: "standalone_application"))
    }

    private func event(_ kind: EventKind, _ number: Int, target: String? = nil,
                       metadata: Destination? = nil, reason: String = "foreground_change",
                       session: String = "s", sourceStatus: String = "ready",
                       monotonicMs: Double? = nil, observedApp: String? = nil) throws -> BehaviorEvent {
        var data: [String: JSONValue] = [:]
        if let metadata { data["target"] = try .encoded(metadata) }
        if let target {
            data["targetID"] = .string(target)
            data["reason"] = .string(reason)
            data["inputCoverage"] = .object([
                "version": .string("foreground-coverage-v2"),
                "status": .string("foreground_only"),
                "sources": .array([.object(["source": .string("foreground:test"),
                                              "status": .string(sourceStatus)])])
            ])
        }
        return BehaviorEvent(id: "e\(number)", sessionID: session,
                             timestampMs: Int64(number * 1_000),
                             monotonicMs: monotonicMs ?? Double(number * 1_000), kind: kind,
                             source: "foreground:test",
                             app: observedApp.map { AppContext(bundleID: $0, name: $0) }, data: data)
    }

    func testPythonFixtureCandidateOrderLayersCountsAndNoFuturePromotion() throws {
        let other = app("other")
        let old = page("page:old", "https://example.test/topic/old")
        let future = page("page:future", "https://example.test/topic/new")
        var history = OnlinePathInputHistory()
        for item in [try event(.targetUpdated, 0, metadata: other),
                     try event(.targetUpdated, 1, metadata: old),
                     try event(.contextChanged, 2, target: old.id),
                     try event(.contextChanged, 3, target: other.id),
                     try event(.targetUpdated, 4, metadata: future)] {
            history.consume(item)
        }
        let candidates = try history.candidates(current: other)
        XCTAssertEqual(candidates.map(\.actionID), [
            "derived-parent:page:07a75184ff7cefdfa6a738d0",
            "derived-parent:page:66a9c86ed3464f39de64c551",
            old.id,
        ])
        XCTAssertNil(candidates[0].recordedTarget)
        XCTAssertEqual(candidates[0].executableLocator, "https://example.test/")
        XCTAssertEqual(candidates[1].executableLocator, "https://example.test/topic/")
        XCTAssertEqual(candidates[0].row.layerKeys, [
            "layer:page:15d3cd8e56a7e344d5b0bf87b6df18339561ebf0b6dfe9a0536f43d2ed61cd5a"
        ])
        XCTAssertEqual(candidates[2].row.layerKeys, [
            "layer:page:31e9371fc680a07a370fdd442f605a73b29da502525ea471c3fddc3ef3a8b58a",
            "layer:page:15d3cd8e56a7e344d5b0bf87b6df18339561ebf0b6dfe9a0536f43d2ed61cd5a",
            "layer:page:71fde4f91ad0a407f2d9eb1bb822c05ed7e80a3b00e02848d02919e49c8ec7bb",
        ])
        XCTAssertEqual(candidates[0].row.numeric["candidate:path_count_all_log"]!, log(2), accuracy: 1e-6)
        XCTAssertEqual(candidates[0].row.numeric["candidate:direct_count_all_log"]!, 0)
        XCTAssertEqual(candidates[2].row.numeric["candidate:direct_count_all_log"]!, log(2), accuracy: 1e-6)
        XCTAssertEqual(candidates[2].row.numeric["context:last_confirmed_dwell_log_s"]!, log(2), accuracy: 1e-6)
        XCTAssertEqual(candidates[2].row.numeric.count, 31)
        XCTAssertEqual(candidates[2].row.categorical.count, 14)
        XCTAssertEqual(history.confirmedDestination(eventID: "e3")?.id, other.id)
        let weights = OnlinePathMLPWeights(
            format: "online-path-mlp-f32-v1", modelVersion: OnlinePathMLPScorer.modelVersion,
            featureSchema: OnlinePathMLPScorer.featureSchema,
            numericFields: OnlinePathMLPScorer.numericFields,
            categoricalFields: OnlinePathMLPScorer.categoricalFields,
            categoricalBuckets: 512, tokenHashPerson: "OffRankV1", hashPerson: "SharedMLPPath1",
            hidden: 16, layer1Weights: Array(repeating: Array(repeating: 0, count: 543), count: 16),
            layer1Bias: Array(repeating: 0, count: 16), layer2Weights: Array(repeating: 0, count: 16),
            layer2Bias: 0, identityKeySHA256: "fixture", trainedThroughEventID: nil)
        let scorer = try OnlinePathMLPScorer(weights: weights, identityKeySHA256: "fixture")
        let pythonDigests = [
            "0c3791d8c864d02d810eaf0208062dd3357ba8fb43c4662207bb6668d3adc335",
            "34dba2377f7f9d0b34d503a2adeb5835ff07cf9f3eaf3ebb5eaafeaaea478bb9",
            "1559bae8e937b79f90918eb91a38ae9ae5959faf2471a418c438d33433adf9c4",
        ]
        for (candidate, expected) in zip(candidates, pythonDigests) {
            let vector = try scorer.featureVector(for: candidate.row)
            var bytes = Data()
            for number in vector {
                var bits = number.bitPattern.littleEndian
                withUnsafeBytes(of: &bits) { bytes.append(contentsOf: $0) }
            }
            let digest = SHA256.hash(data: bytes).map { String(format: "%02x", $0) }.joined()
            XCTAssertEqual(digest, expected, candidate.actionID)
        }
    }

    func testCurrentTargetAndUnknownObservationFailClosed() throws {
        var history = OnlinePathInputHistory()
        let old = page("page:old", "https://example.test/topic/old")
        history.consume(try event(.targetUpdated, 1, metadata: old))
        XCTAssertThrowsError(try history.candidates(current: old))
        history.consume(try event(.contextChanged, 2, target: old.id))
        XCTAssertThrowsError(try history.candidates(current: app("other")))
        history.consume(try event(.observationScope, 3))
        XCTAssertThrowsError(try history.candidates(current: old))
    }

    func testCachedPathsRefreshAfterConfirmedMetadataChangeAndKeepEarlierCounts() throws {
        let current = app("other")
        var known = page("page:reused", "https://example.test/old/item")
        var history = OnlinePathInputHistory()
        for item in [try event(.targetUpdated, 0, metadata: current),
                     try event(.targetUpdated, 1, metadata: known),
                     try event(.contextChanged, 2, target: known.id),
                     try event(.contextChanged, 3, target: current.id)] { history.consume(item) }
        let original = try XCTUnwrap(history.candidates(current: current).first { $0.actionID == known.id })
        known.locator = "https://example.test/new/item"
        history.consume(try event(.targetUpdated, 4, metadata: known))
        // A metadata update alone is not a newly confirmed foreground action.
        XCTAssertEqual(try history.candidates(current: current).first { $0.actionID == known.id }?.semanticKey,
                       original.semanticKey)
        history.consume(try event(.contextChanged, 5, target: known.id))
        history.consume(try event(.contextChanged, 6, target: current.id))
        let updated = try XCTUnwrap(history.candidates(current: current).first { $0.actionID == known.id })
        XCTAssertNotEqual(updated.semanticKey, original.semanticKey)
        XCTAssertEqual(updated.executableLocator, known.locator)
        XCTAssertEqual(updated.row.numeric["candidate:direct_count_all_log"], log(2))
        XCTAssertEqual(original.row.numeric["candidate:direct_count_all_log"], log(2))
        XCTAssertFalse(updated.row.layerKeys.contains(original.semanticKey))
    }

    func testCurrentArrivalIsExcludedAndBackwardWallTimeKeepsExactFrequencies() throws {
        let a = page("page:a", "https://example.test/a"), b = app("other")
        var history = OnlinePathInputHistory()
        history.consume(try event(.targetUpdated, 0, metadata: a))
        history.consume(try event(.targetUpdated, 1, metadata: b))
        var first = try event(.contextChanged, 2, target: a.id)
        first.timestampMs = 1_000_000
        history.consume(first)
        var second = try event(.contextChanged, 3, target: b.id)
        second.timestampMs = 999_000
        history.consume(second)
        let rolledBack = try XCTUnwrap(history.candidates(current: b).first { $0.actionID == a.id })
        XCTAssertEqual(rolledBack.row.numeric["candidate:direct_count_all_log"], 0)
        var third = try event(.contextChanged, 4, target: a.id)
        third.timestampMs = 1_001_000
        history.consume(third)
        // Both page arrivals support the derived homepage, but this new
        // arrival must be excluded at its own prediction cutoff.
        let home = try XCTUnwrap(history.candidates(current: a).first { $0.executableLocator == "https://example.test/" })
        XCTAssertEqual(home.row.numeric["candidate:path_count_all_log"], log(2))
        var fourth = try event(.contextChanged, 5, target: b.id)
        fourth.timestampMs = 1_002_000
        history.consume(fourth)
        let caughtUp = try XCTUnwrap(history.candidates(current: b).first { $0.actionID == a.id })
        XCTAssertEqual(caughtUp.row.numeric["candidate:direct_count_all_log"], log(3))
    }

    func testOneFutureArrivalCanUseAlreadyKnownParentWithoutDuplicateVisits() throws {
        let current = app("other")
        let old = page("page:old", "https://example.test/topic/old")
        let future = page("page:future", "https://example.test/topic/new")
        var history = OnlinePathInputHistory()
        for item in [try event(.targetUpdated, 0, metadata: current),
                     try event(.targetUpdated, 1, metadata: old),
                     try event(.contextChanged, 2, target: old.id),
                     try event(.contextChanged, 3, target: current.id)] { history.consume(item) }
        let frozen = try history.candidates(current: current)
        let useful = OnlinePathInputHistory.usefulActionIndices(for: future, current: current,
                                                                 candidates: frozen)
        XCTAssertEqual(useful.map { frozen[$0].actionID },
                       frozen.prefix(2).map(\.actionID))
        let training = OnlinePathInputHistory.trainingRows(for: future, current: current,
                                                            frozenCandidates: frozen)
        XCTAssertTrue(training.appendedExact)
        XCTAssertEqual(training.originalCandidateCount, frozen.count)
        XCTAssertEqual(training.rows.count, frozen.count + 1)
        XCTAssertEqual(training.positiveIndices, useful + [frozen.count])
        XCTAssertEqual(training.rows.last?.categorical["candidate_source"], "recorded_action")
        XCTAssertEqual(training.rows.last?.numeric["candidate:path_count_all_log"], 0)
    }

    func testFileDirectoryParentsMatchPythonLexicalHierarchy() throws {
        let finder = Destination(id: "directory:test", kind: .directory, name: "A B",
                                 bundleID: TargetEligibilityPolicy.finderBundleID,
                                 locator: "file:///Users/me/Projects/A%20B",
                                 admission: TargetAdmission(basis: "observed_finder_directory"))
        let other = app("other")
        var history = OnlinePathInputHistory()
        for item in [try event(.targetUpdated, 0, metadata: other),
                     try event(.targetUpdated, 1, metadata: finder),
                     try event(.contextChanged, 2, target: finder.id),
                     try event(.contextChanged, 3, target: other.id)] { history.consume(item) }
        let candidates = try history.candidates(current: other)
        XCTAssertEqual(candidates.map(\.actionID), [
            "derived-parent:directory:154bc5a12c7584c4d3140bb6",
            "derived-parent:directory:56516bc6ac9dfa168ba6d494",
            "derived-parent:directory:77a801f915e2e07098f00be8",
            "derived-parent:directory:f4e665b9029679728be3ed48",
            "directory:test",
        ])
        XCTAssertEqual(candidates.last?.row.layerKeys, [
            "layer:directory:08b515e759bad81ac441be2f03faeb08d6f0b7c270dbdbe0741d1a3c3ef1177f",
            "layer:directory:30d2ec2269081e189755a521e64b23500bc767c69ad35e95de021983a7947131",
            "layer:directory:b86490e0dda9f8f25ef8a85c8effb634113ff34d960f7ef6a1327bab9111a4da",
            "layer:directory:665d11e6286d99cdd8abfda329d3b1b92bdae65ade51486856889b2c2f899a54",
            "layer:directory:99aa7f2dbb4720f8ca20573449b10acdbf680216e7125d7db3a1b38a6ddd907a",
        ])
    }

    func testExplicitFileParentAndDifferentOwnerAppsFollowPythonLayerOrder() throws {
        let documentApp = app("com.test.documents")
        let finderApp = app(TargetEligibilityPolicy.finderBundleID)
        let parent = Destination(id: "directory:parent", kind: .directory, name: "Projects",
                                 bundleID: TargetEligibilityPolicy.finderBundleID,
                                 locator: "file:///Users/me/Projects",
                                 admission: TargetAdmission(basis: "observed_finder_directory"))
        let child = Destination(id: "file:child", kind: .file, name: "child.txt",
                                bundleID: "com.test.documents",
                                locator: "file:///Users/me/Projects/child.txt", parentID: parent.id,
                                admission: TargetAdmission(basis: "observed_window_document"))
        let current = app("other")
        var history = OnlinePathInputHistory()
        for (index, item) in [documentApp, finderApp, parent, child, current].enumerated() {
            history.consume(try event(.targetUpdated, 2 * index, metadata: item))
            history.consume(try event(.contextChanged, 2 * index + 1, target: item.id))
        }
        let candidate = try XCTUnwrap(history.candidates(current: current).first {
            $0.actionID == child.id
        })
        XCTAssertEqual(candidate.row.layerKeys, [
            "layer:file:de5fb6857d3d968239ce05a4394bb7b7c8d2e314134285b909336996f178206a",
            "layer:directory:30d2ec2269081e189755a521e64b23500bc767c69ad35e95de021983a7947131",
            "app:com.test.documents",
            "app:com.apple.finder",
            "layer:directory:b86490e0dda9f8f25ef8a85c8effb634113ff34d960f7ef6a1327bab9111a4da",
            "layer:directory:665d11e6286d99cdd8abfda329d3b1b92bdae65ade51486856889b2c2f899a54",
            "layer:directory:99aa7f2dbb4720f8ca20573449b10acdbf680216e7125d7db3a1b38a6ddd907a",
        ])
    }

    func testBrowserResyncDoesNotBecomeAnotherVisit() throws {
        let a = app("other"), b = page("page:old", "https://example.test/topic/old")
        var history = OnlinePathInputHistory()
        for item in [try event(.targetUpdated, 0, metadata: a),
                     try event(.targetUpdated, 1, metadata: b),
                     try event(.contextChanged, 2, target: b.id),
                     try event(.contextChanged, 3, target: b.id, reason: "browser_resync")] {
            history.consume(item)
        }
        XCTAssertThrowsError(try history.candidates(current: b))
        history.consume(try event(.contextChanged, 4, target: a.id))
        let rows = try history.candidates(current: a)
        XCTAssertEqual(try XCTUnwrap(rows.first(where: { $0.actionID == b.id })?.row.numeric["candidate:path_count_all_log"]),
                       log(2), accuracy: 1e-6)
    }

    func testCurrentOwnerUsesSameForegroundEventsAppWhenPageMetadataLacksBundle() throws {
        let browser = app("browser")
        let old = page("page:old", "https://example.test/old")
        let current = Destination(id: "page:current", kind: .page, name: "Current",
                                  origin: "https://example.test",
                                  locator: "https://example.test/current",
                                  admission: TargetAdmission(basis: "supported_browser_page"))
        var history = OnlinePathInputHistory()
        for (index, item) in [browser, old, current].enumerated() {
            history.consume(try event(.targetUpdated, 2 * index, metadata: item))
            history.consume(try event(.contextChanged, 2 * index + 1, target: item.id,
                                      observedApp: "browser"))
        }
        let candidates = try history.candidates(current: current)
        XCTAssertFalse(candidates.contains { $0.actionID == browser.id })
        let oldRow = try XCTUnwrap(candidates.first { $0.actionID == old.id }?.row)
        XCTAssertEqual(oldRow.categorical["origin_bundle_token"],
                       OnlinePathMLPScorer.stableToken("browser"))
        XCTAssertEqual(oldRow.numeric["candidate:same_owner_app"], 1)
    }

    func testFeedbackUsesPositiveValueAncestorsAndKeepsDifferentIDNonHomeSeparate() throws {
        let current = app("other")
        let owner = app(TargetEligibilityPolicy.finderBundleID)
        let directory = Destination(id: "directory:parent", kind: .directory, name: "Parent",
                                    bundleID: TargetEligibilityPolicy.finderBundleID,
                                    locator: "file:///Users/me/Projects",
                                    admission: TargetAdmission(basis: "observed_finder_directory"))
        let known = page("page:known", "https://example.test/topic/item")
        var history = OnlinePathInputHistory()
        for (index, item) in [owner, directory, known, current].enumerated() {
            history.consume(try event(.targetUpdated, 2 * index, metadata: item))
            history.consume(try event(.contextChanged, 2 * index + 1, target: item.id))
        }
        let frozen = try history.candidates(current: current)
        let file = Destination(id: "file:future", kind: .file, name: "Future",
                               bundleID: "com.test.documents",
                               locator: "file:///Users/me/Projects/future.txt",
                               parentID: directory.id,
                               admission: TargetAdmission(basis: "observed_window_document"))
        let filePositives = OnlinePathInputHistory.usefulActionIndices(
            for: file, current: current, candidates: frozen).map { frozen[$0].actionID }
        XCTAssertTrue(filePositives.contains(directory.id))
        XCTAssertFalse(filePositives.contains(owner.id))
        let sameURL = page("page:future", "https://example.test/topic/item")
        let sameURLPositives = OnlinePathInputHistory.usefulActionIndices(
            for: sameURL, current: current, candidates: frozen).map { frozen[$0].actionID }
        XCTAssertFalse(sameURLPositives.contains(known.id))
        let training = OnlinePathInputHistory.trainingRows(for: sameURL, current: current,
                                                            frozenCandidates: frozen)
        XCTAssertTrue(training.appendedExact)
        XCTAssertTrue(training.positiveIndices.contains(frozen.count))
        XCTAssertFalse(training.positiveIndices.contains(try XCTUnwrap(frozen.firstIndex {
            $0.actionID == known.id
        })))
    }

    func testEquivalentHomepageCanTrainWithoutASecondExactLeaf() throws {
        let current = app("other")
        let home = page("page:known-home", "https://example.test/")
        var history = OnlinePathInputHistory()
        for (index, item) in [home, current].enumerated() {
            history.consume(try event(.targetUpdated, 2 * index, metadata: item))
            history.consume(try event(.contextChanged, 2 * index + 1, target: item.id))
        }
        let frozen = try history.candidates(current: current)
        let equivalent = page("page:future-home", "https://example.test/")
        let training = OnlinePathInputHistory.trainingRows(for: equivalent, current: current,
                                                            frozenCandidates: frozen)
        XCTAssertFalse(training.appendedExact)
        XCTAssertEqual(training.positiveIndices.map { frozen[$0].actionID }, [home.id])
    }

    func testCurrentServiceHomepageBlocksItsOtherRecordedVariant() throws {
        let home = Destination(id: "page:chat-home", kind: .page, name: "Chat",
                               bundleID: "browser", origin: "https://chatgpt.com",
                               locator: "https://chatgpt.com/",
                               admission: TargetAdmission(basis: "supported_browser_page"))
        let variant = Destination(id: "page:chat-variant", kind: .page, name: "Chat Variant",
                                  bundleID: "browser", origin: "https://chatgpt.com",
                                  locator: "https://chatgpt.com/?src=search",
                                  admission: TargetAdmission(basis: "supported_browser_page"))
        let unrelated = page("page:unrelated", "https://example.test/other")
        var history = OnlinePathInputHistory()
        for (index, item) in [home, unrelated, variant].enumerated() {
            history.consume(try event(.targetUpdated, 2 * index, metadata: item))
            history.consume(try event(.contextChanged, 2 * index + 1, target: item.id))
        }
        let candidates = try history.candidates(current: variant)
        XCTAssertFalse(candidates.contains { $0.actionID == home.id })
        XCTAssertTrue(candidates.contains { $0.actionID == unrelated.id })
    }

    func testFeedbackLeafUsesParentsKnownAtOutcomeWithoutChangingFrozenInput() throws {
        let previous = app("browser"), current = app("other")
        let parent = Destination(id: "directory:future-parent", kind: .directory, name: "Projects",
                                 bundleID: TargetEligibilityPolicy.finderBundleID,
                                 locator: "file:///Users/me/Projects",
                                 admission: TargetAdmission(basis: "observed_finder_directory"))
        let file = Destination(id: "file:future", kind: .file, name: "Future",
                               bundleID: "com.test.documents",
                               locator: "file:///Users/me/Projects/future.txt", parentID: parent.id,
                               admission: TargetAdmission(basis: "observed_window_document"))
        var history = OnlinePathInputHistory()
        for (index, item) in [previous, current].enumerated() {
            history.consume(try event(.targetUpdated, 2 * index, metadata: item))
            history.consume(try event(.contextChanged, 2 * index + 1, target: item.id))
        }
        let frozen = try history.candidates(current: current)
        XCTAssertFalse(frozen.contains { $0.actionID == parent.id || $0.actionID == file.id })
        history.consume(try event(.targetUpdated, 4, metadata: parent))
        history.consume(try event(.targetUpdated, 5, metadata: file))
        history.consume(try event(.contextChanged, 6, target: file.id))
        let parents = try XCTUnwrap(history.confirmedParentChain(eventID: "e6"))
        XCTAssertEqual(parents.map(\.id), [parent.id])
        var changedParent = parent
        changedParent.locator = "file:///Users/me/Different"
        history.consume(try event(.targetUpdated, 7, metadata: changedParent))
        XCTAssertEqual(history.confirmedParentChain(eventID: "e6")?.first?.locator,
                       parent.locator)
        let training = OnlinePathInputHistory.trainingRows(
            for: file, current: current, frozenCandidates: frozen, confirmedParents: parents)
        XCTAssertTrue(training.appendedExact)
        XCTAssertEqual(training.rows.dropLast().map(\.numeric), frozen.map { $0.row.numeric })
        XCTAssertEqual(training.rows.last?.layerKeys, [
            "layer:file:11816bb18fbe1e1ee90ad972258b598278432090615c7ce24271bb37f6cbc6d4",
            "layer:directory:30d2ec2269081e189755a521e64b23500bc767c69ad35e95de021983a7947131",
            "app:com.test.documents", "app:com.apple.finder",
            "layer:directory:b86490e0dda9f8f25ef8a85c8effb634113ff34d960f7ef6a1327bab9111a4da",
            "layer:directory:665d11e6286d99cdd8abfda329d3b1b92bdae65ade51486856889b2c2f899a54",
            "layer:directory:99aa7f2dbb4720f8ca20573449b10acdbf680216e7125d7db3a1b38a6ddd907a",
        ])
    }

    func testDuplicateConfirmationKeepsOriginalDwellStartAndIdleDoesNotAddVisit() throws {
        let a = page("page:a", "https://example.test/a"), b = app("other")
        var uninterrupted = OnlinePathInputHistory()
        for item in [try event(.targetUpdated, 0, metadata: a),
                     try event(.targetUpdated, 1, metadata: b),
                     try event(.contextChanged, 2, target: a.id),
                     try event(.contextChanged, 3, target: a.id),
                     try event(.contextChanged, 4, target: b.id)] { uninterrupted.consume(item) }
        let normal = try uninterrupted.candidates(current: b)
        XCTAssertEqual(try XCTUnwrap(normal.first?.row.numeric["context:last_confirmed_dwell_log_s"]),
                       log(3), accuracy: 1e-6)

        var interrupted = OnlinePathInputHistory()
        for item in [try event(.targetUpdated, 0, metadata: a),
                     try event(.targetUpdated, 1, metadata: b),
                     try event(.contextChanged, 2, target: a.id),
                     try event(.idleChanged, 3),
                     try event(.contextChanged, 4, target: a.id),
                     try event(.contextChanged, 5, target: b.id)] { interrupted.consume(item) }
        let afterIdle = try interrupted.candidates(current: b)
        XCTAssertEqual(afterIdle.first?.row.numeric["context:has_confirmed_dwell"], 0)
        XCTAssertEqual(try XCTUnwrap(afterIdle.first(where: { $0.actionID == a.id })?.row.numeric["candidate:direct_count_all_log"]),
                       log(2), accuracy: 1e-6)
    }

    func testDwellUsesWholeMonotonicMillisecondsAndExpiresWithSequenceWindow() throws {
        var history = OnlinePathInputHistory()
        let first = page("page:0", "https://example.test/0")
        let second = page("page:1", "https://example.test/1")
        for (index, item) in [first, second].enumerated() {
            history.consume(try event(.targetUpdated, index, metadata: item))
            history.consume(try event(.contextChanged, index + 20, target: item.id,
                                      monotonicMs: [20_000.9, 21_000.1][index]))
        }
        let early = try history.candidates(current: second)
        XCTAssertEqual(try XCTUnwrap(early.first?.row.numeric["context:last_confirmed_dwell_log_s"]),
                       Double(Float(Foundation.log1p(Double(1)))), accuracy: 1e-7)
        for index in 2..<18 {
            let item = page("page:\(index)", "https://example.test/\(index)")
            history.consume(try event(.targetUpdated, 50 + 2 * index, metadata: item))
            history.consume(try event(.contextChanged, 51 + 2 * index, target: item.id,
                                      sourceStatus: "initializing"))
        }
        let latest = page("page:17", "https://example.test/17")
        let rows = try history.candidates(current: latest)
        XCTAssertEqual(rows.first?.row.numeric["context:has_confirmed_dwell"], 0)
        XCTAssertEqual(rows.first?.row.numeric["context:last_confirmed_dwell_log_s"], 0)
    }

    func testWindowVisitSupportsAppPathWithoutDirectAppVisit() throws {
        let current = app("other")
        let window = Destination(id: "window:one", kind: .window, name: "Work",
                                 bundleID: "documents",
                                 admission: TargetAdmission(basis: "standard_work_window"))
        var history = OnlinePathInputHistory()
        for item in [try event(.targetUpdated, 0, metadata: current),
                     try event(.targetUpdated, 1, metadata: window),
                     try event(.contextChanged, 2, target: window.id),
                     try event(.contextChanged, 3, target: current.id)] { history.consume(item) }
        let candidate = try XCTUnwrap(history.candidates(current: current).first {
            $0.actionID == "app:documents"
        })
        XCTAssertEqual(candidate.row.numeric["candidate:path_count_all_log"], log(2))
        XCTAssertEqual(candidate.row.numeric["candidate:direct_count_all_log"], 0)
    }

    func testLargeHistoryMenuQueryUsesCachedCandidates() throws {
        var history = OnlinePathInputHistory()
        let current = app("other")
        history.consume(try event(.targetUpdated, 0, metadata: current))
        for index in 0..<600 {
            let destination = page("page:\(index)", "https://example.test/topic/\(index)")
            history.consume(try event(.targetUpdated, 2 * index + 1, metadata: destination))
            history.consume(try event(.contextChanged, 2 * index + 2, target: destination.id))
        }
        history.consume(try event(.contextChanged, 1_202, target: current.id))
        let started = ProcessInfo.processInfo.systemUptime
        let first = try history.candidates(current: current)
        let firstDuration = ProcessInfo.processInfo.systemUptime - started
        let cachedStarted = ProcessInfo.processInfo.systemUptime
        let second = try history.candidates(current: current)
        let cachedDuration = ProcessInfo.processInfo.systemUptime - cachedStarted
        XCTAssertEqual(first.map(\.actionID), second.map(\.actionID))
        XCTAssertGreaterThan(first.count, 600)
        print("OnlinePath menu query: \(first.count) candidates, first \(firstDuration)s, cached \(cachedDuration)s")
    }

    func testPrivateSnapshotPredictionCoverageWhenRequested() throws {
        guard let modelDirectory = ProcessInfo.processInfo.environment["SMARTAPP_PRIVATE_MODEL_PARITY_DIR"],
              ProcessInfo.processInfo.environment["SMARTAPP_PRIVATE_COVERAGE_AUDIT"] == "1" else { return }
        let runData = try Data(contentsOf: URL(fileURLWithPath: modelDirectory)
            .appendingPathComponent("run.json"))
        let run = try XCTUnwrap(JSONSerialization.jsonObject(with: runData) as? [String: Any])
        let snapshot = try XCTUnwrap(run["snapshot"] as? String)
        let raw = try String(contentsOf: URL(fileURLWithPath: snapshot)
            .appendingPathComponent("events.jsonl"), encoding: .utf8)
        var history = OnlinePathInputHistory()
        var metadata: [String: Destination] = [:]
        var byReason: [String: Int] = [:]
        var byCurrentKind: [String: [String: Int]] = [:]
        var eligible = 0, withCandidates = 0, emptyCandidates = 0
        let nonArrival = Set(["resolution_update", "browser_history", "browser_resync"])
        for line in raw.split(separator: "\n") {
          try autoreleasepool {
            let event = try JSONDecoder().decode(BehaviorEvent.self, from: Data(line.utf8))
            if event.kind == .targetUpdated, let encoded = event.data["target"],
               let target = try? encoded.decoded(Destination.self) {
                metadata[target.id] = target
            }
            history.consume(event)
            guard event.kind == .contextChanged,
                  !nonArrival.contains(event.data["reason"]?.string ?? ""),
                  let coverage = event.data["inputCoverage"]?.object,
                  coverage["version"]?.string == "foreground-coverage-v2",
                  coverage["status"]?.string == "foreground_only",
                  let targetID = event.data["targetID"]?.string,
                  let current = metadata[targetID], TargetEligibilityPolicy.validates(current) else { return }
            eligible += 1
            let reason: String
            if let unavailable = history.predictionUnavailableReason(current: current) {
                switch unavailable {
                case .noConfirmedCutoff: reason = "no_confirmed_cutoff"
                case .currentTargetMismatch: reason = "current_target_mismatch"
                case .invalidCoverage: reason = "invalid_coverage"
                case .invalidFeatureSchema: reason = "invalid_feature_schema"
                case .unsupportedObservation(let detail): reason = detail
                }
            } else {
                let rows = try history.candidates(current: current)
                if rows.isEmpty {
                    emptyCandidates += 1
                    reason = "no_other_candidate"
                } else {
                    withCandidates += 1
                    reason = "mlp_candidate_ready"
                }
            }
            byReason[reason, default: 0] += 1
            byCurrentKind[current.kind.rawValue, default: [:]][reason, default: 0] += 1
          }
        }
        XCTAssertEqual(eligible, byReason.values.reduce(0, +))
        print("PRIVATE_PREDICTION_COVERAGE eligible=\(eligible) with_candidates=\(withCandidates) " +
              "empty=\(emptyCandidates) reasons=\(byReason) by_kind=\(byCurrentKind)")
    }

    #if DEBUG
    func testPrivateLayerPathDiagnosticWhenRequested() throws {
        guard let modelDirectory = ProcessInfo.processInfo.environment["SMARTAPP_PRIVATE_MODEL_PARITY_DIR"],
              ProcessInfo.processInfo.environment["SMARTAPP_PRIVATE_LAYER_DIAGNOSTIC"] == "1" else { return }
        let runData = try Data(contentsOf: URL(fileURLWithPath: modelDirectory)
            .appendingPathComponent("run.json"))
        let run = try XCTUnwrap(JSONSerialization.jsonObject(with: runData) as? [String: Any])
        let snapshot = try XCTUnwrap(run["snapshot"] as? String)
        let raw = try String(contentsOf: URL(fileURLWithPath: snapshot)
            .appendingPathComponent("events.jsonl"), encoding: .utf8)
        let cases = [(cutoff: 3_624, hash: "061663c8d0aa"),
                     (cutoff: 10_458, hash: "ab3bb6eda96b")]
        var history = OnlinePathInputHistory()
        var metadata: [String: Destination] = [:]
        for (index, line) in raw.split(separator: "\n").enumerated() {
            if index > 10_458 { break }
            let event = try JSONDecoder().decode(BehaviorEvent.self, from: Data(line.utf8))
            if event.kind == .targetUpdated, let encoded = event.data["target"],
               let target = try? encoded.decoded(Destination.self) {
                metadata[target.id] = target
            }
            history.consume(event)
            for entry in cases where entry.cutoff == index {
                let actionID = try XCTUnwrap(metadata.keys.first { id in
                    SHA256.hash(data: Data(id.utf8)).map { String(format: "%02x", $0) }
                        .joined().hasPrefix(entry.hash)
                })
                let layers = try XCTUnwrap(history.diagnosticLayerEntries(actionID: actionID))
                let anonymous = layers.map { layer in
                    "\(OnlinePathMLPScorer.stableToken(layer.key).prefix(12)):\(layer.path.joined(separator: ">"))"
                }
                print("PRIVATE_LAYER_PATH cutoff=\(index) action_sha=\(entry.hash) entries=\(anonymous)")
            }
        }
    }
    #endif
}
