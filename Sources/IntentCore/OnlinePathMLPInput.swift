import Foundation
import CryptoKit

/// One action in the frozen, prediction-time candidate list. A derived action
/// exists only in memory; `recordedTarget == nil` must never be persisted as a visit.
public struct OnlinePathCandidate: Sendable {
    public let actionID: String
    public let recordedTarget: Destination?
    public let derivedKind: TargetKind?
    public let executableLocator: String?
    public let source: String
    public let semanticKey: String
    public let row: OnlinePathMLPRow
}

public struct OnlinePathTrainingRows: Sendable {
    public let rows: [OnlinePathMLPRow]
    public let positiveIndices: [Int]
    public let originalCandidateCount: Int
    public let appendedExact: Bool
}

public enum OnlinePathInputError: Error, Equatable {
    case noConfirmedCutoff
    case currentTargetMismatch
    case invalidCoverage
    case unsupportedObservation(String)
    case invalidFeatureSchema
}

private struct PathArrival: Sendable {
    let timestampMs: Int64
    let exactKey: String
    let layerKeys: [String]
}

private struct PathCutoff: Sendable {
    let target: Destination
    let ownerBundle: String?
    let timestampMs: Int64
    let eventID: String
    let context: [String: Double]
    let excludedArrival: PathArrival?
}

private struct PathActionTemplate: Sendable {
    let candidate: OnlinePathCandidate
    let homeGroup: String?
    let numeric: [Double]
    let categorical: [String]
}

/// Incremental mirror of the research model's normal, uninterrupted foreground
/// path. It never reads an open-window inventory or a future label. Candidate
/// derivation is deferred until `candidates` and cached for repeated menu opens.
public struct OnlinePathInputHistory: Sendable {
    public static let usefulActionRuleVersion = "online-path-useful-actions-v1"
    private var metadata: [String: Destination] = [:]
    private var confirmed: [String: Destination] = [:]
    private var frequencies = OnlinePathFrequencyIndex()
    private var arrivalCount = 0
    private var cutoff: PathCutoff?
    private var previousArrival: (targetID: String, sessionID: String, source: String,
                                  monotonicMs: Double, ready: Bool)?
    private var lastDwell: Double?
    private var lastDwellArrivalOrdinal: Int?
    private var dwellTrusted = false
    private var currentSession: String?
    private var cachedCutoffID: String?
    private var cachedCandidates: [OnlinePathCandidate] = []
    private var unsupported: String?
    private var confirmedByEventID: [String: Destination] = [:]
    private var parentChainByEventID: [String: [Destination]] = [:]
    private var excludedResumePending = false
    private var catalogChanged = true
    private var actionCatalog: [String: Destination] = [:]
    private var locationIndex: [String: [Destination]] = [:]
    private var actionTemplates: [PathActionTemplate] = []
    private var arrivalLayerCache: [String: (target: Destination, parents: [Destination], layers: [String])] = [:]
    private var pairOrigin: String?
    private var originPairTokens: [String: String] = [:]

    public init() {}

    /// A verified foreground outcome for matching a mature learning record.
    public func confirmedDestination(eventID: String) -> Destination? {
        confirmedByEventID[eventID]
    }

    /// Explicit parent metadata exactly as known when this foreground target
    /// was confirmed. It is used only after feedback for the training leaf.
    public func confirmedParentChain(eventID: String) -> [Destination]? {
        parentChainByEventID[eventID]
    }

    /// The learning caller releases outcomes only after their pending samples
    /// and interruption bridges are finished. No arbitrary event-count cap can
    /// evict an outcome that is still needed after a long known pause.
    public mutating func releaseConfirmedEvents(keeping eventIDs: Set<String>) {
        for id in confirmedByEventID.keys where !eventIDs.contains(id) {
            confirmedByEventID[id] = nil
            parentChainByEventID[id] = nil
        }
    }

    public mutating func consume(_ event: BehaviorEvent) {
        if currentSession != event.sessionID {
            currentSession = event.sessionID
            previousArrival = nil
            lastDwell = nil
            lastDwellArrivalOrdinal = nil
            dwellTrusted = false
            cutoff = nil
            cachedCutoffID = nil
            excludedResumePending = false
        }
        switch event.kind {
        case .targetUpdated:
            if let raw = event.data["target"], let target = try? raw.decoded(Destination.self),
               TargetEligibilityPolicy.validates(target) {
                metadata[target.id] = target
            }
        case .captureGap, .observationScope, .sessionStart, .sessionEnd:
            previousArrival = nil
            lastDwell = nil
            lastDwellArrivalOrdinal = nil
            dwellTrusted = false
            cutoff = nil
            cachedCutoffID = nil
            if event.kind == .observationScope {
                // Only the first post-exclusion context may have a paired
                // bridge marker that this conservative runtime cannot encode.
                unsupported = "excluded_scope_resume_not_represented"
                excludedResumePending = true
            } else if event.kind == .sessionStart {
                unsupported = nil
                excludedResumePending = false
            } else if event.kind == .captureGap {
                // An unpaired gap resets the sequence in the research builder;
                // normal new foreground confirmations have zero gap features.
                unsupported = nil
            }
        case .sourceStatus:
            if let previousArrival, event.source == previousArrival.source,
               event.data["observationStatus"]?.string != "ready" {
                dwellTrusted = false
                unsupported = "foreground_source_not_ready_after_cutoff"
            }
        case .idleChanged:
            // Idle makes elapsed time unsafe as dwell, but the foreground
            // sequence is not a new independent visit to the same target.
            dwellTrusted = false
            unsupported = "idle_since_last_confirmed_cutoff"
        case .contextChanged:
            consumeConfirmedContext(event)
        default: break
        }
    }

    private mutating func consumeConfirmedContext(_ event: BehaviorEvent) {
        guard let coverage = event.data["inputCoverage"]?.object,
              coverage["version"]?.string == "foreground-coverage-v2",
              coverage["status"]?.string == "foreground_only",
              let targetID = event.data["targetID"]?.string,
              let observed = metadata[targetID],
              let target = Self.action(observed) else {
            cutoff = nil; cachedCutoffID = nil; return
        }
        // A target.update alone does not put an action in the catalog.
        if confirmed[target.id] != target {
            confirmed[target.id] = target
            catalogChanged = true
        }
        confirmedByEventID[event.id] = observed
        var parentChain: [Destination] = []
        var pointer = observed
        var seenParents = Set([observed.id])
        while let parentID = pointer.parentID, let parent = metadata[parentID],
              seenParents.insert(parentID).inserted {
            parentChain.append(parent)
            pointer = parent
        }
        parentChainByEventID[event.id] = parentChain
        if let bundle = observed.bundleID, observed.kind != .app,
           let app = metadata["app:" + bundle], TargetEligibilityPolicy.validates(app) {
            if confirmed[app.id] != app { confirmed[app.id] = app; catalogChanged = true }
        }
        guard let context = Self.context(event: event, lastDwell: lastDwell) else {
            unsupported = "foreground_coverage_cannot_be_encoded"
            cutoff = nil; cachedCutoffID = nil; return
        }
        let nonArrivalReasons: Set<String> = ["resolution_update", "browser_history", "browser_resync"]
        let isResolution = event.data["reason"]?.string.map(nonArrivalReasons.contains) == true
        if isResolution {
            cutoff = nil
            cachedCutoffID = nil
            unsupported = "context_update_has_no_research_cutoff"
            return
        }
        let isIndependent = previousArrival == nil || previousArrival?.targetID != target.id
        let ready = Self.sourceReady(event, coverage: coverage)
        if isIndependent, let previousArrival, dwellTrusted,
           previousArrival.sessionID == event.sessionID && previousArrival.ready && ready,
           let previousMs = Self.researchMonotonic(previousArrival.monotonicMs),
           let currentMs = Self.researchMonotonic(event.monotonicMs), currentMs >= previousMs {
            // The research sequence converts both monotonic samples to whole
            // milliseconds before subtracting, then stores the log as Float32.
            lastDwell = Double(Float(log1p(Double(currentMs - previousMs) / 1_000)))
            // Python retains only the last 16 sequence steps. The closed
            // dwell belongs to the previous arrival, not the new one.
            lastDwellArrivalOrdinal = arrivalCount - 1
        }
        // Consume committed database rows exactly once, in their write order.
        // Duplicate notifications are rejected by the caller's database cursor.
        var appendedArrival: PathArrival?
        if isIndependent {
            // Explicit parent metadata may have been observed before this
            // arrival without the parent ever becoming an executable action.
            // It supports a path-count layer, not a prediction candidate.
            // A window's menu action is its app, but the observed visit's
            // direct frequency belongs to the window *class* layer. Python
            // makes the same separation in RichIndex.records.
            // RichIndex builds an arrival's path from that event's exact
            // metadata and explicit parent chain. Other known targets belong
            // to prediction-time candidate construction, not a retroactive
            // rewrite of this already observed arrival.
            let layers: [String]
            if let cached = arrivalLayerCache[observed.id],
               cached.target == observed, cached.parents == parentChain {
                layers = cached.layers
            } else {
                var lineage = [observed.id: observed]
                for parent in parentChain { lineage[parent.id] = parent }
                layers = Self.layerKeys(for: observed, catalog: lineage, ownerApp: true)
                arrivalLayerCache[observed.id] = (observed, parentChain, layers)
            }
            frequencies.append(timestamp: event.timestampMs, exact: layers[0], layers: layers)
            arrivalCount += 1
            appendedArrival = PathArrival(timestampMs: event.timestampMs, exactKey: layers[0], layerKeys: layers)
        }
        if isIndependent {
            previousArrival = (target.id, event.sessionID, event.source, event.monotonicMs, ready)
            dwellTrusted = ready
        }
        var corrected = context
        // The most recent confirmed dwell is available at this cutoff, but the
        // current arrival's still-open dwell is never used.
        let dwellInSequence = lastDwellArrivalOrdinal.map { $0 >= arrivalCount - 16 } == true
        corrected["last_confirmed_dwell_log_s"] = dwellInSequence ? (lastDwell ?? 0) : 0
        corrected["has_confirmed_dwell"] = dwellInSequence ? 1 : 0
        let observedOwner = target.kind == .app ? target.bundleID : (event.app?.bundleID ?? target.bundleID)
        cutoff = PathCutoff(target: target, ownerBundle: observedOwner,
                            timestampMs: event.timestampMs, eventID: event.id,
                            context: corrected, excludedArrival: appendedArrival)
        cachedCutoffID = nil
        cachedCandidates = []
        if excludedResumePending {
            unsupported = "excluded_scope_resume_not_represented"
            excludedResumePending = false
        } else {
            unsupported = nil
        }
    }

    /// Uses the last confirmed arrival as the cutoff, even when the menu opens
    /// later. This is the timestamp and coverage definition used in training.
    public func predictionUnavailableReason(current: Destination) -> OnlinePathInputError? {
        if let unsupported { return .unsupportedObservation(unsupported) }
        guard let cutoff else { return .noConfirmedCutoff }
        guard Self.action(current)?.id == cutoff.target.id else { return .currentTargetMismatch }
        return nil
    }

    public mutating func candidates(current: Destination) throws -> [OnlinePathCandidate] {
        if let error = predictionUnavailableReason(current: current) { throw error }
        guard let cutoff else { throw OnlinePathInputError.noConfirmedCutoff }
        if cachedCutoffID == cutoff.eventID { return cachedCandidates }
        if catalogChanged { rebuildActionTemplates() }
        let rows = try buildCandidates(cutoff: cutoff)
        cachedCandidates = rows
        cachedCutoffID = cutoff.eventID
        return rows
    }

    /// Exactly one positive set for a matured arrival, never one visit per parent.
    public static func usefulActionIndices(for outcome: Destination, current: Destination,
                                           candidates: [OnlinePathCandidate]) -> [Int] {
        let catalog = Dictionary(uniqueKeysWithValues: candidates.compactMap { item in
            item.recordedTarget.map { ($0.id, $0) }
        })
        let outcomeAction = action(outcome) ?? outcome
        let paths = Dictionary(uniqueKeysWithValues:
            layerEntries(for: outcomeAction, catalog: catalog, ownerApp: true).map { ($0.key, $0.path) })
        let currentAction = action(current) ?? current
        let outcomeSemantic = semanticKey(outcomeAction)
        let outcomeHome = homeGroup(of: outcomeAction)
        let currentHome = homeGroup(of: currentAction)
        let currentBundleToken = candidates.first?.row.categorical["origin_bundle_token"]
        return candidates.enumerated().compactMap { index, candidate in
            if candidate.actionID == currentAction.id ||
                candidate.recordedTarget?.kind == .app &&
                candidate.row.categorical["candidate_bundle_token"] == currentBundleToken {
                return nil
            }
            let proposedHome = homeGroup(kind: candidate.recordedTarget?.kind ?? candidate.derivedKind,
                                         locator: candidate.executableLocator)
            if let currentHome, proposedHome == currentHome { return nil }
            if candidate.actionID == outcomeAction.id {
                return candidate.semanticKey == outcomeSemantic ? index : nil
            }
            if let outcomeHome, proposedHome == outcomeHome { return index }
            // A different ID at the identical non-home URL/path is not an
            // exact hit. Only a real ancestor path can earn a partial label.
            guard candidate.semanticKey != outcomeSemantic,
                  let path = paths[candidate.semanticKey], !path.isEmpty else { return nil }
            if path == ["owner_app"] { return index }
            return path.contains("owner_app") ? nil : index
        }
    }

    /// Attach a newly confirmed exact destination to a *training-only* view.
    /// Counts and time/quality inputs remain frozen at the original prediction.
    /// The appended leaf can be positive but must never enter a menu or the
    /// negative sample pool.
    public static func trainingRows(for outcome: Destination, current: Destination,
                                    frozenCandidates: [OnlinePathCandidate],
                                    confirmedParents: [Destination] = []) -> OnlinePathTrainingRows {
        let original = frozenCandidates.count
        var rows = frozenCandidates.map(\.row)
        let outcomeAction = action(outcome) ?? outcome
        let outcomeSemantic = semanticKey(outcomeAction)
        guard outcomeSemantic != semanticKey(action(current) ?? current) else {
            return OnlinePathTrainingRows(rows: rows, positiveIndices: [],
                                          originalCandidateCount: original, appendedExact: false)
        }
        let outcomeBundleToken = OnlinePathMLPScorer.stableToken(outcomeAction.bundleID)
        if frozenCandidates.contains(where: { candidate in
            candidate.actionID == outcomeAction.id &&
                (candidate.semanticKey != outcomeSemantic ||
                 candidate.row.categorical["candidate_kind"] != outcomeAction.kind.rawValue ||
                 candidate.row.categorical["candidate_bundle_token"] != outcomeBundleToken)
        }) {
            // A reused ID with a changed locator or owner was a different
            // action at prediction time; it cannot supervise this frozen row.
            return OnlinePathTrainingRows(rows: rows, positiveIndices: [],
                                          originalCandidateCount: original, appendedExact: false)
        }
        var useful = usefulActionIndices(for: outcome, current: current, candidates: frozenCandidates)
        let alreadyExact = frozenCandidates.contains { $0.actionID == outcomeAction.id }
        let equivalentHome = frozenCandidates.contains { candidate in
            candidate.actionID != outcomeAction.id &&
                candidate.semanticKey == outcomeSemantic &&
                candidate.row.categorical["candidate_kind"] == outcomeAction.kind.rawValue &&
                candidate.row.categorical["candidate_bundle_token"] == outcomeBundleToken &&
                homeGroup(kind: candidate.recordedTarget?.kind ?? candidate.derivedKind,
                          locator: candidate.executableLocator) == homeGroup(of: outcomeAction) &&
                homeGroup(of: outcomeAction) != nil
        }
        guard !alreadyExact && !equivalentHome, let template = rows.first else {
            return OnlinePathTrainingRows(rows: rows, positiveIndices: useful,
                                          originalCandidateCount: original, appendedExact: false)
        }
        let frozenCatalog = Dictionary(uniqueKeysWithValues: frozenCandidates.compactMap { candidate in
            candidate.recordedTarget.map { ($0.id, $0) }
        })
        var lineage = [outcomeAction.id: outcomeAction]
        var pointer = outcomeAction
        var seenParents = Set([outcomeAction.id])
        let outcomeParents = Dictionary(uniqueKeysWithValues: confirmedParents.map { ($0.id, $0) })
        while let parentID = pointer.parentID, seenParents.insert(parentID).inserted,
              let parent = outcomeParents[parentID] ?? frozenCatalog[parentID] {
            lineage[parentID] = parent
            pointer = parent
        }
        let layers = layerKeys(for: outcomeAction, catalog: lineage, ownerApp: true)
        let parents = Array(layers.dropFirst())
        var numeric = template.numeric
        numeric["candidate:hierarchy_layer_count_log"] = log1p(Double(layers.count))
        numeric["candidate:is_derived_parent"] = 0
        let originBundleToken = template.categorical["origin_bundle_token"]
        numeric["candidate:same_owner_app"] = outcomeAction.bundleID != nil &&
            originBundleToken == outcomeBundleToken ? 1 : 0
        for name in ["5m", "1h", "24h", "all"] {
            numeric["candidate:path_count_\(name)_log"] = 0
            numeric["candidate:direct_count_\(name)_log"] = 0
        }
        var categorical = template.categorical
        categorical["candidate_token"] = OnlinePathMLPScorer.stableToken(outcomeAction.id)
        categorical["origin_candidate_token"] =
            OnlinePathMLPScorer.stableToken((action(current)?.id ?? current.id) + "\0" + outcomeAction.id)
        categorical["candidate_kind"] = outcomeAction.kind.rawValue
        categorical["candidate_bundle_token"] = OnlinePathMLPScorer.stableToken(outcomeAction.bundleID)
        categorical["candidate_source"] = "recorded_action"
        for slot in 0..<4 {
            categorical["candidate_parent_token_\(slot + 1)"] =
                OnlinePathMLPScorer.stableToken(slot < parents.count ? parents[slot] : nil)
        }
        rows.append(OnlinePathMLPRow(numeric: numeric, categorical: categorical, layerKeys: layers))
        useful.append(rows.count - 1)
        return OnlinePathTrainingRows(rows: rows, positiveIndices: useful,
                                      originalCandidateCount: original, appendedExact: true)
    }

    private static func action(_ target: Destination) -> Destination? {
        guard TargetEligibilityPolicy.validates(target) else { return nil }
        if target.kind == .window, let bundle = target.bundleID {
            return Destination(id: "app:" + bundle, kind: .app, name: bundle, bundleID: bundle,
                               admission: TargetAdmission(basis: "standalone_application"))
        }
        return target
    }

    private static func context(event: BehaviorEvent, lastDwell: Double?) -> [String: Double]? {
        guard let coverage = event.data["inputCoverage"]?.object,
              coverage["status"]?.string == "foreground_only" else { return nil }
        var counts = [String: Double]()
        let sources = coverage["sources"]?.array ?? []
        for source in sources {
            if let status = source.object?["status"]?.string { counts[status, default: 0] += 1 }
        }
        let total = max(1, counts.values.reduce(0, +))
        let known = ["ready", "initializing", "unavailable", "failed", "stale"]
        let unknown = counts.filter { !known.contains($0.key) }.values.reduce(0, +)
        var calendar = Calendar(identifier: .gregorian)
        calendar.timeZone = TimeZone(identifier: "America/Los_Angeles")!
        let date = Date(timeIntervalSince1970: Double(event.timestampMs) / 1_000)
        let parts = calendar.dateComponents([.hour, .minute, .weekday], from: date)
        guard let hour = parts.hour, let minute = parts.minute, let weekday = parts.weekday else { return nil }
        let hourValue = Double(hour) + Double(minute) / 60
        let mondayZero = Double((weekday + 5) % 7)
        return [
            "time_hour_sin": sin(2 * .pi * hourValue / 24),
            "time_hour_cos": cos(2 * .pi * hourValue / 24),
            "time_weekday_sin": sin(2 * .pi * mondayZero / 7),
            "time_weekday_cos": cos(2 * .pi * mondayZero / 7),
            "last_confirmed_dwell_log_s": lastDwell ?? 0,
            "has_confirmed_dwell": lastDwell == nil ? 0 : 1,
            "quality_ready_fraction": counts["ready", default: 0] / total,
            "quality_syncing_fraction": counts["initializing", default: 0] / total,
            "quality_unavailable_fraction": counts["unavailable", default: 0] / total,
            "quality_failed_fraction": counts["failed", default: 0] / total,
            "quality_stale_fraction": counts["stale", default: 0] / total,
            "quality_unknown_fraction": sources.isEmpty ? 1 : unknown / total,
            "quality_confirmed": 1,
            "has_same_target_inferred_active": 0,
            "same_target_inferred_active_log_s": 0,
            "has_legacy_short_gap": 0,
            "has_excluded_scope_gap": 0,
            "known_gap_log_s": 0,
            "known_gap_duration": 0,
            "rich_input_missing": 0,
        ]
    }

    private static func sourceReady(_ event: BehaviorEvent, coverage: [String: JSONValue]) -> Bool {
        let matches = (coverage["sources"]?.array ?? []).compactMap(\.object).filter {
            $0["source"]?.string == event.source
        }
        return matches.count == 1 && matches[0]["status"]?.string == "ready"
    }

    private static func researchMonotonic(_ value: Double) -> Int64? {
        guard value.isFinite, value > Double(Int64.min), value < Double(Int64.max) else { return nil }
        return Int64(value)
    }

    private mutating func rebuildActionTemplates() {
        var actions = confirmed
        var derived = Set<String>()
        let recordedLocations = Set(confirmed.values.compactMap { Self.locationKey($0) })
        for action in confirmed.values {
            for parent in Self.locatorParents(of: action) where !recordedLocations.contains(Self.locationKey(parent) ?? "") {
                actions[parent.id] = parent
                derived.insert(parent.id)
            }
        }
        actionCatalog = actions
        locationIndex = Self.structuralIndex(actions)
        var templates: [PathActionTemplate] = []
        templates.reserveCapacity(actions.count)
        for id in actions.keys.sorted() {
            guard let item = actions[id] else { continue }
            let semantic = Self.semanticKey(item)
            let layers = Self.layerKeys(for: item, catalog: actions, ownerApp: false,
                                   locationIndex: locationIndex)
            let source = derived.contains(id) ? "derived_from_known_locator" : "recorded_action"
            var numeric = [Double](repeating: 0, count: OnlinePathMLPScorer.numericFields.count)
            numeric[20] = log1p(Double(layers.count))
            numeric[21] = derived.contains(id) ? 1 : 0
            let parents = Array(layers.dropFirst())
            var categorical = [String](repeating: "none", count: OnlinePathMLPScorer.categoricalFields.count)
            categorical[1] = OnlinePathMLPScorer.stableToken(id)
            categorical[4] = item.kind.rawValue
            categorical[6] = OnlinePathMLPScorer.stableToken(item.bundleID)
            categorical[7] = source
            for slot in 0..<4 { categorical[8 + slot] =
                OnlinePathMLPScorer.stableToken(slot < parents.count ? parents[slot] : nil) }
            let candidate = OnlinePathCandidate(actionID: id,
                recordedTarget: derived.contains(id) ? nil : item,
                derivedKind: derived.contains(id) ? item.kind : nil,
                executableLocator: item.locator, source: source, semanticKey: semantic,
                row: OnlinePathMLPRow(numericValues: numeric, categoricalValues: categorical, layerKeys: layers))
            templates.append(PathActionTemplate(candidate: candidate, homeGroup: Self.homeGroup(of: item),
                                                 numeric: numeric, categorical: categorical))
        }
        actionTemplates = templates
        catalogChanged = false
    }

    private mutating func buildCandidates(cutoff: PathCutoff) throws -> [OnlinePathCandidate] {
        let current = cutoff.target
        let currentBundle = cutoff.ownerBundle
        let currentHome = Self.homeGroup(of: current)
        let currentSemantic = Self.semanticKey(current)
        let currentLayers = Self.layerKeys(for: current, catalog: actionCatalog, ownerApp: true,
                                           locationIndex: locationIndex)
        let currentParents = Array(currentLayers.dropFirst())
        let originToken = OnlinePathMLPScorer.stableToken(current.id)
        let originBundleToken = OnlinePathMLPScorer.stableToken(currentBundle)
        let currentParentTokens = (0..<2).map {
            OnlinePathMLPScorer.stableToken($0 < currentParents.count ? currentParents[$0] : nil)
        }
        let contextFields = Array(OnlinePathMLPScorer.numericFields.prefix(20))
        guard Set(contextFields) == Set(cutoff.context.keys.map { "context:" + $0 }) else {
            throw OnlinePathInputError.invalidFeatureSchema
        }
        let contextValues = contextFields.map { cutoff.context[String($0.dropFirst(8))]! }
        if pairOrigin != current.id { pairOrigin = current.id; originPairTokens.removeAll(keepingCapacity: true) }
        let excludedKeys = Set(cutoff.excludedArrival?.layerKeys ?? [])
        var result: [OnlinePathCandidate] = []
        result.reserveCapacity(actionTemplates.count)
        for template in actionTemplates {
            let candidate = template.candidate
            guard let item = actionCatalog[candidate.actionID] else { continue }
            if candidate.actionID == current.id || (item.kind == .app && item.bundleID == currentBundle) ||
                candidate.semanticKey == currentSemantic ||
                (currentHome != nil && template.homeGroup == currentHome) { continue }
            var frequency = frequencies.counts(for: candidate.semanticKey, at: cutoff.timestampMs)
            // The latest independent arrival is known at the cutoff, but its
            // frequency was not part of the pre-arrival history in v3.
            if excludedKeys.contains(candidate.semanticKey) {
                for slot in 0..<4 {
                    frequency.path[slot] -= 1
                    if cutoff.excludedArrival?.exactKey == candidate.semanticKey { frequency.direct[slot] -= 1 }
                }
            }
            var numeric = template.numeric
            for (column, value) in contextValues.enumerated() { numeric[column] = value }
            numeric[22] = currentBundle != nil && item.bundleID == currentBundle ? 1 : 0
            for slot in 0..<4 {
                numeric[23 + slot] = log1p(Double(frequency.path[slot]))
                numeric[27 + slot] = log1p(Double(frequency.direct[slot]))
            }
            var categorical = template.categorical
            categorical[0] = originToken
            if originPairTokens[candidate.actionID] == nil {
                originPairTokens[candidate.actionID] = OnlinePathMLPScorer.stableToken(current.id + "\0" + candidate.actionID)
            }
            categorical[2] = originPairTokens[candidate.actionID]!
            categorical[3] = current.kind.rawValue
            categorical[5] = originBundleToken
            categorical[12] = currentParentTokens[0]
            categorical[13] = currentParentTokens[1]
            result.append(OnlinePathCandidate(actionID: candidate.actionID, recordedTarget: candidate.recordedTarget,
                derivedKind: candidate.derivedKind, executableLocator: candidate.executableLocator,
                source: candidate.source, semanticKey: candidate.semanticKey,
                row: OnlinePathMLPRow(numericValues: numeric, categoricalValues: categorical, layerKeys: candidate.row.layerKeys)))
        }
        return result
    }

    private static func structuralIndex(_ catalog: [String: Destination]) -> [String: [Destination]] {
        var result: [String: [Destination]] = [:]
        for destination in catalog.values {
            if let key = locationKey(destination) { result[key, default: []].append(destination) }
        }
        return result
    }

    private static func layerKeys(for target: Destination, catalog: [String: Destination],
                                  ownerApp: Bool,
                                  locationIndex suppliedIndex: [String: [Destination]]? = nil) -> [String] {
        layerEntries(for: target, catalog: catalog, ownerApp: ownerApp,
                     locationIndex: suppliedIndex).map(\.key)
    }

    private static func layerEntries(for target: Destination, catalog: [String: Destination],
                                     ownerApp: Bool,
                                     locationIndex suppliedIndex: [String: [Destination]]? = nil)
        -> [(key: String, path: [String])] {
        let exact = semanticKey(target)
        let locationIndex = suppliedIndex ?? structuralIndex(catalog)
        var ancestors: [String: [String]] = [:]
        func add(_ key: String, _ path: [String], preferOnTie: Bool = false) {
            if key != exact && (ancestors[key] == nil || path.count < ancestors[key]!.count ||
                (preferOnTie && path.count == ancestors[key]!.count)) {
                ancestors[key] = path
            }
        }
        let parents = locatorParents(of: target)
        for (offset, parent) in parents.enumerated() {
            let kind = target.kind == .page ? (offset == parents.count - 1 ? "service_home" : "url_parent") : "path_parent"
            let distance = target.kind == .page && kind == "service_home" ? 1 : offset + 1
            let path = Array(repeating: kind, count: distance)
            add(semanticKey(parent), path)
            // Every recorded ancestor is another branch in the Python DAG.
            // Its owner application may contribute an additional distinct
            // layer via owner_app + path_parent/service_home, even though the
            // ancestor and synthetic locator have the same semantic key.
            if let key = locationKey(parent) {
                for recorded in locationIndex[key] ?? [] {
                    if let bundle = recorded.bundleID, recorded.kind != .app,
                       (ownerApp || catalog["app:" + bundle]?.kind == .app) {
                        add("app:" + bundle, ["owner_app"] + path)
                    }
                }
            }
        }
        // ChatGPT permits several already-known service home URLs (for
        // example approved query variants) with distinct semantic identities.
        // They are sibling parent actions, not aliases of an arbitrary host.
        if let service = NavigationSemantics.service(SemanticMetadata.annotate(target).identity),
           !["", "/"].contains(URLComponents(string: target.locator ?? "")?.path ?? "") {
            for home in catalog.values where home.kind == .page && home.id != target.id {
                guard ["", "/"].contains(URLComponents(string: home.locator ?? "")?.path ?? ""),
                      NavigationSemantics.service(SemanticMetadata.annotate(home).identity) == service else { continue }
                add(semanticKey(home), ["service_home"])
            }
        }
        var pointer = target
        var explicitPath: [String] = []
        var seenExplicit: Set<String> = [target.id]
        while let parentID = pointer.parentID, let parent = catalog[parentID],
              seenExplicit.insert(parentID).inserted, seenExplicit.count <= 16 {
            explicitPath.insert("explicit_parent", at: 0)
            // The direct explicit edge wins over an inferred edge to this
            // same first parent ID. At deeper levels Python already inserted
            // an equal-length direct locator edge and keeps that first path.
            add(semanticKey(parent), explicitPath, preferOnTie: seenExplicit.count == 2)
            let parentLocations = locatorParents(of: parent)
            for (offset, ancestor) in parentLocations.enumerated() {
                let edge = parent.kind == .page ?
                    (offset == parentLocations.count - 1 ? "service_home" : "url_parent") : "path_parent"
                let distance = parent.kind == .page && edge == "service_home" ? 1 : offset + 1
                add(semanticKey(ancestor), Array(repeating: edge, count: distance) + explicitPath)
            }
            if let bundle = parent.bundleID, parent.kind != .app,
               (ownerApp || catalog["app:" + bundle]?.kind == .app) {
                add("app:" + bundle, ["owner_app"] + explicitPath,
                    preferOnTie: seenExplicit.count == 2)
            }
            pointer = parent
        }
        if let bundle = target.bundleID, target.kind != .app {
            if ownerApp {
                add("app:" + bundle, ["owner_app"])
            } else if let app = catalog["app:" + bundle], app.kind == .app {
                add(semanticKey(app), ["owner_app"])
            }
        }
        let ordered = ancestors.sorted { left, right in
            if left.value.count != right.value.count { return left.value.count < right.value.count }
            if left.value != right.value { return left.value.lexicographicallyPrecedes(right.value) }
            return left.key < right.key
        }.map { (key: $0.key, path: $0.value) }
        return [(key: exact, path: [])] + ordered
    }

    #if DEBUG
    /// Local parity diagnostics only; neither locations nor catalog state are persisted.
    func diagnosticLayerEntries(actionID: String) -> [(key: String, path: [String])]? {
        guard let action = confirmed[actionID] else { return nil }
        var catalog = confirmed
        let locations = Set(confirmed.values.compactMap { Self.locationKey($0) })
        for known in confirmed.values {
            for parent in Self.locatorParents(of: known)
                where !locations.contains(Self.locationKey(parent) ?? "") {
                catalog[parent.id] = parent
            }
        }
        return Self.layerEntries(for: action, catalog: catalog, ownerApp: false)
    }
    #endif

    private static func locatorParents(of target: Destination) -> [Destination] {
        guard let locator = target.locator, let parts = URLComponents(string: locator) else { return [] }
        if target.kind == .page, let scheme = parts.scheme?.lowercased(),
           ["http", "https"].contains(scheme), let host = parts.host?.lowercased(),
           parts.user == nil, parts.password == nil, !parts.percentEncodedPath.contains("\\") {
            let raw = parts.percentEncodedPath
            guard raw.isEmpty || raw == "/" || (raw.hasPrefix("/") && !raw.contains("//")) else { return [] }
            var segments = raw.isEmpty || raw == "/" ? [] : String(raw.dropFirst()).split(separator: "/", omittingEmptySubsequences: false).map(String.init)
            if segments.last == "" { segments.removeLast() }
            guard segments.allSatisfy({ segment in
                guard !segment.isEmpty, segment != ".", segment != "..",
                      let decoded = segment.removingPercentEncoding else { return false }
                return decoded != "." && decoded != ".." && !decoded.contains("/") && !decoded.contains("\\")
            }) else { return [] }
            guard !segments.isEmpty else { return [] }
            let port = parts.port == (scheme == "http" ? 80 : 443) ? nil : parts.port
            let authority = host.contains(":") ? "[\(host)]" : host
            let origin = "\(scheme)://\(authority)\(port.map { ":\($0)" } ?? "")"
            return (0..<segments.count).reversed().map { depth in
                let path = depth == 0 ? "/" : "/" + segments.prefix(depth).joined(separator: "/") + "/"
                let parentLocator = origin + path
                return derivedTarget(kind: .page, locator: parentLocator, origin: origin)
            }
        }
        if target.kind == .directory || target.kind == .file,
           parts.scheme?.lowercased() == "file", parts.query == nil, parts.fragment == nil,
           parts.user == nil, parts.password == nil, parts.port == nil,
           let path = parts.percentEncodedPath.removingPercentEncoding,
           path.hasPrefix("/"), !parts.percentEncodedPath.lowercased().contains("%2f"),
           !path.contains("\\"), !path.contains("\0") {
            let host = parts.host?.lowercased() == "localhost" ? "" : (parts.host?.lowercased() ?? "")
            var current = (path as NSString).standardizingPath
            var output: [Destination] = []
            while true {
                let parent = (current as NSString).deletingLastPathComponent
                if parent == current || parent.isEmpty { break }
                let allowed = CharacterSet(charactersIn: "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-._~/:@")
                let encoded = parent.addingPercentEncoding(withAllowedCharacters: allowed) ?? parent
                output.append(derivedTarget(kind: .directory, locator: "file://\(host)\(encoded)", origin: nil))
                if parent == "/" { break }
                current = parent
            }
            return output
        }
        return []
    }

    private static func derivedTarget(kind: TargetKind, locator: String, origin: String?) -> Destination {
        let digest = OnlinePathMLPScorer.lowercaseHex(SHA256.hash(data: Data((kind.rawValue + "\0" + locator).utf8)))
        return Destination(id: "derived-parent:\(kind.rawValue):\(digest.prefix(24))", kind: kind,
                           name: locator, origin: origin, locator: locator)
    }

    private static func locationKey(_ target: Destination) -> String? {
        guard let locator = target.locator, let parts = URLComponents(string: locator) else { return nil }
        if target.kind == .page {
            guard parts.query == nil, parts.fragment == nil else { return nil }
            let path = parts.percentEncodedPath.trimmingCharacters(in: CharacterSet(charactersIn: "/"))
            return "page:\(parts.scheme?.lowercased() ?? ""):\(parts.host?.lowercased() ?? ""):\(parts.port ?? 0):\(path)"
        }
        if target.kind == .directory, parts.scheme?.lowercased() == "file",
           parts.query == nil, parts.fragment == nil,
           let decoded = parts.percentEncodedPath.removingPercentEncoding,
           decoded.hasPrefix("/") {
            let host = (parts.host?.lowercased() ?? "").trimmingCharacters(in: CharacterSet(charactersIn: "."))
            let normalizedHost = host == "localhost" ? "" : host
            return "directory:\(normalizedHost):\((decoded as NSString).standardizingPath)"
        }
        return nil
    }

    private static func homeGroup(of target: Destination) -> String? {
        homeGroup(kind: target.kind, locator: target.locator)
    }

    private static func homeGroup(kind: TargetKind?, locator: String?) -> String? {
        guard kind == .page, let locator, let url = URLComponents(string: locator),
              let scheme = url.scheme?.lowercased(), ["http", "https"].contains(scheme),
              let host = url.host?.lowercased(), !host.isEmpty,
              url.user == nil, url.password == nil,
              url.percentEncodedPath.isEmpty || url.percentEncodedPath == "/" else { return nil }
        let page = Destination(id: "home-group", kind: .page, name: "", locator: locator)
        if let service = NavigationSemantics.service(SemanticMetadata.annotate(page).identity) {
            return service
        }
        guard url.query == nil, url.fragment == nil else { return nil }
        let port = url.port == (scheme == "http" ? 80 : 443) ? nil : url.port
        return "web-home:\(scheme)://\(host.trimmingCharacters(in: CharacterSet(charactersIn: ".")))" +
            "\(port.map { ":\($0)" } ?? "")/"
    }

    private static func semanticKey(_ target: Destination) -> String {
        if target.kind == .window, let bundle = target.bundleID { return "window_class:" + bundle }
        guard let locator = target.locator, let parts = URLComponents(string: locator) else { return target.id }
        let value: [Any]
        let kind: String
        if target.kind == .page, let scheme = parts.scheme?.lowercased(),
           ["http", "https"].contains(scheme), let host = parts.host?.lowercased(), !host.isEmpty,
           parts.user == nil, parts.password == nil, !parts.percentEncodedPath.contains("\\") {
            let raw = parts.percentEncodedPath
            guard raw.isEmpty || raw == "/" || (raw.hasPrefix("/") && !raw.contains("//")) else {
                return target.id
            }
            var segments = parts.percentEncodedPath.isEmpty || parts.percentEncodedPath == "/" ? [] :
                String(parts.percentEncodedPath.dropFirst()).split(separator: "/", omittingEmptySubsequences: false).map(String.init)
            if segments.last == "" { segments.removeLast() }
            guard segments.allSatisfy({ segment in
                guard !segment.isEmpty, segment != ".", segment != "..",
                      let decoded = segment.removingPercentEncoding else { return false }
                return decoded != "." && decoded != ".." && !decoded.contains("/") && !decoded.contains("\\")
            }) else { return target.id }
            value = ["page", [scheme, host.trimmingCharacters(in: CharacterSet(charactersIn: ".")),
                               parts.port == (scheme == "http" ? 80 : 443) ? NSNull() : (parts.port.map { $0 as Any } ?? NSNull())],
                     segments, parts.percentEncodedQuery ?? "", parts.percentEncodedFragment ?? ""]
            kind = "page"
        } else if (target.kind == .directory || target.kind == .file),
                  parts.scheme?.lowercased() == "file", parts.query == nil, parts.fragment == nil,
                  parts.user == nil, parts.password == nil, parts.port == nil,
                  parts.percentEncodedPath.hasPrefix("/"),
                  !parts.percentEncodedPath.lowercased().contains("%2f"),
                  !parts.percentEncodedPath.contains("\\"),
                  let decoded = parts.percentEncodedPath.removingPercentEncoding,
                  !decoded.contains("\0") {
            let normalized = (decoded as NSString).standardizingPath
            guard normalized.hasPrefix("/") else { return target.id }
            let host = (parts.host?.lowercased() ?? "").trimmingCharacters(in: CharacterSet(charactersIn: "."))
            value = [target.kind.rawValue, [host == "localhost" ? "" : host, normalized]]
            kind = target.kind.rawValue
        } else { return target.id }
        guard let json = try? JSONSerialization.data(withJSONObject: value, options: [.withoutEscapingSlashes]),
              let normalized = String(data: json, encoding: .utf8) else { return target.id }
        let digest = OnlinePathMLPScorer.lowercaseHex(SHA256.hash(data: Data(normalized.utf8)))
        return "layer:\(kind):\(digest)"
    }
}
