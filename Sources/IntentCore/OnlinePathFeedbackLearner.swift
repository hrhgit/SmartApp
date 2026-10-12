import Foundation

/// A single observed arrival supplies one learning result. The earlier input is
/// identified here but remains frozen by the caller at its original cutoff.
public struct OnlinePathFeedback: Equatable, Sendable {
    public enum Evidence: String, Sendable {
        case completeObservation
        case confirmedFirstStep
        case legacyShortGap
        case excludedScope
    }

    public let inputEventID: String
    public let outcomeEventID: String
    public let evidence: Evidence

    public init(inputEventID: String, outcomeEventID: String, evidence: Evidence) {
        self.inputEventID = inputEventID
        self.outcomeEventID = outcomeEventID
        self.evidence = evidence
    }
}

/// Incremental form of the private full-history training rules. Complete
/// samples use their first confirmed outcome; an incomplete sample can retain
/// that same observed first step. A strict paired interruption can additionally
/// relate two confirmed endpoints without inventing activity in the interval.
public struct OnlinePathFeedbackLearner {
    private struct Confirmed {
        let event: BehaviorEvent
        let destination: Destination
        let actionID: String
        let ownerBundleID: String?
    }

    private struct Bridge {
        enum Kind { case legacyShortGap, excludedScope }
        let kind: Kind
        let before: Confirmed
        let begin: BehaviorEvent
        let markerID: String?
        let category: String?
        var end: BehaviorEvent?
    }

    private var sessionID: String?
    private var supportedSession = false
    private var builder = NavigationSampleBuilder()
    private var metadata: [String: Destination] = [:]
    private var contexts: [String: Confirmed] = [:]
    private var lastContextWithoutBoundary: Confirmed?
    private var bridge: Bridge?
    private var emittedOutcomeIDs = Set<String>()
    private var emittedOutcomeOrder: [String] = []
    var recentOutcomeCount: Int { emittedOutcomeIDs.count }
    /// Inputs whose complete/incomplete sample and any allowed bridge have
    /// finished can be released by the caller without losing a later label.
    public private(set) var discardableInputEventIDs: [String] = []
    public var excludedSessions: Set<String> = []
    /// A pending sample, paired bridge, or latest confirmed endpoint can
    /// still consume an earlier frozen input. Keep exactly those dependencies.
    public var pendingInputEventIDs: Set<String> {
        var ids = builder.pendingAnchorEventIDs
        if let bridge { ids.insert(bridge.before.event.id) }
        if let lastContextWithoutBoundary { ids.insert(lastContextWithoutBoundary.event.id) }
        return ids
    }
    public var requiredConfirmedEventIDs: Set<String> {
        pendingInputEventIDs.union(builder.pendingFirstStepEventIDs)
    }
    var retainedContextCount: Int { contexts.count }

    public init() {}

    public mutating func consume(_ event: BehaviorEvent) -> [OnlinePathFeedback] {
        discardableInputEventIDs = []
        var carried: [OnlinePathFeedback] = []
        var carriedInputs: [String] = []
        if sessionID != event.sessionID || event.kind == .sessionStart {
            if let priorBridge = bridge { carriedInputs.append(priorBridge.before.event.id) }
            if supportedSession, let previous = sessionID,
               !excludedSessions.contains(previous) {
                let unfinished = builder.finish(reason: "session_changed")
                carried = unfinished.compactMap { feedback(from: $0) }
                carriedInputs += unfinished.map(\.anchorEventID)
            }
            sessionID = event.sessionID
            supportedSession = event.kind == .sessionStart &&
                event.data["targetEligibilityPolicyVersion"]?.string
                    .map(TargetEligibilityPolicy.supportedVersions.contains) == true &&
                event.data["stateCoverageVersion"]?.string == StateCoverage.foregroundOnlyVersion
            builder = NavigationSampleBuilder()
            lastContextWithoutBoundary = nil
            bridge = nil
            contexts.removeAll(keepingCapacity: true)
        }
        guard supportedSession, !excludedSessions.contains(event.sessionID) else {
            discardableInputEventIDs = carriedInputs
            return unique(carried)
        }

        if event.kind == .targetUpdated,
           let raw = event.data["target"],
           let target = try? raw.decoded(Destination.self),
           TargetEligibilityPolicy.validates(target) {
            metadata[target.id] = target
        }

        let confirmed = confirmedContext(event)
        if let confirmed {
            contexts[event.id] = confirmed
        }

        let samples = builder.consume(event)
        var output = carried + samples.compactMap { feedback(from: $0) }
        let priorBridgeInput = bridge?.before.event.id
        if let joined = bridgeFeedback(event, confirmed: confirmed) { output.append(joined) }
        let protected = pendingInputEventIDs
        var released = Set(carriedInputs + samples.map(\.anchorEventID))
        if let priorBridgeInput { released.insert(priorBridgeInput) }
        let required = requiredConfirmedEventIDs
        for id in contexts.keys where !required.contains(id) {
            contexts[id] = nil
            released.insert(id)
        }
        released.subtract(protected)
        discardableInputEventIDs = released.sorted()
        return unique(output)
    }

    private mutating func unique(_ feedback: [OnlinePathFeedback]) -> [OnlinePathFeedback] {
        feedback.filter { item in
            guard emittedOutcomeIDs.insert(item.outcomeEventID).inserted else { return false }
            emittedOutcomeOrder.append(item.outcomeEventID)
            if emittedOutcomeOrder.count > 8_192 {
                let old = emittedOutcomeOrder.removeFirst()
                emittedOutcomeIDs.remove(old)
            }
            return true
        }
    }

    private func confirmedContext(_ event: BehaviorEvent) -> Confirmed? {
        guard event.kind == .contextChanged,
              let coverage = event.data["inputCoverage"]?.object,
              coverage["version"]?.string == StateCoverage.foregroundOnlyVersion,
              coverage["status"]?.string == "foreground_only",
              let reason = event.data["reason"]?.string, !reason.isEmpty,
              !["resolution_update", "browser_history", "browser_resync"].contains(reason),
              let targetID = event.data["targetID"]?.string,
              let target = metadata[targetID], TargetEligibilityPolicy.validates(target),
              let actionID = Self.actionID(target) else { return nil }
        return Confirmed(event: event, destination: target, actionID: actionID,
                         ownerBundleID: event.app?.bundleID ?? target.bundleID)
    }

    private func feedback(from sample: NavigationSample) -> OnlinePathFeedback? {
        guard ["observed_horizon", "incomplete"].contains(sample.status),
              sample.schemaVersion == 4,
              sample.labelRuleVersion == "observed-foreground-sequence-30s-v4",
              sample.inventoryEventIDs.isEmpty,
              sample.inputCoverage?.version == StateCoverage.foregroundOnlyVersion,
              sample.inputCoverage?.status == "foreground_only",
              sample.status != "observed_horizon" || sample.outcomeCoverage == "observed",
              let first = sample.outcomes.first,
              let before = contexts[sample.anchorEventID],
              let after = contexts[first.eventID],
              first.targetID == after.destination.id,
              sample.sessionID == before.event.sessionID,
              sample.sessionID == after.event.sessionID,
              sample.inputThroughSequence == before.event.sequence,
              sample.inputCutoffMs == before.event.monotonicMs,
              Self.increasing(before.event, after.event),
              Self.elapsed(before.event, after.event, maximum: nil) != nil,
              Self.isNavigation(before, after) else { return nil }
        return OnlinePathFeedback(inputEventID: before.event.id,
                                  outcomeEventID: after.event.id,
                                  evidence: sample.status == "observed_horizon" ?
                                    .completeObservation : .confirmedFirstStep)
    }

    private mutating func bridgeFeedback(_ event: BehaviorEvent,
                                         confirmed: Confirmed?) -> OnlinePathFeedback? {
        if event.kind == .sessionStart || event.kind == .sessionEnd {
            bridge = nil
            lastContextWithoutBoundary = nil
            return nil
        }

        if event.kind == .captureGap || event.kind == .observationScope {
            defer { lastContextWithoutBoundary = nil }
            if var pending = bridge, pending.end == nil,
               Self.validEnd(event, for: pending) {
                pending.end = event
                bridge = pending
                return nil
            }
            bridge = nil
            if let before = lastContextWithoutBoundary,
               let beginning = Self.bridgeBeginning(event, before: before) {
                bridge = beginning
            }
            return nil
        }

        guard event.kind == .contextChanged else { return nil }
        defer { lastContextWithoutBoundary = confirmed }
        guard let pending = bridge else { return nil }
        bridge = nil
        guard let after = confirmed, let end = pending.end,
              Self.increasing(pending.before.event, pending.begin, end, after.event),
              let duration = Self.elapsed(pending.begin, end,
                                          maximum: pending.kind == .legacyShortGap ? 1_000 : nil),
              Self.elapsed(end, after.event,
                           maximum: pending.kind == .legacyShortGap ? 1_000 : nil) != nil,
              Self.elapsed(pending.before.event, pending.begin, maximum: nil) != nil,
              let total = Self.elapsed(pending.before.event, after.event, maximum: nil),
              total - duration <= 30_000,
              pending.before.destination.id != after.destination.id,
              Self.isNavigation(pending.before, after) else { return nil }
        return OnlinePathFeedback(inputEventID: pending.before.event.id,
                                  outcomeEventID: after.event.id,
                                  evidence: pending.kind == .legacyShortGap ?
                                    .legacyShortGap : .excludedScope)
    }

    private static func bridgeBeginning(_ event: BehaviorEvent, before: Confirmed) -> Bridge? {
        guard event.sessionID == before.event.sessionID else { return nil }
        if event.kind == .captureGap,
           event.data["reason"]?.string == "observation_suspended",
           ["pauseID", "causeID", "pauseVersion", "causeCategory",
            "interruptionID", "interruptionVersion"].allSatisfy({ event.data[$0] == nil }) {
            return Bridge(kind: .legacyShortGap, before: before, begin: event,
                          markerID: nil, category: nil)
        }
        if event.kind == .observationScope,
           validScope(event, phase: "begin"),
           let id = event.data["scopeID"]?.string,
           let category = event.data["category"]?.string {
            return Bridge(kind: .excludedScope, before: before, begin: event,
                          markerID: id, category: category)
        }
        return nil
    }

    private static func validEnd(_ event: BehaviorEvent, for bridge: Bridge) -> Bool {
        guard event.sessionID == bridge.begin.sessionID else { return false }
        switch bridge.kind {
        case .legacyShortGap:
            return event.kind == .captureGap &&
                event.data["reason"]?.string == "observation_resumed"
        case .excludedScope:
            return validScope(event, phase: "end") &&
                event.data["scopeID"]?.string == bridge.markerID &&
                event.data["category"]?.string == bridge.category
        }
    }

    private static func validScope(_ event: BehaviorEvent, phase: String) -> Bool {
        guard event.kind == .observationScope,
              event.source == "lifecycle", event.app == nil,
              Set(event.data.keys) == ["scopeVersion", "scopeID", "phase", "category"],
              event.data["scopeVersion"]?.string == "excluded-foreground-v1",
              event.data["phase"]?.string == phase,
              let id = event.data["scopeID"]?.string, UUID(uuidString: id) != nil,
              let category = event.data["category"]?.string else { return false }
        return ["excluded_app", "excluded_page", "private_browser", "secure_input"].contains(category)
    }

    private static func actionID(_ target: Destination) -> String? {
        if target.kind == .window, let bundle = target.bundleID { return "app:" + bundle }
        return [.app, .page, .directory, .file].contains(target.kind) ? target.id : nil
    }

    private static func isNavigation(_ before: Confirmed, _ after: Confirmed) -> Bool {
        guard before.actionID != after.actionID else { return false }
        return before.ownerBundleID.map { after.actionID != "app:" + $0 } ?? true
    }

    private static func elapsed(_ start: BehaviorEvent, _ end: BehaviorEvent,
                                maximum: Double?) -> Double? {
        let monotonic = end.monotonicMs - start.monotonicMs
        let wall = Double(end.timestampMs - start.timestampMs)
        guard monotonic.isFinite, wall.isFinite, monotonic >= 0, wall >= 0,
              maximum.map({ monotonic <= $0 && wall <= $0 }) ?? true else { return nil }
        return monotonic
    }

    private static func increasing(_ events: BehaviorEvent...) -> Bool {
        let numbers = events.compactMap(\.sequence)
        return numbers.count == events.count &&
            zip(numbers, numbers.dropFirst()).allSatisfy { $0 < $1 }
    }
}
