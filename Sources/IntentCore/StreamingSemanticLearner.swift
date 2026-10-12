import Foundation

/// Conservative online form of the research hard_kind trajectory supervision.
/// Only completed 30s of active observation train. A boundary never proves a short dwell.
/// Observed recommendation outcomes train once, without separately counting UI feedback.
public struct StreamingSemanticLearner {
    private struct Episode {
        var event: BehaviorEvent
        var initial: Destination
        var target: Destination
        var source: String
        var entrance: String?
        var refinements: [(Double, Destination)] = []
        var end: Double?
        var next: String?
        var boundary = false
        var leftCensored: Bool
        var contextIDs: Set<String>
    }
    private struct Input {
        var target: Destination
        var episode: String
        var enabled: Bool
    }
    private var session: String?
    private var builder = NavigationSampleBuilder()
    private var catalog: [String: Destination] = [:]
    private var episodes: [String: Episode] = [:]
    private var anchors: [String: Input] = [:]
    private var active: String?
    private var supportedSession = false
    private var inactivity = KnownInactivityTracker()
    public var allowNewAnchors = true
    public var excludedSessions: Set<String> = []
    public init() {}

    public mutating func consume(_ rawEvent: BehaviorEvent) -> [SemanticLearningRecord] {
        var event = rawEvent
        if session != event.sessionID {
            session = event.sessionID; builder = NavigationSampleBuilder()
            catalog = [:]; episodes = [:]; anchors = [:]; active = nil
            supportedSession = false
        }
        if event.kind == .sessionStart {
            supportedSession = event.data["targetEligibilityPolicyVersion"]?.string.map(TargetEligibilityPolicy.supportedVersions.contains) == true
        }
        guard supportedSession, !excludedSessions.contains(event.sessionID) else { return [] }
        if event.kind == .targetUpdated, let value = event.data["target"], let target = try? value.decoded(Destination.self) {
            if TargetEligibilityPolicy.validates(target) { catalog[target.id] = target }
            else {
                catalog[target.id] = nil
                if let id = active, let current = episodes[id]?.target,
                   current.id == target.id || target.kind == .app && current.bundleID == target.bundleID {
                    close(event, boundary: true)
                }
            }
        }
        let continuity = inactivity.consume(rawEvent)
        let completed = builder.consume(rawEvent)
        event.monotonicMs = inactivity.activeMs
        if continuity == .suspended { return [] }
        if continuity == .boundary || continuity == .filtered {
            close(event, boundary: true)
        } else if event.kind == .sourceStatus,
                  ["unavailable", "stale", "initializing"].contains(event.data["observationStatus"]?.string ?? ""),
                  let id = active, let visit = episodes[id],
                  event.source == visit.source || Self.family(event.source) != "other" && Self.family(event.source) == Self.family(visit.source) {
            close(event, boundary: true)
        } else if event.kind == .contextChanged {
            observe(event)
        }
        var records: [SemanticLearningRecord] = []
        for sample in completed {
            defer { anchors[sample.anchorEventID] = nil }
            let strict = ["complete", "foreground_only"].contains(sample.inputCoverage?.status ?? "") &&
                sample.outcomeCoverage == "observed"
            let legacy = sample.inputCoverage?.status == "unknown" && sample.outcomeCoverage == "initial_coverage_unverified"
            guard sample.status == "observed_horizon", strict || legacy,
                  let input = anchors[sample.anchorEventID], input.enabled,
                  let firstRaw = sample.outcomes.first, firstRaw.targetID != input.target.id else { continue }
            if let record = label(sample, input: input) { records.append(record) }
        }
        // Keep active episodes and pending anchors, never an unbounded raw history.
        let needed = Set(anchors.values.map(\.episode))
        episodes = episodes.filter { key, value in key == active || needed.contains(key) || (value.end ?? event.monotonicMs) >= event.monotonicMs - 60_000 }
        return records
    }
    private mutating func close(_ event: BehaviorEvent, boundary: Bool, next: String? = nil) {
        if let id = active { episodes[id]?.end = event.monotonicMs; episodes[id]?.boundary = boundary; episodes[id]?.next = next }
        active = nil
    }
    private mutating func observe(_ event: BehaviorEvent) {
        guard let targetID = event.data["targetID"]?.string, var target = catalog[targetID] else { close(event, boundary: true); return }
        target.bundleID = event.app?.bundleID ?? target.bundleID
        let resolution = event.data["reason"]?.string == "resolution_update"
        let entrance = event.data["entranceID"]?.string
        if resolution, let id = active, var visit = episodes[id],
            event.data["resolvesContextEventID"]?.string.map(visit.contextIDs.contains) == true ||
            event.data["resolvesContextEventID"]?.string == nil && visit.target.bundleID == target.bundleID {
            if visit.target.id != target.id {
                visit.refinements.append((event.monotonicMs, target)); visit.target = target; visit.source = event.source
            }
            visit.entrance = entrance; visit.contextIDs.insert(event.id); episodes[id] = visit
            return
        }
        if resolution { close(event, boundary: true) }
        if let id = active, let visit = episodes[id], visit.target.id == target.id, visit.entrance == entrance {
            episodes[id]?.contextIDs.insert(event.id)
            anchors[event.id] = Input(target: target, episode: id, enabled: allowNewAnchors)
            return
        }
        if let id = active, let visit = episodes[id] {
            let resource = [.page, .directory, .file].contains(visit.target.kind)
            let coarse = [.app, .window].contains(target.kind)
            close(event, boundary: resource && coarse && visit.target.bundleID == target.bundleID, next: event.id)
        }
        episodes[event.id] = Episode(event: event, initial: target, target: target, source: event.source,
                                      entrance: entrance, leftCensored: resolution, contextIDs: [event.id])
        active = event.id
        if !resolution { anchors[event.id] = Input(target: target, episode: event.id, enabled: allowNewAnchors) }
    }
    private func label(_ sample: NavigationSample, input: Input) -> SemanticLearningRecord? {
        let cutoff = sample.activeInputCutoffMs ?? sample.inputCutoffMs
        let deadline = cutoff + 30_000
        guard let source = episodes[input.episode], !source.boundary, let end = source.end, end <= deadline,
              var next = source.next else { return nil }
        for step in 1...3 {
            guard let episode = episodes[next], !episode.leftCensored,
                  episode.event.monotonicMs > cutoff, episode.event.monotonicMs <= deadline else { return nil }
            var target = episode.initial
            var since = episode.event.monotonicMs
            for (time, refinement) in episode.refinements where time <= deadline { target = refinement; since = time }
            guard let action = SemanticRecommendationModel.action(target) else { return nil }
            let exit = episode.end.flatMap { $0 <= deadline ? $0 : nil }
            let completed = exit != nil && !episode.boundary
            let threshold: Double = [.app, .window].contains(target.kind) ? 5_000 : 2_000
            if !completed || exit! - since >= threshold {
                return NavigationSemantics.derive(SemanticLearningRecord(origin: SemanticRecommendationModel.origin(input.target),
                    annotation: SemanticMetadata.annotate(input.target), label: action, outcomeID: episode.event.id,
                    outcomeTimestamp: Double(episode.event.timestampMs),
                    source: sample.decisionID != nil || episode.event.decisionID != nil ? "recommendation_observed" : "natural_observed",
                    destination: SemanticMetadata.annotate(target).identity, observedStep: step,
                    inputEventID: sample.anchorEventID))
            }
            guard let following = episode.next else { return nil }
            next = following
        }
        return nil
    }
    private static func family(_ source: String) -> String {
        if source.hasPrefix("browser") { return "browser" }
        if source.hasPrefix("ax:") { return "ax" }
        if source.hasPrefix("finder:") { return "finder" }
        if source.hasPrefix("qspace:") { return "qspace" }
        return "other"
    }
}
