import Foundation

/// Coverage describes observations, never an assertion about the entire physical desktop.
public struct SourceCoverage: Codable, Equatable, Sendable {
    public var source: String
    public var status: String
    public var reason: String?
    public var scope: String?
    public var snapshotEventID: String?
    public var statusEventID: String?
    public var observedAtMs: Double?
    public var availableAtMs: Double?
    public var confirmedAtMs: Double?
    public var maxAgeMs: Double?
}

public struct StateCoverage: Codable, Equatable, Sendable {
    public static let inventoryVersion = "state-coverage-v1"
    public static let foregroundOnlyVersion = "foreground-coverage-v2"
    public var version = Self.inventoryVersion
    public var status: String
    public var registryEventID: String?
    public var cutoffSequence: Int?
    public var cutoffMs: Double
    public var boundaryReason: String?
    public var sources: [SourceCoverage]
    public var missingSources: [String]
}

/// Nonexclusive labels: the leaf's identity is separate from its live container/entrance.
public struct ObservedTargetHierarchy: Codable, Equatable, Sendable {
    public var appTargetID: String?
    public var containerTargetID: String?
    public var destinationTargetID: String
    public var entranceID: String?
    public var kind: TargetKind?
    public var locator: String?
}

/// The same as-of reducer is used by live capture and historical exports. It never backfills.
public struct StateCoverageTracker: Sendable {
    private var session: String?
    private var registryEventID: String?
    private var expected: Set<String> = []
    private var sources: [String: SourceCoverage] = [:]
    private var targets: [String: Destination] = [:]
    private var entrances: [String: [TargetEntrance]] = [:]
    private var boundaryReason: String?
    private var foregroundOnly = false
    public init() {}

    public mutating func consume(_ event: BehaviorEvent) {
        if session != event.sessionID || event.kind == .sessionStart {
            self = Self(); session = event.sessionID; boundaryReason = "initial_sync_pending"
            foregroundOnly = event.data["stateCoverageVersion"]?.string == StateCoverage.foregroundOnlyVersion
        }
        if event.kind == .captureGap || event.kind == .observationScope || event.kind == .sessionEnd {
            boundaryReason = event.data["reason"]?.string ?? event.kind.rawValue
            entrances = [:]
            for key in Array(sources.keys) {
                sources[key]?.status = "initializing"; sources[key]?.reason = boundaryReason
                sources[key]?.snapshotEventID = nil; sources[key]?.confirmedAtMs = nil
                sources[key]?.statusEventID = event.id
            }
            return
        }
        if event.kind == .targetUpdated, let target = try? event.data["target"]?.decoded(Destination.self) {
            targets[target.id] = target
        }
        if foregroundOnly {
            if event.kind == .sourceStatus,
               event.data["coverageVersion"]?.string == StateCoverage.foregroundOnlyVersion,
               let status = event.data["observationStatus"]?.string {
                var value = sources[event.source] ?? SourceCoverage(source: event.source, status: "unknown")
                value.status = status; value.reason = event.data["reason"]?.string
                value.statusEventID = event.id; value.observedAtMs = event.monotonicMs
                value.availableAtMs = event.monotonicMs; value.confirmedAtMs = event.monotonicMs
                sources[event.source] = value
            }
            return
        }
        if event.kind == .sourceStatus, event.source == "source_registry",
           event.data["coverageVersion"]?.string == StateCoverage.inventoryVersion,
           let declarations = event.data["expectedSources"]?.array {
            registryEventID = event.id; expected = []
            for declaration in declarations {
                guard let item = declaration.object, let key = item["source"]?.string else { continue }
                expected.insert(key)
                var value = sources[key] ?? SourceCoverage(source: key, status: "initializing")
                value.scope = item["scope"]?.string; value.maxAgeMs = item["maxAgeMs"]?.number
                sources[key] = value
            }
        }
        if event.kind == .inventoryChanged {
            var value = sources[event.source] ?? SourceCoverage(source: event.source, status: "unknown")
            let entries = (try? event.data["entrances"]?.decoded([TargetEntrance].self)) ?? []
            let explicit = event.data["observationStatus"]?.string == "ready"
            value.status = explicit ? "ready" : (entries.isEmpty ? "unknown" : "legacy_observed")
            value.reason = explicit ? nil : (entries.isEmpty ? "legacy_empty_unverified" : "legacy_coverage_unverified")
            value.snapshotEventID = event.id; value.statusEventID = event.id
            value.observedAtMs = event.data["observationStartedMonotonicMs"]?.number ?? event.monotonicMs
            value.availableAtMs = event.monotonicMs; value.confirmedAtMs = event.monotonicMs
            sources[event.source] = value; entrances[event.source] = entries
        }
        if event.kind == .sourceStatus, event.data["coverageVersion"]?.string == StateCoverage.inventoryVersion,
           let status = event.data["observationStatus"]?.string {
            var value = sources[event.source] ?? SourceCoverage(source: event.source, status: "unknown")
            if status == "ready" {
                // A successful check can refresh only the exact state snapshot it checked.
                guard value.status == "ready", let snapshot = event.data["snapshotEventID"]?.string,
                      snapshot == value.snapshotEventID else { return }
                value.confirmedAtMs = event.monotonicMs
            } else if status != "not_applicable" {
                entrances[event.source] = nil
            } else { entrances[event.source] = nil; value.snapshotEventID = nil }
            value.status = status; value.reason = event.data["reason"]?.string
            value.statusEventID = event.id; sources[event.source] = value
        }
    }

    public func snapshot(at time: Double, sequence: Int?) -> StateCoverage {
        if foregroundOnly {
            let values = sources.keys.sorted().compactMap { sources[$0] }
            return StateCoverage(version: StateCoverage.foregroundOnlyVersion, status: "foreground_only",
                registryEventID: nil, cutoffSequence: sequence, cutoffMs: time, boundaryReason: boundaryReason,
                sources: values, missingSources: [])
        }
        let keys = expected.union(sources.keys)
        let values = keys.sorted().compactMap { key -> SourceCoverage? in
            guard var value = sources[key] else { return nil }
            if value.status == "ready", let age = value.maxAgeMs,
               time - (value.confirmedAtMs ?? -.infinity) > age {
                value.status = "stale"; value.reason = "observation_expired"
            }
            return value
        }
        let missing = values.filter { expected.contains($0.source) && !["ready", "not_applicable"].contains($0.status) }.map(\.source)
        let status = registryEventID == nil ? "unknown" : (missing.isEmpty ? "complete" : "partial")
        return StateCoverage(status: status, registryEventID: registryEventID, cutoffSequence: sequence,
            cutoffMs: time, boundaryReason: status == "complete" ? nil : boundaryReason,
            sources: values, missingSources: missing)
    }

    public func hierarchy(for event: BehaviorEvent) -> ObservedTargetHierarchy? {
        guard let id = event.data["targetID"]?.string else { return nil }
        let entryID = event.data["entranceID"]?.string
        let direct: TargetEntrance?
        if let value = event.data["foregroundEntrance"], let decoded = try? value.decoded(TargetEntrance.self),
           decoded.id == entryID, decoded.targetID == id {
            direct = decoded
        } else { direct = nil }
        let entry = direct ?? entrances.values.joined().first { $0.id == entryID && $0.targetID == id }
        let target = targets[id]
        let bundle = entry?.bundleID ?? event.app?.bundleID ?? target?.bundleID
        return ObservedTargetHierarchy(appTargetID: bundle.map { "app:" + $0 },
            containerTargetID: entry?.containerID ?? (target?.kind == .window ? id : nil),
            destinationTargetID: id, entranceID: entryID, kind: target?.kind, locator: target?.locator)
    }
}
