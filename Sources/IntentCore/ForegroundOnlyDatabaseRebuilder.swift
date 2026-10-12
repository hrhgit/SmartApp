import Foundation

/// An auditable result for rebuilding the active database at the foreground-only privacy boundary.
public struct ForegroundOnlyRebuildReport: Codable, Equatable, Sendable {
    public var migrationVersion = ForegroundOnlyDatabaseRebuilder.migrationVersion
    public var sourceEvents = 0
    public var sourceSessions = 0
    public var retainedEvents = 0
    public var retainedSessions = 0
    public var discardedEvents = 0
    public var discardedSessions = 0
    public var discardedByKind: [String: Int] = [:]
    public var discardedSessionReasons: [String: Int] = [:]
    public var removedFields: [String: Int] = [:]
    public var databaseBytes: Int64 = 0
    public init() {}
}

/// Rebuilds a new database instead of editing historical rows in place. Only complete sessions
/// that declared the current foreground-only policy can cross this boundary.
public enum ForegroundOnlyDatabaseRebuilder {
    public static let migrationVersion = "foreground-only-redaction-v1"

    private static let prohibitedKinds: Set<EventKind> = [.inventoryChanged, .menuInvoked, .recommendationShown]
    private static let standardKinds: Set<EventKind> = [
        .sessionStart, .sessionEnd, .heartbeat, .captureGap, .observationScope, .capabilities,
        .appActivated, .contextChanged, .targetUpdated, .sourceStatus,
        .destinationSettled, .idleChanged, .recommendationFeedback
    ]
    private static let forbiddenDataKeys: Set<String> = [
        "entrances", "browserState", "windowState", "desktopInventoryEventIDs",
        "inventoryEventIDs", "candidateTargetIDs", "items", "snapshotEventID",
        "expectedSources", "registryEventID", "backgroundSourcePollSeconds"
    ]
    private static let standardStartKeys: Set<String> = [
        "appVersion", "stateCoverageVersion", "inactivityPolicyVersion", "timezone",
        "utcOffsetSeconds", "targetEligibilityPolicyVersion", "observationPolicy",
        "webPolicy", "localPolicy", "browserObservation", "browserPollSeconds",
        "browserStableDelayMs", "finderForegroundPollSeconds", "qspaceForegroundPollSeconds",
        "backgroundStateCollection", "windowForegroundPollSeconds"
    ]

    public static func rebuild(source: URL, destination: URL) throws -> ForegroundOnlyRebuildReport {
        let sourcePath = source.resolvingSymlinksInPath().standardizedFileURL.path
        let destinationPath = destination.resolvingSymlinksInPath().standardizedFileURL.path
        guard sourcePath != destinationPath else { throw StoreError("Foreground-only rebuild requires a different output database") }
        guard FileManager.default.fileExists(atPath: source.path) else { throw StoreError("Source database does not exist") }
        for suffix in ["", "-wal", "-shm"] where FileManager.default.fileExists(atPath: destination.path + suffix) {
            throw StoreError("Foreground-only rebuild output already exists")
        }

        let reader = try EventStore(url: source, readOnly: true)
        let sourceEvents = try reader.recommendationHistory(seedThroughEventID: nil)
        var sessions: [String: [BehaviorEvent]] = [:]
        var sessionOrder: [String] = []
        for event in sourceEvents {
            if sessions[event.sessionID] == nil { sessionOrder.append(event.sessionID) }
            sessions[event.sessionID, default: []].append(event)
        }

        var report = ForegroundOnlyRebuildReport()
        report.sourceEvents = sourceEvents.count
        report.sourceSessions = sessionOrder.count
        var retained: [BehaviorEvent] = []

        for id in sessionOrder {
            guard let events = sessions[id], let start = events.first else { continue }
            guard events.first?.kind == .sessionStart,
                  events.filter({ $0.kind == .sessionStart }).count == 1 else {
                discardSession(events, reason: "missing_or_duplicate_session_start", report: &report)
                continue
            }
            guard isForegroundOnlyStart(start) else {
                discardSession(events, reason: "legacy_or_non_foreground_policy", report: &report)
                continue
            }
            guard events.last?.kind == .sessionEnd,
                  events.filter({ $0.kind == .sessionEnd }).count == 1 else {
                discardSession(events, reason: "incomplete_session", report: &report)
                continue
            }

            var normalized: [BehaviorEvent] = []
            for event in events {
                if event.kind == .sessionStart {
                    normalized.append(try canonicalStart(event, report: &report))
                } else if let canonical = try canonicalEvent(event, report: &report) {
                    normalized.append(canonical)
                }
            }
            guard normalized.first?.kind == .sessionStart, normalized.last?.kind == .sessionEnd else {
                throw StoreError("A foreground-only session lost its required boundary during rebuild")
            }
            try validate(normalized)
            retained.append(contentsOf: normalized)
            report.retainedSessions += 1
        }

        report.retainedEvents = retained.count
        guard !FileManager.default.fileExists(atPath: destination.path) else { throw StoreError("Foreground-only rebuild output already exists") }
        let destinationStore = try EventStore(url: destination)
        for index in stride(from: 0, to: retained.count, by: 128) {
            try destinationStore.append(Array(retained[index..<min(index + 128, retained.count)]))
        }
        destinationStore.flush()

        let output = try EventStore(url: destination, readOnly: true)
        let outputEvents = try output.recommendationHistory(seedThroughEventID: nil)
        guard outputEvents == retained else { throw StoreError("Foreground-only rebuild verification did not match its selected events") }
        try validate(outputEvents)
        report.databaseBytes = EventStore.databaseBytes(at: destination)
        return report
    }

    private static func isForegroundOnlyStart(_ event: BehaviorEvent) -> Bool {
        event.schemaVersion == 2 && event.source == "lifecycle" &&
            event.data["stateCoverageVersion"]?.string == StateCoverage.foregroundOnlyVersion &&
            event.data["targetEligibilityPolicyVersion"]?.string == TargetEligibilityPolicy.version &&
            event.data["observationPolicy"]?.string == "foreground_destinations_only_no_background_state" &&
            event.data["backgroundStateCollection"]?.bool == false
    }

    private static func canonicalStart(_ event: BehaviorEvent,
                                       report: inout ForegroundOnlyRebuildReport) throws -> BehaviorEvent {
        guard event.schemaVersion == 2, !containsForbiddenData(event.data) else {
            throw StoreError("A session declared foreground-only but contains prohibited historical state")
        }
        var canonical = event
        for key in event.data.keys where !standardStartKeys.contains(key) && key != "clipboardPolicy" {
            throw StoreError("A session declared foreground-only but contains an unknown legacy start field")
        }
        if canonical.data.removeValue(forKey: "clipboardPolicy") != nil {
            report.removedFields["clipboardPolicy", default: 0] += 1
        }
        return canonical
    }

    private static func canonicalEvent(_ event: BehaviorEvent,
                                       report: inout ForegroundOnlyRebuildReport) throws -> BehaviorEvent? {
        guard event.schemaVersion == 2 else {
            throw StoreError("A session declared foreground-only but contains an unsupported event schema")
        }
        guard !containsForbiddenData(event.data), !prohibitedKinds.contains(event.kind) else {
            throw StoreError("A session declared foreground-only but contains prohibited historical state")
        }
        guard standardKinds.contains(event.kind) else {
            report.discardedEvents += 1
            report.discardedByKind[event.kind.rawValue, default: 0] += 1
            return nil
        }
        return event
    }

    private static func containsForbiddenData(_ value: [String: JSONValue]) -> Bool {
        value.contains { key, nested in
            forbiddenDataKeys.contains(key) || containsForbiddenData(nested)
        }
    }

    private static func containsForbiddenData(_ value: JSONValue) -> Bool {
        switch value {
        case .array(let values): return values.contains(where: containsForbiddenData)
        case .object(let values): return containsForbiddenData(values)
        default: return false
        }
    }

    private static func discardSession(_ events: [BehaviorEvent], reason: String,
                                       report: inout ForegroundOnlyRebuildReport) {
        report.discardedSessions += 1
        report.discardedSessionReasons[reason, default: 0] += 1
        report.discardedEvents += events.count
        for event in events { report.discardedByKind[event.kind.rawValue, default: 0] += 1 }
    }

    private static func validate(_ events: [BehaviorEvent]) throws {
        var gate = TargetEventGate()
        for event in events {
            guard event.schemaVersion == 2, standardKinds.contains(event.kind),
                  !containsForbiddenData(event.data), gate.accept(event) else {
                throw StoreError("A selected event does not satisfy the foreground-only target boundary")
            }
        }
    }
}
