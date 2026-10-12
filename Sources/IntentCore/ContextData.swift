import Foundation
import CryptoKit

public enum TargetKind: String, Codable, Sendable { case app, window, page, directory, file }

public struct Destination: Codable, Equatable, Identifiable, Sendable {
    public var id: String
    public var kind: TargetKind
    public var name: String
    public var bundleID: String?
    public var origin: String?
    public var locator: String?
    public var parentID: String?
    public var fileType: String?
    public var admission: TargetAdmission?
    public init(id: String, kind: TargetKind, name: String, bundleID: String? = nil,
                origin: String? = nil, locator: String? = nil, parentID: String? = nil,
                fileType: String? = nil, admission: TargetAdmission? = nil) {
        self.id = id; self.kind = kind; self.name = name; self.bundleID = bundleID
        self.origin = origin; self.locator = locator; self.parentID = parentID; self.fileType = fileType
        self.admission = admission
    }
}

public struct TargetEntrance: Codable, Equatable, Identifiable, Sendable {
    public var id: String
    public var targetID: String
    public var bundleID: String
    public var windowID: String?
    public var tabID: String?
    public var paneID: String?
    public var kind: String
    public var containerID: String?
    public init(id: String, targetID: String, bundleID: String, windowID: String? = nil,
                tabID: String? = nil, paneID: String? = nil, kind: String, containerID: String? = nil) {
        self.id = id; self.targetID = targetID; self.bundleID = bundleID
        self.windowID = windowID; self.tabID = tabID; self.paneID = paneID; self.kind = kind
        self.containerID = containerID
    }
}

/// The secret remains local. IDs are pseudonyms, not a claim that behavioral logs are anonymous.
public struct TargetIdentity: Sendable {
    private let key: Data
    public init(key: Data) { self.key = key }
    public init(keyURL: URL) throws {
        if FileManager.default.fileExists(atPath: keyURL.path) {
            let value = try Data(contentsOf: keyURL)
            guard value.count == 32 else { throw StoreError("Invalid local identity key") }
            key = value
        } else {
            let generated = SymmetricKey(size: .bits256).withUnsafeBytes { Data($0) }
            try FileManager.default.createDirectory(at: keyURL.deletingLastPathComponent(),
                withIntermediateDirectories: true, attributes: [.posixPermissions: 0o700])
            try generated.write(to: keyURL, options: .atomic)
            try FileManager.default.setAttributes([.posixPermissions: 0o600], ofItemAtPath: keyURL.path)
            key = generated
        }
    }
    public func id(_ namespace: String, _ value: String) -> String {
        let digest = HMAC<SHA256>.authenticationCode(for: Data((namespace + "\n" + value).utf8),
                                                     using: SymmetricKey(data: key))
        return namespace + ":" + digest.map { String(format: "%02x", $0) }.joined()
    }
    public func page(_ rawURL: String) -> Destination? {
        guard var parts = URLComponents(string: rawURL),
              let scheme = parts.scheme?.lowercased(), ["http", "https"].contains(scheme),
              let host = parts.host?.lowercased(), !host.isEmpty else { return nil }
        parts.user = nil; parts.password = nil; parts.scheme = scheme; parts.host = host
        if (scheme == "http" && parts.port == 80) || (scheme == "https" && parts.port == 443) {
            parts.port = nil
        }
        if parts.path.isEmpty { parts.path = "/" }
        guard let normalized = parts.string else { return nil }
        var root = URLComponents(); root.scheme = scheme; root.host = host; root.port = parts.port
        guard let origin = root.string else { return nil }
        let targetID = id("page", normalized)
        return Destination(id: targetID, kind: .page, name: normalized,
            origin: origin, locator: normalized)
    }
    public func local(_ url: URL, directory: Bool) -> Destination {
        let location = url.standardizedFileURL
        return Destination(id: id(directory ? "directory" : "file", location.path),
            kind: directory ? .directory : .file, name: location.lastPathComponent,
            locator: location.absoluteString,
            parentID: location.path == "/" ? nil : id("directory", location.deletingLastPathComponent().path),
            fileType: directory ? nil : location.pathExtension.lowercased())
    }
}

public struct NavigationOutcome: Codable, Equatable, Sendable {
    public var eventID: String
    public var targetID: String?
    public var entranceID: String?
    public var offsetMs: Double
    public var reason: String?
    public var hierarchy: ObservedTargetHierarchy?
    public var activeOffsetMs: Double?
}

public struct NavigationSample: Codable, Equatable, Sendable {
    public var schemaVersion = 4
    public var targetEligibilityPolicyVersion: String?
    public var id: String
    public var sessionID: String
    public var anchorEventID: String
    public var inputCutoffMs: Double
    public var inputThroughSequence: Int?
    public var recentContextEventIDs: [String]
    public var activityContextEventIDs: [String]
    public var excludedScopeEventIDs: [String]?
    public var inventoryEventIDs: [String]
    public var origin: String
    public var decisionID: String?
    public var outcomes: [NavigationOutcome] = []
    public var status = "pending"
    public var endReason: String?
    public var labelRuleVersion = "observed-foreground-sequence-30s-v4"
    public var inputCoverage: StateCoverage?
    public var hierarchy: ObservedTargetHierarchy?
    public var resolutionEventIDs: [String]?
    public var outcomeCoverage: String?
    public var activeInputCutoffMs: Double?
    public var completedAtMonotonicMs: Double?
    public var completedThroughSequence: Int?
}

/// Streaming reference samples. Future events are outcomes only; no guessed final intent or __stay__.
public struct NavigationSampleBuilder {
    public static let horizonMs: Double = 30_000
    private var session: String?
    private var history: [BehaviorEvent] = []
    private var excludedScopeEvents: [BehaviorEvent] = []
    private var inventories: [String: String] = [:]
    private var pending: [NavigationSample] = []
    private var coverage = StateCoverageTracker()
    private var inactivity = KnownInactivityTracker()
    private var foregroundOnly = false
    public init() {}
    /// Dependencies for online first-step learning. Later outcomes and stored
    /// lookback references do not need a second in-memory metadata snapshot.
    public var pendingAnchorEventIDs: Set<String> { Set(pending.map(\.anchorEventID)) }
    public var pendingFirstStepEventIDs: Set<String> {
        Set(pending.compactMap { $0.outcomes.first?.eventID })
    }
    public mutating func consume(_ event: BehaviorEvent) -> [NavigationSample] {
        let priorCoverage = coverage
        let continuity = inactivity.consume(event)
        coverage.consume(event)
        var completed: [NavigationSample] = []
        if event.sessionID != session {
            completed += finish(reason: "session_changed")
            history = []; excludedScopeEvents = []; inventories = [:]; session = event.sessionID
            foregroundOnly = event.data["stateCoverageVersion"]?.string == StateCoverage.foregroundOnlyVersion
        }
        if continuity == .suspended {
            if event.kind == .captureGap { inventories = [:] }
            if event.kind == .inventoryChanged { inventories[event.source] = event.id }
            if event.kind == .sourceStatus, let status = event.data["observationStatus"]?.string, status != "ready" {
                inventories[event.source] = nil
            }
            return completed
        }
        if continuity == .filtered {
            completed += finish(reason: "excluded_foreground")
            inventories = [:]
            if event.kind == .observationScope { excludedScopeEvents.append(event) }
            excludedScopeEvents.removeAll { event.monotonicMs - $0.monotonicMs > 3_600_000 }
            return completed
        }
        if continuity == .boundary {
            completed += finish(reason: event.data["reason"]?.string ?? event.kind.rawValue)
            history = []; excludedScopeEvents = []; inventories = [:]
            if event.kind != .contextChanged { return completed }
        }
        var remaining: [NavigationSample] = []
        for var sample in pending {
            if inactivity.activeMs >= (sample.activeInputCutoffMs ?? sample.inputCutoffMs) + Self.horizonMs {
                sample.status = "observed_horizon"
                let deadline = event.monotonicMs - (inactivity.activeMs - (sample.activeInputCutoffMs ?? sample.inputCutoffMs) - Self.horizonMs)
                if !foregroundOnly, priorCoverage.snapshot(at: deadline, sequence: event.sequence).status != "complete",
                   sample.outcomeCoverage == nil { sample.outcomeCoverage = "coverage_unverified" }
                if sample.outcomeCoverage == nil { sample.outcomeCoverage = "observed" }
                sample.completedAtMonotonicMs = event.monotonicMs
                sample.completedThroughSequence = event.sequence
                completed.append(sample)
            } else { remaining.append(sample) }
        }
        pending = remaining
        if !foregroundOnly, coverage.snapshot(at: event.monotonicMs, sequence: event.sequence).status != "complete" {
            for index in pending.indices where pending[index].outcomeCoverage == nil {
                pending[index].outcomeCoverage = "coverage_unverified"
            }
        }
        if !foregroundOnly, event.kind == .inventoryChanged { inventories[event.source] = event.id }
        if event.kind == .sourceStatus,
           let status = event.data["observationStatus"]?.string,
           ["unavailable", "initializing", "stale", "not_applicable"].contains(status) {
            inventories[event.source] = nil
            if status != "not_applicable" {
                for index in pending.indices { pending[index].outcomeCoverage = "source_unavailable" }
            }
        }
        history.removeAll { event.monotonicMs - $0.monotonicMs > 3_600_000 }
        excludedScopeEvents.removeAll { event.monotonicMs - $0.monotonicMs > 3_600_000 }
        // Recommendation decisions retain their own frozen input in raw events; these reference
        // samples are anchored only on observed foreground state, never on a later render time.
        guard event.kind == .contextChanged else { return completed }
        let resolution = event.data["reason"]?.string == "resolution_update"
        if event.kind == .contextChanged {
            for index in pending.indices {
                if resolution {
                    pending[index].resolutionEventIDs = (pending[index].resolutionEventIDs ?? []) + [event.id]
                } else { pending[index].outcomes.append(NavigationOutcome(eventID: event.id,
                    targetID: event.data["targetID"]?.string, entranceID: event.data["entranceID"]?.string,
                    offsetMs: event.monotonicMs - pending[index].inputCutoffMs,
                    reason: event.data["reason"]?.string, hierarchy: coverage.hierarchy(for: event),
                    activeOffsetMs: inactivity.activeMs - (pending[index].activeInputCutoffMs ?? pending[index].inputCutoffMs))) }
            }
            history.append(event)
            if history.count > 300 { history.removeFirst(history.count - 300) }
        }
        // Resolution of a just-activated container is an observation, not a second independent anchor.
        if resolution { return completed }
        let recent = history.filter { event.monotonicMs - $0.monotonicMs <= 300_000 }.suffix(30)
        let inputCoverage = coverage.snapshot(at: event.monotonicMs, sequence: event.sequence)
        pending.append(NavigationSample(id: "sample:" + event.id, sessionID: event.sessionID,
            anchorEventID: event.id, inputCutoffMs: event.monotonicMs, inputThroughSequence: event.sequence,
            recentContextEventIDs: recent.map(\.id), activityContextEventIDs: history.map(\.id),
            excludedScopeEventIDs: excludedScopeEvents.isEmpty ? nil : excludedScopeEvents.map(\.id),
            inventoryEventIDs: foregroundOnly ? [] : inventories.values.sorted(),
            origin: event.decisionID == nil ? "natural_observed" : "recommendation_related",
            decisionID: event.decisionID, inputCoverage: inputCoverage,
            hierarchy: coverage.hierarchy(for: event), resolutionEventIDs: [],
            outcomeCoverage: foregroundOnly || inputCoverage.status == "complete" ? nil : "initial_coverage_unverified",
            activeInputCutoffMs: inactivity.activeMs))
        pending[pending.count - 1].labelRuleVersion = foregroundOnly ? "observed-foreground-sequence-30s-v4" : "observed-active-sequence-30s-v3"
        return completed
    }
    public mutating func finish(reason: String = "export_cutoff") -> [NavigationSample] {
        let result = pending.map { original in
            var value = original; value.status = "incomplete"; value.endReason = reason
            value.outcomeCoverage = "interrupted"; return value
        }
        pending = []; return result
    }
}
