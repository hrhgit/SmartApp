import Foundation

/// Versioned metadata features; locators are never normalized into a different destination.
public struct SemanticProfile: Codable, Equatable, Sendable {
    public struct Identity: Codable, Equatable, Sendable {
        public var id: String?
        public var kind: String
        public var bundleID: String?
        public var locator: String?
    }
    public struct Tag: Codable, Equatable, Sendable {
        public var key: String
        public var kind: String
        public var strength: Double
        public var confidence: Double
    }
    public var identity: Identity
    public var tags: [Tag]

    public func similarity(to other: Self) -> Double {
        if identity.kind == other.identity.kind && identity.kind != "window" {
            if let locator = identity.locator, locator == other.identity.locator { return 1 }
            if identity.locator == nil && other.identity.locator == nil {
                if identity.kind == "app", let owner = identity.bundleID, owner == other.identity.bundleID { return 1 }
                if identity.kind != "app", let id = identity.id, id == other.identity.id { return 1 }
            }
        }
        let eligible: Set<String> = ["site", "url_parent", "local_directory_scope", "observed_parent_id", "name_token", "course_scope", "project_scope", "repository_scope"]
        let right = Dictionary(other.tags.map { ($0.key, $0) }, uniquingKeysWith: { a, _ in a })
        return tags.compactMap { a -> Double? in
            guard eligible.contains(a.kind), let b = right[a.key],
                  a.kind != "local_directory_scope" || min(a.strength, b.strength) >= 0.35 else { return nil }
            return min(a.strength, b.strength) * min(a.confidence, b.confidence)
        }.max() ?? 0
    }
}

public struct SemanticLearningRecord: Codable, Equatable, Sendable {
    public var origin: String
    public var annotation: SemanticProfile
    public var label: String
    public var outcomeID: String
    public var outcomeTimestamp: Double
    public var weight: Double
    public var source: String
    /// Metadata of the observed outcome, never part of the earlier input.
    public var destination: SemanticProfile.Identity?
    public var observedStep: Int?
    public var navigationDisposition: NavigationDisposition?
    public var navigationOrigin: String?
    public var navigationLabel: String?
    /// Identifies the confirmed foreground anchor for another in-memory learner.
    /// Older saved records omit this optional field.
    public var inputEventID: String?
    public init(origin: String, annotation: SemanticProfile, label: String, outcomeID: String,
                outcomeTimestamp: Double, weight: Double = 1, source: String = "natural_observed",
                destination: SemanticProfile.Identity? = nil, observedStep: Int? = nil,
                inputEventID: String? = nil) {
        self.origin = origin; self.annotation = annotation; self.label = label; self.outcomeID = outcomeID
        self.outcomeTimestamp = outcomeTimestamp; self.weight = weight; self.source = source
        self.destination = destination; self.observedStep = observedStep; self.inputEventID = inputEventID
    }
}

public struct SemanticModelState: Codable, Sendable {
    public static let legacyVersion = "semantic-statistics-hard-kind-v1"
    public static let previousVersion = "semantic-navigation-hierarchy-v2"
    /// Dynamic-label online Softmax trained with weighted multiclass cross-entropy.
    public static let version = "semantic-cross-entropy-v3"
    public var version: String = Self.version
    public var records: [SemanticLearningRecord]
    public var identityKeySHA256: String
    public var seedThroughEventID: String?
    public var excludedSessions: [String]
    public var migratedFromVersion: String?
    public var metadataEnrichedAtMs: Int64?
    /// Optional so existing v3 files and research seeds remain readable.
    public var checkpoint: NavigationRecommendationCheckpoint?
    public init(records: [SemanticLearningRecord] = [], identityKeySHA256: String = "",
                seedThroughEventID: String? = nil, excludedSessions: [String] = []) {
        self.records = records; self.identityKeySHA256 = identityKeySHA256
        self.seedThroughEventID = seedThroughEventID; self.excludedSessions = excludedSessions
    }
}

/// Dynamic-label online Softmax trained with weighted multiclass cross-entropy.
///
/// Each matured navigation outcome is one positive class. At update time every
/// class known then receives `lr * weight * (1[label] - P(label|input)) * x`.
/// The input contains the current target, owner/type and a bounded semantic-tag
/// budget; it never contains an open-window or background-inventory feature.
/// Labels are introduced only when their real outcome matures, so predictions
/// never require a future vocabulary. Execution eligibility stays separate.
public struct SemanticRecommendationModel: Sendable {
    public struct Snapshot: Codable, Sendable {
        public var recordCount: Int
        public var weights: [String: [String: Double]]
    }
    public private(set) var records: [SemanticLearningRecord] = []
    private var seen = Set<String>()
    private var weights: [String: [String: Double]] = [:]
    private static let learningRate = 0.3
    private static let semanticBudget = 0.5
    public init(records: [SemanticLearningRecord] = []) { for record in records { learn(record) } }
    public var snapshot: Snapshot { Snapshot(recordCount: records.count, weights: weights) }
    public init?(records: [SemanticLearningRecord], snapshot: Snapshot) {
        guard snapshot.recordCount == records.count,
              Set(records.map { $0.label }) == Set(snapshot.weights.keys),
              snapshot.weights.values.allSatisfy({ !$0.isEmpty && $0.values.allSatisfy(\.isFinite) }) else { return nil }
        var seen = Set<String>()
        for record in records {
            guard record.weight.isFinite, record.weight > 0, record.outcomeTimestamp.isFinite,
                  !record.label.isEmpty, !record.origin.isEmpty,
                  seen.insert(record.origin + "\u{0}" + record.label + "\u{0}" + record.outcomeID).inserted else { return nil }
        }
        self.records = records
        self.seen = seen
        self.weights = snapshot.weights
    }
    public mutating func learn(_ record: SemanticLearningRecord) {
        guard record.weight.isFinite, record.weight > 0, record.outcomeTimestamp.isFinite,
              !record.label.isEmpty, !record.origin.isEmpty else { return }
        let key = record.origin + "\u{0}" + record.label + "\u{0}" + record.outcomeID
        guard seen.insert(key).inserted else { return }
        let features = Self.features(origin: record.origin, annotation: record.annotation)
        if weights[record.label] == nil { weights[record.label] = [:] }
        let probabilities = Self.softmax(logits(features: features))
        let rate = Self.learningRate * record.weight
        for label in weights.keys.sorted() {
            var row = weights[label] ?? [:]
            let gradient = (label == record.label ? 1.0 : 0.0) - (probabilities[label] ?? 0)
            for (feature, value) in features {
                let updated = (row[feature] ?? 0) + rate * gradient * value
                guard updated.isFinite else { continue }
                row[feature] = updated
            }
            weights[label] = row
        }
        records.append(record)
    }
    public static func origin(_ target: Destination) -> String {
        target.kind == .window ? "window_class:" + (target.bundleID ?? "") : target.id
    }
    public static func action(_ target: Destination) -> String? {
        if target.kind == .window || target.kind == .app {
            guard let owner = target.bundleID, !owner.isEmpty else { return nil }
            return "app:" + owner
        }
        guard target.locator != nil else { return nil }
        return target.id
    }
    public func scores(origin: String, annotation: SemanticProfile) -> [String: Double] {
        Self.softmax(logits(features: Self.features(origin: origin, annotation: annotation)))
    }
    public func ranked(origin: String, annotation: SemanticProfile) -> [String] {
        let values = scores(origin: origin, annotation: annotation)
        return values.keys.sorted { values[$0] == values[$1] ? $0 < $1 : values[$0]! > values[$1]! }
    }

    private func logits(features: [String: Double]) -> [String: Double] {
        Dictionary(uniqueKeysWithValues: weights.keys.sorted().map { label in
            (label, features.reduce(0) { $0 + (weights[label]?[$1.key] ?? 0) * $1.value })
        })
    }

    private static func features(origin: String, annotation: SemanticProfile) -> [String: Double] {
        var values: [String: Double] = ["bias": 1, "origin=" + origin: 1]
        let kind = origin.hasPrefix("window_class:") ? "window_class" : origin.split(separator: ":", maxSplits: 1).first.map(String.init) ?? "unknown"
        values["kind=" + kind] = 0.1
        if let bundle = annotation.identity.bundleID, !bundle.isEmpty { values["app=" + bundle] = 0.25 }
        var tags: [String: Double] = [:]
        for tag in annotation.tags {
            guard tag.strength.isFinite, tag.confidence.isFinite,
                  tag.strength > 0, tag.confidence > 0 else { continue }
            tags["semantic=" + tag.key] = max(tags["semantic=" + tag.key] ?? 0, tag.strength * tag.confidence)
        }
        let divisor = max(1, tags.values.reduce(0, +))
        for key in tags.keys.sorted() { values[key] = Self.semanticBudget * (tags[key] ?? 0) / divisor }
        return values
    }

    private static func softmax(_ logits: [String: Double]) -> [String: Double] {
        guard let anchor = logits.values.max(), anchor.isFinite else { return [:] }
        let exponentials = Dictionary(uniqueKeysWithValues: logits.keys.sorted().map { key in
            (key, Foundation.exp(max(-60, min(60, (logits[key] ?? anchor) - anchor))))
        })
        let total = exponentials.values.reduce(0, +)
        return total.isFinite && total > 0 ? exponentials.mapValues { $0 / total } : [:]
    }
}
