import Foundation
import CryptoKit

public struct NavigationRecommendationCheckpoint: Codable, Sendable {
    public var format: String
    public var recordsSHA256: String
    public var semantic: SemanticRecommendationModel.Snapshot
}

/// These describe what the available action can do, not the user's hidden intent.
public enum NavigationDisposition: String, Codable, Sendable {
    case navigation
    case currentDestination = "current_destination"
    case sameApplicationUnspecified = "same_application_unspecified"
}

public enum NavigationSemantics {
    public static let chatService = "service:https://chatgpt.com/"
    public static let chatEntry = "https://chatgpt.com/"

    /// Explicit service routes, not a rule that arbitrary pages on one host are interchangeable.
    /// Locators, anchors and query strings are retained in the original outcome record.
    public static func service(_ identity: SemanticProfile.Identity) -> String? {
        guard identity.kind == "page", let locator = identity.locator,
              let url = URLComponents(string: locator), url.scheme?.lowercased() == "https",
              url.host?.lowercased() == "chatgpt.com", url.port == nil || url.port == 443,
              url.user == nil, url.password == nil else { return nil }
        let knownQueryKeys: Set<String> = ["messageId", "src", "historySearchQuery", "no_universal_links"]
        guard (url.queryItems ?? []).allSatisfy({ knownQueryKeys.contains($0.name) }),
              url.fragment?.hasPrefix("/") != true else { return nil }
        if url.path.isEmpty || url.path == "/" { return chatService }
        let components = url.path.split(separator: "/", omittingEmptySubsequences: false)
        guard components.count == 3, components[0].isEmpty, components[1] == "c",
              UUID(uuidString: String(components[2])) != nil else { return nil }
        return chatService
    }

    public static func entryLocator(for label: String) -> String? {
        label == chatService ? chatEntry : nil
    }

    public static func disposition(label: String, input: SemanticProfile.Identity) -> NavigationDisposition {
        if label == input.id { return .currentDestination }
        if let owner = input.bundleID, label == "app:" + owner { return .sameApplicationUnspecified }
        return .navigation
    }

    public static func derive(_ record: SemanticLearningRecord) -> SemanticLearningRecord {
        var value = record
        value.navigationDisposition = disposition(label: record.label, input: record.annotation.identity)
        value.navigationOrigin = inputOrigin(record.origin, record.annotation.identity)
        value.navigationLabel = value.navigationDisposition == .navigation ?
            (record.destination.flatMap(service) ?? record.label) : nil
        return value
    }

    public static func inputOrigin(_ origin: String, _ identity: SemanticProfile.Identity) -> String {
        service(identity) ?? origin
    }

    /// A final action-effect check also applies to popularity and recency priors.
    /// A different concrete resource in the same application remains a valid action.
    public static func changesDestination(_ target: Destination, current: Destination) -> Bool {
        if target.id == current.id { return false }
        if target.kind == .app, let owner = current.bundleID, target.bundleID == owner { return false }
        if let locator = target.locator, target.kind == current.kind, locator == current.locator { return false }
        return true
    }

    /// Resolve a service prediction only at execution/menu time, using an already admitted
    /// historical locator. This does not fabricate a visit, target update or candidate record.
    public static func resolve(_ label: String, catalog: [String: Destination]) -> Destination? {
        if let target = catalog[label] { return target }
        guard let locator = entryLocator(for: label) else { return nil }
        return catalog.values.filter { $0.kind == .page && $0.locator == locator }.sorted { $0.id < $1.id }.first
    }
}

/// Learns one navigation label per real outcome at the service or exact-resource level.
/// Non-navigation observations remain in records for auditing, but cannot add navigation votes.
public struct NavigationRecommendationModel: Sendable {
    public private(set) var records: [SemanticLearningRecord] = []
    private var seen = Set<String>()
    private var navigation = SemanticRecommendationModel()

    public init(records: [SemanticLearningRecord] = []) {
        for record in records { learn(record) }
    }

    private static let checkpointFormat = "navigation-softmax-v3-weights-1"

    public func checkpoint() throws -> NavigationRecommendationCheckpoint {
        NavigationRecommendationCheckpoint(format: Self.checkpointFormat,
            recordsSHA256: try Self.digest(records), semantic: navigation.snapshot)
    }

    /// Restores learned parameters without running historical gradient updates.
    public static func restored(records: [SemanticLearningRecord],
                                checkpoint: NavigationRecommendationCheckpoint) -> Self? {
        guard checkpoint.format == checkpointFormat,
              checkpoint.recordsSHA256 == (try? digest(records)) else { return nil }
        var result = Self()
        var coarseRecords: [SemanticLearningRecord] = []
        for raw in records {
            guard raw.weight.isFinite, raw.weight > 0, raw.outcomeTimestamp.isFinite,
                  !raw.label.isEmpty, !raw.origin.isEmpty else { return nil }
            let identity = raw.origin + "\u{0}" + raw.label + "\u{0}" + raw.outcomeID
            guard result.seen.insert(identity).inserted else { return nil }
            let record = NavigationSemantics.derive(raw)
            result.records.append(record)
            guard record.navigationDisposition == .navigation else { continue }
            var coarse = record
            coarse.origin = record.navigationOrigin ?? record.origin
            coarse.label = record.navigationLabel ?? record.label
            coarseRecords.append(coarse)
        }
        guard let navigation = SemanticRecommendationModel(records: coarseRecords,
                                                            snapshot: checkpoint.semantic) else { return nil }
        result.navigation = navigation
        return result
    }

    private static func digest(_ records: [SemanticLearningRecord]) throws -> String {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys, .withoutEscapingSlashes]
        return SHA256.hash(data: try encoder.encode(records)).map { String(format: "%02x", $0) }.joined()
    }

    public mutating func learn(_ raw: SemanticLearningRecord) {
        guard raw.weight.isFinite, raw.weight > 0, raw.outcomeTimestamp.isFinite,
              !raw.label.isEmpty, !raw.origin.isEmpty else { return }
        let identity = raw.origin + "\u{0}" + raw.label + "\u{0}" + raw.outcomeID
        guard seen.insert(identity).inserted else { return }
        let record = NavigationSemantics.derive(raw)
        records.append(record)
        guard record.navigationDisposition == .navigation else { return }
        var coarse = record
        coarse.origin = record.navigationOrigin ?? record.origin
        coarse.label = record.navigationLabel ?? record.label
        navigation.learn(coarse)
    }

    public func ranked(current: Destination) -> [String] {
        let values = scores(current: current)
        return values.keys.sorted { values[$0] == values[$1] ? $0 < $1 : values[$0]! > values[$1]! }
    }

    public func scores(current: Destination) -> [String: Double] {
        let annotation = SemanticMetadata.annotate(current)
        let origin = SemanticRecommendationModel.origin(current)
        let primary = navigation.scores(origin: NavigationSemantics.inputOrigin(origin, annotation.identity), annotation: annotation)
        var result: [String: Double] = [:]
        for (label, score) in primary {
            let selected = NavigationSemantics.entryLocator(for: label) == nil ? label : specificLabel(service: label, origin: origin) ?? label
            guard NavigationSemantics.disposition(label: selected, input: annotation.identity) == .navigation else { continue }
            if let entry = NavigationSemantics.entryLocator(for: selected), entry == current.locator { continue }
            result[selected, default: 0] += score
        }
        return result
    }

    /// Specificity uses only distinct, already matured outcomes for this exact input.
    /// Global popularity, inventory size and weak site matches cannot justify a conversation.
    /// Wilson is a conservative fixed evidence rule, not a calibrated intent guarantee.
    private func specificLabel(service: String, origin: String) -> String? {
        var seenOutcomes = Set<String>(), counts: [String: Double] = [:]
        for record in records where record.origin == origin && record.navigationDisposition == .navigation {
            guard let target = record.destination, NavigationSemantics.service(target) == service,
                  seenOutcomes.insert(record.outcomeID).inserted else { continue }
            counts[record.label, default: 0] += min(1, record.weight)
        }
        let total = counts.values.reduce(0, +)
        guard total >= 4, let best = counts.keys.sorted(by: { counts[$0] == counts[$1] ? $0 < $1 : counts[$0]! > counts[$1]! }).first,
              let support = counts[best],
              let locator = records.first(where: { $0.label == best })?.destination?.locator,
              URLComponents(string: locator)?.path.hasPrefix("/c/") == true else { return nil }
        let p = support / total, z2 = 1.96 * 1.96
        let lower = (p + z2 / (2 * total) - 1.96 * sqrt(p * (1 - p) / total + z2 / (4 * total * total))) / (1 + z2 / total)
        return lower > 0.5 ? best : nil
    }
}
