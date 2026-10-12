import Foundation

/// A transient display/execution action. Instances are always regenerated from
/// actual file-opening records; this type is never persisted as a candidate list.
public struct DirectoryRecommendation: Equatable, Identifiable, Sendable {
    public let directory: FileDestination
    public let score: Double
    public let supportingOpenCount: Int
    public let lastOpenedAtMs: Int64

    public var id: String { directory.id }

    public init(directory: FileDestination, score: Double, supportingOpenCount: Int, lastOpenedAtMs: Int64) {
        self.directory = directory
        self.score = score
        self.supportingOpenCount = supportingOpenCount
        self.lastOpenedAtMs = lastOpenedAtMs
    }
}

public struct DirectoryRecommendationEngine: Sendable {
    public init() {}

    /// Scores only parent directories that occur in real, retained open-file
    /// events. An inference can boost an observed directory, never fabricate one.
    public func recommend(
        from events: [FileAccessEvent],
        inferredState: FileStateInference?,
        nowMs: Int64 = Int64(Date().timeIntervalSince1970 * 1_000),
        limit: Int = Int.max
    ) -> [DirectoryRecommendation] {
        var accumulator = DirectoryRecallAccumulator(inferredState: inferredState, nowMs: nowMs)
        for event in events { accumulator.consume(event) }
        return accumulator.recommendations(limit: limit)
    }
}

/// Builds transient candidates while records are read, without retaining the
/// complete history or a saved candidate inventory. Consume newest first to
/// preserve the existing score and tie order.
public struct DirectoryRecallAccumulator {
    private struct Aggregate {
        var directory: FileDestination
        var score: Double
        var support: Int
        var latest: Int64
    }

    private let inferredState: FileStateInference?
    private let nowMs: Int64
    private let terms: [String]
    private var aggregates: [String: Aggregate] = [:]
    private var countedVisits = Set<String>()
    private var matchingEvidenceLocators = Set<String>()

    public init(refinement: String = "", inferredState: FileStateInference? = nil,
                nowMs: Int64 = Int64(Date().timeIntervalSince1970 * 1_000)) {
        self.inferredState = inferredState
        self.nowMs = nowMs
        self.terms = refinement.lowercased().split(whereSeparator: \.isWhitespace).map(String.init)
    }

    public mutating func consume(_ event: FileAccessEvent) {
        if !terms.isEmpty, let parent = event.target.parentDirectoryLocator {
            let name = event.target.name.lowercased()
            if terms.contains(where: { name.contains($0) }) { matchingEvidenceLocators.insert(parent) }
        }
        let locator: String
        let id: String
        if event.target.kind == .directory {
            locator = event.target.locator
            id = event.target.id
        } else {
            guard let parent = event.target.parentDirectoryLocator,
                  let parentID = event.target.parentDirectoryID else { return }
            locator = parent
            id = parentID
        }
        // Multiple sources/files within the same directory visit count once.
        guard countedVisits.insert((event.visitID ?? event.id) + "|" + id).inserted else { return }
        let url = URL(string: locator)
        let name = url?.path == "/" ? "/" : (url?.lastPathComponent ?? locator)
        let age = max(0, nowMs - event.observedAtMs)
        let recency = exp(-Double(age) / (7 * 24 * 60 * 60 * 1_000))
        let stateBoost = inferredState?.preferredDirectoryLocator == locator ? 1.25 : 1.0
        let contribution = (0.05 + recency) * stateBoost
        if var aggregate = aggregates[id] {
            aggregate.score += contribution
            aggregate.support += 1
            aggregate.latest = max(aggregate.latest, event.observedAtMs)
            aggregates[id] = aggregate
        } else {
            aggregates[id] = Aggregate(directory: FileDestination(id: id, kind: .directory, name: name, locator: locator),
                                       score: contribution, support: 1, latest: event.observedAtMs)
        }
    }

    public func recommendations(limit: Int = Int.max) -> [DirectoryRecommendation] {
        aggregates.values.sorted {
            if $0.score != $1.score { return $0.score > $1.score }
            if $0.latest != $1.latest { return $0.latest > $1.latest }
            return $0.directory.locator < $1.directory.locator
        }.prefix(max(0, limit)).map {
            DirectoryRecommendation(directory: $0.directory, score: $0.score,
                                    supportingOpenCount: $0.support, lastOpenedAtMs: $0.latest)
        }
    }

    public func recall(limit: Int = Int.max) -> [DirectoryRecommendation] {
        let all = recommendations()
        let recent = all.sorted { $0.lastOpenedAtMs > $1.lastOpenedAtMs }
        let matching = terms.isEmpty ? [] : all.filter { item in
            let locator = item.directory.locator
            let path = item.directory.url?.path ?? locator
            return matchingEvidenceLocators.contains(locator) || terms.contains { path.lowercased().contains($0) }
        }
        var pool: [DirectoryRecommendation] = []
        var seen = Set<String>()
        for item in matching + recent + all {
            if seen.insert(item.id).inserted { pool.append(item) }
            if pool.count >= limit { break }
        }
        return pool
    }
}
