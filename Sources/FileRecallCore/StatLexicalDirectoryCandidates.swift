import Foundation

/// The production candidate stage selected from the v4 candidate study.
/// It only combines the current, temporary frontmost-window text with already
/// observed directory locators and file names. It never reads the filesystem or
/// persists a candidate list, page text, or refinement.
public struct StatLexicalDirectoryCandidateEngine: Sendable {
    /// The selected candidate count for the product's semantics-enabled path.
    public static let productionCandidateLimit = 32

    public let candidateLimit: Int
    public let halfLifeHours: Double
    public let lexicalMix: Double

    public init(candidateLimit: Int = Self.productionCandidateLimit,
                halfLifeHours: Double = 168,
                lexicalMix: Double = 0.85) {
        self.candidateLimit = candidateLimit
        self.halfLifeHours = halfLifeHours
        self.lexicalMix = lexicalMix
    }

    /// Convenience entry point for callers that already have an event array.
    /// Store-backed product code should use `makeAccumulator` and consume its
    /// snapshot one event at a time instead.
    public func rank(events: [FileAccessEvent], context: PageContext, refinement: String = "",
                     nowMs: Int64 = Int64(Date().timeIntervalSince1970 * 1_000)) throws -> [DirectoryRecommendation] {
        var accumulator = try makeAccumulator(context: context, refinement: refinement, nowMs: nowMs)
        for event in events { accumulator.consume(event) }
        return try accumulator.candidates()
    }

    /// Creates a memory-only accumulator for a single recommendation request.
    public func makeAccumulator(context: PageContext, refinement: String = "",
                                nowMs: Int64 = Int64(Date().timeIntervalSince1970 * 1_000)) throws
        -> StatLexicalDirectoryCandidateAccumulator {
        try StatLexicalDirectoryCandidateAccumulator(context: context, refinement: refinement,
                                                     candidateLimit: candidateLimit,
                                                     halfLifeHours: halfLifeHours,
                                                     lexicalMix: lexicalMix, nowMs: nowMs)
    }
}

/// Streams all retained observations into the selected semantics-aware candidate
/// model. The only retained state is a per-observed-directory summary for this
/// request; source events, page text, refinements, and output candidates are not
/// written to disk.
public struct StatLexicalDirectoryCandidateAccumulator: Sendable {
    private struct Aggregate: Sendable {
        let directory: FileDestination
        var supportingOpenCount = 0
        var lastOpenedAtMs = Int64.min
        var decayedHistoryWeight = 0.0
        private var descriptionTexts = Set<String>()
        var termFrequency: [String: Double] = [:]

        init(directory: FileDestination) {
            self.directory = directory
            addDescription(directory.url?.path ?? directory.locator)
        }

        mutating func addDescription(_ text: String) {
            guard descriptionTexts.insert(text).inserted else { return }
            for token in StatLexicalDirectoryCandidateAccumulator.tokens(text) {
                termFrequency[token, default: 0] += 1
            }
        }
    }

    private let candidateLimit: Int
    private let halfLifeHours: Double
    private let lexicalMix: Double
    private let nowMs: Int64
    private let query: [String: Double]
    private var aggregates: [String: Aggregate] = [:]
    private var countedVisits = Set<String>()

    public init(context: PageContext, refinement: String, candidateLimit: Int,
                halfLifeHours: Double = 168, lexicalMix: Double = 0.85,
                nowMs: Int64 = Int64(Date().timeIntervalSince1970 * 1_000)) throws {
        guard candidateLimit > 0,
              halfLifeHours.isFinite, halfLifeHours > 0,
              lexicalMix.isFinite, (0...1).contains(lexicalMix) else {
            throw RecommendationFailure("候选召回", "候选模型设置无效，未开始本次推荐。")
        }
        guard !context.text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
            throw RecommendationFailure("读取前台页面", "没有可用语义；返回目标窗口后重试。")
        }
        self.candidateLimit = candidateLimit
        self.halfLifeHours = halfLifeHours
        self.lexicalMix = lexicalMix
        self.nowMs = nowMs
        self.query = Self.makeQuery(page: context.text, refinement: refinement)
    }

    /// Adds one real foreground observation. Repeated sources for one visit and
    /// directory are one history vote, while a newly observed file name can still
    /// describe that directory for this temporary request.
    public mutating func consume(_ event: FileAccessEvent) {
        guard let directory = Self.directory(for: event) else { return }
        var aggregate = aggregates[directory.id] ?? Aggregate(directory: directory)
        aggregate.addDescription(event.target.name)

        let visitKey = (event.visitID ?? event.id) + "|" + directory.id
        if countedVisits.insert(visitKey).inserted {
            aggregate.supportingOpenCount += 1
            aggregate.lastOpenedAtMs = max(aggregate.lastOpenedAtMs, event.observedAtMs)
            aggregate.decayedHistoryWeight += historyWeight(at: event.observedAtMs)
        }
        aggregates[directory.id] = aggregate
    }

    /// Ranks every observed directory first, then returns the configured prefix.
    /// When no current-page or temporary-input term matches any observed evidence,
    /// it deliberately stops instead of silently changing to history-only recall.
    public func candidates() throws -> [DirectoryRecommendation] {
        guard !aggregates.isEmpty else { return [] }
        guard !query.isEmpty else {
            throw noLexicalEvidenceFailure()
        }

        let ids = aggregates.keys.sorted()
        let documentCount = Double(ids.count)
        let averageLength = max(1, ids.reduce(0.0) { total, id in
            total + (aggregates[id]?.termFrequency.values.reduce(0, +) ?? 0)
        } / documentCount)

        var documentFrequency: [String: Double] = [:]
        for id in ids {
            guard let aggregate = aggregates[id] else { continue }
            for term in aggregate.termFrequency.keys {
                documentFrequency[term, default: 0] += 1
            }
        }

        let lexicalScores = Dictionary(uniqueKeysWithValues: ids.compactMap { id -> (String, Double)? in
            guard let aggregate = aggregates[id] else { return nil }
            let document = aggregate.termFrequency
            let length = document.values.reduce(0, +)
            let score = query.keys.sorted().reduce(0.0) { total, term in
                guard let frequency = document[term] else { return total }
                let frequencyInDocuments = documentFrequency[term, default: 0]
                let inverseDocumentFrequency = log(1 + (documentCount - frequencyInDocuments + 0.5)
                    / (frequencyInDocuments + 0.5))
                let k1 = 1.2
                let b = 0.75
                return total + query[term, default: 0] * inverseDocumentFrequency * (frequency * (k1 + 1))
                    / (frequency + k1 * (1 - b + b * length / averageLength))
            }
            return (id, score)
        })
        guard lexicalScores.values.contains(where: { $0 > 0 }) else {
            throw noLexicalEvidenceFailure()
        }

        let history = Self.normalized(Dictionary(uniqueKeysWithValues: ids.compactMap { id -> (String, Double)? in
            aggregates[id].map { (id, $0.decayedHistoryWeight) }
        }), ids: ids)
        let lexical = Self.normalized(lexicalScores, ids: ids)
        let scores = Dictionary(uniqueKeysWithValues: ids.map {
            ($0, (1 - lexicalMix) * history[$0, default: 0] + lexicalMix * lexical[$0, default: 0])
        })

        return aggregates.values.sorted {
            let left = scores[$0.directory.id, default: 0]
            let right = scores[$1.directory.id, default: 0]
            return left == right ? $0.directory.locator < $1.directory.locator : left > right
        }.prefix(candidateLimit).map {
            DirectoryRecommendation(directory: $0.directory, score: scores[$0.directory.id, default: 0],
                                    supportingOpenCount: $0.supportingOpenCount,
                                    lastOpenedAtMs: $0.lastOpenedAtMs)
        }
    }

    private func historyWeight(at timestamp: Int64) -> Double {
        let age = max(0, nowMs - timestamp)
        return exp2(-Double(age) / (halfLifeHours * 3_600_000))
    }

    private func noLexicalEvidenceFailure() -> RecommendationFailure {
        RecommendationFailure("候选召回", "当前前台内容没有与已记录目录或文件名对应的词语。可在补充描述中写入项目名或文件名后重试。")
    }

    private static func directory(for event: FileAccessEvent) -> FileDestination? {
        if event.target.kind == .directory {
            let locator = event.target.locator
            let url = URL(string: locator)
            let name = url?.path == "/" ? "/" : (url?.lastPathComponent ?? locator)
            return FileDestination(id: event.target.id, kind: .directory, name: name, locator: locator)
        }
        guard let parentID = event.target.parentDirectoryID,
              let parentLocator = event.target.parentDirectoryLocator else { return nil }
        let url = URL(string: parentLocator)
        let name = url?.path == "/" ? "/" : (url?.lastPathComponent ?? parentLocator)
        return FileDestination(id: parentID, kind: .directory, name: name, locator: parentLocator)
    }

    private static func normalized(_ values: [String: Double], ids: [String]) -> [String: Double] {
        let total = ids.reduce(0.0) { $0 + values[$1, default: 0] }
        guard total.isFinite, total > 0 else {
            return Dictionary(uniqueKeysWithValues: ids.map { ($0, 1 / Double(ids.count)) })
        }
        return Dictionary(uniqueKeysWithValues: ids.map { ($0, values[$0, default: 0] / total) })
    }

    private static func makeQuery(page: String, refinement: String) -> [String: Double] {
        var terms = Dictionary(uniqueKeysWithValues: Set(tokens(page)).map { ($0, 1.0) })
        for term in Set(tokens(refinement)) { terms[term, default: 0] += 4 }
        return terms
    }

    private static let stopWords: Set<String> = [
        "the", "and", "for", "with", "this", "that", "from", "into", "your", "you", "are", "was", "were",
        "has", "have", "will", "can", "not", "all", "file", "files", "folder", "folders", "users", "home"
    ]

    /// English words and Chinese 1/2/3-character pieces. It needs no dictionary,
    /// network, filesystem enumeration, or file-content access.
    private static func tokens(_ text: String) -> [String] {
        let separated = text.replacingOccurrences(of: "([a-z])([A-Z])", with: "$1 $2", options: .regularExpression)
        var result: [String] = []
        var latin = ""
        var chinese: [String] = []

        func flushLatin() {
            if latin.count >= 2, !stopWords.contains(latin) { result.append(latin) }
            latin = ""
        }
        func flushChinese() {
            for width in 1...3 where chinese.count >= width {
                for index in 0...(chinese.count - width) {
                    result.append(chinese[index..<(index + width)].joined())
                }
            }
            chinese = []
        }
        for scalar in separated.lowercased().unicodeScalars {
            let code = scalar.value
            if (0x3400...0x9fff).contains(code) || (0x20000...0x3134f).contains(code) {
                flushLatin()
                chinese.append(String(scalar))
            } else if CharacterSet.alphanumerics.contains(scalar) {
                flushChinese()
                latin.unicodeScalars.append(scalar)
            } else {
                flushLatin()
                flushChinese()
            }
        }
        flushLatin()
        flushChinese()
        return result
    }
}
