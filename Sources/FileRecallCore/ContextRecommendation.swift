import Foundation

public struct RecommendationFailure: LocalizedError, Sendable {
    public let stage: String
    public let reason: String
    public init(_ stage: String, _ reason: String) { self.stage = stage; self.reason = reason }
    public var errorDescription: String? { "\(stage)：\(reason)" }
}

/// Memory-only, deliberately not Codable: page content is never an event.
public struct PageContext: Sendable {
    public let application: String
    public let text: String
    public let capturedAt: Date
    public let applicationBundleID: String?
    public let isTruncated: Bool
    public let visitedNodeCount: Int?
    public init(application: String, text: String, capturedAt: Date = Date(), applicationBundleID: String? = nil,
                isTruncated: Bool = false, visitedNodeCount: Int? = nil) {
        self.application = application; self.text = text; self.capturedAt = capturedAt
        self.applicationBundleID = applicationBundleID
        self.isTruncated = isTruncated; self.visitedNodeCount = visitedNodeCount
    }
}

public struct DecisionOption: Codable, Sendable {
    public let id: String
    public let description: String
    public init(id: String, description: String) { self.id = id; self.description = description }
}

public struct ChoiceDecision: Decodable, Sendable {
    public let type: String
    public let choice: String
    public let confidence: Double
    public let probabilities: [String: Double]
    public init(type: String = "choice", choice: String, confidence: Double, probabilities: [String: Double]) {
        self.type = type; self.choice = choice; self.confidence = confidence; self.probabilities = probabilities
    }
    public func validated(options: [DecisionOption]) throws -> ChoiceDecision {
        let keys = Set(options.map(\.id))
        guard type == "choice", keys.contains(choice), Set(probabilities.keys) == keys,
              confidence.isFinite, (0...1).contains(confidence),
              probabilities.values.allSatisfy({ $0.isFinite && (0...1).contains($0) }),
              abs(probabilities.values.reduce(0, +) - 1) < 0.02,
              let peak = probabilities.values.max(), probabilities[choice, default: -1] >= peak - 0.0001 else {
            throw RecommendationFailure("模型响应", "选项或概率无效；请重试。")
        }
        return self
    }
}

public protocol DirectoryDeciding: Sendable {
    func decide(context: PageContext, refinement: String, options: [DecisionOption]) async throws -> ChoiceDecision
}

public struct ContextRecommendationResult: Sendable {
    public let directories: [DirectoryRecommendation]
    /// Raw per-option probabilities from the same final comparison.
    public let probabilitiesByDirectoryID: [String: Double]
    public let uncertain: Bool
    public let confidence: Double
}

public struct ContextRecommendationEngine: Sendable {
    // Kev's protocol accepts at most 255 options; one option is reserved for "uncertain".
    private static let maximumDirectoriesPerRequest = 254
    public init() {}

    /// Recall from all retained events again on every refinement. Never enumerate disk.
    public func recall(events: [FileAccessEvent], refinement: String) -> [DirectoryRecommendation] {
        var accumulator = DirectoryRecallAccumulator(refinement: refinement)
        for event in events { accumulator.consume(event) }
        return accumulator.recall()
    }

    public func recommend(context: PageContext, events: [FileAccessEvent], refinement: String,
                          decider: any DirectoryDeciding) async throws -> ContextRecommendationResult {
        guard !context.text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
            throw RecommendationFailure("读取前台页面", "没有可用语义；返回目标窗口后重试。")
        }
        return try await recommend(context: context, candidates: recall(events: events, refinement: refinement),
                                   refinement: refinement, decider: decider)
    }

    public func recommend(context: PageContext, candidates: [DirectoryRecommendation], refinement: String,
                          decider: any DirectoryDeciding) async throws -> ContextRecommendationResult {
        guard !context.text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
            throw RecommendationFailure("读取前台页面", "没有可用语义；返回目标窗口后重试。")
        }
        guard !candidates.isEmpty else {
            throw RecommendationFailure("候选召回", "暂无已记录目录；请先开始记录并打开本地文件。")
        }
        var finalists = candidates
        var uncertaintyByGroup: [Double] = []
        while finalists.count > Self.maximumDirectoriesPerRequest {
            // Every observed candidate enters one bounded protocol request. Only each
            // request's selected directory advances; probabilities across requests
            // cannot be compared, so the survivors meet in another tournament round.
            let grouped = finalists.sorted { $0.directory.locator < $1.directory.locator }
            var survivors: [DirectoryRecommendation] = []
            for start in stride(from: 0, to: grouped.count, by: Self.maximumDirectoriesPerRequest) {
                try Task.checkCancellation()
                let group = Array(grouped[start..<min(start + Self.maximumDirectoriesPerRequest, grouped.count)])
                let answer = try await choose(group, context, refinement, decider)
                if answer.choice == "uncertain" {
                    uncertaintyByGroup.append(answer.probabilities["uncertain", default: 0])
                } else if let index = Int(answer.choice.dropFirst()), group.indices.contains(index) {
                    survivors.append(group[index])
                }
            }
            guard !survivors.isEmpty else {
                let confidence = uncertaintyByGroup.isEmpty ? 1
                    : uncertaintyByGroup.reduce(0, +) / Double(uncertaintyByGroup.count)
                return ContextRecommendationResult(directories: [], probabilitiesByDirectoryID: [:],
                                                   uncertain: true, confidence: confidence)
            }
            finalists = survivors
        }
        let answer = try await choose(finalists, context, refinement, decider)
        let uncertain = answer.choice == "uncertain"
        let rankedDirectories = ranked(finalists, answer)
        let probabilities = Dictionary(uniqueKeysWithValues: finalists.enumerated().map {
            ($0.element.id, answer.probabilities["c\($0.offset)", default: 0])
        })
        return ContextRecommendationResult(directories: rankedDirectories, probabilitiesByDirectoryID: probabilities,
                                           uncertain: uncertain, confidence: answer.confidence)
    }

    private func choose(_ items: [DirectoryRecommendation], _ context: PageContext,
                        _ refinement: String, _ decider: any DirectoryDeciding) async throws -> ChoiceDecision {
        var options = items.enumerated().map { index, item in
            DecisionOption(id: "c\(index)", description: item.directory.url?.path ?? item.directory.locator)
        }
        options.append(DecisionOption(id: "uncertain", description: "Insufficient evidence, or none of these directories fits the user's task."))
        return try await decider.decide(context: context, refinement: refinement, options: options).validated(options: options)
    }

    private func ranked(_ items: [DirectoryRecommendation], _ answer: ChoiceDecision) -> [DirectoryRecommendation] {
        items.enumerated().sorted {
            let lhs = answer.probabilities["c\($0.offset)", default: 0]
            let rhs = answer.probabilities["c\($1.offset)", default: 0]
            return lhs == rhs ? $0.offset < $1.offset : lhs > rhs
        }.map(\.element)
    }
}
