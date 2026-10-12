import Foundation

/// Private offline parity fixture; never used by the collector or UI.
public struct SemanticModelValidation: Codable {
    public struct Update: Codable { public var readyIndex: Int; public var record: SemanticLearningRecord }
    public struct Query: Codable { public var position: Int; public var index: Int; public var origin: String; public var target: Destination; public var expected: [String] }
    public struct Annotation: Codable { public var target: Destination; public var expected: SemanticProfile }
    public var updates: [Update]
    public var queries: [Query]
    public var annotations: [Annotation]
    public var modelVersion: String?

    public func run() -> [String: JSONValue] {
        var model = SemanticRecommendationModel(), navigation = NavigationRecommendationModel(), cursor = 0
        var mismatches: [JSONValue] = [], annotationMismatches: [JSONValue] = []
        let kinds: Set<String> = ["site", "url_parent", "local_directory_scope", "observed_parent_id", "name_token", "course_scope", "project_scope", "repository_scope"]
        for (index, item) in annotations.enumerated() {
            let native = SemanticMetadata.annotate(item.target)
            let expected = item.expected.tags.filter { kinds.contains($0.kind) }.sorted { $0.key < $1.key }
            if native.tags != expected {
                annotationMismatches.append(.object(["index": .number(Double(index)),
                    "native": (try? .encoded(native.tags)) ?? .null, "expected": (try? .encoded(expected)) ?? .null]))
            }
        }
        for query in queries {
            while cursor < updates.count && updates[cursor].readyIndex < query.index {
                model.learn(updates[cursor].record); navigation.learn(updates[cursor].record); cursor += 1
            }
            let predicted = Array((modelVersion == SemanticModelState.version ? navigation.ranked(current: query.target) :
                model.ranked(origin: query.origin, annotation: SemanticMetadata.annotate(query.target))).prefix(4))
            if predicted != query.expected {
                mismatches.append(.object(["position": .number(Double(query.position)), "predicted": .array(predicted.map(JSONValue.string)), "expected": .array(query.expected.map(JSONValue.string))]))
            }
        }
        return ["queries": .number(Double(queries.count)), "annotations": .number(Double(annotations.count)),
                "rankingMismatches": .array(mismatches), "annotationMismatches": .array(annotationMismatches)]
    }
}
