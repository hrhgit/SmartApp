import Foundation

/// Bounded summaries of actual timer observations. A summary is known only at its end;
/// it does not assert that every source was healthy throughout the interval.
public struct HeartbeatSummary: Sendable {
    private var first: Double?
    private var last: Double?
    private var checks = 0
    private var maximumGap = 0.0
    public init() {}
    @discardableResult public mutating func observe(at time: Double) -> Bool {
        if let last { maximumGap = max(maximumGap, time - last) }
        if first == nil { first = time }
        last = time; checks += 1
        return time - (first ?? time) >= 30_000
    }
    public static func isIdentityFree(_ value: JSONValue) -> Bool {
        guard let fields = value.object,
              fields["summaryVersion"]?.string == "collector-heartbeat-v1" else { return false }
        let numeric = Set(["firstObservedMonotonicMs", "lastObservedMonotonicMs", "checkCount", "maximumCheckGapMs"])
        return Set(fields.keys) == numeric.union(["summaryVersion"]) &&
            numeric.allSatisfy { fields[$0]?.number.map { $0.isFinite && $0 >= 0 } == true }
    }
    public mutating func take() -> [String: JSONValue]? {
        guard let first, let last else { return nil }
        let result: [String: JSONValue] = [
            "summaryVersion": .string("collector-heartbeat-v1"),
            "firstObservedMonotonicMs": .number(first),
            "lastObservedMonotonicMs": .number(last),
            "checkCount": .number(Double(checks)),
            "maximumCheckGapMs": .number(maximumGap)
        ]
        self = Self()
        return result
    }
}
