import Foundation

/// Counts observed destination transitions, not collector callbacks or inferred input gestures.
/// The first observation after a boundary is a baseline, never proof of a user operation.
public struct UserActivityCounter {
    public private(set) var count = 0
    private var session: String?
    private var enabled = false
    private var gate = TargetEventGate()
    private var previous: (target: String, entrance: String?, source: String)?
    public init() {}

    @discardableResult public mutating func consume(_ event: BehaviorEvent) -> Bool {
        if session != event.sessionID || event.kind == .sessionStart {
            session = event.sessionID; previous = nil; gate.reset()
            enabled = event.kind == .sessionStart &&
                event.data["targetEligibilityPolicyVersion"]?.string.map(TargetEligibilityPolicy.supportedVersions.contains) == true
        }
        guard enabled else { return false }
        guard gate.accept(event) else { previous = nil; return false }
        if event.kind == .captureGap || event.kind == .observationScope || event.kind == .sessionEnd {
            previous = nil; return false
        }
        if event.kind == .sourceStatus, event.source == previous?.source,
           ["unavailable", "stale", "initializing", "not_applicable"].contains(event.data["observationStatus"]?.string ?? "") {
            previous = nil
        }
        guard event.kind == .contextChanged, let target = event.data["targetID"]?.string else { return false }
        let entrance = event.data["entranceID"]?.string
        let changed = previous.map { $0.target != target || $0.entrance != entrance } ?? false
        previous = (target, entrance, event.source)
        guard changed, event.data["reason"]?.string != "resolution_update" else { return false }
        count += 1
        return true
    }
}
