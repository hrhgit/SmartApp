import Foundation

/// Raw archives remain readable; only independently admitted sessions feed training.
public struct EligibleNavigationReplay {
    private var session: String?
    private var enabled = false
    private var policyVersion: String?
    private var gate = TargetEventGate()
    private var builder = NavigationSampleBuilder()
    public private(set) var excludedEvents = 0
    public init() {}
    public mutating func consume(_ event: BehaviorEvent) -> [NavigationSample] {
        var output: [NavigationSample] = []
        if event.sessionID != session || event.kind == .sessionStart {
            output = stamp(builder.finish(reason: "session_changed"))
            builder = NavigationSampleBuilder(); gate.reset(); session = event.sessionID
            policyVersion = event.data["targetEligibilityPolicyVersion"]?.string
            enabled = event.kind == .sessionStart &&
                policyVersion.map(TargetEligibilityPolicy.supportedVersions.contains) == true
        }
        guard enabled else { excludedEvents += 1; return output }
        if gate.accept(event) {
            output += stamp(builder.consume(event))
        } else {
            excludedEvents += 1
            var gap = event; gap.kind = .captureGap; gap.app = nil
            gap.data = ["reason": .string("target_eligibility_boundary")]
            _ = gate.accept(gap)
            output += stamp(builder.consume(gap))
        }
        return output
    }
    public mutating func finish() -> [NavigationSample] { stamp(builder.finish()) }
    private func stamp(_ values: [NavigationSample]) -> [NavigationSample] {
        values.map { value in
            var result = value; result.targetEligibilityPolicyVersion = policyVersion; return result
        }
    }
}
