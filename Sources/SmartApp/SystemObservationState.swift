import Foundation

enum SystemObservationSignal: String, CaseIterable {
    case systemSleep = "system_sleep"
    case screenLocked = "screen_locked"
    case sessionInactive = "session_inactive"
    case displaySleep = "display_sleep"
}

struct SystemObservationState {
    private(set) var active: Set<SystemObservationSignal> = []
    var blocksObservation: Bool { !active.subtracting([.displaySleep]).isEmpty }
    var permitsContinuity: Bool { blocksObservation && !active.contains(.sessionInactive) }
    var reasons: [String] { active.subtracting([.displaySleep]).map(\.rawValue).sorted() }
    mutating func update(_ signal: SystemObservationSignal, active enabled: Bool) -> Bool {
        let previous = active
        if enabled { active.insert(signal) } else { active.remove(signal) }
        return previous != active
    }
}
