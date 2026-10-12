import Foundation

/// A paired, explicitly observed inactivity interval is not an unknown capture loss.
/// Old readers still see capture.gap and conservatively break the sequence.
public struct KnownInactivityTracker {
    public enum Disposition { case normal, suspended, resumed, filtered, boundary }
    private struct Pause {
        var id: String
        var start: Double
        var wallStart: Int64
        var end: Double?
        var context: BehaviorEvent
    }
    private var session: String?
    private var lastContext: BehaviorEvent?
    private var pause: Pause?
    private var excludedScopeID: String?
    private var excludedMs = 0.0
    private var rawMs = 0.0
    public private(set) var activeMs = 0.0
    public var isSuspended: Bool { pause != nil }
    public init() {}

    public mutating func consume(_ event: BehaviorEvent) -> Disposition {
        if event.sessionID != session || event.kind == .sessionStart {
            self = Self(); session = event.sessionID
        }
        rawMs = event.monotonicMs
        activeMs = rawMs - excludedMs
        if let pause { activeMs -= max(0, min(event.monotonicMs, pause.end ?? event.monotonicMs) - pause.start) }
        if event.kind == .observationScope {
            let phase = event.data["phase"]?.string
            let id = event.data["scopeID"]?.string
            if phase == "begin", excludedScopeID == nil, let id, UUID(uuidString: id) != nil {
                invalidate(); excludedScopeID = id; return .filtered
            }
            if phase == "end", excludedScopeID == id, id != nil {
                excludedScopeID = nil; return .filtered
            }
            invalidate(); excludedScopeID = nil; return .boundary
        }
        if event.kind == .captureGap {
            excludedScopeID = nil
            let data = event.data
            if data["pauseVersion"]?.string == "known-inactivity-v1",
               let id = data["pauseID"]?.string, UUID(uuidString: id) != nil {
                let phase = data["pausePhase"]?.string
                if phase == "begin", pause == nil, let context = lastContext,
                   let reasons = data["pauseReasons"]?.array, !reasons.isEmpty,
                   reasons.allSatisfy({ ["system_sleep", "screen_locked"].contains($0.string ?? "") }) {
                    pause = Pause(id: id, start: event.monotonicMs, wallStart: event.timestampMs, context: context)
                    return .suspended
                }
                if phase == "end", var current = pause, current.id == id, current.end == nil,
                   event.monotonicMs >= current.start, event.timestampMs >= current.wallStart {
                    current.end = event.monotonicMs; pause = current
                    return .suspended
                }
            }
            invalidate()
            return .boundary
        }
        if event.kind == .sessionEnd { invalidate(); excludedScopeID = nil; return .boundary }
        if excludedScopeID != nil {
            if event.kind == .contextChanged {
                invalidate(); excludedScopeID = nil; lastContext = event; return .boundary
            }
            return .suspended
        }
        if let current = pause {
            if event.kind == .contextChanged {
                let valid = current.end != nil &&
                    event.data["continuityPauseID"]?.string == current.id &&
                    event.data["resolvesContextEventID"]?.string == current.context.id &&
                    event.data["reason"]?.string == "resolution_update" &&
                    event.data["targetID"] == current.context.data["targetID"] &&
                    event.data["entranceID"] == current.context.data["entranceID"]
                if valid {
                    excludedMs += current.end! - current.start
                    pause = nil; lastContext = event
                    activeMs = event.monotonicMs - excludedMs
                    return .resumed
                }
                invalidate(); lastContext = event
                return .boundary
            }
            return .suspended
        }
        if event.kind == .contextChanged { lastContext = event }
        return .normal
    }
    private mutating func invalidate() {
        // Even an unpaired pause never becomes observed use time. Consumers close
        // their sequence at this boundary, so no time is stitched across it.
        if let pause { excludedMs += max(0, min(rawMs, pause.end ?? rawMs) - pause.start) }
        pause = nil; lastContext = nil
        activeMs = rawMs - excludedMs
    }
}
