import Foundation

public struct TrainingSample: Codable, Equatable, Sendable {
    public var schemaVersion = 1
    public var sessionID: String
    public var triggerEventID: String
    public var triggerTimestampMs: Int64
    public var triggerMonotonicMs: Double
    public var sourceApp: String
    public var features: [String: Double]
    public var inputEventIDs: [String]
    public var targetApp: String
    public var outcomeEventID: String?
    public var outcomeDelayMs: Double?
    public var labelMethod = "next_app_activation_within_2s"
}

/// Streaming builder: bounded rolling history, independent observation and prediction windows.
/// Supply events in session/monotonic order. Gaps invalidate pending labels and history.
public struct SampleBuilder {
    public static let lookbackMs = 3_000.0
    public static let horizonMs = 2_000.0
    public static let triggerIntervalMs = 750.0
    public static let stayLabel = "__stay__"
    private var sessionID: String?
    private var history: [BehaviorEvent] = []
    private var pending: [TrainingSample] = []
    private var lastTrigger = -Double.infinity
    private var lastObserved = -Double.infinity

    public init() {}

    public mutating func consume(_ event: BehaviorEvent) -> [TrainingSample] {
        if sessionID != event.sessionID {
            reset(); sessionID = event.sessionID
        }
        // A gap in callbacks/heartbeats is a missing observation interval, not a negative label.
        if event.kind == .captureGap || event.kind == .observationScope || event.kind == .sessionEnd ||
            event.monotonicMs < lastObserved || (lastObserved.isFinite && event.monotonicMs - lastObserved > 5_000) {
            reset(); sessionID = event.sessionID
            lastObserved = event.monotonicMs
            return []
        }
        lastObserved = event.monotonicMs
        var completed: [TrainingSample] = []
        var remaining: [TrainingSample] = []
        for var candidate in pending {
            let delta = event.monotonicMs - candidate.triggerMonotonicMs
            if delta > 0 && delta <= Self.horizonMs && event.kind == .appActivated,
               let target = event.app?.bundleID, target != candidate.sourceApp {
                candidate.targetApp = target
                candidate.outcomeEventID = event.id
                candidate.outcomeDelayMs = delta
                completed.append(candidate)
            } else if delta >= Self.horizonMs {
                completed.append(candidate)
            } else {
                remaining.append(candidate)
            }
        }
        pending = remaining
        history.removeAll { event.monotonicMs - $0.monotonicMs > Self.lookbackMs }
        if event.kind.isInput { history.append(event) }
        if event.kind.isTrigger, let app = event.app,
           event.monotonicMs - lastTrigger >= Self.triggerIntervalMs {
            // No target application, future event, or user annotation is included in features.
            let inputs = history.filter { $0.app?.bundleID == app.bundleID }
            pending.append(TrainingSample(
                sessionID: event.sessionID, triggerEventID: event.id,
                triggerTimestampMs: event.timestampMs, triggerMonotonicMs: event.monotonicMs,
                sourceApp: app.bundleID, features: features(inputs), inputEventIDs: inputs.map(\.id),
                targetApp: Self.stayLabel
            ))
            lastTrigger = event.monotonicMs
        }
        return completed
    }

    private mutating func reset() {
        history.removeAll(keepingCapacity: true); pending.removeAll(keepingCapacity: true)
        lastTrigger = -Double.infinity; lastObserved = -Double.infinity
    }

    private func features(_ inputs: [BehaviorEvent]) -> [String: Double] {
        var result: [String: Double] = [
            "pointer_moves": 0, "pointer_drags": 0, "clicks": 0,
            "scroll_events": 0, "scroll_points_abs_x": 0, "scroll_points_abs_y": 0,
            "scroll_lines_abs_x": 0, "scroll_lines_abs_y": 0,
            "typing_keys": 0, "shortcut_keys": 0, "navigation_keys": 0,
            "gesture_events": 0, "pointer_distance": 0, "input_events": Double(inputs.count)
        ]
        var previous: BehaviorEvent?
        for event in inputs {
            switch event.kind {
            case .pointerMoved: result["pointer_moves", default: 0] += 1
            case .pointerDragged: result["pointer_drags", default: 0] += 1
            case .mouseDown: result["clicks", default: 0] += 1
            case .scroll:
                result["scroll_events", default: 0] += 1
                let unit = event.data["deltaUnit"]?.string == "points" ? "points" : "lines"
                result["scroll_\(unit)_abs_x", default: 0] += abs(event.data["deltaX"]?.number ?? 0)
                result["scroll_\(unit)_abs_y", default: 0] += abs(event.data["deltaY"]?.number ?? 0)
            case .keyDown:
                let category = event.data["category"]?.string ?? "typing"
                result[category + "_keys", default: 0] += 1
            case .magnify, .rotate, .swipe: result["gesture_events", default: 0] += 1
            default: break
            }
            if let x = event.data["x"]?.number, let y = event.data["y"]?.number {
                if let previous,
                   let px = previous.data["x"]?.number, let py = previous.data["y"]?.number,
                   event.data["displayID"] == previous.data["displayID"] {
                    result["pointer_distance", default: 0] += hypot(x - px, y - py)
                }
                previous = event
                result["last_x"] = x; result["last_y"] = y
            }
        }
        if let first = inputs.first, let last = inputs.last {
            result["observed_input_span_ms"] = last.monotonicMs - first.monotonicMs
        }
        return result
    }
}
