import Foundation

/// Only aggregate calibration parameters are saved in app preferences, never pointer samples.
public struct ShakeCalibration: Codable, Equatable, Sendable {
    public let amplitude: Double
    public let duration: Double
    public let recordedAt: Date

    public init(amplitude: Double, duration: Double, recordedAt: Date = Date()) {
        self.amplitude = amplitude; self.duration = duration; self.recordedAt = recordedAt
    }

    public var isValid: Bool {
        amplitude.isFinite && amplitude >= 8 && amplitude <= 10_000 &&
        duration.isFinite && duration > 0 && duration <= 1.6
    }
}

public struct ShakeConfiguration: Equatable, Sendable {
    public let minimumTravel: Double
    public let timeWindow: Double

    public init(sensitivity: Double = 0.5, calibration: ShakeCalibration? = nil) {
        let sensitivity = Self.normalizedSensitivity(sensitivity)
        let calibration = calibration.flatMap { $0.isValid ? $0 : nil }
        let travel = calibration.map { min(200, max(8, $0.amplitude * 0.65)) } ?? 25
        let window = calibration.map { min(1.6, max(0.4, $0.duration * 1.5)) } ?? 0.8
        minimumTravel = max(4, travel * pow(2, (0.5 - sensitivity) * 2))
        timeWindow = window * (0.75 + sensitivity * 0.5)
    }

    public static func normalizedSensitivity(_ value: Double) -> Double {
        value.isFinite ? min(1, max(0, value)) : 0.5
    }
}

/// A bounded, transient detector shared by calibration and normal invocation.
/// Positions and individual legs cannot be encoded or exported.
private struct ShakeMotion: Sendable {
    private struct Leg: Sendable {
        let amplitude: Double
        let start: Double
    }
    private var lastX: Double?
    private var direction = 0.0
    private var travel = 0.0
    private var legStart = 0.0
    private var previousTime = 0.0
    private var legs: [Leg] = []
    var reversalCount: Int { legs.count }

    mutating func reset() { self = Self() }

    mutating func move(x: Double, time: Double, dragging: Bool,
                       minimumTravel: Double, timeWindow: Double) -> ShakeCalibration? {
        guard x.isFinite, time.isFinite, !dragging else { reset(); return nil }
        if time < previousTime || time - previousTime > 0.35 { reset() }
        let previousTime = self.previousTime
        self.previousTime = time
        legs.removeAll { time - $0.start > timeWindow }
        guard let previous = lastX else { lastX = x; legStart = time; return nil }
        let dx = x - previous
        // Retain the reference point so sub-two-point events accumulate on high-rate mice.
        guard abs(dx) >= 2 else { return nil }
        lastX = x
        let sign = dx > 0 ? 1.0 : -1.0
        if sign == direction { travel += abs(dx) }
        else {
            if travel >= minimumTravel, time - legStart <= timeWindow {
                legs.append(Leg(amplitude: travel, start: legStart))
            }
            direction = sign; travel = abs(dx); legStart = previousTime
        }
        guard legs.count >= 4 else { return nil }
        let amplitudes = legs.suffix(4).map(\.amplitude).sorted()
        let result = ShakeCalibration(amplitude: (amplitudes[1] + amplitudes[2]) / 2,
                                      duration: time - legs[legs.count - 4].start)
        reset()
        return result
    }
}

public struct ShakeDetector: Sendable {
    private let configuration: ShakeConfiguration
    private var motion = ShakeMotion()
    private var cooldownUntil = 0.0

    public init(configuration: ShakeConfiguration = ShakeConfiguration()) {
        self.configuration = configuration
    }
    public mutating func reset() { motion.reset() }
    public mutating func move(x: Double, time: Double, dragging: Bool) -> Bool {
        guard time >= cooldownUntil else { reset(); return false }
        guard motion.move(x: x, time: time, dragging: dragging,
                          minimumTravel: configuration.minimumTravel,
                          timeWindow: configuration.timeWindow) != nil else { return false }
        cooldownUntil = time + 3
        return true
    }
}

public struct ShakeCalibrator: Sendable {
    private var motion = ShakeMotion()
    public var reversalCount: Int { motion.reversalCount }
    public init() {}
    public mutating func reset() { motion.reset() }
    public mutating func move(x: Double, time: Double, dragging: Bool) -> ShakeCalibration? {
        motion.move(x: x, time: time, dragging: dragging, minimumTravel: 8, timeWindow: 1.6)
    }
}

/// Detects two complete, unaccompanied taps of the same Command key.
/// No key text or non-modifier key codes are needed or retained.
public struct CommandDoubleTapDetector: Sendable {
    private var pressedKey: UInt16?
    private var pressedAt = 0.0
    private var firstKey: UInt16?
    private var firstReleasedAt: Double?
    private var cooldownUntil = 0.0
    public init() {}

    public mutating func cancel() {
        pressedKey = nil; firstKey = nil; firstReleasedAt = nil
    }

    public mutating func flagsChanged(keyCode: UInt16, commandDown: Bool,
                                      otherModifiersDown: Bool, time: Double) -> Bool {
        guard time.isFinite, time >= cooldownUntil, !otherModifiersDown,
              keyCode == 54 || keyCode == 55 else { cancel(); return false }
        if commandDown {
            // A second modifier transition while held includes overlapping left/right Command.
            guard pressedKey == nil else { cancel(); return false }
            if let released = firstReleasedAt,
               time < released || time - released > 0.35 || firstKey != keyCode {
                cancel()
            }
            pressedKey = keyCode; pressedAt = time
            return false
        }
        guard pressedKey == keyCode, time >= pressedAt, time - pressedAt <= 0.25 else {
            cancel(); return false
        }
        pressedKey = nil
        if firstReleasedAt != nil, firstKey == keyCode {
            cancel(); cooldownUntil = time + 0.6
            return true
        }
        firstKey = keyCode; firstReleasedAt = time
        return false
    }
}
