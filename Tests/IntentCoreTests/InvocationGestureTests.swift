import XCTest
@testable import IntentCore

final class InvocationGestureTests: XCTestCase {
    private func shake(_ detector: inout ShakeDetector, amplitude: Double = 40,
                       interval: Double = 0.06, start: Double = 1,
                       stepsPerLeg: Int = 1, dragging: Bool = false) -> Int {
        var count = 0
        for step in 0...(6 * stepsPerLeg) {
            let leg = step / stepsPerLeg
            let fraction = Double(step % stepsPerLeg) / Double(stepsPerLeg)
            let x = leg.isMultiple(of: 2) ? fraction * amplitude : (1 - fraction) * amplitude
            if detector.move(x: x, time: start + Double(step) * interval / Double(stepsPerLeg), dragging: dragging) {
                count += 1
            }
        }
        return count
    }

    func testSensitivityChangesRequiredAmplitude() {
        var low = ShakeDetector(configuration: ShakeConfiguration(sensitivity: 0))
        var high = ShakeDetector(configuration: ShakeConfiguration(sensitivity: 1))
        XCTAssertEqual(shake(&low, amplitude: 20), 0)
        XCTAssertEqual(shake(&high, amplitude: 20), 1)
    }

    func testSensitivityAlsoAccommodatesSlowerShake() {
        var low = ShakeDetector(configuration: ShakeConfiguration(sensitivity: 0))
        var high = ShakeDetector(configuration: ShakeConfiguration(sensitivity: 1))
        XCTAssertEqual(shake(&low, amplitude: 80, interval: 0.16), 0)
        XCTAssertEqual(shake(&high, amplitude: 80, interval: 0.16), 1)
    }

    func testHighRateMouseWithSubPointEventsStillTriggers() {
        var detector = ShakeDetector()
        XCTAssertEqual(shake(&detector, stepsPerLeg: 80), 1)
    }

    func testJitterStraightMovementAndSparseReversalsDoNotTrigger() {
        var detector = ShakeDetector(configuration: ShakeConfiguration(sensitivity: 1))
        XCTAssertEqual(shake(&detector, amplitude: 2), 0)
        for i in 0..<100 {
            XCTAssertFalse(detector.move(x: Double(i) * 40, time: 3 + Double(i) * 0.01, dragging: false))
        }
        XCTAssertEqual(shake(&detector, interval: 0.4, start: 6), 0)
    }

    func testDraggingAndCooldownClearPartialGestures() {
        var detector = ShakeDetector()
        XCTAssertEqual(shake(&detector, dragging: true), 0)
        XCTAssertEqual(shake(&detector, start: 2), 1)
        XCTAssertEqual(shake(&detector, start: 2.5), 0)
        XCTAssertEqual(shake(&detector, start: 6), 1)
    }

    func testCalibrationMeasuresStrengthAndReplaysAtItsOwnPace() throws {
        var calibrator = ShakeCalibrator()
        var result: ShakeCalibration?
        for (index, x) in [400.0, 470, 400, 470, 400, 470].enumerated() {
            result = calibrator.move(x: x, time: 1 + Double(index) * 0.2, dragging: false) ?? result
        }
        let calibration = try XCTUnwrap(result)
        XCTAssertEqual(calibration.amplitude, 70)
        XCTAssertEqual(calibration.duration, 1, accuracy: 0.0001)
        XCTAssertTrue(calibration.isValid)
        var detector = ShakeDetector(configuration: ShakeConfiguration(calibration: calibration))
        XCTAssertEqual(shake(&detector, amplitude: 70, interval: 0.2), 1)
        let data = try JSONEncoder().encode(calibration)
        let fields = try XCTUnwrap(JSONSerialization.jsonObject(with: data) as? [String: Any])
        XCTAssertEqual(Set(fields.keys), ["amplitude", "duration", "recordedAt"])
        XCTAssertEqual(try JSONDecoder().decode(ShakeCalibration.self, from: data), calibration)
    }

    func testCalibrationRejectsDraggingTinyMotionAndIncompleteDemonstrations() {
        var calibrator = ShakeCalibrator()
        for i in 0..<30 {
            XCTAssertNil(calibrator.move(x: i.isMultiple(of: 2) ? 0 : 100, time: Double(i) * 0.05, dragging: true))
        }
        for i in 0..<30 {
            XCTAssertNil(calibrator.move(x: i.isMultiple(of: 2) ? 0 : 4, time: 3 + Double(i) * 0.05, dragging: false))
        }
        for (index, x) in [0.0, 50, 0, 50].enumerated() {
            XCTAssertNil(calibrator.move(x: x, time: 6 + Double(index) * 0.06, dragging: false))
        }
        calibrator.reset()
        XCTAssertEqual(calibrator.reversalCount, 0)
        XCTAssertNil(calibrator.move(x: 0, time: 6.25, dragging: false))
    }

    func testInvalidPreferencesFallBackToUsableConfiguration() {
        XCTAssertEqual(ShakeConfiguration(sensitivity: .nan), ShakeConfiguration())
        XCTAssertEqual(ShakeConfiguration(sensitivity: -20), ShakeConfiguration(sensitivity: 0))
        XCTAssertEqual(ShakeConfiguration(sensitivity: 20), ShakeConfiguration(sensitivity: 1))
        let invalid = ShakeCalibration(amplitude: .infinity, duration: -1)
        XCTAssertFalse(invalid.isValid)
        XCTAssertEqual(ShakeConfiguration(calibration: invalid), ShakeConfiguration())
    }

    private func command(_ detector: inout CommandDoubleTapDetector, _ down: Bool,
                         _ time: Double, key: UInt16 = 55, other: Bool = false) -> Bool {
        detector.flagsChanged(keyCode: key, commandDown: down, otherModifiersDown: other, time: time)
    }

    func testDoubleCommandRequiresTwoCompleteTapsOnEitherSide() {
        for key: UInt16 in [54, 55] {
            var detector = CommandDoubleTapDetector()
            XCTAssertFalse(command(&detector, true, 1, key: key))
            XCTAssertFalse(command(&detector, false, 1.07, key: key))
            XCTAssertFalse(command(&detector, true, 1.2, key: key))
            XCTAssertTrue(command(&detector, false, 1.27, key: key))
        }
    }

    func testCommandShortcutOnEitherTapNeverInvokes() {
        for interruptedTap in [0, 1] {
            var detector = CommandDoubleTapDetector()
            for tap in 0..<2 {
                let time = 1 + Double(tap) * 0.2
                XCTAssertFalse(command(&detector, true, time))
                if tap == interruptedTap { detector.cancel() } // keyDown, click, scroll, or drag
                XCTAssertFalse(command(&detector, false, time + 0.07))
            }
        }
    }

    func testCommandChordsLongHoldsAndSlowTapsDoNotInvoke() {
        var detector = CommandDoubleTapDetector()
        XCTAssertFalse(command(&detector, true, 1, other: true))
        XCTAssertFalse(command(&detector, false, 1.1))
        XCTAssertFalse(command(&detector, true, 1.2))
        XCTAssertFalse(command(&detector, false, 1.6))
        XCTAssertFalse(command(&detector, true, 2))
        XCTAssertFalse(command(&detector, false, 2.07))
        XCTAssertFalse(command(&detector, true, 3))
        XCTAssertFalse(command(&detector, false, 3.07))
    }

    func testAlternatingAndOverlappingCommandKeysDoNotInvoke() {
        var detector = CommandDoubleTapDetector()
        XCTAssertFalse(command(&detector, true, 1))
        XCTAssertFalse(command(&detector, false, 1.06))
        XCTAssertFalse(command(&detector, true, 1.16, key: 54))
        XCTAssertFalse(command(&detector, false, 1.22, key: 54))
        detector.cancel()
        XCTAssertFalse(command(&detector, true, 2))
        XCTAssertFalse(command(&detector, true, 2.02, key: 54))
        XCTAssertFalse(command(&detector, true, 2.05))
        XCTAssertFalse(command(&detector, false, 2.07, key: 54))
    }

    func testInterveningModifierOrActionBreaksDoubleTap() {
        var detector = CommandDoubleTapDetector()
        XCTAssertFalse(command(&detector, true, 1))
        XCTAssertFalse(command(&detector, false, 1.06))
        XCTAssertFalse(command(&detector, false, 1.1, key: 56))
        XCTAssertFalse(command(&detector, true, 1.2))
        XCTAssertFalse(command(&detector, false, 1.27))
        detector.cancel()
        XCTAssertFalse(command(&detector, true, 1.4))
        XCTAssertFalse(command(&detector, false, 1.47))
    }

    func testCommandCooldownSuppressesRepeatedInvocationAndThenRecovers() {
        var detector = CommandDoubleTapDetector()
        var invocations = 0
        for tap in 0..<4 {
            let time = 1 + Double(tap) * 0.14
            XCTAssertFalse(command(&detector, true, time))
            if command(&detector, false, time + 0.06) { invocations += 1 }
        }
        XCTAssertEqual(invocations, 1)
        XCTAssertFalse(command(&detector, true, 3))
        XCTAssertFalse(command(&detector, false, 3.06))
        XCTAssertFalse(command(&detector, true, 3.2))
        XCTAssertTrue(command(&detector, false, 3.26))
    }
}
