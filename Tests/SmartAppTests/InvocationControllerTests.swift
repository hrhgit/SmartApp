import XCTest
import IntentCore
@testable import SmartApp

final class InvocationControllerTests: XCTestCase {
    @MainActor private func withController(_ body: (InvocationController, InputMonitor, UserDefaults) throws -> Void) rethrows {
        let suite = "local.herh.SmartApp.invocation-test.\(UUID().uuidString)"
        let defaults = UserDefaults(suiteName: suite)!
        let input = InputMonitor()
        let controller = InvocationController(defaults: defaults, input: input)
        defer { controller.stop(); defaults.removePersistentDomain(forName: suite) }
        try body(controller, input, defaults)
    }

    @MainActor func testPreferencesSaveImmediatelyAndReloadIndependently() {
        withController { controller, _, defaults in
            controller.shakeEnabled = false
            controller.doubleCommandEnabled = true
            controller.sensitivity = 0.83
            let restored = InvocationController(defaults: defaults)
            defer { restored.stop() }
            XCTAssertFalse(restored.shakeEnabled)
            XCTAssertTrue(restored.doubleCommandEnabled)
            XCTAssertEqual(restored.sensitivity, 0.83)
            XCTAssertFalse(restored.isMonitoring)
        }
    }

    @MainActor func testCalibrationWorksWhilePausedAndPersistsOnlyAggregates() throws {
        try withController { controller, input, defaults in
            var invocations = 0
            controller.onInvoke = { invocations += 1 }
            controller.beginCalibration()
            XCTAssertTrue(controller.isCalibrating)
            input.onInvoke?()
            XCTAssertEqual(invocations, 0)
            let sample = ShakeCalibration(amplitude: 90, duration: 0.65)
            input.onCalibration?(sample)
            XCTAssertFalse(controller.isCalibrating)
            XCTAssertFalse(controller.isMonitoring)
            XCTAssertFalse(controller.isCollecting)
            XCTAssertEqual(controller.sensitivity, 0.5)
            let stored = try XCTUnwrap(defaults.data(forKey: "shakeCalibrationV1"))
            XCTAssertEqual(try JSONDecoder().decode(ShakeCalibration.self, from: stored), sample)
            let restored = InvocationController(defaults: defaults)
            defer { restored.stop() }
            XCTAssertEqual(restored.calibration, sample)
        }
    }

    @MainActor func testCancelAndInvalidDemonstrationPreservePreviousSettings() {
        withController { controller, input, _ in
            controller.beginCalibration()
            let sample = ShakeCalibration(amplitude: 60, duration: 0.55)
            input.onCalibration?(sample)
            controller.sensitivity = 0.73
            controller.beginCalibration()
            input.onCalibration?(ShakeCalibration(amplitude: 0, duration: 0))
            XCTAssertTrue(controller.isCalibrating)
            controller.cancelCalibration()
            XCTAssertEqual(controller.calibration, sample)
            XCTAssertEqual(controller.sensitivity, 0.73)
            XCTAssertFalse(controller.isMonitoring)
        }
    }

    @MainActor func testResetClearsCalibrationAndPreservesInvocationToggles() {
        withController { controller, input, defaults in
            controller.shakeEnabled = false
            controller.doubleCommandEnabled = false
            controller.beginCalibration()
            input.onCalibration?(ShakeCalibration(amplitude: 60, duration: 0.55))
            controller.restoreDefaults()
            XCTAssertNil(controller.calibration)
            XCTAssertNil(defaults.data(forKey: "shakeCalibrationV1"))
            XCTAssertEqual(controller.sensitivity, 0.5)
            XCTAssertFalse(controller.shakeEnabled)
            XCTAssertFalse(controller.doubleCommandEnabled)
        }
    }
}
