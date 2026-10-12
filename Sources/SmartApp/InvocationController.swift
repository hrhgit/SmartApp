import Foundation
import IntentCore
import Observation

@MainActor @Observable
final class InvocationController {
    var shakeEnabled: Bool { didSet { saveAndApply() } }
    var doubleCommandEnabled: Bool { didSet { saveAndApply() } }
    var sensitivity: Double { didSet { saveAndApply() } }
    private(set) var calibration: ShakeCalibration?
    private(set) var isCalibrating = false
    private(set) var calibrationSeconds = 0
    private(set) var calibrationReversals = 0
    private(set) var calibrationMessage: LocalizedStringResource?
    private(set) var isCollecting = false
    private(set) var isMonitoring = false

    @ObservationIgnored var onInvoke: (() -> Void)?
    @ObservationIgnored private let input: InputMonitor
    @ObservationIgnored private let defaults: UserDefaults
    @ObservationIgnored private var calibrationTask: Task<Void, Never>?

    init(defaults: UserDefaults = .standard, input: InputMonitor = InputMonitor()) {
        self.defaults = defaults; self.input = input
        shakeEnabled = defaults.object(forKey: "shakeEnabledV2") as? Bool ?? true
        doubleCommandEnabled = defaults.object(forKey: "doubleCommandEnabledV1") as? Bool ?? true
        sensitivity = ShakeConfiguration.normalizedSensitivity(
            defaults.object(forKey: "shakeSensitivityV1") as? Double ?? 0.5)
        if let data = defaults.data(forKey: "shakeCalibrationV1"),
           let saved = try? JSONDecoder().decode(ShakeCalibration.self, from: data), saved.isValid {
            calibration = saved
        }
        input.onInvoke = { [weak self] in
            guard let self, !self.isCalibrating else { return }
            self.onInvoke?()
        }
        input.onCalibrationProgress = { [weak self] in self?.calibrationReversals = $0 }
        input.onCalibration = { [weak self] in self?.finishCalibration($0) }
    }

    func setCollecting(_ collecting: Bool) {
        cancelCalibration()
        isCollecting = collecting
        refreshMonitoring()
    }

    func refreshMonitoring() {
        isMonitoring = input.start(shakeEnabled: isCollecting && shakeEnabled,
            commandEnabled: isCollecting && doubleCommandEnabled && !isCalibrating,
            configuration: ShakeConfiguration(sensitivity: sensitivity, calibration: calibration),
            calibrating: isCalibrating)
    }

    func beginCalibration() {
        guard !isCalibrating else { return }
        isCalibrating = true; calibrationSeconds = 6; calibrationReversals = 0
        calibrationMessage = nil
        refreshMonitoring()
        guard isMonitoring else {
            isCalibrating = false; input.stop()
            calibrationMessage = "无法连接鼠标事件，请重试。"
            return
        }
        calibrationTask = Task { [weak self] in
            for remaining in (0...5).reversed() {
                do { try await Task.sleep(for: .seconds(1)) } catch { return }
                guard let self, self.isCalibrating else { return }
                self.calibrationSeconds = remaining
            }
            guard let self else { return }
            self.cancelCalibration()
            self.calibrationMessage = "未识别到完整晃动，原设置已保留。请在短时间内左右来回晃动，至少转向 4 次。"
        }
    }

    func cancelCalibration() {
        guard isCalibrating else { return }
        calibrationTask?.cancel(); calibrationTask = nil
        isCalibrating = false; calibrationSeconds = 0; calibrationReversals = 0
        calibrationMessage = "已取消示范，原设置已保留。"
        refreshMonitoring()
    }

    func restoreDefaults() {
        cancelCalibration()
        calibration = nil
        defaults.removeObject(forKey: "shakeCalibrationV1")
        sensitivity = 0.5
        saveAndApply()
        calibrationMessage = "已恢复默认晃动灵敏度。"
    }

    func stop() {
        calibrationTask?.cancel(); calibrationTask = nil; isCalibrating = false
        isCollecting = false; input.stop(); isMonitoring = false
    }

    private func finishCalibration(_ result: ShakeCalibration) {
        guard isCalibrating, result.isValid else { return }
        calibrationTask?.cancel(); calibrationTask = nil
        isCalibrating = false; calibrationSeconds = 0; calibrationReversals = 4
        calibration = result
        if let data = try? JSONEncoder().encode(result) { defaults.set(data, forKey: "shakeCalibrationV1") }
        // The midpoint now represents this user's demonstration, with room to fine-tune either way.
        sensitivity = 0.5
        saveAndApply()
        calibrationMessage = "已记录示范强度并应用。可继续拖动滑块微调。"
    }

    private func saveAndApply() {
        defaults.set(shakeEnabled, forKey: "shakeEnabledV2")
        defaults.set(doubleCommandEnabled, forKey: "doubleCommandEnabledV1")
        defaults.set(sensitivity, forKey: "shakeSensitivityV1")
        refreshMonitoring()
    }
}
