import AppKit
import func Carbon.IsSecureEventInputEnabled
import IntentCore

/// Invocation-only input. Never reads typed text or forwards events to the data recorder.
@MainActor
final class InputMonitor {
    var onInvoke: (() -> Void)?
    var onCalibration: ((ShakeCalibration) -> Void)?
    var onCalibrationProgress: ((Int) -> Void)?
    private var globalToken: Any?
    private var localToken: Any?
    private var detector = ShakeDetector()
    private var commandDetector = CommandDoubleTapDetector()
    private var calibrator = ShakeCalibrator()
    private var calibrationStartsAt: Double?
    private var shakeEnabled = false
    private var commandEnabled = false

    func start(shakeEnabled: Bool, commandEnabled: Bool, configuration: ShakeConfiguration,
               calibrating: Bool = false) -> Bool {
        stop()
        self.shakeEnabled = shakeEnabled; self.commandEnabled = commandEnabled
        detector = ShakeDetector(configuration: configuration)
        if calibrating { calibrationStartsAt = ProcessInfo.processInfo.systemUptime + 1 }
        guard shakeEnabled || commandEnabled || calibrating else { return false }
        var mask: NSEvent.EventTypeMask = [.leftMouseDown, .rightMouseDown, .otherMouseDown,
                                         .leftMouseDragged, .rightMouseDragged, .otherMouseDragged, .scrollWheel]
        if shakeEnabled || calibrating { mask.insert(.mouseMoved) }
        if commandEnabled { mask.formUnion([.flagsChanged, .keyDown]) }
        globalToken = NSEvent.addGlobalMonitorForEvents(matching: mask) { [weak self] event in
            MainActor.assumeIsolated { self?.handle(event) }
        }
        localToken = NSEvent.addLocalMonitorForEvents(matching: mask) { [weak self] event in
            MainActor.assumeIsolated { self?.handle(event) }
            return event
        }
        return globalToken != nil && localToken != nil
    }

    func stop() {
        if let globalToken { NSEvent.removeMonitor(globalToken) }
        if let localToken { NSEvent.removeMonitor(localToken) }
        globalToken = nil; localToken = nil; calibrationStartsAt = nil
        detector = ShakeDetector(); commandDetector = CommandDoubleTapDetector(); calibrator.reset()
    }

    private func handle(_ event: NSEvent) {
        guard !IsSecureEventInputEnabled() else {
            detector.reset(); commandDetector.cancel(); calibrator.reset()
            return
        }
        switch event.type {
        case .mouseMoved:
            // Use screen coordinates for both local and global events, including window crossings.
            let x = NSEvent.mouseLocation.x
            let dragging = NSEvent.pressedMouseButtons != 0
            if let startsAt = calibrationStartsAt {
                guard event.timestamp >= startsAt else { return }
                if let result = calibrator.move(x: x, time: event.timestamp, dragging: dragging) {
                    onCalibration?(result)
                } else { onCalibrationProgress?(calibrator.reversalCount) }
            } else if shakeEnabled, detector.move(x: x, time: event.timestamp, dragging: dragging) {
                commandDetector.cancel(); onInvoke?()
            }
        case .flagsChanged where commandEnabled && calibrationStartsAt == nil:
            let otherModifiers: NSEvent.ModifierFlags = [.shift, .control, .option, .function]
            if commandDetector.flagsChanged(keyCode: event.keyCode,
                commandDown: event.modifierFlags.contains(.command),
                otherModifiersDown: !event.modifierFlags.intersection(otherModifiers).isEmpty,
                time: event.timestamp) {
                detector.reset(); onInvoke?()
            }
        case .keyDown:
            // Seeing another key is sufficient to reject shortcuts; do not inspect its contents.
            commandDetector.cancel()
        default:
            commandDetector.cancel(); detector.reset(); calibrator.reset()
        }
    }
}
