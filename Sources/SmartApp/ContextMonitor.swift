import AppKit
import ApplicationServices
import IntentCore

struct WindowObservation {
    let destination: Destination
    let entrance: TargetEntrance
    let document: URL?
    var minimized: Bool? = nil
}

@MainActor
final class ContextMonitor {
    var onActivate: ((NSRunningApplication?) -> Void)?
    var onLifecycle: ((EventKind, NSRunningApplication) -> Void)?
    var onAvailability: ((SystemObservationSignal, Bool) -> Void)?
    var onWindowsChanged: ((pid_t) -> Void)?
    private var tokens: [NSObjectProtocol] = []
    private var distributedTokens: [NSObjectProtocol] = []
    private var windows: [(AXUIElement, String, pid_t)] = []
    private(set) var secureFieldFocused = false
    private(set) var lastWindowStatus = "not_observed"
    private var epoch = UUID().uuidString
    private var observers: [pid_t: AXObserver] = [:]

    static func context(_ app: NSRunningApplication?) -> AppContext? {
        guard let app, let bundleID = app.bundleIdentifier else { return nil }
        return AppContext(bundleID: bundleID, name: app.localizedName ?? bundleID)
    }

    func start() {
        stop(); epoch = UUID().uuidString
        observe(NSWorkspace.didActivateApplicationNotification) { [weak self] app in self?.onActivate?(app) }
        for (name, kind) in [(NSWorkspace.didLaunchApplicationNotification, EventKind.appLaunched),
                             (NSWorkspace.didTerminateApplicationNotification, EventKind.appTerminated)] {
            observe(name) { [weak self] app in
                guard let app else { return }
                self?.onLifecycle?(kind, app)
            }
        }
        for (name, signal, enabled) in [
            (NSWorkspace.willSleepNotification, SystemObservationSignal.systemSleep, true),
            (NSWorkspace.didWakeNotification, .systemSleep, false),
            (NSWorkspace.sessionDidResignActiveNotification, .sessionInactive, true),
            (NSWorkspace.sessionDidBecomeActiveNotification, .sessionInactive, false),
            (NSWorkspace.screensDidSleepNotification, .displaySleep, true),
            (NSWorkspace.screensDidWakeNotification, .displaySleep, false)
        ] {
            observe(name) { [weak self] _ in self?.onAvailability?(signal, enabled) }
        }
        // Workspace session notifications describe user switching, not screen locking.
        // Only a received lock/unlock pair can establish a known lock interval.
        for (name, enabled) in [("com.apple.screenIsLocked", true), ("com.apple.screenIsUnlocked", false)] {
            distributedTokens.append(DistributedNotificationCenter.default().addObserver(
                forName: Notification.Name(name), object: nil, queue: .main) { [weak self] _ in
                    MainActor.assumeIsolated { self?.onAvailability?(.screenLocked, enabled) }
                })
        }
        if let session = CGSessionCopyCurrentDictionary() as? [String: Any],
           let onConsole = session[kCGSessionOnConsoleKey as String] as? Bool, !onConsole {
            onAvailability?(.sessionInactive, true)
        }
    }
    func stop() {
        for token in tokens { NSWorkspace.shared.notificationCenter.removeObserver(token) }
        for token in distributedTokens { DistributedNotificationCenter.default().removeObserver(token) }
        distributedTokens = []
        tokens = []; windows = []; secureFieldFocused = false
        for observer in observers.values { CFRunLoopRemoveSource(CFRunLoopGetMain(), AXObserverGetRunLoopSource(observer), .commonModes) }
        observers = [:]
    }

    /// Window handles are transient activation aids, not a background inventory.
    /// Keep at most the current foreground application's observed window.
    func retainOnlyForegroundWindows(for pid: pid_t?) {
        for key in observers.keys.filter({ observedPID in pid.map { activePID in observedPID != activePID } ?? true }) {
            if let observer = observers.removeValue(forKey: key) {
                CFRunLoopRemoveSource(CFRunLoopGetMain(), AXObserverGetRunLoopSource(observer), .commonModes)
            }
        }
        windows.removeAll { item in pid.map { activePID in item.2 != activePID } ?? true }
    }

    func refreshSecureField(_ app: NSRunningApplication?) {
        secureFieldFocused = false
        guard AXIsProcessTrusted(), let app else { return }
        let application = AXUIElementCreateApplication(app.processIdentifier)
        AXUIElementSetMessagingTimeout(application, 0.05)
        if let focused = element(application, kAXFocusedUIElementAttribute) {
            secureFieldFocused = string(focused, kAXSubroleAttribute) == kAXSecureTextFieldSubrole
        }
        if let window = element(application, kAXFocusedWindowAttribute),
           string(window, kAXSubroleAttribute) == kAXSystemDialogSubrole { secureFieldFocused = true }
    }

    func admission(_ app: NSRunningApplication, rules: [String]) -> AdmissionDecision {
        var facts = ApplicationClassifier.evidence(app)
        let first = TargetEligibilityPolicy.application(facts, rules: rules)
        guard first.disposition == .unknown, first.reason == "work_window_unconfirmed", AXIsProcessTrusted() else { return first }
        // Never enumerate an application's windows to prove eligibility: that
        // would read background state.  At most inspect the focused window of
        // the macOS-frontmost application that we are already observing.
        guard NSWorkspace.shared.frontmostApplication?.processIdentifier == app.processIdentifier else { return first }
        let application = AXUIElementCreateApplication(app.processIdentifier)
        AXUIElementSetMessagingTimeout(application, 0.05)
        if let focused = element(application, kAXFocusedWindowAttribute) {
            let windowFacts = evidence(focused)
            facts.independentWindow = windowFacts.main &&
                TargetEligibilityPolicy.window(windowFacts, owner: .init(.allowed, "probe")).isAllowed
        }
        return TargetEligibilityPolicy.application(facts, rules: rules)
    }

    func installedAdmission(_ bundle: String, rules: [String]) -> AdmissionDecision {
        if let app = NSRunningApplication.runningApplications(withBundleIdentifier: bundle).first { return admission(app, rules: rules) }
        let facts = ApplicationClassifier.evidence(bundleID: bundle,
            url: NSWorkspace.shared.urlForApplication(withBundleIdentifier: bundle), activation: .regular)
        return TargetEligibilityPolicy.application(facts, rules: rules)
    }

    func observeWindows(_ app: NSRunningApplication, readDocuments: Bool, rules: [String]) -> ([WindowObservation], String?) {
        let owner = admission(app, rules: rules)
        guard owner.isAllowed else { lastWindowStatus = "excluded"; return ([], nil) }
        guard AXIsProcessTrusted(), let bundle = app.bundleIdentifier else { lastWindowStatus = "permission_missing"; return ([], nil) }
        let application = AXUIElementCreateApplication(app.processIdentifier)
        AXUIElementSetMessagingTimeout(application, 0.08)
        watch(application, pid: app.processIdentifier)
        let current = element(application, kAXFocusedWindowAttribute)
        if let current, evidence(current).isSystemDialog,
           NSWorkspace.shared.frontmostApplication?.processIdentifier == app.processIdentifier {
            secureFieldFocused = true; lastWindowStatus = "sensitive"; return ([], nil)
        }
        lastWindowStatus = "ready"
        guard let current else { return ([], nil) }
        let decision = TargetEligibilityPolicy.window(evidence(current), owner: owner)
        guard decision.isAllowed else { return ([], nil) }
        let identifier: String
        if let existing = windows.first(where: { $0.2 == app.processIdentifier && CFEqual($0.0, current) }) {
            identifier = existing.1
        } else {
            identifier = "window:\(epoch):\(UUID().uuidString)"
            windows.removeAll { $0.2 == app.processIdentifier }
            windows.append((current, identifier, app.processIdentifier))
        }
        let document = readDocuments ? string(current, kAXDocumentAttribute).flatMap(URL.init(string:)) : nil
        if let observer = observers[app.processIdentifier] {
            for name in [kAXUIElementDestroyedNotification, kAXWindowMiniaturizedNotification, kAXWindowDeminiaturizedNotification] {
                AXObserverAddNotification(observer, current, name as CFString, Unmanaged.passUnretained(self).toOpaque())
            }
        }
        let observation = WindowObservation(destination: Destination(id: identifier, kind: .window,
            name: "\(app.localizedName ?? bundle) · 窗口 \(identifier.suffix(4))", bundleID: bundle, admission: decision.admission),
            entrance: TargetEntrance(id: identifier, targetID: identifier, bundleID: bundle,
                                     windowID: identifier, kind: "ax_window", containerID: identifier), document: document,
            minimized: boolean(current, kAXMinimizedAttribute))
        return ([observation], identifier)
    }

    private func watch(_ application: AXUIElement, pid: pid_t) {
        for dead in observers.keys.filter({ NSRunningApplication(processIdentifier: $0) == nil }) {
            if let observer = observers.removeValue(forKey: dead) { CFRunLoopRemoveSource(CFRunLoopGetMain(), AXObserverGetRunLoopSource(observer), .commonModes) }
        }
        guard observers[pid] == nil else { return }
        var value: AXObserver?
        guard AXObserverCreate(pid, { _, element, _, context in
            guard let context else { return }
            var pid: pid_t = 0
            guard AXUIElementGetPid(element, &pid) == .success else { return }
            MainActor.assumeIsolated {
                Unmanaged<ContextMonitor>.fromOpaque(context).takeUnretainedValue().onWindowsChanged?(pid)
            }
        }, &value) == .success, let observer = value else { return }
        observers[pid] = observer
        for name in [kAXFocusedWindowChangedNotification, kAXWindowCreatedNotification] {
            // Unsupported notifications retain the periodic observation fallback.
            AXObserverAddNotification(observer, application, name as CFString, Unmanaged.passUnretained(self).toOpaque())
        }
        CFRunLoopAddSource(CFRunLoopGetMain(), AXObserverGetRunLoopSource(observer), .commonModes)
    }

    func raise(_ id: String, rules: [String], expectedDocument: URL? = nil) -> Bool {
        guard let window = windows.first(where: { $0.1 == id }),
              let app = NSRunningApplication(processIdentifier: window.2),
              TargetEligibilityPolicy.window(evidence(window.0), owner: admission(app, rules: rules)).isAllowed else { return false }
        if let expectedDocument {
            guard let raw = string(window.0, kAXDocumentAttribute), let current = URL(string: raw), current.isFileURL,
                  current.standardizedFileURL == expectedDocument.standardizedFileURL else { return false }
        }
        app.activate()
        return AXUIElementPerformAction(window.0, kAXRaiseAction as CFString) == .success
    }
    func canRaise(_ id: String, rules: [String]) -> Bool {
        guard let window = windows.first(where: { $0.1 == id }), let app = NSRunningApplication(processIdentifier: window.2) else { return false }
        return TargetEligibilityPolicy.window(evidence(window.0), owner: admission(app, rules: rules)).isAllowed
    }
    private func evidence(_ window: AXUIElement) -> WindowEvidence {
        var actions: CFArray?
        let result = AXUIElementCopyActionNames(window, &actions)
        return WindowEvidence(role: string(window, kAXRoleAttribute), subrole: string(window, kAXSubroleAttribute),
            modal: boolean(window, kAXModalAttribute),
            canRaise: result == .success && (actions as? [String])?.contains(kAXRaiseAction) == true,
            main: boolean(window, kAXMainAttribute) == true)
    }
    private func boolean(_ item: AXUIElement, _ name: String) -> Bool? {
        var value: CFTypeRef?
        guard AXUIElementCopyAttributeValue(item, name as CFString, &value) == .success else { return nil }
        return (value as? NSNumber)?.boolValue
    }
    private func element(_ parent: AXUIElement, _ name: String) -> AXUIElement? {
        var value: CFTypeRef?
        guard AXUIElementCopyAttributeValue(parent, name as CFString, &value) == .success,
              let value, CFGetTypeID(value) == AXUIElementGetTypeID() else { return nil }
        return unsafeDowncast(value, to: AXUIElement.self)
    }
    private func string(_ item: AXUIElement, _ name: String) -> String? {
        var value: CFTypeRef?
        guard AXUIElementCopyAttributeValue(item, name as CFString, &value) == .success else { return nil }
        return value as? String
    }
    private func observe(_ name: Notification.Name, action: @escaping @MainActor (NSRunningApplication?) -> Void) {
        tokens.append(NSWorkspace.shared.notificationCenter.addObserver(forName: name, object: nil, queue: .main) { note in
            let app = note.userInfo?[NSWorkspace.applicationUserInfoKey] as? NSRunningApplication
            MainActor.assumeIsolated { action(app) }
        })
    }
}
