import AppKit
@preconcurrency import ApplicationServices
import func Carbon.IsSecureEventInputEnabled
import CryptoKit
import IntentCore
import Observation
import ServiceManagement
import FileRecallAX

struct SourceHealth: Identifiable, Equatable {
    let id: String
    var name: String
    var status: String
}
struct NavigationChoice: Identifiable, Equatable {
    let target: Destination
    let entrance: TargetEntrance?
    var derivedActionID: String? = nil
    var rankingSource: String? = nil
    var id: String { entrance?.id ?? target.id }
}
struct PickerSession {
    let id: String
    let choices: [NavigationChoice]
    let contextID: String?
    let inputSequence: Int
    let started: Double
    var modelVersion = "recency-fallback-v1"
}

@MainActor @Observable
final class RecorderModel {
    private(set) var isCollecting = false
    private(set) var accessibilityAuthorized = false
    private(set) var inputAuthorized = false
    private(set) var activeAppName = "等待开始"
    private(set) var activeTargetName = "等待目标变化"
    private(set) var suspension: String?
    private(set) var recent: [BehaviorEvent] = []
    private(set) var sources: [SourceHealth] = []
    private(set) var stats = StoreStats()
    private(set) var sessionEventCount = 0
    @ObservationIgnored private var activityCounter = UserActivityCounter()
    @ObservationIgnored private var heartbeatSummary = HeartbeatSummary()
    @ObservationIgnored private var coverageReasons: [String: String] = [:]
    private(set) var sessionSwitches = 0
    private(set) var targetCount = 0
    private(set) var registeredTargets: [Destination] = []
    private(set) var isExporting = false
    private(set) var loginEnabled = false
    private(set) var isWritingBacklog = false
    private(set) var pendingWriteEvents = 0
    var collectOnLogin: Bool {
        didSet { defaults.set(collectOnLogin, forKey: "collectOnLoginV1") }
    }
    var notice: String?
    var errorMessage: String?
    var exclusionText: String
    var excludedSitesText: String
    let invocation: InvocationController
    let clipboard: ClipboardController
    var onShowPicker: ((PickerSession) -> Void)?
    let databaseURL: URL
    static let bundleID = CapturePolicy.ownBundleID
    static let defaultExclusions = "com.1password.1password\ncom.agilebits.onepassword7\ncom.bitwarden.desktop\ncom.apple.Passwords\ncom.apple.loginwindow"
    private static let writeBatchEventLimit = 128
    private static let writeBatchByteLimit = 1 * 1_024 * 1_024
    private static let writeBatchMaximumAge: TimeInterval = 10
    private static let maximumQueuedEvents = 1_024
    private static let maximumQueuedBytes = 16 * 1_024 * 1_024
    private static let databaseByteLimit: Int64 = 2 * 1_024 * 1_024 * 1_024

    @ObservationIgnored private var store: EventStore?
    @ObservationIgnored private var identity: TargetIdentity?
    @ObservationIgnored private let context = ContextMonitor()
    @ObservationIgnored private var timer: Timer?
    @ObservationIgnored private var sessionID: String?
    @ObservationIgnored private var sequence = 0
    @ObservationIgnored private var buffer: [BehaviorEvent] = []
    @ObservationIgnored private var bufferedBytes = 0
    @ObservationIgnored private var bufferedSince: Double?
    @ObservationIgnored private var queuedEvents = 0
    @ObservationIgnored private var queuedBytes = 0
    private struct PendingWriteBatch {
        let events: [BehaviorEvent]
        let bytes: Int
    }
    @ObservationIgnored private var pendingWriteBatches: [PendingWriteBatch] = []
    @ObservationIgnored private var inFlightWriteBatches = 0
    @ObservationIgnored private var writeFailure = false
    @ObservationIgnored private var resumeAfterBacklogDrain = false
    @ObservationIgnored private var isStoppingCollection = false
    @ObservationIgnored private var catalog: [String: Destination] = [:]
    @ObservationIgnored private var inventories: [String: [TargetEntrance]] = [:]
    @ObservationIgnored private let browserBridge: BrowserExtensionBridge
    @ObservationIgnored private var extensionClients: [String: String] = [:]
    @ObservationIgnored private var extensionSources: [String: String] = [:]
    @ObservationIgnored private var lastSettledContext: String?
    @ObservationIgnored private var stableBrowserLocation: (key: String, since: Double)?
    @ObservationIgnored private var persistedTargetIDs = Set<String>()
    @ObservationIgnored private var sessionTargetIDs = Set<String>()
    @ObservationIgnored private var lastVisited: [String: Double] = [:]
    @ObservationIgnored private var currentApp: AppContext?
    @ObservationIgnored private var currentPID: pid_t?
    @ObservationIgnored private var lastContextKey: String?
    @ObservationIgnored private var lastContextID: String?
    @ObservationIgnored private var currentTargetID: String?
    @ObservationIgnored private var activatedAt = 0.0
    @ObservationIgnored private var systemState = SystemObservationState()
    private var systemUnavailable: String? { systemState.blocksObservation ? systemState.reasons.joined(separator: ",") : nil }
    private struct Inactivity {
        var id: String
        var target: String
        var entrance: String?
        var contextID: String
        var bundle: String?
        var source: String?
        var reasons: [String]
        var endedAt: Double?
    }
    @ObservationIgnored private var inactivity: Inactivity?
    @ObservationIgnored private var excludedScope: (id: String, category: String)?
    @ObservationIgnored private var activeInterruption: (id: String, category: String, signals: [String])?
    @ObservationIgnored private var privateBrowser = false
    @ObservationIgnored private var excludedBrowserPage = false
    @ObservationIgnored private var finderBusy = false
    @ObservationIgnored private var finderDue = 0.0
    @ObservationIgnored private var qspaceBusy = false
    @ObservationIgnored private var qspaceDue = 0.0
    @ObservationIgnored private var sourceTimes: [String: Double] = [:]
    @ObservationIgnored private var coverageTracker = StateCoverageTracker()
    @ObservationIgnored private var coverageStatuses: [String: String] = [:]
    @ObservationIgnored private var coverageChecks: [String: Double] = [:]
    @ObservationIgnored private var lastContextSource: String?
    @ObservationIgnored private var observationResyncPending = false
    @ObservationIgnored private var lastIdle: Bool?
    @ObservationIgnored private var tickCount = 0
    @ObservationIgnored private var lastTick = ProcessInfo.processInfo.systemUptime
    @ObservationIgnored private var lastTickWall = Date().timeIntervalSince1970
    @ObservationIgnored private var exclusionRules: [String] = []
    @ObservationIgnored private var excludedSites: [String] = []
    @ObservationIgnored private var activeDecision: (id: String, until: Double)?
    @ObservationIgnored private var activePicker: PickerSession?
    @ObservationIgnored private var lastPermission: PermissionStatus?
    @ObservationIgnored private var statsBusy = false
    @ObservationIgnored private var statsReader: EventStore?
    @ObservationIgnored private let liveObservation: Bool
    @ObservationIgnored private let defaults: UserDefaults
    @ObservationIgnored private let destinationActions: DestinationActions
    @ObservationIgnored private var eventGate = TargetEventGate()
    @ObservationIgnored private var generation = ObservationGeneration()
    @ObservationIgnored private var foregroundAdmission = AdmissionDecision(.unknown, "not_observed")
    @ObservationIgnored private var admittedProcesses: [pid_t: AppContext] = [:]
    @ObservationIgnored private var pickerShownAt: Double?
    @ObservationIgnored private var recommender: FastRecommendationService?

    private var registrationURL: URL { databaseURL.deletingLastPathComponent().appendingPathComponent("registered-targets.json") }
    private var secureInputEnabled: Bool { liveObservation && IsSecureEventInputEnabled() }
    var statusText: String {
        if isWritingBacklog { return "正在自动分批写入 \(pendingWriteEvents) 条积压" }
        return !isCollecting ? "已暂停" : (suspension ?? "正在采集上下文")
    }
    private var hasPendingWrites: Bool {
        !buffer.isEmpty || !pendingWriteBatches.isEmpty || inFlightWriteBatches > 0
    }
    /// Ephemeral handles for the currently foreground target only; never persisted as inventory.
    var availableEntrances: [TargetEntrance] { inventories.values.flatMap { $0 } }

    init(dataDirectory: URL? = nil, defaults: UserDefaults = .standard, liveObservation: Bool = true,
         destinationActions: DestinationActions = DestinationActions()) {
        databaseURL = dataDirectory?.appendingPathComponent("context-v2.sqlite") ?? EventStore.contextURL
        browserBridge = BrowserExtensionBridge(directory: databaseURL.deletingLastPathComponent())
        self.defaults = defaults; self.liveObservation = liveObservation
        collectOnLogin = defaults.bool(forKey: "collectOnLoginV1")
        self.destinationActions = destinationActions
        invocation = InvocationController(defaults: defaults)
        clipboard = ClipboardController(directory: databaseURL.deletingLastPathComponent(), defaults: defaults, liveObservation: liveObservation)
        exclusionText = defaults.string(forKey: "excludedBundleIDs") ?? Self.defaultExclusions
        excludedSitesText = defaults.string(forKey: "excludedSitesV2") ?? ""
        exclusionRules = exclusionText.split(whereSeparator: { $0.isWhitespace || $0 == "," }).map(String.init)
        excludedSites = excludedSitesText.split(whereSeparator: { $0.isWhitespace || $0 == "," }).map { $0.lowercased() }
        clipboard.excludedBundles = { [weak self] in self?.exclusionRules ?? [] }
        do {
            let store = try EventStore(url: databaseURL); self.store = store
            identity = try TargetIdentity(keyURL: databaseURL.deletingLastPathComponent().appendingPathComponent("identity-v2.key"))
            for target in try store.knownTargets() where target.kind != .window && TargetEligibilityPolicy.validates(target, rules: exclusionRules) {
                catalog[target.id] = target
                persistedTargetIDs.insert(target.id)
            }
            if let data = try? Data(contentsOf: registrationURL) {
                registeredTargets = try JSONDecoder().decode([Destination].self, from: data)
                for target in registeredTargets { if let allowed = admitLocal(target) { catalog[allowed.id] = allowed } }
            }
            // Old source identities are not republished into the live dashboard.
            recommender = FastRecommendationService(dataDirectory: databaseURL.deletingLastPathComponent(), store: store)
        } catch { errorMessage = error.localizedDescription }
        loginEnabled = liveObservation && SMAppService.mainApp.status == .enabled
        context.onActivate = { [weak self] in self?.activate($0) }
        context.onWindowsChanged = { [weak self] pid in
            guard let self, self.isCollecting, self.currentPID == pid,
                  let running = NSRunningApplication(processIdentifier: pid) else { return }
            self.observeFront(running)
        }
        context.onLifecycle = { [weak self] kind, running in
            guard let self, self.isCollecting else { return }
            let pid = running.processIdentifier
            guard kind == .appTerminated, self.currentPID == pid else { return }
            self.setForegroundEntrance(self.lastContextSource ?? "workspace", nil)
            if let source = self.lastContextSource { self.setSourceStatus(source, "not_applicable", reason: "foreground_application_terminated") }
            self.admittedProcesses[pid] = nil
        }
        context.onAvailability = { [weak self] signal, active in self?.systemObservationChanged(signal, active: active) }
        invocation.onInvoke = { [weak self] in self?.invokePicker() }
        browserBridge.onSnapshot = { [weak self] client, bundle, message, started in
            self?.acceptExtension(message, client: client, bundle: bundle, started: started)
        }
        browserBridge.onDisconnect = { [weak self] client, bundle in
            guard let self else { return }
            self.extensionClients[client] = nil
            if self.isCollecting {
                if let source = self.extensionSources.removeValue(forKey: client) {
                    if self.lastContextSource == source { self.setSourceStatus(source, "unavailable", reason: "browser_extension_disconnected") }
                }
                self.health("browser:" + bundle, bundle, "扩展已断开，等待重新连接；不回退轮询")
            }
        }
        browserBridge.onError = { [weak self] message in self?.errorMessage = message }
        if liveObservation { timer = Timer.scheduledTimer(withTimeInterval: 0.5, repeats: true) { [weak self] _ in
            MainActor.assumeIsolated { self?.tick() }
        } }
        if liveObservation { refreshPermissions(); refreshStats() }
    }

    func toggle() {
        guard !isWritingBacklog else { return }
        isCollecting ? pause() : start()
    }
    func startAtLaunch(isLogin: Bool, explicitlyCollect: Bool = false) {
        if LaunchContext.shouldCollect(isLogin: isLogin, loginEnabled: loginEnabled,
                                       consent: collectOnLogin, explicitlyCollect: explicitlyCollect) { start() }
    }
    func prepareBrowserConnection() { if liveObservation { browserBridge.start() } }
    func start() {
        guard !isCollecting, let store, identity != nil else { return }
        let databaseBytes = EventStore.databaseBytes(at: databaseURL)
        stats.bytes = databaseBytes
        guard databaseBytes < Self.databaseByteLimit else { errorMessage = "数据达到 2 GB 限额，请先导出归档。"; return }
        if hasPendingWrites {
            writeFailure = false
            isWritingBacklog = true
            resumeAfterBacklogDrain = true
            errorMessage = nil
            notice = "检测到尚未完成的写入，正在自动分批入库。"
            flush()
            finishAutomaticBacklogRecoveryIfPossible()
            return
        }
        // A previous session may have stopped before its async callbacks reached
        // the main actor. Establish the boundary before accepting new events.
        store.flush()
        errorMessage = nil
        activityCounter = UserActivityCounter(); heartbeatSummary = HeartbeatSummary(); coverageReasons = [:]
        sessionID = UUID().uuidString; sequence = 0; sessionEventCount = 0; sessionSwitches = 0
        sessionTargetIDs = []
        generation.invalidate(); eventGate.reset(); eventGate.rules = exclusionRules
        currentApp = nil; currentPID = nil; lastContextKey = nil; lastContextID = nil
        lastContextSource = nil; observationResyncPending = false
        inventories = [:]; lastSettledContext = nil
        coverageTracker = StateCoverageTracker(); coverageStatuses = [:]; coverageChecks = [:]; coverageReasons = [:]
        extensionSources = [:]; sourceTimes = [:]
        finderDue = 0; qspaceDue = 0; lastIdle = nil
        systemState = SystemObservationState(); inactivity = nil; excludedScope = nil; activeInterruption = nil
        privateBrowser = false; excludedBrowserPage = false; activeDecision = nil
        isCollecting = true; suspension = nil; lastTick = ProcessInfo.processInfo.systemUptime
        lastTickWall = Date().timeIntervalSince1970
        record(.sessionStart, source: "lifecycle", data: ["appVersion": .string("0.4.0"),
            "stateCoverageVersion": .string(StateCoverage.foregroundOnlyVersion),
            "inactivityPolicyVersion": .string("known-inactivity-v1"),
            "timezone": .string(TimeZone.current.identifier), "utcOffsetSeconds": .number(Double(TimeZone.current.secondsFromGMT())),
            "targetEligibilityPolicyVersion": .string(TargetEligibilityPolicy.version),
            "observationPolicy": .string("foreground_destinations_only_no_background_state"), "webPolicy": .string("frontmost_active_tab_full_url_and_local_keyed_page_id"),
            "localPolicy": .string("observed_directories_and_window_documents"),
            "browserObservation": .string("extension_foreground_native_messaging_v2"),
            "browserPollSeconds": .null, "browserStableDelayMs": .number(1200),
            "finderForegroundPollSeconds": .number(0.5), "qspaceForegroundPollSeconds": .number(0.5),
            "backgroundStateCollection": .bool(false), "windowForegroundPollSeconds": .number(0.5)])
        if liveObservation {
            prepareBrowserConnection()
            browserBridge.setForegroundBrowser(nil)
            browserBridge.setCapturing(true)
            context.start(); refreshPermissions(); invocation.setCollecting(true)
            activate(NSWorkspace.shared.frontmostApplication)
        }
        record(.capabilities, source: "lifecycle", data: ["keyboardCapture": .bool(false), "pointerPersistence": .bool(false)])
        if liveObservation { refreshPermissions() }
    }
    func pause(reason: String = "user_pause", waitForCommit: Bool = true) {
        guard isCollecting, !isStoppingCollection else { return }
        isStoppingCollection = true
        defer { isStoppingCollection = false }
        record(.sessionEnd, source: "lifecycle", data: ["reason": .string(reason)])
        generation.invalidate()
        browserBridge.setCapturing(false)
        browserBridge.setForegroundBrowser(nil)
        isCollecting = false; invocation.setCollecting(false); context.stop()
        if liveObservation { Task.detached(priority: .utility) { AutomationScript.closeDirectorySession() } }
        flush(waitForCommit: waitForCommit); sessionID = nil; suspension = nil; inactivity = nil; excludedScope = nil; activeInterruption = nil; activePicker = nil; pickerShownAt = nil; activeDecision = nil
        if liveObservation { refreshPermissions(); refreshStats() }
    }
    func shutdown() {
        resumeAfterBacklogDrain = false
        if isCollecting { pause(reason: "app_shutdown") }
        else { flush(waitForCommit: true) }
        isWritingBacklog = false
        browserBridge.stop(); invocation.stop(); timer?.invalidate(); store?.flush(); recommender?.flush()
        clipboard.shutdown()
    }

    func refreshPermissions() {
        let previous = accessibilityAuthorized
        accessibilityAuthorized = AXIsProcessTrusted(); inputAuthorized = CGPreflightListenEventAccess()
        if previous != accessibilityAuthorized { invocation.refreshMonitoring() }
        if isCollecting {
            health("accessibility", "窗口权限", accessibilityAuthorized ? "已授权" : "需要辅助功能权限")
            if previous != accessibilityAuthorized {
                generation.invalidate()
                for key in Array(coverageStatuses.keys) where key.hasPrefix("ax:") {
                    setSourceStatus(key, accessibilityAuthorized ? "initializing" : "unavailable", reason: "accessibility_changed")
                }
            }
        }
        let status = PermissionStatus(schemaVersion: 2, pid: ProcessInfo.processInfo.processIdentifier,
            bundlePath: Bundle.main.bundlePath, inputMonitoring: inputAuthorized, accessibility: accessibilityAuthorized,
            inputTapConnected: false, collecting: isCollecting)
        if status != lastPermission {
            do { try status.write(); lastPermission = status } catch { errorMessage = error.localizedDescription }
        }
    }
    func requestAccessibilityPermission() {
        let options = [kAXTrustedCheckOptionPrompt.takeUnretainedValue() as String: true] as CFDictionary
        _ = AXIsProcessTrustedWithOptions(options)
        openPrivacy("Privacy_Accessibility")
    }
    func openAutomationSettings() { openPrivacy("Privacy_Automation") }
    func retrySources() {
        browserBridge.resync()
        finderDue = 0; qspaceDue = 0; refreshPermissions(); invocation.refreshMonitoring()
        notice = "正在重新连接运行中的浏览器、Finder 与 QSpace Pro。"
    }
    func saveSettings() {
        exclusionRules = exclusionText.split(whereSeparator: { $0.isWhitespace || $0 == "," }).map(String.init)
        excludedSites = excludedSitesText.split(whereSeparator: { $0.isWhitespace || $0 == "," }).map { $0.lowercased() }
        defaults.set(exclusionText, forKey: "excludedBundleIDs")
        defaults.set(excludedSitesText, forKey: "excludedSitesV2")
        eventGate.rules = exclusionRules; generation.invalidate()
        if isCollecting {
            gap("settings_changed"); inventories = [:]; lastContextKey = nil; finderDue = 0; qspaceDue = 0
            if liveObservation { activate(NSWorkspace.shared.frontmostApplication) }
        }
        notice = "采集设置已保存。"
    }
    func setLoginEnabled(_ enabled: Bool) {
        do {
            if enabled { try SMAppService.mainApp.register() } else { try SMAppService.mainApp.unregister() }
            loginEnabled = SMAppService.mainApp.status == .enabled
            if SMAppService.mainApp.status == .requiresApproval { SMAppService.openSystemSettingsLoginItems() }
        } catch { errorMessage = error.localizedDescription }
    }

    func registerLocalTargets() {
        let panel = NSOpenPanel(); panel.title = "选择要记录和快速打开的目录或文件"
        panel.canChooseFiles = true; panel.canChooseDirectories = true; panel.allowsMultipleSelection = true
        panel.prompt = "登记目标"
        guard panel.runModal() == .OK, let identity else { return }
        for url in panel.urls {
            let directory = (try? url.resourceValues(forKeys: [.isDirectoryKey]).isDirectory) == true
            guard let target = admitLocal(identity.local(url, directory: directory), registering: true) else { continue }
            if !registeredTargets.contains(where: { $0.id == target.id }) { registeredTargets.append(target) }
            upsert(target)
            if !directory {
                guard let parent = admitLocal(identity.local(url.deletingLastPathComponent(), directory: true), registering: true) else { continue }
                if !registeredTargets.contains(where: { $0.id == parent.id }) { registeredTargets.append(parent) }
                upsert(parent)
            }
        }
        do {
            try JSONEncoder().encode(registeredTargets).write(to: registrationURL, options: .atomic)
            try FileManager.default.setAttributes([.posixPermissions: 0o600], ofItemAtPath: registrationURL.path)
            notice = "已添加快捷目标。Finder 与 QSpace Pro 浏览的目录会自动记录，无需手动添加。"
        } catch { errorMessage = error.localizedDescription }
    }

    func invokePicker() {
        guard isCollecting, !invocation.isCalibrating, activePicker == nil,
              systemUnavailable == nil, !secureInputEnabled, !context.secureFieldFocused,
              suspension == nil || currentPID == ProcessInfo.processInfo.processIdentifier else { return }
        // The path model must see this very foreground arrival. The collector
        // normally batches writes; an explicit menu opening closes that batch.
        if recommender?.requiresCommittedCutoff() == true { flush(waitForCommit: true) }
        let entrances = availableEntrances
        var input = currentTargetID.flatMap { catalog[$0] }
        if input?.bundleID == nil { input?.bundleID = currentApp?.bundleID }
        let onlineRanking = input.flatMap { recommender?.onlineMenuRanking(current: $0) }
        let sorted = catalog.values.filter { target in
            MainMenuTargetScope.includes(target.kind) && target.id != currentTargetID && eligibleNow(target) &&
            (input.map { NavigationSemantics.changesDestination(target, current: $0) } ?? true) &&
            (target.kind == .app || target.locator != nil || entrances.contains(where: { $0.targetID == target.id }))
        }.sorted { left, right in
            let a = lastVisited[left.id] ?? 0, b = lastVisited[right.id] ?? 0
            return a == b ? left.name < right.name : a > b
        }
        var chosen: [NavigationChoice] = []; var used = Set<String>()
        if let onlineRanking, let input {
            var qualified: [String: NavigationChoice] = [:]
            for scored in onlineRanking.ranked {
                let candidate = scored.candidate
                guard MainMenuTargetScope.includes(candidate.recordedTarget?.kind ?? candidate.derivedKind) else { continue }
                let choice: NavigationChoice
                if let recorded = candidate.recordedTarget {
                    guard let target = catalog[recorded.id], target.kind == recorded.kind,
                          target.locator == recorded.locator,
                          NavigationSemantics.changesDestination(target, current: input),
                          eligibleNow(target) else { continue }
                    let entrance = entrances.filter { $0.targetID == target.id }.sorted { $0.id < $1.id }.first
                    choice = NavigationChoice(target: target, entrance: entrance,
                                              rankingSource: "path_model")
                } else {
                    guard let derived = derivedChoice(candidate),
                          NavigationSemantics.changesDestination(derived.target, current: input),
                          eligibleChoice(derived) else { continue }
                    choice = derived
                }
                guard used.insert(choice.id).inserted else { continue }
                qualified[candidate.actionID] = choice
            }
            do {
                let preview = try onlineRanking.preview.restrictingPreview(toActionIDs: Set(qualified.keys))
                chosen = preview.selected.compactMap { qualified[$0.actionID] }
            } catch { errorMessage = error.localizedDescription }
        }
        // Recency is used only when the current model cannot answer. A valid
        // calibrated choice keeps its size, including an explicit empty set.
        if onlineRanking == nil {
            for target in sorted {
                if used.contains(target.id) { continue }
                if target.kind == .app, chosen.contains(where: { $0.target.bundleID == target.bundleID }) { continue }
                if let parent = target.parentID, used.contains(parent) { continue }
                if chosen.contains(where: { $0.target.parentID == target.id }) { continue }
                let entrance = entrances.filter { $0.targetID == target.id }.sorted { $0.id < $1.id }.first
                chosen.append(NavigationChoice(target: target, entrance: entrance,
                                               rankingSource: nil))
                used.insert(target.id)
                if chosen.count == DynamicRecommendationConfiguration().maximumItems { break }
            }
        }
        var picker = PickerSession(id: UUID().uuidString, choices: chosen, contextID: lastContextID,
                                   inputSequence: sequence, started: ProcessInfo.processInfo.systemUptime)
        if onlineRanking != nil { picker.modelVersion = OnlinePathMLPScorer.modelVersion }
        activePicker = picker; pickerShownAt = nil
        onShowPicker?(picker)
    }
    func pickerShown(_ picker: PickerSession) {
        guard activePicker?.id == picker.id else { return }
        pickerShownAt = ProcessInfo.processInfo.systemUptime
    }
    /// A cancelled session is only transient UI state: it must not create data or affect later navigation.
    func pickerClosed(_ picker: PickerSession) {
        guard activePicker?.id == picker.id else { return }
        activePicker = nil; pickerShownAt = nil
        if activeDecision?.id == picker.id { activeDecision = nil }
    }
    func leavePickerForTools() { if let activePicker { pickerClosed(activePicker) } }

    /// Capture only the source handle now; body text is read after the user enters FileRecall.
    func captureRecallSource() -> Result<FocusedAXWindow, Error> {
        Result {
            guard !secureInputEnabled, systemUnavailable == nil, !privateBrowser, !excludedBrowserPage,
                  let app = NSWorkspace.shared.frontmostApplication,
                  context.admission(app, rules: exclusionRules).isAllowed else {
                throw StoreError("请在允许的工作窗口唤出菜单，再进入目录查找。")
            }
            return try FocusedAXWindow.capture()
        }
    }
    func prepareDirectoryLookup() { flush(waitForCommit: true) }
    var recallExclusionRules: [String] { exclusionRules }
    func validateRecallSource(_ source: FocusedAXWindow) async throws -> String? {
        guard !secureInputEnabled, systemUnavailable == nil else { throw StoreError("当前窗口暂不可读取，请重新唤出。") }
        guard source.isBrowser else { return nil }
        guard let bundle = source.applicationBundleID, BrowserReader.supported.contains(bundle),
              NSWorkspace.shared.frontmostApplication?.bundleIdentifier == bundle else {
            throw StoreError("目录查找目前需要 Chrome、Edge 或 Brave 的普通活动页面；请保持该页面在前台。")
        }
        let message = try await browserBridge.inspectForeground(bundle: bundle)
        try Task.checkCancellation()
        guard NSWorkspace.shared.frontmostApplication?.bundleIdentifier == bundle,
              let url = message?.snapshot.windows.first?.tabs.first?.url, !url.isEmpty,
              !isSiteExcluded(url) else { throw StoreError("当前不是允许读取的普通活动页面，已停止查找。") }
        return url
    }

    /// A requested parent is justified by a confirmed child, not a new catalog visit.
    func openRecallTarget(_ requested: Destination, decisionID: String, preferredEntrance: TargetEntrance? = nil) async -> Bool {
        guard [.directory, .file].contains(requested.kind), let raw = requested.locator,
              let url = URL(string: raw), url.isFileURL, let identity else { return false }
        let target = catalog[requested.id] ?? requested
        guard target.locator == raw, target.kind == requested.kind,
              identity.local(url, directory: target.kind == .directory).id == target.id,
              let facts = try? url.resourceValues(forKeys: [.isDirectoryKey, .isRegularFileKey]),
              target.kind == .directory ? facts.isDirectory == true : facts.isRegularFile == true else { return false }
        let entrance = preferredEntrance.flatMap { entry in
            entry.targetID == target.id && eventGate.valid(entry) ? entry : nil
        } ?? availableEntrances.first { $0.targetID == target.id && entranceUsable($0) }
        let isDerived = catalog[target.id] == nil
        if !isDerived, !eligibleNow(target) { return false }
        if isDerived {
            // Verify against a real known child locator at execution too. No disk enumeration.
            guard target.kind == .directory, catalog.values.contains(where: { child in
                guard child.kind == .file, TargetEligibilityPolicy.validates(child, rules: exclusionRules),
                      let childURL = child.locator.flatMap(URL.init(string:)) else { return false }
                return childURL.standardizedFileURL.deletingLastPathComponent().path == url.standardizedFileURL.path
            }) else { return false }
        }
        var fields: [String: JSONValue] = ["action": .string("select"), "modelVersion": .string("filerecall-stat-lexical-kev-v1")]
        if isDerived {
            fields["selectedTargetID"] = .string(target.id); fields["selectedLocator"] = .string(raw)
            fields["selectedDerivedActionID"] = .string("filerecall-parent:" + target.id)
        } else { fields["targetID"] = .string(target.id) }
        activeDecision = (decisionID, ProcessInfo.processInfo.systemUptime + 30)
        record(.recommendationFeedback, source: "filerecall", data: fields, decision: decisionID)
        var success = false
        if let entrance, entrance.kind == "finder_window", let window = entrance.windowID {
            let action = destinationActions.activateFinder
            success = await Task.detached { action(window, raw) }.value
        } else if let entrance, entrance.kind == "qspace_pane", let window = entrance.windowID, let pane = entrance.paneID {
            let action = destinationActions.activateQSpace
            success = await Task.detached { action(window, pane, raw) }.value
        } else if let entrance, entrance.kind == "ax_window" {
            success = context.raise(entrance.id, rules: exclusionRules, expectedDocument: target.kind == .file ? url : nil)
        }
        if !success, target.kind == .directory,
           (target.bundleID == QSpaceReader.bundleID || currentApp?.bundleID == QSpaceReader.bundleID) {
            let action = destinationActions.openQSpace
            success = await Task.detached { action(url) }.value
        }
        if !success { success = destinationActions.openURL(url) }
        fields["action"] = .string("execution_result"); fields["success"] = .bool(success)
        record(.recommendationFeedback, source: "filerecall", data: fields, decision: decisionID)
        return success
    }
    func choose(_ requested: NavigationChoice, in picker: PickerSession) {
        guard activePicker?.id == picker.id,
              let displayed = activePicker?.choices.first(where: {
                  $0.target.id == requested.target.id && $0.derivedActionID == requested.derivedActionID
              }),
              requested.entrance.map({ $0.targetID == displayed.target.id }) != false else { return }
        // The displayed destination and model source stay authoritative. A
        // freshly resolved live entrance for that same target may be used.
        let choice = NavigationChoice(target: displayed.target,
                                      entrance: requested.entrance ?? displayed.entrance,
                                      derivedActionID: displayed.derivedActionID,
                                      rankingSource: displayed.rankingSource)
        activePicker = nil
        guard (suspension == nil || currentPID == ProcessInfo.processInfo.processIdentifier), eligibleChoice(choice),
              (reopenChoiceURL(choice) != nil || choice.entrance.map({ entranceUsable($0) }) != false) else {
            pickerShownAt = nil; notice = "目标已不可用，请重新唤出。"; return
        }
        pickerShownAt = nil
        activeDecision = (picker.id, ProcessInfo.processInfo.systemUptime + 30)
        let epoch = sessionID
        let ticket = generation.token
        let actions = destinationActions
        let sourceBundle = currentApp?.bundleID
        var selectionData: [String: JSONValue] = ["action": .string("select"),
            "modelVersion": .string(picker.modelVersion),
            "entranceID": choice.entrance.map { .string($0.id) } ?? .null,
            "selectionMs": .number((ProcessInfo.processInfo.systemUptime - picker.started) * 1_000)]
        selectionData.merge(choiceFeedbackFields(choice), uniquingKeysWith: { _, new in new })
        record(.recommendationFeedback, source: "smartapp_menu", data: selectionData, decision: picker.id)
        Task {
            guard isCollecting, sessionID == epoch, generation.accepts(ticket), eligibleChoice(choice) else { return }
            var success = false
            if let entrance = choice.entrance, entrance.kind == "browser_extension_tab",
               let client = extensionClients.first(where: { entrance.id.contains(":" + $0.key + ":") })?.key,
               let window = entrance.windowID, let tab = entrance.tabID, let url = choice.target.locator {
                success = await browserBridge.activate(clientID: client, window: window, tab: tab, url: url)
            } else if let entrance = choice.entrance, entrance.kind == "browser_tab",
               let window = entrance.windowID, let tab = entrance.tabID {
                let snapshot = await Task.detached(priority: .userInitiated) { actions.readBrowser(entrance.bundleID) }.value.snapshot
                let observed = snapshot?.windows.first(where: { $0.id == window })?.tabs.first(where: { $0.id == tab })
                if generation.accepts(ticket), eligibleChoice(choice),
                   let observed, identity?.page(observed.url)?.id == choice.target.id {
                    success = await Task.detached(priority: .userInitiated) {
                        actions.activateBrowser(entrance.bundleID, window, tab, observed.url)
                    }.value
                } else { success = false }
            } else if let entrance = choice.entrance, entrance.kind == "finder_window", let window = entrance.windowID {
                let expected = choice.target.locator
                success = await Task.detached(priority: .userInitiated) {
                    actions.activateFinder(window, expected)
                }.value
            } else if let entrance = choice.entrance, entrance.kind == "qspace_pane",
                      let window = entrance.windowID, let pane = entrance.paneID {
                let expected = choice.target.locator
                success = await Task.detached(priority: .userInitiated) {
                    actions.activateQSpace(window, pane, expected)
                }.value
            } else if let entrance = choice.entrance, entrance.kind == "ax_window" {
                success = context.raise(entrance.id, rules: exclusionRules,
                    expectedDocument: choice.target.kind == .file ? choice.target.locator.flatMap(URL.init(string:)) : nil)
            } else if choice.target.kind == .app, let bundle = choice.target.bundleID, let url = NSWorkspace.shared.urlForApplication(withBundleIdentifier: bundle) {
                do { _ = try await NSWorkspace.shared.openApplication(at: url, configuration: NSWorkspace.OpenConfiguration()); success = true }
                catch { success = false }
            } else { success = false }
            // A live entrance is only an optimization; its closure/navigation must not
            // prevent returning to the persisted destination.
            if !success, isCollecting, sessionID == epoch, generation.accepts(ticket), eligibleChoice(choice),
               let url = reopenChoiceURL(choice) {
                let qspaceTarget = choice.target.bundleID == QSpaceReader.bundleID ||
                    choice.entrance?.kind == "qspace_pane" ||
                    (choice.derivedActionID != nil && sourceBundle == QSpaceReader.bundleID &&
                     choice.target.kind == .directory)
                if qspaceTarget {
                    success = await Task.detached(priority: .userInitiated) { actions.openQSpace(url) }.value
                }
                if !success { success = actions.openURL(url) }
            }
            guard isCollecting, sessionID == epoch else { return }
            var resultData: [String: JSONValue] = ["action": .string("execution_result"),
                "modelVersion": .string(picker.modelVersion),
                "success": .bool(success)]
            resultData.merge(choiceFeedbackFields(choice), uniquingKeysWith: { _, new in new })
            record(.recommendationFeedback, source: "smartapp_menu", data: resultData, decision: picker.id)
            if !success { notice = "目标可能已关闭或权限不足，请通过原来的方式打开。" }
        }
    }

    func exportDataset() {
        guard let store, !isExporting else { return }
        let panel = NSOpenPanel(); panel.title = "选择数据导出的父文件夹"
        panel.canChooseFiles = false; panel.canChooseDirectories = true; panel.canCreateDirectories = true
        guard panel.runModal() == .OK, let parent = panel.url else { return }
        let directory = parent.appendingPathComponent("SmartApp-context-\(Int(Date().timeIntervalSince1970))-\(UUID().uuidString.prefix(6))")
        flush(waitForCommit: true); isExporting = true
        Task {
            do {
                let report = try await Task.detached(priority: .utility) { store.flush(); return try store.export(to: directory) }.value
                notice = "已导出 \(report.events) 条事件、\(report.samples) 个序列样本，其中 \(report.partialSamples) 个观察尚不完整。"
                NSWorkspace.shared.activateFileViewerSelecting([directory])
            } catch { errorMessage = error.localizedDescription }
            isExporting = false
        }
    }
    func revealData() { NSWorkspace.shared.activateFileViewerSelecting([databaseURL]) }

    private func activate(_ running: NSRunningApplication?) {
        guard isCollecting, !systemState.blocksObservation else { return }
        let changed = currentPID != running?.processIdentifier
        if changed {
            let previousBundle = currentApp?.bundleID
            if (previousBundle == FinderReader.bundleID || previousBundle == QSpaceReader.bundleID) &&
               running?.bundleIdentifier != FinderReader.bundleID && running?.bundleIdentifier != QSpaceReader.bundleID {
                Task.detached(priority: .utility) { AutomationScript.closeDirectorySession() }
            }
            observationResyncPending = false
            stableBrowserLocation = nil
            generation.invalidate(); privateBrowser = false; excludedBrowserPage = false
            activatedAt = ProcessInfo.processInfo.systemUptime; lastContextKey = nil
            inventories = [:]; sourceTimes = [:]
            if running?.bundleIdentifier == FinderReader.bundleID { finderDue = 0 }
            if running?.bundleIdentifier == QSpaceReader.bundleID { qspaceDue = 0 }
        }
        let browserBundle = running?.bundleIdentifier.flatMap { BrowserReader.supported.contains($0) ? $0 : nil }
        browserBridge.setForegroundBrowser(browserBundle)
        let dedicatedSource = browserBundle != nil || running?.bundleIdentifier == FinderReader.bundleID ||
            running?.bundleIdentifier == QSpaceReader.bundleID
        context.retainOnlyForegroundWindows(for: dedicatedSource ? nil : running?.processIdentifier)
        foregroundAdmission = running.map { context.admission($0, rules: exclusionRules) } ?? .init(.unknown, "not_observed")
        let next = foregroundAdmission.isAllowed ? ContextMonitor.context(running) : nil
        currentApp = next; currentPID = running?.processIdentifier
        context.refreshSecureField(foregroundAdmission.isAllowed ? running : nil)
        activeAppName = next?.name ?? "未纳入采集的来源"
        updateSuspension()
        guard suspension == nil, let next else { return }
        guard let running, admitApplication(running) != nil else { return }
        if changed { record(.appActivated, source: "workspace", app: next) }
        observeFront(running)
        if changed { browserBridge.resync(bundle: browserBundle); pollBrowsers(); pollFinder(); pollQSpace() }
    }
    private func observeFront(_ running: NSRunningApplication?) {
        guard suspension == nil, let running, let app = currentApp else { return }
        if BrowserReader.supported.contains(app.bundleID) || app.bundleID == FinderReader.bundleID || app.bundleID == QSpaceReader.bundleID {
            if lastContextKey == nil { emitContext(targetID: "app:" + app.bundleID, entranceID: nil, source: "workspace", reason: "app_activation") }
            return
        }
        let (windows, focused) = context.observeWindows(running, readDocuments: true, rules: exclusionRules)
        health("windows", "应用窗口", context.lastWindowStatus == "ready" ? "前台窗口观察中" : "当前窗口信息不可用")
        updateSuspension()
        guard suspension == nil else { return }
        guard context.lastWindowStatus == "ready" else {
            setSourceStatus("ax:\(running.processIdentifier)", "unavailable", reason: context.lastWindowStatus)
            if lastContextKey == nil { emitContext(targetID: "app:" + app.bundleID, entranceID: nil, source: "workspace", reason: "window_unavailable") }
            return
        }
        var entries: [TargetEntrance] = []
        var targetsByEntrance: [String: Destination] = [:]
        for window in windows {
            _ = upsert(window.destination)
            var entrance = window.entrance
            var observedTarget = window.destination
            if let document = window.document, let target = observedDocument(document, bundle: app.bundleID), upsert(target) {
                entrance.targetID = target.id; observedTarget = target
            }
            entries.append(entrance)
            targetsByEntrance[entrance.id] = observedTarget
        }
        let selected = entries.first { $0.id == focused }
        if let selected, let target = targetsByEntrance[selected.id] { _ = upsert(target) }
        setForegroundEntrance("ax:\(running.processIdentifier)", selected)
        emitContext(targetID: selected?.targetID ?? "app:" + app.bundleID, entranceID: selected?.id,
                    source: selected == nil ? "workspace" : "ax:\(running.processIdentifier)", reason: "foreground_change")
    }
    func emitContext(targetID: String, entranceID: String?, source: String, reason: String,
                             observationStarted: Double? = nil) {
        let key = targetID + ":" + (entranceID ?? "")
        guard (key != lastContextKey || observationResyncPending), suspension == nil, eventGate.known(targetID) else { return }
        var resumedPauseID: String?
        if let pending = inactivity, pending.endedAt != nil {
            if targetID == pending.target && entranceID == pending.entrance {
                resumedPauseID = pending.id
                lastContextID = pending.contextID
                observationResyncPending = true
            } else if catalog[targetID]?.kind == .app, catalog[pending.target]?.kind != .app,
                      targetID == currentApp.map({ "app:" + $0.bundleID }),
                      pending.bundle == currentApp?.bundleID {
                // Wait for the concrete child; a coarse wake snapshot proves no continuity.
                return
            } else {
                gap("inactive_resume_target_changed")
                observationResyncPending = true
            }
        }
        let actualReason = observationResyncPending || (lastContextKey?.hasPrefix("app:") == true && entranceID != nil) ? "resolution_update" : reason
        let targetName = catalog[targetID]?.name ?? activeAppName
        var data: [String: JSONValue] = ["targetID": .string(targetID), "entranceID": entranceID.map(JSONValue.string) ?? .null,
            "reason": .string(actualReason), "targetName": .string(targetName), "association": .string("observed_not_causal")]
        if let entranceID, let entrance = availableEntrances.first(where: { $0.id == entranceID && $0.targetID == targetID }),
           let encoded = try? JSONValue.encoded(entrance) {
            data["foregroundEntrance"] = encoded
        }
        if actualReason == "resolution_update", let lastContextID {
            data["resolvesContextEventID"] = .string(lastContextID)
            data["resolutionEvidence"] = .string(observationResyncPending ? "source_recovered_behavior_unobserved" : "container_then_observed_child_not_confirmed_intent")
        }
        if let resumedPauseID {
            data["continuityPauseID"] = .string(resumedPauseID)
            data["resolutionEvidence"] = .string("same_target_after_known_inactivity")
        }
        data["inputCoverage"] = try? .encoded(coverageTracker.snapshot(at: ProcessInfo.processInfo.systemUptime * 1000, sequence: sequence))
        if let observationStarted { data["observationStartedMonotonicMs"] = .number(observationStarted * 1_000) }
        guard let id = record(.contextChanged, source: source, app: currentApp, data: data) else { return }
        lastVisited[targetID] = ProcessInfo.processInfo.systemUptime
        lastContextID = id; lastContextKey = key; currentTargetID = targetID; activeTargetName = targetName
        lastContextSource = source
        observationResyncPending = false
        if resumedPauseID != nil { inactivity = nil }
    }
    private func pollBrowsers() {
        guard isCollecting, suspension == nil, let app = currentApp, BrowserReader.supported.contains(app.bundleID) else { return }
        health("browser:" + app.bundleID, app.name,
               browserBridge.hasClient(app.bundleID) ? "浏览器扩展已连接（仅前台事件）" : "等待安装或连接浏览器扩展")
    }
    private func acceptExtension(_ message: BrowserBridgeMessage, client: String, bundle: String, started: Double) {
        guard isCollecting, systemUnavailable == nil, !secureInputEnabled, !context.secureFieldFocused else { return }
        activate(NSWorkspace.shared.frontmostApplication)
        guard browserBridge.contains(client) else { return }
        guard let identity, let epoch = sessionID,
              let running = NSRunningApplication.runningApplications(withBundleIdentifier: bundle).first,
              context.admission(running, rules: exclusionRules).isAllowed else { return }
        let stillFront = currentPID == running.processIdentifier && started >= activatedAt
        let hasFocusedWindow = message.snapshot.windows.contains(where: \.front)
        guard stillFront, hasFocusedWindow else { return }
        let resolved = BrowserExtensionObservation(message: message, identity: identity, bundle: bundle,
            pid: running.processIdentifier, epoch: epoch, client: client, excluded: isSiteExcluded)
        excludedBrowserPage = resolved.observation.excludedFront
        updateSuspension()
        guard suspension == nil, admitApplication(running) != nil else { return }
        extensionClients[client] = bundle
        let source = "browser-extension:" + bundle + ":" + (message.profileID ?? client)
        extensionSources[client] = source
        persistExtension(resolved, message: message, source: source, foreground: true,
                         fallbackID: "app:" + bundle, started: started)
    }
    func persistExtension(_ resolved: BrowserExtensionObservation, message: BrowserBridgeMessage, source: String,
                          foreground: Bool, fallbackID: String, started: Double) {
        guard isCollecting, suspension == nil, foreground, !resolved.observation.excludedFront else { return }
        let observation = resolved.observation.foregroundOnly()
        for target in observation.targets { _ = upsert(target) }
        setForegroundEntrance(source, observation.foreground)
        let stableKey = source + ":" + (resolved.observation.foreground?.targetID ?? fallbackID) + ":" + (resolved.observation.foreground?.id ?? "")
        if stableBrowserLocation?.key != stableKey { stableBrowserLocation = (stableKey, started) }
        emitContext(targetID: observation.foreground?.targetID ?? fallbackID,
            entranceID: observation.foreground?.id, source: source, reason: "browser_" + message.reason,
            observationStarted: started)
        if message.reason == "settled", let entrance = observation.foreground,
           let stable = stableBrowserLocation, stable.key == stableKey, started - stable.since >= 1.0,
           let contextID = lastContextID, lastSettledContext != contextID {
            lastSettledContext = contextID
            var data: [String: JSONValue] = [
                "targetID": .string(entrance.targetID), "entranceID": .string(entrance.id),
                "contextEventID": .string(contextID), "evidence": .string("foreground_location_unchanged_1200ms"),
                "confirmedIntent": .bool(false)
            ]
            if let encoded = try? JSONValue.encoded(entrance) { data["foregroundEntrance"] = encoded }
            record(.destinationSettled, source: source, app: currentApp, data: data)
        }
    }

    private func pollFinder() {
        guard isCollecting, suspension == nil, !finderBusy, systemUnavailable == nil,
              !secureInputEnabled, !context.secureFieldFocused,
              let app = NSRunningApplication.runningApplications(withBundleIdentifier: FinderReader.bundleID).first,
              currentPID == app.processIdentifier, NSWorkspace.shared.frontmostApplication?.processIdentifier == app.processIdentifier,
              context.admission(app, rules: exclusionRules).isAllowed else { return }
        let now = ProcessInfo.processInfo.systemUptime
        guard now >= finderDue else { return }
        finderBusy = true; finderDue = now + 0.5
        let epoch = sessionID, pid = app.processIdentifier, ticket = generation.token
        Task {
            let result = await Task.detached(priority: .utility) { FinderReader.read(expectedPID: pid) }.value
            finderBusy = false
            if isCollecting { activate(NSWorkspace.shared.frontmostApplication) }
            guard isCollecting, sessionID == epoch, generation.accepts(ticket), suspension == nil,
                  currentPID == pid, NSWorkspace.shared.frontmostApplication?.processIdentifier == pid,
                  NSRunningApplication(processIdentifier: pid)?.bundleIdentifier == FinderReader.bundleID,
                  context.admission(app, rules: exclusionRules).isAllowed else { return }
            if result.status == "stale" {
                finderDue = 0
                setSourceStatus("finder:" + FinderReader.bundleID, "stale", reason: "front_changed_during_read")
                return
            }
            guard let snapshot = result.snapshot, let identity, let epoch else {
                finderDue = ProcessInfo.processInfo.systemUptime + 30
                health("finder:" + FinderReader.bundleID, "Finder 目录",
                       result.status == "automation_denied" ? "需要 Finder 自动化权限" : "连接失败，30 秒后重试")
                setSourceStatus("finder:" + FinderReader.bundleID, "unavailable", reason: result.status)
                return
            }
            guard admitApplication(app) != nil else { return }
            health("finder:" + FinderReader.bundleID, "Finder 目录", "目录导航已连接")
            acceptObservation(snapshot.resolve(identity: identity, pid: pid, epoch: epoch),
                              source: "finder:" + FinderReader.bundleID, foreground: true,
                              fallbackID: "app:" + FinderReader.bundleID, reason: "directory_or_window_change", started: now)
        }
    }

    private func pollQSpace() {
        guard isCollecting, suspension == nil, !qspaceBusy, systemUnavailable == nil,
              !secureInputEnabled, !context.secureFieldFocused,
              let app = NSRunningApplication.runningApplications(withBundleIdentifier: QSpaceReader.bundleID).first,
              currentPID == app.processIdentifier, NSWorkspace.shared.frontmostApplication?.processIdentifier == app.processIdentifier,
              context.admission(app, rules: exclusionRules).isAllowed else { return }
        let now = ProcessInfo.processInfo.systemUptime
        guard now >= qspaceDue else { return }
        qspaceBusy = true; qspaceDue = now + 0.5
        let epoch = sessionID, pid = app.processIdentifier, ticket = generation.token
        Task {
            let result = await Task.detached(priority: .utility) { QSpaceReader.read(expectedPID: pid) }.value
            qspaceBusy = false
            if isCollecting { activate(NSWorkspace.shared.frontmostApplication) }
            guard isCollecting, sessionID == epoch, generation.accepts(ticket), suspension == nil,
                  currentPID == pid, NSWorkspace.shared.frontmostApplication?.processIdentifier == pid,
                  NSRunningApplication(processIdentifier: pid)?.bundleIdentifier == QSpaceReader.bundleID,
                  context.admission(app, rules: exclusionRules).isAllowed else { return }
            if result.status == "stale" {
                qspaceDue = 0
                setSourceStatus("qspace:" + QSpaceReader.bundleID, "stale", reason: "front_changed_during_read")
                return
            }
            guard let snapshot = result.snapshot, let identity, let epoch else {
                qspaceDue = ProcessInfo.processInfo.systemUptime + 30
                health("qspace:" + QSpaceReader.bundleID, "QSpace Pro 目录",
                       result.status == "automation_denied" ? "需要 QSpace Pro 自动化权限" : "连接失败，30 秒后重试")
                setSourceStatus("qspace:" + QSpaceReader.bundleID, "unavailable", reason: result.status)
                return
            }
            guard admitApplication(app) != nil else { return }
            health("qspace:" + QSpaceReader.bundleID, "QSpace Pro 目录", "目录窗格导航已连接")
            acceptObservation(snapshot.resolve(identity: identity, pid: pid, epoch: epoch),
                              source: "qspace:" + QSpaceReader.bundleID, foreground: true,
                              fallbackID: "app:" + QSpaceReader.bundleID, reason: "directory_or_pane_change", started: now)
        }
    }

    func acceptObservation(_ observation: DestinationObservation, source: String, foreground: Bool,
                           fallbackID: String, reason: String, started: Double) {
        guard isCollecting, suspension == nil, foreground, !observation.excludedFront else { return }
        let current = observation.foregroundOnly()
        for target in current.targets { _ = upsert(target) }
        setForegroundEntrance(source, current.foreground)
        emitContext(targetID: current.foreground?.targetID ?? fallbackID,
                    entranceID: current.foreground?.id, source: source, reason: reason, observationStarted: started)
    }

    func systemObservationChanged(_ signal: SystemObservationSignal, active: Bool) {
        let wasBlocked = systemState.blocksObservation
        guard systemState.update(signal, active: active), isCollecting else { return }
        record(.sourceStatus, source: "system_availability", data: ["systemSignal": .string(signal.rawValue), "active": .bool(active)])
        if signal == .displaySleep { return }
        if systemState.blocksObservation {
            Task.detached(priority: .utility) { AutomationScript.closeDirectorySession() }
        }
        if !wasBlocked, systemState.permitsContinuity, suspension == nil,
           let target = currentTargetID, let contextID = lastContextID {
            let entrance = availableEntrances.first { $0.targetID == target && target + ":" + $0.id == lastContextKey }?.id
            inactivity = Inactivity(id: UUID().uuidString, target: target, entrance: entrance,
                                    contextID: contextID, bundle: currentApp?.bundleID ?? catalog[target]?.bundleID,
                                    source: lastContextSource,
                                    reasons: systemState.reasons)
            recordInactivity("begin")
            clearObservationState()
            flush(waitForCommit: true)
        } else if wasBlocked, !systemState.blocksObservation, inactivity != nil {
            recordInactivity("end")
            inactivity?.endedAt = ProcessInfo.processInfo.systemUptime
            lastTick = ProcessInfo.processInfo.systemUptime
            lastTickWall = Date().timeIntervalSince1970
            lastIdle = nil
            suspension = nil
            clearObservationState()
            flush(waitForCommit: true)
        } else if systemState.blocksObservation, !systemState.permitsContinuity, inactivity != nil {
            gap("user_session_inactive")
        }
        if !systemState.blocksObservation, liveObservation {
            activate(NSWorkspace.shared.frontmostApplication)
        } else if systemState.blocksObservation { updateSuspension() }
    }
    private func recordInactivity(_ phase: String) {
        guard let inactivity else { return }
        record(.captureGap, source: "lifecycle", data: [
            "reason": .string("known_inactivity_" + phase), "pauseVersion": .string("known-inactivity-v1"),
            "pauseID": .string(inactivity.id), "pausePhase": .string(phase),
            "pauseReasons": .array(inactivity.reasons.map(JSONValue.string))])
    }
    private func updateSuspension() {
        guard isCollecting else { return }
        if inactivity != nil, systemState.permitsContinuity {
            suspension = systemState.active.contains(.systemSleep) ? "系统休眠期间暂停计时" : "锁屏期间暂停计时"
            return
        }
        let reason: String?
        if systemUnavailable != nil { reason = "系统状态导致观察暂停" }
        else if secureInputEnabled || context.secureFieldFocused { reason = "安全输入期间暂停" }
        else if privateBrowser { reason = "隐私窗口期间暂停" }
        else if excludedBrowserPage { reason = "当前页面未纳入采集" }
        else if currentApp == nil { reason = foregroundAdmission.disposition == .excluded ? "当前应用已排除" : "当前前台对象尚未确认" }
        else if currentApp.map({ isExcluded($0.bundleID) }) == true { reason = "当前应用已排除" }
        else { reason = nil }
        let category: String?
        if systemUnavailable != nil { category = nil }
        else if secureInputEnabled || context.secureFieldFocused { category = "secure_input" }
        else if privateBrowser { category = "private_browser" }
        else if excludedBrowserPage { category = "excluded_page" }
        else if foregroundAdmission.disposition == .excluded || currentApp.map({ isExcluded($0.bundleID) }) == true { category = "excluded_app" }
        else { category = nil }
        let interruptionCategory: String?
        if reason == nil || category != nil { interruptionCategory = nil }
        else if systemUnavailable != nil { interruptionCategory = "system_unavailable" }
        else { interruptionCategory = "foreground_unconfirmed" }
        let interruptionSignals = interruptionCategory == "system_unavailable" ? systemState.reasons : []
        if reason != suspension || category != excludedScope?.category ||
            interruptionCategory != activeInterruption?.category || interruptionSignals != (activeInterruption?.signals ?? []) {
            if reason != nil, liveObservation {
                Task.detached(priority: .utility) { AutomationScript.closeDirectorySession() }
            }
            generation.invalidate()
            if let previous = excludedScope, category != previous.category {
                recordScope("end", id: previous.id, category: previous.category)
                excludedScope = nil
            }
            if let previous = activeInterruption,
               interruptionCategory != previous.category || interruptionSignals != previous.signals {
                recordInterruption("end", id: previous.id, category: previous.category, signals: previous.signals)
                activeInterruption = nil
            }
            if let category {
                if excludedScope == nil {
                    let id = UUID().uuidString
                    recordScope("begin", id: id, category: category)
                    excludedScope = (id, category)
                }
            } else if let interruptionCategory, activeInterruption == nil {
                let id = UUID().uuidString
                recordInterruption("begin", id: id, category: interruptionCategory, signals: interruptionSignals)
                activeInterruption = (id, interruptionCategory, interruptionSignals)
            }
            clearObservationState()
            suspension = reason; lastContextKey = nil
            currentTargetID = nil; lastContextID = nil; activeTargetName = "等待合格目标"
            inventories = [:]; sourceTimes = [:]
            activePicker = nil; pickerShownAt = nil; activeDecision = nil
            if reason == nil {
                finderDue = 0; qspaceDue = 0
            }
        }
    }
    private func recordScope(_ phase: String, id: String, category: String) {
        record(.observationScope, source: "lifecycle", data: [
            "scopeVersion": .string("excluded-foreground-v1"), "scopeID": .string(id),
            "phase": .string(phase), "category": .string(category)])
    }
    private func recordInterruption(_ phase: String, id: String, category: String, signals: [String]) {
        var data: [String: JSONValue] = [
            "reason": .string(phase == "begin" ? "observation_suspended" : "observation_resumed"),
            "interruptionVersion": .string("interruption-cause-v1"),
            "interruptionID": .string(id), "interruptionPhase": .string(phase),
            "causeCategory": .string(category)]
        if !signals.isEmpty { data["causeSignals"] = .array(signals.map(JSONValue.string)) }
        record(.captureGap, source: "lifecycle", data: data)
    }
    private func isExcluded(_ bundle: String) -> Bool { CapturePolicy.isExcluded(bundle, rules: exclusionRules) }
    private func isSiteExcluded(_ origin: String?) -> Bool {
        guard let origin, let host = URL(string: origin)?.host else { return false }
        return excludedSites.contains { host == $0 || host.hasSuffix("." + $0) }
    }
    private func admitApplication(_ running: NSRunningApplication) -> AppContext? {
        guard suspension == nil else { return nil }
        let decision = context.admission(running, rules: exclusionRules)
        guard decision.isAllowed, let app = ContextMonitor.context(running) else { return nil }
        let target = Destination(id: "app:" + app.bundleID, kind: .app, name: app.name,
                                 bundleID: app.bundleID, admission: decision.admission)
        guard upsert(target) else { return nil }
        admittedProcesses[running.processIdentifier] = app
        return app
    }
    private func admitLocal(_ target: Destination, registering: Bool = false) -> Destination? {
        guard [.file, .directory].contains(target.kind), let url = URL(string: target.locator ?? ""), url.isFileURL,
              let values = try? url.resourceValues(forKeys: [.isDirectoryKey, .isRegularFileKey]),
              target.kind == .directory ? values.isDirectory == true : values.isRegularFile == true else { return nil }
        if TargetEligibilityPolicy.isObservedDirectory(target) || TargetEligibilityPolicy.isObservedDocument(target),
           TargetEligibilityPolicy.validates(target, rules: exclusionRules) { return target }
        let decision = TargetEligibilityPolicy.local(registered: registering || registeredTargets.contains(where: {
            $0.id == target.id && $0.locator == target.locator && $0.kind == target.kind
        }), validFileURL: true)
        guard decision.isAllowed else { return nil }
        var result = target; result.admission = decision.admission; return result
    }
    func eligibleNow(_ target: Destination) -> Bool {
        guard TargetEligibilityPolicy.validates(target, rules: exclusionRules) else { return false }
        switch target.kind {
        case .app: return target.bundleID.map { context.installedAdmission($0, rules: exclusionRules).isAllowed } == true
        case .file, .directory: return admitLocal(target) != nil
        case .page:
            return !isSiteExcluded(target.origin) && (reopenURL(target) != nil ||
                availableEntrances.contains { $0.targetID == target.id && entranceUsable($0) })
        case .window:
            return availableEntrances.contains { $0.targetID == target.id && entranceUsable($0) }
        }
    }
    private func derivedChoice(_ candidate: OnlinePathCandidate) -> NavigationChoice? {
        guard candidate.recordedTarget == nil,
              candidate.source == "derived_from_known_locator",
              let kind = candidate.derivedKind,
              let locator = candidate.executableLocator,
              let identity, let url = URL(string: locator) else { return nil }
        let target: Destination
        switch kind {
        case .page:
            guard let page = identity.page(locator) else { return nil }
            target = page
        case .directory:
            guard url.isFileURL else { return nil }
            target = identity.local(url, directory: true)
        default: return nil
        }
        let choice = NavigationChoice(target: target, entrance: nil,
                                      derivedActionID: candidate.actionID,
                                      rankingSource: "path_model")
        return verifiedDerivedURL(choice) == nil ? nil : choice
    }
    private func verifiedDerivedURL(_ choice: NavigationChoice) -> URL? {
        guard let actionID = choice.derivedActionID,
              let locator = choice.target.locator,
              let url = URL(string: locator),
              [.page, .directory].contains(choice.target.kind) else { return nil }
        let digest = SHA256.hash(data: Data((choice.target.kind.rawValue + "\0" + locator).utf8))
            .map { String(format: "%02x", $0) }.joined()
        guard actionID == "derived-parent:\(choice.target.kind.rawValue):\(digest.prefix(24))" else { return nil }
        switch choice.target.kind {
        case .page:
            guard identity?.page(locator)?.id == choice.target.id,
                  !isSiteExcluded(choice.target.origin) else { return nil }
        case .directory:
            guard url.isFileURL,
                  identity?.local(url, directory: true).id == choice.target.id,
                  (try? url.resourceValues(forKeys: [.isDirectoryKey]).isDirectory) == true else { return nil }
        default: return nil
        }
        return url
    }
    private func eligibleChoice(_ choice: NavigationChoice) -> Bool {
        choice.derivedActionID == nil ? eligibleNow(choice.target) : verifiedDerivedURL(choice) != nil
    }
    private func reopenChoiceURL(_ choice: NavigationChoice) -> URL? {
        choice.derivedActionID == nil ? reopenURL(choice.target) : verifiedDerivedURL(choice)
    }
    private func choiceFeedbackFields(_ choice: NavigationChoice) -> [String: JSONValue] {
        var fields: [String: JSONValue] = [:]
        if let source = choice.rankingSource { fields["choiceRankingSource"] = .string(source) }
        guard let actionID = choice.derivedActionID else {
            fields["targetID"] = .string(choice.target.id)
            return fields
        }
        // A selected derived parent is feedback about an actual choice, not a
        // newly observed visit or a saved candidate list.
        fields["selectedTargetID"] = .string(choice.target.id)
        fields["selectedDerivedActionID"] = .string(actionID)
        fields["selectedLocator"] = choice.target.locator.map(JSONValue.string) ?? .null
        return fields
    }
    func reopenURL(_ target: Destination) -> URL? {
        guard let locator = target.locator, let url = URL(string: locator) else { return nil }
        switch target.kind {
        case .page:
            return identity?.page(locator)?.id == target.id && !isSiteExcluded(target.origin) ? url : nil
        case .file, .directory: return admitLocal(target) != nil ? url : nil
        default: return nil
        }
    }
    private func entranceUsable(_ entrance: TargetEntrance) -> Bool {
        guard eventGate.valid(entrance), context.installedAdmission(entrance.bundleID, rules: exclusionRules).isAllowed else { return false }
        switch entrance.kind {
        case "browser_extension_tab":
            return extensionClients.contains { entrance.id.contains(":" + $0.key + ":") && browserBridge.contains($0.key) } && availableEntrances.contains(entrance)
        case "ax_window": return context.canRaise(entrance.id, rules: exclusionRules)
        case "browser_tab":
            return BrowserReader.supported.contains(entrance.bundleID) &&
                ProcessInfo.processInfo.systemUptime - (sourceTimes["browser:" + entrance.bundleID] ?? 0) < 5 &&
                availableEntrances.contains(entrance)
        case "finder_window":
            return entrance.bundleID == FinderReader.bundleID &&
                ProcessInfo.processInfo.systemUptime - (sourceTimes["finder:" + FinderReader.bundleID] ?? 0) < 10 &&
                availableEntrances.contains(entrance)
        case "qspace_pane":
            return entrance.bundleID == QSpaceReader.bundleID && entrance.windowID != nil && entrance.paneID != nil &&
                ProcessInfo.processInfo.systemUptime - (sourceTimes["qspace:" + QSpaceReader.bundleID] ?? 0) < 10 &&
                availableEntrances.contains(entrance)
        case "running_app": return !NSRunningApplication.runningApplications(withBundleIdentifier: entrance.bundleID).isEmpty
        default: return false
        }
    }
    @discardableResult func upsert(_ target: Destination) -> Bool {
        guard suspension == nil, TargetEligibilityPolicy.validates(target, rules: exclusionRules),
              target.kind != .page || !isSiteExcluded(target.origin) else { return false }
        let changed = catalog[target.id] != target
        // The final event gate starts each session empty. Repeat only a target
        // that has actually been observed in this session, before its arrival.
        let needsPersistence = changed || !persistedTargetIDs.contains(target.id) ||
            (isCollecting && !sessionTargetIDs.contains(target.id))
        if changed { catalog[target.id] = target }
        if needsPersistence {
            if isCollecting, let encoded = try? JSONValue.encoded(target) {
                guard record(.targetUpdated, source: "target_catalog", data: ["target": encoded]) != nil else { return false }
                persistedTargetIDs.insert(target.id)
                sessionTargetIDs.insert(target.id)
            }
        }
        targetCount = catalog.count
        return true
    }
    /// Retain a live activation handle only while its target is foreground. Unlike
    /// historic inventory snapshots, this never persists an available-target list.
    func setForegroundEntrance(_ source: String, _ entrance: TargetEntrance?) {
        guard suspension == nil else { return }
        inventories = [:]; sourceTimes = [:]
        guard let entrance else { return }
        guard eventGate.valid(entrance) else { return }
        inventories[source] = [entrance]
        sourceTimes[source] = ProcessInfo.processInfo.systemUptime
        setSourceStatus(source, "ready")
    }

    func setSourceStatus(_ source: String, _ status: String, reason: String? = nil) {
        guard isCollecting, suspension == nil else { return }
        if let pending = inactivity, pending.endedAt != nil, pending.source == source,
           ["unavailable", "stale", "not_applicable"].contains(status) {
            gap("inactive_resume_source_unavailable")
            observationResyncPending = true
        }
        let now = ProcessInfo.processInfo.systemUptime
        guard coverageStatuses[source] != status || coverageReasons[source] != reason ||
            (status == "ready" && now - (coverageChecks[source] ?? 0) >= 10) else { return }
        var data: [String: JSONValue] = ["coverageVersion": .string(StateCoverage.foregroundOnlyVersion),
            "observationStatus": .string(status)]
        if let reason { data["reason"] = .string(reason) }
        guard record(.sourceStatus, source: source, data: data) != nil else { return }
        coverageReasons[source] = reason
        coverageStatuses[source] = status; coverageChecks[source] = now
        if status != "ready" {
            inventories[source] = nil; sourceTimes[source] = nil
            if lastContextSource == source { observationResyncPending = true; currentTargetID = nil }
        }
    }
    func observedDocument(_ url: URL, bundle: String) -> Destination? {
        guard url.isFileURL, let identity, eventGate.known("app:" + bundle) else { return nil }
        if let registered = registeredTargets.first(where: { URL(string: $0.locator ?? "")?.standardizedFileURL == url.standardizedFileURL }),
           let allowed = admitLocal(registered) { return allowed }
        guard let facts = try? url.resourceValues(forKeys: [.isRegularFileKey]), facts.isRegularFile == true else { return nil }
        var target = identity.local(url, directory: false); target.bundleID = bundle
        target.admission = TargetEligibilityPolicy.document(validFileURL: true, isRegularFile: true,
            owner: .init(.allowed, "standalone_application")).admission
        return target
    }
    private func health(_ id: String, _ name: String, _ status: String) {
        guard suspension == nil else { return }
        if let index = sources.firstIndex(where: { $0.id == id }) {
            guard sources[index].status != status else { return }
            sources[index].status = status
        } else { sources.append(SourceHealth(id: id, name: name, status: status)) }
        record(.sourceStatus, source: id, data: ["status": .string(status)])
    }
    @discardableResult func record(_ kind: EventKind, source: String, app: AppContext? = nil,
        data: [String: JSONValue] = [:], decision: String? = nil) -> String? {
        guard let sessionID, !kind.isInput else { return nil }
        // Transient UI, clipboard, and legacy inventory records are outside the foreground-only dataset.
        guard ![.menuInvoked, .recommendationShown, .clipboardChanged, .clipboardMetadata, .inventoryChanged].contains(kind) else { return nil }
        let boundary = [EventKind.captureGap, .observationScope, .sessionStart, .sessionEnd, .heartbeat, .capabilities].contains(kind)
            || (kind == .sourceStatus && source == "system_availability" && app == nil &&
                Set(data.keys) == ["systemSignal", "active"] &&
                data["systemSignal"]?.string.flatMap(SystemObservationSignal.init(rawValue:)) != nil)
        let ownMenu = currentPID == ProcessInfo.processInfo.processIdentifier &&
            kind == .recommendationFeedback
        guard suspension == nil || boundary || ownMenu else { return nil }
        var event = BehaviorEvent(sessionID: sessionID, kind: kind, source: source, app: app, data: data)
        if kind == .captureGap || kind == .sessionEnd,
           let summary = heartbeatSummary.take() {
            event.data["collectorHeartbeatSummary"] = .object(summary)
        }
        guard eventGate.accept(event) else { return nil }
        sequence += 1
        event.schemaVersion = 2; event.sequence = sequence
        let related = activeDecision.flatMap { ProcessInfo.processInfo.systemUptime <= $0.until ? $0.id : nil }
        event.decisionID = decision ?? ((kind == .contextChanged || kind == .appActivated) ? related : nil)
        coverageTracker.consume(event)
        let isUserAction = activityCounter.consume(event)
        if buffer.isEmpty { bufferedSince = ProcessInfo.processInfo.systemUptime }
        buffer.append(event); bufferedBytes += estimatedWriteBytes(event); sessionEventCount += 1
        updatePendingWriteCount()
        if isUserAction {
            sessionSwitches += 1
            recent.insert(event, at: 0); if recent.count > 80 { recent.removeLast(recent.count - 80) }
        }
        let boundaryFlush = kind == .sessionStart || kind == .captureGap || kind == .observationScope
        if buffer.count >= Self.writeBatchEventLimit || bufferedBytes >= Self.writeBatchByteLimit || boundaryFlush {
            flush(waitForCommit: boundaryFlush)
        }
        return event.id
    }
    func gap(_ reason: String) {
        inactivity = nil
        let categories: [String: String] = [
            "settings_changed": "settings_change", "user_session_inactive": "user_session_inactive",
            "inactive_resume_target_changed": "resume_target_changed",
            "inactive_resume_source_unavailable": "resume_source_unavailable",
            "inactive_resume_unverified": "resume_unverified",
            "scheduler_or_clock_gap": "scheduler_or_clock_gap"]
        var data: [String: JSONValue] = ["reason": .string(reason)]
        if let category = categories[reason] { data["causeCategory"] = .string(category) }
        record(.captureGap, source: "lifecycle", data: data)
        clearObservationState()
    }
    private func clearObservationState() {
        observationResyncPending = false
        stableBrowserLocation = nil; lastSettledContext = nil
        generation.invalidate()
        lastContextKey = nil; lastContextID = nil; currentTargetID = nil
        inventories = [:]; sourceTimes = [:]
        coverageStatuses = [:]; coverageChecks = [:]; coverageReasons = [:]
        finderDue = 0; qspaceDue = 0
        browserBridge.resync()
        activePicker = nil; pickerShownAt = nil; activeDecision = nil
    }
    private func tick() {
        tickCount += 1
        let now = ProcessInfo.processInfo.systemUptime
        let wall = Date().timeIntervalSince1970
        if isCollecting {
            if let endedAt = inactivity?.endedAt, now - endedAt > 10 {
                gap("inactive_resume_unverified"); observationResyncPending = true
            }
            if (now - lastTick > 4 || wall - lastTickWall > 4 || wall < lastTickWall),
               systemUnavailable == nil, inactivity == nil {
                gap("scheduler_or_clock_gap"); lastContextKey = nil
            }
            let front = NSWorkspace.shared.frontmostApplication
            activate(front)
            pollBrowsers(); pollFinder(); pollQSpace()
            if tickCount.isMultiple(of: 4) {
                refreshPermissions()
                let seconds = CGEventSource.secondsSinceLastEventType(.combinedSessionState, eventType: CGEventType(rawValue: UInt32.max)!)
                let idle = seconds >= 60
                if idle != lastIdle { lastIdle = idle; record(.idleChanged, source: "system_idle", data: ["idle": .bool(idle), "thresholdSeconds": .number(60)]) }
            }
            observeHeartbeat(at: now * 1_000)
            if tickCount.isMultiple(of: 40), suspension == nil { browserBridge.resync() }
            if let bufferedSince, now - bufferedSince >= Self.writeBatchMaximumAge { flush() }
        }
        if tickCount.isMultiple(of: 30) { refreshDatabaseSize() }
        if tickCount.isMultiple(of: 120) { refreshStats() }
        lastTick = now
        lastTickWall = wall
    }
    func observeHeartbeat(at time: Double) {
        if heartbeatSummary.observe(at: time) { flushHeartbeatSummary() }
    }
    private func flushHeartbeatSummary() {
        guard var data = heartbeatSummary.take() else { return }
        let now = ProcessInfo.processInfo.systemUptime
        data["suspended"] = .bool(suspension != nil)
        data["sourceAgesSeconds"] = .object(sourceTimes.mapValues { .number(max(0, now - $0)) })
        record(.heartbeat, source: "lifecycle", data: data)
    }
    private func flush(waitForCommit: Bool = false) {
        guard let store else { return }
        if !buffer.isEmpty {
            let batch = buffer, batchBytes = bufferedBytes
            buffer = []; bufferedBytes = 0; bufferedSince = nil
            queuedEvents += batch.count; queuedBytes += batchBytes
            pendingWriteBatches.append(.init(events: batch, bytes: batchBytes))
            updatePendingWriteCount()
            let backpressured = queuedEvents > Self.maximumQueuedEvents || queuedBytes > Self.maximumQueuedBytes
            if backpressured, isCollecting, !isStoppingCollection {
                beginAutomaticBacklogRecovery()
            }
        }
        if waitForCommit {
            // A boundary must include batches queued before this call, not only
            // the current in-memory buffer.
            schedulePendingWrites(using: store, force: true)
            store.flush()
        } else {
            schedulePendingWrites(using: store)
        }
    }
    private func updatePendingWriteCount() {
        pendingWriteEvents = queuedEvents + buffer.count
    }
    private func schedulePendingWrites(using store: EventStore, force: Bool = false) {
        guard !writeFailure else { return }
        if force {
            while !pendingWriteBatches.isEmpty { submitNextWrite(using: store) }
        } else if inFlightWriteBatches == 0, !pendingWriteBatches.isEmpty {
            submitNextWrite(using: store)
        }
    }
    private func submitNextWrite(using store: EventStore) {
        guard !pendingWriteBatches.isEmpty else { return }
        let batch = pendingWriteBatches.removeFirst()
        inFlightWriteBatches += 1
        let learning = recommender
        store.appendWithSequencesAsync(batch.events) { [weak self] result in
            if case .success(let sequences) = result {
                if let through = sequences.last { learning?.consumeCommitted(throughDatabaseSequence: through) }
            }
            Task { @MainActor in
                guard let self else { return }
                self.finishWrite(batch, result: result.map { _ in () })
            }
        }
    }
    private func finishWrite(_ batch: PendingWriteBatch, result: Result<Void, StoreError>) {
        inFlightWriteBatches = max(0, inFlightWriteBatches - 1)
        switch result {
        case .success:
            queuedEvents = max(0, queuedEvents - batch.events.count)
            queuedBytes = max(0, queuedBytes - batch.bytes)
            updatePendingWriteCount()
            stats.events += batch.events.count
            if let store { schedulePendingWrites(using: store) }
            finishAutomaticBacklogRecoveryIfPossible()
        case .failure(let error):
            // Retain the batch in this process so a later normal start can retry it
            // before collecting a new session. Never silently discard observations.
            pendingWriteBatches.insert(batch, at: 0)
            writeFailure = true
            isWritingBacklog = false
            resumeAfterBacklogDrain = false
            notice = nil
            updatePendingWriteCount()
            if isCollecting { pause(reason: "write_failure", waitForCommit: false) }
            errorMessage = "采集写入失败：\(error.localizedDescription)。已保护性暂停；解决磁盘或权限问题后重新开始，SmartApp 会先自动继续入库。"
        }
    }
    private func beginAutomaticBacklogRecovery() {
        guard isCollecting, !isWritingBacklog else { return }
        isWritingBacklog = true
        resumeAfterBacklogDrain = true
        errorMessage = nil
        notice = "写入暂时变慢，正在自动以小批次入库；完成后会恢复采集。"
        pause(reason: "write_backpressure", waitForCommit: false)
    }
    private func finishAutomaticBacklogRecoveryIfPossible() {
        guard isWritingBacklog, !isCollecting, !isStoppingCollection, !hasPendingWrites else { return }
        isWritingBacklog = false
        let shouldResume = resumeAfterBacklogDrain
        resumeAfterBacklogDrain = false
        guard shouldResume else { return }
        notice = "积压已自动写入，已恢复采集。"
        errorMessage = nil
        start()
    }
    private func refreshStats() {
        guard !statsBusy else { return }; statsBusy = true
        let url = databaseURL
        let reader = statsReader
        Task {
            do {
                let (latest, activeReader) = try await Task.detached(priority: .utility) {
                    // Keep the reader's event cursor and target-switch state between refreshes.
                    // The first read still checks all history; later reads process only new events.
                    let activeReader = try reader ?? EventStore(url: url, readOnly: true)
                    return (try activeReader.stats(), activeReader)
                }.value
                statsReader = activeReader
                stats = latest
                refreshDatabaseSize()
            } catch { errorMessage = error.localizedDescription }
            statsBusy = false
        }
    }
    private func refreshDatabaseSize() {
        let bytes = EventStore.databaseBytes(at: databaseURL)
        stats.bytes = bytes
        if isCollecting && bytes >= Self.databaseByteLimit {
            pause()
            errorMessage = "数据达到 2 GB 限额，已暂停采集。"
        }
    }
    private func estimatedWriteBytes(_ event: BehaviorEvent) -> Int {
        max(1, (try? JSONEncoder().encode(event).count) ?? 1_024)
    }
    private func openPrivacy(_ pane: String) {
        if let url = URL(string: "x-apple.systempreferences:com.apple.preference.security?\(pane)") { NSWorkspace.shared.open(url) }
    }
}
