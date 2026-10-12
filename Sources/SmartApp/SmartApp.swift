import AppKit
import SwiftUI
import Carbon
import IntentCore
import FileRecallAX
import FileRecallCore

enum LaunchContext {
    static func shouldCollect(isLogin: Bool, loginEnabled: Bool, consent: Bool, explicitlyCollect: Bool) -> Bool {
        explicitlyCollect || (isLogin && loginEnabled && consent)
    }
    static func isLogin(_ event: NSAppleEventDescriptor?) -> Bool {
        event?.eventID == kAEOpenApplication &&
            event?.paramDescriptor(forKeyword: keyAEPropData)?.enumCodeValue == keyAELaunchedAsLogInItem
    }
}

@MainActor
final class PickerPanel: NSPanel {
    var dismiss: (() -> Void)?
    var dismissWhenResigning = true
    override var canBecomeKey: Bool { true }
    override func cancelOperation(_ sender: Any?) { dismiss?() }
    override func resignKey() { super.resignKey(); if dismissWhenResigning { dismiss?() } }
}

@MainActor
final class AppDelegate: NSObject, NSApplicationDelegate, NSMenuDelegate, NSWindowDelegate {
    let model = RecorderModel()
    private var statusItem: NSStatusItem?
    private var dashboard: NSWindow?
    private var picker: PickerPanel?
    private var toolsPanel: PickerPanel?
    private var toolsClickMonitor: Any?
    private var toolsKeyMonitor: Any?
    private var toolsActivationMonitor: NSObjectProtocol?
    private struct RecallInvocation {
        let source: Result<FocusedAXWindow, Error>
        let clipboard: ClipboardRecommendationContext?
        let at: Date
        let pointer: CGPoint
    }
    private var recallInvocation: RecallInvocation?
    private var returningToMenu = false
    private lazy var directoryLookup: DirectoryLookupController = {
        let lookup = DirectoryLookupController(databaseURL: model.databaseURL,
            excludedBundles: { [weak self] in self?.model.recallExclusionRules ?? [] },
            validateSource: { [weak self] source in
                guard let self else { throw CancellationError() }
                return try await self.model.validateRecallSource(source)
            })
        lookup.onOpen = { [weak self] target, decisionID in
            await self?.model.openRecallTarget(target, decisionID: decisionID) ?? false
        }
        lookup.onCopyPath = { path in
            NSPasteboard.general.clearContents(); NSPasteboard.general.setString(path, forType: .string)
        }
        return lookup
    }()

    func applicationDidFinishLaunching(_ notification: Notification) {
        let others = NSRunningApplication.runningApplications(withBundleIdentifier: RecorderModel.bundleID)
        if let existing = others.first(where: { $0.processIdentifier != ProcessInfo.processInfo.processIdentifier }) {
            existing.activate(); NSApp.terminate(nil); return
        }
        NSApp.setActivationPolicy(.accessory)
        model.prepareBrowserConnection()
        let item = NSStatusBar.system.statusItem(withLength: NSStatusItem.squareLength)
        item.button?.image = SmartAppBranding.menuBarIcon()
            ?? NSImage(systemSymbolName: "cursorarrow.rays", accessibilityDescription: "SmartApp")
        let menu = NSMenu(); menu.delegate = self; menu.autoenablesItems = false
        item.menu = menu; statusItem = item
        model.onShowPicker = { [weak self] session in self?.showPicker(session) }
        model.clipboard.onFindDirectories = { [weak self] entry in self?.findUsingClipboard(entry) }
        model.clipboard.startAtLaunch()
        let isLogin = LaunchContext.isLogin(NSAppleEventManager.shared().currentAppleEvent)
        model.startAtLaunch(isLogin: isLogin, explicitlyCollect: CommandLine.arguments.contains("--collect"))
        if !isLogin && !CommandLine.arguments.contains("--background") { showDashboard() }
    }
    func applicationShouldHandleReopen(_ sender: NSApplication, hasVisibleWindows flag: Bool) -> Bool { showDashboard(); return true }
    func applicationShouldTerminateAfterLastWindowClosed(_ sender: NSApplication) -> Bool { false }
    func applicationShouldTerminate(_ sender: NSApplication) -> NSApplication.TerminateReply {
        dismissTools(); directoryLookup.shutdown()
        model.shutdown(); return .terminateNow
    }
    func windowWillClose(_ notification: Notification) {
        if notification.object as? NSWindow === dashboard {
            model.invocation.cancelCalibration(); model.clipboard.clearSelection()
        }
    }
    func menuWillOpen(_ menu: NSMenu) {
        // Freeze all rows for the duration of native menu tracking.
        menu.removeAllItems()
        let status = NSMenuItem(title: "SmartApp · \(model.statusText)", action: nil, keyEquivalent: "")
        status.isEnabled = false; menu.addItem(status)
        let count = NSMenuItem(title: "本次 \(model.sessionSwitches) 次目标切换", action: nil, keyEquivalent: "")
        count.isEnabled = false; menu.addItem(count)
        menu.addItem(.separator())
        let toggleItem = add(menu, model.isWritingBacklog ? "正在自动入库…" : (model.isCollecting ? "暂停采集" : "开始采集"), #selector(toggle))
        toggleItem.isEnabled = !model.isWritingBacklog
        add(menu, "打开采集面板", #selector(showDashboard))
        add(menu, "唤出推荐目标", #selector(showDestinations))
        add(menu, "查看剪贴板历史", #selector(showClipboard))
        add(menu, "导出上下文数据…", #selector(export))
        menu.addItem(.separator())
        add(menu, "退出 SmartApp", #selector(quit), key: "q")
    }
    @discardableResult private func add(_ menu: NSMenu, _ title: String, _ action: Selector, key: String = "") -> NSMenuItem {
        let item = NSMenuItem(title: title, action: action, keyEquivalent: key); item.target = self
        menu.addItem(item)
        return item
    }
    @objc private func toggle() { model.toggle() }
    @objc private func export() { model.exportDataset() }
    @objc private func quit() { NSApp.terminate(nil) }
    @objc private func showDestinations() { model.invokePicker() }
    @objc private func showClipboard() {
        if recallInvocation == nil { recallInvocation = captureInvocation() }
        closePickerForTransition()
        model.clipboard.reload()
        let panel = makeToolsPanel(title: "SmartApp · 剪贴板", size: CGSize(width: 780, height: 560))
        panel.contentView = NSHostingView(rootView: ClipboardHistoryView(model: model.clipboard, close: { [weak self] in self?.dismissTools() }))
        panel.makeKeyAndOrderFront(nil)
        installToolsMonitors(directory: false)
    }
    @objc private func showDashboard() {
        if dashboard == nil {
            let window = NSWindow(contentRect: NSRect(x: 0, y: 0, width: 1060, height: 800),
                                  styleMask: [.titled, .closable, .miniaturizable, .resizable], backing: .buffered, defer: false)
            window.title = "SmartApp · 上下文采集"
            window.acceptsMouseMovedEvents = true
            window.delegate = self
            window.minSize = NSSize(width: 900, height: 640)
            window.contentView = NSHostingView(rootView: DashboardView(model: model))
            window.isReleasedWhenClosed = false; window.center(); dashboard = window
        }
        dashboard?.makeKeyAndOrderFront(nil); NSApp.activate()
    }
    private func showPicker(_ session: PickerSession) {
        if !returningToMenu {
            dismissTools()
            recallInvocation = captureInvocation()
        }
        returningToMenu = false
        picker?.dismiss = nil
        picker?.orderOut(nil)
        let point = NSEvent.mouseLocation
        let screen = NSScreen.screens.first(where: { $0.frame.contains(point) }) ?? NSScreen.main
        let visibleFrame = screen?.visibleFrame ?? NSRect(x: 0, y: 0, width: 1000, height: 800)
        let frame = DestinationPickerLayout.frame(around: point, visibleFrame: visibleFrame)
        let scale = frame.width / DestinationPickerLayout.diameter
        let panel = PickerPanel(contentRect: frame,
                                styleMask: [.borderless, .nonactivatingPanel], backing: .buffered, defer: false)
        panel.level = .floating; panel.isOpaque = false; panel.backgroundColor = .clear
        panel.hasShadow = false; panel.isReleasedWhenClosed = false
        panel.acceptsMouseMovedEvents = true
        panel.collectionBehavior = [.canJoinAllSpaces, .fullScreenAuxiliary]
        panel.title = "推荐目标"
        let dismiss: () -> Void = { [weak self, weak panel] in
            panel?.dismiss = nil; panel?.orderOut(nil); self?.model.pickerClosed(session)
            self?.recallInvocation = nil
        }
        panel.dismiss = dismiss
        let view = DestinationPickerView(items: DestinationPickerItem.resolve(session.choices), choose: { [weak self, weak panel] choice in
            panel?.dismiss = nil; panel?.orderOut(nil); self?.model.choose(choice, in: session)
            self?.recallInvocation = nil
        }, close: dismiss, findDirectories: { [weak self] in
            self?.model.pickerClosed(session); self?.closePickerForTransition(); self?.showDirectoryLookup()
        }, showClipboard: { [weak self] in
            self?.model.pickerClosed(session); self?.showClipboard()
        })
        let hosting = NSHostingView(rootView: view.scaleEffect(scale).frame(width: frame.width, height: frame.height))
        hosting.sizingOptions = []
        panel.contentView = hosting
        picker = panel; panel.makeKeyAndOrderFront(nil)
        DispatchQueue.main.async { [weak self] in self?.model.pickerShown(session) }
    }

    private func captureInvocation() -> RecallInvocation {
        let at = Date()
        return .init(source: model.captureRecallSource(), clipboard: model.clipboard.freezeForRecommendation(at: at), at: at, pointer: NSEvent.mouseLocation)
    }
    private func closePickerForTransition() {
        // A tool transition is not a recommendation selection or training event.
        model.leavePickerForTools(); picker?.dismiss = nil; picker?.orderOut(nil)
    }
    private func showDirectoryLookup(explicitClipboard: ClipboardEntry? = nil) {
        guard let invocation = recallInvocation else { return }
        let clipboard = explicitClipboard.map { ClipboardRecommendationContext(entry: $0, explicitlySelected: true) } ?? invocation.clipboard
        model.prepareDirectoryLookup()
        let panel = makeToolsPanel(title: "SmartApp · FileRecall", size: CGSize(width: 760, height: 540))
        panel.contentView = NSHostingView(rootView: DirectoryLookupView(model: directoryLookup,
            close: { [weak self] in self?.dismissTools() }, back: { [weak self] in self?.returnToPicker() }))
        panel.makeKeyAndOrderFront(nil)
        installToolsMonitors(directory: true)
        directoryLookup.begin(source: invocation.source, clipboard: clipboard, at: invocation.at)
    }
    private func findUsingClipboard(_ entry: ClipboardEntry) {
        guard recallInvocation != nil else {
            model.clipboard.error = "请先回到工作窗口，唤出菜单后点击剪贴板，再选择这条内容查找目录。"
            return
        }
        model.clipboard.clearSelection()
        showDirectoryLookup(explicitClipboard: entry)
    }
    private func makeToolsPanel(title: String, size: CGSize) -> PickerPanel {
        removeToolsMonitors()
        directoryLookup.close()
        toolsPanel?.dismiss = nil; toolsPanel?.orderOut(nil)
        let pointer = recallInvocation?.pointer ?? NSEvent.mouseLocation
        let screen = NSScreen.screens.first { $0.frame.contains(pointer) } ?? NSScreen.main
        let frame = PopupPlacement.frame(at: pointer, size: size, visibleFrame: screen?.visibleFrame ?? CGRect(x: 0, y: 0, width: 1000, height: 700))
        let panel = PickerPanel(contentRect: frame, styleMask: [.borderless, .nonactivatingPanel], backing: .buffered, defer: false)
        panel.level = .floating; panel.isOpaque = false; panel.backgroundColor = .clear
        panel.hasShadow = true; panel.isReleasedWhenClosed = false; panel.isFloatingPanel = true
        panel.dismissWhenResigning = false
        panel.collectionBehavior = [.canJoinAllSpaces, .fullScreenAuxiliary]
        panel.title = title; panel.dismiss = { [weak self] in self?.dismissTools() }
        toolsPanel = panel
        return panel
    }
    private func returnToPicker() {
        dismissTools(keepInvocation: true); returningToMenu = true; model.invokePicker()
    }
    private func dismissTools(keepInvocation: Bool = false) {
        toolsPanel?.dismiss = nil; toolsPanel?.orderOut(nil); toolsPanel = nil
        removeToolsMonitors(); directoryLookup.close(); model.clipboard.clearSelection()
        if !keepInvocation { recallInvocation = nil }
    }
    private func installToolsMonitors(directory: Bool) {
        removeToolsMonitors()
        toolsClickMonitor = NSEvent.addGlobalMonitorForEvents(matching: [.leftMouseDown, .rightMouseDown]) { [weak self] _ in
            MainActor.assumeIsolated { self?.dismissTools() }
        }
        let sourceBundle = try? recallInvocation?.source.get().applicationBundleID
        toolsActivationMonitor = NSWorkspace.shared.notificationCenter.addObserver(forName: NSWorkspace.didActivateApplicationNotification,
            object: nil, queue: .main) { [weak self] notification in
                let bundleID = (notification.userInfo?[NSWorkspace.applicationUserInfoKey] as? NSRunningApplication)?.bundleIdentifier
                MainActor.assumeIsolated {
                    guard let bundleID, bundleID != sourceBundle, bundleID != RecorderModel.bundleID else { return }
                    self?.dismissTools()
                }
            }
        toolsKeyMonitor = NSEvent.addLocalMonitorForEvents(matching: [.keyDown, .leftMouseDown, .rightMouseDown]) { [weak self] event in
            let consumed = MainActor.assumeIsolated {
                guard let self, let panel = self.toolsPanel, panel.isVisible else { return false }
                if event.type != .keyDown {
                    if let window = event.window, window !== panel, window.parent !== panel,
                       window.sheetParent !== panel, window.level != .popUpMenu {
                        self.dismissTools()
                    }
                    return false
                }
                if event.keyCode == 53 { self.dismissTools(); return true }
                guard directory, !(panel.firstResponder is NSTextView) else { return false }
                if event.keyCode == 125 { self.directoryLookup.moveSelection(1); return true }
                if event.keyCode == 126 { self.directoryLookup.moveSelection(-1); return true }
                if event.keyCode == 36, let row = self.directoryLookup.selectedRow {
                    Task { if await self.directoryLookup.open(row) { self.dismissTools() } }; return true
                }
                if event.modifierFlags.contains(.command), event.charactersIgnoringModifiers == "c", let row = self.directoryLookup.selectedRow {
                    self.directoryLookup.copy(row); return true
                }
                return false
            }
            return consumed ? nil : event
        }
    }
    private func removeToolsMonitors() {
        if let toolsClickMonitor { NSEvent.removeMonitor(toolsClickMonitor) }
        if let toolsKeyMonitor { NSEvent.removeMonitor(toolsKeyMonitor) }
        if let toolsActivationMonitor { NSWorkspace.shared.notificationCenter.removeObserver(toolsActivationMonitor) }
        toolsActivationMonitor = nil
        toolsClickMonitor = nil; toolsKeyMonitor = nil
    }
}

@main
@MainActor
enum SmartApp {
    static func main() {
        let app = NSApplication.shared
        let delegate = AppDelegate(); app.delegate = delegate
        withExtendedLifetime(delegate) { app.run() }
    }
}
