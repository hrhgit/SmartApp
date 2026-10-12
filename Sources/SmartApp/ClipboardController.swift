import AppKit
import Observation
import IntentCore
import func Carbon.IsSecureEventInputEnabled

enum ClipboardTransfer {
    static let supportedTypes: Set<NSPasteboard.PasteboardType> = [
        .string, .URL, .fileURL, .png, .tiff, .rtf, .rtfd, .html, .pdf, .init("public.jpeg")
    ]
    static let concealedTypes: Set<String> = ["org.nspasteboard.ConcealedType", "org.nspasteboard.TransientType",
        "com.agilebits.onepassword", "com.1password.1password", "de.petermaurer.TransientPasteboardType"]

    @MainActor static func read(_ board: NSPasteboard, origin: ClipboardCaptureOrigin, at: Date = Date()) throws -> ClipboardEntry? {
        if #available(macOS 15.4, *), board.accessBehavior == .alwaysDeny {
            throw StoreError("系统已禁止 SmartApp 读取剪贴板，请在系统设置的隐私与安全性中修改剪贴板权限。")
        }
        let version = board.changeCount
        let sourceItems = board.pasteboardItems ?? []
        guard !sourceItems.contains(where: { item in item.types.contains { concealedTypes.contains($0.rawValue) } }) else { return nil }
        var records: [ClipboardItemRecord] = []
        for item in sourceItems {
            let types = item.types.filter { supportedTypes.contains($0) }
            var representations: [ClipboardRepresentation] = []
            for type in types {
                guard let data = item.data(forType: type) else {
                    throw StoreError("系统没有提供剪贴板内容（\(type.rawValue)）。请重新复制后重试。")
                }
                representations.append(.init(type: type.rawValue, data: data))
            }
            guard !representations.isEmpty else { continue }
            records.append(.init(representations: representations, text: item.string(forType: .string) ?? item.string(forType: .URL),
                                 fileLocator: item.string(forType: .fileURL)))
        }
        guard board.changeCount == version else { throw StoreError("读取期间剪贴板已更新，等待新的内容。") }
        guard !records.isEmpty else { return nil }
        return .init(observedAt: at, origin: origin, items: records)
    }

    @MainActor static func write(_ entry: ClipboardEntry, to board: NSPasteboard) throws {
        let items = entry.items.map { record -> NSPasteboardItem in
            let item = NSPasteboardItem()
            for representation in record.representations {
                item.setData(representation.data, forType: .init(representation.type))
            }
            return item
        }
        guard !items.isEmpty else { throw StoreError("这条记录没有可复制内容。") }
        board.clearContents()
        guard board.writeObjects(items) else { throw StoreError("无法写入系统剪贴板。") }
    }
}

@MainActor @Observable
final class ClipboardController {
    private(set) var isRecording = false
    private(set) var isTemporarilyPaused = false
    private(set) var summaries: [ClipboardSummary] = []
    private(set) var selectedEntry: ClipboardEntry?
    private(set) var previewImage: NSImage?
    private(set) var totalCount = 0
    private(set) var hasMore = false
    private(set) var isLoading = false
    private(set) var copiedID: String?
    var search = "" { didSet { reload() } }
    var pinnedOnly = false { didSet { reload() } }
    var error: String?
    var onFindDirectories: ((ClipboardEntry) -> Void)?
    var excludedBundles: () -> [String] = { [] }

    @ObservationIgnored private let store: ClipboardHistoryStore?
    @ObservationIgnored private let board: NSPasteboard
    @ObservationIgnored private let liveObservation: Bool
    @ObservationIgnored private let defaults: UserDefaults
    @ObservationIgnored private var timer: Timer?
    @ObservationIgnored private var tracker = ClipboardObservationTracker()
    @ObservationIgnored private var lifecycleTokens: [NSObjectProtocol] = []
    @ObservationIgnored private var lockTokens: [NSObjectProtocol] = []
    @ObservationIgnored private var blockers = Set<String>()
    @ObservationIgnored private var privacyBlocked = false
    @ObservationIgnored private let ioQueue = DispatchQueue(label: "SmartApp.clipboard-io", qos: .utility)
    @ObservationIgnored private let writeGroup = DispatchGroup()
    @ObservationIgnored private var queryGeneration = UUID()
    @ObservationIgnored private var selectionGeneration = UUID()

    init(directory: URL, defaults: UserDefaults = .standard, board: NSPasteboard = .general, liveObservation: Bool = true) {
        self.defaults = defaults; self.board = board; self.liveObservation = liveObservation
        do { store = try ClipboardHistoryStore(url: directory.appendingPathComponent("clipboard/history-v1.sqlite")) }
        catch { store = nil; self.error = error.localizedDescription }
        reload()
    }

    func startAtLaunch() {
        guard liveObservation, defaults.bool(forKey: "clipboardRecordingEnabledV1") else { return }
        start()
    }

    func toggle() { isRecording ? stop() : start() }
    func start() {
        guard !isRecording, store != nil else { return }
        isRecording = true; error = nil
        defaults.set(true, forKey: "clipboardRecordingEnabledV1")
        installBoundaryObservers()
        tracker.begin(version: board.changeCount)
        // Existing contents can be browsed, but their original update time is unknown.
        capture(origin: .existingTimeUnknown)
        if liveObservation {
            let timer = Timer(timeInterval: 0.5, repeats: true) { [weak self] _ in
                MainActor.assumeIsolated { self?.poll() }
            }
            RunLoop.main.add(timer, forMode: .common); self.timer = timer
        }
    }
    func stop() {
        isRecording = false; isTemporarilyPaused = false
        defaults.set(false, forKey: "clipboardRecordingEnabledV1")
        timer?.invalidate(); timer = nil; tracker.reset(); privacyBlocked = false; blockers = []
        removeBoundaryObservers()
    }

    func poll() {
        guard isRecording else { return }
        let blocked = !blockers.isEmpty || !allowsReading()
        isTemporarilyPaused = blocked
        if blocked { tracker.reset(); privacyBlocked = true; return }
        if privacyBlocked || tracker.version == nil {
            tracker.begin(version: board.changeCount); privacyBlocked = false
            return
        }
        guard tracker.changed(to: board.changeCount) else { return }
        capture(origin: .observedChange)
    }

    func freezeForRecommendation(at: Date = Date()) -> ClipboardRecommendationContext? {
        guard isRecording, blockers.isEmpty, allowsReading() else { return nil }
        poll()
        return tracker.automaticContext(currentVersion: board.changeCount, at: at)
    }

    private func allowsReading() -> Bool {
        guard liveObservation else { return true }
        guard !IsSecureEventInputEnabled(), let bundle = NSWorkspace.shared.frontmostApplication?.bundleIdentifier else { return false }
        return bundle == CapturePolicy.ownBundleID || !CapturePolicy.isExcluded(bundle, rules: excludedBundles())
    }

    private func capture(origin: ClipboardCaptureOrigin) {
        guard blockers.isEmpty, allowsReading(), let store else { tracker.reset(); privacyBlocked = true; return }
        do {
            let version = board.changeCount
            guard let entry = try ClipboardTransfer.read(board, origin: origin) else { return }
            tracker.accepted(entry, version: version)
            writeGroup.enter()
            ioQueue.async { [weak self, writeGroup] in
                defer { writeGroup.leave() }
                do {
                    try store.insert(entry)
                    Task { @MainActor [weak self] in self?.reload() }
                } catch {
                    let message = error.localizedDescription
                    Task { @MainActor [weak self] in self?.stop(); self?.error = message }
                }
            }
        } catch { self.error = error.localizedDescription }
    }

    func reload(loadMore: Bool = false) {
        guard let store else { return }
        let token = UUID(); queryGeneration = token; isLoading = true
        let search = search; let pinnedOnly = pinnedOnly
        let offset = loadMore ? summaries.count : 0
        ioQueue.async { [weak self] in
            do {
                let rows = try store.summaries(search: search, pinnedOnly: pinnedOnly, offset: offset, limit: 100)
                let count = try store.count()
                Task { @MainActor [weak self] in
                    guard let self, self.queryGeneration == token else { return }
                    if loadMore { self.summaries.append(contentsOf: rows) } else { self.summaries = rows }
                    self.totalCount = count; self.hasMore = rows.count == 100; self.isLoading = false
                }
            } catch {
                let message = error.localizedDescription
                Task { @MainActor [weak self] in
                    guard let self, self.queryGeneration == token else { return }
                    self.error = message; self.isLoading = false
                }
            }
        }
    }

    func select(id: String) {
        guard let store else { return }
        let token = UUID(); selectionGeneration = token
        selectedEntry = nil; previewImage = nil
        ioQueue.async { [weak self] in
            do {
                let entry = try store.entry(id: id)
                Task { @MainActor [weak self] in
                    guard let self, self.selectionGeneration == token else { return }
                    self.selectedEntry = entry
                    if let entry, entry.kind == .image,
                       let data = entry.items.flatMap(\.representations).first(where: { ["public.png", "public.tiff", "public.jpeg"].contains($0.type) })?.data {
                        self.previewImage = NSImage(data: data)
                    }
                }
            } catch {
                let message = error.localizedDescription
                Task { @MainActor [weak self] in self?.error = message }
            }
        }
    }
    func copySelected() {
        guard let selectedEntry else { return }
        do { try ClipboardTransfer.write(selectedEntry, to: board); copiedID = selectedEntry.id; poll() }
        catch { self.error = error.localizedDescription }
    }
    func findSelected() { if let selectedEntry { onFindDirectories?(selectedEntry) } }
    func pin(_ summary: ClipboardSummary) {
        mutate { try $0.setPinned(id: summary.id, pinned: !summary.isPinned) }
        if selectedEntry?.id == summary.id { selectedEntry?.isPinned = !summary.isPinned }
    }
    func delete(_ summary: ClipboardSummary) {
        tracker.remove(id: summary.id)
        if selectedEntry?.id == summary.id { clearSelection() }
        mutate { try $0.delete(id: summary.id) }
    }
    func clear() {
        tracker.begin(version: board.changeCount); clearSelection()
        mutate { try $0.clear() }
    }
    func clearSelection() { selectionGeneration = UUID(); selectedEntry = nil; previewImage = nil }
    private func mutate(_ mutation: @escaping @Sendable (ClipboardHistoryStore) throws -> Void) {
        guard let store else { return }
        writeGroup.enter()
        ioQueue.async { [weak self, writeGroup] in
            defer { writeGroup.leave() }
            do { try mutation(store); Task { @MainActor [weak self] in self?.reload() } }
            catch { let message = error.localizedDescription; Task { @MainActor [weak self] in self?.error = message } }
        }
    }

    private func installBoundaryObservers() {
        guard liveObservation else { return }
        for (notification, name, blocked) in [(NSWorkspace.willSleepNotification, "sleep", true),
            (NSWorkspace.didWakeNotification, "sleep", false),
            (NSWorkspace.sessionDidResignActiveNotification, "session", true),
            (NSWorkspace.sessionDidBecomeActiveNotification, "session", false)] {
            lifecycleTokens.append(NSWorkspace.shared.notificationCenter.addObserver(forName: notification, object: nil, queue: .main) { [weak self] _ in
                MainActor.assumeIsolated { self?.setBoundary(name, blocked: blocked) }
            })
        }
        for (name, blocked) in [("com.apple.screenIsLocked", true), ("com.apple.screenIsUnlocked", false)] {
            lockTokens.append(DistributedNotificationCenter.default().addObserver(forName: Notification.Name(name), object: nil, queue: .main) { [weak self] _ in
                MainActor.assumeIsolated { self?.setBoundary("lock", blocked: blocked) }
            })
        }
    }
    func setBoundary(_ name: String, blocked: Bool) {
        if blocked { blockers.insert(name) } else { blockers.remove(name) }
        tracker.reset(); isTemporarilyPaused = !blockers.isEmpty; privacyBlocked = true
    }
    private func removeBoundaryObservers() {
        for token in lifecycleTokens { NSWorkspace.shared.notificationCenter.removeObserver(token) }
        for token in lockTokens { DistributedNotificationCenter.default().removeObserver(token) }
        lifecycleTokens = []; lockTokens = []
    }
    func shutdown() {
        timer?.invalidate(); timer = nil; tracker.reset(); removeBoundaryObservers()
        writeGroup.wait(); store?.flush(); clearSelection()
    }
}
