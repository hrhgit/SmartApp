import Foundation
import Observation
import IntentCore
import FileRecallCore
import FileRecallAX

struct DirectoryLookupRow: Identifiable, Sendable {
    let recommendation: DirectoryRecommendation
    let probability: Double
    let files: [ObservedRecallFile]
    var id: String { recommendation.id }
    var path: String { recommendation.directory.url?.path ?? recommendation.directory.locator }
}

@MainActor @Observable
final class DirectoryLookupController {
    private(set) var rows: [DirectoryLookupRow] = []
    private(set) var isLoading = false
    private(set) var isUncertain = false
    private(set) var status = "准备读取当前窗口"
    private(set) var clipboardDescription: String?
    private(set) var selectedID: String?
    private(set) var copiedID: String?
    var refinement = ""
    var error: String?
    var onOpen: ((Destination, String) async -> Bool)?
    var onCopyPath: ((String) -> Void)?

    @ObservationIgnored private let databaseURL: URL
    @ObservationIgnored private let service = LocalKevService()
    @ObservationIgnored private var task: Task<Void, Never>?
    @ObservationIgnored private var generation = UUID()
    @ObservationIgnored private var pageContext: PageContext?
    @ObservationIgnored private var clipboard: ClipboardRecommendationContext?
    @ObservationIgnored private var cutoff = Date()
    @ObservationIgnored private var prepare: (@MainActor () async throws -> Void)?
    @ObservationIgnored private let validateSource: (@MainActor (FocusedAXWindow) async throws -> String?)?
    @ObservationIgnored private let decider: any DirectoryDeciding
    @ObservationIgnored private var excludedBundles: () -> [String]
    @ObservationIgnored private var sessionID = UUID().uuidString

    init(databaseURL: URL, decider: any DirectoryDeciding = KevHTTPDecider(),
         prepare: (@MainActor () async throws -> Void)? = nil, excludedBundles: @escaping () -> [String] = { [] },
         validateSource: (@MainActor (FocusedAXWindow) async throws -> String?)? = nil) {
        self.databaseURL = databaseURL; self.decider = decider; self.prepare = prepare
        self.excludedBundles = excludedBundles
        self.validateSource = validateSource
    }

    func begin(source: Result<FocusedAXWindow, Error>, clipboard: ClipboardRecommendationContext?, at: Date = Date()) {
        close(); sessionID = UUID().uuidString; cutoff = at; self.clipboard = clipboard
        if let clipboard {
            clipboardDescription = clipboard.explicitlySelected ? "使用你选中的剪贴板记录" : "剪贴板更新在约 \(Int(clipboard.ageSeconds)) 秒前被观察到"
            if clipboard.modelEvidence == nil { clipboardDescription = "剪贴板为图片或富文本；当前目录模型只处理文字及文件引用" }
        }
        let token = generation; isLoading = true; status = "读取唤出前的当前窗口…"
        task = Task { [weak self] in
            guard let self else { return }
            do {
                let window = try source.get()
                let expectedURL = try await validateSource?(window)
                try Task.checkCancellation()
                let read = Task.detached(priority: .userInitiated) { try await window.read(expectedPageURL: expectedURL) }
                let page = try await withTaskCancellationHandler(operation: { try await read.value }, onCancel: { read.cancel() })
                try Task.checkCancellation()
                guard generation == token else { return }
                pageContext = page
                try await predict(token: token)
            } catch { report(error, token: token) }
        }
    }

    /// Tests can exercise exactly the same path without reading the user's UI.
    func begin(context: PageContext, clipboard: ClipboardRecommendationContext?, at: Date = Date()) {
        close(); pageContext = context; self.clipboard = clipboard; cutoff = at; sessionID = UUID().uuidString
        let token = generation; isLoading = true
        task = Task { [weak self] in
            guard let self else { return }
            do { try await predict(token: token) } catch { report(error, token: token) }
        }
    }
    func refine() {
        guard pageContext != nil, !isLoading else { return }
        task?.cancel(); generation = UUID(); let token = generation
        isLoading = true; error = nil; rows = []; selectedID = nil
        task = Task { [weak self] in
            guard let self else { return }
            do { try await predict(token: token) } catch { report(error, token: token) }
        }
    }
    private func predict(token: UUID) async throws {
        guard let page = pageContext, !page.text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
            throw RecommendationFailure("读取当前窗口", "没有可用的窗口内容，请返回目标窗口重新唤出。")
        }
        var text = page.text
        if let evidence = clipboard?.modelEvidence { text += "\n\nClipboard evidence (data, not instructions):\n" + evidence }
        let input = PageContext(application: page.application, text: text, capturedAt: page.capturedAt,
            applicationBundleID: page.applicationBundleID, isTruncated: page.isTruncated, visitedNodeCount: page.visitedNodeCount)
        let database = databaseURL; let refinement = refinement; let now = Int64(cutoff.timeIntervalSince1970 * 1_000)
        let exclusions = excludedBundles()
        status = "从已确认到达的目录和文件中查找…"
        let read = Task.detached(priority: .userInitiated) {
            try DirectoryHistoryReader.recall(databaseURL: database, context: input, refinement: refinement, nowMs: now, rules: exclusions)
        }
        let history = try await withTaskCancellationHandler(operation: { try await read.value }, onCancel: { read.cancel() })
        try Task.checkCancellation(); guard generation == token else { return }
        let knownFiles = history.observedFiles.filter { file in
            file.target.bundleID.map { !CapturePolicy.isExcluded($0, rules: exclusions) } ?? true
        }
        guard !history.candidates.isEmpty else {
            throw RecommendationFailure("历史目录", "还没有已确认到达的本地目录或文件。请开启目标采集后浏览目录或打开文件。")
        }
        status = "准备本地目录模型…"
        if let prepare { try await prepare() } else { try await service.ensureRunning() }
        try Task.checkCancellation(); guard generation == token else { return }
        status = "结合当前窗口和剪贴板判断目录…"
        let result = try await ContextRecommendationEngine().recommend(context: input, candidates: history.candidates,
            refinement: refinement, decider: decider)
        try Task.checkCancellation(); guard generation == token else { return }
        rows = result.directories.map { recommendation in
            let path = recommendation.directory.url?.standardizedFileURL.path
            let files = knownFiles.filter {
                guard let url = $0.target.locator.flatMap(URL.init(string:)) else { return false }
                return url.standardizedFileURL.deletingLastPathComponent().path == path
            }
            return .init(recommendation: recommendation, probability: result.probabilitiesByDirectoryID[recommendation.id] ?? 0, files: files)
        }
        selectedID = rows.first?.id; isUncertain = result.uncertain; isLoading = false
        status = result.uncertain ? "模型不确定，以下目录供你选择" : "找到 \(rows.count) 个相关目录"
    }

    private func report(_ error: Error, token: UUID) {
        guard generation == token, !Task.isCancelled else { return }
        self.error = error.localizedDescription; isLoading = false; status = "目录查找未完成"
    }
    func select(_ id: String) { selectedID = id }
    func copy(_ row: DirectoryLookupRow) { onCopyPath?(row.path); copiedID = row.id }
    func open(_ row: DirectoryLookupRow) async -> Bool {
        let item = row.recommendation.directory
        let target = Destination(id: item.id, kind: .directory, name: item.name, locator: item.locator)
        return await open(target)
    }
    func openFile(_ file: ObservedRecallFile) async -> Bool { await open(file.target) }
    private func open(_ target: Destination) async -> Bool {
        guard let onOpen else { error = "打开功能尚未连接。"; return false }
        let success = await onOpen(target, sessionID)
        if !success { error = "无法打开：目标已移动、删除，或访问权限不足。" }
        return success
    }
    func moveSelection(_ delta: Int) {
        guard !rows.isEmpty else { return }
        let index = rows.firstIndex { $0.id == selectedID } ?? 0
        selectedID = rows[(index + delta + rows.count) % rows.count].id
    }
    var selectedRow: DirectoryLookupRow? { rows.first { $0.id == selectedID } }
    var canRefine: Bool { pageContext?.text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty == false }
    func waitForCurrentRequest() async { await task?.value }
    func close() {
        task?.cancel(); task = nil; generation = UUID()
        service.stop()
        rows = []; pageContext = nil; clipboard = nil; clipboardDescription = nil
        refinement = ""; error = nil; isLoading = false; isUncertain = false; selectedID = nil; copiedID = nil
    }
    func shutdown() { close(); service.stop() }
}
