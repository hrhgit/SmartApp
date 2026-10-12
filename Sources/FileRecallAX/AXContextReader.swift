import AppKit
import ApplicationServices
import FileRecallCore

/// Captured before our panel is shown; no window-list or screenshot fallback.
public struct FocusedAXWindow: @unchecked Sendable {
    let element: AXUIElement
    let pid: pid_t
    public let application: String
    public let applicationBundleID: String?
    public var isBrowser: Bool {
        ["com.apple.Safari", "com.apple.SafariTechnologyPreview", "com.google.Chrome", "com.microsoft.edgemac",
         "org.mozilla.firefox", "com.brave.Browser", "company.thebrowser.Browser"].contains(applicationBundleID ?? "")
    }

    @MainActor public static func capture(expectedTitle: String? = nil) throws -> FocusedAXWindow {
        guard AXIsProcessTrusted() else {
            throw RecommendationFailure("读取前台页面", "缺少辅助功能权限。请在系统设置 → 隐私与安全性 → 辅助功能中允许 SmartApp，再回到目标窗口重试。")
        }
        guard let app = NSWorkspace.shared.frontmostApplication,
              app.processIdentifier != ProcessInfo.processInfo.processIdentifier else {
            throw RecommendationFailure("读取前台页面", "请先聚焦需要推荐目录的应用，再唤出菜单并点击目录查找。")
        }
        let root = AXUIElementCreateApplication(app.processIdentifier)
        var value: CFTypeRef?
        let code = AXUIElementCopyAttributeValue(root, kAXFocusedWindowAttribute as CFString, &value)
        guard code == .success, let value, CFGetTypeID(value) == AXUIElementGetTypeID() else {
            throw RecommendationFailure("读取前台页面", "\(app.localizedName ?? "应用")未提供聚焦窗口（AX \(code.rawValue)）；请回到目标窗口后重试。")
        }
        let window = unsafeDowncast(value, to: AXUIElement.self)
        if let expectedTitle {
            var title: CFTypeRef?
            let status = AXUIElementCopyAttributeValue(window, kAXTitleAttribute as CFString, &title)
            guard status == .success, let title = title as? String, !expectedTitle.isEmpty,
                  title.contains(expectedTitle) else {
                throw RecommendationFailure("场景核对", "当前聚焦窗口不是指定的人工场景；请先在浏览器打开并聚焦对应页面。未调用模型。")
            }
        }
        return FocusedAXWindow(element: window,
                               pid: app.processIdentifier, application: app.localizedName ?? app.bundleIdentifier ?? "应用",
                               applicationBundleID: app.bundleIdentifier)
    }

    public func read(expectedPageURL: String? = nil) async throws -> PageContext {
        var queue: [(AXUIElement, Int, Bool)] = [(element, 0, false)]
        var index = 0
        var seen: [AXUIElement] = []
        var lines: [String] = []
        var webLines: [String] = []
        var unique = Set<String>()
        var semanticCount = 0
        var webSemanticCount = 0
        var hasWebArea = false
        let browser = isBrowser
        var currentWebArea: AXUIElement?
        while index < queue.count {
            try Task.checkCancellation()
            let stillCurrent = await MainActor.run {
                let front = NSWorkspace.shared.frontmostApplication?.processIdentifier
                guard front == pid || (!browser && front == ProcessInfo.processInfo.processIdentifier) else { return false }
                let root = AXUIElementCreateApplication(pid)
                var current: CFTypeRef?
                guard AXUIElementCopyAttributeValue(root, kAXFocusedWindowAttribute as CFString, &current) == .success,
                      let current, CFGetTypeID(current) == AXUIElementGetTypeID() else { return false }
                return CFEqual(current, element)
            }
            guard stillCurrent else { throw RecommendationFailure("读取前台页面", "读取期间聚焦应用或窗口已切换，请在目标窗口重新唤出。") }
            let (node, depth, webAncestor) = queue[index]; index += 1
            if seen.contains(where: { CFEqual($0, node) }) { continue }
            seen.append(node)
            if (try strictValue(node, "AXHidden") as? Bool) == true { continue }
            let role = try strictValue(node, kAXRoleAttribute) as? String ?? ""
            if role == "AXSecureTextField" { continue }
            let webContent = webAncestor || role == "AXWebArea"
            if role == "AXWebArea" {
                // Embedded frames belong to the verified current page. Only a
                // second independent page root makes the active page ambiguous.
                if browser && !webAncestor {
                    guard currentWebArea == nil else { throw RecommendationFailure("读取前台页面", "窗口暴露了多个页面，无法确认唯一当前页面；已停止查找。") }
                    currentWebArea = node
                }
                hasWebArea = true
            }
            if browser, webContent {
                guard let expectedPageURL, let webArea = currentWebArea,
                      let actual = try strictValue(webArea, kAXURLAttribute),
                      let url = (actual as? URL)?.absoluteString ?? actual as? String,
                      URL(string: url) == URL(string: expectedPageURL) else {
                    throw RecommendationFailure("读取前台页面", "当前页面地址无法核对或已变化，请回到页面重新唤出。")
                }
            }
            // Window chrome is unrelated task noise. Do not traverse toolbar/menu descendants
            // (address fields and background tab labels cannot make an empty page readable).
            if ["AXToolbar", "AXMenuBar"].contains(role) || (browser && role == "AXTabGroup") { continue }
            var parts: [String] = []
            var hasContentValue = false
            // Editor values/body text are retained. Empty wrappers and repeated labels are omitted.
            for key in [kAXTitleAttribute, kAXDescriptionAttribute, kAXValueAttribute, kAXHelpAttribute] {
                if browser && !webContent { continue }
                if let value = (try strictValue(node, key) as? String)?.trimmingCharacters(in: .whitespacesAndNewlines), !value.isEmpty,
                   !parts.contains(value) {
                    parts.append(value)
                    if key == kAXValueAttribute { hasContentValue = true }
                }
            }
            let text = parts.joined(separator: " · ")
            if !text.isEmpty, unique.insert(role + "|" + text).inserted {
                let line = role + ": " + text
                lines.append(line)
                if webContent { webLines.append(line) }
                // A window title alone is not usable page semantics.
                if depth > 0 && (!["AXWindow", "AXButton", "AXWebArea", "AXScrollArea", "AXRadioButton", "AXTabGroup"].contains(role) || hasContentValue) {
                    semanticCount += 1
                    if webContent { webSemanticCount += 1 }
                }
            }
            let children = try strictValue(node, kAXChildrenAttribute) as? [AXUIElement] ?? []
            queue += children.map { ($0, depth + 1, webContent) }
        }
        let useWebArea = browser
        if browser && !hasWebArea { throw RecommendationFailure("读取前台页面", "浏览器未提供可核对的活动页面内容，已停止查找。") }
        guard (useWebArea ? webSemanticCount : semanticCount) > 0 else {
            throw RecommendationFailure("读取前台页面", "\(application)没有提供可用的页面内容（仅标题、空树或 AX 读取失败）。已停止，未调用模型；请确认该应用的辅助功能支持后重试。")
        }
        let contentLines = useWebArea ? webLines : lines
        return PageContext(application: application, text: contentLines.joined(separator: "\n"), applicationBundleID: applicationBundleID,
                           visitedNodeCount: index)
    }
    private func strictValue(_ node: AXUIElement, _ key: String) throws -> CFTypeRef? {
        var value: CFTypeRef?
        let status = AXUIElementCopyAttributeValue(node, key as CFString, &value)
        switch status {
        case .success: return value
        case .attributeUnsupported, .noValue: return nil
        default:
            throw RecommendationFailure("读取前台页面", "\(application) 的 AX 读取失败（\(key)，代码 \(status.rawValue)）。已停止本次推荐，请返回目标窗口重试。")
        }
    }

}
