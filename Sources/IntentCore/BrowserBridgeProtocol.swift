import Foundation

/// Only location/state metadata crosses this boundary; no DOM, input text or titles.
public struct BrowserBridgeTab: Codable, Equatable, Sendable {
    public let id: String
    public let url: String
    public let index: Int
    public let pinned: Bool
}
public struct BrowserBridgeWindow: Codable, Equatable, Sendable {
    public let id: String
    public let front: Bool
    public let activeID: String
    public let minimized: Bool
    public let tabs: [BrowserBridgeTab]
}
public struct BrowserBridgeSnapshot: Codable, Equatable, Sendable {
    public let windows: [BrowserBridgeWindow]
}
public struct BrowserBridgeMessage: Codable, Sendable {
    public let version: Int
    public let sequence: Int
    public let capturedAt: Double
    public let reason: String
    public let requestID: String?
    public let profileID: String?
    public let snapshot: BrowserBridgeSnapshot
    public var isValid: Bool {
        guard version == 2, sequence > 0, capturedAt.isFinite,
              profileID == nil || UUID(uuidString: profileID!) != nil,
              Self.reasons.contains(reason), snapshot.windows.count <= 1,
              Set(snapshot.windows.map(\.id)).count == snapshot.windows.count,
              snapshot.windows.filter(\.front).count == snapshot.windows.count else { return false }
        var ids = Set<String>()
        for w in snapshot.windows {
            guard Self.numericID(w.id), Self.numericID(w.activeID), w.tabs.count <= 1,
                  !w.minimized, w.tabs.allSatisfy({ $0.id == w.activeID }) else { return false }
            for t in w.tabs {
                guard Self.numericID(t.id), ids.insert(t.id).inserted, t.url.utf8.count <= 65_536,
                      t.index >= 0 else { return false }
            }
        }
        return true
    }
    public static let reasons: Set<String> = ["initial", "resync", "tab_activated", "window_focus",
        "navigation", "history", "fragment", "tab_created", "tab_removed", "tab_updated",
        "tab_moved", "tab_attached", "tab_detached", "tab_replaced", "window_created",
        "window_removed", "window_bounds", "settled"]
    private static func numericID(_ value: String) -> Bool {
        !value.isEmpty && value.count < 20 && value.utf8.allSatisfy { (48...57).contains($0) }
    }
}

/// Native Messaging framing; bounds apply before allocating/decoding payloads.
public enum BrowserBridgeWire {
    public static let maximum = 4 * 1_024 * 1_024
    public static func frame(_ data: Data) throws -> Data {
        guard !data.isEmpty, data.count <= maximum else { throw StoreError("Invalid bridge frame size") }
        var length = UInt32(data.count).littleEndian
        var result = withUnsafeBytes(of: &length) { Data($0) }
        result.append(data); return result
    }
    public static func take(from buffer: inout Data) throws -> Data? {
        guard buffer.count >= 4 else { return nil }
        let size = buffer.prefix(4).enumerated().reduce(0) { $0 | Int($1.element) << ($1.offset * 8) }
        guard size > 0, size <= maximum else { throw StoreError("Invalid bridge frame size") }
        guard buffer.count >= size + 4 else { return nil }
        let result = Data(buffer.dropFirst(4).prefix(size)); buffer.removeFirst(size + 4)
        return result
    }
}
