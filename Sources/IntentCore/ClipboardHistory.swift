import Foundation
import CSQLite

public enum ClipboardCaptureOrigin: String, Codable, Sendable {
    case observedChange
    case existingTimeUnknown
}

public enum ClipboardContentKind: String, Codable, Sendable {
    case text, link, image, files, richContent
}

public struct ClipboardRepresentation: Codable, Equatable, Sendable {
    public let type: String
    public let data: Data
    public init(type: String, data: Data) { self.type = type; self.data = data }
}

/// The original representations are kept for replay; no file contents are opened.
public struct ClipboardItemRecord: Codable, Equatable, Sendable {
    public let representations: [ClipboardRepresentation]
    public let text: String?
    public let fileLocator: String?
    public init(representations: [ClipboardRepresentation], text: String? = nil, fileLocator: String? = nil) {
        self.representations = representations; self.text = text; self.fileLocator = fileLocator
    }
}

public struct ClipboardEntry: Codable, Equatable, Identifiable, Sendable {
    public let id: String
    public let observedAt: Date
    public let origin: ClipboardCaptureOrigin
    public let items: [ClipboardItemRecord]
    public var isPinned: Bool

    public init(id: String = UUID().uuidString, observedAt: Date = Date(),
                origin: ClipboardCaptureOrigin, items: [ClipboardItemRecord], isPinned: Bool = false) {
        self.id = id; self.observedAt = observedAt; self.origin = origin
        self.items = items; self.isPinned = isPinned
    }

    public var kind: ClipboardContentKind {
        if items.contains(where: { $0.fileLocator != nil }) { return .files }
        if items.contains(where: { $0.representations.contains { ["public.png", "public.tiff", "public.jpeg"].contains($0.type) } }) {
            return .image
        }
        if let text, let url = URL(string: text.trimmingCharacters(in: .whitespacesAndNewlines)),
           ["http", "https"].contains(url.scheme?.lowercased() ?? ""), url.host != nil { return .link }
        return text == nil ? .richContent : .text
    }

    public var text: String? {
        let values = items.compactMap(\.text).filter { !$0.isEmpty }
        return values.isEmpty ? nil : values.joined(separator: "\n")
    }
    public var fileLocators: [String] { items.compactMap(\.fileLocator) }
    public var title: String {
        if kind == .files {
            return fileLocators.compactMap { URL(string: $0)?.lastPathComponent }.joined(separator: ", ")
        }
        if let text { return text.components(separatedBy: .newlines).first(where: { !$0.isEmpty }) ?? text }
        return kind == .image ? "图片" : "富文本或其他内容"
    }
    public var searchText: String {
        [text ?? "", fileLocators.joined(separator: "\n")].joined(separator: "\n")
    }
    public var byteCount: Int { items.reduce(0) { $0 + $1.representations.reduce(0) { $0 + $1.data.count } } }
}

public struct ClipboardSummary: Equatable, Identifiable, Sendable {
    public let id: String
    public let observedAt: Date
    public let origin: ClipboardCaptureOrigin
    public let kind: ClipboardContentKind
    public let title: String
    public let isPinned: Bool
    public let byteCount: Int
}

/// This evidence exists only during an uninterrupted observation period.
/// It records recency, never whether someone copied or successfully pasted.
public struct ClipboardRecommendationContext: Equatable, Sendable {
    public let entry: ClipboardEntry
    public let explicitlySelected: Bool
    public let requestAt: Date
    public init(entry: ClipboardEntry, explicitlySelected: Bool, requestAt: Date = Date()) {
        self.entry = entry; self.explicitlySelected = explicitlySelected; self.requestAt = requestAt
    }
    public var ageSeconds: TimeInterval { max(0, requestAt.timeIntervalSince(entry.observedAt)) }
    public var modelEvidence: String? {
        var parts: [String] = []
        if let text = entry.text { parts.append(text) }
        if !entry.fileLocators.isEmpty { parts.append(entry.fileLocators.joined(separator: "\n")) }
        guard !parts.isEmpty else { return nil }
        let source = explicitlySelected ? "The user explicitly selected this history entry for this request."
            : "An update was observed \(Int(ageSeconds)) seconds before this request. This does not prove it is still relevant or unused."
        let time = entry.origin == .existingTimeUnknown ? "Original clipboard update time is unknown." : "Time refers to observation, not an exact Copy event."
        return "\(source) \(time)\n\(parts.joined(separator: "\n"))"
    }
}

public struct ClipboardObservationTracker: Sendable {
    public private(set) var version: Int?
    public private(set) var latest: ClipboardEntry?
    public init() {}
    public mutating func begin(version: Int) { self.version = version; latest = nil }
    public mutating func reset() { version = nil; latest = nil }
    public mutating func changed(to version: Int) -> Bool {
        guard self.version != version else { return false }
        self.version = version; latest = nil; return true
    }
    public mutating func accepted(_ entry: ClipboardEntry, version: Int) {
        guard self.version == version, entry.origin == .observedChange else { return }
        latest = entry
    }
    public mutating func remove(id: String) { if latest?.id == id { latest = nil } }
    public func automaticContext(currentVersion: Int, at: Date = Date()) -> ClipboardRecommendationContext? {
        guard version == currentVersion, let latest else { return nil }
        return .init(entry: latest, explicitlySelected: false, requestAt: at)
    }
}

/// Separate from EventStore: contents cannot enter foreground exports or model replay.
public final class ClipboardHistoryStore: @unchecked Sendable {
    private let queue = DispatchQueue(label: "SmartApp.clipboard-history")
    private var db: OpaquePointer?
    public let url: URL
    private let transient = unsafeBitCast(-1, to: sqlite3_destructor_type.self)

    public init(url: URL) throws {
        self.url = url
        try FileManager.default.createDirectory(at: url.deletingLastPathComponent(), withIntermediateDirectories: true,
                                                attributes: [.posixPermissions: 0o700])
        guard sqlite3_open_v2(url.path, &db, SQLITE_OPEN_CREATE | SQLITE_OPEN_READWRITE | SQLITE_OPEN_FULLMUTEX, nil) == SQLITE_OK else {
            if let db { sqlite3_close(db) }; db = nil
            throw StoreError("无法打开剪贴板历史库。")
        }
        do {
            sqlite3_busy_timeout(db, 5_000)
            try execute("PRAGMA journal_mode=WAL")
            try execute("PRAGMA secure_delete=ON")
            try execute("CREATE TABLE IF NOT EXISTS clipboard_entries (seq INTEGER PRIMARY KEY AUTOINCREMENT, id TEXT UNIQUE NOT NULL, observed_at REAL NOT NULL, origin TEXT NOT NULL, kind TEXT NOT NULL, title TEXT NOT NULL, search_text TEXT NOT NULL, pinned INTEGER NOT NULL DEFAULT 0, byte_count INTEGER NOT NULL, payload BLOB NOT NULL)")
            try execute("CREATE INDEX IF NOT EXISTS clipboard_order ON clipboard_entries(pinned DESC, seq DESC)")
            try FileManager.default.setAttributes([.posixPermissions: 0o600], ofItemAtPath: url.path)
        } catch { sqlite3_close(db); db = nil; throw error }
    }
    deinit { if let db { sqlite3_close(db) } }

    public func insert(_ entry: ClipboardEntry) throws {
        let data = try JSONEncoder().encode(entry)
        try queue.sync {
            let sql = try prepare("INSERT INTO clipboard_entries(id,observed_at,origin,kind,title,search_text,pinned,byte_count,payload) VALUES(?,?,?,?,?,?,?,?,?)")
            defer { sqlite3_finalize(sql) }
            bind(entry.id, to: sql, at: 1); sqlite3_bind_double(sql, 2, entry.observedAt.timeIntervalSince1970)
            bind(entry.origin.rawValue, to: sql, at: 3); bind(entry.kind.rawValue, to: sql, at: 4)
            bind(entry.title, to: sql, at: 5); bind(entry.searchText, to: sql, at: 6)
            sqlite3_bind_int(sql, 7, entry.isPinned ? 1 : 0); sqlite3_bind_int64(sql, 8, Int64(entry.byteCount))
            let bound = data.withUnsafeBytes { sqlite3_bind_blob64(sql, 9, $0.baseAddress, UInt64($0.count), transient) }
            guard bound == SQLITE_OK else { throw failure() }
            _ = try step(sql)
        }
    }
    public func summaries(search: String = "", pinnedOnly: Bool = false, offset: Int = 0, limit: Int = 100) throws -> [ClipboardSummary] {
        try queue.sync {
            let sql = try prepare("SELECT id,observed_at,origin,kind,title,pinned,byte_count FROM clipboard_entries WHERE (?=0 OR pinned=1) AND (?='' OR search_text LIKE ? ESCAPE '\\') ORDER BY pinned DESC,seq DESC LIMIT ? OFFSET ?")
            defer { sqlite3_finalize(sql) }
            sqlite3_bind_int(sql, 1, pinnedOnly ? 1 : 0); bind(search, to: sql, at: 2)
            let escaped = search.replacingOccurrences(of: "\\", with: "\\\\").replacingOccurrences(of: "%", with: "\\%").replacingOccurrences(of: "_", with: "\\_")
            bind("%\(escaped)%", to: sql, at: 3)
            sqlite3_bind_int64(sql, 4, Int64(max(1, limit))); sqlite3_bind_int64(sql, 5, Int64(max(0, offset)))
            var result: [ClipboardSummary] = []
            while try step(sql) == SQLITE_ROW {
                guard let origin = ClipboardCaptureOrigin(rawValue: string(sql, 2)),
                      let kind = ClipboardContentKind(rawValue: string(sql, 3)) else { throw StoreError("剪贴板历史格式不兼容。") }
                result.append(.init(id: string(sql, 0), observedAt: Date(timeIntervalSince1970: sqlite3_column_double(sql, 1)),
                    origin: origin, kind: kind, title: string(sql, 4), isPinned: sqlite3_column_int(sql, 5) != 0,
                    byteCount: Int(sqlite3_column_int64(sql, 6))))
            }
            return result
        }
    }
    public func entry(id: String) throws -> ClipboardEntry? {
        try queue.sync {
            let sql = try prepare("SELECT payload,pinned FROM clipboard_entries WHERE id=?")
            defer { sqlite3_finalize(sql) }; bind(id, to: sql, at: 1)
            guard try step(sql) == SQLITE_ROW, let bytes = sqlite3_column_blob(sql, 0) else { return nil }
            var entry = try JSONDecoder().decode(ClipboardEntry.self, from: Data(bytes: bytes, count: Int(sqlite3_column_bytes(sql, 0))))
            entry.isPinned = sqlite3_column_int(sql, 1) != 0
            return entry
        }
    }
    public func count() throws -> Int {
        try queue.sync {
            let sql = try prepare("SELECT COUNT(*) FROM clipboard_entries")
            defer { sqlite3_finalize(sql) }; _ = try step(sql); return Int(sqlite3_column_int64(sql, 0))
        }
    }
    public func setPinned(id: String, pinned: Bool) throws {
        try queue.sync {
            let sql = try prepare("UPDATE clipboard_entries SET pinned=? WHERE id=?")
            defer { sqlite3_finalize(sql) }; sqlite3_bind_int(sql, 1, pinned ? 1 : 0); bind(id, to: sql, at: 2); _ = try step(sql)
        }
    }
    public func delete(id: String) throws {
        try queue.sync {
            let sql = try prepare("DELETE FROM clipboard_entries WHERE id=?")
            defer { sqlite3_finalize(sql) }; bind(id, to: sql, at: 1); _ = try step(sql)
            try checkpointAfterRemoval()
        }
    }
    public func clear() throws { try queue.sync { try execute("DELETE FROM clipboard_entries"); try checkpointAfterRemoval() } }
    public func flush() { queue.sync { _ = sqlite3_wal_checkpoint_v2(db, nil, SQLITE_CHECKPOINT_PASSIVE, nil, nil) } }

    private func execute(_ sql: String) throws {
        guard sqlite3_exec(db, sql, nil, nil, nil) == SQLITE_OK else { throw failure() }
    }
    private func prepare(_ sql: String) throws -> OpaquePointer {
        var statement: OpaquePointer?
        guard sqlite3_prepare_v2(db, sql, -1, &statement, nil) == SQLITE_OK, let statement else { throw failure() }
        return statement
    }
    private func step(_ statement: OpaquePointer) throws -> Int32 {
        let code = sqlite3_step(statement)
        guard code == SQLITE_ROW || code == SQLITE_DONE else { throw failure() }
        return code
    }
    private func bind(_ value: String, to statement: OpaquePointer, at index: Int32) {
        value.utf8CString.withUnsafeBufferPointer { buffer in
            _ = sqlite3_bind_text64(statement, index, buffer.baseAddress, UInt64(buffer.count - 1), transient, UInt8(SQLITE_UTF8))
        }
    }
    private func string(_ statement: OpaquePointer, _ index: Int32) -> String {
        guard let bytes = sqlite3_column_text(statement, index) else { return "" }
        return String(decoding: UnsafeBufferPointer(start: bytes, count: Int(sqlite3_column_bytes(statement, index))), as: UTF8.self)
    }
    private func checkpointAfterRemoval() throws {
        guard sqlite3_wal_checkpoint_v2(db, nil, SQLITE_CHECKPOINT_TRUNCATE, nil, nil) == SQLITE_OK else { throw failure() }
    }
    private func failure() -> StoreError { StoreError("剪贴板历史保存失败：\(db.map { String(cString: sqlite3_errmsg($0)) } ?? "数据库已关闭")") }
}
