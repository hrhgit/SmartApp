import CryptoKit
import Foundation

/// A concrete location that the user has actually exposed by opening a file.
/// `locator` remains local because reopening needs the complete `file://` URL.
public struct FileDestination: Codable, Equatable, Identifiable, Sendable {
    public enum Kind: String, Codable, Sendable { case file, directory }

    public let id: String
    public let kind: Kind
    public let name: String
    public let locator: String
    public let parentDirectoryID: String?
    public let parentDirectoryLocator: String?
    public let fileType: String?

    public init(
        id: String,
        kind: Kind,
        name: String,
        locator: String,
        parentDirectoryID: String? = nil,
        parentDirectoryLocator: String? = nil,
        fileType: String? = nil
    ) {
        self.id = id
        self.kind = kind
        self.name = name
        self.locator = locator
        self.parentDirectoryID = parentDirectoryID
        self.parentDirectoryLocator = parentDirectoryLocator
        self.fileType = fileType
    }

    public var url: URL? { URL(string: locator) }
}

public enum FileAccessKind: String, Codable, Sendable { case openedFile, observedFile, observedDirectory }

/// The raw history intentionally contains only confirmed foreground file openings.
/// A parent directory is carried as the file's reopenable location, not as a
/// separately collected folder inventory.
public struct FileAccessEvent: Codable, Equatable, Identifiable, Sendable {
    public let id: String
    public let kind: FileAccessKind
    public let visitID: String?
    public let observedAtMs: Int64
    public let source: String
    public let applicationBundleID: String?
    public let target: FileDestination

    public init(
        id: String = UUID().uuidString,
        kind: FileAccessKind = .observedFile,
        visitID: String? = nil,
        observedAtMs: Int64 = Int64(Date().timeIntervalSince1970 * 1_000),
        source: String,
        applicationBundleID: String?,
        target: FileDestination
    ) {
        self.id = id
        self.kind = kind
        self.visitID = visitID
        self.observedAtMs = observedAtMs
        self.source = source
        self.applicationBundleID = applicationBundleID
        self.target = target
    }
}

public enum FileIdentityError: LocalizedError, Sendable {
    case invalidKey

    public var errorDescription: String? {
        switch self {
        case .invalidKey: "FileRecall identity key must be 32 bytes."
        }
    }
}

/// Matches SmartApp's local HMAC identity pattern while limiting the namespace
/// to local file and parent-directory locations.
public struct FileIdentity: Sendable {
    private let key: Data

    public init(key: Data) throws {
        guard key.count == 32 else { throw FileIdentityError.invalidKey }
        self.key = key
    }

    public init(keyURL: URL) throws {
        let value: Data
        if FileManager.default.fileExists(atPath: keyURL.path) {
            value = try Data(contentsOf: keyURL)
        } else {
            value = SymmetricKey(size: .bits256).withUnsafeBytes { Data($0) }
            try FileManager.default.createDirectory(
                at: keyURL.deletingLastPathComponent(),
                withIntermediateDirectories: true,
                attributes: [.posixPermissions: 0o700]
            )
            try value.write(to: keyURL, options: .atomic)
            try FileManager.default.setAttributes([.posixPermissions: 0o600], ofItemAtPath: keyURL.path)
        }
        try self.init(key: value)
    }

    public func file(at rawURL: URL) -> FileDestination? {
        guard FileObservationPolicy.accepts(rawURL) else { return nil }
        let location = rawURL.standardizedFileURL
        let name = location.lastPathComponent
        guard !name.isEmpty, name != "/" else { return nil }
        let parent = location.deletingLastPathComponent().standardizedFileURL
        let parentID = id(namespace: "directory", value: parent.path)
        return FileDestination(
            id: id(namespace: "file", value: location.path),
            kind: .file,
            name: name,
            locator: location.absoluteString,
            parentDirectoryID: parentID,
            parentDirectoryLocator: parent.absoluteString,
            fileType: location.pathExtension.isEmpty ? nil : location.pathExtension.lowercased()
        )
    }

    public func directory(at rawURL: URL) -> FileDestination? {
        guard rawURL.isFileURL else { return nil }
        let location = rawURL.standardizedFileURL
        let name = location.path == "/" ? "/" : location.lastPathComponent
        return FileDestination(
            id: id(namespace: "directory", value: location.path),
            kind: .directory,
            name: name,
            locator: location.absoluteString
        )
    }

    private func id(namespace: String, value: String) -> String {
        let digest = HMAC<SHA256>.authenticationCode(
            for: Data((namespace + "\n" + value).utf8),
            using: SymmetricKey(data: key)
        )
        return namespace + ":" + digest.map { String(format: "%02x", $0) }.joined()
    }
}

/// The collector only accepts a direct file URL from the currently foreground
/// application's focused document attribute. It never expands a directory.
public enum FileObservationPolicy {
    public static let version = "foreground-document-file-v1"

    public static func accepts(_ url: URL) -> Bool {
        url.isFileURL && url.standardizedFileURL.path != "/"
    }
}
