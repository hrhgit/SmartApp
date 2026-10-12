import Foundation

/// The only information an eventual AI provider may receive from this module.
/// It deliberately has no file contents, directory listings, background app
/// state, keystrokes, or candidate inventory.
public struct FileStateInput: Codable, Equatable, Sendable {
    public let activeApplicationBundleID: String?
    public let recentFiles: [FileDestination]

    public init(activeApplicationBundleID: String?, recentFiles: [FileDestination]) {
        self.activeApplicationBundleID = activeApplicationBundleID
        self.recentFiles = recentFiles
    }
}

public struct FileStateInference: Codable, Equatable, Sendable {
    public let summary: String
    public let preferredDirectoryLocator: String?
    public let confidence: Double
    public let evidenceCount: Int

    public init(summary: String, preferredDirectoryLocator: String?, confidence: Double, evidenceCount: Int) {
        self.summary = summary
        self.preferredDirectoryLocator = preferredDirectoryLocator
        self.confidence = confidence
        self.evidenceCount = evidenceCount
    }
}

/// A future local or user-enabled remote model plugs in here. The return value
/// may rank existing directories but must never introduce a new path.
public protocol FileStateInferring: Sendable {
    func inferState(from input: FileStateInput) -> FileStateInference?
}

/// Privacy-preserving bootstrap used until a real AI provider is deliberately
/// configured. It identifies the most recurrent recent parent directory from
/// already-recorded file openings; it does not inspect file contents.
public struct LocalPathStateInferrer: FileStateInferring {
    public init() {}

    public func inferState(from input: FileStateInput) -> FileStateInference? {
        let files = input.recentFiles.filter { $0.kind == .file }
        guard !files.isEmpty else { return nil }

        var counts: [String: Int] = [:]
        var names: [String: String] = [:]
        for file in files {
            guard let locator = file.parentDirectoryLocator else { continue }
            counts[locator, default: 0] += 1
            if names[locator] == nil {
                names[locator] = URL(string: locator)?.lastPathComponent ?? locator
            }
        }
        guard let locator = counts.keys.sorted(by: {
            counts[$0, default: 0] == counts[$1, default: 0] ? $0 < $1 : counts[$0, default: 0] > counts[$1, default: 0]
        }).first else { return nil }

        let evidence = counts[locator, default: 0]
        let confidence = min(0.8, 0.25 + Double(evidence) / Double(max(files.count, 1)))
        return FileStateInference(
            summary: "最近正在处理 \(names[locator] ?? "该目录") 中的文件",
            preferredDirectoryLocator: locator,
            confidence: confidence,
            evidenceCount: evidence
        )
    }
}
