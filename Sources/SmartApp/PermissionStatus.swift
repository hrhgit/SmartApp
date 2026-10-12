import AppKit
import ApplicationServices
import IntentCore
import IntentCore

/// A local diagnostic written by the actual app process, so the CLI need not inspect TCC databases
/// or accidentally report the terminal's permissions as SmartApp's permissions.
struct PermissionStatus: Codable, Equatable {
    var targetEligibilityPolicyVersion = TargetEligibilityPolicy.version
    let schemaVersion: Int
    let pid: Int32
    let bundlePath: String
    let inputMonitoring: Bool
    let accessibility: Bool
    let inputTapConnected: Bool
    let collecting: Bool

    static var url: URL {
        EventStore.defaultURL.deletingLastPathComponent().appendingPathComponent("permission-status.json")
    }

    func write() throws {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys, .withoutEscapingSlashes]
        try encoder.encode(self).write(to: Self.url, options: .atomic)
        try FileManager.default.setAttributes([.posixPermissions: 0o600], ofItemAtPath: Self.url.path)
    }
}
