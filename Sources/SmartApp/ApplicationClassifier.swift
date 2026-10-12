import AppKit
import IntentCore

@MainActor
enum ApplicationClassifier {
    static func evidence(bundleID: String?, url: URL?, activation: ApplicationEvidence.Activation) -> ApplicationEvidence {
        guard let url = url?.resolvingSymlinksInPath(), let bundle = Bundle(url: url) else {
            return ApplicationEvidence(bundleID: bundleID, standaloneApplication: false, activation: activation)
        }
        let ancestors = url.deletingLastPathComponent().pathComponents
        let component = ancestors.contains { ["app", "xpc", "appex"].contains(URL(fileURLWithPath: $0).pathExtension.lowercased()) }
        let packageType = bundle.object(forInfoDictionaryKey: "CFBundlePackageType") as? String
        // Finder is a standalone GUI application with Apple's special FNDR type.
        let applicationPackage = packageType == "APPL" || (packageType == "FNDR" && bundleID == "com.apple.finder")
        let standalone = url.pathExtension.lowercased() == "app" && !component &&
            bundle.bundleIdentifier == bundleID && applicationPackage &&
            bundle.executableURL.map { FileManager.default.isExecutableFile(atPath: $0.path) } == true
        return ApplicationEvidence(bundleID: bundleID, standaloneApplication: standalone,
            backgroundOnly: (bundle.object(forInfoDictionaryKey: "LSBackgroundOnly") as? NSNumber)?.boolValue == true,
            uiElement: (bundle.object(forInfoDictionaryKey: "LSUIElement") as? NSNumber)?.boolValue == true,
            activation: activation)
    }
    static func evidence(_ app: NSRunningApplication) -> ApplicationEvidence {
        let activation: ApplicationEvidence.Activation
        switch app.activationPolicy {
        case .regular: activation = .regular
        case .accessory: activation = .accessory
        case .prohibited: activation = .prohibited
        @unknown default: activation = .unknown
        }
        return evidence(bundleID: app.bundleIdentifier, url: app.bundleURL, activation: activation)
    }
}
