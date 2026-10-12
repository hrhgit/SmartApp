import Foundation

public enum CapturePolicy {
    public static let ownBundleID = "local.herh.SmartApp"
    // System authentication/permission panels are never useful prediction targets.
    public static let alwaysExcluded: Set<String> = [
        ownBundleID, "com.apple.UserNotificationCenter", "com.apple.SecurityAgent",
        "com.apple.loginwindow", "com.apple.systemuiserver", "com.apple.notificationcenterui"
    ]

    public static func isExcluded(_ bundleID: String, rules: [String]) -> Bool {
        if alwaysExcluded.contains(bundleID) { return true }
        if ["com.apple.SecurityAgent", "com.apple.SecurityAgentHelper"].contains(where: {
            bundleID == $0 || bundleID.hasPrefix($0 + ".")
        }) { return true }
        return rules.contains { rule in
            if rule.hasSuffix(".*") { return bundleID.hasPrefix(String(rule.dropLast())) }
            return bundleID == rule
        }
    }
}
