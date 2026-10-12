import AppKit

/// Keep destination execution testable without moving the user's real windows.
struct DestinationActions: Sendable {
    var readBrowser: @Sendable (String) -> BrowserResult = { BrowserReader.read(bundleID: $0) }
    var activateBrowser: @Sendable (String, String, String, String) -> Bool = {
        BrowserReader.activate(bundleID: $0, windowID: $1, tabID: $2, expectedURL: $3)
    }
    var activateFinder: @Sendable (String, String?) -> Bool = {
        FinderReader.activate(windowID: $0, expectedURL: $1)
    }
    var activateQSpace: @Sendable (String, String, String?) -> Bool = {
        QSpaceReader.activate(windowID: $0, paneID: $1, expectedURL: $2)
    }
    var openQSpace: @Sendable (URL) -> Bool = { QSpaceReader.open(url: $0) }
    var openURL: @MainActor @Sendable (URL) -> Bool = { NSWorkspace.shared.open($0) }
}
