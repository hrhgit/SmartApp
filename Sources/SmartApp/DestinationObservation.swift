import Foundation
import IntentCore

/// A collector returns only the current frontmost destination. It never carries
/// background window, tab, pane or candidate inventory.
struct DestinationObservation: Sendable {
    var targets: [Destination] = []
    var entrances: [TargetEntrance] = []
    var foreground: TargetEntrance?
    var excludedFront = false

    func foregroundOnly() -> DestinationObservation {
        guard let foreground else {
            var empty = DestinationObservation(); empty.excludedFront = excludedFront
            return empty
        }
        var result = DestinationObservation()
        result.foreground = foreground; result.excludedFront = excludedFront
        result.entrances = [foreground]
        let IDs = Set([foreground.targetID, foreground.containerID].compactMap { $0 })
        result.targets = targets.filter { IDs.contains($0.id) }
        return result
    }
}

extension BrowserSnapshot {
    func resolve(identity: TargetIdentity, bundle: String, pid: Int32, epoch: String,
                 entranceKind: String = "browser_tab",
                 excluded: (String?) -> Bool) -> DestinationObservation {
        var result = DestinationObservation()
        guard BrowserReader.supported.contains(bundle) else { return result }
        result.excludedFront = privateFront
        guard let window = windows.first(where: \.front) else { return result }
        let windowID = "browser-window:\(epoch):\(pid):\(window.id)"
        result.targets.append(Destination(id: windowID, kind: .window,
            name: "\(bundle) · 浏览器窗口 \(window.id)", bundleID: bundle,
            admission: TargetAdmission(basis: "supported_browser_window")))
        guard let tab = window.tabs.first(where: { $0.id == window.activeID }),
              var page = identity.page(tab.url), !excluded(page.origin), !privateFront else {
            result.excludedFront = true; return result
        }
        page.admission = TargetEligibilityPolicy.page(origin: page.origin, supportedSource: true, excluded: false).admission
        result.targets.append(page)
        let entrance = TargetEntrance(id: "tab:\(epoch):\(pid):\(tab.id)", targetID: page.id,
            bundleID: bundle, windowID: window.id, tabID: tab.id, kind: entranceKind, containerID: windowID)
        result.entrances.append(entrance); result.foreground = entrance
        return result
    }
}
