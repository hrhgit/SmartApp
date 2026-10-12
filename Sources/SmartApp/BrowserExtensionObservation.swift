import Foundation
import IntentCore

struct BrowserExtensionObservation {
    let observation: DestinationObservation
    init(message: BrowserBridgeMessage, identity: TargetIdentity, bundle: String, pid: Int32,
         epoch: String, client: String, excluded: (String?) -> Bool) {
        let snapshot = BrowserSnapshot(privateFront: false, windows: message.snapshot.windows.map {
            BrowserWindow(id: $0.id, front: $0.front, activeID: $0.activeID,
                tabs: $0.tabs.map { BrowserTab(id: $0.id, url: $0.url) })
        })
        let observation = snapshot.resolve(identity: identity, bundle: bundle, pid: pid,
            epoch: epoch + ":" + client, entranceKind: "browser_extension_tab", excluded: excluded)
        self.observation = observation
    }
}
