import AppKit
import IntentCore

struct BrowserTab: Decodable, Sendable { let id: String; let url: String }
struct BrowserWindow: Decodable, Sendable {
    let id: String
    let front: Bool
    let activeID: String
    let tabs: [BrowserTab]
}
struct BrowserSnapshot: Decodable, Sendable {
    let privateFront: Bool
    let windows: [BrowserWindow]
}
struct BrowserResult: Sendable {
    let snapshot: BrowserSnapshot?
    let status: String
}

/// A bounded helper process keeps Apple-event delays off the UI thread.
enum BrowserReader {
    static let supported: Set<String> = ["com.google.Chrome", "com.microsoft.edgemac", "com.brave.Browser"]
    static func read(bundleID: String) -> BrowserResult {
        guard supported.contains(bundleID) else { return BrowserResult(snapshot: nil, status: "unsupported") }
        let script = """
        const app = Application("\(bundleID)");
        const result = {privateFront:false, windows:[]};
        if (app.running()) {
          // Compatibility revalidation only: inspect one current window and its
          // active tab, never enumerate a browser's windows or tab inventory.
          const w = app.windows()[0];
          if (w && w.visible() && !w.minimized()) {
            if (w.mode() !== "normal") result.privateFront = true;
            else {
              const tab = w.activeTab();
              result.windows.push({id:String(w.id()), front:true, activeID:String(tab.id()),
                tabs:[{id:String(tab.id()), url:tab.url()}]});
            }
          }
        }
        JSON.stringify(result);
        """
        let result = AutomationScript.run(script)
        guard result.code == 0, let data = result.output.data(using: .utf8),
              let snapshot = try? JSONDecoder().decode(BrowserSnapshot.self, from: data) else {
            return BrowserResult(snapshot: nil, status: result.code == -1743 ? "automation_denied" : "unavailable")
        }
        return BrowserResult(snapshot: snapshot, status: "ready")
    }

    static func activate(bundleID: String, windowID: String, tabID: String, expectedURL: String) -> Bool {
        guard supported.contains(bundleID), let window = Int(windowID), let tab = Int(tabID) else { return false }
        let script = """
        const app = Application("\(bundleID)");
        let found = false;
        if (app.running()) {
          app.windows().forEach(function(w) {
            if (Number(w.id()) !== \(window) || w.mode() !== "normal") return;
            w.tabs().forEach(function(t,i) {
              if (Number(t.id()) === \(tab) && t.url() === \(AutomationScript.literal(expectedURL))) {
                w.activeTabIndex = i+1; w.index = 1; found = true;
              }
            });
          });
          if (found) app.activate();
        }
        String(found);
        """
        let result = AutomationScript.run(script)
        return result.code == 0 && result.output.trimmingCharacters(in: .whitespacesAndNewlines) == "true"
    }

}
