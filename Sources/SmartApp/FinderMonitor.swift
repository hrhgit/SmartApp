import Foundation
import AppKit
import IntentCore

struct FinderWindow: Decodable, Sendable {
    let id: String
    let front: Bool
    let url: String?
}
struct FinderSnapshot: Decodable, Sendable {
    let windows: [FinderWindow]

    func resolve(identity: TargetIdentity, pid: Int32, epoch: String) -> DestinationObservation {
        var result = DestinationObservation()
        guard let window = windows.first(where: \.front) else { return result }
        let windowID = "finder-window:\(epoch):\(pid):\(window.id)"
        let container = Destination(id: windowID, kind: .window, name: "Finder · 窗口 \(window.id)",
            bundleID: FinderReader.bundleID, admission: TargetAdmission(basis: "supported_finder_window"))
        result.targets.append(container)
        var targetID = windowID
        if let raw = window.url, let url = URL(string: raw), url.isFileURL,
           let values = try? url.resourceValues(forKeys: [.isDirectoryKey]), values.isDirectory == true {
            var directory = identity.local(url, directory: true)
            directory.bundleID = FinderReader.bundleID
            directory.admission = TargetEligibilityPolicy.finderDirectory(validFileURL: true, isDirectory: true,
                owner: .init(.allowed, "standalone_application")).admission
            result.targets.append(directory); targetID = directory.id
        }
        let entrance = TargetEntrance(id: windowID, targetID: targetID, bundleID: FinderReader.bundleID,
            windowID: window.id, kind: "finder_window", containerID: windowID)
        result.entrances.append(entrance); result.foreground = entrance
        return result
    }
}
struct FinderResult: Sendable {
    let snapshot: FinderSnapshot?
    let status: String
}

enum FinderReader {
    static let bundleID = "com.apple.finder"

    static let observationScript = """
        const app = Application("com.apple.finder");
        const result = {windows:[]};
        if (app.running()) {
          const w = app.finderWindows()[0];
          if (w && w.visible() && !w.collapsed()) {
            let url = null;
            try { url = w.target().url(); } catch(e) { }
            result.windows.push({id:String(w.id()), front:true, url:url || null});
          }
        }
        JSON.stringify(result);
        """

    static func read(expectedPID: Int32) -> FinderResult {
        guard NSWorkspace.shared.frontmostApplication?.processIdentifier == expectedPID else {
            return FinderResult(snapshot: nil, status: "stale")
        }
        if let native = DirectoryNativeTransport.read(source: "finder", pid: expectedPID) {
            if native.status == "stale" { return FinderResult(snapshot: nil, status: "stale") }
            if native.status == "ready", let data = native.snapshot,
               let snapshot = try? JSONDecoder().decode(FinderSnapshot.self, from: data) {
                AutomationScript.closeScriptDirectorySession()
                return FinderResult(snapshot: snapshot, status: "ready")
            }
        }
        guard NSWorkspace.shared.frontmostApplication?.processIdentifier == expectedPID else {
            return FinderResult(snapshot: nil, status: "stale")
        }
        let result = AutomationScript.readDirectory(observationScript)
        guard result.code == 0, let data = result.output.data(using: .utf8),
              let snapshot = try? JSONDecoder().decode(FinderSnapshot.self, from: data) else {
            return FinderResult(snapshot: nil, status: result.code == -1743 ? "automation_denied" : "unavailable")
        }
        return FinderResult(snapshot: snapshot, status: "ready")
    }

    static func activate(windowID: String, expectedURL: String?) -> Bool {
        guard let window = Int(windowID) else { return false }
        let expected = expectedURL.map(AutomationScript.literal) ?? "null"
        let script = """
        const app = Application("com.apple.finder");
        const expected = \(expected);
        let found = false;
        if (app.running()) {
          app.finderWindows().forEach(function(w) {
            if (Number(w.id()) !== \(window)) return;
            if (expected !== null) {
              let actual = null;
              try { actual = w.target().url(); } catch(e) { return; }
              if (actual !== expected) return;
            }
            w.collapsed = false; w.index = 1; found = true;
          });
          if (found) app.activate();
        }
        String(found);
        """
        let result = AutomationScript.run(script)
        return result.code == 0 && result.output.trimmingCharacters(in: .whitespacesAndNewlines) == "true"
    }
}
