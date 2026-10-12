import Foundation
import AppKit
import IntentCore

struct QSpacePane: Decodable, Sendable {
    let id: String
    let active: Bool
    let url: String?
}

struct QSpaceWindow: Decodable, Sendable {
    let id: String
    let front: Bool
    let panes: [QSpacePane]
}

struct QSpaceSnapshot: Decodable, Sendable {
    let windows: [QSpaceWindow]

    func resolve(identity: TargetIdentity, pid: Int32, epoch: String) -> DestinationObservation {
        var result = DestinationObservation()
        guard let window = windows.first(where: \.front), let pane = window.panes.first(where: \.active) else { return result }
        let containerID = "qspace-window:\(epoch):\(pid):\(window.id)"
        result.targets.append(Destination(id: containerID, kind: .window,
            name: "QSpace Pro · 窗口 \(window.id)", bundleID: QSpaceReader.bundleID,
            admission: TargetAdmission(basis: "supported_qspace_window")))
        var targetID = containerID
        if let raw = pane.url, let url = URL(string: raw), url.isFileURL,
           let values = try? url.resourceValues(forKeys: [.isDirectoryKey]), values.isDirectory == true {
            var directory = identity.local(url, directory: true)
            directory.bundleID = QSpaceReader.bundleID
            directory.admission = TargetEligibilityPolicy.qspaceDirectory(validFileURL: true, isDirectory: true,
                owner: .init(.allowed, "standalone_application")).admission
            result.targets.append(directory)
            targetID = directory.id
        }
        let entrance = TargetEntrance(id: "qspace-pane:\(epoch):\(pid):\(window.id):\(pane.id)",
            targetID: targetID, bundleID: QSpaceReader.bundleID, windowID: window.id, paneID: pane.id,
            kind: "qspace_pane", containerID: containerID)
        result.entrances.append(entrance); result.foreground = entrance
        return result
    }
}

struct QSpaceResult: Sendable {
    let snapshot: QSpaceSnapshot?
    let status: String
}

enum QSpaceReader {
    static let bundleID = TargetEligibilityPolicy.qspaceBundleID

    static let observationScript = """
        const app = Application("com.jinghaoshe.qspace.pro");
        const result = {windows:[]};
        if (app.running()) {
          const window = app.windows()[0];
          if (window) {
            let id = null;
            let bounds = null;
            try { id = String(window.id()); bounds = window.bounds(); } catch(e) { }
            if (id && bounds && Number(bounds.width) > 0 && Number(bounds.height) > 0) {
              let pane = null;
              try { pane = window.activatedPane; } catch(e) { }
              if (pane) {
                let paneID = null;
                let url = null;
                try { paneID = String(pane.id()); url = String(pane.rootItem.urlstr()); } catch(e) { }
                if (paneID) result.windows.push({id:id, front:true, panes:[{id:paneID, active:true, url:url || null}]});
              }
            }
          }
        }
        JSON.stringify(result);
        """

    static func read(expectedPID: Int32) -> QSpaceResult {
        guard NSWorkspace.shared.frontmostApplication?.processIdentifier == expectedPID else {
            return QSpaceResult(snapshot: nil, status: "stale")
        }
        if let native = DirectoryNativeTransport.read(source: "qspace", pid: expectedPID) {
            if native.status == "stale" { return QSpaceResult(snapshot: nil, status: "stale") }
            if native.status == "ready", let data = native.snapshot,
               let snapshot = try? JSONDecoder().decode(QSpaceSnapshot.self, from: data) {
                AutomationScript.closeScriptDirectorySession()
                return QSpaceResult(snapshot: snapshot, status: "ready")
            }
        }
        guard NSWorkspace.shared.frontmostApplication?.processIdentifier == expectedPID else {
            return QSpaceResult(snapshot: nil, status: "stale")
        }
        let result = AutomationScript.readDirectory(observationScript)
        guard result.code == 0, let data = result.output.data(using: .utf8),
              let snapshot = try? JSONDecoder().decode(QSpaceSnapshot.self, from: data) else {
            return QSpaceResult(snapshot: nil, status: result.code == -1743 ? "automation_denied" : "unavailable")
        }
        return QSpaceResult(snapshot: snapshot, status: "ready")
    }

    static func activate(windowID: String, paneID: String, expectedURL: String?) -> Bool {
        let expected = expectedURL.map(AutomationScript.literal) ?? "null"
        let script = """
        const app = Application("\(bundleID)");
        const expected = \(expected);
        let found = false;
        if (app.running()) {
          const window = app.windows().find(function(candidate) {
            return String(candidate.id()) === \(AutomationScript.literal(windowID));
          });
          if (window) {
            const pane = window.panes().find(function(candidate) {
              return String(candidate.id()) === \(AutomationScript.literal(paneID));
            });
            if (pane) {
              let actual = null;
              try { actual = String(pane.rootItem.urlstr()); } catch (error) { }
              if (expected === null || actual === expected) {
                window.activate(); pane.activate(); found = true;
              }
            }
          }
        }
        String(found);
        """
        let result = AutomationScript.run(script)
        return result.code == 0 && result.output.trimmingCharacters(in: .whitespacesAndNewlines) == "true"
    }

    static func open(url: URL) -> Bool {
        guard url.isFileURL else { return false }
        let script = """
        tell application id "\(bundleID)"
            open urlstr \(AutomationScript.appleScriptLiteral(url.absoluteString))
            return "true"
        end tell
        """
        let result = AutomationScript.run(script, language: .appleScript)
        return result.code == 0 && result.output.trimmingCharacters(in: .whitespacesAndNewlines) == "true"
    }
}
