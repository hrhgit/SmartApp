import XCTest
import IntentCore
@testable import SmartApp

final class BrowserExtensionTests: XCTestCase {
    @MainActor func testExplicitPageInspectionDoesNotEmitCollectionAndCanBeCancelled() async throws {
        for mode in ["inspection", "cancel-inspection"] {
            let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
            let bridge = BrowserExtensionBridge(directory: root)
            defer { bridge.stop(); try? FileManager.default.removeItem(at: root) }
            var arrivals = 0
            bridge.onSnapshot = { _, _, _, _ in arrivals += 1 }
            bridge.start()
            let endpoint = root.appendingPathComponent("runtime/browser-bridge-v1/endpoint.json")
            for _ in 0..<100 where !FileManager.default.fileExists(atPath: endpoint.path) { try await Task.sleep(for: .milliseconds(20)) }
            let script = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent()
                .appendingPathComponent("BrowserExtensionTests/bridge-client.mjs").path
            let peer = Task.detached { () throws -> Int32 in
                let process = Process(); process.executableURL = URL(fileURLWithPath: "/usr/bin/env")
                process.arguments = ["node", script, endpoint.path, mode]
                try process.run(); process.waitUntilExit(); return process.terminationStatus
            }
            for _ in 0..<100 where !bridge.supportsForegroundInspection("com.google.Chrome") { try await Task.sleep(for: .milliseconds(20)) }
            XCTAssertTrue(bridge.supportsForegroundInspection("com.google.Chrome"))
            if mode == "inspection" {
                let page = try await bridge.inspectForeground(bundle: "com.google.Chrome")
                XCTAssertEqual(page?.snapshot.windows.first?.tabs.first?.url, "https://example.com/inspection")
            } else {
                let lookup = Task { try await bridge.inspectForeground(bundle: "com.google.Chrome") }
                try await Task.sleep(for: .milliseconds(50)); lookup.cancel()
                do { _ = try await lookup.value; XCTFail("Cancelled lookup must fail") }
                catch { XCTAssertTrue(error is CancellationError) }
                bridge.stop()
            }
            let peerExit = try await peer.value
            XCTAssertEqual(peerExit, 0)
            XCTAssertEqual(arrivals, 0)
        }
    }
    @MainActor func testAuthenticatedTransportAndReplayRejection() async throws {
        let root=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        let bridge=BrowserExtensionBridge(directory:root)
        defer { bridge.stop(); try? FileManager.default.removeItem(at:root) }
        var count=0, disconnects=0
        bridge.onSnapshot={ _,_,message,_ in XCTAssertTrue(message.isValid); count += 1 }
        bridge.onDisconnect={ _,_ in disconnects += 1 }
        bridge.start(); bridge.setCapturing(true); bridge.setForegroundBrowser("com.google.Chrome")
        let endpoint=root.appendingPathComponent("runtime/browser-bridge-v1/endpoint.json")
        for _ in 0..<100 where !FileManager.default.fileExists(atPath:endpoint.path) { try await Task.sleep(for:.milliseconds(20)) }
        XCTAssertTrue(FileManager.default.fileExists(atPath:endpoint.path))
        let script=URL(fileURLWithPath:#filePath).deletingLastPathComponent().deletingLastPathComponent()
            .appendingPathComponent("BrowserExtensionTests/bridge-client.mjs").path
        for mode in ["bad","good"] {
            let code=try await Task.detached { () -> Int32 in
                let process=Process(); process.executableURL=URL(fileURLWithPath:"/usr/bin/env")
                process.arguments=["node",script,endpoint.path,mode]
                try process.run(); process.waitUntilExit(); return process.terminationStatus
            }.value
            XCTAssertEqual(code,0)
            if mode == "bad" { XCTAssertEqual(count,0) }
        }
        XCTAssertEqual(count,1); XCTAssertEqual(disconnects,1)
    }
    func message(reason: String = "initial", active: String = "1", url: String = "https://example.com/a?q=full#fragment") throws -> BrowserBridgeMessage {
        let object: [String: Any] = ["version":2,"sequence":1,"capturedAt":Date().timeIntervalSince1970 * 1000,
            "reason":reason,"snapshot":["windows":[["id":"9","front":true,"minimized":false,"activeID":active,
            "tabs":[["id":active,"url":url,"index":0,"pinned":false]]]]]]
        return try JSONDecoder().decode(BrowserBridgeMessage.self, from: JSONSerialization.data(withJSONObject: object))
    }
    func testWirePartialFramesBatchingAndOversize() throws {
        let data = Data("fixture".utf8), frame = try BrowserBridgeWire.frame(data)
        var buffer = Data(frame.prefix(5)); XCTAssertNil(try BrowserBridgeWire.take(from: &buffer))
        buffer.append(frame.dropFirst(5)); buffer.append(frame)
        XCTAssertEqual(try BrowserBridgeWire.take(from: &buffer), data)
        XCTAssertEqual(try BrowserBridgeWire.take(from: &buffer), data); XCTAssertTrue(buffer.isEmpty)
        buffer = Data([255,255,255,127]); XCTAssertThrowsError(try BrowserBridgeWire.take(from: &buffer))
    }
    func testValidationAndProfileIdentity() throws {
        let m = try message(); XCTAssertTrue(m.isValid)
        let identity = TargetIdentity(key: Data(repeating: 8,count:32))
        let a = BrowserExtensionObservation(message:m,identity:identity,bundle:"com.google.Chrome",pid:7,epoch:"s",client:"a",excluded:{ $0 == "https://excluded.example" })
        let b = BrowserExtensionObservation(message:m,identity:identity,bundle:"com.google.Chrome",pid:7,epoch:"s",client:"b",excluded:{ _ in false })
        XCTAssertEqual(a.observation.entrances.count,1)
        XCTAssertNotEqual(a.observation.foreground?.id,b.observation.foreground?.id)
        XCTAssertEqual(a.observation.targets.filter { $0.kind == .page }.count,1)
        XCTAssertFalse(String(describing:a.observation).contains("background"))
    }
    @MainActor func testForegroundExtensionPersistsNoBackgroundInventory() throws {
        let root=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        let suite="smartapp.extension-test."+UUID().uuidString
        let defaults=UserDefaults(suiteName:suite)!
        defer { defaults.removePersistentDomain(forName:suite); try? FileManager.default.removeItem(at:root) }
        let recorder=RecorderModel(dataDirectory:root,defaults:defaults,liveObservation:false)
        let identity=try TargetIdentity(keyURL:root.appendingPathComponent("identity-v2.key"))
        recorder.start()
        let bundle="com.google.Chrome",source="browser-extension:com.google.Chrome:fixture"
        XCTAssertTrue(recorder.upsert(Destination(id:"app:"+bundle,kind:.app,name:"Chrome",bundleID:bundle,admission:TargetAdmission(basis:"standalone_application"))))
        func accept(_ m:BrowserBridgeMessage,_ foreground:Bool,_ time:Double) {
            let r=BrowserExtensionObservation(message:m,identity:identity,bundle:bundle,pid:7,epoch:"s",client:"fixture",excluded:{$0 == "https://excluded.example"})
            recorder.persistExtension(r,message:m,source:source,foreground:foreground,fallbackID:"app:"+bundle,started:time)
        }
        accept(try message(),false,1)
        XCTAssertFalse(recorder.recent.contains{$0.kind == .contextChanged})
        accept(try message(reason:"window_focus"),true,2)
        accept(try message(reason:"tab_updated"),true,2.1)
        accept(try message(reason:"settled"),true,3.3)
        accept(try message(reason:"settled"),true,3.4)
        accept(try message(reason:"tab_activated",active:"2",url:"https://example.com/next"),true,4)
        XCTAssertEqual(recorder.recent.filter{$0.kind == .contextChanged}.count,1)
        XCTAssertFalse(recorder.recent.contains{$0.kind == .inventoryChanged})
        recorder.shutdown()
        let store=try EventStore(url:root.appendingPathComponent("context-v2.sqlite"),readOnly:true)
        let events=try store.recent(limit:200)
        XCTAssertEqual(events.filter{$0.kind == .contextChanged}.count,2)
        XCTAssertEqual(events.filter{$0.kind == .destinationSettled}.count,1)
        XCTAssertFalse(events.contains{$0.kind == .inventoryChanged && $0.source == source})
        XCTAssertFalse(String(decoding:try JSONEncoder().encode(events),as:UTF8.self).contains("excluded.example"))
        let export=root.appendingPathComponent("export")
        XCTAssertEqual(try store.export(to:export).samples,2)
        let samples=try String(contentsOf:export.appendingPathComponent("navigation-samples.jsonl"),encoding:.utf8)
            .split(separator:"\n").map{try JSONDecoder().decode(NavigationSample.self,from:Data($0.utf8))}
        for sample in samples {
            XCTAssertEqual(sample.schemaVersion,4)
            XCTAssertEqual(sample.inputCoverage?.status,"foreground_only")
            XCTAssertTrue(sample.inventoryEventIDs.isEmpty)
        }
    }
}
