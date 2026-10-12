import Foundation
import Network
import IntentCore

@MainActor final class BrowserExtensionBridge {
    static let extensionID = "hhlflmdclnimbejikiajjfjeeibjnngh"
    final class Client {
        let connection: NWConnection
        var buffer = Data()
        var bundle: String?
        var sequence = 0
        var requestID: String?
        var synchronized = false
        var queuedBytes = 0
        var supportsForegroundInspection = false
        init(_ connection: NWConnection) { self.connection = connection }
    }
    var onSnapshot: ((String, String, BrowserBridgeMessage, Double) -> Void)?
    var onDisconnect: ((String, String) -> Void)?
    var onError: ((String) -> Void)?
    private var listener: NWListener?
    private var clients: [String: Client] = [:]
    private var pending: [String: CheckedContinuation<Bool, Never>] = [:]
    private var inspections: [String: (client: String, continuation: CheckedContinuation<BrowserBridgeMessage?, Error>)] = [:]
    private var enabled = false
    private var foregroundBundle: String?
    private let token = UUID().uuidString + UUID().uuidString
    private let directory: URL
    init(directory: URL) { self.directory = directory.appendingPathComponent("runtime/browser-bridge-v1") }

    func start() {
        guard listener == nil else { return }
        do {
            try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true, attributes: [.posixPermissions: 0o700])
            let parameters = NWParameters.tcp
            parameters.requiredLocalEndpoint = .hostPort(host: "127.0.0.1", port: .any)
            let listener = try NWListener(using: parameters)
            self.listener = listener
            listener.stateUpdateHandler = { [weak self] state in
                Task { @MainActor in
                    guard let self else { return }
                    switch state {
                    case .ready:
                        guard let port = self.listener?.port else { return }
                        do {
                            let data = try JSONEncoder().encode(["port": JSONValue.number(Double(port.rawValue)),
                                "token": .string(self.token), "origin": .string("chrome-extension://" + Self.extensionID + "/")])
                            let url = self.directory.appendingPathComponent("endpoint.json")
                            try data.write(to: url, options: .atomic)
                            try FileManager.default.setAttributes([.posixPermissions: 0o600], ofItemAtPath: url.path)
                        } catch { self.onError?("无法写入浏览器连接配置") }
                    case .failed: self.onError?("浏览器本地连接启动失败")
                    default: break
                    }
                }
            }
            listener.newConnectionHandler = { [weak self] connection in
                Task { @MainActor in self?.attach(connection) }
            }
            listener.start(queue: .main)
        } catch { onError?("浏览器本地连接启动失败") }
    }
    func stop() {
        for id in Array(clients.keys) { disconnect(id) }
        listener?.cancel(); listener = nil
        try? FileManager.default.removeItem(at: directory.appendingPathComponent("endpoint.json"))
    }
    func setCapturing(_ value: Bool) {
        enabled = value
        for id in clients.keys.sorted() { sendCapture(id) }
    }
    /// The extension must not obtain any tab/window data while its browser is not
    /// the macOS foreground application.
    func setForegroundBrowser(_ bundle: String?) {
        guard foregroundBundle != bundle else { return }
        foregroundBundle = bundle
        for id in clients.keys.sorted() { sendCapture(id) }
    }
    func resync(bundle: String? = nil) {
        guard enabled else { return }
        let requested = bundle ?? foregroundBundle
        for (id, client) in clients where client.bundle != nil && client.bundle == requested && client.bundle == foregroundBundle {
            client.synchronized = false
            let request = UUID().uuidString; client.requestID = request
            send(id, ["type": .string("resync"), "requestID": .string(request)])
        }
    }
    func hasClient(_ bundle: String) -> Bool { clients.values.contains { $0.bundle == bundle } }
    func supportsForegroundInspection(_ bundle: String) -> Bool {
        clients.values.contains { $0.bundle == bundle && $0.supportsForegroundInspection }
    }
    func contains(_ id: String) -> Bool { clients[id]?.synchronized == true }
    /// A user-requested read of one current normal page. It never enters onSnapshot.
    func inspectForeground(bundle: String) async throws -> BrowserBridgeMessage? {
        let peers = clients.keys.filter { clients[$0]?.bundle == bundle && clients[$0]?.supportsForegroundInspection == true }.sorted()
        guard !peers.isEmpty else {
            throw StoreError("请安装或重新加载 0.4 版 SmartApp 浏览器扩展，再回到当前页面查找目录。")
        }
        for id in peers {
            try Task.checkCancellation()
            let request = UUID().uuidString
            let message: BrowserBridgeMessage? = try await withTaskCancellationHandler {
                try await withCheckedThrowingContinuation { continuation in
                    guard !Task.isCancelled else { continuation.resume(throwing: CancellationError()); return }
                    inspections[request] = (id, continuation)
                    send(id, ["type": .string("inspect_foreground"), "requestID": .string(request), "foreground": .bool(true)])
                }
            } onCancel: {
                Task { @MainActor [weak self] in self?.inspections.removeValue(forKey: request)?.continuation.resume(throwing: CancellationError()) }
            }
            if message?.snapshot.windows.first?.tabs.first != nil { return message }
        }
        return nil
    }
    func activate(clientID: String, window: String, tab: String, url: String) async -> Bool {
        guard clients[clientID]?.synchronized == true else { return false }
        let request = UUID().uuidString
        return await withCheckedContinuation { continuation in
            pending[request] = continuation
            send(clientID, ["type": .string("activate"), "requestID": .string(request),
                "windowID": .string(window), "tabID": .string(tab), "url": .string(url)])
            DispatchQueue.main.asyncAfter(deadline: .now() + 3) { [weak self] in self?.pending.removeValue(forKey: request)?.resume(returning: false) }
        }
    }
    private func attach(_ connection: NWConnection) {
        guard clients.count < 16 else { connection.cancel(); return }
        let id = UUID().uuidString
        clients[id] = Client(connection)
        connection.stateUpdateHandler = { [weak self] state in
            Task { @MainActor in
                switch state { case .failed, .cancelled: self?.disconnect(id); default: break }
            }
        }
        connection.start(queue: .main); receive(id)
        DispatchQueue.main.asyncAfter(deadline: .now() + 5) { [weak self] in
            if self?.clients[id]?.bundle == nil { self?.disconnect(id) }
        }
    }
    private func receive(_ id: String) {
        guard let client = clients[id] else { return }
        client.connection.receive(minimumIncompleteLength: 1, maximumLength: 65536) { [weak self] data, _, done, error in
            Task { @MainActor in
                guard let self, let client = self.clients[id] else { return }
                do {
                    if let data { client.buffer.append(data) }
                    while let payload = try BrowserBridgeWire.take(from: &client.buffer) { try self.consume(payload, id: id) }
                    if done || error != nil { self.disconnect(id) } else { self.receive(id) }
                } catch { self.disconnect(id) }
            }
        }
    }
    private func consume(_ payload: Data, id: String) throws {
        guard let client = clients[id] else { return }
        if client.bundle == nil {
            let hello = try JSONDecoder().decode([String: JSONValue].self, from: payload)
            guard hello["type"]?.string == "hello", hello["token"]?.string == token,
                  let bundle = hello["bundle"]?.string, BrowserReader.supported.contains(bundle) else { throw StoreError("Bridge authentication failed") }
            client.bundle = bundle
            sendCapture(id)
            return
        }
        if let result = try? JSONDecoder().decode([String: JSONValue].self, from: payload) {
            if result["type"]?.string == "capabilities" {
                client.supportsForegroundInspection = result["foregroundInspection"]?.bool == true; return
            }
            if result["type"]?.string == "foreground_inspection" {
                guard let request = result["requestID"]?.string, let pending = inspections[request], pending.client == id else { return }
                inspections.removeValue(forKey: request)
                do {
                    guard let message = try result["message"]?.decoded(BrowserBridgeMessage.self), message.isValid else {
                        throw StoreError("无法确认当前浏览器的普通活动页面，已停止查找。")
                    }
                    pending.continuation.resume(returning: message)
                } catch { pending.continuation.resume(throwing: error) }
                return
            }
            if result["type"]?.string == "activation_result" {
                if let request = result["requestID"]?.string { pending.removeValue(forKey: request)?.resume(returning: result["success"]?.bool == true) }
                return
            }
        }
        let message = try JSONDecoder().decode(BrowserBridgeMessage.self, from: payload)
        guard message.isValid, message.sequence == client.sequence + 1 else { throw StoreError("Invalid browser state") }
        client.sequence = message.sequence
        guard enabled else { return }
        if let request = client.requestID {
            guard message.requestID == request else { return }
            client.requestID = nil; client.synchronized = true
        }
        guard client.synchronized, abs(Date().timeIntervalSince1970 * 1000 - message.capturedAt) < 5000 else { return }
        onSnapshot?(id, client.bundle!, message, ProcessInfo.processInfo.systemUptime)
    }
    private func send(_ id: String, _ value: [String: JSONValue]) {
        guard let client = clients[id], let data = try? BrowserBridgeWire.frame(JSONEncoder().encode(value)) else { return }
        client.queuedBytes += data.count
        guard client.queuedBytes < 1_048_576 else { disconnect(id); return }
        client.connection.send(content: data, completion: .contentProcessed { [weak self] error in
            Task { @MainActor in
                self?.clients[id]?.queuedBytes -= data.count
                if error != nil { self?.disconnect(id) }
            }
        })
    }
    private func sendCapture(_ id: String) {
        guard let client = clients[id], let bundle = client.bundle else { return }
        client.synchronized = false
        let request = UUID().uuidString; client.requestID = request
        let foreground = enabled && bundle == foregroundBundle
        send(id, ["type": .string("capture"), "enabled": .bool(enabled),
            "foreground": .bool(foreground), "requestID": .string(request)])
    }
    private func disconnect(_ id: String) {
        guard let client = clients.removeValue(forKey: id) else { return }
        for request in inspections.keys.filter({ inspections[$0]?.client == id }) {
            inspections.removeValue(forKey: request)?.continuation.resume(throwing: StoreError("浏览器连接已断开，请重新唤出目录查找。"))
        }
        client.connection.cancel()
        if let bundle = client.bundle { onDisconnect?(id, bundle) }
    }
}
