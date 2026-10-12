import Foundation
import Darwin
import IntentCore

// A persistent stdio <-> localhost relay. Nothing from the browser is stored here.
func writeAll(_ fd: Int32, _ data: Data) throws {
    try data.withUnsafeBytes { bytes in
        var offset = 0
        while offset < bytes.count {
            let n = Darwin.write(fd, bytes.baseAddress!.advanced(by: offset), bytes.count - offset)
            if n < 0 && errno == EINTR { continue }
            guard n > 0 else { throw StoreError("Bridge write failed") }
            offset += n
        }
    }
}
do {
    signal(SIGPIPE, SIG_IGN)
    let args = CommandLine.arguments
    guard args.count >= 3, ["com.google.Chrome", "com.microsoft.edgemac", "com.brave.Browser"].contains(args[1]),
          args[2].hasPrefix("chrome-extension://"), args[2].hasSuffix("/") else { throw StoreError("Invalid host invocation") }
    let root = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0]
        .appendingPathComponent("SmartApp/runtime/browser-bridge-v1")
    let config = try JSONDecoder().decode([String: JSONValue].self, from: Data(contentsOf: root.appendingPathComponent("endpoint.json")))
    guard let port = config["port"]?.number, port >= 1, port <= 65535,
          let token = config["token"]?.string,
          let origin = config["origin"]?.string, args[2] == origin else { throw StoreError("Invalid bridge endpoint") }
    let fd = socket(AF_INET, SOCK_STREAM, 0)
    guard fd >= 0 else { throw StoreError("Bridge socket unavailable") }
    defer { close(fd) }
    var address = sockaddr_in()
    address.sin_len = UInt8(MemoryLayout<sockaddr_in>.size)
    address.sin_family = sa_family_t(AF_INET)
    address.sin_port = UInt16(port).bigEndian
    address.sin_addr.s_addr = inet_addr("127.0.0.1")
    let connected = withUnsafePointer(to: &address) { ptr in
        ptr.withMemoryRebound(to: sockaddr.self, capacity: 1) { Darwin.connect(fd, $0, socklen_t(MemoryLayout<sockaddr_in>.size)) }
    }
    guard connected == 0 else { throw StoreError("SmartApp is not running") }
    let hello: [String: JSONValue] = ["type": .string("hello"), "token": .string(token), "bundle": .string(args[1])]
    try writeAll(fd, BrowserBridgeWire.frame(JSONEncoder().encode(hello)))
    var descriptors = [pollfd(fd: STDIN_FILENO, events: Int16(POLLIN), revents: 0), pollfd(fd: fd, events: Int16(POLLIN), revents: 0)]
    var buffers = [Data(), Data()]
    var bytes = [UInt8](repeating: 0, count: 65536)
    relay: while true {
        let result = poll(&descriptors, 2, -1)
        if result < 0 && errno == EINTR { continue }
        guard result > 0 else { break }
        for i in 0..<2 {
            if descriptors[i].revents & Int16(POLLIN) != 0 {
                let n = Darwin.read(descriptors[i].fd, &bytes, bytes.count)
                guard n > 0 else { break relay }
                buffers[i].append(contentsOf: bytes.prefix(n))
                while let payload = try BrowserBridgeWire.take(from: &buffers[i]) {
                    try writeAll(i == 0 ? fd : STDOUT_FILENO, BrowserBridgeWire.frame(payload))
                }
            } else if descriptors[i].revents & Int16(POLLHUP | POLLERR | POLLNVAL) != 0 { break relay }
        }
    }
} catch {
    // No URL, token, payload, or input text in diagnostics.
    FileHandle.standardError.write(Data("SmartApp browser bridge unavailable. Start SmartApp and reconnect the extension.\n".utf8))
    exit(1)
}
