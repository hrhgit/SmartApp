import Foundation
import Darwin

/// Apple events can block on an unresponsive app or a permission dialog.
/// Run off the main actor, drain one pipe, and bound every helper's lifetime.
enum AutomationScript {
    enum Language: String { case javaScript = "JavaScript", appleScript = "AppleScript" }

    /// Foreground directory reads are frequent, so keep one JavaScript interpreter warm.
    /// A protocol error or timeout closes it and uses the existing one-shot path.
    private static let directorySession = DirectorySession()

    static func readDirectory(_ script: String) -> (code: Int, output: String) {
        directorySession.run(script) ?? run(script)
    }

    static func closeDirectorySession() {
        DirectoryNativeTransport.close()
        directorySession.close()
    }

    static func closeScriptDirectorySession() { directorySession.close() }

    static func run(_ script: String, language: Language = .javaScript, timeout: Double = 6) -> (code: Int, output: String) {
        let process = Process()
        process.executableURL = URL(fileURLWithPath: "/usr/bin/osascript")
        process.arguments = ["-l", language.rawValue, "-e", script]
        let pipe = Pipe()
        process.standardOutput = pipe; process.standardError = pipe
        do { try process.run() } catch { return (-1, "") }
        let deadline = DispatchWorkItem {
            if process.isRunning { kill(process.processIdentifier, SIGKILL) }
        }
        DispatchQueue.global(qos: .utility).asyncAfter(deadline: .now() + timeout, execute: deadline)
        let data = pipe.fileHandleForReading.readDataToEndOfFile()
        process.waitUntilExit(); deadline.cancel()
        let output = String(decoding: data, as: UTF8.self)
        return (output.contains("(-1743)") ? -1743 : Int(process.terminationStatus), output)
    }

    static func literal(_ value: String) -> String {
        // JSON string encoding also escapes quotes and backslashes for JavaScript.
        String(decoding: try! JSONEncoder().encode(value), as: UTF8.self)
    }

    static func appleScriptLiteral(_ value: String) -> String {
        // Locators are URL-encoded before they reach this boundary. Escape defensively so a
        // malformed automation value cannot change the generated script structure.
        let escaped = value
            .replacingOccurrences(of: "\\", with: "\\\\")
            .replacingOccurrences(of: "\"", with: "\\\"")
            .replacingOccurrences(of: "\r", with: "\\r")
            .replacingOccurrences(of: "\n", with: "\\n")
        return "\"\(escaped)\""
    }
}

private final class DirectorySession: @unchecked Sendable {
    private struct Reply: Decodable {
        let id: UInt64
        let output: String
    }

    private let lock = NSLock()
    private var process: Process?
    private var terminal: FileHandle?
    private var buffered = Data()
    private var nextID: UInt64 = 0

    func run(_ script: String) -> (code: Int, output: String)? {
        lock.lock()
        defer { lock.unlock() }
        guard startIfNeeded(), let terminal else { reset(); return nil }
        nextID &+= 1
        let id = nextID
        let command = "JSON.stringify({id:\(id),output:eval(\(AutomationScript.literal(script)))})\n"
        do {
            try terminal.write(contentsOf: Data(command.utf8))
        } catch {
            reset(); return nil
        }
        let deadline = ProcessInfo.processInfo.systemUptime + 2
        while let line = readLine(until: deadline) {
            var response = line
            while response.hasPrefix(">> ") { response.removeFirst(3) }
            if response.hasPrefix("!! ") { reset(); return nil }
            guard response.hasPrefix("=> ") else { continue }
            guard let quoted = response.dropFirst(3).data(using: .utf8),
                  let json = try? JSONDecoder().decode(String.self, from: quoted),
                  let reply = try? JSONDecoder().decode(Reply.self, from: Data(json.utf8)),
                  reply.id == id else { reset(); return nil }
            return (0, reply.output)
        }
        reset()
        return nil
    }

    func close() {
        lock.lock()
        reset()
        lock.unlock()
    }

    private func startIfNeeded() -> Bool {
        if process?.isRunning == true { return true }
        reset()
        let process = Process()
        var master: Int32 = -1, slave: Int32 = -1
        guard openpty(&master, &slave, nil, nil, nil) == 0 else { return false }
        let terminal = FileHandle(fileDescriptor: master, closeOnDealloc: true)
        let childTerminal = FileHandle(fileDescriptor: slave, closeOnDealloc: true)
        process.executableURL = URL(fileURLWithPath: "/usr/bin/osascript")
        process.arguments = ["-i", "-l", "JavaScript"]
        process.standardInput = childTerminal
        process.standardOutput = childTerminal
        process.standardError = childTerminal
        do { try process.run() } catch {
            terminal.closeFile(); childTerminal.closeFile()
            return false
        }
        childTerminal.closeFile()
        self.process = process
        self.terminal = terminal
        return true
    }

    private func readLine(until deadline: Double) -> String? {
        guard let terminal else { return nil }
        while true {
            if let newline = buffered.firstIndex(of: 10) {
                let line = Data(buffered[..<newline])
                buffered.removeSubrange(...newline)
                return String(decoding: line, as: UTF8.self).trimmingCharacters(in: .whitespacesAndNewlines)
            }
            if buffered.count > 1_048_576 { return nil }
            let remaining = deadline - ProcessInfo.processInfo.systemUptime
            guard remaining > 0 else { return nil }
            let fd = terminal.fileDescriptor
            var descriptor = pollfd(fd: fd, events: Int16(POLLIN | POLLHUP), revents: 0)
            let ready = Darwin.poll(&descriptor, 1, Int32(ceil(remaining * 1_000)))
            if ready < 0 && errno == EINTR { continue }
            guard ready > 0 else { return nil }
            var bytes = [UInt8](repeating: 0, count: 4096)
            let count = bytes.withUnsafeMutableBytes { Darwin.read(fd, $0.baseAddress, $0.count) }
            guard count > 0 else { return nil }
            buffered.append(contentsOf: bytes[..<count])
        }
    }

    private func reset() {
        if let process, process.isRunning {
            kill(process.processIdentifier, SIGKILL)
            process.waitUntilExit()
        }
        terminal?.closeFile()
        process = nil
        terminal = nil
        buffered.removeAll(keepingCapacity: true)
    }
}
