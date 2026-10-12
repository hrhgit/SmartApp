import Foundation
import Darwin

/// A disposable native Apple-event reader. The collector remains responsible for
/// foreground admission and confirms the same app again after each reply.
enum DirectoryNativeTransport {
    static let session = DirectoryNativeSession(
        executableURL: Bundle.main.bundleURL.appendingPathComponent("Contents/MacOS/smartapp-directory-host"))

    static func read(source: String, pid: Int32) -> DirectoryNativeSession.Reply? {
        session.read(source: source, pid: pid)
    }

    static func close() { session.close() }
}

final class DirectoryNativeSession: @unchecked Sendable {
    struct Reply: Sendable {
        let status: String
        let snapshot: Data?
    }

    private let executableURL: URL
    private let lock = NSLock()
    private var process: Process?
    private var input: FileHandle?
    private var output: FileHandle?
    private var buffered = Data()
    private var nextID: UInt64 = 0
    private var retryAfter: Double = 0

    init(executableURL: URL) { self.executableURL = executableURL }

    func read(source: String, pid: Int32) -> Reply? {
        lock.lock()
        defer { lock.unlock() }
        guard ProcessInfo.processInfo.systemUptime >= retryAfter else { return nil }
        guard startIfNeeded(), let input else { failed(); return nil }
        nextID &+= 1
        let id = nextID
        let request: [String: Any] = ["id": id, "source": source, "pid": pid]
        guard var bytes = try? JSONSerialization.data(withJSONObject: request) else { failed(); return nil }
        bytes.append(0x0A)
        do { try input.write(contentsOf: bytes) }
        catch { failed(); return nil }
        guard let line = readLine(until: ProcessInfo.processInfo.systemUptime + 2.5),
              let object = try? JSONSerialization.jsonObject(with: line) as? [String: Any],
              (object["id"] as? NSNumber)?.uint64Value == id,
              let status = object["status"] as? String else { failed(); return nil }
        if status == "stale" { return Reply(status: status, snapshot: nil) }
        guard status == "ready", let snapshot = object["snapshot"] as? [String: Any],
              let snapshotBytes = try? JSONSerialization.data(withJSONObject: snapshot),
              snapshotBytes.count <= 1_048_576 else { failed(); return nil }
        return Reply(status: status, snapshot: snapshotBytes)
    }

    func close() {
        lock.lock()
        reset()
        retryAfter = 0
        lock.unlock()
    }

    private func startIfNeeded() -> Bool {
        if process?.isRunning == true { return true }
        reset()
        guard FileManager.default.isExecutableFile(atPath: executableURL.path) else { return false }
        let process = Process()
        let inputPipe = Pipe(), outputPipe = Pipe()
        process.executableURL = executableURL
        process.standardInput = inputPipe
        process.standardOutput = outputPipe
        process.standardError = FileHandle.nullDevice
        do { try process.run() } catch { return false }
        inputPipe.fileHandleForReading.closeFile()
        outputPipe.fileHandleForWriting.closeFile()
        self.process = process
        input = inputPipe.fileHandleForWriting
        output = outputPipe.fileHandleForReading
        return true
    }

    private func readLine(until deadline: Double) -> Data? {
        guard let output else { return nil }
        while true {
            if let newline = buffered.firstIndex(of: 0x0A) {
                let line = Data(buffered[..<newline])
                buffered.removeSubrange(...newline)
                return line
            }
            if buffered.count > 1_048_576 { return nil }
            let remaining = deadline - ProcessInfo.processInfo.systemUptime
            guard remaining > 0 else { return nil }
            var descriptor = pollfd(fd: output.fileDescriptor, events: Int16(POLLIN | POLLHUP), revents: 0)
            let ready = Darwin.poll(&descriptor, 1, Int32(ceil(remaining * 1_000)))
            if ready < 0 && errno == EINTR { continue }
            guard ready > 0 else { return nil }
            var bytes = [UInt8](repeating: 0, count: 4096)
            let count = bytes.withUnsafeMutableBytes { Darwin.read(output.fileDescriptor, $0.baseAddress, $0.count) }
            guard count > 0 else { return nil }
            buffered.append(contentsOf: bytes[..<count])
        }
    }

    private func failed() {
        reset()
        retryAfter = ProcessInfo.processInfo.systemUptime + 30
    }

    private func reset() {
        if let process, process.isRunning {
            kill(process.processIdentifier, SIGKILL)
            process.waitUntilExit()
        }
        input?.closeFile(); output?.closeFile()
        process = nil; input = nil; output = nil
        buffered.removeAll(keepingCapacity: true)
    }
}
