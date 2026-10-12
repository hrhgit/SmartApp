import Foundation
import FileRecallCore

@MainActor
final class LocalKevService {
    private var process: Process?
    private var stopping = false
    private let session: URLSession = {
        let configuration = URLSessionConfiguration.ephemeral
        configuration.timeoutIntervalForRequest = .greatestFiniteMagnitude
        configuration.timeoutIntervalForResource = .greatestFiniteMagnitude
        configuration.urlCache = nil
        return URLSession(configuration: configuration)
    }()

    func ensureRunning() async throws {
        if stopping {
            while process?.isRunning == true {
                try Task.checkCancellation()
                try await Task.sleep(for: .milliseconds(100))
            }
            process = nil; stopping = false
        }
        if try await isReady() { return }
        try Task.checkCancellation()
        if process?.isRunning != true {
            let support = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0].appendingPathComponent("FileRecall")
            let checkout = support.appendingPathComponent("kev")
            let python = checkout.appendingPathComponent(".venv/bin/python")
            guard FileManager.default.isExecutableFile(atPath: python.path) else {
                throw RecommendationFailure("启动本地模型", "未找到已安装的 Kev Python 环境：\(python.path)")
            }
            let child = Process()
            child.executableURL = python
            child.currentDirectoryURL = checkout
            let adapter: URL
            if let bundled = Bundle.main.url(forResource: "KevSmartAppServer", withExtension: "py") { adapter = bundled }
            else {
                #if DEBUG
                adapter = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent()
                    .deletingLastPathComponent().appendingPathComponent("Resources/KevSmartAppServer.py")
                #else
                throw RecommendationFailure("启动本地模型", "应用包缺少完整输入校验程序，请重新构建 SmartApp。")
                #endif
            }
            child.arguments = [adapter.path, "--run", KevHTTPDecider.model, "--host", "127.0.0.1", "--port", "8009"]
            var environment = ProcessInfo.processInfo.environment
            environment["HF_HOME"] = support.appendingPathComponent("model-cache").path
            environment["HF_HUB_OFFLINE"] = "1"
            environment["KEV_PREFIX_CACHE"] = "0"
            environment["TRANSFORMERS_OFFLINE"] = "1"
            child.environment = environment
            child.standardOutput = FileHandle.nullDevice
            child.standardError = FileHandle.nullDevice
            do { try child.run() }
            catch { throw RecommendationFailure("启动本地模型", error.localizedDescription) }
            process = child
        }
        while true {
            try Task.checkCancellation()
            if try await isReady() { return }
            if process?.isRunning == false {
                throw RecommendationFailure("启动本地模型", "Kev 已退出（状态码 \(process?.terminationStatus ?? -1)）。请检查模型缓存和 8009 端口。")
            }
            try await Task.sleep(for: .milliseconds(500))
        }
    }

    func stop() {
        if process?.isRunning == true {
            if !stopping { process?.terminate(); stopping = true }
        } else { process = nil; stopping = false }
    }

    private func isReady() async throws -> Bool {
        guard let (data, response) = try? await session.data(from: URL(string: "http://127.0.0.1:8009/v1/models")!),
              (response as? HTTPURLResponse)?.statusCode == 200,
              let object = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              let models = object["models"] as? [[String: Any]] else { return false }
        guard models.contains(where: { $0["run"] as? String == KevHTTPDecider.model }) else {
            throw RecommendationFailure("启动本地模型", "8009 端口已被另一套模型占用，请关闭该服务后重试。")
        }
        guard process?.isRunning == true, !stopping else {
            throw RecommendationFailure("启动本地模型", "8009 端口已有其他程序的服务，请关闭该服务后重试。")
        }
        guard (response as? HTTPURLResponse)?.value(forHTTPHeaderField: "X-SmartApp-Context-Policy") == "complete-input-v1",
              models.allSatisfy({ ($0["prefix_cache"] as? [String: Any])?["size"] as? Int == 0 }) else {
            throw RecommendationFailure("启动本地模型", "8009 端口的服务没有启用完整输入校验或仍保留内容缓存，请关闭该服务后重试。")
        }
        return true
    }
}
