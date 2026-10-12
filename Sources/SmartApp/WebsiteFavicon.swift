import AppKit
import Foundation
import IntentCore

/// Website artwork is derived only for the transient picker presentation.
enum WebsiteFavicon {
    static func url(for target: Destination) -> URL? {
        guard target.kind == .page,
              let origin = target.origin,
              var components = URLComponents(string: origin),
              components.scheme?.lowercased() == "https",
              components.host?.isEmpty == false,
              components.user == nil,
              components.password == nil else { return nil }
        components.path = "/favicon.ico"
        components.query = nil
        components.fragment = nil
        return components.url
    }
}

/// Keeps site icons out of capture storage, exports, and the on-disk URL cache.
@MainActor
final class WebsiteFaviconLoader {
    static let shared = WebsiteFaviconLoader()

    private var cache = WebsiteFaviconMemoryCache()

    func image(for url: URL) async -> NSImage? {
        if let image = cache.image(for: url) { return image }
        if cache.isUnavailable(url) { return nil }
        let data = await Self.loadData(from: url)
        guard !Task.isCancelled else { return nil }
        guard let data, let image = NSImage(data: data) else {
            cache.markUnavailable(url)
            return nil
        }
        cache.insert(image, for: url, encodedByteCount: data.count)
        return image
    }

    private nonisolated static func loadData(from url: URL) async -> Data? {
        let maximumByteCount = 512 * 1_024
        let configuration = URLSessionConfiguration.ephemeral
        configuration.httpShouldSetCookies = false
        configuration.httpCookieStorage = nil
        configuration.urlCache = nil
        configuration.requestCachePolicy = .reloadIgnoringLocalCacheData
        configuration.timeoutIntervalForRequest = 2
        configuration.timeoutIntervalForResource = 3
        let session = URLSession(configuration: configuration)
        var request = URLRequest(url: url, cachePolicy: .reloadIgnoringLocalCacheData, timeoutInterval: 2)
        request.setValue("image/*", forHTTPHeaderField: "Accept")

        do {
            let (bytes, response) = try await session.bytes(for: request)
            guard let response = response as? HTTPURLResponse,
                  (200..<300).contains(response.statusCode),
                  let responseURL = response.url,
                  responseURL.scheme?.lowercased() == "https",
                  responseURL.host?.lowercased() == url.host?.lowercased(),
                  responseURL.port == url.port else { return nil }
            if let value = response.value(forHTTPHeaderField: "Content-Length"),
               let count = Int(value), count > maximumByteCount { return nil }

            var data = Data()
            data.reserveCapacity(min(maximumByteCount, 32 * 1_024))
            for try await byte in bytes {
                guard !Task.isCancelled, data.count < maximumByteCount else { return nil }
                data.append(byte)
            }
            return data
        } catch {
            return nil
        }
    }
}

/// Small, predictable menu-only cache. Failed sites are retried after a short delay.
@MainActor
struct WebsiteFaviconMemoryCache {
    private struct ImageEntry {
        let image: NSImage
        let cost: Int
        var lastUse: UInt64
    }
    private struct FailedEntry {
        let retryAfter: Date
        var lastUse: UInt64
    }

    private let maximumImages: Int
    private let maximumBytes: Int
    private let maximumFailures: Int
    private var images: [URL: ImageEntry] = [:]
    private var failures: [URL: FailedEntry] = [:]
    private var clock: UInt64 = 0
    private(set) var imageBytes = 0
    var imageCount: Int { images.count }
    var failureCount: Int { failures.count }

    init(maximumImages: Int = 64, maximumBytes: Int = 8 * 1_024 * 1_024, maximumFailures: Int = 128) {
        self.maximumImages = maximumImages
        self.maximumBytes = maximumBytes
        self.maximumFailures = maximumFailures
    }

    mutating func image(for url: URL) -> NSImage? {
        guard var entry = images[url] else { return nil }
        entry.lastUse = nextUse()
        images[url] = entry
        return entry.image
    }

    mutating func insert(_ image: NSImage, for url: URL, encodedByteCount: Int) {
        failures.removeValue(forKey: url)
        if let previous = images.removeValue(forKey: url) { imageBytes -= previous.cost }
        let cost = max(encodedByteCount, Self.decodedByteCount(image))
        guard maximumImages > 0, cost <= maximumBytes else { return }
        images[url] = ImageEntry(image: image, cost: cost, lastUse: nextUse())
        imageBytes += cost
        while images.count > maximumImages || imageBytes > maximumBytes {
            guard let oldest = images.min(by: { $0.value.lastUse < $1.value.lastUse })?.key,
                  let removed = images.removeValue(forKey: oldest) else { break }
            imageBytes -= removed.cost
        }
    }

    mutating func isUnavailable(_ url: URL, now: Date = Date()) -> Bool {
        guard var entry = failures[url] else { return false }
        guard entry.retryAfter > now else { failures.removeValue(forKey: url); return false }
        entry.lastUse = nextUse()
        failures[url] = entry
        return true
    }

    mutating func markUnavailable(_ url: URL, now: Date = Date()) {
        guard maximumFailures > 0 else { return }
        failures[url] = FailedEntry(retryAfter: now.addingTimeInterval(5 * 60), lastUse: nextUse())
        while failures.count > maximumFailures {
            guard let oldest = failures.min(by: { $0.value.lastUse < $1.value.lastUse })?.key else { break }
            failures.removeValue(forKey: oldest)
        }
    }

    private mutating func nextUse() -> UInt64 { clock &+= 1; return clock }

    private static func decodedByteCount(_ image: NSImage) -> Int {
        let sizes = image.representations.isEmpty
            ? [image.size]
            : image.representations.map {
                NSSize(width: $0.pixelsWide > 0 ? CGFloat($0.pixelsWide) : image.size.width,
                       height: $0.pixelsHigh > 0 ? CGFloat($0.pixelsHigh) : image.size.height)
            }
        return sizes.reduce(0) { total, size in
            guard size.width.isFinite, size.height.isFinite,
                  (0...16_384).contains(size.width), (0...16_384).contains(size.height) else { return Int.max }
            let width = max(0, Int(size.width)), height = max(0, Int(size.height))
            let (pixels, pixelOverflow) = width.multipliedReportingOverflow(by: height)
            let (bytes, byteOverflow) = pixels.multipliedReportingOverflow(by: 4)
            let (sum, sumOverflow) = total.addingReportingOverflow(bytes)
            return pixelOverflow || byteOverflow || sumOverflow ? Int.max : sum
        }
    }
}
