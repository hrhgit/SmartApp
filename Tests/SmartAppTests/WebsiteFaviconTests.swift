import XCTest
import IntentCore
@testable import SmartApp

final class WebsiteFaviconTests: XCTestCase {
    private func page(origin: String?, locator: String = "https://example.test/path?token=private#section") -> Destination {
        Destination(id: "page:test", kind: .page, name: locator, origin: origin, locator: locator,
                    admission: TargetAdmission(basis: "supported_browser_page"))
    }

    func testFaviconUsesOnlyTheHTTPSOrigin() {
        let target = page(origin: "https://example.test:8443")
        XCTAssertEqual(WebsiteFavicon.url(for: target)?.absoluteString, "https://example.test:8443/favicon.ico")
    }

    func testFaviconRejectsNonWebsiteAndUnsafeOrigins() {
        XCTAssertNil(WebsiteFavicon.url(for: page(origin: "http://example.test")))
        XCTAssertNil(WebsiteFavicon.url(for: page(origin: "https://user:secret@example.test")))
        XCTAssertNil(WebsiteFavicon.url(for: Destination(id: "app:test", kind: .app, name: "Test", bundleID: "test")))
    }

    @MainActor func testPagePickerItemCarriesOnlyTheDerivedFaviconURL() throws {
        let target = page(origin: "https://example.test")
        let item = try XCTUnwrap(DestinationPickerItem.resolve([NavigationChoice(target: target, entrance: nil)]).first)
        XCTAssertEqual(item.faviconURL?.absoluteString, "https://example.test/favicon.ico")
    }

    @MainActor func testPagePickerItemHidesHTTPProtocolInItsMenuLabel() throws {
        let locator = "https://example.test/path?token=private#section"
        let item = try XCTUnwrap(DestinationPickerItem.resolve([
            NavigationChoice(target: page(origin: "https://example.test", locator: locator), entrance: nil)
        ]).first)

        XCTAssertEqual(item.name, "example.test/path?token=private#section")
        XCTAssertEqual(item.choice.target.locator, locator)
        XCTAssertEqual(DestinationPickerItem.pageDisplayName("HTTP://example.test/a"), "example.test/a")
    }

    @MainActor func testIconCacheEvictsLeastRecentlyUsedAndRetriesFailures() {
        var cache = WebsiteFaviconMemoryCache(maximumImages: 2, maximumBytes: 200, maximumFailures: 2)
        let a = URL(string: "https://a.test/favicon.ico")!
        let b = URL(string: "https://b.test/favicon.ico")!
        let c = URL(string: "https://c.test/favicon.ico")!
        let icon = NSImage(size: NSSize(width: 1, height: 1))
        cache.insert(icon, for: a, encodedByteCount: 80)
        cache.insert(icon, for: b, encodedByteCount: 80)
        XCTAssertNotNil(cache.image(for: a))
        cache.insert(icon, for: c, encodedByteCount: 80)
        XCTAssertNil(cache.image(for: b))
        XCTAssertEqual(cache.imageCount, 2)
        XCTAssertEqual(cache.imageBytes, 160)
        cache.insert(icon, for: b, encodedByteCount: 300)
        XCTAssertNil(cache.image(for: b))
        XCTAssertEqual(cache.imageCount, 2)
        let pixels = NSBitmapImageRep(bitmapDataPlanes: nil, pixelsWide: 16, pixelsHigh: 16,
                                      bitsPerSample: 8, samplesPerPixel: 4, hasAlpha: true,
                                      isPlanar: false, colorSpaceName: .deviceRGB,
                                      bytesPerRow: 0, bitsPerPixel: 0)!
        let decodedIcon = NSImage(size: NSSize(width: 16, height: 16))
        decodedIcon.addRepresentation(pixels)
        cache.insert(decodedIcon, for: b, encodedByteCount: 20)
        XCTAssertNil(cache.image(for: b))
        XCTAssertEqual(cache.imageCount, 2)

        let now = Date(timeIntervalSince1970: 1_000)
        cache.markUnavailable(a, now: now)
        cache.markUnavailable(b, now: now)
        cache.markUnavailable(c, now: now)
        XCTAssertEqual(cache.failureCount, 2)
        XCTAssertFalse(cache.isUnavailable(a, now: now))
        XCTAssertTrue(cache.isUnavailable(b, now: now))
        XCTAssertFalse(cache.isUnavailable(b, now: now.addingTimeInterval(301)))
    }
}
