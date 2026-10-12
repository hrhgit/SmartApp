import XCTest
import AppKit
import SwiftUI
import IntentCore
@testable import SmartApp

final class DestinationPickerLayoutTests: XCTestCase {
    func testFirstSlotStartsUpperRightAndSlotsGrowClockwiseWithoutFillingCircle() {
        let first = DestinationPickerLayout.offset(at: 0, count: 1)
        XCTAssertGreaterThan(first.width, 0)
        XCTAssertLessThan(first.height, 0)
        XCTAssertEqual(first.width, -first.height, accuracy: 0.001)
        for count in 1...6 {
            for index in 0..<count {
                XCTAssertEqual(DestinationPickerLayout.offset(at: index, count: count),
                               DestinationPickerLayout.offset(at: index, count: 6))
            }
        }
        let second = DestinationPickerLayout.offset(at: 1, count: 2)
        XCTAssertGreaterThan(second.height, first.height)
        XCTAssertGreaterThan(second.width, 0)
        XCTAssertLessThan(DestinationPickerLayout.angleStep * 5, 2 * .pi)
    }

    func testSixHitRegionsAreDisjointAndInsidePanel() {
        let size = DestinationPickerLayout.hoverSize
        let center = DestinationPickerLayout.diameter / 2
        let captionOffset = (DestinationPickerLayout.itemSize.height - DestinationPickerLayout.iconDiameter) / 2
        let panel = CGRect(x: 0, y: 0, width: center * 2, height: center * 2)
        let frames = (0..<6).map { index -> CGRect in
            let offset = DestinationPickerLayout.offset(at: index, count: 6)
            return CGRect(x: center + offset.width - size.width / 2,
                          y: center + offset.height + captionOffset - size.height / 2,
                          width: size.width, height: size.height)
        }
        for (index, frame) in frames.enumerated() {
            XCTAssertTrue(panel.contains(frame), "slot \(index)")
            for later in frames.dropFirst(index + 1) { XCTAssertFalse(frame.intersects(later)) }
        }
        let tools = [(DestinationPickerLayout.fileToolCenter, DestinationPickerLayout.fileToolSize),
                     (DestinationPickerLayout.clipboardToolCenter, DestinationPickerLayout.clipboardToolSize)].map { point, size in
            CGRect(x: center + point.x - size.width / 2, y: center + point.y - size.height / 2,
                   width: size.width, height: size.height)
        }
        for tool in tools {
            XCTAssertTrue(panel.contains(tool))
            for frame in frames { XCTAssertFalse(tool.intersects(frame)) }
        }
        XCTAssertFalse(tools[0].intersects(tools[1]))
    }

    func testPanelStaysOnScreenAtEveryEdgeAndOnSmallScreen() {
        for screen in [CGRect(x: 100, y: 50, width: 1_200, height: 800),
                       CGRect(x: -400, y: -200, width: 340, height: 300)] {
            for point in [CGPoint(x: screen.minX, y: screen.minY), CGPoint(x: screen.maxX, y: screen.maxY),
                          CGPoint(x: screen.minX, y: screen.maxY), CGPoint(x: screen.maxX, y: screen.minY)] {
                let frame = DestinationPickerLayout.frame(around: point, visibleFrame: screen)
                XCTAssertTrue(screen.insetBy(dx: 8, dy: 8).contains(frame))
                XCTAssertEqual(frame.width, frame.height)
            }
        }
    }

    @MainActor func testRenderNativeOneThreeAndSixItemPreviews() throws {
        let root = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent()
            .deletingLastPathComponent().appendingPathComponent("build/menu-previews")
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        let labels = ["Safari", "文件", "日历", "终端", "邮件", "设置"]
        let bundles = ["com.apple.Safari", "com.apple.finder", "com.apple.iCal", "com.apple.Terminal",
                       "com.apple.mail", "com.apple.systempreferences"]
        let symbols = ["safari", "folder.fill", "calendar", "terminal.fill", "envelope.fill", "gearshape.fill"]
        let items = (0..<6).map { index in
            let target = Destination(id: "preview-\(index)", kind: .app, name: labels[index],
                                     bundleID: bundles[index], admission: TargetAdmission(basis: "standalone_application"))
            let icon = NSWorkspace.shared.urlForApplication(withBundleIdentifier: bundles[index])
                .map { NSWorkspace.shared.icon(forFile: $0.path) }
            return DestinationPickerItem(choice: NavigationChoice(target: target, entrance: nil),
                icon: icon, faviconURL: nil, symbol: symbols[index], kindLabel: "应用")
        }
        for count in [1, 3, 6] {
            let content = ZStack {
                LinearGradient(colors: [Color(red: 0.86, green: 0.91, blue: 0.97),
                                        Color(red: 0.98, green: 0.97, blue: 0.93)],
                               startPoint: .topLeading, endPoint: .bottomTrailing)
                DestinationPickerView(items: Array(items.prefix(count)), choose: { _ in }, close: {},
                                      initiallyExpanded: true)
            }.frame(width: DestinationPickerLayout.diameter, height: DestinationPickerLayout.diameter)
            _ = NSApplication.shared
            let frame = CGRect(x: -10_000, y: -10_000, width: DestinationPickerLayout.diameter,
                               height: DestinationPickerLayout.diameter)
            let window = NSWindow(contentRect: frame, styleMask: .borderless, backing: .buffered, defer: false)
            window.isReleasedWhenClosed = false
            let hosting = NSHostingView(rootView: content)
            hosting.frame = CGRect(origin: .zero, size: frame.size)
            window.contentView = hosting
            window.orderFront(nil)
            hosting.layoutSubtreeIfNeeded()
            window.displayIfNeeded()
            let bitmap = try XCTUnwrap(hosting.bitmapImageRepForCachingDisplay(in: hosting.bounds))
            hosting.cacheDisplay(in: hosting.bounds, to: bitmap)
            window.orderOut(nil)
            let png = try XCTUnwrap(bitmap.representation(using: .png, properties: [:]))
            try png.write(to: root.appendingPathComponent("menu-\(count).png"))
            XCTAssertGreaterThan(png.count, 10_000)
        }
    }
}
