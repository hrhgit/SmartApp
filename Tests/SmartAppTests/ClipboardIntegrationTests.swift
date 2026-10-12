import XCTest
import AppKit
import SwiftUI
import IntentCore
@testable import SmartApp

final class ClipboardIntegrationTests: XCTestCase {
    @MainActor func testRenderHistoryPreviewWithSyntheticContent() async throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        let suite = "smartapp.clipboard-preview." + UUID().uuidString
        let defaults = UserDefaults(suiteName: suite)!
        let board = NSPasteboard(name: .init(suite))
        defer { defaults.removePersistentDomain(forName: suite); board.releaseGlobally(); try? FileManager.default.removeItem(at: root) }
        let store = try ClipboardHistoryStore(url: root.appendingPathComponent("clipboard/history-v1.sqlite"))
        let selected = ClipboardEntry(origin: .observedChange, items: [.init(representations: [.init(type: "public.utf8-plain-text", data: Data("Project Alpha notes".utf8))],
            text: "Project Alpha notes\nCheck the project folder and the latest draft.\nThis is synthetic preview content.")])
        try store.insert(selected)
        try store.insert(.init(origin: .existingTimeUnknown, items: [.init(representations: [], text: "https://example.test/project-alpha")], isPinned: true))
        let model = ClipboardController(directory: root, defaults: defaults, board: board, liveObservation: false)
        for _ in 0..<100 where model.summaries.isEmpty { try await Task.sleep(for: .milliseconds(10)) }
        model.select(id: selected.id)
        for _ in 0..<100 where model.selectedEntry == nil { try await Task.sleep(for: .milliseconds(10)) }
        XCTAssertEqual(model.selectedEntry?.id, selected.id)
        try IntegrationPreviewRenderer.render(ClipboardHistoryView(model: model, close: {}), name: "clipboard", size: CGSize(width: 780, height: 560))
        model.shutdown()
    }
    @MainActor func testRichMultipleItemsRoundTripAndConcealedContentsAreExcluded() throws {
        let board = NSPasteboard(name: .init("smartapp.clipboard-test." + UUID().uuidString))
        defer { board.releaseGlobally() }
        let file = NSPasteboardItem()
        file.setString("file:///tmp/fixture-file.txt", forType: .fileURL)
        let text = NSPasteboardItem()
        text.setString("hello project-alpha", forType: .string)
        let rich = Data("{\\rtf1 hello}".utf8)
        text.setData(rich, forType: .rtf)
        XCTAssertTrue(board.writeObjects([file, text]))
        let entry = try XCTUnwrap(ClipboardTransfer.read(board, origin: .observedChange))
        XCTAssertEqual(entry.items.count, 2)
        XCTAssertEqual(entry.fileLocators, ["file:///tmp/fixture-file.txt"])
        try ClipboardTransfer.write(entry, to: board)
        let replay = try XCTUnwrap(ClipboardTransfer.read(board, origin: .observedChange))
        XCTAssertEqual(replay.items, entry.items)
        board.clearContents()
        let secret = NSPasteboardItem(); secret.setString("not-to-be-read", forType: .string)
        secret.setData(Data(), forType: .init("org.nspasteboard.ConcealedType"))
        XCTAssertTrue(board.writeObjects([secret]))
        XCTAssertNil(try ClipboardTransfer.read(board, origin: .observedChange))
    }
    @MainActor func testControllerStartsWithUnknownTimeAndOnlyObservedChangesEnterAutomaticContext() throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        let suite = "smartapp.clipboard-test." + UUID().uuidString
        let defaults = UserDefaults(suiteName: suite)!
        let board = NSPasteboard(name: .init(suite))
        defer { defaults.removePersistentDomain(forName: suite); board.releaseGlobally(); try? FileManager.default.removeItem(at: root) }
        board.setString("before recording", forType: .string)
        let controller = ClipboardController(directory: root, defaults: defaults, board: board, liveObservation: false)
        XCTAssertFalse(controller.isRecording)
        controller.start(); XCTAssertNil(controller.freezeForRecommendation())
        board.clearContents(); board.setString("new task", forType: .string); controller.poll()
        XCTAssertEqual(controller.freezeForRecommendation()?.entry.text, "new task")
        controller.setBoundary("sleep", blocked: true)
        board.clearContents(); board.setString("while asleep", forType: .string)
        controller.setBoundary("sleep", blocked: false); controller.poll()
        XCTAssertNil(controller.freezeForRecommendation())
        controller.shutdown()
        let store = try ClipboardHistoryStore(url: root.appendingPathComponent("clipboard/history-v1.sqlite"))
        XCTAssertEqual(try store.count(), 2)
        XCTAssertTrue(try store.summaries().contains { $0.origin == .existingTimeUnknown })
    }
    @MainActor func testClipboardCannotEnterForegroundHistoryOrExports() throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: root) }
        let recorder = RecorderModel(dataDirectory: root, liveObservation: false)
        recorder.start()
        XCTAssertNil(recorder.record(.clipboardChanged, source: "fixture", data: ["text": .string("private clipboard fixture")]))
        recorder.shutdown()
        let store = try EventStore(url: recorder.databaseURL, readOnly: true)
        let output = root.appendingPathComponent("export")
        _ = try store.export(to: output)
        for name in ["events.jsonl", "target-history.jsonl", "navigation-samples.jsonl"] {
            let data = try String(contentsOf: output.appendingPathComponent(name), encoding: .utf8)
            XCTAssertFalse(data.contains("private clipboard fixture"))
        }
    }
}

enum IntegrationPreviewRenderer {
    @MainActor static func render<V: View>(_ view: V, name: String, size: CGSize) throws {
        let output = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent()
            .deletingLastPathComponent().appendingPathComponent("build/integration-previews")
        try FileManager.default.createDirectory(at: output, withIntermediateDirectories: true)
        _ = NSApplication.shared
        let window = NSWindow(contentRect: CGRect(origin: CGPoint(x: -10_000, y: -10_000), size: size),
            styleMask: .borderless, backing: .buffered, defer: false)
        window.isReleasedWhenClosed = false
        let hosting = NSHostingView(rootView: view.environment(\.colorScheme, .light))
        hosting.frame = CGRect(origin: .zero, size: size); window.contentView = hosting; window.orderFront(nil)
        hosting.layoutSubtreeIfNeeded(); window.displayIfNeeded()
        let bitmap = try XCTUnwrap(hosting.bitmapImageRepForCachingDisplay(in: hosting.bounds))
        hosting.cacheDisplay(in: hosting.bounds, to: bitmap); window.orderOut(nil)
        let png = try XCTUnwrap(bitmap.representation(using: .png, properties: [:]))
        try png.write(to: output.appendingPathComponent(name + ".png"))
    }
}
