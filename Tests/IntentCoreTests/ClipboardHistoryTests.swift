import XCTest
@testable import IntentCore

final class ClipboardHistoryTests: XCTestCase {
    private func entry(_ text: String, origin: ClipboardCaptureOrigin = .observedChange) -> ClipboardEntry {
        .init(origin: origin, items: [.init(representations: [.init(type: "public.utf8-plain-text", data: Data(text.utf8))], text: text)])
    }
    func testExistingContentAndResumedObservationCannotBecomeFreshEvidence() throws {
        var tracker = ClipboardObservationTracker()
        tracker.begin(version: 3)
        tracker.accepted(entry("old", origin: .existingTimeUnknown), version: 3)
        XCTAssertNil(tracker.automaticContext(currentVersion: 3))
        XCTAssertTrue(tracker.changed(to: 4))
        let fresh = entry("new")
        tracker.accepted(fresh, version: 4)
        XCTAssertEqual(tracker.automaticContext(currentVersion: 4)?.entry.id, fresh.id)
        XCTAssertNil(tracker.automaticContext(currentVersion: 5))
        tracker.reset(); tracker.begin(version: 5)
        XCTAssertNil(tracker.automaticContext(currentVersion: 5))
        // Repeated reads and pastes have no effect on this update version.
        XCTAssertFalse(tracker.changed(to: 5))
    }
    func testStaleInFlightReadCannotReplaceNewerContentAndDeleteInvalidatesContext() throws {
        var tracker = ClipboardObservationTracker()
        tracker.begin(version: 0)
        XCTAssertTrue(tracker.changed(to: 1)); XCTAssertTrue(tracker.changed(to: 2))
        tracker.accepted(entry("stale"), version: 1)
        XCTAssertNil(tracker.latest)
        let fresh = entry("latest"); tracker.accepted(fresh, version: 2)
        tracker.remove(id: fresh.id)
        XCTAssertNil(tracker.automaticContext(currentVersion: 2))
    }
    func testHistoryPreservesAllBytesAcrossReopenAndPagingDoesNotLimitRetention() throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: root) }
        let url = root.appendingPathComponent("clipboard/history-v1.sqlite")
        var firstID = ""
        do {
            let store = try ClipboardHistoryStore(url: url)
            for i in 0..<130 {
                let text = "entry \(i) %_" + String(repeating: " longer content", count: i)
                let row = entry(text)
                if i == 0 { firstID = row.id }
                try store.insert(row)
            }
            XCTAssertEqual(try store.count(), 130)
            XCTAssertEqual(try store.summaries().count, 100)
            XCTAssertEqual(try store.summaries(offset: 100).count, 30)
            XCTAssertEqual(try store.summaries(search: "%_").count, 100)
            try store.setPinned(id: firstID, pinned: true)
            XCTAssertEqual(try store.summaries().first?.id, firstID)
            store.flush()
        }
        let reopened = try ClipboardHistoryStore(url: url)
        XCTAssertEqual(try reopened.count(), 130)
        XCTAssertTrue(try XCTUnwrap(reopened.entry(id: firstID)).isPinned)
        let newest = try XCTUnwrap(reopened.summaries().first { !$0.isPinned })
        let saved = try XCTUnwrap(reopened.entry(id: newest.id))
        XCTAssertEqual(saved.items[0].representations[0].data, Data(try XCTUnwrap(saved.text).utf8))
        XCTAssertEqual(saved.text?.components(separatedBy: " longer content").count, 130)
        try reopened.delete(id: firstID); XCTAssertNil(try reopened.entry(id: firstID))
        try reopened.clear(); XCTAssertEqual(try reopened.count(), 0)
        let permissions = try FileManager.default.attributesOfItem(atPath: url.path)[.posixPermissions] as? Int
        XCTAssertEqual(permissions, 0o600)
    }
    func testExplicitOldSelectionKeepsUnknownTimeAndNeverClaimsSuccessfulUse() {
        let context = ClipboardRecommendationContext(entry: entry("project-alpha", origin: .existingTimeUnknown), explicitlySelected: true)
        XCTAssertTrue(context.modelEvidence?.contains("explicitly selected") == true)
        XCTAssertTrue(context.modelEvidence?.contains("unknown") == true)
        let automatic = ClipboardRecommendationContext(entry: entry("project-alpha"), explicitlySelected: false)
        XCTAssertTrue(automatic.modelEvidence?.contains("does not prove") == true)
    }
}
