import XCTest
@testable import IntentCore

final class OnlinePathFrequencyIndexTests: XCTestCase {
    func testExactInclusiveWindowsAndClockChangesMatchArrivalReplay() {
        var index = OnlinePathFrequencyIndex()
        var arrivals: [(timestamp: Int64, exact: String, layers: [String])] = []
        let timestamps: [Int64] = [0, 300_000, 300_001, 3_600_000, 86_400_000,
                                   86_400_001, 200, 0, 3_600_001, 170_000_000]
        for number in 0..<500 {
            let exact = number % 2 == 0 ? "leaf:a" : "leaf:b"
            let layers = [exact, "parent", "parent"]
            let timestamp = timestamps[number % timestamps.count]
            index.append(timestamp: timestamp, exact: exact, layers: layers)
            arrivals.append((timestamp, exact, layers))
            for cutoff in timestamps + [-1, 300_002, 86_400_200, 172_800_000] {
                for key in ["leaf:a", "leaf:b", "parent", "unknown"] {
                    let included = arrivals.filter { $0.timestamp <= cutoff && $0.layers.contains(key) }
                    let windows: [Int64?] = [300_000, 3_600_000, 86_400_000, nil]
                    let path = windows.map { window in
                        included.filter { window == nil || cutoff - $0.timestamp <= window! }.count
                    }
                    let direct = windows.map { window in
                        included.filter { $0.exact == key && (window == nil || cutoff - $0.timestamp <= window!) }.count
                    }
                    let actual = index.counts(for: key, at: cutoff)
                    XCTAssertEqual(actual.path, path)
                    XCTAssertEqual(actual.direct, direct)
                }
            }
        }
    }
}
