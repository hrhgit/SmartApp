import Foundation

/// An exact cumulative time index, rather than a retained list of full
/// arrivals. Each occurrence is one Int64 timestamp; binary range counts keep
/// the original inclusive windows and exclude future wall times after a clock
/// adjustment. No target, event ID, parent list or raw event is repeated here.
struct OnlinePathFrequencyIndex: Sendable {
    struct Timeline: Sendable {
        var path: [Int64] = []
        var direct: [Int64] = []

        mutating func append(_ timestamp: Int64, direct isDirect: Bool) {
            Self.insert(timestamp, into: &path)
            if isDirect { Self.insert(timestamp, into: &direct) }
        }

        private static func insert(_ timestamp: Int64, into values: inout [Int64]) {
            if values.last.map({ $0 > timestamp }) == true {
                values.insert(timestamp, at: upperBound(values, timestamp))
            } else { values.append(timestamp) }
        }

        static func upperBound(_ values: [Int64], _ value: Int64) -> Int {
            var lower = 0, upper = values.count
            while lower < upper {
                let middle = lower + (upper - lower) / 2
                if values[middle] <= value { lower = middle + 1 } else { upper = middle }
            }
            return lower
        }

        private static func lowerBound(_ values: [Int64], _ value: Int64) -> Int {
            var lower = 0, upper = values.count
            while lower < upper {
                let middle = lower + (upper - lower) / 2
                if values[middle] < value { lower = middle + 1 } else { upper = middle }
            }
            return lower
        }

        func counts(at cutoff: Int64) -> (path: [Int], direct: [Int]) {
            func windows(_ values: [Int64]) -> [Int] {
                let through = Self.upperBound(values, cutoff)
                var result = [Int]()
                for window: Int64 in [300_000, 3_600_000, 86_400_000] {
                    let (begin, overflow) = cutoff.subtractingReportingOverflow(window)
                    result.append(through - (overflow ? 0 : Self.lowerBound(values, begin)))
                }
                result.append(through)
                return result
            }
            return (windows(path), windows(direct))
        }
    }

    private var timelines: [String: Timeline] = [:]

    mutating func append(timestamp: Int64, exact: String, layers: [String]) {
        for key in Set(layers) {
            timelines[key, default: Timeline()].append(timestamp, direct: key == exact)
        }
    }

    func counts(for key: String, at cutoff: Int64) -> (path: [Int], direct: [Int]) {
        timelines[key]?.counts(at: cutoff) ?? ([0, 0, 0, 0], [0, 0, 0, 0])
    }
}
