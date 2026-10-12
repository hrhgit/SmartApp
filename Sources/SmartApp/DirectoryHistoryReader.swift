import Foundation
import IntentCore
import FileRecallCore

struct ObservedRecallFile: Equatable, Identifiable, Sendable {
    let target: Destination
    let lastObservedAtMs: Int64
    var id: String { target.id }
}

struct DirectoryRecallHistory: Sendable {
    let candidates: [DirectoryRecommendation]
    let observedFiles: [ObservedRecallFile]
}

/// Only confirmed foreground arrivals enter FileRecall. A catalog update alone
/// is not a visit, and later metadata is never backfilled into earlier events.
struct DirectoryHistoryReplay {
    private var session: String?
    private var enabled = false
    private var gate = TargetEventGate()
    private var metadata: [String: Destination] = [:]
    private var visit: String?
    private var previous: (target: String, entrance: String?)?
    private var resolutionRoot: (context: String, visit: String)?
    private(set) var files: [String: ObservedRecallFile] = [:]
    private var nowMs: Int64
    private let rules: [String]
    private var accumulator: StatLexicalDirectoryCandidateAccumulator

    init(context: PageContext, refinement: String, nowMs: Int64 = Int64(Date().timeIntervalSince1970 * 1_000), rules: [String] = []) throws {
        self.nowMs = nowMs
        self.rules = rules; gate.rules = rules
        accumulator = try StatLexicalDirectoryCandidateEngine().makeAccumulator(context: context, refinement: refinement, nowMs: nowMs)
    }
    mutating func consume(_ event: BehaviorEvent) throws {
        if event.sessionID != session || event.kind == .sessionStart {
            session = event.sessionID; gate.reset(); metadata = [:]; visit = nil; previous = nil; resolutionRoot = nil
            enabled = event.kind == .sessionStart &&
                event.data["stateCoverageVersion"]?.string == StateCoverage.foregroundOnlyVersion &&
                event.data["targetEligibilityPolicyVersion"]?.string.map(TargetEligibilityPolicy.supportedVersions.contains) == true
        }
        guard enabled, event.timestampMs <= nowMs, gate.accept(event) else { return }
        if [.captureGap, .observationScope, .sessionEnd].contains(event.kind) {
            previous = nil; visit = nil
            if event.kind == .sessionEnd { resolutionRoot = nil }
            return
        }
        if event.kind == .targetUpdated, let target = try event.data["target"]?.decoded(Destination.self) {
            metadata[target.id] = target; return
        }
        guard event.kind == .contextChanged, let id = event.data["targetID"]?.string else { return }
        let entrance = event.data["entranceID"]?.string
        let isResolution = event.data["reason"]?.string == "resolution_update"
        let same = previous?.target == id && previous?.entrance == entrance
        if isResolution, let resolved = event.data["resolvesContextEventID"]?.string,
           let root = resolutionRoot, root.context == resolved {
            visit = root.visit
        } else if !same && !isResolution { visit = event.id }
        if visit == nil { visit = event.id }
        if let visit { resolutionRoot = (event.id, visit) }
        previous = (id, entrance)
        guard let target = metadata[id], TargetEligibilityPolicy.validates(target, rules: rules),
              let raw = target.locator, let url = URL(string: raw), url.isFileURL,
              [.directory, .file].contains(target.kind) else { return }
        let parent = url.standardizedFileURL.deletingLastPathComponent()
        let file = FileDestination(id: target.id, kind: target.kind == .file ? .file : .directory,
            name: target.name, locator: raw,
            parentDirectoryID: target.kind == .file ? target.parentID : nil,
            parentDirectoryLocator: target.kind == .file ? parent.absoluteString : nil, fileType: target.fileType)
        accumulator.consume(.init(id: event.id, kind: target.kind == .file ? .observedFile : .observedDirectory,
            visitID: visit, observedAtMs: event.timestampMs, source: event.source,
            applicationBundleID: event.app?.bundleID, target: file))
        if target.kind == .file { files[target.id] = .init(target: target, lastObservedAtMs: event.timestampMs) }
    }
    func result() throws -> DirectoryRecallHistory {
        .init(candidates: try accumulator.candidates(), observedFiles: files.values.sorted {
            $0.lastObservedAtMs == $1.lastObservedAtMs ? $0.id < $1.id : $0.lastObservedAtMs > $1.lastObservedAtMs
        })
    }
}

enum DirectoryHistoryReader {
    static func recall(databaseURL: URL, context: PageContext, refinement: String, nowMs: Int64, rules: [String] = []) throws -> DirectoryRecallHistory {
        let reader = try EventStore(url: databaseURL, readOnly: true)
        var replay = try DirectoryHistoryReplay(context: context, refinement: refinement, nowMs: nowMs, rules: rules)
        try reader.replayRecommendationHistory(seedThroughEventID: nil) { _, event in
            try Task.checkCancellation(); try replay.consume(event)
        }
        return try replay.result()
    }
}
