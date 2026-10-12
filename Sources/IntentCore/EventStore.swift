import Foundation
import CSQLite

public struct StoreStats: Codable, Equatable, Sendable {
    public var events: Int = 0
    public var userActions: Int = 0
    public var sessions: Int = 0
    public var applications: Int = 0
    public var annotations: Int = 0
    public var bytes: Int64 = 0
    public var targets: Int = 0
    public init() {}
}

public struct ExportReport: Codable, Equatable, Sendable {
    public var schemaVersion = 1
    public var targetEligibilityPolicyVersion: String?
    public var eventsExcludedFromSamples: Int?
    public var sampleSchemaVersion: Int?
    public var stateCoverageVersion: String?
    public var inputCoverageCounts: [String: Int]?
    public var outcomeCoverageCounts: [String: Int]?
    public var createdAtMs = Int64(Date().timeIntervalSince1970 * 1_000)
    public var events: Int = 0
    public var samples: Int = 0
    public var switchSamples: Int = 0
    public var staySamples: Int = 0
    public var annotations: Int = 0
    public var completeSamples: Int = 0
    public var partialSamples: Int = 0
    public var targetRecords: Int = 0
    public var lookbackMs = SampleBuilder.lookbackMs
    public var predictionHorizonMs = SampleBuilder.horizonMs
    public var notes = [
        "Labels are temporal associations, not proof of causation or user intent.",
        "Features contain only input events at or before each trigger. Future app activations are labels only.",
        "__stay__ means no observed app activation in the complete following 2 seconds.",
        "Incomplete horizons, session boundaries, and capture gaps are omitted.",
        "Split train/test chronologically by session or day. Overlapping windows must not cross splits.",
        "Gesture capture is best effort. Continuous scrolling does not uniquely identify a trackpad.",
        "Text key identity, input text, clipboard, window titles, URLs and screenshots are not collected."
    ]
}

public struct StoreError: LocalizedError, Sendable {
    public var message: String
    public var errorDescription: String? { message }
    public init(_ message: String) { self.message = message }
}

/// A serial writer. Exports use an independent read-only connection and a WAL snapshot.
public final class EventStore: @unchecked Sendable {
    public let url: URL
    private var db: OpaquePointer?
    private let readOnly: Bool
    private let queue = DispatchQueue(label: "local.smartapp.database", qos: .utility)
    private let encoder: JSONEncoder
    private let decoder = JSONDecoder()
    private var activityCounter = UserActivityCounter()
    private var activitySequence: Int64 = 0

    public static var defaultURL: URL {
        FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0]
            .appendingPathComponent("SmartApp", isDirectory: true)
            .appendingPathComponent("events.sqlite")
    }
    public static var contextURL: URL {
        defaultURL.deletingLastPathComponent().appendingPathComponent("context-v2.sqlite")
    }

    public init(url: URL, readOnly: Bool = false) throws {
        self.url = url
        self.readOnly = readOnly
        encoder = JSONEncoder(); encoder.outputFormatting = [.sortedKeys, .withoutEscapingSlashes]
        if !readOnly {
            try FileManager.default.createDirectory(at: url.deletingLastPathComponent(),
                                                    withIntermediateDirectories: true,
                                                    attributes: [.posixPermissions: 0o700])
        }
        let flags = readOnly ? SQLITE_OPEN_READONLY : (SQLITE_OPEN_READWRITE | SQLITE_OPEN_CREATE)
        guard sqlite3_open_v2(url.path, &db, flags | SQLITE_OPEN_FULLMUTEX, nil) == SQLITE_OK else {
            let error = StoreError("Cannot open database: \(url.path)")
            if db != nil { sqlite3_close(db); db = nil }
            throw error
        }
        do {
            sqlite3_busy_timeout(db, 5_000)
            let version = try scalar("PRAGMA user_version")
            guard version <= 1 else { throw StoreError("Database schema is newer than this app; upgrade SmartApp.") }
            if !readOnly {
                try execute("PRAGMA journal_mode=WAL; PRAGMA synchronous=NORMAL;")
                try execute("""
                    CREATE TABLE IF NOT EXISTS events (
                        seq INTEGER PRIMARY KEY AUTOINCREMENT,
                        id TEXT NOT NULL UNIQUE,
                        session_id TEXT NOT NULL,
                        timestamp_ms INTEGER NOT NULL,
                        monotonic_ms REAL NOT NULL,
                        kind TEXT NOT NULL,
                        app_bundle_id TEXT,
                        json TEXT NOT NULL
                    );
                    CREATE INDEX IF NOT EXISTS events_session_time ON events(session_id, monotonic_ms, seq);
                    CREATE INDEX IF NOT EXISTS events_kind ON events(kind);
                    CREATE INDEX IF NOT EXISTS events_app ON events(app_bundle_id);
                    PRAGMA user_version=1;
                    """)
                try FileManager.default.setAttributes([.posixPermissions: 0o600], ofItemAtPath: url.path)
            }
        } catch {
            sqlite3_close(db); db = nil; throw error
        }
    }

    deinit { sqlite3_close(db) }

    public func append(_ events: [BehaviorEvent]) throws {
        try queue.sync { _ = try insert(events) }
    }

    public func appendWithSequences(_ events: [BehaviorEvent]) throws -> [Int64] {
        try queue.sync { try insert(events) }
    }

    public func appendAsync(_ events: [BehaviorEvent], completion: @escaping @Sendable (Result<Void, StoreError>) -> Void) {
        appendWithSequencesAsync(events) { completion($0.map { _ in () }) }
    }

    /// Positions are returned only after the entire batch commits successfully.
    public func appendWithSequencesAsync(_ events: [BehaviorEvent], completion: @escaping @Sendable (Result<[Int64], StoreError>) -> Void) {
        queue.async {
            do { completion(.success(try self.insert(events))) }
            catch { completion(.failure(StoreError(error.localizedDescription))) }
        }
    }

    public func flush() { queue.sync {} }

    /// The live writer must not serialize a filesystem-size check behind queued writes.
    public static func databaseBytes(at url: URL) -> Int64 {
        ["", "-wal", "-shm"].reduce(into: Int64(0)) { total, suffix in
            let attributes = try? FileManager.default.attributesOfItem(atPath: url.path + suffix)
            total += (attributes?[.size] as? NSNumber)?.int64Value ?? 0
        }
    }

    /// SQLite online backup includes committed WAL content. Never copy only the live .sqlite file.
    public func backup(to destination: URL) throws {
        guard !FileManager.default.fileExists(atPath: destination.path) else { throw StoreError("Backup already exists") }
        try FileManager.default.createDirectory(at: destination.deletingLastPathComponent(), withIntermediateDirectories: true,
                                              attributes: [.posixPermissions: 0o700])
        // Create with private permissions before SQLite opens it.
        guard FileManager.default.createFile(atPath: destination.path, contents: nil, attributes: [.posixPermissions: 0o600]) else {
            throw StoreError("Cannot create backup")
        }
        return try queue.sync {
            var target: OpaquePointer?
            guard sqlite3_open_v2(destination.path, &target, SQLITE_OPEN_READWRITE | SQLITE_OPEN_FULLMUTEX, nil) == SQLITE_OK else {
                if target != nil { sqlite3_close(target) }; throw StoreError("Cannot open backup")
            }
            defer { sqlite3_close(target) }
            guard let backup = sqlite3_backup_init(target, "main", db, "main") else { throw StoreError("Cannot initialize backup") }
            let deadline = Date().addingTimeInterval(30)
            var result: Int32
            repeat {
                result = sqlite3_backup_step(backup, 256)
                if result == SQLITE_BUSY || result == SQLITE_LOCKED { Thread.sleep(forTimeInterval: 0.01) }
            } while [SQLITE_OK, SQLITE_BUSY, SQLITE_LOCKED].contains(result) && Date() < deadline
            let finished = sqlite3_backup_finish(backup)
            guard result == SQLITE_DONE, finished == SQLITE_OK else { throw StoreError("Backup incomplete; retained for inspection, do not restore it") }
            var check: OpaquePointer?
            guard sqlite3_prepare_v2(target, "PRAGMA quick_check", -1, &check, nil) == SQLITE_OK else { throw StoreError("Cannot verify backup") }
            defer { sqlite3_finalize(check) }
            guard sqlite3_step(check) == SQLITE_ROW, let value = sqlite3_column_text(check, 0), String(cString: value) == "ok" else {
                throw StoreError("Backup integrity verification failed")
            }
        }
    }

    public func stats() throws -> StoreStats {
        if !readOnly { return try EventStore(url: url, readOnly: true).stats() }
        return try queue.sync {
            var stats = StoreStats()
            stats.events = try scalar("SELECT COUNT(*) FROM events")
            let activity = try prepare("SELECT json, seq FROM events WHERE seq > \(activitySequence) ORDER BY seq")
            defer { sqlite3_finalize(activity) }
            while try step(activity) == SQLITE_ROW {
                try autoreleasepool {
                    activityCounter.consume(try decode(activity))
                    activitySequence = sqlite3_column_int64(activity, 1)
                }
            }
            stats.userActions = activityCounter.count
            stats.sessions = try scalar("SELECT COUNT(DISTINCT session_id) FROM events")
            stats.applications = try scalar("SELECT COUNT(DISTINCT app_bundle_id) FROM events WHERE app_bundle_id IS NOT NULL")
            stats.annotations = try scalar("SELECT COUNT(*) FROM events WHERE kind='annotation'")
            stats.targets = try scalar("SELECT COUNT(*) FROM events WHERE kind='target.update'")
            stats.bytes = Self.databaseBytes(at: url)
            return stats
        }
    }

    public func recent(limit: Int = 80) throws -> [BehaviorEvent] {
        if !readOnly { return try EventStore(url: url, readOnly: true).recent(limit: limit) }
        return try queue.sync {
            let statement = try prepare("SELECT json FROM events ORDER BY seq DESC LIMIT \(max(1, min(limit, 1_000)))")
            defer { sqlite3_finalize(statement) }
            var events: [BehaviorEvent] = []
            while try step(statement) == SQLITE_ROW { events.append(try decode(statement)) }
            return events
        }
    }

    /// Start at the seed session boundary so pending observations stay causal.
    public func recommendationHistory(seedThroughEventID: String?) throws -> [BehaviorEvent] {
        var result: [BehaviorEvent] = []
        try replayRecommendationHistory(seedThroughEventID: seedThroughEventID) { _, event in result.append(event) }
        return result
    }

    /// Replays a database snapshot in write order without retaining the full history in memory.
    public func replayRecommendationHistory(seedThroughEventID: String?, consume: (Int64, BehaviorEvent) throws -> Void) throws {
        if !readOnly {
            try EventStore(url: url, readOnly: true).replayRecommendationHistory(seedThroughEventID: seedThroughEventID, consume: consume)
            return
        }
        try queue.sync {
            var firstSequence: Int64 = 0
            if let id = seedThroughEventID {
                let lookup = try prepare("SELECT MIN(seq) FROM events WHERE session_id=(SELECT session_id FROM events WHERE id=?)")
                defer { sqlite3_finalize(lookup) }
                sqlite3_bind_text(lookup, 1, id, -1, unsafeBitCast(-1, to: sqlite3_destructor_type.self))
                guard try step(lookup) == SQLITE_ROW, sqlite3_column_type(lookup, 0) != SQLITE_NULL else {
                    throw StoreError("Recommendation seed is not part of this database")
                }
                firstSequence = sqlite3_column_int64(lookup, 0)
            }
            let statement = try prepare("SELECT json, seq FROM events WHERE seq >= \(firstSequence) ORDER BY seq")
            defer { sqlite3_finalize(statement) }
            while try step(statement) == SQLITE_ROW {
                // Startup may replay years of history in one queue block.
                // Release Foundation's temporary JSON objects after each row
                // while keeping the SQLite cursor and caller state intact.
                try autoreleasepool {
                    let event = try decode(statement)
                    try consume(sqlite3_column_int64(statement, 1), event)
                }
            }
        }
    }

    /// Stream only committed positions on a distinct reader connection. The
    /// upper bound freezes this drain while the writer continues appending;
    /// queued learning work needs only that position, never an event payload.
    public func replayCommitted(after sequence: Int64, through lastSequence: Int64,
                                consume: (Int64, BehaviorEvent) throws -> Void) throws {
        guard lastSequence > sequence else { return }
        if !readOnly {
            try EventStore(url: url, readOnly: true).replayCommitted(after: sequence, through: lastSequence, consume: consume)
            return
        }
        try queue.sync {
            let statement = try prepare("SELECT json, seq FROM events WHERE seq > ? AND seq <= ? ORDER BY seq")
            defer { sqlite3_finalize(statement) }
            sqlite3_bind_int64(statement, 1, sequence)
            sqlite3_bind_int64(statement, 2, lastSequence)
            while try step(statement) == SQLITE_ROW {
                try autoreleasepool { try consume(sqlite3_column_int64(statement, 1), decode(statement)) }
            }
        }
    }

    public func export(to directory: URL) throws -> ExportReport {
        // A distinct connection prevents a large export from delaying input-event writes.
        let reader = try EventStore(url: url, readOnly: true)
        return try reader.queue.sync {
            let first = try reader.prepare("SELECT json FROM events ORDER BY seq LIMIT 1")
            defer { sqlite3_finalize(first) }
            if try reader.step(first) == SQLITE_ROW, try reader.decode(first).schemaVersion >= 2 {
                return try reader.exportContextSnapshot(to: directory)
            }
            return try reader.exportSnapshot(to: directory)
        }
    }

    public func knownTargets() throws -> [Destination] {
        if !readOnly { return try EventStore(url: url, readOnly: true).knownTargets() }
        return try queue.sync {
            let statement = try prepare("SELECT json FROM events WHERE kind='target.update' ORDER BY seq")
            defer { sqlite3_finalize(statement) }
            var targets: [String: Destination] = [:]
            while try step(statement) == SQLITE_ROW {
                try autoreleasepool {
                    let event = try decode(statement)
                    if let value = event.data["target"], let target = try? value.decoded(Destination.self) {
                        targets[target.id] = target
                    }
                }
            }
            return targets.values.sorted { $0.id < $1.id }
        }
    }

    private func exportContextSnapshot(to directory: URL) throws -> ExportReport {
        guard !FileManager.default.fileExists(atPath: directory.path) else {
            throw StoreError("Export directory already exists. Choose a new directory.")
        }
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true,
                                                attributes: [.posixPermissions: 0o700])
        var succeeded = false
        defer { if !succeeded { try? FileManager.default.removeItem(at: directory) } }
        let raw = try outputFile(directory.appendingPathComponent("events.jsonl"))
        let samples = try outputFile(directory.appendingPathComponent("navigation-samples.jsonl"))
        let targets = try outputFile(directory.appendingPathComponent("target-history.jsonl"))
        defer { try? raw.close(); try? samples.close(); try? targets.close() }
        try execute("BEGIN")
        defer { try? execute("ROLLBACK") }
        let statement = try prepare("SELECT json FROM events ORDER BY seq")
        defer { sqlite3_finalize(statement) }
        var report = ExportReport()
        report.schemaVersion = 2; report.lookbackMs = 300_000
        report.sampleSchemaVersion = 4; report.stateCoverageVersion = StateCoverage.foregroundOnlyVersion
        report.inputCoverageCounts = [:]; report.outcomeCoverageCounts = [:]
        report.targetEligibilityPolicyVersion = TargetEligibilityPolicy.version
        report.predictionHorizonMs = NavigationSampleBuilder.horizonMs
        report.notes = [
            "Context v2. Legacy input datasets remain in events.sqlite.",
            "Raw events and target history are archives, not a prefiltered training view. Sessions without a supported admission policy are excluded from samples; each sample retains its session policy version.",
            "Rejected or unknown target references break training history; target admission never uses future observations.",
            "Events use observation time, session sequence and monotonic time. Polling can miss short transitions.",
            "New samples reference decision-time foreground arrivals only; historic inventory events remain raw archive data and are never added to new inputs.",
            "Activity references cover at most 300 context events in the preceding hour; raw logs remain continuous.",
            "Incomplete horizons are retained; no final intent or __stay__ label is guessed.",
            "Menu suggestions can use a versioned online semantic model; actual selections/execution retain the model version. Related navigation is marked separately; displayed lists are not collected.",
            "New captures preserve full web URLs (path, query, fragment; no embedded credentials) and observed Finder/QSpace Pro directory locators in raw data and target history. Legacy hashed pages may have no locator until observed again.",
            "No pointer trajectories, keyboard events, clipboard text, page bodies, screenshots or media content.",
            "New capture never reads, transmits, persists or exports background application, window, tab or pane state. Respect foreground source status, gaps and observation intervals.",
            "Sample v4 separates inputCoverage from outcomeCoverage; foreground_only marks the intentional no-background boundary, and observed_horizon alone does not establish complete coverage.",
            "known-inactivity-v1 pause pairs preserve sequence only after same-target revalidation. Active horizons/dwell exclude the paired interval; raw timestamps and input cutoffs stay unchanged. Unpaired/legacy gaps remain interruptions.",
            "excluded-foreground-v1 observation.scope markers contain no target identity. Crossing outcomes are censored; later samples may retain prior observed context with explicit excludedScopeEventIDs. Old unclassified capture gaps remain interruptions.",
            "interruption-cause-v1 capture.gap pairs identify system-unavailable or unconfirmed-foreground boundaries without target identity; system causeSignals distinguish sleep, lock and inactive sessions. Point gaps carry a reason category. These remain interruptions, and legacy gaps are not reclassified.",
            "Legacy sessions have unknown source coverage; old empty lists are unverified, not observed empty desktops. No later state is backfilled.",
            "resolutionEventIDs are future observation refinements, not independent next actions or input features. Hierarchies resolve only against then-known targets and entrances.",
            "Split chronologically; reconstruct historical features from events, never the latest catalog."
        ]
        var builder = EligibleNavigationReplay()
        func writeSamples(_ values: [NavigationSample]) throws {
            for sample in values {
                try writeLine(sample, to: samples); report.samples += 1
                report.inputCoverageCounts?[sample.inputCoverage?.status ?? "unknown", default: 0] += 1
                report.outcomeCoverageCounts?[sample.outcomeCoverage ?? "unknown", default: 0] += 1
                if sample.status == "observed_horizon" { report.completeSamples += 1 }
                else { report.partialSamples += 1 }
            }
        }
        while try step(statement) == SQLITE_ROW {
            try autoreleasepool {
                let event = try decode(statement)
                try writeLine(event, to: raw); report.events += 1
                if event.kind == .targetUpdated {
                    try writeLine(event, to: targets); report.targetRecords += 1
                }
                if event.kind == .annotation { report.annotations += 1 }
                try writeSamples(builder.consume(event))
            }
        }
        try writeSamples(builder.finish())
        report.eventsExcludedFromSamples = builder.excludedEvents
        try raw.synchronize(); try samples.synchronize(); try targets.synchronize()
        let manifest = try outputFile(directory.appendingPathComponent("manifest.json"))
        try manifest.write(contentsOf: encoder.encode(report)); try manifest.close()
        succeeded = true
        return report
    }

    private func exportSnapshot(to directory: URL) throws -> ExportReport {
        guard !FileManager.default.fileExists(atPath: directory.path) else {
            throw StoreError("Export directory already exists. Choose a new directory.")
        }
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true,
                                                attributes: [.posixPermissions: 0o700])
        var succeeded = false
        defer { if !succeeded { try? FileManager.default.removeItem(at: directory) } }
        let raw = try outputFile(directory.appendingPathComponent("events.jsonl"))
        let samples = try outputFile(directory.appendingPathComponent("samples.jsonl"))
        let annotations = try outputFile(directory.appendingPathComponent("annotations.jsonl"))
        defer { try? raw.close(); try? samples.close(); try? annotations.close() }
        try execute("BEGIN")
        defer { try? execute("ROLLBACK") }
        let statement = try prepare("SELECT json FROM events ORDER BY session_id, monotonic_ms, seq")
        defer { sqlite3_finalize(statement) }
        var report = ExportReport()
        var builder = SampleBuilder()
        while try step(statement) == SQLITE_ROW {
            try autoreleasepool {
                let event = try decode(statement)
                try writeLine(event, to: raw)
                report.events += 1
                if event.kind == .annotation {
                    try writeLine(event, to: annotations); report.annotations += 1
                }
                for sample in builder.consume(event) {
                    try writeLine(sample, to: samples); report.samples += 1
                    if sample.targetApp == SampleBuilder.stayLabel { report.staySamples += 1 }
                    else { report.switchSamples += 1 }
                }
            }
        }
        try raw.synchronize(); try samples.synchronize(); try annotations.synchronize()
        let manifest = try outputFile(directory.appendingPathComponent("manifest.json"))
        try manifest.write(contentsOf: encoder.encode(report)); try manifest.close()
        succeeded = true
        return report
    }

    private func insert(_ events: [BehaviorEvent]) throws -> [Int64] {
        guard !events.isEmpty else { return [] }
        try execute("BEGIN IMMEDIATE")
        var committed = false
        defer { if !committed { try? execute("ROLLBACK") } }
        let statement = try prepare("INSERT INTO events(id,session_id,timestamp_ms,monotonic_ms,kind,app_bundle_id,json) VALUES(?,?,?,?,?,?,?)")
        defer { sqlite3_finalize(statement) }
        var sequences: [Int64] = []
        sequences.reserveCapacity(events.count)
        for event in events {
            sqlite3_reset(statement); sqlite3_clear_bindings(statement)
            bind(event.id, at: 1, to: statement); bind(event.sessionID, at: 2, to: statement)
            sqlite3_bind_int64(statement, 3, event.timestampMs)
            sqlite3_bind_double(statement, 4, event.monotonicMs)
            bind(event.kind.rawValue, at: 5, to: statement)
            if let bundleID = event.app?.bundleID { bind(bundleID, at: 6, to: statement) }
            else { sqlite3_bind_null(statement, 6) }
            let json = String(decoding: try encoder.encode(event), as: UTF8.self)
            bind(json, at: 7, to: statement)
            _ = try step(statement)
            sequences.append(sqlite3_last_insert_rowid(db))
        }
        try execute("COMMIT"); committed = true
        return sequences
    }

    private func writeLine<T: Encodable>(_ value: T, to file: FileHandle) throws {
        var data = try encoder.encode(value); data.append(0x0A)
        try file.write(contentsOf: data)
    }

    private func outputFile(_ url: URL) throws -> FileHandle {
        guard FileManager.default.createFile(atPath: url.path, contents: nil,
                                              attributes: [.posixPermissions: 0o600]) else {
            throw StoreError("Cannot create export file: \(url.path)")
        }
        return try FileHandle(forWritingTo: url)
    }

    private func execute(_ sql: String) throws {
        guard sqlite3_exec(db, sql, nil, nil, nil) == SQLITE_OK else { throw failure() }
    }

    private func prepare(_ sql: String) throws -> OpaquePointer {
        var statement: OpaquePointer?
        guard sqlite3_prepare_v2(db, sql, -1, &statement, nil) == SQLITE_OK, let statement else { throw failure() }
        return statement
    }

    private func step(_ statement: OpaquePointer) throws -> Int32 {
        let result = sqlite3_step(statement)
        guard result == SQLITE_ROW || result == SQLITE_DONE else { throw failure() }
        return result
    }

    private func scalar(_ sql: String) throws -> Int {
        let statement = try prepare(sql); defer { sqlite3_finalize(statement) }
        return try step(statement) == SQLITE_ROW ? Int(sqlite3_column_int64(statement, 0)) : 0
    }

    private func bind(_ text: String, at index: Int32, to statement: OpaquePointer) {
        _ = text.withCString { sqlite3_bind_text(statement, index, $0, -1, unsafeBitCast(-1, to: sqlite3_destructor_type.self)) }
    }

    private func decode(_ statement: OpaquePointer) throws -> BehaviorEvent {
        guard let text = sqlite3_column_text(statement, 0) else { throw StoreError("Missing event JSON") }
        return try decoder.decode(BehaviorEvent.self, from: Data(String(cString: text).utf8))
    }

    private func failure() -> StoreError { StoreError(String(cString: sqlite3_errmsg(db))) }
}
