import Foundation
import IntentCore
import Darwin

@main
enum CLI {
    static func main() {
        do {
            var arguments = Array(CommandLine.arguments.dropFirst())
            let command = arguments.first ?? "help"
            if !arguments.isEmpty { arguments.removeFirst() }
            var database = EventStore.contextURL
            if let index = arguments.firstIndex(of: "--database") {
                guard index + 1 < arguments.count else { throw StoreError("--database requires a path") }
                database = URL(fileURLWithPath: arguments[index + 1])
                arguments.removeSubrange(index...index + 1)
            }
            switch command {
            case "validate-model":
                guard arguments.count == 4, arguments[0] == "--input", arguments[2] == "--output" else {
                    throw StoreError("Usage: smartapp-cli validate-model --input PRIVATE_FIXTURE --output NEW_PRIVATE_REPORT")
                }
                let output = URL(fileURLWithPath: arguments[3])
                guard !FileManager.default.fileExists(atPath: output.path) else { throw StoreError("Validation output already exists") }
                let fixture = try JSONDecoder().decode(SemanticModelValidation.self, from: Data(contentsOf: URL(fileURLWithPath: arguments[1])))
                let result = fixture.run()
                let encoder = JSONEncoder(); encoder.outputFormatting = [.sortedKeys, .withoutEscapingSlashes]
                guard FileManager.default.createFile(atPath: output.path, contents: try encoder.encode(result), attributes: [.posixPermissions: 0o600]) else { throw StoreError("Cannot write model validation") }
                try printJSON(["queries": result["queries"]!, "annotations": result["annotations"]!,
                    "rankingMismatches": .number(Double(result["rankingMismatches"]?.array?.count ?? 0)),
                    "annotationMismatches": .number(Double(result["annotationMismatches"]?.array?.count ?? 0))])
            case "status":
                guard arguments.isEmpty else { throw StoreError("Unexpected arguments") }
                let store = try EventStore(url: database, readOnly: true)
                try printJSON(store.stats())
            case "recent":
                guard arguments.isEmpty else { throw StoreError("Unexpected arguments") }
                let store = try EventStore(url: database, readOnly: true)
                for event in try store.recent(limit: 20).reversed() { try printJSON(event) }
            case "export", "migrate-state":
                guard arguments.count == 2, arguments[0] == "--output" else {
                    throw StoreError("Usage: smartapp-cli export --output NEW_DIRECTORY [--database PATH]")
                }
                let store = try EventStore(url: database, readOnly: true)
                try printJSON(store.export(to: URL(fileURLWithPath: arguments[1], isDirectory: true)))
            case "backup":
                guard arguments.count == 2, arguments[0] == "--output" else { throw StoreError("Usage: smartapp-cli backup --output NEW_DATABASE [--database PATH]") }
                let store = try EventStore(url: database, readOnly: true)
                try store.backup(to: URL(fileURLWithPath: arguments[1]))
                try printJSON(["status": "verified", "path": arguments[1]])
            case "rebuild-foreground":
                guard arguments.count == 2, arguments[0] == "--output" else {
                    throw StoreError("Usage: smartapp-cli rebuild-foreground --output NEW_DATABASE [--database PATH]")
                }
                try printJSON(try ForegroundOnlyDatabaseRebuilder.rebuild(
                    source: database, destination: URL(fileURLWithPath: arguments[1])))
            case "help", "--help", "-h":
                print("""
                SmartApp — local behavior dataset utilities
                status [--database PATH]                      Database counts and size
                validate-model --input PRIVATE_FIXTURE --output NEW_PRIVATE_REPORT   Native model parity check
                recent [--database PATH]                      Last 20 recorded events (JSONL)
                export --output NEW_DIRECTORY [--database PATH]   Snapshot raw events and training samples
                backup --output NEW_DATABASE [--database PATH]    Verified SQLite online backup including WAL
                migrate-state --output NEW_DIRECTORY [--database PATH]   Rebuild v3 state-aware samples; retain raw events
                rebuild-foreground --output NEW_DATABASE [--database PATH]  Build a foreground-only standard database

                Default database: \(EventStore.contextURL.path)
                Legacy v1 database: \(EventStore.defaultURL.path)
                The CLI never starts a collector or requests permissions.
                """)
            default: throw StoreError("Unknown command: \(command). Run smartapp-cli help.")
            }
        } catch {
            FileHandle.standardError.write(Data("smartapp-cli: \(error.localizedDescription)\n".utf8))
            exit(1)
        }
    }

    private static func printJSON<T: Encodable>(_ value: T) throws {
        let encoder = JSONEncoder(); encoder.outputFormatting = [.sortedKeys, .withoutEscapingSlashes]
        print(String(decoding: try encoder.encode(value), as: UTF8.self))
    }
}
