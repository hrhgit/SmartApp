import Foundation

/// Native equivalent of the eligible similarity tags in target-metadata-semantics-v1.
/// Never reads a file, page body, global catalog, or future observation.
public enum SemanticMetadata {
    public static func annotate(_ target: Destination) -> SemanticProfile {
        var tags: [String: SemanticProfile.Tag] = [:]
        func add(_ kind: String, _ value: Any, _ strength: Double, _ confidence: Double) {
            guard let data = try? JSONSerialization.data(withJSONObject: value, options: [.fragmentsAllowed, .sortedKeys, .withoutEscapingSlashes]) else { return }
            let key = kind + ":" + String(decoding: data, as: UTF8.self)
            if (tags[key].map { $0.strength * $0.confidence } ?? -1) <= strength * confidence {
                tags[key] = .init(key: key, kind: kind, strength: strength, confidence: confidence)
            }
        }
        let kind = target.kind.rawValue, bundle: Any = target.bundleID as Any? ?? NSNull()
        var parts: [String]?, parsed: URLComponents?
        if let locator = target.locator { parsed = URLComponents(string: locator) }
        else if let origin = target.origin { parsed = URLComponents(string: origin) }
        let scheme = parsed?.scheme?.lowercased() ?? ""
        let host = parsed?.host?.lowercased() ?? ""
        let pairs = query(parsed?.percentEncodedQuery)
        let accounts: [[String]] = pairs.filter { SemanticMetadataRules.accounts.contains($0[0].lowercased()) }
            .map { [$0[0].lowercased(), $0[1]] }.sorted(by: lexical)
        let root = scheme + "://" + host + (parsed?.port.map { ":\($0)" } ?? "")
        func scopes(_ namespace: [Any], _ path: [String], local: Bool, directory: Bool) {
            if path.count > 1 {
                for i in 0..<(path.count - 1) {
                    let part = path[i].lowercased(), identifier = path[i + 1]
                    let inRoot = local && ["users", "home", "volumes", "mnt"].contains(path[0].lowercased()) && i < 2
                    if !inRoot, let scope = SemanticMetadataRules.scopeMarkers[part], !SemanticMetadataRules.generic.contains(identifier.lowercased()) {
                        add(scope, [namespace, Array(path.prefix(i + 2))], 0.94, 0.92)
                    }
                    if local && ["school", "classes", "coursework"].contains(part), matches(identifier, "^[A-Za-z]{2,8}[-_ ]?[0-9]{2,4}[A-Za-z]?$") {
                        add("course_scope", [namespace, Array(path.prefix(i + 2))], 0.90, 0.88)
                    }
                }
            }
            let end = local && directory ? path.count : path.count - 1
            if end > 0 {
                for depth in 1...end {
                    let prefix = Array(path.prefix(depth))
                    if local {
                        let minimum = path.first.map { ["users", "home", "volumes", "mnt"].contains($0.lowercased()) } == true ? 3 : 2
                        add("local_directory_scope", [namespace, prefix], depth >= minimum && specific(prefix.last!) ? 0.88 : 0.12, 0.96)
                    } else {
                        add("url_parent", [namespace, prefix], depth >= 2 && specific(prefix.last!) ? 0.65 : 0.12, 0.90)
                    }
                }
            }
        }
        if let locator = target.locator {
            if ["http", "https"].contains(scheme), !host.isEmpty {
                add("site", root, 0.18, 0.99)
                let namespace: [Any] = ["web", root, accounts]
                parts = pathParts(parsed?.percentEncodedPath ?? "")
                if let parts {
                    scopes(namespace, parts, local: false, directory: false)
                    if SemanticMetadataRules.repositoryHosts.contains(host), parts.count >= 2,
                       !SemanticMetadataRules.repositoryReserved.contains(parts[0].lowercased()), specific(parts[0]), specific(parts[1]),
                       host != "gitlab.com" || parts.count == 2 || parts.contains("-") {
                        let end = host == "gitlab.com" ? (parts.firstIndex(of: "-") ?? 2) : 2
                        add("repository_scope", [namespace, Array(parts.prefix(end))], 0.96, 0.95)
                    }
                }
                var grouped: [String: Set<String>] = [:]
                for pair in pairs {
                    if let scope = SemanticMetadataRules.queryScopes[pair[0].lowercased()], !pair[1].isEmpty { grouped[scope, default: []].insert(pair[1]) }
                }
                for (scope, values) in grouped where values.count == 1 {
                    add(scope, [namespace, "explicit_query_scope", parsed?.percentEncodedPath ?? "", values.first!], 0.91, 0.88)
                }
                if let fragment = parsed?.percentEncodedFragment, fragment.hasPrefix("/"),
                   let route = URLComponents(string: fragment), let fragmentParts = pathParts(route.percentEncodedPath) {
                    let fragmentAccounts = query(route.percentEncodedQuery).filter { SemanticMetadataRules.accounts.contains($0[0].lowercased()) }
                        .map { [$0[0].lowercased(), $0[1]] }.sorted(by: lexical)
                    scopes(["fragment_route", namespace, parsed?.percentEncodedPath ?? "", fragmentAccounts], fragmentParts, local: false, directory: false)
                }
            } else if scheme == "file" || scheme.isEmpty && locator.hasPrefix("/") {
                parts = pathParts(scheme.isEmpty ? locator : parsed?.percentEncodedPath ?? "")
                if let parts, target.kind == .directory || target.kind == .file {
                    scopes(["local", host], parts, local: true, directory: target.kind == .directory)
                }
            }
        } else if ["http", "https"].contains(scheme), !host.isEmpty { add("site", root, 0.18, 0.95) }
        if let parent = target.parentID { add("observed_parent_id", parent, 0.22, 0.95) }
        let nameNamespace: [Any]
        if parsed == nil { nameNamespace = ["application_metadata", bundle, kind] }
        else if ["http", "https"].contains(scheme) {
            var owner: [String] = []
            let path = parts ?? []
            if path.count > 1 {
                for i in 0..<(path.count - 1) where ["users", "user", "u", "accounts", "account", "tenants", "organizations", "orgs", "workspaces"].contains(path[i].lowercased()) {
                    owner = Array(path.prefix(i + 2)); break
                }
            }
            if SemanticMetadataRules.repositoryHosts.contains(host), !path.isEmpty { owner = Array(path.prefix(1)) }
            nameNamespace = ["web", scheme, host, parsed?.port as Any? ?? NSNull(), accounts, owner]
        } else if scheme == "file" || scheme.isEmpty && (parsed?.path.hasPrefix("/") ?? false) {
            let path = parts ?? []
            let depth = path.first.map { ["users", "home", "volumes", "mnt"].contains($0.lowercased()) } == true ? 2 : 1
            nameNamespace = ["local", host, Array(path.prefix(depth))]
        } else { nameNamespace = ["application_metadata", bundle, kind] }
        if let filename = parts?.last { for token in tokens(filename) { add("name_token", [nameNamespace, token], 0.18, 0.45) } }
        if !target.name.contains("://") && !target.name.hasPrefix("/") {
            for token in tokens(target.name) { add("name_token", [nameNamespace, token], 0.18, 0.45) }
        }
        return SemanticProfile(identity: .init(id: target.id, kind: kind, bundleID: target.bundleID, locator: target.locator), tags: tags.keys.sorted().map { tags[$0]! })
    }
    private static func lexical(_ a: [String], _ b: [String]) -> Bool { a.lexicographicallyPrecedes(b) }
    private static func query(_ raw: String?) -> [[String]] {
        guard let raw else { return [] }
        return raw.split(separator: "&").prefix(512).map { field in
            let pair = field.split(separator: "=", maxSplits: 1, omittingEmptySubsequences: false)
            return [String(pair[0]), pair.count > 1 ? String(pair[1]) : ""].map { $0.replacingOccurrences(of: "+", with: " ").removingPercentEncoding ?? $0 }
        }
    }
    private static func pathParts(_ value: String) -> [String]? {
        let parts = value.split(separator: "/").map { String($0).removingPercentEncoding ?? String($0) }
        guard !parts.contains(where: { [".", ".."].contains($0) || $0.contains("/") || $0.contains("\\") }) else { return nil }
        return parts
    }
    private static func matches(_ value: String, _ pattern: String) -> Bool { value.range(of: pattern, options: .regularExpression) != nil }
    private static func opaque(_ value: String) -> Bool { matches(value, "^(?:[0-9]+|[0-9a-fA-F]{12,}|[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12})$") }
    private static func specific(_ value: String) -> Bool { !value.isEmpty && !SemanticMetadataRules.generic.contains(value.lowercased()) && !opaque(value) }
    private static func tokens(_ name: String) -> [String] {
        let value = name.replacingOccurrences(of: "([a-z])([A-Z])", with: "$1 $2", options: .regularExpression).lowercased()
        guard let regex = try? NSRegularExpression(pattern: "[^\\W_]+") else { return [] }
        let matches = regex.matches(in: value, range: NSRange(value.startIndex..., in: value)).compactMap { match -> String? in
            guard let range = Range(match.range, in: value) else { return nil }
            let token = String(value[range])
            return (2...40).contains(token.count) && !SemanticMetadataRules.generic.contains(token) && SemanticMetadataRules.roles[token] == nil && !opaque(token) ? token : nil
        }
        return Array(Set(matches).sorted().prefix(16))
    }
}
