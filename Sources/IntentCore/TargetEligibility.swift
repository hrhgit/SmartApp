import Foundation

public enum AdmissionDisposition: String, Codable, Sendable { case allowed, excluded, unknown }

/// Only positive, content-free evidence is persisted with a destination.
public struct TargetAdmission: Codable, Equatable, Sendable {
    public var policyVersion: String
    public var basis: String
    public init(basis: String, policyVersion: String = TargetEligibilityPolicy.version) {
        self.basis = basis; self.policyVersion = policyVersion
    }
}

public struct AdmissionDecision: Equatable, Sendable {
    public let disposition: AdmissionDisposition
    public let reason: String
    public var policyVersion: String { TargetEligibilityPolicy.version }
    public var admission: TargetAdmission? {
        disposition == .allowed ? TargetAdmission(basis: reason) : nil
    }
    public var isAllowed: Bool { disposition == .allowed }
    public init(_ disposition: AdmissionDisposition, _ reason: String) {
        self.disposition = disposition; self.reason = reason
    }
}

/// Deliberately not Codable: raw source metadata is transient classification input.
public struct ApplicationEvidence: Sendable {
    public enum Activation: Sendable { case regular, accessory, prohibited, unknown }
    public var bundleID: String?
    public var standaloneApplication: Bool
    public var backgroundOnly: Bool
    public var uiElement: Bool
    public var activation: Activation
    public var independentWindow: Bool
    public init(bundleID: String?, standaloneApplication: Bool, backgroundOnly: Bool = false,
                uiElement: Bool = false, activation: Activation, independentWindow: Bool = false) {
        self.bundleID = bundleID; self.standaloneApplication = standaloneApplication
        self.backgroundOnly = backgroundOnly; self.uiElement = uiElement
        self.activation = activation; self.independentWindow = independentWindow
    }
}

public struct WindowEvidence: Sendable {
    public var role: String?
    public var subrole: String?
    public var modal: Bool?
    public var canRaise: Bool
    public var main: Bool
    public init(role: String?, subrole: String?, modal: Bool?, canRaise: Bool, main: Bool = false) {
        self.role = role; self.subrole = subrole; self.modal = modal
        self.canRaise = canRaise; self.main = main
    }
    public var isSystemDialog: Bool { subrole == "AXSystemDialog" }
}

public enum TargetEligibilityPolicy {
    public static let version = "target-eligibility-v4"
    public static let supportedVersions: Set<String> = ["target-eligibility-v1", "target-eligibility-v2", "target-eligibility-v3", version]
    public static let finderBundleID = "com.apple.finder"
    public static let qspaceBundleID = "com.jinghaoshe.qspace.pro"
    public static func application(_ value: ApplicationEvidence, rules: [String] = []) -> AdmissionDecision {
        guard let bundle = value.bundleID, !bundle.isEmpty else { return .init(.unknown, "missing_bundle") }
        if CapturePolicy.isExcluded(bundle, rules: rules) { return .init(.excluded, "excluded_source") }
        if value.backgroundOnly || value.activation == .prohibited { return .init(.excluded, "background_service") }
        guard value.standaloneApplication else { return .init(.excluded, "non_application_component") }
        if value.activation == .regular && !value.uiElement { return .init(.allowed, "standalone_application") }
        if value.activation == .accessory || value.uiElement {
            return value.independentWindow ? .init(.allowed, "independent_work_window") : .init(.unknown, "work_window_unconfirmed")
        }
        return .init(.unknown, "activation_unconfirmed")
    }
    public static func window(_ value: WindowEvidence, owner: AdmissionDecision) -> AdmissionDecision {
        guard owner.isAllowed else { return owner }
        if value.isSystemDialog || value.modal == true { return .init(.excluded, "dialog") }
        guard value.role == "AXWindow", value.subrole == "AXStandardWindow" else {
            return .init(value.subrole == nil ? .unknown : .excluded, "non_work_window")
        }
        guard value.modal == false, value.canRaise else { return .init(.unknown, "window_action_unconfirmed") }
        return .init(.allowed, "standard_work_window")
    }
    public static func page(origin: String?, supportedSource: Bool, excluded: Bool) -> AdmissionDecision {
        if excluded { return .init(.excluded, "excluded_site") }
        guard supportedSource, let origin, let url = URL(string: origin),
              ["http", "https"].contains(url.scheme?.lowercased() ?? ""), url.host?.isEmpty == false else {
            return .init(.unknown, "page_unconfirmed")
        }
        return .init(.allowed, "supported_browser_page")
    }
    public static func local(registered: Bool, validFileURL: Bool) -> AdmissionDecision {
        registered && validFileURL ? .init(.allowed, "registered_local_target") : .init(.unknown, "unregistered_local_target")
    }
    public static func document(validFileURL: Bool, isRegularFile: Bool, owner: AdmissionDecision) -> AdmissionDecision {
        guard owner.isAllowed else { return owner }
        return validFileURL && isRegularFile ? .init(.allowed, "observed_window_document") : .init(.unknown, "document_unconfirmed")
    }
    public static func isObservedDocument(_ target: Destination) -> Bool {
        target.kind == .file && target.bundleID != nil && target.admission?.policyVersion == version &&
            target.admission?.basis == "observed_window_document"
    }
    public static func finderDirectory(validFileURL: Bool, isDirectory: Bool, owner: AdmissionDecision) -> AdmissionDecision {
        guard owner.isAllowed else { return owner }
        return validFileURL && isDirectory ? .init(.allowed, "observed_finder_directory") : .init(.unknown, "directory_unconfirmed")
    }
    public static func qspaceDirectory(validFileURL: Bool, isDirectory: Bool, owner: AdmissionDecision) -> AdmissionDecision {
        guard owner.isAllowed else { return owner }
        return validFileURL && isDirectory ? .init(.allowed, "observed_qspace_directory") : .init(.unknown, "directory_unconfirmed")
    }
    public static func isObservedDirectory(_ target: Destination) -> Bool {
        guard target.kind == .directory, let admission = target.admission else { return false }
        let finder = target.bundleID == finderBundleID && admission.basis == "observed_finder_directory" &&
            ["target-eligibility-v2", "target-eligibility-v3", version].contains(admission.policyVersion)
        let qspace = target.bundleID == qspaceBundleID && admission.basis == "observed_qspace_directory" &&
            ["target-eligibility-v3", version].contains(admission.policyVersion)
        return finder || qspace
    }
    public static func validates(_ target: Destination, rules: [String] = []) -> Bool {
        guard let admission = target.admission, supportedVersions.contains(admission.policyVersion) else { return false }
        if let bundle = target.bundleID, CapturePolicy.isExcluded(bundle, rules: rules) { return false }
        switch target.kind {
        case .app:
            return target.bundleID.map { target.id == "app:" + $0 } == true && ["standalone_application", "independent_work_window"].contains(admission.basis)
        case .window:
            return target.bundleID != nil && (["standard_work_window", "supported_browser_window"].contains(admission.basis) ||
                (["target-eligibility-v2", "target-eligibility-v3", version].contains(admission.policyVersion) &&
                    admission.basis == "supported_finder_window" && target.bundleID == finderBundleID) ||
                (["target-eligibility-v3", version].contains(admission.policyVersion) && admission.basis == "supported_qspace_window" && target.bundleID == qspaceBundleID))
        case .page:
            return admission.basis == "supported_browser_page" && page(origin: target.origin, supportedSource: true, excluded: false).isAllowed
        case .file, .directory:
            return (admission.basis == "registered_local_target" || isObservedDirectory(target) || isObservedDocument(target)) &&
                URL(string: target.locator ?? "")?.isFileURL == true
        }
    }
}

/// A callback is usable only within the exact uninterrupted observation generation.
public struct ObservationGeneration: Sendable {
    public private(set) var token = UUID()
    public init() {}
    public mutating func invalidate() { token = UUID() }
    public func accepts(_ token: UUID) -> Bool { token == self.token }
}

/// Shared final boundary for live events and replay. Nothing rejected is serialized here.
public struct TargetEventGate {
    public private(set) var targets: [String: Destination] = [:]
    private var entrances: [String: [TargetEntrance]] = [:]
    private var inventoryIDs: [String: String] = [:]
    public var rules: [String] = []
    public init() {}
    public mutating func reset() { targets = [:]; entrances = [:]; inventoryIDs = [:] }
    public mutating func accept(_ event: BehaviorEvent) -> Bool {
        if event.kind == .sessionStart { reset(); return true }
        if event.kind == .captureGap {
            let pauseKeys: Set<String> = ["pauseVersion", "pauseID", "pausePhase", "pauseReasons"]
            let interruptionKeys: Set<String> = ["interruptionVersion", "interruptionID", "interruptionPhase", "causeCategory", "causeSignals"]
            let hasPause = event.data.keys.contains(where: pauseKeys.contains)
            let hasInterruption = event.data.keys.contains(where: { ["interruptionVersion", "interruptionID", "interruptionPhase", "causeSignals"].contains($0) })
            guard event.app == nil, event.data.keys.allSatisfy({ $0 == "reason" || $0 == "collectorHeartbeatSummary" || pauseKeys.contains($0) || interruptionKeys.contains($0) }),
                  event.data["collectorHeartbeatSummary"].map(HeartbeatSummary.isIdentityFree) ?? true else { return false }
            if hasPause {
                guard event.data["pauseVersion"]?.string == "known-inactivity-v1",
                      event.data["pauseID"]?.string.flatMap(UUID.init(uuidString:)) != nil,
                      ["begin", "end"].contains(event.data["pausePhase"]?.string ?? ""),
                      let reasons = event.data["pauseReasons"]?.array, !reasons.isEmpty,
                      reasons.allSatisfy({ ["system_sleep", "screen_locked"].contains($0.string ?? "") }) else { return false }
            }
            let pairedCategories = ["system_unavailable", "foreground_unconfirmed"]
            let pointCategories = [
                "settings_changed": "settings_change", "user_session_inactive": "user_session_inactive",
                "inactive_resume_target_changed": "resume_target_changed",
                "inactive_resume_source_unavailable": "resume_source_unavailable",
                "inactive_resume_unverified": "resume_unverified",
                "scheduler_or_clock_gap": "scheduler_or_clock_gap"]
            if hasInterruption {
                guard !hasPause, event.data["interruptionVersion"]?.string == "interruption-cause-v1",
                      event.data["interruptionID"]?.string.flatMap(UUID.init(uuidString:)) != nil,
                      let phase = event.data["interruptionPhase"]?.string, ["begin", "end"].contains(phase),
                      let category = event.data["causeCategory"]?.string, pairedCategories.contains(category),
                      event.data["reason"]?.string == (phase == "begin" ? "observation_suspended" : "observation_resumed") else { return false }
                if category == "system_unavailable" {
                    guard let signals = event.data["causeSignals"]?.array, !signals.isEmpty,
                          signals.allSatisfy({ ["system_sleep", "screen_locked", "session_inactive"].contains($0.string ?? "") }),
                          Set(signals.compactMap(\.string)).count == signals.count else { return false }
                } else if event.data["causeSignals"] != nil { return false }
            } else if let category = event.data["causeCategory"]?.string {
                guard !hasPause, pointCategories[event.data["reason"]?.string ?? ""] == category else { return false }
            }
            entrances = [:]; inventoryIDs = [:]; return true
        }
        if event.kind == .observationScope {
            guard event.app == nil, event.source == "lifecycle",
                  Set(event.data.keys) == ["scopeVersion", "scopeID", "phase", "category"],
                  event.data["scopeVersion"]?.string == "excluded-foreground-v1",
                  event.data["scopeID"]?.string.flatMap(UUID.init(uuidString:)) != nil,
                  ["begin", "end"].contains(event.data["phase"]?.string ?? ""),
                  ["excluded_app", "excluded_page", "private_browser", "secure_input"].contains(event.data["category"]?.string ?? "") else { return false }
            entrances = [:]; inventoryIDs = [:]; return true
        }
        for prefix in ["browser:", "finder:", "qspace:"] where event.source.hasPrefix(prefix) {
            if !known("app:" + String(event.source.dropFirst(prefix.count))) { return false }
        }
        if event.source.hasPrefix("browser-extension:") {
            let parts = event.source.split(separator: ":")
            guard parts.count == 3, known("app:" + parts[1]) else { return false }
        }
        if let app = event.app {
            guard !CapturePolicy.isExcluded(app.bundleID, rules: rules),
                  targets["app:" + app.bundleID].map({ TargetEligibilityPolicy.validates($0, rules: rules) }) == true else { return false }
        }
        switch event.kind {
        case .targetUpdated:
            guard let target = try? event.data["target"]?.decoded(Destination.self),
                  TargetEligibilityPolicy.validates(target, rules: rules) else { return false }
            if target.kind == .window || TargetEligibilityPolicy.isObservedDocument(target),
               target.bundleID.map({ known("app:" + $0) }) != true { return false }
            targets[target.id] = target
        case .inventoryChanged:
            guard let values = try? event.data["entrances"]?.decoded([TargetEntrance].self),
                  values.allSatisfy({ valid($0) }) else { return false }
            if let state = event.data["browserState"]?.object {
                guard let windows = state["windows"]?.array else { return false }
                for value in windows {
                    guard let window = value.object, let container = window["containerID"]?.string, known(container),
                          let tabs = window["tabs"]?.array else { return false }
                    let entryIDs = Set(values.filter { $0.containerID == container }.map(\.id))
                    if let selected = window["selectedEntranceID"]?.string, !entryIDs.contains(selected) { return false }
                    guard tabs.allSatisfy({ $0.object?["entranceID"]?.string.map(entryIDs.contains) == true }) else { return false }
                }
            }
            entrances[event.source] = values
            inventoryIDs[event.source] = event.id
        case .sourceStatus:
            if event.data["coverageVersion"]?.string == "state-coverage-v1" {
                if event.source == "source_registry" {
                    guard let declarations = event.data["expectedSources"]?.array else { return false }
                    for value in declarations {
                        guard let entry = value.object, let source = entry["source"]?.string else { return false }
                        if source != "running_apps", entry["ownerBundleID"]?.string.map({ known("app:" + $0) }) != true { return false }
                    }
                } else {
                    guard let status = event.data["observationStatus"]?.string,
                          ["ready", "initializing", "unavailable", "stale", "not_applicable"].contains(status) else { return false }
                    if status == "ready" {
                        guard let id = event.data["snapshotEventID"]?.string, id == inventoryIDs[event.source] else { return false }
                    } else { entrances[event.source] = nil; inventoryIDs[event.source] = nil }
                }
            } else if event.data["coverageVersion"]?.string == StateCoverage.foregroundOnlyVersion {
                guard event.source != "source_registry", event.data["snapshotEventID"] == nil,
                      event.data["expectedSources"] == nil,
                      let status = event.data["observationStatus"]?.string,
                      ["ready", "initializing", "unavailable", "stale", "not_applicable"].contains(status) else { return false }
            }
        case .contextChanged, .destinationSettled:
            guard let id = event.data["targetID"]?.string, known(id) else { return false }
            if let entrance = event.data["entranceID"]?.string {
                if let value = event.data["foregroundEntrance"], let direct = try? value.decoded(TargetEntrance.self) {
                    guard direct.id == entrance, direct.targetID == id, valid(direct) else { return false }
                } else {
                    // Inventory-backed entrances remain readable for historical archives only.
                    guard entrances.values.flatMap({ $0 }).contains(where: { $0.id == entrance && $0.targetID == id }) else { return false }
                }
            }
        case .recommendationShown:
            // Kept in the enum only so old archives remain decodable. New streams never
            // persist a menu's transient candidate list.
            return false
        case .recommendationFeedback:
            if let id = event.data["targetID"]?.string, !known(id) { return false }
        default: break
        }
        return true
    }
    public func known(_ id: String) -> Bool { targets[id].map { TargetEligibilityPolicy.validates($0, rules: rules) } == true }
    public func valid(_ entrance: TargetEntrance) -> Bool {
        guard known(entrance.targetID), known("app:" + entrance.bundleID),
              entrance.containerID.map(known) != false,
              !CapturePolicy.isExcluded(entrance.bundleID, rules: rules) else { return false }
        if entrance.kind == "qspace_pane" {
            return entrance.bundleID == TargetEligibilityPolicy.qspaceBundleID &&
                entrance.windowID?.isEmpty == false && entrance.paneID?.isEmpty == false &&
                entrance.containerID?.hasPrefix("qspace-window:") == true
        }
        return true
    }
}
