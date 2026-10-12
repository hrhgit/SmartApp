import Foundation

public enum JSONValue: Codable, Equatable, Sendable {
    case string(String), number(Double), bool(Bool), null
    case array([JSONValue]), object([String: JSONValue])

    public init(from decoder: Decoder) throws {
        let value = try decoder.singleValueContainer()
        if value.decodeNil() { self = .null }
        else if let bool = try? value.decode(Bool.self) { self = .bool(bool) }
        else if let number = try? value.decode(Double.self) { self = .number(number) }
        else if let array = try? value.decode([JSONValue].self) { self = .array(array) }
        else if let object = try? value.decode([String: JSONValue].self) { self = .object(object) }
        else { self = .string(try value.decode(String.self)) }
    }

    public func encode(to encoder: Encoder) throws {
        var value = encoder.singleValueContainer()
        switch self {
        case .string(let string): try value.encode(string)
        case .number(let number): try value.encode(number)
        case .bool(let bool): try value.encode(bool)
        case .null: try value.encodeNil()
        case .array(let array): try value.encode(array)
        case .object(let object): try value.encode(object)
        }
    }

    public var string: String? { if case .string(let value) = self { value } else { nil } }
    public var number: Double? { if case .number(let value) = self { value } else { nil } }
    public var bool: Bool? { if case .bool(let value) = self { value } else { nil } }
    public var array: [JSONValue]? { if case .array(let value) = self { value } else { nil } }
    public var object: [String: JSONValue]? { if case .object(let value) = self { value } else { nil } }
    public static func encoded<T: Encodable>(_ value: T) throws -> JSONValue {
        try JSONDecoder().decode(JSONValue.self, from: JSONEncoder().encode(value))
    }
    public func decoded<T: Decodable>(_ type: T.Type) throws -> T {
        try JSONDecoder().decode(type, from: JSONEncoder().encode(self))
    }
}

public enum EventKind: String, Codable, CaseIterable, Sendable {
    case sessionStart = "session.start", sessionEnd = "session.end", heartbeat
    case captureGap = "capture.gap", observationScope = "observation.scope", capabilities = "capture.capabilities"
    case pointerMoved = "pointer.move", pointerDragged = "pointer.drag"
    case mouseDown = "mouse.down", mouseUp = "mouse.up", scroll
    case keyDown = "key.down", keyUp = "key.up", modifiersChanged = "key.modifiers"
    case magnify = "gesture.magnify", rotate = "gesture.rotate", swipe = "gesture.swipe"
    case appActivated = "app.activate", appLaunched = "app.launch", appTerminated = "app.terminate"
    case spaceChanged = "space.change", windowFocused = "window.focus", windowCreated = "window.create"
    case uiFocusChanged = "ui.focus", annotation
    case contextChanged = "context.change", targetUpdated = "target.update"
    case inventoryChanged = "inventory.change", sourceStatus = "source.status"
    case destinationSettled = "destination.settled"
    case clipboardChanged = "clipboard.change", clipboardMetadata = "clipboard.metadata"
    case idleChanged = "idle.change"
    case menuInvoked = "menu.invoke", recommendationShown = "recommendation.show"
    case recommendationFeedback = "recommendation.feedback"

    public var isInput: Bool {
        switch self {
        case .pointerMoved, .pointerDragged, .mouseDown, .mouseUp, .scroll,
             .keyDown, .keyUp, .modifiersChanged, .magnify, .rotate, .swipe: true
        default: false
        }
    }

    public var isTrigger: Bool {
        switch self {
        case .mouseDown, .keyDown, .scroll, .magnify, .rotate, .swipe: true
        default: false
        }
    }

    public var isOutcome: Bool {
        switch self {
        case .appActivated, .appLaunched, .appTerminated, .spaceChanged, .windowFocused, .windowCreated: true
        default: false
        }
    }
}

public struct AppContext: Codable, Equatable, Sendable {
    public var bundleID: String
    public var name: String
    public init(bundleID: String, name: String) { self.bundleID = bundleID; self.name = name }
}

public struct BehaviorEvent: Codable, Equatable, Identifiable, Sendable {
    public var schemaVersion = 1
    public var id: String
    public var sessionID: String
    public var timestampMs: Int64
    public var monotonicMs: Double
    public var kind: EventKind
    public var source: String
    public var app: AppContext?
    public var data: [String: JSONValue]
    public var sequence: Int?
    public var decisionID: String?

    public init(id: String = UUID().uuidString, sessionID: String,
                timestampMs: Int64 = Int64(Date().timeIntervalSince1970 * 1_000),
                monotonicMs: Double = ProcessInfo.processInfo.systemUptime * 1_000,
                kind: EventKind, source: String, app: AppContext? = nil,
                data: [String: JSONValue] = [:]) {
        self.id = id; self.sessionID = sessionID; self.timestampMs = timestampMs
        self.monotonicMs = monotonicMs; self.kind = kind; self.source = source
        self.app = app; self.data = data
    }
}

/// Classifies physical keys without ever accepting or generating typed text.
public enum KeyPrivacy {
    private static let navigation: [Int: String] = [
        36: "return", 48: "tab", 51: "delete", 53: "escape", 76: "enter",
        115: "home", 116: "page_up", 117: "forward_delete", 119: "end",
        121: "page_down", 123: "left", 124: "right", 125: "down", 126: "up",
        122: "f1", 120: "f2", 99: "f3", 118: "f4", 96: "f5", 97: "f6",
        98: "f7", 100: "f8", 101: "f9", 109: "f10", 103: "f11", 111: "f12"
    ]

    public static func fields(keyCode: Int, command: Bool, control: Bool,
                              option: Bool, shift: Bool, repeated: Bool) -> [String: JSONValue] {
        var fields: [String: JSONValue] = [
            "command": .bool(command), "control": .bool(control),
            "option": .bool(option), "shift": .bool(shift), "repeat": .bool(repeated)
        ]
        // Option alone may enter composed text. Only command/control chords keep physical key IDs.
        if command || control {
            fields["category"] = .string("shortcut")
            fields["keyCode"] = .number(Double(keyCode))
        } else if let key = navigation[keyCode] {
            fields["category"] = .string("navigation")
            fields["key"] = .string(key)
        } else {
            fields["category"] = .string("typing")
        }
        return fields
    }
}
