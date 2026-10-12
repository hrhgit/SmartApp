import IntentCore

enum MainMenuTargetScope {
    static let version = "app-window-page-v1"
    static func includes(_ kind: TargetKind?) -> Bool {
        guard let kind else { return false }
        return [.app, .window, .page].contains(kind)
    }
}
