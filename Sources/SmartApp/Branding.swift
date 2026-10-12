import AppKit

enum SmartAppBranding {
    static func icon(size: CGFloat? = nil) -> NSImage? {
        guard let url = Bundle.main.url(forResource: "AppIcon", withExtension: "icns"),
              let image = NSImage(contentsOf: url) else { return nil }
        if let size {
            image.size = NSSize(width: size, height: size)
        }
        image.isTemplate = false
        return image
    }

    static func menuBarIcon(size: CGFloat = 18) -> NSImage? {
        guard let url = Bundle.main.url(forResource: "MenuBarIcon", withExtension: "png"),
              let image = NSImage(contentsOf: url) else { return nil }
        image.size = NSSize(width: size, height: size)
        image.isTemplate = true
        return image
    }
}
