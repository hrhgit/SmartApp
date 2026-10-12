import AppKit

// Draw the app's existing cursor identity as vector paths at each native icon size.
// No screenshot, external artwork, or image-generation service is needed.
let project = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent()
let resources = project.appendingPathComponent("Resources", isDirectory: true)
let iconset = project.appendingPathComponent("build/AppIcon.iconset", isDirectory: true)
try FileManager.default.createDirectory(at: iconset, withIntermediateDirectories: true)

func drawIcon(size: Int) throws -> Data {
    guard let bitmap = NSBitmapImageRep(bitmapDataPlanes: nil, pixelsWide: size, pixelsHigh: size,
                                       bitsPerSample: 8, samplesPerPixel: 4, hasAlpha: true,
                                       isPlanar: false, colorSpaceName: .deviceRGB,
                                       bytesPerRow: 0, bitsPerPixel: 0),
          let graphics = NSGraphicsContext(bitmapImageRep: bitmap) else {
        throw NSError(domain: "SmartApp.Icon", code: 1)
    }
    NSGraphicsContext.saveGraphicsState()
    NSGraphicsContext.current = graphics
    defer { NSGraphicsContext.restoreGraphicsState() }
    let context = graphics.cgContext
    context.scaleBy(x: CGFloat(size) / 1_024, y: CGFloat(size) / 1_024)
    context.setAllowsAntialiasing(true)

    let tile = NSBezierPath(roundedRect: NSRect(x: 88, y: 88, width: 848, height: 848), xRadius: 192, yRadius: 192)
    context.saveGState()
    context.setShadow(offset: CGSize(width: 0, height: -9), blur: 21,
                      color: NSColor(calibratedWhite: 0, alpha: 0.20).cgColor)
    NSColor(calibratedRed: 0.24, green: 0.31, blue: 0.74, alpha: 1).setFill()
    tile.fill()
    context.restoreGState()
    let gradient = NSGradient(colors: [
        NSColor(calibratedRed: 0.28, green: 0.34, blue: 0.82, alpha: 1),
        NSColor(calibratedRed: 0.40, green: 0.49, blue: 0.96, alpha: 1)
    ])!
    gradient.draw(in: tile, angle: 75)
    NSColor.white.withAlphaComponent(0.20).setStroke()
    tile.lineWidth = 2; tile.stroke()

    // Three fading motion trails echo the menu-bar symbol and trajectory data.
    for (index, line) in [(CGPoint(x: 539, y: 706), CGPoint(x: 738, y: 706)),
                          (CGPoint(x: 584, y: 610), CGPoint(x: 784, y: 610)),
                          (CGPoint(x: 627, y: 514), CGPoint(x: 755, y: 514))].enumerated() {
        let path = NSBezierPath()
        path.move(to: line.0); path.line(to: line.1)
        path.lineCapStyle = .round; path.lineWidth = 26
        NSColor.white.withAlphaComponent(0.80 - Double(index) * 0.17).setStroke()
        path.stroke()
    }

    // Optical centering and a generous arrow silhouette stay legible at 16–32 px.
    let cursor = NSBezierPath()
    cursor.move(to: CGPoint(x: 305, y: 770))
    cursor.line(to: CGPoint(x: 329, y: 314))
    cursor.line(to: CGPoint(x: 428, y: 425))
    cursor.line(to: CGPoint(x: 522, y: 252))
    cursor.curve(to: CGPoint(x: 550, y: 242), controlPoint1: CGPoint(x: 528, y: 239), controlPoint2: CGPoint(x: 539, y: 236))
    cursor.line(to: CGPoint(x: 610, y: 274))
    cursor.curve(to: CGPoint(x: 619, y: 302), controlPoint1: CGPoint(x: 623, y: 280), controlPoint2: CGPoint(x: 626, y: 291))
    cursor.line(to: CGPoint(x: 525, y: 475))
    cursor.line(to: CGPoint(x: 670, y: 494))
    cursor.close()
    context.saveGState()
    context.setShadow(offset: CGSize(width: 0, height: -8), blur: 13,
                      color: NSColor(calibratedRed: 0.10, green: 0.14, blue: 0.39, alpha: 0.22).cgColor)
    NSColor.white.setFill(); cursor.fill()
    context.restoreGState()

    guard let png = bitmap.representation(using: .png, properties: [:]) else {
        throw NSError(domain: "SmartApp.Icon", code: 2)
    }
    return png
}

func drawMenuBarIcon(size: Int) throws -> Data {
    guard let bitmap = NSBitmapImageRep(bitmapDataPlanes: nil, pixelsWide: size, pixelsHigh: size,
                                       bitsPerSample: 8, samplesPerPixel: 4, hasAlpha: true,
                                       isPlanar: false, colorSpaceName: .deviceRGB,
                                       bytesPerRow: 0, bitsPerPixel: 0),
          let graphics = NSGraphicsContext(bitmapImageRep: bitmap) else {
        throw NSError(domain: "SmartApp.Icon", code: 3)
    }
    NSGraphicsContext.saveGraphicsState()
    NSGraphicsContext.current = graphics
    defer { NSGraphicsContext.restoreGraphicsState() }
    let context = graphics.cgContext
    context.scaleBy(x: CGFloat(size) / 1_024, y: CGFloat(size) / 1_024)
    context.setAllowsAntialiasing(true)

    // The source paths live in the AppIcon canvas. Fit their visible bounds into
    // the menu-bar canvas so transparent margins do not make the glyph look tiny.
    let contentBounds = CGRect(x: 290, y: 232, width: 512, height: 544)
    let padding: CGFloat = 64
    let fit = min((1_024 - padding * 2) / contentBounds.width,
                  (1_024 - padding * 2) / contentBounds.height)
    context.translateBy(x: (1_024 - contentBounds.width * fit) / 2, y: padding)
    context.scaleBy(x: fit, y: fit)
    context.translateBy(x: -contentBounds.minX, y: -contentBounds.minY)

    // Keep the same large pointer and three right-side motion trails, without the colored tile.
    for line in [(CGPoint(x: 539, y: 706), CGPoint(x: 738, y: 706)),
                 (CGPoint(x: 584, y: 610), CGPoint(x: 784, y: 610)),
                 (CGPoint(x: 627, y: 514), CGPoint(x: 755, y: 514))] {
        let path = NSBezierPath()
        path.move(to: line.0); path.line(to: line.1)
        path.lineCapStyle = .round; path.lineWidth = 26 / fit
        NSColor.white.setStroke(); path.stroke()
    }

    let cursor = NSBezierPath()
    cursor.move(to: CGPoint(x: 305, y: 770))
    cursor.line(to: CGPoint(x: 329, y: 314))
    cursor.line(to: CGPoint(x: 428, y: 425))
    cursor.line(to: CGPoint(x: 522, y: 252))
    cursor.curve(to: CGPoint(x: 550, y: 242), controlPoint1: CGPoint(x: 528, y: 239), controlPoint2: CGPoint(x: 539, y: 236))
    cursor.line(to: CGPoint(x: 610, y: 274))
    cursor.curve(to: CGPoint(x: 619, y: 302), controlPoint1: CGPoint(x: 623, y: 280), controlPoint2: CGPoint(x: 626, y: 291))
    cursor.line(to: CGPoint(x: 525, y: 475))
    cursor.line(to: CGPoint(x: 670, y: 494))
    cursor.close()
    NSColor.white.setFill(); cursor.fill()

    guard let png = bitmap.representation(using: .png, properties: [:]) else {
        throw NSError(domain: "SmartApp.Icon", code: 4)
    }
    return png
}

for points in [16, 32, 128, 256, 512] {
    for scale in [1, 2] {
        let suffix = scale == 2 ? "@2x" : ""
        let name = "icon_\(points)x\(points)\(suffix).png"
        try drawIcon(size: points * scale).write(to: iconset.appendingPathComponent(name))
    }
}
try drawIcon(size: 512).write(to: resources.appendingPathComponent("AppIcon-preview.png"))
try drawMenuBarIcon(size: 64).write(to: resources.appendingPathComponent("MenuBarIcon.png"))

let tool = Process()
tool.executableURL = URL(fileURLWithPath: "/usr/bin/iconutil")
tool.arguments = ["--convert", "icns", "--output", resources.appendingPathComponent("AppIcon.icns").path, iconset.path]
try tool.run(); tool.waitUntilExit()
guard tool.terminationStatus == 0 else { throw NSError(domain: "SmartApp.Icon", code: Int(tool.terminationStatus)) }
print("Generated Resources/AppIcon.icns, AppIcon-preview.png, and MenuBarIcon.png")
