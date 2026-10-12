import Foundation
import CoreGraphics

public enum PopupPlacement {
    /// AppKit screen coordinates: origin is bottom-left, including negative display origins.
    public static func frame(at pointer: CGPoint, size: CGSize, visibleFrame: CGRect) -> CGRect {
        let bounds = visibleFrame.insetBy(dx: 8, dy: 8)
        let width = min(size.width, bounds.width)
        let height = min(size.height, bounds.height)
        let below = pointer.y - 8 - height
        let preferredY = below >= bounds.minY ? below : pointer.y + 8
        return CGRect(x: min(max(pointer.x, bounds.minX), bounds.maxX - width),
                      y: min(max(preferredY, bounds.minY), bounds.maxY - height), width: width, height: height)
    }
}
