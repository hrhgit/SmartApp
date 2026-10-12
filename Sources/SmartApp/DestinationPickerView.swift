import AppKit
import IntentCore
import SwiftUI
import UniformTypeIdentifiers

enum DestinationPickerLayout {
    static let diameter: CGFloat = 520
    static let radius: CGFloat = 176
    static let iconDiameter: CGFloat = 64
    static let itemSize = CGSize(width: 104, height: 96)
    static let hoverPadding: CGFloat = 8
    static let startAngle = -Double.pi / 4
    static let angleStep = 2 * Double.pi / Double(DynamicRecommendationConfiguration().maximumItems)
    static let fileToolCenter = CGPoint(x: 0, y: 112)
    static let fileToolSize = CGSize(width: 122, height: 34)
    static let clipboardToolCenter = CGPoint(x: 90, y: -10)
    static let clipboardToolSize = CGSize(width: 38, height: 38)

    static var hoverSize: CGSize {
        CGSize(width: itemSize.width + hoverPadding * 2, height: itemSize.height + hoverPadding * 2)
    }

    static func offset(at index: Int, count: Int) -> CGSize {
        // The occupied arc grows clockwise from upper-right. Existing slots
        // stay put when the number changes; fewer items never fill the circle.
        let angle = startAngle + angleStep * Double(index)
        return CGSize(width: cos(angle) * radius, height: sin(angle) * radius)
    }

    static func frame(around pointer: CGPoint, visibleFrame: CGRect) -> CGRect {
        let available = visibleFrame.insetBy(dx: 8, dy: 8)
        let side = min(diameter, available.width, available.height)
        return CGRect(x: min(max(pointer.x - side / 2, available.minX), available.maxX - side),
                      y: min(max(pointer.y - side / 2, available.minY), available.maxY - side),
                      width: side, height: side)
    }
}

private struct PickerOccupiedArc: Shape {
    let count: Int
    func path(in rect: CGRect) -> Path {
        guard count > 1 else { return Path() }
        var path = Path()
        path.addArc(center: CGPoint(x: rect.midX, y: rect.midY), radius: DestinationPickerLayout.radius,
                    startAngle: .radians(DestinationPickerLayout.startAngle),
                    endAngle: .radians(DestinationPickerLayout.startAngle +
                                      DestinationPickerLayout.angleStep * Double(count - 1)), clockwise: false)
        return path
    }
}

/// Presentation is resolved once per invocation. Icons stay local and never enter the dataset.
struct DestinationPickerItem: Identifiable {
    let choice: NavigationChoice
    let icon: NSImage?
    let faviconURL: URL?
    let symbol: String
    let kindLabel: LocalizedStringKey
    var id: String { choice.id }
    var name: String {
        guard choice.target.kind == .page else { return choice.target.name }
        return Self.pageDisplayName(choice.target.locator ?? choice.target.name)
    }

    /// Keeps menu labels scannable without changing the saved, reopenable locator.
    static func pageDisplayName(_ address: String) -> String {
        for prefix in ["https://", "http://"] {
            if address.range(of: prefix, options: [.anchored, .caseInsensitive]) != nil {
                return String(address.dropFirst(prefix.count))
            }
        }
        return address
    }

    var badge: String? {
        switch choice.target.kind {
        case .page: "globe"
        case .window: "macwindow"
        default: nil
        }
    }

    @MainActor
    static func resolve(_ choices: [NavigationChoice]) -> [Self] {
        var appIcons: [String: NSImage] = [:]
        return choices.map { choice in
            let symbol: String
            let label: LocalizedStringKey
            switch choice.target.kind {
            case .app: (symbol, label) = ("app.fill", "应用")
            case .window: (symbol, label) = ("macwindow", "窗口")
            case .page: (symbol, label) = ("globe", "网页标签")
            case .directory: (symbol, label) = ("folder.fill", "目录")
            case .file: (symbol, label) = ("doc.fill", "文件")
            }

            var icon: NSImage?
            if choice.target.kind == .directory || choice.target.kind == .file {
                if let locator = choice.target.locator, let url = URL(string: locator), url.isFileURL {
                    icon = NSWorkspace.shared.icon(forFile: url.path)
                } else {
                    let type = choice.target.kind == .directory ? UTType.folder :
                        (choice.target.fileType.flatMap { UTType(filenameExtension: $0) } ?? .data)
                    icon = NSWorkspace.shared.icon(for: type)
                }
            } else if let bundleID = choice.entrance?.bundleID ?? choice.target.bundleID {
                if let cached = appIcons[bundleID] {
                    icon = cached
                } else if let url = NSWorkspace.shared.urlForApplication(withBundleIdentifier: bundleID) {
                    let image = NSWorkspace.shared.icon(forFile: url.path)
                    appIcons[bundleID] = image
                    icon = image
                }
            }
            return Self(choice: choice, icon: icon, faviconURL: WebsiteFavicon.url(for: choice.target),
                        symbol: symbol, kindLabel: label)
        }
    }
}

struct DestinationPickerView: View {
    let items: [DestinationPickerItem]
    let choose: (NavigationChoice) -> Void
    let close: () -> Void
    let findDirectories: () -> Void
    let showClipboard: () -> Void
    @Environment(\.accessibilityReduceMotion) private var reduceMotion
    @Environment(\.accessibilityReduceTransparency) private var reduceTransparency
    @State private var expanded = false
    @State private var hoveredID: String?

    init(items: [DestinationPickerItem], choose: @escaping (NavigationChoice) -> Void,
         close: @escaping () -> Void, initiallyExpanded: Bool = false,
         findDirectories: @escaping () -> Void = {}, showClipboard: @escaping () -> Void = {}) {
        self.items = items
        self.choose = choose
        self.close = close
        self.findDirectories = findDirectories; self.showClipboard = showClipboard
        _expanded = State(initialValue: initiallyExpanded)
    }

    var body: some View {
        PickerGlassContainer {
            ZStack {
                Color.clear.contentShape(Rectangle()).onTapGesture(perform: close)

                PickerOccupiedArc(count: items.count)
                    .stroke(.primary.opacity(reduceTransparency ? 0.28 : 0.09), lineWidth: 1)
                    .frame(width: DestinationPickerLayout.diameter, height: DestinationPickerLayout.diameter)
                    .opacity(expanded && !items.isEmpty ? 1 : 0)
                    .allowsHitTesting(false)
                    .accessibilityHidden(true)

                // EnumeratedSequence's collection conformance needs macOS 26;
                // keep macOS 14 support and identify rows by their target IDs.
                ForEach(Array(items.enumerated()), id: \.element.id) { index, item in
                    let progress: CGFloat = reduceMotion || expanded ? 1 : 0.45
                    let angle = DestinationPickerLayout.startAngle +
                        (reduceMotion || expanded ? DestinationPickerLayout.angleStep * Double(index) : 0)
                    PickerDestinationButton(item: item, highlighted: hoveredID == item.id) {
                        choose(item.choice)
                    }
                    // The icon, rather than the combined icon + caption, sits on the orbit.
                    .offset(y: (DestinationPickerLayout.itemSize.height - DestinationPickerLayout.iconDiameter) / 2)
                    .rotationEffect(.radians(-angle))
                    .scaleEffect(reduceMotion || expanded ? 1 : 0.55)
                    .offset(x: DestinationPickerLayout.radius * progress)
                    .rotationEffect(.radians(angle))
                    .opacity(expanded ? 1 : 0)
                    .animation(reduceMotion ? .easeOut(duration: 0.12) :
                        .spring(response: 0.4, dampingFraction: 0.82).delay(Double(index) * 0.035), value: expanded)
                    .onHover { hovering in
                        if hovering { hoveredID = item.id }
                        else if hoveredID == item.id { hoveredID = nil }
                    }
                }

                PickerHub(hoveredItem: items.first { $0.id == hoveredID }, isEmpty: items.isEmpty, close: close)
                    .opacity(expanded ? 1 : 0)
                    .animation(.easeOut(duration: 0.14), value: expanded)

                PickerTools(findDirectories: findDirectories, showClipboard: showClipboard)
                    .opacity(expanded ? 1 : 0)
            }
            .frame(width: DestinationPickerLayout.diameter, height: DestinationPickerLayout.diameter)
        }
        .onAppear { expanded = true }
        .onExitCommand(perform: close)
    }
}

private struct PickerTools: View {
    let findDirectories: () -> Void
    let showClipboard: () -> Void
    var body: some View {
        ZStack {
            Button(action: findDirectories) {
                Label("目录与文件", systemImage: "folder.badge.magnifyingglass")
                    .font(.callout.weight(.medium)).foregroundStyle(.primary).frame(width: DestinationPickerLayout.fileToolSize.width,
                                                         height: DestinationPickerLayout.fileToolSize.height)
                    .modifier(PickerSurface(shape: Capsule(), interactive: true))
            }.buttonStyle(PickerPressStyle()).offset(x: DestinationPickerLayout.fileToolCenter.x, y: DestinationPickerLayout.fileToolCenter.y)
                .help("进入 FileRecall，结合当前窗口查找目录与文件")
            Button(action: showClipboard) {
                Image(systemName: "clipboard").font(.system(size: 17)).foregroundStyle(.primary)
                    .frame(width: 38, height: 38).modifier(PickerSurface(shape: Circle(), interactive: true))
            }.buttonStyle(PickerPressStyle()).offset(x: DestinationPickerLayout.clipboardToolCenter.x, y: DestinationPickerLayout.clipboardToolCenter.y)
                .help("查看剪贴板历史").accessibilityLabel("查看剪贴板历史")
        }
    }
}

private struct PickerDestinationButton: View {
    let item: DestinationPickerItem
    let highlighted: Bool
    let action: () -> Void
    @Environment(\.accessibilityReduceMotion) private var reduceMotion
    @State private var hover = PickerHoverResponse.idle
    @Namespace private var hoverSpace

    var body: some View {
        Button(action: action) {
            VStack(spacing: 7) {
                PickerHoverIcon(icon: item.icon, faviconURL: item.faviconURL, symbol: item.symbol, badge: item.badge,
                                highlighted: highlighted, hover: reduceMotion ? .idle : hover)

                Text(item.name)
                    .font(.caption.weight(.medium))
                    .lineLimit(1)
                    .truncationMode(.middle)
                    .padding(.horizontal, 10)
                    .padding(.vertical, 4)
                    .modifier(PickerCaptionSurface())
            }
            .frame(width: DestinationPickerLayout.itemSize.width,
                   height: DestinationPickerLayout.itemSize.height, alignment: .top)
            .padding(DestinationPickerLayout.hoverPadding)
            .contentShape(RoundedRectangle(cornerRadius: 26))
        }
        .buttonStyle(PickerPressStyle())
        // Observe an untransformed region, including a little room for the floating icon.
        // Only the icon inside it moves; pointer coordinates never chase the animation.
        .frame(width: DestinationPickerLayout.hoverSize.width, height: DestinationPickerLayout.hoverSize.height)
        .onContinuousHover(coordinateSpace: .named(hoverSpace)) { phase in
            guard !reduceMotion else { return }
            switch phase {
            case .active(let location): hover = PickerHoverResponse(location: location)
            case .ended: hover = .idle
            }
        }
        .coordinateSpace(name: hoverSpace)
        .onChange(of: reduceMotion) { _, _ in hover = .idle }
        .onDisappear { hover = .idle }
        .help(item.name)
        .accessibilityLabel(Text(item.name))
        .accessibilityValue(Text(item.kindLabel))
        .accessibilityHint("打开这个目标")
    }
}

/// Just the current local presentation response: no event history, timers, or recorder access.
private struct PickerHoverResponse: Equatable {
    var direction: CGSize = .zero
    var proximity: CGFloat = 0
    var isActive = false
    static let idle = Self()

    init() {}

    init(location: CGPoint) {
        let radius = DestinationPickerLayout.iconDiameter / 2
        let dx = location.x - DestinationPickerLayout.hoverSize.width / 2
        let dy = location.y - (DestinationPickerLayout.hoverPadding + radius)
        let distance = hypot(dx, dy)
        // Full response over the icon, with a smooth falloff in the surrounding hit area.
        let falloff = min(max((distance - radius) / 32, 0), 1)
        proximity = 1 - falloff * falloff * (3 - 2 * falloff)
        let divisor = max(radius, distance)
        direction = CGSize(width: dx / divisor * proximity, height: dy / divisor * proximity)
        isActive = true
    }
}

private struct PickerHoverIcon: View {
    let icon: NSImage?
    let faviconURL: URL?
    let symbol: String
    let badge: String?
    let highlighted: Bool
    let hover: PickerHoverResponse
    @Environment(\.accessibilityReduceMotion) private var reduceMotion
    @Environment(\.accessibilityReduceTransparency) private var reduceTransparency
    @Environment(\.colorSchemeContrast) private var contrast

    var body: some View {
        let x = hover.direction.width
        let y = hover.direction.height
        let light = UnitPoint(x: 0.5 + x * 0.35, y: 0.5 + y * 0.35)
        ZStack {
            Circle()
                .fill(RadialGradient(colors: [.white.opacity(0.42), .white.opacity(0)],
                                     center: light, startRadius: 0, endRadius: 58))
                .opacity(reduceTransparency ? 0 : hover.proximity)
                .allowsHitTesting(false)
            PickerDestinationIcon(icon: icon, faviconURL: faviconURL, symbol: symbol, badge: badge)
                .offset(x: x * 2, y: y * 2)
        }
        .frame(width: DestinationPickerLayout.iconDiameter, height: DestinationPickerLayout.iconDiameter)
        // Keep native glass as the material; one spring system owns the hover transforms.
        .modifier(PickerSurface(shape: Circle(), interactive: false))
        .overlay {
            Circle()
                .strokeBorder(Color.accentColor.opacity(highlighted ? (contrast == .increased ? 0.95 : 0.65) : 0), lineWidth: 1.5)
                .allowsHitTesting(false)
        }
        // No preset lift or enlargement: a pointer at the icon's center produces zero displacement.
        .rotation3DEffect(.degrees(-Double(y) * 6), axis: (x: 1, y: 0, z: 0), perspective: 0.45)
        .rotation3DEffect(.degrees(Double(x) * 6), axis: (x: 0, y: 1, z: 0), perspective: 0.45)
        .shadow(color: .black.opacity(0.13 + hover.proximity * 0.06),
                radius: 9 + hover.proximity * 4, x: -x * 3, y: 5 - y * 3)
        .offset(x: x * 9, y: y * 9)
        .animation(reduceMotion ? nil : (hover.isActive ?
            .interactiveSpring(duration: 0.18, extraBounce: 0, blendDuration: 0.08) :
            .spring(duration: 0.32, bounce: 0.08, blendDuration: 0.08)), value: hover)
        .animation(.easeOut(duration: 0.15), value: highlighted)
    }
}

private struct PickerDestinationIcon: View {
    let icon: NSImage?
    let faviconURL: URL?
    let symbol: String
    let badge: String?

    var body: some View {
        ZStack(alignment: .bottomTrailing) {
            Group {
                if let faviconURL {
                    WebsiteFaviconImage(faviconURL: faviconURL, fallbackIcon: icon, symbol: symbol)
                } else {
                    PickerIconArtwork(icon: icon, symbol: symbol)
                }
            }
            .frame(width: 52, height: 52)
            if let badge {
                Image(systemName: badge)
                    .font(.system(size: 10, weight: .semibold))
                    .frame(width: 21, height: 21)
                    .background(Color(nsColor: .windowBackgroundColor), in: Circle())
                    .overlay { Circle().strokeBorder(.primary.opacity(0.12), lineWidth: 0.5) }
                    .offset(x: 3, y: 3)
            }
        }
        .accessibilityHidden(true)
    }
}

private struct WebsiteFaviconImage: View {
    let faviconURL: URL
    let fallbackIcon: NSImage?
    let symbol: String
    @State private var favicon: NSImage?

    var body: some View {
        PickerIconArtwork(icon: favicon ?? fallbackIcon, symbol: symbol)
            .task(id: faviconURL) {
                favicon = nil
                let image = await WebsiteFaviconLoader.shared.image(for: faviconURL)
                guard !Task.isCancelled else { return }
                favicon = image
            }
    }
}

private struct PickerIconArtwork: View {
    let icon: NSImage?
    let symbol: String

    var body: some View {
        if let icon {
            Image(nsImage: icon).resizable().interpolation(.high).scaledToFit()
        } else {
            Image(systemName: symbol).resizable().scaledToFit().padding(10).foregroundStyle(.tint)
        }
    }
}

private struct PickerHub: View {
    let hoveredItem: DestinationPickerItem?
    let isEmpty: Bool
    let close: () -> Void

    var body: some View {
        Button(action: close) {
            VStack(spacing: 5) {
                Image(systemName: hoveredItem == nil ? "cursorarrow.rays" : "arrow.up.forward")
                    .font(.system(size: 17, weight: .medium))
                    .foregroundStyle(.secondary)
                if let hoveredItem {
                    Text(hoveredItem.name).font(.caption.weight(.semibold)).lineLimit(2)
                    Text(hoveredItem.kindLabel).font(.caption2).foregroundStyle(.secondary)
                } else {
                    Text(isEmpty ? "暂无推荐目标" : "接下来去哪").font(.callout.weight(.semibold))
                    Text(isEmpty ? "继续使用后再试" : "点击图标打开").font(.caption2).foregroundStyle(.secondary)
                }
                Text("Esc 关闭").font(.caption2).foregroundStyle(.secondary).padding(.top, 3)
            }
            .multilineTextAlignment(.center)
            .padding(.horizontal, 12)
            .frame(width: 124, height: 124)
            .modifier(PickerSurface(shape: Circle(), interactive: false))
            .contentShape(Circle())
            .shadow(color: .black.opacity(0.10), radius: 12, y: 4)
        }
        .buttonStyle(PickerPressStyle())
        .help("关闭推荐目标菜单")
        .accessibilityLabel("关闭推荐目标菜单")
        .accessibilityHint(isEmpty ? "暂无推荐目标，继续使用后再试" : "也可以按 Escape 关闭")
    }
}

private struct PickerPressStyle: ButtonStyle {
    @Environment(\.accessibilityReduceMotion) private var reduceMotion

    func makeBody(configuration: Configuration) -> some View {
        configuration.label
            .scaleEffect(configuration.isPressed && !reduceMotion ? 0.95 : 1)
            .opacity(configuration.isPressed ? 0.8 : 1)
            .animation(.easeOut(duration: 0.1), value: configuration.isPressed)
    }
}

private struct PickerGlassContainer<Content: View>: View {
    @ViewBuilder let content: Content

    var body: some View {
        if #available(macOS 26.0, *) {
            GlassEffectContainer(spacing: 16) { content }
        } else {
            content
        }
    }
}

private struct PickerSurface<S: Shape>: ViewModifier {
    let shape: S
    let interactive: Bool
    @Environment(\.accessibilityReduceTransparency) private var reduceTransparency
    @Environment(\.colorSchemeContrast) private var contrast

    func body(content: Content) -> some View {
        if #available(macOS 26.0, *) {
            content
                .background(Color(nsColor: .windowBackgroundColor).opacity(reduceTransparency ? 1 : 0), in: shape)
                .glassEffect(reduceTransparency ? .identity : .regular.interactive(interactive), in: shape)
                .overlay { shape.stroke(.primary.opacity(contrast == .increased ? 0.6 : 0), lineWidth: 1) }
        } else {
            content
                .background(.regularMaterial, in: shape)
                .background(Color(nsColor: .windowBackgroundColor).opacity(reduceTransparency ? 1 : 0), in: shape)
                .overlay { shape.stroke(.primary.opacity(contrast == .increased ? 0.6 : 0.14), lineWidth: 1) }
        }
    }
}

private struct PickerCaptionSurface: ViewModifier {
    @Environment(\.accessibilityReduceTransparency) private var reduceTransparency

    func body(content: Content) -> some View {
        content
            .background(.regularMaterial, in: Capsule())
            .background(Color(nsColor: .windowBackgroundColor).opacity(reduceTransparency ? 1 : 0), in: Capsule())
    }
}
