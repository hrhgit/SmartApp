import SwiftUI
import AppKit
import IntentCore

enum DashboardPage: String, CaseIterable, Identifiable {
    case overview, events, targets, clipboard, settings
    var id: String { rawValue }
    var title: LocalizedStringResource {
        switch self {
        case .overview: "采集概览"
        case .events: "目标时间线"
        case .targets: "本地目标与导出"
        case .clipboard: "剪贴板历史"
        case .settings: "采集设置"
        }
    }
    var symbol: String {
        switch self {
        case .overview: "square.grid.2x2"
        case .events: "list.bullet.rectangle"
        case .targets: "folder"
        case .clipboard: "clipboard"
        case .settings: "slider.horizontal.3"
        }
    }
}

struct DashboardView: View {
    @Bindable var model: RecorderModel
    @State private var page: DashboardPage = .overview
    var body: some View {
        HStack(spacing: 0) {
            DashboardSidebar(selection: $page, collecting: model.isCollecting)
            Divider()
            VStack(spacing: 0) {
                DashboardHeader(model: model, page: page)
                Divider()
                ScrollView {
                    VStack(alignment: .leading, spacing: 20) {
                        if let error = model.errorMessage {
                            Label(error, systemImage: "exclamationmark.triangle").foregroundStyle(.red).textSelection(.enabled)
                        }
                        if let notice = model.notice {
                            Label(notice, systemImage: "info.circle").foregroundStyle(.secondary).textSelection(.enabled)
                        }
                        switch page {
                        case .overview: ContextOverview(model: model)
                        case .events: EventTimeline(events: model.recent)
                        case .targets: TargetsAndExport(model: model)
                        case .clipboard: ClipboardHistoryView(model: model.clipboard, close: nil).frame(height: 560)
                        case .settings: CollectorSettings(model: model)
                        }
                    }.padding(24).frame(maxWidth: .infinity, alignment: .leading)
                }
            }
        }
        .background(Color(nsColor: .windowBackgroundColor))
        .tint(.indigo)
    }
}
private struct DashboardSidebar: View {
    @Binding var selection: DashboardPage
    let collecting: Bool
    var body: some View {
        VStack(alignment: .leading, spacing: 24) {
            HStack(spacing: 10) {
                SmartAppBrandIcon(size: 32)
                VStack(alignment: .leading) {
                    Text(verbatim: "SmartApp").font(.title3.bold())
                    Text("上下文采集 · v0.2").font(.caption).foregroundStyle(.secondary)
                }
            }
            ForEach(DashboardPage.allCases) { page in
                Button { selection = page } label: {
                    Label(page.title, systemImage: page.symbol)
                        .frame(maxWidth: .infinity, alignment: .leading)
                        .padding(10)
                        .background(selection == page ? Color.indigo.opacity(0.12) : .clear,
                                    in: RoundedRectangle(cornerRadius: 9))
                }.buttonStyle(.plain)
            }
            Spacer()
            Label(collecting ? "后台采集中" : "采集已暂停", systemImage: collecting ? "record.circle" : "pause.circle")
                .font(.caption).foregroundStyle(collecting ? .green : .secondary)
            Text("记录到达的目标和顺序。\n数据仅存储在这台 Mac。").font(.caption).foregroundStyle(.secondary)
        }.padding(20).frame(width: 200)
    }
}
private struct DashboardHeader: View {
    let model: RecorderModel
    let page: DashboardPage
    var body: some View {
        HStack {
            VStack(alignment: .leading, spacing: 6) {
                Text(page.title).font(.title2.bold())
                Text(model.statusText).font(.callout).foregroundStyle(.secondary)
            }
            Spacer()
            Button(model.isWritingBacklog ? "正在自动入库…" : (model.isCollecting ? "暂停采集" : "开始采集")) { model.toggle() }
                .buttonStyle(.borderedProminent).controlSize(.large)
                .disabled(model.isWritingBacklog)
        }.padding(24)
    }
}
private struct ContextOverview: View {
    let model: RecorderModel
    var body: some View {
        VStack(alignment: .leading, spacing: 20) {
            VStack(alignment: .leading, spacing: 10) {
                Text("观察工作过程，积累下一步目标的数据").font(.title3.bold())
                Text("应用与窗口 → 页面或文件 → 后续多步变化")
                    .foregroundStyle(.secondary)
                Text("记录目标顺序、必要的当前入口和来源质量。剪贴板历史独立保存；鼠标仅用于晃动唤出，不记录键盘、快捷键和轨迹。")
                    .font(.callout).foregroundStyle(.secondary)
            }.card()
            HStack(spacing: 16) {
                MetricCard(title: "累计目标切换", value: model.stats.userActions.formatted())
                MetricCard(title: "本次目标切换", value: model.sessionSwitches.formatted())
                MetricCard(title: "已知目标", value: model.targetCount.formatted())
            }
            Text("仅统计已观察到的目标切换；不含初始快照、后台状态和同一次到达的信息补全。")
                .font(.caption).foregroundStyle(.secondary)
            VStack(alignment: .leading, spacing: 12) {
                Text("最近观察").font(.headline)
                LabeledContent("应用", value: model.activeAppName)
                LabeledContent("目标", value: model.activeTargetName)
                Text("打开面板时暂停记录面板内的操作，关闭窗口后继续观察其他应用。")
                    .font(.caption).foregroundStyle(.secondary)
            }.card()
            SourceStatusView(sources: model.sources)
            EventTimeline(events: Array(model.recent.prefix(6)))
        }
    }
}
private struct MetricCard: View {
    let title: LocalizedStringResource
    let value: String
    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text(title).font(.callout).foregroundStyle(.secondary)
            Text(value).font(.largeTitle.weight(.semibold)).monospacedDigit()
        }.frame(maxWidth: .infinity, alignment: .leading).card()
    }
}
private struct SourceStatusView: View {
    let sources: [SourceHealth]
    var body: some View {
        VStack(alignment: .leading, spacing: 14) {
            Text("数据源状态").font(.headline)
            if sources.isEmpty { Text("开始采集后显示连接状态").foregroundStyle(.secondary) }
            ForEach(sources) { source in
                LabeledContent(source.name, value: source.status).font(.callout)
            }
            Text("Chrome / Edge / Brave 通过扩展按事件记录标签页状态、到达与稳定停留；未连接时不扫描浏览器。Finder 与 QSpace Pro 按周期观察，短暂变化可能遗漏。")
                .font(.caption).foregroundStyle(.secondary)
        }.card()
    }
}
struct EventTimeline: View {
    let events: [BehaviorEvent]
    var body: some View {
        VStack(alignment: .leading, spacing: 14) {
            Text("最近的目标切换").font(.headline)
            if events.isEmpty {
                VStack(spacing: 8) {
                    SmartAppBrandIcon(size: 48)
                    Text("等待采集").font(.headline)
                    Text("开始后切换应用或浏览器标签页。")
                        .font(.callout).foregroundStyle(.secondary)
                }
                .frame(maxWidth: .infinity)
                .padding(.vertical, 20)
            }
            LazyVStack(spacing: 0) {
                ForEach(events) { event in EventRow(event: event) }
            }
        }.card()
    }
}
private struct SmartAppBrandIcon: View {
    let size: CGFloat

    var body: some View {
        Group {
            if let icon = SmartAppBranding.icon(size: size) {
                Image(nsImage: icon).resizable().interpolation(.high).scaledToFit()
            } else {
                Image(systemName: "cursorarrow.rays")
                    .font(.system(size: size * 0.7, weight: .medium))
                    .foregroundStyle(.indigo)
            }
        }
        .frame(width: size, height: size)
        .accessibilityHidden(true)
    }
}
private struct EventRow: View {
    let event: BehaviorEvent
    var body: some View {
        HStack(alignment: .top, spacing: 12) {
            Image(systemName: event.kind == .contextChanged ? "arrow.right.circle.fill" : "circle.dotted")
                .foregroundStyle(.indigo).frame(width: 24)
            VStack(alignment: .leading, spacing: 4) {
                Text(event.data["targetName"]?.string ?? event.data["status"]?.string ?? event.kind.rawValue)
                    .font(.callout).lineLimit(2)
                Text(event.source + " · " + (event.app?.name ?? "上下文采集器"))
                    .font(.caption).foregroundStyle(.secondary)
            }
            Spacer()
            Text(Date(timeIntervalSince1970: Double(event.timestampMs) / 1_000), format: .dateTime.hour().minute().second())
                .font(.caption.monospacedDigit()).foregroundStyle(.secondary)
        }.padding(.vertical, 10).accessibilityElement(children: .combine)
    }
}
private struct TargetsAndExport: View {
    let model: RecorderModel
    var body: some View {
        VStack(alignment: .leading, spacing: 20) {
            VStack(alignment: .leading, spacing: 14) {
                Text("自动采集与快捷目标").font(.headline)
                Text("浏览器页面及 Finder、QSpace Pro 中浏览的目录会自动记录完整地址，无需提前添加。也可以手动添加文件或目录作为快捷目标；添加文件时同时保留其所在目录。")
                    .font(.callout).foregroundStyle(.secondary)
                Button("添加快捷目标…") { model.registerLocalTargets() }.buttonStyle(.borderedProminent)
                ForEach(model.registeredTargets) { target in
                    Label(target.name, systemImage: target.kind == .directory ? "folder" : "doc")
                }
            }.card()
            VStack(alignment: .leading, spacing: 14) {
                Text("导出上下文数据集").font(.headline)
                Text("完整时间线、目标变更历史，以及决策前历史对应的后续 30 秒目标序列。保留观察不完整的样本，不猜测唯一最终意图。")
                    .font(.callout).foregroundStyle(.secondary)
                Text("events.jsonl · target-history.jsonl\nnavigation-samples.jsonl · manifest.json")
                    .font(.caption.monospaced()).foregroundStyle(.indigo)
                HStack {
                    Button(model.isExporting ? "导出中…" : "导出数据…") { model.exportDataset() }
                        .buttonStyle(.borderedProminent).disabled(model.isExporting)
                    Button("查看数据库") { model.revealData() }
                }
                Text("当前库仅保留前台-only 标准会话；旧归档需单独保存，不能重新混入运行库。")
                    .font(.caption).foregroundStyle(.secondary)
            }.card()
        }
    }
}
private struct CollectorSettings: View {
    @Bindable var model: RecorderModel
    var body: some View {
        VStack(alignment: .leading, spacing: 20) {
            InvocationSettingsView(controller: model.invocation, accessibilityAuthorized: model.accessibilityAuthorized)
            VStack(alignment: .leading, spacing: 14) {
                Text("连接权限").font(.headline)
                LabeledContent("辅助功能 / Device Control and Data Access", value: model.accessibilityAuthorized ? "已授权" : "未授权")
                Text("辅助功能用于识别应用窗口。Chrome、Edge、Brave 通过 SmartApp 浏览器扩展连接，事件驱动记录标签页状态与地址；Finder 和 QSpace Pro 分别需要“自动化 / Automation”权限读取目录路径。")
                    .font(.callout).foregroundStyle(.secondary)
                HStack {
                    Button("辅助功能设置") { model.requestAccessibilityPermission() }
                    Button("自动化设置") { model.openAutomationSettings() }
                    Button("重新检测与连接") { model.retrySources() }
                }
                Text("保存完整网页地址与目录路径，供快速打开和训练导出使用；不读取网页或文件正文。双击 Command 手势仅在内存中识别。")
                    .font(.caption).foregroundStyle(.secondary)
            }.card()
            VStack(alignment: .leading, spacing: 14) {
                Text("采集范围").font(.headline)
                Text("排除应用 Bundle ID，每行一个，支持 com.example.*")
                TextEditor(text: $model.exclusionText).font(.body.monospaced()).frame(height: 105)
                Text("排除网站域名，每行一个，包含其子域名")
                TextEditor(text: $model.excludedSitesText).font(.body.monospaced()).frame(height: 70)
                Button("保存采集设置") { model.saveSettings() }.buttonStyle(.borderedProminent)
                Text("隐私浏览窗口不记录页面。剪贴板记录在独立页面开关，保存文字、链接、图片和文件引用，不进入目标采集、行为训练或导出。进入 FileRecall 时才临时读取唤出前的窗口内容。")
                    .font(.caption).foregroundStyle(.secondary)
            }.card()
            VStack(alignment: .leading, spacing: 14) {
                Toggle("登录时启动 SmartApp", isOn: Binding(get: { model.loginEnabled }, set: { model.setLoginEnabled($0) }))
                Toggle("同意登录启动后自动开始采集", isOn: $model.collectOnLogin)
                    .disabled(!model.loginEnabled)
                Text("自动采集默认关闭，仅在启用登录启动并勾选同意后生效。手动打开仍默认暂停；关闭面板不停止已开始的采集。达到 2 GB 后自动暂停。")
                    .font(.caption).foregroundStyle(.secondary)
            }.card()
            VStack(alignment: .leading, spacing: 14) {
                Text("写入保护").font(.headline)
                if model.isWritingBacklog {
                    LabeledContent("当前积压", value: "\(model.pendingWriteEvents) 条，正在分批入库")
                    ProgressView()
                }
                Text("写入瞬时积压超过安全上限时，会先保护性暂停当前会话，再按最多 128 条的顺序小批次自动入库；完成后自动恢复采集，无需手动处理。")
                    .font(.callout).foregroundStyle(.secondary)
            }.card()
        }
    }
}
private struct InvocationSettingsView: View {
    @Bindable var controller: InvocationController
    let accessibilityAuthorized: Bool

    var body: some View {
        VStack(alignment: .leading, spacing: 16) {
            HStack {
                Text("唤出方式").font(.headline)
                Spacer()
                Text("自动保存").font(.caption).foregroundStyle(.secondary)
            }
            Toggle("晃动鼠标唤出近期目标菜单", isOn: $controller.shakeEnabled)
                .disabled(controller.isCalibrating)
            VStack(alignment: .leading, spacing: 8) {
                HStack {
                    Text("晃动灵敏度")
                    Spacer()
                    Text(controller.sensitivity, format: .percent.precision(.fractionLength(0)))
                        .monospacedDigit().foregroundStyle(.secondary)
                }
                Slider(value: $controller.sensitivity, in: 0...1, step: 0.01) {
                    Text("晃动灵敏度")
                } minimumValueLabel: {
                    Text("较低")
                } maximumValueLabel: {
                    Text("较高")
                }
                .disabled(controller.isCalibrating)
                .accessibilityIdentifier("shake-sensitivity")
                Text("越高越容易唤出；左右快速来回晃动，拖拽时不触发。")
                    .font(.caption).foregroundStyle(.secondary)
            }
            ShakeCalibrationView(controller: controller)
            Divider()
            Toggle("双击 Command（⌘）唤出", isOn: $controller.doubleCommandEnabled)
                .disabled(controller.isCalibrating)
            Text("快速按下并松开同一侧 ⌘ 两次。组合键、长按和鼠标点击会取消识别。")
                .font(.caption).foregroundStyle(.secondary)
            if controller.doubleCommandEnabled && !accessibilityAuthorized {
                Label("全局双击 Command 需要辅助功能权限，可在下方开启。", systemImage: "lock")
                    .font(.callout).foregroundStyle(.orange)
            }
            if !controller.isCollecting {
                Text("开始采集后可唤出近期目标；暂停时也可调整设置和示范校准。")
                    .font(.caption).foregroundStyle(.secondary)
            }
        }
        .card()
        .onDisappear { controller.cancelCalibration() }
    }
}

private struct ShakeCalibrationView: View {
    let controller: InvocationController

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            HStack {
                Button("开始示范") { controller.beginCalibration() }
                    .disabled(controller.isCalibrating)
                if controller.isCalibrating {
                    Button("取消示范") { controller.cancelCalibration() }
                } else {
                    Button("恢复默认灵敏度") { controller.restoreDefaults() }
                }
            }
            if controller.isCalibrating {
                if controller.calibrationSeconds > 5 {
                    Text("准备，1 秒后开始晃动…").font(.callout)
                } else {
                    Text("请左右来回晃动 · 剩余 \(controller.calibrationSeconds) 秒 · 已转向 \(controller.calibrationReversals)/4 次")
                        .font(.callout).monospacedDigit()
                }
                ProgressView(value: Double(controller.calibrationReversals), total: 4)
            }
            if let message = controller.calibrationMessage {
                Text(message).font(.callout).foregroundStyle(.secondary)
            }
            if let calibration = controller.calibration {
                Text("已记录强度：单程幅度 \(calibration.amplitude, format: .number.precision(.fractionLength(0))) 点，4 次转向用时 \(calibration.duration, format: .number.precision(.fractionLength(2))) 秒")
                    .font(.callout)
                Text("示范时间：\(calibration.recordedAt, format: .dateTime.month().day().hour().minute())")
                    .font(.caption).foregroundStyle(.secondary)
            }
            Text("示范成功后自动应用并保存强度；只保存幅度和时间参数，不保存鼠标轨迹。")
                .font(.caption).foregroundStyle(.secondary)
        }
    }
}

private struct Card: ViewModifier {
    func body(content: Content) -> some View {
        content.padding(20).frame(maxWidth: .infinity, alignment: .leading)
            .background(Color(nsColor: .controlBackgroundColor), in: RoundedRectangle(cornerRadius: 14))
    }
}
private extension View { func card() -> some View { modifier(Card()) } }
