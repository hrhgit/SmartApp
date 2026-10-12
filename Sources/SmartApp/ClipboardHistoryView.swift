import AppKit
import SwiftUI
import IntentCore

struct ClipboardHistoryView: View {
    @Bindable var model: ClipboardController
    let close: (() -> Void)?
    @State private var confirmingClear = false

    var body: some View {
        VStack(spacing: 0) {
            ClipboardHistoryToolbar(model: model, clear: { confirmingClear = true }, close: close)
            Divider()
            HStack(spacing: 0) {
                ClipboardHistoryList(model: model)
                    .frame(minWidth: 270, idealWidth: 300, maxWidth: 340)
                Divider()
                ClipboardHistoryPreview(model: model)
                    .frame(minWidth: 300, maxWidth: .infinity, maxHeight: .infinity)
            }
            if let error = model.error {
                Text(error).font(.callout).foregroundStyle(.red).textSelection(.enabled)
                    .frame(maxWidth: .infinity, alignment: .leading).padding(12)
            }
            Divider()
            HStack {
                Text("共 \(model.totalCount) 条 · 保存在本机，独立于行为训练")
                Spacer()
                Text("查看历史不会调用 AI")
            }.font(.caption).foregroundStyle(.secondary).padding(12)
        }
        .frame(minHeight: 480)
        .background(.regularMaterial)
        .confirmationDialog("清空全部剪贴板历史，包括收藏？", isPresented: $confirmingClear, titleVisibility: .visible) {
            Button("清空全部历史", role: .destructive) { model.clear() }
            Button("取消", role: .cancel) {}
        }
        .onDisappear { model.clearSelection() }
    }
}

private struct ClipboardHistoryToolbar: View {
    @Bindable var model: ClipboardController
    let clear: () -> Void
    let close: (() -> Void)?

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            HStack {
                Label("剪贴板", systemImage: "clipboard").font(.headline)
                Text(model.isTemporarilyPaused ? "暂时暂停" : (model.isRecording ? "正在记录" : "已暂停"))
                    .font(.caption).foregroundStyle(.secondary)
                Spacer()
                Button(model.isRecording ? "暂停记录" : "开始记录") { model.toggle() }
                    .buttonStyle(.bordered)
                Button("清空", role: .destructive, action: clear).disabled(model.totalCount == 0)
                if let close {
                    Button(action: close) { Image(systemName: "xmark") }
                        .help("关闭剪贴板").accessibilityLabel("关闭剪贴板")
                }
            }
            HStack {
                TextField("搜索文字、链接或文件路径", text: $model.search).textFieldStyle(.roundedBorder)
                Toggle("只看收藏", isOn: $model.pinnedOnly).toggleStyle(.checkbox)
            }
        }.padding(16)
    }
}

private struct ClipboardHistoryList: View {
    let model: ClipboardController
    var body: some View {
        ScrollView {
            LazyVStack(spacing: 4) {
                ForEach(model.summaries) { summary in
                    ClipboardHistoryRow(summary: summary, selected: model.selectedEntry?.id == summary.id,
                        select: { model.select(id: summary.id) }, pin: { model.pin(summary) }, delete: { model.delete(summary) })
                }
                if model.hasMore {
                    Button("加载更多") { model.reload(loadMore: true) }.disabled(model.isLoading).padding(12)
                }
                if model.summaries.isEmpty && !model.isLoading {
                    Text(model.isRecording ? "复制内容后会出现在这里" : "开启记录后，复制的内容会保存在这里")
                        .font(.callout).foregroundStyle(.secondary).padding(24)
                }
            }.padding(8)
        }
    }
}

private struct ClipboardHistoryRow: View {
    let summary: ClipboardSummary
    let selected: Bool
    let select: () -> Void
    let pin: () -> Void
    let delete: () -> Void

    var body: some View {
        Button(action: select) {
            HStack(alignment: .top, spacing: 10) {
                Image(systemName: summary.kind.symbol).font(.title3).frame(width: 24)
                VStack(alignment: .leading, spacing: 6) {
                    Text(summary.title).lineLimit(2).frame(maxWidth: .infinity, alignment: .leading)
                    HStack {
                        Text(summary.observedAt, format: .dateTime.month().day().hour().minute())
                        if summary.origin == .existingTimeUnknown { Text("更新时间未知") }
                        Spacer(minLength: 0)
                        if summary.isPinned { Image(systemName: "pin.fill") }
                    }.font(.caption2).foregroundStyle(.secondary)
                }
            }.padding(10).contentShape(Rectangle())
        }
        .buttonStyle(.plain)
        .background(selected ? Color.accentColor.opacity(0.12) : .clear, in: RoundedRectangle(cornerRadius: 8))
        .contextMenu {
            Button(summary.isPinned ? "取消收藏" : "收藏", action: pin)
            Button("删除", role: .destructive, action: delete)
        }
    }
}

private struct ClipboardHistoryPreview: View {
    let model: ClipboardController
    var body: some View {
        VStack(alignment: .leading, spacing: 14) {
            if let entry = model.selectedEntry {
                HStack {
                    Label { Text(entry.kind.title) } icon: { Image(systemName: entry.kind.symbol) }.font(.headline)
                    Spacer()
                    Text(Int64(entry.byteCount), format: .byteCount(style: .file)).font(.caption).foregroundStyle(.secondary)
                }
                Text(entry.origin == .existingTimeUnknown ? "打开记录时已存在，原更新时间未知" : "观察到更新的时间：\(entry.observedAt.formatted(date: .abbreviated, time: .standard))")
                    .font(.caption).foregroundStyle(.secondary)
                ScrollView {
                    VStack(alignment: .leading, spacing: 12) {
                        if let image = model.previewImage {
                            Image(nsImage: image).resizable().scaledToFit().frame(maxWidth: .infinity)
                        }
                        if let text = entry.text { Text(text).font(.body).textSelection(.enabled).frame(maxWidth: .infinity, alignment: .leading) }
                        if !entry.fileLocators.isEmpty {
                            Text(entry.fileLocators.compactMap { URL(string: $0)?.path }.joined(separator: "\n"))
                                .font(.body.monospaced()).textSelection(.enabled)
                        }
                        if entry.text == nil && entry.fileLocators.isEmpty && model.previewImage == nil {
                            Text("已保留原始格式，可以重新复制。")
                        }
                    }
                }.frame(maxWidth: .infinity, maxHeight: .infinity)
                HStack {
                    Button(model.copiedID == entry.id ? "已复制" : "重新复制") { model.copySelected() }
                        .buttonStyle(.borderedProminent).keyboardShortcut("c", modifiers: .command)
                    if let summary = model.summaries.first(where: { $0.id == entry.id }) {
                        Button(entry.isPinned ? "取消收藏" : "收藏") { model.pin(summary) }.buttonStyle(.bordered)
                    }
                }
                Button("用这条查找目录") { model.findSelected() }.buttonStyle(.bordered)
                    .disabled(entry.text == nil && entry.fileLocators.isEmpty)
                Text("只有明确选择查找时，才把这条内容用于本次目录判断。图片可查看和重新复制；当前目录模型处理文字及文件引用。")
                    .font(.caption).foregroundStyle(.secondary)
            } else {
                Spacer()
                Image(systemName: "clipboard").font(.system(size: 36)).foregroundStyle(.secondary)
                    .frame(maxWidth: .infinity)
                Text("选择一条内容查看").foregroundStyle(.secondary).frame(maxWidth: .infinity)
                Spacer()
            }
        }.padding(18)
    }
}

extension ClipboardContentKind {
    var symbol: String {
        switch self { case .text: "text.alignleft"; case .link: "link"; case .image: "photo"; case .files: "doc.on.doc"; case .richContent: "doc.richtext" }
    }
    var title: LocalizedStringResource {
        switch self { case .text: "文字"; case .link: "链接"; case .image: "图片"; case .files: "文件引用"; case .richContent: "富文本" }
    }
}
