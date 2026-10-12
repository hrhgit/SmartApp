import SwiftUI

struct DirectoryLookupView: View {
    @Bindable var model: DirectoryLookupController
    let close: () -> Void
    let back: () -> Void

    var body: some View {
        VStack(spacing: 0) {
            DirectoryLookupHeader(close: close, back: back)
            Divider()
            DirectoryLookupStatus(status: model.status, clipboard: model.clipboardDescription, uncertain: model.isUncertain)
            if model.isLoading {
                HStack(spacing: 12) { ProgressView(); Text(model.status).foregroundStyle(.secondary) }
                    .frame(maxWidth: .infinity, maxHeight: .infinity)
            } else {
                DirectoryLookupResults(model: model, close: close)
            }
            if let error = model.error {
                VStack(alignment: .leading, spacing: 8) {
                    Text(error).font(.callout).foregroundStyle(.red).textSelection(.enabled)
                    if !model.isLoading && model.canRefine { Button("用补充描述重试") { model.refine() } }
                }.frame(maxWidth: .infinity, alignment: .leading).padding(16)
            }
            DirectoryLookupRefinement(model: model)
            Divider()
            HStack {
                Text("↑↓ 选择 · ↵ 打开 · ⌘C 复制路径")
                Spacer()
                Text("在本机判断 · 仅使用已观察路径")
            }.font(.caption).foregroundStyle(.secondary).padding(12)
        }
        .background(.regularMaterial, in: RoundedRectangle(cornerRadius: 20))
        .onExitCommand(perform: close)
    }
}

private struct DirectoryLookupHeader: View {
    let close: () -> Void
    let back: () -> Void
    var body: some View {
        HStack(spacing: 12) {
            Button(action: back) { Image(systemName: "chevron.left") }.help("返回推荐菜单").accessibilityLabel("返回推荐菜单")
            Label("FileRecall · 目录与文件", systemImage: "folder.badge.magnifyingglass").font(.headline)
            Spacer()
            Button(action: close) { Image(systemName: "xmark") }.help("关闭").accessibilityLabel("关闭目录查找")
        }.padding(16)
    }
}

private struct DirectoryLookupStatus: View {
    let status: String
    let clipboard: String?
    let uncertain: Bool
    var body: some View {
        VStack(alignment: .leading, spacing: 5) {
            Text(status).foregroundStyle(uncertain ? .orange : .secondary)
            if let clipboard { Label(clipboard, systemImage: "clipboard").foregroundStyle(.secondary) }
        }.font(.caption).frame(maxWidth: .infinity, alignment: .leading).padding(.horizontal, 16).padding(.vertical, 10)
    }
}

private struct DirectoryLookupResults: View {
    let model: DirectoryLookupController
    let close: () -> Void
    var body: some View {
        ScrollView {
            LazyVStack(spacing: 6) {
                ForEach(model.rows) { row in
                    DirectoryLookupResultRow(row: row, selected: model.selectedID == row.id, copied: model.copiedID == row.id,
                        select: { model.select(row.id) }, copy: { model.copy(row) },
                        open: { Task { if await model.open(row) { close() } } },
                        openFile: { file in Task { if await model.openFile(file) { close() } } })
                }
            }.padding(.horizontal, 12)
        }.frame(maxWidth: .infinity, maxHeight: .infinity)
    }
}

private struct DirectoryLookupResultRow: View {
    let row: DirectoryLookupRow
    let selected: Bool
    let copied: Bool
    let select: () -> Void
    let copy: () -> Void
    let open: () -> Void
    let openFile: (ObservedRecallFile) -> Void

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            HStack(spacing: 10) {
                Button(action: copy) { Image(systemName: copied ? "checkmark" : "doc.on.doc") }
                    .buttonStyle(.plain).help("复制完整路径").accessibilityLabel("复制完整路径")
                Button(action: open) {
                    HStack {
                        Image(systemName: "folder")
                        Text(row.path).font(.callout.monospaced()).frame(maxWidth: .infinity, alignment: .leading)
                        Text(row.probability, format: .percent.precision(.fractionLength(1))).font(.caption.monospacedDigit())
                            .foregroundStyle(.secondary)
                            .help("本地模型在这次比较中给出的选项概率，不是实测准确率")
                    }.contentShape(Rectangle())
                }.buttonStyle(.plain)
            }
            if !row.files.isEmpty {
                DisclosureGroup("\(row.files.count) 个曾在前台观察到的文件") {
                    VStack(alignment: .leading, spacing: 6) {
                        ForEach(row.files) { file in
                            Button { openFile(file) } label: {
                                Label(file.target.name, systemImage: "doc").frame(maxWidth: .infinity, alignment: .leading)
                            }.buttonStyle(.plain)
                        }
                    }.padding(.top, 6)
                }.font(.caption).padding(.leading, 28)
            }
        }
        .padding(10)
        .background(selected ? Color.accentColor.opacity(0.10) : .clear, in: RoundedRectangle(cornerRadius: 9))
        .onHover { if $0 { select() } }
    }
}

private struct DirectoryLookupRefinement: View {
    @Bindable var model: DirectoryLookupController
    var body: some View {
        HStack {
            Image(systemName: "text.bubble").foregroundStyle(.secondary)
            TextField("补充项目名、文件名或当前任务", text: $model.refinement).textFieldStyle(.roundedBorder)
                .onSubmit { model.refine() }.disabled(model.isLoading || !model.canRefine)
            Button("重新查找") { model.refine() }.disabled(model.isLoading || !model.canRefine)
        }.padding(12)
    }
}
