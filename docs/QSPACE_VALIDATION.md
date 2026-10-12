# QSpace Pro 目录采集验证

> 历史验收记录：当前版本只读取前台活动窗格；本文提及的后台窗格库存不再采集。

2026-09-20，QSpace Pro 6.3.2.020，Bundle ID `com.jinghaoshe.qspace.pro`。

## 实现边界

- 只读取实际显露窗格的 `root item.urlstr`，只接受存在的本地 `file://` 目录；不读取 `selection`、目录内容、文件名列表或全盘资源。
- 每个窗格以临时窗口 ID、窗格 ID 和承载窗口关联。前台窗口的活动窗格才形成自然到达；同一窗口的其他窗格及后台窗口只作为上下文库存。
- 无根目录、非 file URL、无效路径或文件位置不提升为目录目标，而是回退为 QSpace Pro 窗口目标。
- 回放先按窗口 ID、窗格 ID 和保存的完整目录 URL 复核并激活原窗格。原窗格不存在、已导航或应用未运行时，才把保存的 URL 交给 QSpace Pro 打开；通用系统打开接口只是该失败链路的最后回退。

## 本机实测

- 静态脚本字典确认 QSpace Pro 提供 `windows`、`panes`、`activated pane`、`root item.urlstr`、窗格 `activate` 与 URL `open` 命令。
- 读取实机窗口快照成功，发现 1 个可见窗口和 2 个可见窗格；读取耗时约 0.31 秒。测试只检查窗格数量、ID 和 URL 是否可用，不将路径输出到此文档。
- 用完整窗口／窗格 ID 重新查找目标、核对目标窗格根 URL 后激活另一窗格；实测结果为 `found=true`、`switched=true`、`restored=true`，并恢复原活动窗格。
- 应用级 URL `open` 会新建 QSpace Pro 窗口，因此实现只把它作为原窗格不可用时的回退；实测完成后原有双窗格工作区保持原状。
- 上述验证直接调用 QSpace Pro Automation，没有启动 SmartApp 采集会话，也没有写入 SmartApp 行为数据。

## 自动回归

`swift test` 通过 72 项：新增测试覆盖真实目录与文件／虚拟窗格的准入差异、前台活动窗格和后台库存分离、现存窗格优先激活、失效窗格回退给 QSpace Pro 打开、策略版本与 Bundle ID 边界，以及 AppleScript 备用打开调用。

如果 SmartApp → QSpace Pro 尚未获 Automation 授权，首次开始采集时 macOS 会单独请求授权；拒绝或超时时面板清楚显示来源不可用、清空该来源入口，并在 30 秒后重试，不会把无法确认的目录写入目标库。
