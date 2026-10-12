# 网页与 Finder 目录自动采集验收

> 历史验收记录：其中的后台库存行为属于当时版本，已不适用于当前前台专用采集器。

2026-09-18，SmartApp 0.2.0 build 6，`target-eligibility-v2`。

## 已交付行为

- Chrome / Edge / Brave 自动保存完整 HTTP(S) URL，包括路径、查询和片段；保留旧 ID 算法，旧记录再次观察后补齐地址，不倒填历史。
- Finder 自动读取文件浏览窗口当前目录，不要求手动登记；同窗口切换目录／显露标签页产生目标变化，后台窗口只进入入口库存。
- 持久化 `locator` 进入原始记录、目标历史和训练导出。地址可在重启后恢复；失效标签页／关闭窗口回退到保存地址重新打开。
- 前台约 0.5 秒轮询，应用切换立即刷新，后台约 5 秒补充。异步请求不重叠，6 秒超时，失败后 30 秒重试；并非无损事件流。
- 面板中的手动操作改为可选的“添加快捷目标”。规则同步到根目录 `AGENTS.md`，替换先前不符合需求的训练脱敏约束。

## 自动化验证

`swift test`：63 项通过（IntentCore 48，SmartApp 15）。

覆盖：完整 URL 编码、重复参数与片段；页面／标签页身份；Finder 未登记目录、中文和特殊字符路径、文件／虚拟目标回退；父目录关联；后台不产生访问；重复观察去重；SQLite、导出与重启；网站排除；已导航标签页／已关闭 Finder 窗口的重新打开调用；旧策略样本兼容；异步 helper 超时；系统真实 Finder 的 `FNDR` 包类型准入。

`./scripts/build-app.sh release` 与本机安装完成，签名严格校验通过。安装产物与构建二进制 SHA-256 相同：

```text
06281d7c53548da17a9fc625e378215e539d69b545a6fe5efd6feb519f28d745
```

## 实机验证

用户已完成 Finder 自动化授权与新版辅助功能重新授权。实际安装位置为 `~/Applications/SmartApp.app`；运行诊断确认 `collecting=true`、`accessibility=true`。真实数据源状态为“页面与标签页已连接”和“目录导航已连接”。

在独立 Chrome 测试标签页、独立 Finder 测试窗口中验证，SQLite 及 `target-history.jsonl` 均保留：

```text
https://example.com/?smartapp_capture_test=20260918#first
https://example.com/?smartapp_capture_test=20260918&step=second#child-route
file:///tmp/smartapp-live-capture.FMrIai/
file:///tmp/smartapp-live-capture.FMrIai/child/
```

首个网页确有前台访问记录。Finder 父目录首次解析记为 `resolution_update`，同窗口进入子目录记为 `directory_or_window_change`。第二个网址变更实际在后台完成，只更新目标库存，没有误记为前台访问。两条初次前台快照从请求开始到写入的耗时约 503 / 529 毫秒，不能等同于用户操作到采集的完整延迟。

运行中的数据库成功导出快照：3270 条事件、77 条目标历史、125 个样本（75 完整、50 不完整），样本准入排除事件数 0。快照仅保存在本机临时验收目录 `/tmp/smartapp-live-capture.FMrIai/export`，不进入源码。真实 Finder 自动目录的准入依据为 `observed_finder_directory`。

测试标签页和 Finder 测试窗口已关闭，原有窗口保留。验收会话 `E17A21A0-82F5-4578-9DA6-732F884C114C` 已结束，重新开启日常采集。训练时应排除该验收会话及上述合成目标；原始数据未清理，测试目标可能仍存在于历史目标库。

旧 build 5 安装包保存在 `~/Applications/SmartApp.backup.20260918-144628.app`；后续中间构建另有安装脚本备份。源码修改前快照在 `/tmp/smartapp-capture-backup.bT6Ntr/source-before.tar.gz`。未删除用户行为数据或偏好。

## 验证范围

实机验证使用 Chrome 与 Finder；Edge / Brave 共享 Chromium 适配器，本次未单独启动验证。Safari 尚未适配；QSpace Pro 的后续目录适配与独立实机脚本验证见 [QSpace Pro 目录验证](QSPACE_VALIDATION.md)，不追溯写入本次 2026-09-18 验收会话。窗口关闭后的执行回退用注入系统打开接口的集成测试验证，没有宣称所有浏览器／文件管理器的原生 UI 均已验收。
