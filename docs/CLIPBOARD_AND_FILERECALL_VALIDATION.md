# 剪贴板与 FileRecall 接入验证

日期：2026-10-10。版本：0.4.0 / build 14，macOS arm64。实现与边界见 [功能说明](CLIPBOARD_AND_FILERECALL.md)。

## 已执行的检查

- Swift 完整检查：SmartAppTests 71 项（含 1 项默认跳过的真实模型连接检查）、IntentCoreTests 152 项，零失败。单独启用真实模型检查后，该项也通过。
- 浏览器 worker：9 项通过。核对暂停与后台时不查询、当前普通窗口只查一个活动标签、完整 URL、无痕排除、临时查找不生成采集事件；测试替身会拒绝 `populate:true` 和未限定活动标签的查询。
- Python 完整输入校验：2 项通过。使用本机已安装的 Kev 编码器及合成文字，不加载权重；过长内容在任何模型计算前以 422 拒绝，小内容保持完整且缓存关闭。
- 本地模型实际连接：使用合成窗口文字、剪贴板和目录历史，先完成一次请求，再关闭、清空本次内容并重新启动完成第二次请求。两次流程合计 10.058 秒，零失败；没有下载、调用云模型或读取真实剪贴板／页面正文。该时间包含启动和等待旧进程退出，不是交互延迟基准。
- Release 应用构建成功：`build/SmartApp.app` 为 arm64，版本 0.4.0，Bundle ID `local.herh.SmartApp`。固定 `SmartApp Local Release` 身份、稳定指定要求及应用／目录读取 helper 校验通过，浏览器 helper 另行通过严格签名检查。包内包含完整输入校验程序。

## 功能与边界覆盖

剪贴板检查覆盖启动已有内容的未知时间、恢复后基线、版本更新、旧读取失效、删除后不再自动使用、历史显式选择、富文本／多个文件引用格式重新复制、隐蔽标记排除、所有原字节保留、重启读取、收藏、搜索转义及分页不删除。独立历史 130 条全部保留，每页 100 条仅为显示分页。目录与数据库权限分别为 0700／0600；清空、删除执行安全删除及事务日志清理。

前台事件与导出检查确认剪贴板正文不能写入 `context-v2.sqlite`，原行为导出不包含合成剪贴板内容。目录查找只重放明确当前前台策略的会话；单独目标更新、未实际到达目录、旧库存会话、预测截点后的到达与补全重复不会充作目录使用次数。

查找输入检查确认剪贴板文字同时进入临时目录候选和最终模型判断，关闭后清除窗口内容、剪贴板与结果。空窗口内容在模型调用前停止，不能用剪贴板替代读不到的页面继续推荐。普通菜单不含文件／目录，原到达记录继续保留。原 Finder 与 QSpace Pro 执行检查改为 FileRecall 实际选择入口，仍核对优先激活原窗口／窗格和失败后重开。

浏览器原生连接使用合成 TCP 对端核对认证、顺序拒绝和显式查询。显式查询不触发 `onSnapshot`，取消时释放等待；采集暂停也不把查询回复变成行为事件。真实网页正文仅在用户未来主动点击查找时读取，验证过程没有读取个人网页。

## 界面与证据

通过真实 AppKit 窗口离屏渲染合成内容并查看图片：

- `build/menu-previews/menu-{1,3,6}.png`：固定 FileRecall 与剪贴板入口；几何检查确认两工具不与最多六个推荐点击区域重叠，并覆盖屏幕边缘缩放与避让。
- `build/integration-previews/clipboard.png`：搜索、收藏筛选、完整内容预览、重新复制及查找入口。
- `build/integration-previews/filerecall.png`：目录结果、历史文件展开入口及补充描述。

离屏渲染中的玻璃效果不等同于真实桌面的透明度。图中目录、链接、文字及模型概率是合成验证内容，不是个人到达记录或推荐效果报告。

构建目录保留本次日志及源码校验清单：`clipboard-filerecall-tests.log`、`clipboard-filerecall-browser-tests.log`、`clipboard-filerecall-strict-model-tests.log`、`clipboard-filerecall-kev-smoke.log`、`clipboard-filerecall-release.log`、`clipboard-filerecall-source.sha256`。

## 尚未验证与安装状态

没有替换或重启正在运行的旧 SmartApp，也没有修改真实用户行为库、旧 FileRecall 历史或共享模型源码。尚未在用户真实工作窗口中手动走查菜单点击、剪贴板系统授权、当前网页辅助功能树／URL 及无痕切换；目前的界面证据是合成渲染和代码检查。

合成模型调用只证明本机协议、完整输入校验及关闭后重启可用，不证明目录推荐准确率或普通推荐有改善。目录候选沿用既有 `stat-lexical@32`，本次没有做个人行为准确率评测。真实文件通过推荐目录展开选择，尚未单独排序文件。旧 FileRecall 历史与 Terminal／VS Code 独立采集入口未迁入；浏览器需安装并重新加载 0.4 版扩展。

## 重现

```sh
swift test
node --test Tests/BrowserExtensionTests/state.test.mjs
"$HOME/Library/Application Support/FileRecall/kev/.venv/bin/python" Tests/KevStrictServerTests/test_server.py
SMARTAPP_TEST_KEV=1 swift test --filter DirectoryLookupIntegrationTests.testLiveLocalKevWithSyntheticContextWhenRequested
./scripts/build-app.sh release
```

模型连接检查需要本机已有 Kev 环境和缓存，8009 端口没有其他程序的服务；检查只启动并结束自己拥有的进程。安装与扩展更新按 [功能说明](CLIPBOARD_AND_FILERECALL.md) 和 [浏览器安装说明](BROWSER_EXTENSION.md) 执行。
