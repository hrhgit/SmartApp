# 浏览器扩展接入 v2（仅前台活动页）

## 安装

先退出旧 SmartApp，然后在项目根目录运行：

```sh
./scripts/install-app.sh
node scripts/install-browser-extension.mjs
```

打开 Chrome 的 `chrome://extensions`（Edge 为 `edge://extensions`，Brave 为 `brave://extensions`），开启开发者模式，选择「加载已解压的扩展」，选择：

```text
~/Library/Application Support/SmartApp/integrations/browser-v1/extension
```

固定扩展 ID：`hhlflmdclnimbejikiajjfjeeibjnngh`。代码中的公开 manifest key 只用于固定开发版 ID，不是私钥或认证秘密。此版本为本机开发版，未发布浏览器商店。每个需要采集的浏览器个人资料须分别加载。Safari/Firefox 尚不支持。

启动 SmartApp；扩展弹窗应显示已连接。SmartApp 默认暂停；只有开始采集且该浏览器是 macOS 前台应用时，扩展才读取其聚焦普通窗口的活动页。暂停或浏览器退到后台时扩展停止读取，不保存离线浏览历史；恢复前台后采集新快照。更新代码后重新运行安装脚本，并在扩展页面点击重新加载。安装脚本不会修改浏览器 Preferences 或自动授权扩展。

## 分工与事件

0.4 版另支持用户点击 FileRecall 时的一次临时 `inspect_foreground` 核对。扩展声明 `foregroundInspection` 能力，仅回复当前普通活动页的位置；SmartApp 再核对前台归属、网站排除和辅助功能页面 URL。这次回复不触发到达采集，不进入训练或导出。旧扩展继续按原协议采集，但目录查找会提示重新加载 0.4 版。实现使用 `getLastFocused(populate:false)` 后单独查询该窗口的活动标签，不取全窗口标签后再过滤。

- MV3 service worker 使用 tabs/windows/webNavigation 事件；没有 content script、DOM/正文读取、截图、键盘或点击采集。URL 保留查询及片段，去掉内嵌账号密码。非 HTTP(S) 页面不作为目的地。
- `connectNative` 保持一个宿主进程，避免每次事件启动脚本。宿主只转发数据，不持久化浏览记录。SmartApp 的回环 TCP 监听仅绑定 127.0.0.1，随机端口与每次启动随机令牌存于 0700 运行目录的 0600 文件；握手认证后才接收消息。无外网接口、网页可调用端点或远程发送。
- 初次连接、开始/恢复采集、观察缺口后和桌面前台应用变化时请求新快照。平时只由浏览器事件触发，没有浏览器轮询。断线的 30 秒单次重连闹钟只重连通道，不读取标签页。
- 协议 v2 的每条快照至多包含一个聚焦、未最小化的普通窗口及其一个活动标签页。后台窗口、后台标签页、标签页顺序、固定状态和打开库存不会被读取、传输或写入 `inventory.change`。
- 前台到达要求扩展报告该唯一 focused 普通窗口，且 SmartApp 独立确认浏览器是 macOS 前台应用。标签页 active 只表示该前台窗口内的当前页。
- 首次显露/导航产生实际位置变化时记录 `context.change`。同一位置 1200 ms 后进行一次事件关联的重读，满足原生侧持续时间检查才记录 `destination.settled`；后者不是第二次到达，不另建训练锚点，也不是已确认用户意图。
- 同一个 URL 在不同窗口/个人资料中的入口独立；连接 ID、采集会话和进程共同隔离原生编号。选择快捷目标时通过对应扩展核对当前 URL 后激活；已关闭/导航/断线则使用保存地址重新打开。

## 格式与训练

原始数据仍使用 `context-v2.sqlite`。来源为 `browser-extension:<bundleID>:<connectionID>`。

协议 v2 仅发送版本、序号、时间、触发原因，以及上述一个前台页的位置。完整 URL 通过该前台入口关联的 `target.update.locator` 保存，不能用最新目标表倒填过去。原生端把入口嵌入对应 `context.change.data.foregroundEntrance`，不写浏览器库存快照。`source.status` 使用 `foreground-coverage-v2`，仅表明当前前台来源的观察质量。

新序列样本为 schema v4：`inventoryEventIDs` 恒为空，`inputCoverage.status=foreground_only` 明确表示没有后台状态输入。`destination.settled` 通过 contextEventID 关联到达，携带稳定停留证据而不是确认意图。在线推荐使用动态标签 Softmax 的加权交叉熵更新；执行时才解析已知入口或保存的地址。

## 边界

这是事件触发的前台状态观测，不承诺浏览器每次重定向都完整记录：API 查询与事件是异步的，极短中间状态可能合并；事件 reason 表示读取的触发原因，不证明因果。后台事件若没有改变前台页不会产生记录。一次最多排队 32 次查询，超限断开并记录观察缺口。协议对帧大小、版本、顺序、元素数量和身份字段做检查；断线/暂停或失去 macOS 前台资格后不重放旧快照。无痕扩展禁用，不能读取无痕 URL。

新版本不在扩展断线时回退 AppleScript 扫描。未安装扩展时只有浏览器应用级观测，面板显示等待连接。Finder 与 QSpace Pro 同样只读取当前前台窗口／活动窗格。

## 测试

```sh
node --test Tests/BrowserExtensionTests/*.test.mjs
swift test
./scripts/build-app.sh release
```

官方协议与事件依据：[Native Messaging](https://developer.chrome.com/docs/extensions/develop/concepts/native-messaging)、[tabs](https://developer.chrome.com/docs/extensions/reference/api/tabs)、[webNavigation](https://developer.chrome.com/docs/extensions/reference/api/webNavigation)。
