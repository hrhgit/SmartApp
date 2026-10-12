# 浏览器扩展验收 · 2026-09-18

> 历史验收记录：当前扩展协议为 v2，只读取 macOS 前台浏览器的活动页，不再采集后台库存。

## 安装结果

- SmartApp 0.2.0 build 7 已安装到固定路径 `~/Applications/SmartApp.app`，主应用及原生宿主使用既有本地签名身份。签名校验通过，实际应用辅助功能权限仍为 true。
- 原应用备份：`~/Applications/SmartApp.backup.20260918-154818.app`。
- Chrome 已通过扩展管理页加载 SmartApp Browser Context 0.1.0，ID `hhlflmdclnimbejikiajjfjeeibjnngh`，页面显示 enabled，原生宿主进程已实际建立。
- Native Messaging 已注册 Chrome/Edge/Brave；本轮只在 Chrome 做实机验证，Edge/Brave 兼容代码未做实机验收。
- 用户已明确同意扩展读取普通窗口标签页 URL/状态并仅发送本机 SmartApp。开发者模式已开启，未发布到扩展商店，未修改其他扩展设置。

## 实际浏览与存储

测试会话：`55D022D2-816F-40AD-B5F6-D8636DACCC5C`。测试 HTTP 服务仅绑定 127.0.0.1:63398；服务已停止。以下均为合成地址：

1. `http://127.0.0.1:63398/?smartapp_extension_test=entry#start`：完整地址入库；实际前台到达及与其关联的稳定停留记录均存在。
2. 点击页面按钮执行 history.pushState，标题保持不变：`/detail?smartapp_extension_test=spa#section` 入库。
3. 只改变 fragment：`/detail?smartapp_extension_test=spa#second-section` 入库。
4. 浏览器在后台期间，上述 SPA/fragment 变化只更新库存，不产生前台 context.change；此项根据实际系统前台状态与数据库核对，不把自动化操作等同于用户访问。
5. 使用链接菜单 Open Link in New Tab 新建真正未选中的标签页：UI 确认新标签 off、原标签 on。数据库库存包含新入口，清理前核对其 tabID 1588871041 的 context.change 次数为 **0**；另两个已访问测试入口各为 **1**。
6. 页面地址使用此前 target.update 与当前入口关联；库存保留窗口归属、选中入口、顺序、固定状态和浏览器前台状态。后台库存和访问标签分离。

验收期间用户也使用了原有 Chrome 页面，因此该会话不是纯合成数据集。不要把整段会话当作干净训练/测试样本；项目记录只保留合成地址与汇总，不复制用户原有页面内容或 URL。无需删除真实数据库；原始验收会话可在训练划分中明确排除。

## 自动化覆盖

- Node：状态序列化、完整 URL/凭据去除、隐私/特殊窗口过滤、暂停/恢复、在途快照失效、后台标签页、顶层 SPA/fragment 事件、一次性稳定重读与位置变化不误判。
- Swift：帧拆包/多包/尺寸上限、真实回环连接的握手认证与重复序号拒绝、跨个人资料入口隔离、排除页面不落盘、状态改变去重、稳定事件不重复生成访问锚点、SQLite/导出与输入库存时序。
- 完整 `swift test` 与 `node --test Tests/BrowserExtensionTests/*.test.mjs` 用于回归；构建与固定签名验证通过。

## 收尾与边界

测试生成的三个浏览标签页和扩展安装页已关闭；用户原有页面保留。SmartApp 恢复到测试前的**暂停**状态。扩展和本地宿主安装保留，点击 SmartApp「开始采集」即可启用。

没有浏览器周期扫描。断线闹钟只重试连接；有限的 1200ms 稳定重读属于事件后检查。异步事件查询不是无损导航日志；多个人资料同时前台切换、浏览器重启、长时间运行以及本机实际快捷菜单激活扩展标签页仍需更长实机覆盖，不能由本轮结果推断全部通过。现有 fallback 打开保存 URL 的回归测试仍通过。

Finder 未在此次扩展接入中改为事件驱动，仍保持已安装的原采集实现。
