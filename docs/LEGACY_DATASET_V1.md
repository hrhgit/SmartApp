# 事件与训练样本 v1

## 事件 envelope

所有 `events.jsonl` 行都是独立 JSON 对象：

| 字段 | 类型 | 含义 |
| --- | --- | --- |
| `schemaVersion` | integer | 当前为 1 |
| `id` | string | 事件唯一 UUID |
| `sessionID` | string | 每次手动开始的新 UUID；暂停/重启不连接 |
| `timestampMs` | integer | UTC Unix 毫秒，用于跨会话的日历定位 |
| `monotonicMs` | number | 系统 uptime 毫秒，记录回调收到时刻，窗口计算使用此时钟 |
| `kind` | string | `pointer.move` / `app.activate` 等类型 |
| `source` | string | `cg_event_tap`、`nsevent_global`、`ns_workspace`、`accessibility`、`manual`、`lifecycle` |
| `app` | object? | 该输入所在/该结果涉及的 `bundleID` 和 `name` |
| `data` | object | 类型相关的数值、布尔或短字符串 |

时间戳是**观察到事件的时刻**，不是物理设备上的精确按压时间。事件来自不同公开通知链路，交付延迟和前台状态竞态可能影响关联。不得将短时间相关标注成严格因果。

显示器坐标 `x, y` 为 `[0,1]`，原点左上。同一会话可能有多台显示器，以 `displayID` 区分。归一化轨迹长度是采样近似值。显示器 ID 不作为跨设备稳定身份。

`scroll.data.phase` 与 `momentumPhase` 保留 Core Graphics 原始枚举数值；不要与 NSEvent.Phase 的位值混用。`continuous=true` 时 `deltaUnit=points`，否则 `lines`。滚动逐事件保留增量和阶段；20 Hz 限速只用于指针移动/拖动，避免丢失滚动尾部或混合阶段。

键盘事件字段 `category` 为 `typing`、`navigation` 或 `shortcut`：

- `typing` 无字符和物理键码，包含空格和常规字符；仍保留时序及修饰键布尔值。
- `navigation` 保留导航/功能键名称。
- `shortcut` 仅 Command 或 Control 组合保留 `keyCode`，不是键盘布局解析后的文字。
- 从不调用字符提取接口，不采集粘贴内容。快捷键本身仍可能透露用户操作，按个人数据管理。

`gesture.*` 的 `delivery=global_best_effort` 只表示确实收到了事件，不能据此断言其他时间没有做手势。`space.change` 的 `origin=unknown`，不会伪造三指/四指手势类型。

`capture.capabilities` 保存输入/AX 权限及监听器状态。`capture.gap` 表示不得跨越的观察边界，其 payload 不包含被排除应用身份。`heartbeat` 每秒证明采集器仍在运行，采集暂停/被排除/锁屏期间不发送正常心跳。

## 样本构造

输入触发器：鼠标按下、按键按下、滚动、实际收到的手势。触发间隔至少 750 ms。

对于触发时间 `t`：

1. 特征只使用同一会话、同一来源应用、`[t−3000,t]` 范围的输入事件。
2. 在 `(t,t+2000]` 内观察到的第一个不同应用的激活，作为 `targetApp`。
3. 观察完整窗口且没有目标，标为 `__stay__`。
4. 暂停、会话切换、`capture.gap`、倒退时钟、超过 5 秒无事件/心跳，全部丢弃未完成候选并清空前文。
5. 文件结尾剩余候选不导出。未来事件不进入特征。

特征：移动/拖动数量、点击次数、滚动事件数量及绝对增量、键盘类别计数、观察到的手势数量、归一化采样路径长度、最后坐标、窗口内输入数和观测跨度。滚动绝对增量按 points/lines 分成独立特征，不直接混加不同单位。

`sourceApp`、`triggerTimestampMs`、`sessionID` 用于上下文/拆分/追溯。`inputEventIDs` 只属于特征窗口；`outcomeEventID` 与 `outcomeDelayMs` 是结果证据，不能作为预测输入。`targetApp=__stay__` 时无结果 ID。

多次相邻触发可能指向同一激活，是有意保留的候选决策时刻，但高度相关。按会话/日期拆分；计算转移准确率时也建议按 `outcomeEventID` 去重。应用切换不一定由某次输入导致，标签属于弱监督。

## 人工标注

`annotation.data.intent` 与 `note` 为用户主动提交的文本。`anchorEventID` 指向进入 SmartApp 前最近的外部输入/结果。`rangeStartMonotonicMs`、`rangeEndMonotonicMs` 定义人工回顾的 10 秒区间；`sessionID` 是锚点会话。写备注耗时不改变锚点。

自动样本与人工意图没有强行一对一合并。训练脚本可按会话及区间关联人工标签，必要时由人工细化范围和修正错配。
