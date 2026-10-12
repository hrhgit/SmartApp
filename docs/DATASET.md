# 上下文数据 v2：当前前台采集与历史兼容

## 当前到达采集边界（应用 0.4.0，样本 v4）

以下规则适用于所有新会话，并优先于本文后续保留的历史 v2/v3 描述：

- 只采集并持久化已确认的**前台到达**、其必要层级/定位符和来源质量。不会读取、传输、持久化、导出或训练后台应用、窗口、标签页、窗格、打开库存或独立候选名单。剪贴板管理使用独立私有历史库，不进入本文的到达数据或训练导出；FileRecall 主动查找时的窗口语义与剪贴板输入也不进入本文事件，见 [独立功能边界](CLIPBOARD_AND_FILERECALL.md)。
- 浏览器协议 v2 至多传入 macOS 前台浏览器的一个聚焦普通窗口和一个活动标签页；Finder 只读前台窗口；QSpace Pro 只读前台窗口的活动窗格。后台事件不会生成前台记录。
- `context.change` 可内嵌当次 `foregroundEntrance`，不再创建新 `inventory.change`。旧库存事件只允许留在离线历史归档，不能存在于当前运行库或回填到新样本。
- `NavigationSample.schemaVersion=4`，`labelRuleVersion=observed-foreground-sequence-30s-v4`。新样本的 `inventoryEventIDs` 为空，`inputCoverage.version=foreground-coverage-v2` 且 `status=foreground_only`；这表示刻意没有后台输入，而非确认桌面为空。
- 实际应用只运行 `shared-online-path-mlp-v1` 一套在线学习模型，以已成熟的真实前台结果继续更新。输入含截至预测时已知的前台目标及层级、近期和长期频率、时间、已确认停留与来源质量；候选及执行入口只在内存临时推导，不写回采集库。旧 `semantic-cross-entropy-v3` 不再参与应用运行、学习或保存。

## 前台-only 重建与标准字段

当前 `context-v2.sqlite` 的标准会话必须有 `schemaVersion=2`、`stateCoverageVersion=foreground-coverage-v2`、`targetEligibilityPolicyVersion=target-eligibility-v4`、`observationPolicy=foreground_destinations_only_no_background_state` 和 `backgroundStateCollection=false`，并以 `session.end` 正常结束。

`smartapp-cli rebuild-foreground --database INPUT --output NEW_DATABASE` 从输入库生成新库，不在原库上删列或改行。它只迁入完整的标准会话，剔除旧策略和未完成会话；会移除过时的 `clipboardPolicy` 会话字段及非目标辅助事件。`inventory.change`、候选列表、后台状态键或其引用出现在声称符合当前策略的会话中时，重建会失败而不是悄悄清洗该会话。输出会重放目标准入检查，并核对写出的事件与选择结果一致后才报告成功。

## 观察暂停与登录采集（build 11）

已确认的前台排除范围另记为 `observation.scope` 的 begin/end（`excluded-foreground-v1`，只含随机配对 ID 和粗类别，不含对象身份），不计作 `capture.gap` 故障断点。排除期间不采集目标；原始样本仍在开始时以未知结果结束，恢复后重新观察。离线研究可以在同一会话内核对配对起止和两侧已确认的合格目标，把排除范围当作“下一可推荐目标”序列中的跳过段，无时长上限；派生标签必须保留排除类别和时长，不得写成完整观察到的直接操作或连续停留。旧版普通 `capture.gap` 没有足够信息识别具体原因，原始记录保持原样。

新发生的普通观察中断仍写入 `capture.gap`，但使用 `interruptionVersion=interruption-cause-v1`、同一随机 `interruptionID`、`interruptionPhase=begin/end` 和无对象身份的 `causeCategory` 区分系统状态暂不可观察与前台对象尚未确认。系统状态中断另有 `causeSignals`，明确列出休眠、锁屏、用户会话切换；同时出现多个信号时一起记录，信号变化时分段配对。设置变更、用户会话退出、恢复时目标变化／来源不可用／无法核对、调度或时钟缺口等单次边界也在 `causeCategory` 中说明。类别不包含应用、窗口、页面或文件身份；尚未结束的中断不补造 end，旧版通用暂停／恢复也不补猜原因。分类本身不使普通中断变成可跨越的连续观察。

### 旧版断点可以怎样有限处理

旧数据的原始事件和会话边界不改写。已带成对休眠／锁屏标记，且恢复后确实核对到同一具体目标和入口的区间，可以按现有规则扣除暂停时间并接续；没有恢复核对的区间仍中断。普通 `observation_suspended`／`observation_resumed` 缺少当时的原因，即使间隔很短、两侧是同一目标，也不能据此删除断点、补造中间停留或认定期间没有其他操作。

后续研究只在派生数据里利用旧断点：同一会话内通用暂停／恢复顺序无嵌套、暂停不超过 1 秒、恢复后 1 秒内重新确认前台目标，且没有其他明确中断时，可将两侧已确认目标接成推定的“下一可推荐目标”，并标出断点和间隔。两侧是同一具体目标时不增加独立到达；不同目标的接续也不能写成已完整观察到的直接操作。断点时间不计停留，不能补造中间行为或具体暂停原因。其他断点继续切断相邻关系；断点前已确认的到达仍可用于长期访问频次，序列模型须保留未知区间标记。每种派生处理记录规则版本、受影响条数及训练／评测分母变化；原始库和原始时间保持不变。旧版连续性核查见用户数据目录下私有的 `training/session-continuity-audit-20260929T2200Z/report.md`，其结论仅针对原始记录可否确认为连续观察，不是本次推定训练规则的效果结论。

- 单纯熄屏和空闲不暂停观察。系统休眠、屏幕锁定、用户会话切换分别记录为 `source.status / system_availability` 的 `systemSignal` 与 `active`；多个状态独立维护，屏幕唤醒不会清除仍然生效的休眠或锁定。会话切换不是锁屏通知。
- 正常应用生命周期、某一来源断连/读取失败及后台尚未同步不触发全局暂停。辅助功能权限变化只失效相关窗口来源，已确认合格的应用级及其他健康来源继续观察；来源状态仍如实缺失，不能冒充完整网页或窗口标签。
- 从正常观察进入明确的系统休眠/锁屏时，保存 `capture.gap` 标记，包含 `pauseVersion=known-inactivity-v1`、随机 `pauseID`、`pausePhase=begin/end`、枚举 `pauseReasons`。不包含目标身份。原始 UTC、单调时间与各次系统通知均保留；重叠的睡眠/锁定状态全部解除才写 end。
- 唤醒/解锁后先使旧入口失效并重新读取。同一具体目标和入口得到核对后，记录 `resolution_update`、`continuityPauseID`、`resolvesContextEventID` 和 `resolutionEvidence=same_target_after_known_inactivity`，不形成新到达。当前具体目标尚未取得时不拿应用级快照替代确认，最多等待 10 秒。
- 恢复后目标/入口变化、对应来源失败、恢复超时、无法确认的前台、用户会话切换、主动停止、进程退出及无法解释的调度/时钟缺口仍中断序列。已确认的前台排除按上文单独处理。通知未配对或进程跨会话的记录不拼接；旧 `observation_suspended` 等记录也不追认为休眠。锁屏使用显式锁定通知，未收到通知时不能据熄屏或空闲推断已锁屏。
- 新样本使用 schema 4，`labelRuleVersion=observed-foreground-sequence-30s-v4`：`activeInputCutoffMs` 与 outcome 的 `activeOffsetMs` 使用扣除已记录暂停区间的单调时间；`inputCutoffMs`、`offsetMs` 保留原时间口径。`completedAtMonotonicMs`、`completedThroughSequence` 给出实际成熟事件，评估不能继续用原始锚点加 30 秒提前读取标签。未完整观察到结束的暂停不会成熟为完整样本。
- 前台来源状态在暂停后失效、恢复时重新确认，不把恢复状态填回暂停前。只有行为关联可有条件延续；旧读取器不识别新暂停标记时继续按 capture.gap 保守切断。
- 「同意登录启动后自动开始采集」与「登录时启动 SmartApp」分别设置，默认不同意。仅当两个设置都启用且系统启动事件确认为登录项时开始采集；手动打开仍暂停，显式 `--collect` 保持有效。登录启动不弹出采集面板。

## 历史状态覆盖修复（v3 及更早归档）

原始事件继续使用兼容的 v2 格式；本节记录的是旧 v3 样本和 `state-coverage-v1` 库存快照的解读方式。新采集及新导出以文首的前台 v4 规则为准，仍兼容 v1/v2/v3 的历史准入记录。

- `source.status` 的结构化部分使用 `coverageVersion=state-coverage-v1`。`source_registry` 声明当时运行的合格来源、观察范围和新鲜度上限；普通状态区分 `initializing`、`ready`、`unavailable`、`stale`、`not_applicable`。覆盖 complete 仅指这个已声明范围，不代表整个操作系统及任意应用内部对象。
- 每次成功快照包含 `observationStatus=ready`，包括第一次观察到的空集合。读取失败保存 unavailable，不写伪空快照。未变化的成功检查每约 5 秒保存一次对应快照引用；按来源新鲜度计算过期，不能仅凭快照内容未变判断来源仍健康。
- 局部来源失败只失效其入口；其他来源继续保留。全局休眠、排除、权限边界继续清空关联。恢复观察以补全证据保存，不凭空增加一次到达；原始行为和缺失时段都保留。
- 历史前台与异步后台不保证同时取得。启动或恢复前的样本有明确覆盖标记，后来的快照不会回填过去。当前扩展仅在浏览器为 macOS 前台时以事件触发导航观测；不再有后台连接状态或窗口校验。
- 旧归档中的普通后台工作窗口快照可按当时契约解释；当前采集器不读取或校验后台工作窗口。
- v4 自动识别合格实际工作窗口暴露的本地常规文件 URL，无须预先登记；不读取正文、不枚举磁盘或选择项。文件身份来自路径，入口来自当前窗口；执行前重新核对文档地址，窗口已导航时使用保存地址重开。无法提供文档 URL 的应用仍为窗口级观察。
- `NavigationSample.schemaVersion=3`、`labelRuleVersion=observed-sequence-30s-v2`。`inputCoverage` 保存截至锚点的来源覆盖，`outcomeCoverage` 独立描述后续观察是否可靠；`status=observed_horizon` 本身不证明完整性。`hierarchy` 同时保存应用、容器、具体目标和定位符，按当时目标及入口解析。
- `resolutionEventIDs` 是截点之后的观测补全引用，不是输入特征，也不进入独立行为 outcomes。原始 `resolution_update` 事件仍保留；新记录在有依据时附 `resolvesContextEventID`，不据此宣称确定意图。
- 旧数据没有来源完整性证明时标 `unknown`；旧空列表标 `legacy_empty_unverified`。这不否定其已观察到的目标和序列，也不把从未采到的状态编造出来。新规则由同一个回放器用于后续导出及历史迁移。

可恢复迁移命令及产物说明见[算法研究区](../research/algorithm-research/README.md)。以下为原始 v2 字段背景；与本节不同的失败/空集合或样本规则，以本节新版规则为准。

## 目标准入策略

从 `target-eligibility-v1` 起，各采集入口先判断对象资格，目标和前台上下文写入还会经过统一检查。当前 `target-eligibility-v3` 新增 QSpace Pro 的已观察目录、承载窗口与窗格入口，保留 v1/v2 的合法目标兼容校验。判断结果为允许、排除或未知，未知默认不记录为目标。应用包结构、激活策略、后台标记与窗口角色仅在前台准入检查中使用，不保存原始分类元数据。

- 普通独立应用可以纳入；后台服务、扩展、嵌入式 helper 及内置／用户排除来源不纳入。`SecurityAgentHelper` 本体及其点分子标识强制排除。名称含 Helper、Apple 来源或没有 Dock 图标本身均不构成排除理由。
- 特殊工具需观察到标准、非模态、可激活的独立主窗口；窗口权限或属性缺失时不猜测。已确认的认证／系统对话框、安全输入与隐私来源触发无对象身份的前台排除范围；无法确认前台对象时保留真实观察中断。普通窗口无法识别时只回退到已确认的合格应用。
- 网页限定为受支持浏览器的 macOS 前台普通窗口的活动标签页、HTTP(S) 页面及允许站点；后台浏览器标签页不读取。Finder 自动读取前台文件浏览窗口目标；QSpace Pro 自动读取前台窗口活动窗格的根目录 URL，并确认真实目录；两者均无需手动登记。Finder 隐藏标签页、QSpace Pro 非活动窗格/选择项、目录内容及全盘资源不枚举。普通文件和额外快捷目录仍可手动添加。
- 合格目标的 `admission` 保存 `policyVersion` 和无内容的 `basis`。这是规则判断依据，不代表已确认用户最终意图。旧目标可解码，但不能仅凭旧身份重新获得准入。
- 观察中断清除当前前台关联和临时入口，使所有在途回调失效；恢复后重新观察。拒绝对象的名称、Bundle ID、窗口 ID 不进入缺口事件。

`session.start`、导出清单和序列样本保存 `targetEligibilityPolicyVersion`。原始事件和目标历史导出仍是原始归档，不能直接当作已过滤的训练输入；默认序列样本接受已知 v1/v2/v3 策略会话，按事件当时的准入记录校验引用，并保留各会话策略版本。无策略版本的旧 v2 会话不生成默认样本；无效引用切断样本时间线，清单的 `eventsExcludedFromSamples` 记录被排除的事件数。v1 旧格式读取保持兼容。

推荐菜单在内存中冻结候选及输入截止点。未选择即关闭时不保存唤出、展示或关闭，也不关联随后自然导航；选择时才保存该次菜单与反馈。此时事件时间是写入观察时间，原唤出／展示时刻另存为 `invokedMonotonicMs`／`shownMonotonicMs`，原输入截止点保持不变。

准入和当前可达性分开：展示与执行前复核入口，失效不会删除过去合法的访问。浏览器执行前再读取普通标签页身份，并在激活操作中核对地址；Finder 激活时核对目录；QSpace Pro 激活时核对窗口、窗格和根目录 URL。入口已导航、关闭或过期时改用持久化地址打开原目标；网页交给默认浏览器，QSpace Pro 目录优先由 QSpace Pro 打开，其余目录交给系统文件打开接口。不存在的本地路径不会继续成为候选。系统属性可能缺失或被第三方应用不准确地报告；无法确认目录时只保留窗口。

## 事件格式

事件保存在独立 `context-v2.sqlite`，每行 JSON 有：

| 字段 | 含义 |
| --- | --- |
| schemaVersion | 2 |
| id / sessionID | 唯一事件、采集会话 |
| sequence | 同一采集会话内的写入顺序 |
| timestampMs / monotonicMs | 观察时刻的 UTC 毫秒与系统 uptime 毫秒，不是物理操作的精确时刻 |
| kind / source | 事件类型和来源 |
| app | 能观察到的前台/生命周期应用 Bundle ID、名称 |
| decisionID | 推荐相关事件的关联 ID；时间关联不证明推荐造成了行为 |
| data | 按事件定义的对象，可以含数组、对象、数字、布尔、字符串和 null |

## 主要类型

- `session.start/end`：应用版本、时区、采集策略、轮询间隔、结束原因。
- `target.update`：`data.target` 为目标 ID、类型、显示名、可选应用/站点/定位/父目录等。窗口与标签页入口身份仅在本次观察会话内有效。
- `inventory.change`：仅为历史 v1/v3 归档兼容保留的完整替换快照。新会话不会写入该事件，也不会将其用作新样本输入。
- `context.change`：前台目标与具体入口、变化原因。应用/窗口首次显露后异步补齐网页为 `resolution_update`，不作为另一条独立自然样本锚点。真实的后续页面/标签页变化继续记录。
- `app.activate/launch/terminate`：应用生命周期；不能推断用了哪个快捷键或启动器。
- `clipboard.change` / `clipboard.metadata`：仅为离线旧归档解码保留；当前采集器不写入，前台-only 重建也不会迁入。
- `capture.gap`、`source.status`：采集边界、来源缺失、恢复与权限变化。
- `idle.change`：系统聚合空闲状态，仅一个布尔值和阈值，不包含输入事件；空闲不等于没有阅读。
- `heartbeat`：采集进程存活、是否暂停观察、来源距最近观察的时间。暂停状态心跳不代表完整观察。
- `menu.invoke`、`recommendation.show/feedback`：新会话仅在实际选择时保存选择、执行结果和模型/策略版本；不保存唤出、展示、候选集合或菜单顺序。未选择关闭的会话没有原始事件或样本关联。

网页标识继续使用本地随机密钥 HMAC-SHA256，保持与旧版同一 URL 的 ID 相同。`locator` 保存去除内嵌账号密码后的完整 URL，包括路径、查询参数顺序、重复参数、百分号编码和片段；只归一化协议、主机、默认端口与空路径。`origin` 保存协议/主机/端口，显示名使用 URL。原始事件及训练导出的目标历史均保留 `locator`，可在事件时间点解析网站结构与语义。密钥损坏时不静默创建另一套身份。没有页面正文或窗口/网页标题采集。旧版已丢弃的地址无法从 ID 恢复；再次观察到页面时补齐地址，不倒填历史。

Finder 目录来源为前台 Finder Automation 文件浏览窗口的 `target.URL`，准入依据为 `observed_finder_directory`，承载窗口为 `supported_finder_window`；`locator` 保存标准化完整 file URL，`parentID` 是父目录身份（父目录无需已经成为目标）。不读取后台窗口；同一前台窗口切换目录或标签页按目标变化记录。无真实目录的虚拟视图回退到窗口。

QSpace Pro 目录来源为前台窗口活动窗格的 `root item.urlstr`，仅接受存在的本地 `file://` 目录，准入依据为 `observed_qspace_directory`，承载窗口为 `supported_qspace_window`。入口保存临时窗口 ID 与窗格 ID，类型为 `qspace_pane`；不读取其他窗格或后台窗口。无根目录、非 file URL 或不是目录的活动窗格回退到承载窗口。回放优先按窗口／窗格和期望 URL 激活原窗格；该入口失效时再将保存的目录 URL 交给 QSpace Pro。QSpace Pro Automation 被拒绝或超时时清除该来源入口并记录观察缺口。手动添加仍保存到 `registered-targets.json`，作为快捷目标补充；自动目录不依赖该文件。Safari 内部目标适配尚未实现。

浏览器使用扩展协议 v2 的事件驱动前台观察，不再周期轮询；首次/恢复连接及浏览器成为桌面前台时同步唯一活动页，断线明确形成观察缺口。不会写 `browserState` 或浏览器库存。`destination.settled` 是与到达关联的稳定停留证据，不作为重复访问或额外训练锚点。完整协议见 [浏览器扩展](BROWSER_EXTENSION.md)。Finder 与 QSpace Pro 仍以约 0.5 秒周期观察，应用切换立即触发；不刷新后台入口。两个目录来源各自异步执行、同一来源不重叠；优先由原生辅助进程读取当前前台，单次请求超过 2.5 秒即终止并回退原脚本，最终失败后 30 秒重试。所有来源都不声称无损事件流。快照含 `observationStartedMonotonicMs`；暂停、排除或重新同步之前的旧回调不得恢复为当前访问。

## 序列样本

样本是引用而非复制整段历史：

- `anchorEventID` / `inputCutoffMs` / `inputThroughSequence`：当前目标已观察到、待预测后续目标时的截止点。
- `recentContextEventIDs`：此前五分钟最多 30 个上下文事件。
- `activityContextEventIDs`：此前一小时最多 300 个上下文事件。它是有界的较长历史，不是已经识别出的课程/项目标签；完整原始日志仍保留。
- `inventoryEventIDs`：新 v4 样本恒为空；历史样本中可引用当时库存快照，但不会回填或混入新样本。
- `outcomes`：之后观察到的目标、入口、事件 ID、相对时间与变化原因，禁止用于决策前特征。
- `origin`：自然观察或推荐相关。推荐关联窗口从展示附近开始，选择后延长 30 秒；并不证明因果。
- `status`：`observed_horizon` 表示跨过实验观察时限，`incomplete` 表示会话结束、边界或导出时限中断。仍需结合各数据源状态判断覆盖。
- `labelRuleVersion`：新样本为 `observed-foreground-sequence-30s-v4`。没有唯一最终目的地、负例点击或 `__stay__` 标签。

采集不在 30 秒后结束。30 秒、5 分钟和一小时只是首轮样本生成参数。可以基于完整事件重新构造不同时间跨度。原始样本在真实观察中断处清空跨边界历史，在已确认的前台排除处保留此前历史并标出被排除区间；导出尾部的不完整结果明确保留。离线派生研究可按上文的严格规则补用断点前已观察到的第一步、接续旧版极短断点或跳过配对的排除区间，并与原始样本口径分开报告。

推荐展示输入在唤出时仅在内存冻结；选择后才保存实际选择和执行结果。没有选择不标所有候选为负例，打开成功不等于任务完成。当前在线模型为 `shared-online-path-mlp-v1`，以成熟真实到达的有效第一步反馈继续学习；预测输入保持冻结，不保存候选或展示列表。

## 兼容与存储

v1 的 `events.sqlite` 和旧导出规则保持独立；旧版字段定义见 `LEGACY_DATASET_V1.md`。v2 数据库仍沿用 SQLite 的 events 容器表（物理 user_version 为 1），事件与导出契约版本是 2；两者不是同一个版本号。

文件仅在本机保存。目录 0700，数据与密钥文件 0600；无网络上传。原始日志不自动按天清除，数据库本体、WAL 与 SHM 合计 2 GB 时停止新增。写入以 128 条／1 MiB／10 秒的有界批次提交，观察边界和退出强制落盘。导出为读取事务中的快照，不改变正在进行的采集。

## 心跳摘要与用户操作计数

新采集将约 0.5 秒一次的采集器运行检查合并为约 30 秒一条 `heartbeat`，替代原来的每 10 秒一条。`summaryVersion=collector-heartbeat-v1` 保留实际检查起止单调时间、检查次数及最大检查间隔；`sourceAgesSeconds` 只描述摘要写入时的来源年龄，不证明区间内所有来源均成功，也不倒填早先预测截点。调度中断、暂停和会话结束即时保存边界，未满周期的摘要作为边界上的 `collectorHeartbeatSummary` 保留，严格不含对象身份；不会在中断前补发一条成功心跳来完成未观察到的训练窗口。原始旧数据和已有导出不改写。

来源状态变化及失败原因变化立即记录，未变化的成功确认最短间隔从 5 秒改为 10 秒；来源有效期保持不变，未观察到成功时不得刷新确认时间。现有训练回放继续以实际事件时间推进 30 秒窗口；安静期间周期唤醒的延迟上限从约 10 秒变为约 30 秒，摘要不作为目标到达或独立标签。

界面的「累计目标切换」「本次目标切换」、菜单栏计数及目标时间线使用同一 `UserActivityCounter` 口径：只计合格会话中已观察到的前台目标/入口变化。会话初始和观察中断后的第一条记录作为基线；`resolution_update` 更新基线但不计数。同一次到达的应用激活、目标详情、稳定停留、来源状态、剪贴板元数据及推荐反馈不另计操作。无法确认所属合格会话的旧记录不计入。此指标是可观察的目标切换次数，不声称能还原每次点击、快捷键或用户唯一意图。CLI 的原始 `events` 计数仍作为存储诊断保留，另提供 `userActions`。
