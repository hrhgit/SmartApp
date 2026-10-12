# 分差校准与动态推荐菜单

状态：算法已接入 `FastRecommendationService`、`RecorderModel` 和 `DestinationPickerView`。继续使用 `shared-online-path-mlp-v1` 模型的原始排序分数，菜单数量按门槛变化。用户确认漏选目标 10%、暂时最多 6 项；第一项从鼠标右上方开始，后续按固定 60° 间距顺时针沿圆弧展开，数量少时不均分整圈。没有最低数量，不补齐到 4 项。

本次已完成合成数据检查、原生视图布局检查和应用构建；尚未以个人真实行为测量动态菜单的实际漏选率、平均数量或长期效果。10% 是调整目标，不是单次置信度或已经达到的实际表现。

## 输入与选择

0.4 版默认菜单范围为 `app-window-page-v1`：只保留应用、窗口和网页动作；目录与文件交给固定 FileRecall 工具入口。菜单预览和自然到达时冻结的校准分数都使用这一范围。本地到达仍进入原在线模型学习和前台历史，但不作为普通菜单的漏选反馈。匹配规则版本包含菜单范围，旧校准状态重新积累，有效权重保留；FileRecall 的目录选项概率不参与这套分差调整。

调用方提供预测当时的临时目标分数，以及模型、输入字段、匹配规则的版本。算法按照分数降序排列；同分时按动作 ID 排序。分差定义为：

```text
目标分差 = 本次最高分 − 该目标分数
显示条件 = 目标分差 ≤ 已校准门槛
```

算法默认不限制数量，调用方可显式提供 `maximumCount`；应用按用户要求传入 6。先按分差门槛选择，再取最多 6 项；不补齐数量，不把分数转成概率。分数、预测和返回目标列表只留在内存，不提供 `Codable`；可保存的 `ScoreMarginConformalState` 只含派生分差、版本、参数和汇总，不含目标身份或列表。算法本身不读取或写入文件、事件库、菜单状态或后台库存。

菜单先沿用已有规则核对目标资格、打开入口、是否改变当前目标并去除重复入口，再以合格目标中的最高分计算分差、筛选和应用上限。这是只读 `preview`，不会发出学习预测或更新校准。目标被筛掉后可让原本较后位置的合格目标进入六项范围，不会为了凑数量加入门槛外目标。模型不可用时沿用原最近访问排序，展示上限同样为 6。

## 校准

一次独立、已确认且合法的后续到达对应一条 `ScoreMarginObservation`。先创建不可变的 `ScoreMarginPrediction`，后续确认到达后再调用 `observe`。每条记录只保存该次结果的分差及核对版本、顺序、重复反馈所需的标记，不保存候选列表。

调用方使用现有唯一评分入口确定 `acceptableActionIDs`，算法不重新判断网址、目录、服务或上级的关系：

- 校准具体目标完全一致的覆盖时，传入真实具体目标的动作 ID。
- 校准合法路径成功的覆盖时，传入真实具体目标及按现有评分规则能得分的上级动作 ID。同一次行为只取其中最高分动作的分差，不复制成多条样本。
- 两种口径使用不同的 `matchingRule`，不能混合拟合或复用门槛。部分收益继续由现有评分入口计算，不由该算法指定。
- 正确目标及可得分上级都不在当时的评分列表中时，分差记为正无穷；该次行为仍留在样本总数中，并单独计入 `unavailableOutcomeCount`。即使选择全部已有目标，该情况仍然是漏选。
- 尚未确认后续到达、观察中断、无合法目标变化或关闭菜单未选择时，不生成校准样本。空的匹配 ID 集合会被拒绝，不能用它代替未知结果；已确认合法结果且没有任何匹配入口时，调用 `observeUnavailableOutcome` 明确计为缺失正确入口。

漏选比例 `targetMissRate = α` 必须显式提供，满足 `0 < α < 1`。对 `n` 条校准行为，门槛为升序分差中的第 `ceil((n + 1) × (1 − α))` 条，采用从 1 开始的编号；边界相等也保留。如果编号超过 `n`，或者对应分差为正无穷，结果明确为 `allCandidates`，不会虚构有限门槛。

校准行为必须发生在该模型训练所用行为之后。批量校准只接受同一模型权重修订、训练边界、输入及匹配规则；选择时还核对该预测发生在校准行为确认之后。重复预测或同一到达反馈被拒绝。调用方必须提供真实的训练边界和唯一时间顺序。

有限样本覆盖结论需要校准与测试数据满足相应条件，个人连续行为通常有关联，正确入口也可能缺失。这里的漏选比例是校准目标，不是每次推荐的置信度；实际覆盖与列表大小仍需同口径验证。

## 调用示例

以下标识、序号和分数都是人工示例。实际调用应使用真实权重修订及现有评分规则的版本或哈希。

```swift
let context = try ScoreMarginContext(
    modelVersion: "example-ranker-v1",
    featureSchema: "example-input-v1",
    matchingRule: "example-exact-v1")

func forecast(_ id: String, at sequence: Int64, scores: [Double]) throws -> ScoreMarginPrediction {
    try ScoreMarginPrediction(
        predictionID: id, context: context, modelRevision: "example-weights-1",
        trainedThroughSequence: 0, cutoffSequence: sequence,
        candidates: scores.enumerated().map {
            ScoreMarginCandidate(actionID: "action-\($0.offset)", score: $0.element)
        })
}

// 两次结果的分差分别为 1 和 0；每次结果在预测之后确认。
let first = try forecast("prediction-1", at: 10, scores: [9, 8])
let second = try forecast("prediction-2", at: 12, scores: [9, 9])
let observations = [
    try first.observe(outcomeID: "arrival-1", outcomeSequence: 11,
                      acceptableActionIDs: ["action-1"]),
    try second.observe(outcomeID: "arrival-2", outcomeSequence: 13,
                       acceptableActionIDs: ["action-1"])
]
let calibration = try ScoreMarginCalibration(observations: observations, targetMissRate: 0.4)
// 修正后的编号为 ceil(3 × 0.6) = 2，门槛为 1。
let current = try forecast("prediction-3", at: 20, scores: [9, 8.75, 8.5, 5])
let decision = try calibration.select(current)
let selected = decision.selected  // action-0、action-1、action-2
```

## 独立在线调整

`AdaptiveScoreMarginConformal` 可以从已完成的校准结果或无样本状态开始。`learningRate = γ` 必须显式设置为有限正数。应用当前显式使用 `γ = 0.005`，它是本次实现的调整速度，尚未通过个人行为选参。更新方向为：

```text
miss = 当时选择的目标是否都没有匹配该次真实结果（漏选 1，覆盖 0）
α_next = α_current + γ × (α_target − miss)
```

每次先用过去的分差和当前 `α` 选目标，冻结该次选择；收到真实结果后，依据当时的门槛和显式数量上限判断漏选，再更新 `α` 并加入一条新分差。通过门槛却排在第 7 位及以后的正确入口仍是六项策略的漏选，不得当作已展示覆盖。未来信息不会重算旧预测。同一模型与输入定义下允许权重修订继续变化；更换模型、输入或匹配规则需要重新建立校准。

```swift
var adaptive = try AdaptiveScoreMarginConformal(calibration: calibration, learningRate: 0.05)
let frozen = try adaptive.select(current)
// 确认结果后才更新；不能用展示或关闭菜单代替真实到达。
let covered = try adaptive.observe(frozen, outcomeID: "arrival-3", outcomeSequence: 21,
                                   acceptableActionIDs: ["action-2"])
```

支持同时存在多个待确认预测，但反馈或丢弃必须按发出预测的顺序处理；重复反馈、其他控制器的选择及时间倒退会被拒绝，失败不改变校准状态。没有已确认结果时调用 `discard`，仅释放待处理标记，不增加校准行为，也不更新门槛。

`α` 不被强制夹回 `[0, 1]`。`α ≤ 0` 的门槛允许全部现有目标，应用仍显示最多 6 项；`α ≥ 1` 明确选择空集合，菜单显示暂无推荐目标。这些是算法边界。正确入口缺失或六项上限造成的漏选不能仅靠放宽门槛解决，因此不能照搬完整标签空间下的长期覆盖保证。延迟反馈、持续变更权重和相关行为的实际效果也尚未测量。

校准分差与本次运行的去重标记不按历史长度或超时截断。批量拟合排序需要 `O(n log n)`；每次在线插入需要 `O(n)`。本次未另外引入时间兜底、数量补齐或旧模型。

## 应用中的时间边界、保存与汇总

应用逐条读取已提交的前台历史，模型训练截止前的行为只恢复历史输入，不再充作校准行为。截止之后，每次合法到达时冻结原分数和临时候选；后续第一步真实行为通过现有 `OnlinePathFeedbackLearner` 确认成熟，再按 `OnlinePathInputHistory.usefulActionIndices` 的 `online-path-useful-actions-v1` 规则判断具体目标或合法上级。新确认的具体目标可随后加入模型训练，不能追加进先前校准预测。

新到达的预测先冻结，再处理该条事件刚确认成熟的旧反馈，保持旧分数与旧门槛。预测按数据库全局序号核对先后；多个待确认预测按发出顺序处理，不能用各会话从头计数的序号比较跨会话时间。观察中断而未确认结果的预测丢弃，不当成漏选。

校准状态与当前模型权重、训练截止事件一起原子写入 `models/shared-online-path-mlp-v1/state.json`。仅保存分差（无穷分差编码为 `null`）、版本、参数和汇总；不保存预测/结果/动作身份、URL、路径、候选或菜单清单。重启后恢复已完成反馈的校准分布，未完成预测不持久化，也不用新权重重算旧预测。旧权重没有校准字段时正常启动，无样本门槛允许全部历史候选，菜单仍最多 6 项；随着新确认行为积累再调整。校准版本、规则、参数或状态内容不匹配时报告原因，保留有效排序模型并重新积累校准。

`runtime/recommendation-status.json` 增加以下无目标身份的汇总：

| 字段 | 含义 |
| --- | --- |
| `dynamicMenuMaximumItems` | 用户确认的展示上限，当前 6 |
| `dynamicMenuTargetMissRate` | 调整目标，当前 0.1，非实际漏选率 |
| `marginCalibrationSamples` | 分差分布中的已完成合法行为数，包含正确入口缺失 |
| `marginUnavailableOutcomes` | 预测时具体目标及合法上级均不在评分列表的行为数 |
| `marginFeedbackCount` | 用冻结预测判断覆盖并更新调整值的反馈数 |
| `marginFeedbackThresholdCovered` | 原分差门槛覆盖的反馈数，未应用六项上限 |
| `marginFeedbackCovered` | 门槛和六项上限共同覆盖的反馈数 |
| `marginFeedbackCapLimited` | 门槛可覆盖、六项上限导致漏选的反馈数 |
| `marginCalibrationError` | 校准状态或反馈校验错误 |

这些覆盖汇总来自自然到达时冻结的预测，不是菜单调用次数或实际展示清单统计；菜单还会核对执行入口，且可以在同一目标停留期间反复调用。因此不能把这些汇总直接宣称为实际菜单达到九成覆盖。唤出、展示和关闭菜单均不增加校准样本；实际选择和确认到达仍走原反馈流程。

## 验证与后续效果评估

`ScoreMarginConformalTests` 的 22 项检查包括有限样本编号、逐个留出所有可能分差排名的覆盖结果、同分与负分、分数整体平移、边界包含、不同选项数量、正确入口缺失、未知结果丢弃、训练与结果时间边界、版本隔离、旧选择冻结、反馈顺序与去重、在线更新方向、六项上限漏选和纯分差状态恢复。

应用接入检查覆盖：恢复门槛后只显示 1 项且不补齐 4 项、重复菜单读取不学习、关闭不写菜单数据、校准与模型共同保存并重启恢复、坏校准不弃用有效模型、连续提交不漏处理；布局检查覆盖右上起点、顺时针固定位置、六个点击区域不重叠、屏幕边缘及小屏避让。`build/menu-previews/menu-{1,3,6}.png` 是用合成目标和本机应用图标生成的原生视图预览，不含个人推荐目标。

```sh
swift test --filter 'ScoreMarginConformalTests|DynamicRecommendationIntegrationTests|DestinationPickerLayoutTests|LiveOnlinePathIntegrationTests'
swift test
./scripts/build-app.sh release
```

当前验证结果见 [动态菜单接入验证](DYNAMIC_MENU_VALIDATION.md)。后续比较真实动态列表效果，应在预测截点临时核对同一份合格且会改变目标的入口，真实目标路径关系继续通过现有共享评分入口确定。保留全部合法样本及既定有依据分组，同时报告成功、完全一致及服务接受的 Top1/Top2/Top4、动态列表实际覆盖、平均数量与数量分布，并与相同样本的固定四项及无情景统计对照。不能把列表变长带来的覆盖增加说成排序模型更准确，也不能忽略六项限制或正确入口缺失造成的漏选。

算法依据：[保形预测通用分差/分位数框架](https://arxiv.org/abs/2107.07511)、[自适应保形推断的反馈更新](https://arxiv.org/abs/2106.00170)。分差函数和暂缺正确入口的处理是本项目的明确选择。
