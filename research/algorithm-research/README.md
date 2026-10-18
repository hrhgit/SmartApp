# SmartApp 算法研究区

> 当前产品采集边界是仅前台到达，运行时模型为 `semantic-cross-entropy-v3`。本目录中引用后台状态/库存的研究和历史快照只可用于既有归档的离线、私有对照，不能修改采集器、不能要求新增后台采集，也不能把历史结果作为当前产品能力声明。

这里存放**可复现的研究代码、方法说明与实验约定**；不存放真实用户行为日志、完整 URL、目录路径、身份密钥或生成的分析结果。

真实数据、规范导出快照及训练产物始终留在：

```text
~/Library/Application Support/SmartApp/
├── exports/context-v2/<UTC 时间>-<UUID>/   # 不可变原始快照
└── training/<run ID>/                       # 本次分析、特征、评估与模型产物
```

这与 [数据目录规范](../../docs/DATA_DIRECTORIES.md) 一致：源码目录只保存可审查的算法实现，个人数据和派生产物不进入源码或构建目录。

## 断点清洗与离线模型研究

`scripts/cleaned_foreground_data.py` 从不可变原始快照派生训练记录：补用中断前已确认的下一步；仅在同一会话、无嵌套且起止可核对时接续不超过 1 秒的旧版短断点；配对的前台排除区间可跳过，时长不设上限。接续后的两个目标是相邻的**可推荐目标**，不代表中间没有发生其他操作。断点和排除区间的时间不计为停留，恢复到同一具体目标不增加一次到达。所有新训练记录保留原事件 ID、来源、目标层级、观察质量和完整观察是否结束的证据，完整明细只存私有 training run。

`scripts/run_break_cleaning_study.py` 在原有固定 598 条训练、149 条测试的切分上比较原清洗、只补第一步、只接续短断点、两者合用，以及接续样本半权重的对照。原测试成员和统一评分入口保持不变；测试期额外可补的第一步只作另一组诊断。统计模型、模拟逐步更新的统计模型与批量 CatBoost 分别报告；序列模型的时间、层级、频次、已确认停留和来源质量输入由 `scripts/sequence_asof_features.py` 按每个预测截点构造。批量模型不拿测试期答案训练，逐步更新只吸收已经发生且成熟的结果；将来才知道的停留或断点时长不能进入此前输入。

```sh
<private-python-with-torch-and-catboost> research/algorithm-research/scripts/run_break_cleaning_study.py \
  --snapshot "/absolute/path/to/exports/context-v2/<UTC>-<UUID>" \
  --split "/absolute/path/to/training/offline-batch-split-v1-<run>" \
  --batch-selection "/absolute/path/to/training/offline-batch-rankers-v1-<run>" \
  --output "$HOME/Library/Application Support/SmartApp/training/break-cleaning-study-v1-<new-run>"
```

报告先列固定测试中预测前已有可得分路径的主榜，再列全部合法样本和测试期新增样本诊断。`success` 表示推荐到预测前可打开的真实目标或有用上级；`exact` 只表示目标身份完全一致，单凭猜中未来才出现的 ID 不算可打开的成功；旧服务接受口径另列。抽象服务名须在推理时解析为具体可打开的首页动作，才计入新成功。普通 URL／目录上级的部分收益目前以明确命名的实验曲线报告，不能写成已约定的固定收益或与旧综合分混用。所有模型的 Top1、Top2、Top4、输出覆盖、逐样本得失及评分代码／成员哈希均须保留。若固定测试没有提升，只能说明数据利用率增加及误连风险，不能声称预测变好。

`scripts/run_path_menu_ablation.py` 在已封存的断点清洗结果上，把路径菜单的变化拆为“只重排预测前已有动作”和“再加入由已知网址／目录推导的可打开上级”。它先逐项核对原模型分数与封存结果一致，再在相同测试成员上报告两步的新增命中、失去命中、完全一致、服务接受、实验部分收益和已定固定排位分。此对照只检验推理时的离线名单，不改变训练或线上推荐。

```sh
<private-python-with-torch-and-catboost> research/algorithm-research/scripts/run_path_menu_ablation.py \
  --snapshot "/absolute/path/to/exports/context-v2/<UTC>-<UUID>" \
  --split "/absolute/path/to/training/offline-batch-split-v1-<run>" \
  --batch-selection "/absolute/path/to/training/offline-batch-rankers-v1-<run>" \
  --base-run "/absolute/path/to/training/break-cleaning-study-v1-<run>" \
  --output "$HOME/Library/Application Support/SmartApp/training/path-menu-ablation-v1-<new-run>"
```

## 各模型共用的预测输入

`scripts/shared_model_input.py` 是新的研究输入入口。它按每个预测时刻生成一份带版本的内存样本：当前前台目标及各级归属、此前的访问次数、时间、已确认停留、观察质量、前台到达顺序，以及从当时已知网址或目录临时推导的可打开上级。真实下一步由另一个对象在反馈成熟后附上；同一次到达的具体目标和可用上级共用一个结果 ID，不能拆成多次到达。完整 URL 和本地路径只在私有原始快照与内存中的可执行入口使用，传给模型的是不透明标识和数值，不把候选清单保存成采集记录。

预测候选只取截至该时刻已获前台确认的入口，并使用确认当刻的目标信息；单独出现的目标信息更新不证明用户曾到达该目标。由这些已确认网址和路径推导的上级可以临时成为候选，但不增加实际到达次数。若下一步首次确认了一个此前不可推荐的具体目标，它只在反馈形成后加入**训练专用**视图，作为这一次到达的正例；原预测候选及其顺序保持不变，新目标不能回到原预测时刻凑命中。在线模型在反馈形成后才更新，批量模型只学习固定测试切点之前的反馈。

`scripts/shared_model_adapters.py` 将同一份候选名单和目标关系转换为统计、在线小模型、CatBoost、RNN、LSTM、Transformer 各自适合的形式。每个适配器必须声明读取或有意不用的字段；新增字段、调整列顺序或更改序列宽度而没有同步适配时，测试会报错。批量表格模型与在线小模型还收到固定级数的上级身份标识，使“知道父级”既能进入输入，也能作为可学习的输出。序列模型读取层级序列、访问次数及适用的时间等数值，表格模型专用的类别列则明确不用。

各模型使用统一的目标顺序和“具体目标／有用上级”关系，但输入形式不同。声明某字段已转交模型，不等于模型已经学会有效利用它；统计基准有意只用当前目标和历史次数，不使用时间、停留或观察质量。新增共享字段时要明确每种模型读取或不用它，再运行相应的联合测试和固定样本重跑。

`scripts/run_shared_model_input_study.py` 只用已经固定的 598/149 切分和 `both` 清洗，在相同评分成员上重新训练共用输入版本。旧无情景统计模型作为历史参考单独列出；新统计基准、在线小模型、CatBoost 和三个序列架构使用共用候选与路径关系。训练时上级是同一次到达的另一个有用推荐选项，不赋予未经约定的固定分值；评分仍分别报告路径成功、具体目标完全一致、服务接受与部分收益。此脚本只做私有离线研究，不改采集器或正在运行的推荐模型。更新共用字段后，运行 `test_shared_model_input.py`、`test_shared_model_adapters.py`、`test_shared_tabular_runner.py` 和 `test_shared_sequence_runner.py`，再用新私有输出目录重跑研究，不覆盖旧结果。

```sh
<private-python-with-torch-and-catboost> research/algorithm-research/scripts/run_shared_model_input_study.py \
  --snapshot "/absolute/path/to/exports/context-v2/<UTC>-<UUID>" \
  --split "/absolute/path/to/training/offline-batch-split-v1-<run>" \
  --batch-selection "/absolute/path/to/training/offline-batch-rankers-v1-<run>" \
  --output "$HOME/Library/Application Support/SmartApp/training/shared-model-input-study-v1-<new-run>"
```

## 当前统一评分入口与样本合法性

分层决策与损失：`evaluation_contract.minimum_risk_action` 在调用方提供的具体目标概率分布上，选择后验期望损失最小的动作。`decision_loss` 同时用于决策和统一评测：精确目标命中 0，正确服务首页回退 0.4，错误具体页面或错误服务 1，对应匹配收益为 1、0.6、0；实际目标本就是首页时，打开首页收益为 1。只有真正的上层入口获得部分收益，错误会话不能因同属一个服务获得同样优惠。`measure` 始终附带第一项输出的实际平均损失，与预测期望损失分开；现有 TopK 继续作为兼容诊断。

`measure` 同时返回 `rank_score` 综合排名分：分别取实际输出 Top1、Top2、Top3、Top4 的最佳匹配收益，然后平均。单一精确命中位于第 1–4 位时，得分为 1、0.75、0.5、0.25；仅正确首页回退时为 0.6、0.45、0.3、0.15；前四项均不匹配为 0。每个前缀只取最佳收益，重复首页不加分；首页在先、具体页面在后时，后续前缀可从 0.6 提升至 1。此项统一重算所有策略，不以频率改变评价权重，也不改变本次已冻结的模型选择。完整定义见 [统一评价方案](UNIFIED_EVALUATION.md)。

当前实现状态：公共损失、最小风险决策函数、历史预测损失复核，以及统计／学习混合层级分布上的离线决策对照已经实现。`compare_bayes_frequency.py` 先因果回放既定学习器并核对全部旧预测，再将其服务概率按成熟的具体会话记录展开，保留未知子页面和未知目标的质量；固定损失、全局频率损失、情景频率损失共用这一个预测分布。具体会话条件分布使用全局记录作为经验贝叶斯先验、当前起点与相似时间背景计数作为局部证据，先验浓度固定为 2。它是旧模型上的离线决策实验，没有替换原训练缓存、`NavigationHierarchy.specific` 的旧门槛或安装的运行时。

## 密封的前台序列 Shadow 实验

`scripts/run_offline_sequence_shadow.py` 在已经完成的 `offline-batch-split-v1` 因果切分上比较冻结的单步无情景统计基线、RNN、LSTM 与 Transformer 残差模型。它仅从预测截点前、带 `foreground_only` 覆盖证据的 `context.change` 取有序到达；解析补全、浏览器历史/重同步记录不成为 token，所有会话切换和 `capture.gap` 都切断序列。候选仅在内存中从截至截点的目标更新临时重建。

每个族只在外层训练段的时间前向 fit/calibration/selection 中选择小型预先声明的配置；之后才使用全部成熟训练成员重训，并对固定 held-out 成员评分一次。报告始终同时给出精确与服务接受 Top1、Top2、Top4、RankScore、覆盖率及相对无情景基线的样本级 help/harm。模型 checkpoint 只保存哈希桶张量和架构配置，不保存目标词表、候选清单、逐样本预测、完整 URL 或路径。

```sh
<private-python-with-torch> research/algorithm-research/scripts/run_offline_sequence_shadow.py \
  --snapshot "/absolute/path/to/exports/context-v2/<UTC>-<UUID>" \
  --split "/absolute/path/to/training/offline-batch-split-v1-<run>" \
  --output "$HOME/Library/Application Support/SmartApp/training/offline-sequence-shadow-v1-<new-run>"

<private-python-with-torch> research/algorithm-research/scripts/test_run_offline_sequence_shadow.py
```

输出必须是新的私有 training 目录，包含冻结源码、副本校验、聚合评分、配对计数和私有模型权重；不得把其内容复制进源码或用作线上模型更新。Shadow 的离线分数不是部署收益或真实用户选择收益。

`scripts/evaluate_offline_sequence_evidence.py` 是完成 Shadow 后的补充冻结重放：外层时间切点固定，只有预测截点的当前规范来源单位在该切点前已独立确认到达过，才进入 prospective history-evidence cohort。成员资格只读取当前单位与切点前独立到达，绝不读取真实下一步、命中、模型输出、候选或切点后到达。它重放不可变 checkpoint，并以同一评分入口同时写出全部合法固定测试、证据 cohort、单位类别／到达次数分层、以及逐当前来源单位的私有聚合；逐单位文件只存不透明 token，不存原始动作 ID、URL、路径、候选或逐样本预测。

```sh
<private-python-with-torch> research/algorithm-research/scripts/evaluate_offline_sequence_evidence.py \
  --source-run "$HOME/Library/Application Support/SmartApp/training/offline-sequence-shadow-v1-<completed-run>" \
  --snapshot "/absolute/path/to/exports/context-v2/<UTC>-<UUID>" \
  --split "$HOME/Library/Application Support/SmartApp/training/offline-batch-split-v1-<run>" \
  --output "$HOME/Library/Application Support/SmartApp/training/offline-sequence-shadow-evidence-v1-<new-run>"
```

原 hinge 排序模型的 softmax 分数尚未校准，因此此处“最小风险”仅指相对于所估计的概率。今后的完整模型接入应在训练阶段保留具体目标与未知目标分布，用较早成熟反馈学习／校准概率，再调用公共决策函数。不得把评分数据里的事后层级映射作为预测输入，也不得把服务总概率与其子页面概率作为互斥结果重复相加。Top1 是给定概率和代价下的最小风险动作；Top2/Top4 实验按新增菜单项带来的期望损失下降贪心选择，假定用户能选择最合适的一项，不宣称集合全局最优。

所有当前时间背景实验、统计调参、融合、残差与联合排序实验，以及贡献和漏选诊断，共用 [`evaluation_contract.py`](scripts/evaluation_contract.py)。先执行 `prepare_evaluation`，再调用 `measure`、`grouped_metrics`、`paired_topk` 或 `hits`。未经过审查的评分行直接报错。模型依赖可以从旧 run 冻结加载，评分模块必须在导入这些依赖前从当前源码加载；没有评分版本开关或旧实现回退。

- 先根据当前身份、标签身份、时间关系及目标粒度检查可评分性。标签仍是当前目标，或只是当前所属应用时，不计入分母；真实窗口切换被压成当前应用名时同样列为标签信息丢失，等待修复。明确的同应用内不同页面、目录、文件或具体窗口切换仍可评分。
- 审查不读取预测、排名或是否有训练支持。未知目标、模型词表缺失、低频、训练与参考支持不一致，都不能因此排除。原数据构建的独立到达、停留及目标准入检查继续生效；本入口不把身份检查冒充重新验证所有原始采集证据。
- 既有服务接受成绩采用已确认的 ChatGPT 起始页／普通会话服务归并；同时始终输出 `exact` 精确目标诊断。新断点研究另列 `success`：只有预测前已能打开且位于真实目标路径上的网址、目录等上级才算成功，并给实验性的部分收益。其他网站不能仅按域名归并，应用名也不能代替具体网页。Top1/2/4 只看实际输出的前 1/2/4 个位置，评分不去重、补位或用完整排名替换菜单输出。
- 每个新 run 保存 `evaluation-audit-private.json`，记录代码和规则哈希、成员和标签哈希、逐行排除原因及前后分母。配对、选参和漏选使用同一合法集。`require_comparable` 拒绝比较没有统一契约或成员不同的分数；旧报告分数只能作为历史记录，比较时统一重算历史预测。
- 评分审查只影响评测和选参，不删除原始采集、不改写冻结训练标签，也不自动修复窗口身份。合法性修正、服务评分修正与模型进步必须分别解释。下文的旧实验说明和旧 run 数值是历史记录，不是可以切换使用的评分方案。

无需训练即可审查并统一重算一个或多个历史预测 run：

```sh
python3 research/algorithm-research/scripts/rescore_predictions.py \
  --dataset '<私有数据集 run>' \
  --prediction-run '<已有预测 run>' \
  --prediction-run '<另一个已有预测 run>' \
  --output '<新的私有 training run>'
```

本次冻结数据复核：234 条原评测中排除 6 条当前应用粗标签，剩余 228 条；有证据子集从 89 条变为 84 条（旧 60＋新增 24），占合法评测 36.8%。既定参考模型不重新选模，其服务接受 Top1/2/4 为 61/76/80 除以 84，精确目标为 57/72/76 除以 84。分数变化来自口径和分母纠正，没有重新训练或部署。

## 频率作为漏选代价的对照

关于“策略使用不同代价时，如何公平比较”的研究与协议建议见 [统一评价方案](UNIFIED_EVALUATION.md)。`study_evaluation_preferences.py` 复用冻结预测，通过同一个评分入口重算交叉指标，给出偏好变化下的胜负区间与会话稳定性诊断；它不改变既定主评分、不选新模型，也不将敏感性系数用作模型融合参数。

`compare_evidence_scopes.py --source '<冻结预测 run>' --reference '<当前统一评分 run>' --output '<新私有 run>'` 将比较扩展到全部合法评测样本，同时保留主证据、其他支持、无已记录支持三个互斥分组，以及去掉历史选参验证集后的测试分组。它通过公共评分入口重算现有预测，校验原有分组成绩及分组加总，统一报告首页收益 0.6 下的综合分和精确／服务接受 TopK。无支持是既定证据检验的诊断结果，不意味着输入没有可学习线索；这些旧数据也不被称为全新独立测试。

`compare_bayes_frequency.py --residual-run '<已有残差模型 run>' --output '<新私有 run>'` 比较三个预先声明的决策目标：固定 0/0.4/1 代价、全局频率加权代价、当前目标与相似时间背景下的频率加权代价。高频目标漏选更贵。每种策略都最小化 `sum_y P(y|x) * w(y,x) * L(action,y)`，而不是把频率直接宣称为实际操作成本。既有 run 保留生成时的代价；新比较必须在当前入口重算，不能直接混用旧报告分数。

频率仅来自成熟、独立的派生导航结果，区别于后台状态、心跳或窗口库存；全局计数按独立结果去重，情景计数按当前起点及独立结果去重。全局加一平滑，局部计数加一个全局伪样本以向全局收缩，权重在同一个预测结果空间内归一化为均值 1。比较过程不调代价强度、不重新挑模型、不用较晚结果选择权重；预测分布、代价向量、成熟边界、逐样本菜单及代码哈希保存在私有 run。

统一评测同时报告固定实际损失和交叉偏好评分：每一列用同一套截点前频率权重给所有策略评分，`measure_weighted_loss` 计算实际加权损失除以权重总和，并校验样本与权重哈希。不能拿某方案的全局加权分数和另一方案的情景加权分数直接比较。

本批 84 条合法有证据样本（覆盖 84/228=36.8%）：固定风险决策的服务接受 Top1/2/4 为 61/75/80，精确目标为 60/74/80；全局频率代价为 53/65/72（服务）和 53/61/70（精确）；情景频率代价为 57/71/80（服务）和 57/70/79（精确）。固定实际平均损失分别为 0.2798、0.3690、0.3214。全局偏好列中，全局方案将固定方案的 0.1087 降至 0.0460；情景偏好列中，情景方案将 0.1354 降至 0.0960。结果体现常用目标保护与整体命中率的取舍，不构成新数据上的长期最优证明；概率估计仍需校准。

## 动作身份与会话粒度诊断

`scripts/diagnose_action_granularity.py` 只读审计当前模型，并按原成熟边界回放冻结评测集。使用 `--dataset`、`--comparison` 指定原语义实验，`--bundle`、`--host` 指定待诊断服务，`--native-cli` 指定已构建的 CLI，`--output` 指定新的私有 training run。

它区分原始到达、窗口折叠后的应用动作、短停留清洗后的返回标签，以及服务下分别竞争的具体网页标签。保持评分标签、成员和参数不变，对照仅隐藏当前动作、仅删除训练动作别名和二者结合；这些是诊断消融，不是上线方案。当前模型的两种输入粒度通过原生 CLI 校验，冻结基线必须逐条复现原预测。报告和完整 URL 仅写入私有 run；不更改采集、线上模型或安装包。

会话服务的改进设计应分别评估服务选择与具体资源选择；不能把服务入口命中计作原精确 URL 命中，也不能通过删除原始到达来制造没有返回的轨迹。

`navigation_hierarchy.py` 与原生 `NavigationRecommendationModel` 实现同一 v2 派生规则：保留所有实际结果及原始标签，将当前目标/含糊同应用动作从导航正例分开；普通 ChatGPT 对话与起始页使用服务层标签。具体对话只有在相同原始起点的独立成熟结果中占据明确多数时才下钻，规则固定为至少 4 份有效支持且 95% Wilson 下界大于 0.5。这是保守的证据门槛，不是已校准的意图置信保证，不用全局热度或后台库存作为具体对话证据。

`evaluate_navigation_hierarchy.py --dataset <原语义数据集> --comparison <原比较run> --native-cli <新版smartapp-cli> --focus-bundle <实际应用bundle> --output <新私有run>` 在相同成熟边界、原参数及冻结成员上对照 v1/v2。分别报告精确目标与服务级 Top1/Top4、原有/扩展证据分组和模型输出中的当前应用动作；精确评分只使用截点前已观察的首页定位符解析服务动作。原生排序必须逐条匹配 Python，当前模型的迁移复现与历史回放严格分开。产物、定位符和具体诊断只写入私有 run，参数不根据已看过的测试集反复选择。

模型定义、采集缺口及修复验收条件见 [状态序列预测与采集修复方案](state-sequence-collection-plan.md)。本轮实现范围见文档开头与 [v3 状态覆盖契约](../../docs/DATASET.md)。

## 状态样本迁移

构建新版 `build/smartapp-cli` 后运行：

```sh
python3 research/algorithm-research/scripts/migrate_state.py
```

迁移器先在线备份当前数据库（含 WAL），附备份身份密钥与存在时的登记文件；再从该一致备份生成私有 `training/state-v3-migration-<UTC>-<UUID>/`。它逐条比对原始事件、检查历史导出包含关系及校验值、检查输入时间界限并确认补全不再作为独立行为标签。成功后更新 `training/current-state-dataset.json` 指向新版产物。原始数据库及既有不可变导出保持原样，后续应用和 CLI 导出自动使用相同新版规则。

产物包含 `migration-report.md`、`migration.json`、`SHA256SUMS` 及完整可回放导出。旧状态无法证明覆盖时为 unknown；未采到的窗口语义、地址或后台状态不回填。v3 样本不能仅按 `status=observed_horizon` 筛选，还必须按实验需要检查 `inputCoverage` 与 `outcomeCoverage`。

## 描述性分析

`scripts/describe_snapshot.py` 读取一次规范 v2 导出快照，输出：

- `descriptive-analysis.md`：面向研究的中文摘要；
- `metrics.json`：后续实验可读取的聚合指标。

默认输出不会复写已有 run，也不会复制原始 JSONL。默认报告只保留聚合计数、时间范围、目标类型、会话/序列统计和匿名化的集中度指标；不写出目标名称、URL、目录路径、目标 ID、浏览器标签页 ID 或带实例标识的采集 source。

需要审阅具体目标时，显式传入 `--include-targets`。这会在**私有** run 内额外写出 `concrete-target-arrivals.csv`，并在报告中展示排行；它含目标名称和完整 URL/本地路径，但仍不输出目标 ID、标签页 ID、进程 ID 或原始 source 实例标识。不要把该 detailed run 提交到源码或共享给未获授权的人。

推荐的运行方式：

```sh
# 1. 从正在写入的数据库生成一个一致的、可校验的只读快照。
./scripts/export-data.sh

# 2. 为研究结果建立一个新的 run 目录（不可覆盖）。
python3 research/algorithm-research/scripts/describe_snapshot.py \
  --snapshot "/absolute/path/to/exports/context-v2/<UTC>-<UUID>" \
  --output "/Users/you/Library/Application Support/SmartApp/training/descriptive-v1-<UTC>-<UUID>"

# 如需在私有研究报告中查看实际应用、窗口、网页和目录目标：
python3 research/algorithm-research/scripts/describe_snapshot.py \
  --snapshot "/absolute/path/to/exports/context-v2/<UTC>-<UUID>" \
  --output "/Users/you/Library/Application Support/SmartApp/training/descriptive-v1-targets-<UTC>-<UUID>" \
  --include-targets --target-limit 25
```

脚本会在存在 `SHA256SUMS` 时先验证快照完整性，再分析 `events.jsonl`、`target-history.jsonl`、`navigation-samples.jsonl` 和 `manifest.json`。它只依赖 Python 标准库。

## 状态与行为加权近邻：离线回放

`scripts/evaluate_context_knn.py` 实现可增量追加记忆的近邻模型，仅用于离线研究，不修改运行中的菜单。每次预测分别编码当前前台/后台状态与最近 8 次行为：状态包含应用、窗口类别、具体目标、URL/路径层级、选中入口和来源覆盖；行为包含目标语义、位置和近期集合。窗口实例标签限定本会话，实例 ID 不作跨会话语义特征。

两个特征组以加权 Jaccard 比较；来源缺失不当作空集合，旧已观察库存使用较低的证据权重。状态内部前台/后台初始权重为 0.4/0.6，状态组总权重可调，默认实验配置为 0.55。先按相似度取 k 个有当前可用目标标签的历史近邻，再按相似度平方与经历时间衰减投票。近期访问用于补足位置；本实现尚未学习相似度、做分数校准或证明推荐有效性。

```sh
./scripts/export-data.sh
python3 research/algorithm-research/scripts/evaluate_context_knn.py \
  --snapshot "/absolute/path/to/exports/context-v2/<UTC>-<UUID>" \
  --output "$HOME/Library/Application Support/SmartApp/training/context-knn-<new-run-id>"

python3 -m unittest discover -s research/algorithm-research/scripts \
  -p 'test_evaluate_context_knn.py' -v
```

要求不可变快照有完整校验值、样本格式 v3，且准入回放排除事件数为 0。脚本强制产物进入私有 `training/` 新目录，以 0700/0600 权限创建。已知受控验收会话和合成目标被排除，可重复指定 `--exclude-session` 补充排除清单。原始库与导出不会被修改。

评估规则：

- 预测完整 30 秒窗口内第一次独立到达；同目的地下一次到达、没有后续到达、采集边界中断单独排除，不伪造停留或负标签。
- 仅接受完整输入与后续覆盖的样本，或明确标为覆盖未知的旧样本；后者结果单列为探索性分析。显式来源失败/部分覆盖不混入这两组。
- 时间顺序分成初始学习 60%、验证 20%、保留测试 20%；验证标签必须在测试开始前成熟才能调参。重叠窗口不随机打散。
- 每个样本完整观察 30 秒后才进入记忆，先预测再学习；测试期可以消费已经成熟的新样本，但不改变超参数。这是在线增量回放，不是测试期冻结模型。
- 在验证段搜索状态权重、k、半衰期；测试比较近期访问、一阶转移、初始 0.55 近邻、选参近邻、行为单组、去后台特征对照。测试结果不用于再次选择参数。
- 分开报告精确目标 Top-1/Top-4、候选覆盖率、候选内命中率与宽松的同应用命中率。历史路径实际存在性无法重建，候选只表示当时已知合法定位符/入口的可重开近似。
- 回放命中率不等于真实菜单点击率、最终意图识别或因果收益。少量完整样本与有关联的连续行为不能支持稳定泛化结论。

每个 run 包含 `report-private.md`、`metrics.json`、`validation-grid.json`、逐样本预测与时间引用审计、具体失误案例、到达/库存排行、近邻记忆产物，以及输入校验值、代码摘要和固定源码。**这些文件含完整 URL、路径或个人行为，不能复制进源码、共享报告或版本控制。**

## 前台停留分布与临时目标候选

`scripts/describe_dwell.py` 使用完整事件流统计网页、窗口、应用及目录/文件的前台观测时长，不限于 30 秒预测样本。相同目标的重复通知不重启计时，同次到达的解析补全不增加访问；具体目标从被确认时开始计时。跨同应用内部导航合并的应用前台区间另行统计。

```sh
python3 research/algorithm-research/scripts/describe_dwell.py \
  --snapshot "/absolute/path/to/exports/context-v2/<UTC>-<UUID>" \
  --output "$HOME/Library/Application Support/SmartApp/training/dwell-<new-run-id>"

python3 -m unittest discover -s research/algorithm-research/scripts \
  -p 'test_describe_dwell.py' -v
```

可加 `--plot` 输出独立 PNG/SVG 图，需该 Python 环境安装 matplotlib。产物强制写入全新的私有训练目录，包括聚合分布、完整/删失区间、具体目标排行、逐条短停留候选、输入校验值与固定源码。产物包含个人目标和定位符，不能复制进源码或共享报告。

会话结束、采集缺口、前台来源失败、导出截断，以及同应用观察粒度退化均不当作确认离开；未知时长不标成短访问。空闲标志单列，不直接扣除阅读时间。描述性结果区分起点来源覆盖，并提供右删失敏感性对照，不声称恢复真实意图或有效阅读时长。

默认仅并列标记小于 0.5、1、2、3、5 秒的候选，辅助比较返回前一目标、同站点导航和层级变化。**不会删除原始事件或修改训练标签**。短停留不等于误选；未来派生标签只能在离开已被观察到后确定，且清洗前后应保留对照评估。

停留分析 v2 按事件当时已观察到的库存入口关联窗口和标签页：同窗口、同标签页且 URL 改变才计为连续导航；同窗口换标签页、换窗口与换应用分别记录。身份缺失保持未知，不能用同站点替代归属证据。同一 URL 在不同标签页的独立到达不再合并。

`scripts/evaluate_dwell_cleaning.py` 比较 3 秒与 5 秒的派生标签：跳过短暂连续导航中间步骤及其他疑似临时目标，保存各自原因。输入仍是原截点已观察到的完整历史；返回当前目标、30 秒内未达到停留阈值、观测边界不足分别单列。只使用原 30 秒窗口内可确认的标签，满窗口后才能增量学习；不删除真实记录，也不改变运行中的推荐菜单。

```sh
python3 research/algorithm-research/scripts/evaluate_dwell_cleaning.py \
  --snapshot "/absolute/path/to/exports/context-v2/<UTC>-<UUID>" \
  --output "$HOME/Library/Application Support/SmartApp/training/dwell-cleaning-<new-run-id>"
```

该实验固定原评估截点和时间分割，并沿用固定参数及先前验证段参数，不重新调参。除各阈值总体指标，还报告原训练/清洗训练在相同新标签上的表现，以及 3 秒/5 秒共有且标签相同的测试子集。原测试数据此前已查看，因此这些都是探索性结果；改变标签与测试子集后的百分比不能直接解释为模型能力提升。后台缺失时相似度可能被高估的原型局限仍需另外处理。

`scripts/diagnose_dwell_errors.py --run <清洗实验目录> --output <新的私有训练目录>` 可复查清洗后的错误。它验证原 run 与输入快照校验值，使用 run 内冻结的近邻源码重建截点，核对已保存预测、标签和近邻证据，不改变模型或预测。错误按尚未观察到目标、已知目标不在候选、无成熟学习样本、正确目标历史未进入近邻、层级去重及投票落后分类；来源覆盖、具体目标类型、近期访问对照和私有案例另列。分类表示失败所在环节，并非已证明的行为因果。

### 清洗标签上的参数搜索

`scripts/tune_dwell_knn.py --run <清洗实验目录> --output <新的私有训练目录>` 分别在两档清洗标签上搜索状态/历史权重、近邻数量及时间衰减半衰期。运行前保存固定搜索范围与平局规则；状态权重 0 单列为纯历史对照，不作为上下文模型。只按测试开始前已成熟的验证段 Top4、其次 MRR4 选参。原时间顺序前 60% 用于学习、中间 20% 验证、最后 20% 测试；测试过程中只增量学习当时已经成熟的样本。

搜索复用近邻排序以减少重复计算，并与原 run 冻结实现核对评分及旧参数预测；不改算法、候选或标签。“最优”仅限本次网格中的验证最优，测试结果不用于选参。产物包括完整验证网格、参数、Top1/Top4、来源覆盖分组、旧参数与近期访问对照和私有逐样本预测。

`scripts/summarize_tuned_history.py --tuning-run <调参目录> --diagnosis-run <同一清洗实验的诊断目录> --output <新的私有训练目录>` 在后期测试中另列有/无成熟目标历史的表现，保持已选参数不变。有历史是指预测时已存在该具体目标的成熟清洗标签；此分组依据实际结果事后定义，仅用于诊断，不代替后期整体指标，也不声称线上可以提前知道这一分组。

### 优先验证有证据的相似情景

`scripts/evaluate_evidence_model.py --source-run <清洗目录> --tuning-run <调参目录> --output <新的私有训练目录>` 实现选择性情景记忆：前台匹配、背景相似度及共同已观察来源先过门槛，再看多个历史行为片段的后续一致性。缺失背景不重新归一化成满分；没有足够重复支持、后续行为冲突或目标不可达时暂不预测，不用近期访问补位。

`scripts/evaluate_context_rules.py` 使用相同参数形式，试验另一种条件规则模型：前台具体目标/语义类别加一个已观察的后台应用或站点条件，可进一步要求前一步应用相同。只从已成熟历史学习规则，检查支持数及一致率；多个匹配规则不能被当成多次独立行为证据。

两者都按后续到达事件去重学习样本，并限制同一 30 秒行为片段的总权重。主评估依据预测时刻每 30 秒最多取一个截点，另存完整截点结果。有证据与否只依赖预测时可用信息，不使用本次正确目标来筛选测试样本。窗口类别经验只有能唯一对应当前具体候选窗口时才使用。

验证阶段预先要求至少 15 个被接受截点且 Top1 不低于 80%，再优先覆盖率；不满足时输出 `too_few_accepted_validation_samples` 或 `validation_target_not_met`，不得将少量全对样本宣传成可靠保证。Wilson 分数仅用于描述性排序，不能声称相关样本和参数搜索下的独立二项风险保证。旧覆盖未知、严格来源覆盖和相同接受截点的基线分别报告；这些脚本不更新线上菜单。

### 上下文聚类与类内行为频次

`scripts/evaluate_context_clusters.py --source-run <清洗目录> --tuning-run <调参目录> --output <新的私有训练目录>` 实验“情景类别 → 类内后续到达频次”：使用 KMeans 对前台、后台库存与近期访问编码聚类，比较相同最近 8 次到达的顺序、频次、是否出现三种表示，以及无近期历史的状态对照。词表、类别中心和共同条件仅由早期输入拟合；类别数与历史权重按验证段 Top1、Top4、MRR 依次选择，选参落盘后再计算后期测试。需要 numpy、scipy、scikit-learn 和 threadpoolctl，使用已具备依赖的 Python 环境运行。

加 `--extended` 可测试训练段 IDF 加权后台、聚类时前台权重为 0 或 30%，以及全历史、最近 6 小时、最近 1 小时的类内频率。全局频次、相同前台频次对照也在同样的计数时窗中通过验证选参，避免把近期效应归因于聚类。原始到达顺序始终保留，忽略顺序只影响派生输入表示。

标签满 30 秒观察窗后进入计数，每个类别内同一个实际后续到达只计一次；不同类别的支持数不能相加为全局独立到达数。不用近期访问填空，无法映射到当前具体候选的历史答案仍保留在一致率分母中。窗口按稳定应用窗口类别聚合，仅在当前候选可唯一对应时推荐，准确率仍核对具体目标。类别报告区分 100% 交集与至少 80% 共同条件，并只用测试开始前已成熟的后续行为描述类别；另存独立到达与后台库存排行。高一致率门槛只依赖预测时可用信息，不依据本次真实答案筛选。此前已查看的测试段只支持探索性比较。

```sh
<python-with-sklearn> -m unittest discover \
  -s research/algorithm-research/scripts -p 'test_evaluate_context_clusters.py' -v
```

### 有序界面序列：先启用长度 1

`scripts/evaluate_ordered_current.py --source-run <清洗目录> --tuning-run <调参目录> --cluster-run <聚类扩展实验目录> --output <新的私有训练目录>` 将当前目标与历史目标序列统一为 `OrderedContext`：`[当前]`、`[前一个, 当前]` 是相同类型的不同长度，顺序保留。当前实验只启用长度 1，不把近期无序频次或后台相似度加入分数。

这是相同当前目标下的增量转移频次模型。只有此前已成熟的标签才能进入计数，同一条件下同一个实际后续到达只计一次；窗口使用可确认的稳定语义类别，实际推荐仍要求唯一映射到当前具体入口。先按频次排序，同频次时按该起点下最近的后续到达时间排序，最终仍同时才按 ID 排序。候选不足或无历史时可返回空结果，不通过全局近期访问补位。

脚本仅用 Python 标准库，沿用固定后期评估和 3 秒/5 秒清洗；包含相同起点历史支持分组、预测覆盖率、严格来源子集及全部截点的次要结果。ID 平局对照必须复现之前的相同前台频次基线。模型不做参数搜索，也不修改线上菜单；更长有序序列留作未来扩展，没有混入本轮结果。

```sh
python3 -m unittest discover -s research/algorithm-research/scripts \
  -p 'test_evaluate_ordered_current.py' -v
```

### 多步前台序列：离线统计 Markov / n-gram

`scripts/compare_multistep_statistical.py --snapshot <v4快照> --split <因果切分目录> --output <新的私有训练目录>` 独立研究多步前台时间序列输入，不改线上模型或菜单。它只读取已确认的 v4 前台到达；`resolution_update` 是元数据补全而不是新到达，明确的会话/观察边界会截断序列，睡眠/锁屏只有在同一具体目标连续性成立时才延续。后台库存、候选列表、菜单展示和页面/文件正文不进入输入。

保留现有 `statistics/none` 作为单步无序列主基线，并预注册二阶/三阶 Markov、带时间衰减与近期性、以及可变后缀模型。参数选择只在训练段内再做因果验证；固定测试是冻结训练后的单次回放，prequential 是每次预测后才吸收已经成熟的标签，二者分开报告。两种模式都经过统一 `evaluation_contract.py`，同时报告 exact/service accepted 的 Top1、Top2、Top4、RankScore、输出覆盖、可排序真值覆盖及逐样本 help/harm。

报告另列预测前可识别的共同 evidence cohort：以 `statistics/none` 首个输出为锚点，只用截点前已成熟的同起点后续支持按既定 strong/repeated/single/none 门槛分组，所有模型在同一成员上重算；不使用真实下一步或模型命中入组。固定测试和 prequential 的 cohort 分开记录，详情仅保存在私有聚合/审计产物中。

运行产物只写入新的私有训练目录，权限受限且不保存预测列表、候选列表或原始样本正文；源码与测试留在仓库，运行状态、输入摘要、指标和哈希用于复核。该实验结果只代表离线研究，不替换生产模型。

```sh
python3 -B -m unittest discover -s research/algorithm-research/scripts \
  -p 'test_multistep_statistical_models.py' -v
python3 -B research/algorithm-research/scripts/compare_multistep_statistical.py \
  --snapshot <foreground-v4-snapshot> \
  --split <offline-batch-split> \
  --output <new-private-training-run>
```

### 有证据样本与固定评分口径

`scripts/extract_current_evidence.py` 从已冻结的长度 1 模型回放中挑出“预测前已有同一起点 → 实际下一步历史支持”的后期测试样本。每条附去重历史到达 ID、输入/标签成熟时刻、支持次数、占全部后续的比例、最近支持年龄和候选遗漏状态。至少 3 次且占比 ≥70% 为反复且集中支持，至少 2 次但未满足集中门槛为重复支持，恰好 1 次单列；这些是描述性分组，不是校准概率。

先用 `--prepare` 固定计划，再以相同参数去掉该选项进行提取。输出目录只能是新建的私有训练 run，不覆盖已有产物。

```sh
python3 research/algorithm-research/scripts/extract_current_evidence.py \
  --run "/absolute/path/to/ordered-current-source-run" \
  --output "$HOME/Library/Application Support/SmartApp/training/current-evidence-<new-run-id>" \
  --prepare
```

主评分保留全部有证据样本，包括排序错误、未预测及候选遗漏，以精确目标 Top4 为主并并列 Top1、样本数、覆盖率。按真实答案是否有支持筛出的集只用于事后诊断；仅读取预测前历史的首选支持门槛另行统计。窗口类别支持不能冒充同一个具体窗口的经验。后续调参应使用验证段；已查看的旧测试集作为固定诊断基准，泛化仍需新数据验证。

`inputs-test-5s-private.jsonl` 为全部原测试截点的无答案输入；`evidence-test-5s-private.jsonl` 为完整有证据样本及答案；`evidence-misses-5s-private.jsonl` 保留全部未进前四的案例。完整 URL/路径只在个人 run 中保存，独立到达与库存排行沿用源 run 并分开存放。

`scripts/score_current_evidence.py --benchmark-run <证据 run> --predictions <预测 JSON> --output <新评分 run>` 评分新预测。格式为 `{"3": [{"sample_id": "...", "predictions": ["target-key"]}], "5": [...]}`，须对各阈值全部原后期截点各提交一次；空列表代表不预测，仍算未命中。评分器保持证据成员不变，拒绝漏样本、重复目标、超过四个候选和不符合冻结候选/层级规则的预测。候选生成若有修复，应保留测试成员、另行核对新候选的预测前证据，不能通过删除候选遗漏案例提高分数。

```sh
python3 -m unittest discover -s research/algorithm-research/scripts \
  -p 'test_extract_current_evidence.py' -v
```

### 固定证据集：统计与在线学习模型比较

`scripts/compare_online_models.py --expansion-run <扩展证据 run> --output <新私有 run>` 将评分与训练清洗分开：主评分固定为扩展后的 5 秒证据成员、正确目标及候选，历史训练分别使用原始首次到达监督、3 秒稳定标签与 5 秒稳定标签。原生 3 秒评分集合独立报告，不将换标签或换分母当作提升。默认输出全部配置的逐截点预测、Top4/Top1、证据强度、原后期段、严格来源覆盖、实际更新/预测延迟和完整哈希。

模型接口是 `learn(record)` 与 `scores(query)`。`stat_model_suite.py` 包括条件频次、最近转移、有限记忆、时间/事件衰减、贝叶斯平滑与排序融合；`online_model_suite.py` 包括动态类别的在线线性模型、朴素贝叶斯、Hoeffding 树/森林、二元候选 MLP、流式近邻、在线聚类和按当前界面独立的 Softmax。所有模型从此前已成熟的标签学习，不预先声明未来目标词表。MLP 未选中目标的负采样是训练假设，不当作用户明确拒绝。

主特征只有当前目标、所属应用与类型，上下文长度仍是 1。时刻核与候选最近访问单列为额外特征实验。标签成熟边界严格为原 30 秒观察窗结束后的事件截点；每个当前类别下的同一个实际后续到达只学习一次。窗口语义类别必须能唯一映射到当前候选，所有方法执行同一最多四条及层级去重规则。候选缺失、弃权均留在分母。

依赖固定于 `online-benchmark-requirements.txt`，建议安装到独立虚拟环境，或专用依赖目录再通过 `PYTHONPATH` 使用，不升级共享环境。已验证 Python 3.9 与这些依赖。为了可重复的轻量 CPU 计时，可设置 `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1`。统计模型可用标准库独立运行 `--suite stat`；`--suite supplemental` 只运行后增的近邻、聚类与局部 Softmax。

`scripts/search_online_fusions.py --runs <基础 run> <可选补充 run> --output <新私有 run>` 验证上游哈希，在各清洗因子内选择有限数量的家族代表，搜索两模型加权倒数排名融合。`fused_online_models.py` 提供真正可逐条更新的组合模型；最终入选结构在三种清洗监督下重新完整回放，并核对其预测与缓存排序融合完全一致。

这些参数按固定、已多次查看的证据开发集 Top4 选优，因此只能称本次搜索的开发最佳。报告另保留仅按较早段选基础模型、在较晚段评分的控制；并列规则包含配置 ID 和清洗因子，不能借用完整开发集排序打破较早段并列。初始 sweep 中的旧选参摘要可能缺少后一并列项，最终融合汇总会重新计算，不能直接引用初始摘要。融合代表的选择使用完整开发集，其结果不称独立后期验证。

统计模型当前保存研究记忆，部分打分扫描该当前类别的历史；正式接入可为最终选定公式增加增量缓存或有限队列，不应把本次约千条数据下的延迟外推到无限历史。所有实验只生成离线私有产物，不替换线上菜单。

`scripts/refine_online_winner.py` 对入选结构做小范围固定网格核对；`scripts/restrict_online_evidence.py` 检查预测前同当前界面历史支持限制，支持集合仅来自与模型相同清洗因子的已成熟训练流，不读取评分答案。它是算法的预测限制，不能借此从评分分母删除错误。`scripts/finalize_online_comparison.py --runs <已完成比较 run> ... --output <新私有汇总 run>` 校验各来源哈希，再汇总最高结果、固定结构的清洗对照、分组与全部剩余错误。

```sh
<python-with-online-dependencies> -m unittest discover \
  -s research/algorithm-research/scripts -p 'test_*model*.py'
```

### 当前规则：不使用独立候选列表或输出范围限制

不再采集或使用当时允许推荐哪些目标的独立列表，也不保存菜单展示清单。后台应用、窗口、标签页和窗格库存继续采集、导出，作为可选状态输入，模型可单独禁用；实际选择、执行结果和真实到达可保留。模型输出不受当前库存、已采集候选列表或同起点支持集合硬性过滤，执行入口在预测后解析。历史已学类别是分类模型的输出词表，不是额外采集的候选名单，也不能事先加入未来答案。

`unrestricted_prediction.py` 提供当前无候选限制的预测入口。`replay_without_candidates.py --source-run <完成的动作实验> --output <新私有run>` 从冻结训练流逐条学习，只给评分器传入当前界面特征与已成熟历史标签词表，直接排名输出前四条。保留旧模型参数和清洗，仅检查移除限制的影响；旧的门控实验及下方脚本属于历史基线，不能作为当前默认约束。

`remove_collected_candidate_lists.py --root <SmartApp用户数据目录> --audit-directory <新私有run>` 是用户明确授权后执行的一次性历史清理，不是启动时迁移。仅递归去除采集字段 `candidateTargetIDs`，保留其他字段及事件行；使用 SQLite 事务校验及 JSONL 原子替换。被清理快照有显式 redaction 标记和更新后的 SHA256SUMS，旧训练输入哈希会失配，应重新派生输入，不能静默伪造原始校验值。运行中的采集代码由独立任务修改，本维护脚本不承担未来写入控制。

### 语义增强、清洗与多步监督

`semantic_annotations.py` 仅从预测截点已观察的目标元数据构造确定性标签，不读取正文或访问网络。完整 URL、查询、片段与本地路径作为精确身份保持原样；课程、项目、仓库、具体目录范围与站点/名称弱关系分开。同格式、同 PDF 扩展名或通用下载角色不单独构成相似历史证据。多标签取最强共享证据，不将同次行为复制成多个独立支持。

`prepare_semantic_trajectory.py` 显式核验授权字段清理的迁移链，重建截点输入、冻结旧评测成员、生成原有支持和新增 exact/strong/weak 证据分组。相似来源必须在截点前已产生同一精确下一步目标的成熟历史；扩展的是有支持的评测子集，不是伪造新行为。后台状态保留在私有数据中，本轮模型仍以当前界面为输入。

`trajectory_supervision.py` 从独立到达链提取原 30 秒窗口内的前 3 步，补全观测属于同次到达，真实返回仍保留顺序。未观察到的后续步保持缺失，观测中断不当作确认短停留。`resolved`、`hard_kind`、`soft_kind`、`soft_quickcheck` 比较不删除、硬过滤、时长降权及依据过去快速返回习惯的降权；准备脚本额外生成 `soft_overlap`，根据预测前密集重复 anchor 降权。上述清洗只影响训练标签和权重，单步评分答案仍固定为原 5 秒口径；多步评分始终固定为 resolved 到达链。

`semantic_online.py` 提供可逐条更新的精确起点统计加语义软共享，保留相似度绝对强弱；`semantic_online_classifier.py` 提供动态类别、可加权的在线 Softmax 与多项式朴素贝叶斯。预测头 0 学习清洗后的首个保留目标，头 1/2/3 学习原有序步骤。只接受成熟历史标签，不预先声明未来类别，不使用库存或同起点支持集过滤输出。

```sh
python3 research/algorithm-research/scripts/prepare_semantic_trajectory.py \
  --source-run "/absolute/path/to/completed-action-cleaning-run" \
  --output "$HOME/Library/Application Support/SmartApp/training/semantic-dataset-<new-run-id>"

python3 research/algorithm-research/scripts/compare_semantic_trajectory.py \
  --dataset "/absolute/path/to/semantic-dataset-run" \
  --baseline-run "/absolute/path/to/no-candidate-control-run" \
  --output "$HOME/Library/Application Support/SmartApp/training/semantic-comparison-<new-run-id>"

python3 research/algorithm-research/scripts/summarize_semantic_trajectory.py \
  --run "/absolute/path/to/semantic-comparison-run" \
  --output "$HOME/Library/Application Support/SmartApp/training/semantic-summary-<new-run-id>"
```

比较前保存搜索计划。原有证据 Top4 优先，新增语义证据 Top4 次之；报告原有、新增强/弱、合并及全部截点，分开固定架构的清洗对照与固定清洗的语义对照。多步菜单对各步骤概率分布加权再排序，另报精确的逐步 Top1/Top4、未来目标加权覆盖和完整三步子集，不将任一步命中当成下一步准确率。选优后脱离搜索组件缓存直接回放核验。已查看的数据只作为开发比较，不称独立测试保证；不修改线上菜单或采集数据库。

### 历史实验：可启动与重开动作、独立清洗探索

`scripts/compare_action_cleaning.py --output <新私有 run> --workers 3` 重新比较上述 27 个统计与在线模型家族、210 组基础参数，以及有限的两模型排序融合。主成员仍是原 62 条有证据样本，评分答案仍来自冻结的 5 秒观测定义；训练清洗作为独立实验因子变化。

`reopenable_actions.py` 将没有跨会话定位符的普通窗口聚合到所属应用的激活/启动动作，具体网页、目录与文件保留精确目的地。候选来自截点前已确认准入的应用和有效历史定位符，不能因不在当前库存而丢掉可重开目标，也不能从之后的到达补入新目标。报告分别列动作层、原精确目标层、应用层，以及与当前前台相同应用的含糊样本；命中应用不宣称恢复原窗口或页面。旧预测只映射原已展示的四项，去重后不能从第五名补位。

`explore_training_cleaning.py` 预设 19 种处理方式：原始到达、只补全同次到达、原 3/5 秒对照及 1/2/8/10 秒、仅确证同标签页导航、仅快速 A→B→A、确证导航与跨目标切换、按目标类型使用不同阈值，以及只从过去完整停留分布估计阈值。自适应阈值只使用原预测截点之前已经结束的独立停留；未来补全、之后才发生的退出、未知停留和观测中断不会当作已知短停留。所有方案保留原 30 秒标签成熟边界，并按起点与实际后续到达去重。选择性规则保留下来的短到达仍只是观测，不被称为确认意图。

运行同时保存固定架构下的清洗对照、每种清洗下重新选架构的结果、旧推荐单纯重评分的对照及各样本的私有决策依据。搜索按已看过的开发数据选优；较晚段单列并不使它变成全新测试。`--smoke --cleanings raw 3 5 --config-ids stat_frequency_conditional` 可先验证最小回放；所有输出须使用新目录，不覆盖完成的 run。完整测试覆盖自适应截止时间、后续信息泄漏、可重开候选准入、层级评分及融合重放一致性。

`hierarchical_action_model.py` 与 `compare_hierarchical_actions.py` 另测同一次成熟到达的双层监督：主模型学习具体动作，辅助模型仅从该到达已记录的所属应用学习应用目标，分别更新一次，不能把层级计成重复到达。辅助应用评分与主模型排名融合，权重 0 必须逐条复现原模型；应用推荐不能代替具体页面或路径命中。`summarize_action_experiments.py --main-run <完成的主实验> --hierarchy-run <完成的双层实验> --output <新私有汇总>` 校验来源哈希，再用中文清洗名称汇总结果、固定模型消融和剩余错误。

### 当前方案：长度 1 前台与近期、长期背景联合学习

前台输入固定为 `[当前目标]`，不加入前一步、最近有序路径或顺序后缀。历史行为只通过独立的无序背景统计进入模型：近期 5/60 分钟、过去 24 小时、截至截点的全部记录，分别保留频次、最近访问衰减值、已观察非空闲停留和覆盖标记。各尺度、通道分别归一化，短期相对长期的变化另作背景特征；多尺度统计不计为额外到达。所谓长期仅指已有记录，不能将几天数据视为长期稳定习惯。后台采集与导出继续，模型不读取后台库存。

`prepare_multibackground_dataset.py` 从新的一致导出追加样本，要求旧事件是新事件的完整不变前缀，并逐项验证旧预测截点与旧评分标签。新增截点使用相同训练清洗、成熟边界与评分规则，单独作为 `new_test`；其历史证据只查截点前已成熟的记录。历史菜单展示排除必须有显式审计，不能悄悄接受无法解释的准入排除。

`multibackground_model.py` 直接学习当前目标与背景的联合关系，比较带显式交互项的 softmax 和 32 单元 tanh 神经网络（带线性分支）。输出类别只由成熟行为标签增量引入，不从未来标签、近期目标集合或后台库存构造。新样本触发一次 AdaGrad 更新，批内可复习至多 7 个此前成熟的样本；这是优化器复习，不是新增独立行为。停留与频次沿用真实观测边界，暂停、中断、空闲、初始/恢复和同次补全不会被解释为持续使用或重复到达。

`compare_multibackground.py` 固定比较两种架构、两种学习率、当前/近期/长期/组合背景及组合频次通道，共 20 套权重。每套分别评分纯学习输出与预设 50% 旧统计模型融合；仅用旧较早验证段的固定有证据 Top4 优先选参，再汇总旧较晚段与新增数据。新增数据允许按严格时间顺序在线学习已成熟标签，但不用于选配置。旧基线必须逐条复现，服务级命中与精确会话命中分开。旧数据已用于开发，不称独立测试。

运行需要 Python 与 NumPy。产物只进入新私有训练目录，包含冻结计划、单独选参文件、全部对照、输入、预测、报告及四组背景模型的真实权重、优化器、训练记忆和随机数状态；同时验证重载预测和继续增量更新一致。脚本不修改采集与运行中模型。

```sh
python3 -m unittest discover -s research/algorithm-research/scripts \
  -p 'test_multibackground_model.py' -v

./scripts/export-data.sh

python3 research/algorithm-research/scripts/prepare_multibackground_dataset.py \
  --base-dataset "/absolute/path/to/frozen-semantic-dataset" \
  --snapshot "/absolute/path/to/new-context-v2-export" \
  --output "$HOME/Library/Application Support/SmartApp/training/multibackground-dataset-<new-run-id>"

OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
python3 research/algorithm-research/scripts/compare_multibackground.py \
  --dataset "/absolute/path/to/appended-multibackground-dataset" \
  --baseline-run "/absolute/path/to/frozen-navigation-hierarchy-evaluation" \
  --output "$HOME/Library/Application Support/SmartApp/training/multibackground-experiment-<new-run-id>"
```

### 后台状态替换与输入贡献诊断

`diagnose_background_contributions.py` 固定已完成时间背景实验选出的架构、学习率、种子、标签和时间截点，直接比较“当前＋后台状态”与“当前＋时间背景”，并保留仅当前、近期、长期及两类背景合用的辅助对照。原统计基线本身不使用后台库存，不能把它与时间模型的差异称作后台被替换后的收益。

后台状态取冻结样本在截点已观察的来源组，各来源归一化并保留质量与缺失/空的区别，不限制输出类别。新的特征移除变体从头按时间重训，去掉衍生交互项，分别评分纯学习分支与固定 50% 统计融合。当前特征消融只移除学习分支的对应输入；原统计分支和输出语义仍保留当前目标。贡献以完整组减去移除组的 Top1/Top4 命中差，以及修正/损失次数表示，不能相加为 100%，融合系数也不是贡献率。

这是已查看评测结果后的解释性分析，不再选参或部署；后台模型沿用时间模型的参数，未证明后台模型各自调参后的最佳能力。依赖优先冻结上轮实际源码，必须逐条复现旧统计基线和完整时间模型的预测；样本、标签成熟边界、更新次数及原始快照哈希保持一致。

```sh
python3 -m unittest discover -s research/algorithm-research/scripts \
  -p 'test_diagnose_background_contributions.py'

OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
python3 research/algorithm-research/scripts/diagnose_background_contributions.py \
  --experiment "/absolute/path/to/completed-multibackground-experiment" \
  --dataset "/absolute/path/to/its-frozen-dataset" \
  --output "$HOME/Library/Application Support/SmartApp/training/background-replacement-<new-run-id>"
```

### 纯统计模型的时间背景与后台状态调参

`context_statistics.py` 将无序背景作为条件统计的一部分：按历史截点与当前截点的背景相似度，分别重新分配同起点和语义支持的票数，保留各组证据总量及完整已成熟输出类别。时间背景使用近期/长期尺度的频次、最近访问衰减和停留；后台状态使用各观察来源的目标身份或完整已记录语义特征。尺度、通道和来源独立归一化，缺失来源保持中性，已观察空集合单独表示。前台序列始终为 1，服务起点折叠后按原统计模型再次去重，不把重叠监督计为独立行为。

`compare_context_statistics.py` 复用已完成多背景实验的冻结数据与源码。仅当前组调 27 组通用参数，时间与后台各调 972 组，使用同一旧早期验证段选择；在选择落盘后才汇总较晚与新增成绩。参数包括语义支持、近期目的地回退、历史半衰期、背景特征组合、背景权重及相似度幂。另给出固定通用参数及仅移除背景的对照，区分调参收益与背景收益。主报告先给全部已约定有证据样本合计，再列原固定集、新增段和语义扩展组；不以合计成绩重新选参。产物包含统计记忆、模型配置、重载核对、验证网格及私有预测，无神经网络融合，不部署。

```sh
python3 -m unittest discover -s research/algorithm-research/scripts \
  -p 'test_context_statistics.py'

OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
python3 research/algorithm-research/scripts/compare_context_statistics.py \
  --experiment "/absolute/path/to/completed-multibackground-experiment" \
  --dataset "/absolute/path/to/its-frozen-dataset" \
  --output "$HOME/Library/Application Support/SmartApp/training/context-statistics-<new-run-id>"
```

### 时间背景统计与学习模型融合

`compare_temporal_fusion.py` 固定两个已由早期验证选出的时间背景分支，仅搜索学习分数占比 0、0.1、…、1。统计和学习分数分别归一化后加权；学习分支明确调用纯 `scores`，不调用含旧统计融合的 `serving_scores`，避免把旧统计分支重复混入。两个端点必须逐条复现各自已有预测，历史学习分数按成熟边界重新回放，不能使用最终权重回测早期样本。

仅按原早期验证段选择比例，完全同分优先更小学习权重；选择落盘后才汇总新旧有证据样本。报告同时列出两个单模型、50/50、验证选中比例及此前融合的对照。产物保存因果分支分数、预测、配对修正/损失和引用不可变分支模型的融合清单；验证最终学习权重一致及清单重载结果，不部署或更改采集。

```sh
python3 -m unittest discover -s research/algorithm-research/scripts \
  -p 'test_compare_temporal_fusion.py'

OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
python3 research/algorithm-research/scripts/compare_temporal_fusion.py \
  --statistics-run "/absolute/path/to/completed-context-statistics-run" \
  --learning-run "/absolute/path/to/completed-multibackground-experiment" \
  --output "$HOME/Library/Application Support/SmartApp/training/temporal-fusion-<new-run-id>"
```

### 统计基础上的情景修正与 Top2 训练

`statistical_residual.py` 使用 `log(统计概率) + 学习修正` 排序，从零修正开始，不另设统计/学习混合比例。共享线性修正器学习目标、当前目标与目标的关系，以及候选目标在多时间尺度背景中的频次、最近访问、停留、覆盖和短长差异。这里的候选指全部已成熟可输出目标，不受后台库存或同起点支持集限制。分别比较整体交叉熵与针对第二高错误目标的 Top2 margin hinge，并保留关闭情景特征的消融。训练仍有学习率等超参数，不宣称免调参。

`compare_statistical_residual.py` 复用已完成融合 run 的冻结依赖、数据、清洗与统计参数，在全部观察截点保存当时的统计先验、输出词表及特征。后续标签成熟后才更新修正器；复习旧样本仍用原先验，不重算包含其自身答案的统计分数。原截点尚不能输出的标签只进入统计记忆，残差训练明确跳过，评测分母不变。训练用全部有效成熟导航关系，不按事后有证据诊断组筛样本。

本轮按旧较早验证段的原证据 Top2、Top1，以及扩展证据等预先声明的顺序选择；保留零修正统计基线和旧固定融合对照。先保存选择，再计算晚段、新增段和合计成绩。保存真实参数、优化器、训练索引、随机状态及所有截点的因果稀疏特征，验证基线、重载和继续学习一致。只写私有研究产物，不部署或修改采集。

```sh
python3 -m unittest discover -s research/algorithm-research/scripts \
  -p 'test_statistical_residual.py' -v

OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
python3 research/algorithm-research/scripts/compare_statistical_residual.py \
  --fusion-run "/absolute/path/to/completed-temporal-fusion-run" \
  --output "$HOME/Library/Application Support/SmartApp/training/statistical-residual-top2-<new-run-id>"
```

### 联合排序的架构与稳定性对照

`joint_ranking_alternatives.py` 在相同的冻结统计先验和情景特征上比较线性 Top2 hinge、线性交叉熵、前二边界 logistic、可学习情景门控、16 单元非线性修正。门控同时学习统计先验与修正输出的使用强度，不搜索固定混合比例；所有模型零输出初始化时均复现统计基线。

`compare_joint_alternatives.py` 每类使用两个学习率、三个固定种子，共 30 次训练。按较早验证段三个种子的合计成绩选择架构和学习率，不挑最好种子；主表使用预先固定参考种子并附种子范围。在原验证段之后另按时间等分四段，每段只用此前已成熟的评测结果选择方案，报告随后该段的表现。原统计参数来自早期验证，所以前推检查不能移到这之前。所有历史数据此前均已查看，不称全新独立测试。

复用并校验上一轮因果训练缓存；参考种子的线性模型必须逐条复现上轮预测和最终权重。全部模型保存参数、优化器、去重和复习随机状态，验证重载及继续学习一致。按整个采集会话进行配对重采样，只用来探索稳定性，不将少量会话上的区间当作选模后显著性证明。保持相同清洗、精确目标标签、有证据合计口径和私有数据目录，不部署。

```sh
python3 -m unittest discover -s research/algorithm-research/scripts \
  -p 'test_joint_ranking_alternatives.py' -v

OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
python3 research/algorithm-research/scripts/compare_joint_alternatives.py \
  --residual-run "/absolute/path/to/completed-statistical-residual-run" \
  --output "$HOME/Library/Application Support/SmartApp/training/joint-ranking-alternatives-<new-run-id>"
```

### Top4 漏选的因果证据检查

`diagnose_top4_misses.py` 固定当前已选残差模型及全部评测标签，逐条复现历史预测并检查漏选发生在哪一层：预测前是否已观察目标、是否有成熟训练类别、同应用输出过滤、服务与具体会话归并、参考证据是否真正进入训练、统计排序和学习修正、多时间尺度背景与既有模型对照。主要报告全部有证据样本，另保存全部评测的漏选明细。

诊断严格区分预测前线索与预测后的标签/轨迹核对。窗口类别背景和应用输出标签的对齐缺口只作事实记录，未重训前不宣称修复增益；逐例找到某个对照命中也不是可部署的组合成绩。同一后续结果可能被多个评测截点引用，因此同时记录不同结果事件数。不删除结构性漏选、不改变分母、不部署，完整 URL/路径与逐条证据只写私有 run。

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
python3 research/algorithm-research/scripts/diagnose_top4_misses.py \
  --alternatives-run "/absolute/path/to/completed-joint-alternatives-run" \
  --output "$HOME/Library/Application Support/SmartApp/training/top4-miss-audit-<new-run-id>"
```

### 历史对照：时间背景相似度与有序后缀

以下保留此前实验的复现入口；其中顺序特征不属于上面的当前方案。

`scripts/temporal_context.py` 从原始事件单向重放构建预测截点的时间背景，保留当前目标与最近有序目标后缀，并比较 5、15、60 分钟窗口及多窗口组合。背景通道包括独立到达次数、时间衰减次数、距最近访问的衰减值和已观察的非空闲前台停留。初始/恢复快照不增加次数；同次到达的补全不重复计数，已冻结输入不随后来补全变化，具体子目标的停留不倒填到尚未识别它的时候。中断、来源失效和会话切换截断停留与有序后缀；先前已确认的访问可在时间窗口内保留，但缺失时段不视为持续使用或确认无行为。所有特征只引用截至截点的事件。

`scripts/compare_temporal_context.py` 在已有语义数据集上训练 35 组增量条件统计模型。当前目标、语义共享、`hard_kind` 清洗、导航层级及精确评分标签保持一致，时间背景只重新分配既有转移证据的权重，不限定可输出的历史目标。对照包括当前运行时模型、仅有序后缀、窗口频次和丰富时间背景；本轮各模型都不读取后台库存。旧运行时模型的固定截点预测必须完全复现。服务级命中单独列出，不替代精确会话命中。

参数仅按较早验证段已成熟的固定有证据样本精确 Top4 选取，选参文件落盘后才汇总较晚段指标，同时报告 Top1、样本数、返回预测比例、证据分组及时间特征覆盖。既有数据已经多次查看，因此较晚段仍属于开发检查，不称全新独立测试；按实际后续目标定义的证据分组不代表线上可识别的接受集。完整配置、输入、训练后的模型记忆、预测、错误案例、中文报告、源文件与哈希均保存到新建私有 run，原始采集与运行中模型保持不变。

```sh
python3 -m unittest discover -s research/algorithm-research/scripts \
  -p 'test_temporal_context.py' -v

python3 research/algorithm-research/scripts/compare_temporal_context.py \
  --dataset "/absolute/path/to/semantic-trajectory-dataset-run" \
  --baseline-run "/absolute/path/to/navigation-hierarchy-evaluation-run" \
  --output "$HOME/Library/Application Support/SmartApp/training/temporal-context-<new-run-id>"
```

`scripts/export_temporal_models.py --run <完成的时间背景run> --output <新私有run>` 将验证段选出的顺序、频次、丰富时间背景模型分别保存，复用同一份已成熟训练记忆并逐项校验重载后的分数。它不重新选参、不使用较晚段来挑模型，也不部署到 `models/`；模型文件含版本、配置、输入截点、标签成熟时刻及训练快照终点。

### 历史模型在统一完整验证集复测

完整对照使用 `prepare_complete_validation.py --fixed <原完整固定验证数据run> --fresh <新快照数据run> --output <新私有完整数据run>`。按样本ID合并原完整验证成员与新增验证成员，验证旧输入和评分标签不变；保留训练专用截点作历史输入，不能把它们充作验证样本。所有模型使用同一次合法性审查后的集合。

将完整数据run传给下述 `compare_fresh_models.py`，再用 `complete_fresh_comparison.py` 补齐家族。`evaluation_scope=complete_two_cohorts` 时只评分完整验证集与完整有证据验证集，主排名统一采用现有 `rank_score.mean×100` 综合分（精确收益1、正确首页回退0.6，Top1至Top4前缀最佳收益均值），不以Top4单项排序。`summarize_complete_validation.py --run <完整合并run> --output <新私有汇总run>` 独立重算两组全部模型成绩，生成仅含这两列综合分的报告。完整集包含历史开发及选参成员，不称全新独立测试。

以下仅新增时段流程保留作可复现历史实验，不能替代完整验证集结果：

`prepare_fresh_validation.py --snapshot <最新完整快照> --previous-snapshot <此前最后评测快照> --output <新私有数据run>` 验证原始事件前缀不变，只将此前快照之后的自然截点纳入新验证集。模型训练仍可使用此前历史；验证阶段严格按时间学习已成熟反馈。统一使用 `hard_kind` 训练清洗、稳定停留评分标签和 `evaluation_contract.py` 合法性审查，先冻结成员、边界及哈希，再运行模型。

`compare_fresh_models.py --dataset <新私有数据run> --output <新私有比较run>` 从头重训早期统计／在线模型、上下文统计、联合／残差排序、多背景、时间相似度、早期近邻／聚类／选择性规则、语义统计与风险决策。配置沿用历史选择或预声明适配，不用新验证成绩选参；旧有序路径、库存候选及同起点输出硬限制不带入当前协议。选择性方案空输出仍占同一分母。每个模型必须对全部共同截点提供输出；保存真实状态并检查重载一致。

`complete_fresh_comparison.py --base <刚完成的新比较run> --output <新私有合并run>` 补齐时间统计／学习固定比例融合、语义分类及分层应用辅助头，校验新回放融合端点，合并的基础预测只来自同一新验证集的刚完成重训。所有模型重新接受相同评分和固定／全局频率／情景频率权重交叉评分。配置数包含种子及对照，不等于独立算法数；不声称逐个复刻所有历史超参数或旧清洗协议。

`summarize_fresh_comparison.py --run <新私有合并run> --output <新私有汇总run>` 独立重算全部模型、证据分组及交叉成本，验证源产物哈希和评分成员一致，生成中文代表方案、随机种子范围及配对变化表。先报告全部有证据合计及覆盖率，再看全部合法样本；少量新样本不能用来宣布长期冠军或自动部署。实验脚本使用 `online-benchmark-requirements.txt` 的 Python 3.12 环境；全部个人数据和模型只写入私有训练目录。

## 研究边界

- `context.change` 的 `resolution_update` 是同一前台到达的补全，不作为独立预测锚点。
- 当前研究输入只使用已确认的前台到达，不读取、持久化或训练使用后台应用、窗口、标签页或目录库存；旧快照中的历史字段只在明确的兼容审计中按其旧语义处理。
- 输入状态覆盖与后续标签覆盖分别评估；缺失、失败或过期不等于成功观察到空集合，后来补采的信息不倒填为截点前输入。
- `capture.gap`、会话结束和导出截断是观察边界；不把缺失观察解释为负反馈或“停留”。
- 训练/验证必须按完整会话或时间顺序切分，不能随机打散重叠序列；决策后 outcomes 不能混入输入特征。
- 当前按固定清洗与标签评估增量模型及多时间尺度背景，输出不受当前库存或同起点支持集限制；部署前需另行验证运行时表现。


### 明确休眠/锁屏的连续性（build 11）

`known_inactivity.py` 以逐事件因果回放验证 pause begin/end 与同目标恢复，保留原时间字段并添加派生 activeMonotonicMs。停留分析、时间背景、语义轨迹与当前 kNN 样本读取共同使用该契约；模型标签仍与输入隔离。新样本的成熟位置取 completedThroughSequence/completedAtMonotonicMs，不能把一小时休眠当作完成 30 秒行为观察。历史未标明原因的 gap 保持中断；已冻结的旧 run 与原始采集库不重写。新 run 保存辅助脚本及哈希。

完整 Python 回归环境：`uv run --python 3.12 --with-requirements research/algorithm-research/online-benchmark-requirements.txt python -m unittest discover -s research/algorithm-research/scripts -p 'test_*.py'`。
