# PCHP 双语术语表（V381）

本表是中英文主文、补充材料、图题与图内标签的统一用词依据。原则是：优先采用电池健康估计、机器学习和在线优化文献中的常用术语；仅在确有形式化含义时保留数学术语，并在首次出现时给出普通语言解释。

| English | 中文 | 使用说明 |
|---|---|---|
| capacity retention | 容量保持率 | 本文主要预测目标。只有在测试规程足以支持健康状态解释时才与 SOH 联系，不把所有归一化容量一概称为 SOH。 |
| state of health (SOH) | 健康状态（SOH） | 用于领域背景和采用标准参考容量的实验；不替代数据集实际定义的目标。 |
| cross-domain learning | 跨域学习 | 指训练域与目标域在电芯、协议或工况上的变化；不使用含义更窄的“域适应”统称所有实验。 |
| source domain / target domain | 源域 / 目标域 | 训练与外部评估的标准称谓。 |
| protected source predictor | 受保护源域预测器 | 提供基准预测；“受保护”表示后续更新以它为损害参照，并不表示绝对安全。 |
| protected causal state | 受保护因果状态 | 由当前及既往记录递推得到、不可被未来记录改写的非增状态。 |
| candidate predictor using changes from commissioning | 利用投运初期变化的候选预测器 | 中文优先说明信息来源，避免“记录特异候选”等生硬表达。 |
| prefix causality | 前缀因果性 | 表示时刻 \(t\) 的输出仅依赖截至 \(t\) 的信息；这里是在线信息约束，不宣称电化学因果识别。 |
| prefix-causal harm-budget projection (PCHP) | 前缀因果损害预算投影（PCHP） | 方法全称；标题、摘要和首次出现处使用全称，之后使用 PCHP。 |
| harm budget | 损害预算 | 相对于受保护状态的逐记录最坏绝对损失增量上限 \(\delta\)；它是预测损失约束，不是物理安全阈值。 |
| harm-budget interval | 损害预算区间 | 满足当前逐记录损失约束的精确可行区间；不使用“损害管”。 |
| online viability kernel | 在线可行核 | 同时满足当前预算与未来递推可行性的输出集合；首次出现时补充“可继续在线执行的精确集合”。 |
| recursively feasible | 递归可行 | 当前输出不会破坏后续时刻继续满足约束的可能性。 |
| prefix-wise \(\ell_\infty\)-nonexpansiveness | 前缀 \(\ell_\infty\) 非扩张性 | 表示映射不会放大有界前缀扰动；保留标准数学术语并配直白解释。 |
| domain-equal MAE | 域等权 MAE | 先在电芯内汇总，再在域内等权汇总电芯，最后等权汇总各域。 |
| dataset-equal MAE | 数据集等权 MAE | 外部确认中先在电芯内汇总，再在数据集内等权汇总电芯，最后等权汇总各数据集。 |
| cell-level MAE / cell-equal MAE | 电芯级 MAE / 电芯等权 MAE | 前者指单电芯统计量，后者指各电芯等权的汇总量。 |
| constant budget-feasible offset | 常数预算可行偏移 | 与 PCHP 使用相同预算、范围和单调约束的常数偏移对照；短称“常数偏移”。 |
| source-tuned unprotected comparator | 源域调优无保护比较器 | 保留前缀因果性、范围和单调性，但不受损害预算约束，用于衡量保护的精度代价。 |
| protocol-frozen external mechanism confirmation | 协议冻结外部机制确认 | 数据未影响被检验的方法设计、比较器、估计目标、代价规程或判定标准；不等同于盲法前瞻采集或独立实验室重复。 |
| outcome-blind, protocol-locked external validation | 结果盲、协议锁定外部验证 | 在访问留出结果前锁定预测、规则和判定标准；避免使用可能暗示前瞻采集的“前瞻锁定”。 |
| continuous asymmetric decision cost | 连续非对称决策代价 | 将低估和高估容量保持率赋予不同代价的连续指标。 |
| binary review decision | 二元复核决策 | 根据容量保持率阈值决定是否复核；不使用“硬动作”。 |
| missed degradation / unnecessary review | 漏检退化 / 不必要复核 | 分别对应高估退化电池和低估未退化电池的决策后果。 |
| single-run stress test | 单次压力测试 | 指冻结流程的一次外部压力测试；不写容易产生营销感的“one-shot / 一次性”。 |
| accuracy trade-off under harm control | 损害控制下的精度权衡 | 描述保护约束与精度之间的经验关系；不使用“精度边界”暗示未经证明的最优上限。 |

## 明确弃用的表达

- `surface / 表面`（指数据集时）：改为 `dataset / 数据集`。
- `surface-equal / 表面等权`：改为 `dataset-equal / 数据集等权`。
- `cell-macro MAE / 电芯宏平均 MAE`：按层级改为域等权、数据集等权或电芯等权 MAE。
- `safe shift / 安全平移`：改为常数预算可行偏移，避免把相对预测损失约束误写成物理安全。
- `harm tube / 损害管`：改为损害预算区间。
- `hard action / 硬动作`：改为二元复核决策。
- `opportunity cost of the tube / 损害管机会成本`：改为损害控制牺牲的精度。
- `accuracy boundary / 精度边界`（经验比较场景）：改为精度参考或精度权衡。
- `prospectively locked / 前瞻锁定`（公开历史数据场景）：改为结果盲、协议锁定。
- `falsification / 证伪`（一般稳健性检验场景）：改为验证、检验或反例检查；仅在严格哲学或统计含义成立时使用。

## 语言风格约束

- 英文采用“现实问题—方法动作—保证—证据—实际含义”的直接句法，避免名词堆叠和连续被动语态。
- 中文不逐词照搬英文结构，优先使用“谁在什么条件下做什么、得到什么结果”的主动表达。
- “causal / 因果”每次涉及方法含义时均限定为信息时序上的前缀因果，不延伸为电化学机理因果。
- “harm / 损害”均限定为相对于受保护状态的预测损失增量，不表述为电池安全、热安全或现场事故风险。
- 经验数据支持“改善、降低、符合约束”；理论结果支持“保证、必要充分、非扩张”。两类证据不互换措辞。
