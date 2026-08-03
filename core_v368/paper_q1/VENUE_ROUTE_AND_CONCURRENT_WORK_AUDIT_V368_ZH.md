# PCHP 目标期刊路线与同期工作审计（V368）

核验日期：2026-08-02  
稿件类型：方法型为主、系统/应用型为辅  
科学冻结基线：`rccp_causal_manuscript_v366/`  
当前投稿工作稿：`rccp_causal_manuscript_v368/`  
路线状态：`APPLIED_ENERGY_PRIMARY_ROUTE`  

## 1. 一句话投稿记忆

PCHP 解决的不是“再训练一个更准的 SOH 网络”，而是新车队容量结果尚不可得时，如何允许学习候选更新，同时保证每条已经发布的前缀因果预测相对于受保护状态的最坏结果一致损害不超过声明预算。

## 2. 官方范围核验

### 2.1 Applied Energy：主投

官方范围：<https://www.sciencedirect.com/journal/applied-energy>  
官方投稿指南：<https://www.sciencedirect.com/journal/applied-energy/publish/guide-for-authors>

范围匹配：

- 官方范围明确包括 energy storage、applications of artificial intelligence in energy、optimization and decision-making methods for energy systems problems、data analytics for energy problems，以及交通电气化；
- 强调连接 research、development 与 implementation；
- 接收 Full Length Research Article；
- 摘要不得超过 \(250\) 词；
- 要求 \(3\) 至 \(5\) 条 highlights，每条最多 \(85\) 个字符；
- 要求研究数据声明，并采用“存入仓库并链接；不能共享时说明原因”的数据政策；
- 使用生成式 AI 辅助稿件准备时必须声明，且作者承担全部核验责任；
- 允许 LaTeX 可编辑源文件，不要求初投即采用不可逆的排版形式。

契合判定：`HIGH_METHOD_AND_APPLICATION_FIT`。

主要优点：

1. PCHP 的部署问题、风险预算、在线不可回写和跨域储能数据均处于官方范围内；
2. 期刊已发表多源迁移、贝叶斯迁移和物理信息迁移的电池健康研究，说明编辑部接受方法驱动的电池预测论文；
3. 现有稿件的 \(12\) 域嵌套开发评估、基于留出公开数据的结果盲且前瞻锁定外部验证、压力失败边界和复现链明显强于常见的少数据集精度比较；
4. 可采用订阅出版路线，不必把开放获取费用作为首投前提。

主要风险：

1. V368 英文摘要保守计数为 \(250\) 词，满足官方 \(250\) 词上限；
2. 正文约 \(7{,}949\) 个正文词，理论与协议细节较重，存在编辑端“应用价值不够前置”的风险；
3. 真实车队或新实验室验证缺失，必须把贡献限定为可审计预测更新契约，不能暗示电化学或事故安全；
4. 代码发布许可仍待机构确认，不能以 `LICENSE_PENDING` 状态投稿；
5. 作者、机构、通讯方式、CRediT、基金、利益冲突和 AI 使用声明仍未完成。

### 2.2 Energy and AI：第一备投

官方范围：<https://www.sciencedirect.com/journal/energy-and-ai>

范围匹配：

- 官方定位就是能源与人工智能交叉；
- 明确包括面向能源问题的专用 AI 方法、数据科学、混合数据—物理建模、智能控制，以及能源 AI 的安全、可靠性和伦理；
- 接收 full-length research article；
- 当前为完全开放获取期刊，官方页面列出的 APC 为 USD \(3{,}990\)，实际费用须在投稿时按机构和减免政策再次核对。

契合判定：`VERY_HIGH_SCOPE_FIT_WITH_OPEN_ACCESS_COST`。

相对 Applied Energy 的优势是“AI 可靠性契约”更直接属于期刊核心；劣势是开放获取成本，以及是否满足作者所在单位最新“一区 Top”认定必须另行核验。

### 2.3 Energy：第二备投

官方范围：<https://www.sciencedirect.com/journal/energy>  
官方投稿指南：<https://www.sciencedirect.com/journal/energy/publish/guide-for-authors>

范围匹配：

- 官方范围包括 energy storage、energy and AI、energy and transportation；
- Full Length Article 的正文限制为 \(5{,}000\) 至 \(7{,}000\) 词，摘要不超过 \(250\) 词；
- 当前正文需要至少再压缩约 \(1{,}000\) 个正文词才能进入格式安全区；
- 期刊整体更强调系统级能源应用，单纯电芯 SOH 方法的编辑契合度低于前两者。

契合判定：`MODERATE_TO_HIGH_FIT_AFTER_COMPRESSION`。

### 2.4 Journal of Power Sources：当前不首投

官方范围：<https://www.sciencedirect.com/journal/journal-of-power-sources>

官方范围虽然明确接收 AI、机器学习和多尺度仿真，但同时写明这些工作应由实验验证支撑，并欢迎实验及“经实验验证的”理论与计算贡献。当前公开数据上的结果隔离验证不等于作者团队的新实验室验证。

契合判定：`SUBMISSION_BLOCKER_WITHOUT_EXPERIMENTAL_VALIDATION`。

如果未来获得合作实验室的全新前瞻批次，可以重新评估；在现阶段把它列为主投会显著增加范围拒稿风险。

## 3. 同期期刊论文与差异边界

检索边界：目标期刊官方页面、ScienceDirect 出版记录与 DOI 页面；关键词覆盖 battery SOH、cross-domain、transfer learning、semi-supervised、field application、physics-informed、negative transfer 与 risk control。该检索只能支持限定范围内的定位，不能证明“首次”。

| 同期工作 | 直接证据 | 与 PCHP 的关系 | 状态 |
|---|---|---|---|
| Duan 等，Applied Energy \(2024\)，多源集成迁移学习，DOI <https://doi.org/10.1016/j.apenergy.2024.124245> | 官方摘要说明：在无目标标签的健康预测中进行源域选择、顺序感知 MMD 和多源域适应；仅以 \(12\) 个数据集中的 \(2\) 个作为目标 | 解决“如何提高无标签目标域精度”，不提供对所有隐藏结果成立的逐记录相对损害上界，也不处理已发布前缀不可回写 | `NEAR_NEIGHBOR` |
| Liu 等，Applied Energy \(2025\)，贝叶斯迁移健康评估，DOI <https://doi.org/10.1016/j.apenergy.2024.125260> | 官方记录说明：按电池相似性迁移信息，并对不同场景采用参数更新策略，在两个数据集上验证 | 与跨电池迁移和个体运行状态相关，但保证对象、信息边界和在线输出契约不同 | `NEAR_NEIGHBOR` |
| Hadzalic 等，Energy and AI \(2025\)，\(3{,}000\) 辆车半监督 SOH，DOI <https://doi.org/10.1016/j.egyai.2025.100575> | 官方全文页面说明：多视图协同训练、置信度伪标签和随新标签到达的增量重训练；使用 \(34\) 个国家的真实车队标准容量测量 | 真实车队证据强于 PCHP，但方法允许标签逐步进入并重训练；PCHP 的独特问题是结果到达前的损害约束与历史输出不可回写 | `NEAR_NEIGHBOR_AND_STRONG_FIELD_EVIDENCE` |
| Applied Energy \(2026\) 物理信息迁移学习，DOI <https://doi.org/10.1016/j.apenergy.2025.127161> | 官方摘要说明：把物理知识嵌入激活函数用于电池健康迁移预测 | 说明目标期刊重视物理结合；PCHP 不应宣称物理机理建模，而应强调物理输出范围和在线风险契约 | `ADJACENT_PHYSICS_INFORMED_WORK` |
| Energy and AI \(2026\) 半监督深核 SOH，DOI <https://doi.org/10.1016/j.egyai.2026.100748> | 官方页面说明：在动态放电数据上以少量标签、GPR 不确定性和解释工具提高精度 | 其核心是不确定性与少标签精度，不是相对于受保护预测器、对所有隐藏结果成立的确定性损害预算 | `NEAR_NEIGHBOR` |

限定结论：在上述检索边界内，没有发现同时组合“结果一致的精确损害域、受保护预测状态、前缀因果不可回写、递归非增可行性和跨域电池 SOH 部署”的直接先例；通用投影、度量恒等式、平方损失代数、在线预测与安全回归均存在先验成分。因此允许表述组合差异，不允许使用普遍“首次”或“现有方法不能”的绝对句式。

## 4. 主投路线需要的稿件重排

### 4.1 保留在主文的证据

1. 新车队无容量标签、负迁移和历史记录不可回写的现实场景；
2. PCHP 的受保护状态、损害预算投影和唯一候选保真解释；
3. 绝对损失精确域、损失选择边界及一个紧凑的平方损失结果；
4. \(12\) 域严格嵌套开发结果与候选无关预算可行偏移控制；
5. 强无保护比较器揭示的损害控制—精度机会成本；
6. BaSyTec 结果盲、前瞻锁定的外部验证，并明确其不是前瞻采集或独立实验室重复；
7. NASA 压力失败边界；
8. 一段明确的部署含义和成本标定边界。

### 4.2 移入补充材料的内容

1. 度量空间恒等式的完整证明；
2. 平方损失上下边界的完整代数推导；
3. \(200{,}000\) 配置和 \(256{,}000\) 次递归执行的全部数值审计表；
4. 外部数据解析器失败的完整时间线；
5. 逐协议哈希、所有敏感性明细和非核心复现清单；
6. 不改变核心结论的次级负结果。

## 5. 投稿前硬门槛

| 门槛 | 当前状态 | 关闭条件 |
|---|---|---|
| Applied Energy 摘要 \(\le 250\) 词 | `PASS` | V368 保守计数为 \(250\) 词 |
| 主文应用价值前置 | `PASS_DRAFT` | 引言从新车队上线且容量结果不可得的运维场景起笔，Discussion 与图 \(1\) 保持同一问题契约 |
| 同期工作更新 | `PASS_DRAFT` | 已加入并语义核验 \(2025\)–\(2026\) 近邻工作；投稿前仍需做一次日期刷新 |
| 作者创建代码许可 | `OPEN_COMPLIANCE` | 导师/机构确认许可证并替换 `LICENSE_PENDING` |
| AI 使用声明 | `OPEN_COMPLIANCE` | 作者核对并加入真实、具体的 Elsevier 声明 |
| 作者与机构信息 | `OPEN_COMPLIANCE` | 作者和导师提供最终信息 |
| 电池物理终审 | `OPEN_SCIENTIFIC` | 导师确认物理范围、SOH 定义和运维解释 |
| 最新一区 Top 认定 | `OPEN_INSTITUTIONAL` | 以作者学校/单位投稿时使用的最新中科院或 JCR 清单为准 |

## 6. 决策

当前冻结路线：

1. 主投：Applied Energy，Full Length Research Article；
2. 第一备投：Energy and AI，Full Length Research Article；
3. 第二备投：Energy，Full Length Article；
4. 暂缓：Journal of Power Sources，除非获得新实验室验证。

这一路线不等于录用概率保证。它的作用是让后续每一次删改都服务于明确的编辑问题：PCHP 是否为能源 AI 部署增加了一个现有迁移精度方法没有提供、且已被多域和结果隔离证据验证的可审计风险控制能力。
