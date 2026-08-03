# PCHP 主张—来源与原创性审计（V365）

审计日期：2026-08-02  
审计对象：`rccp_causal_manuscript_v364/main_en.tex`、`main_zh.tex` 与 `references.bib`  
审计目的：检验相关工作陈述是否被原始来源支持，并把“与最近邻工作的可核验区别”同“无法证明的普适首次”分开。

## 1. 可允许的原创性表述

当前证据允许的最强表述是：

> 在本次检索覆盖的安全回归、在线再校准、无标签共形迁移、受保护测试时适应、电池后校准、递归可行性及基线相对不伤害学习文献中，未发现与 PCHP 完全相同的组合：仅由源域学习的候选、前缀因果受保护状态、递归可行的非增输出，以及对任意隐藏真实值成立的逐记录绝对损失精确损害管。

该表述属于 `SCOPED_GAP_IDENTIFIED`，不是优先权证明。正文不得写成“首个”“首次提出”或“据我们所知第一个”，除非进一步完成可复查的系统检索、同义词扩展、引文追踪和独立复核。

## 2. 检索边界

检索主题覆盖：

1. `safe regression`、`no worse regression`、`negative transfer protection`；
2. `baseline-relative regret`、`online recalibration`、`calibrated regression arbitrary sequences`；
3. `conformal transport unlabeled target covariates`、`covariate shift prediction intervals`；
4. `protected test-time adaptation`、`safe online adaptation no shift`；
5. `battery state of health safe calibration`、`battery conformal transfer`；
6. `recursive feasibility monotone constraints`、`history-coupled online constraints`；
7. `counterfactual harm baseline policy`、`do-no-harm learning`。

优先核查论文正式页面、出版社全文、会议论文集或作者提交的完整论文；搜索摘要只用于发现候选，不作为支持复杂技术主张的最终证据。

## 3. 主张—来源核验矩阵

| 最近邻工作 | 正文使用的主张 | 原始证据位置 | 审计状态 | 与 PCHP 的边界 |
|---|---|---|---|---|
| Li et al., SAFER, AAAI 2017 | 将监督预测投影至半监督回归器构成的凸集；当完整真实标签向量位于该集合时，获得平方损失不劣结论 | 正式论文的方法定义、关于真值向量位于凸集时的定理，以及条件失效时的残差上界 | `SUPPORTED_EXACT`、`NEAR_NEIGHBOR` | 保护完整向量平方误差；条件依赖真实向量在集合内；不是逐记录、结果一致的绝对损失证书，也没有轨迹递归 |
| Marx et al., TMLR 2025；Deshpande et al., UAI 2025 | 在任意或对抗序列上同时追求校准与相对基线的低或渐近消失累积遗憾 | TMLR/OpenReview 正式论文摘要与主要结果；PMLR UAI 正式页面与论文 | `SUPPORTED_EXACT`、`NEAR_NEIGHBOR` | 结果依次揭示并用于更新；保护对象是概率校准和累积遗憾，不是当前结果揭示前的逐点绝对损失管 |
| Tuwani & Beam, 2023 | 利用无标签目标协变量，在所声明的协变量偏移模型下恢复边际覆盖 | PubMed Central 全文的方法与覆盖率目标 | `SUPPORTED_EXACT`、`NEAR_NEIGHBOR` | 保护预测集合的边际覆盖率，依赖所声明的偏移条件；不保护已发布点预测相对基线的损失 |
| Filgueira da Silva et al., 2026 | 电池迁移中结合表征对齐与共形预测区间 | arXiv 完整预印本的方法概述与实验设计 | `SUPPORTED_NARROWER`、`NEAR_NEIGHBOR` | 面向区间覆盖与迁移校准；不是无条件的基线相对逐点损害证书 |
| Bar et al., POEM, NeurIPS 2024 | 检测熵偏移并适应模型参数；无偏移时的准确率和校准保持属于经验结果 | NeurIPS 正式论文摘要、方法和无偏移实验 | `SUPPORTED_EXACT`、`NEAR_NEIGHBOR` | 保护围绕自适应训练过程且为经验保持；模型参数会更新，不是冻结模型上的确定性逐点证书 |
| Nyachionjeka & Bayoumi, 2026 | 利用带标签目标留出集，只在误差改善时选择单调后校准器 | 期刊全文的方法流程与选择规则 | `SUPPORTED_EXACT`、`NEAR_NEIGHBOR` | 需要目标标签并按留出误差选择；不是结果未知时对所有真实值成立的证书 |
| Löfberg, 2012；相关在线约束与预测控制工作 | 递归可行性、时变约束或历史耦合可行集本身已有长期研究 | 原论文定理和问题设定 | `SUPPORTED_EXACT`、`NOT_NOVEL_ALONE` | PCHP 不把递归可行性或时变约束本身作为原创性；增量只在一维损害管诱导的精确可行条件及其与前缀因果输出的结合 |
| Vaskov et al., L4DC 2024 | 可相对安全策略定义并限制反事实伤害 | PMLR 正式论文摘要、定义与约束 | `SUPPORTED_EXACT`、`CONCEPTUAL_NEIGHBOR` | 对象是强化学习策略、状态和控制约束；不是冻结回归器的逐记录绝对损失区域 |
| PCHP 的完整组合 | 仅源域候选、前缀因果受保护状态、递归可行非增输出、任意隐藏结果下的逐记录绝对损失精确预算 | 本文定理、推论与算法契约 | `SCOPED_GAP_IDENTIFIED` | 当前检索未发现完全相同组合；只能支持限定区别 |
| “PCHP 是普适意义上的首个方法” | 优先权主张 | 当前证据不足 | `PRIORITY_PROOF_INSUFFICIENT` | 禁止写入标题、摘要、贡献或结论 |

## 4. 原始来源入口

- SAFER（AAAI 正式论文）：<https://ojs.aaai.org/index.php/AAAI/article/download/10856/10715>
- Calibrated Probabilistic Forecasts for Arbitrary Sequences（TMLR/OpenReview）：<https://openreview.net/forum?id=nuIUTHGlM5>
- Calibrated Regression Against an Adversary Without Regret（PMLR）：<https://proceedings.mlr.press/v286/deshpande25a.html>
- Safe and Reliable Transport of Prediction Models（PubMed Central）：<https://pmc.ncbi.nlm.nih.gov/articles/PMC10760294/>
- Protected Online Entropy Matching（NeurIPS）：<https://proceedings.neurips.cc/paper_files/paper/2024/hash/9b35a0a20d617dc68ae98a7a57df2f51-Abstract-Conference.html>
- Safe-Calibrated Battery State-of-Health Estimation（期刊全文）：<https://www.mdpi.com/2032-6653/17/3/149>
- Conformalized Transfer Learning for Battery State-of-Health Prediction（arXiv）：<https://arxiv.org/abs/2603.24475>
- Do-No-Harm Counterfactual Harm Constraints for Policy Learning（PMLR）：<https://proceedings.mlr.press/v242/vaskov24a.html>

## 5. 已落实到正文的风险控制

1. 中英文相关工作均补入“保证对象—信息或成立条件—与 PCHP 的区别”对照表。
2. SAFER 不再被笼统写成无条件安全，而明确写出完整真值向量位于凸集的主要条件。
3. 在线校准只主张校准与累积遗憾，不把它误写为逐点不伤害。
4. 共形方法只主张集合覆盖，不把覆盖率等同于点预测损失保护。
5. POEM 的无偏移保持明确标注为经验结论。
6. 电池后校准明确标出目标标签与留出集选择条件。
7. 原创性被限定为组合式、部署契约层面的差异；正文显式声明其不构成普适优先权。

## 6. 尚不能由文献审计消除的风险

1. 同义命名遗漏风险：相同数学结构可能出现在不同领域而未使用 `safe regression` 或 `harm budget` 术语。
2. 2026 年新近预印本变化快，正式投稿前必须按投稿日重新运行近邻检索。
3. 即使组合未见完全相同，审稿人仍可能认为构件组合的概念增量不足；最终说服力依赖定理紧性、反例、消融和跨域冻结确认共同构成的证据链。
4. 该审计不能替代真实部署或前瞻实验；正文必须继续保持“部署相关证据”而非“已完成工业验证”的措辞。

## 7. 审计结论

当前版本的原创性主张已经从不可证的宽泛“首次”收缩为可防守的限定区别。最近邻文献确实覆盖了安全投影、累积遗憾、共形覆盖、受保护适应和目标监督校准，但其保护对象或信息边界均不同。PCHP 最值得保留的强方法增量是：把精确、结果一致的逐点绝对损失预算，与源域冻结候选、前缀因果状态和递归非增可行性联结为一个可执行的电池部署契约。
