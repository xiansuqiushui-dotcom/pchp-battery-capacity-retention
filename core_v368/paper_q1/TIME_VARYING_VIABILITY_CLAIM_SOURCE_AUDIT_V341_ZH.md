# V341 时变递归可行性主张—来源核验

## 1. 核验范围

- 核验日期：2026-08-02。
- 核验对象：PCHP 在逐记录变化的非对称成本与损害预算下的递归可行性、候选序列无关可行性和最小当前预算。
- 检索边界：控制理论中的 recursive feasibility / strong recursive feasibility，带时变约束的 MPC，带时变约束的在线凸优化，以及 baseline-relative / asymmetric absolute-loss / monotone projection 等交叉词组。
- 主要检索式：`recursive feasibility time-varying constraints`、`strong recursive feasibility all feasible inputs`、`online convex optimization time-varying constraints`、`asymmetric absolute loss recursive feasibility`、`harm tube regression projection baseline`、`minimum feasible budget online projection monotone`。
- 证据类型：出版社记录、作者公开全文、PMLR、机构论文库和 arXiv 原始论文。搜索结果摘要只用于发现文献，不单独支撑最终定位。

本核验只能支持有边界的差异化定位，不能证明普遍意义上的“首次”。

## 2. 主张—来源矩阵

| ID | 待核验主张 | 主张类型 | 原始来源与证据位置 | 语义核验 | 结论与修正动作 |
|---|---|---|---|---|---|
| V341-C1 | 递归可行性本身是既有控制理论概念 | 背景事实 | Löfberg, *Automatica* 2012, DOI `10.1016/j.automatica.2011.12.003`，作者公开全文第 1--2 页、Definitions 2.1--2.2 | 来源把递归可行性定义为优化问题在后续时刻持续有解，并讨论其失效检测 | `SUPPORTED_EXACT`；正文不得把 recursive feasibility 本身称为新概念 |
| V341-C2 | 对所有可行控制输入序列保持可行对应既有的 strong recursive feasibility 思想 | 能力归属 | Löfberg 2012，Definition 2.2：对所有初始可行状态和所有可行控制输入序列保持后续可行 | 与“所有候选序列均不使 PCHP 递归空集”的量词结构相邻，但对象分别是 MPC 控制输入与 PCHP 投影候选 | `SUPPORTED_NARROWER`；仅写“analogous to strong recursive feasibility”，不得宣称两者为同一一般定理 |
| V341-C3 | 时变约束下的递归可行性已有专门研究 | 背景事实 | Knaup and Tsiotras, *Automatica* 2026, DOI `10.1016/j.automatica.2026.112957`，摘要；研究时变线性系统、无界随机扰动与机会约束下的递归可行性 | 直接支持“时变环境中的递归可行性不是空白领域”，但其系统、约束和结论与 PCHP 不同 | `SUPPORTED_EXACT`；作为控制领域近邻，不作为 PCHP 公式的来源 |
| V341-C4 | 在线优化已经处理时变约束 | 背景事实/近邻 | Garber and Kretzu, ICML 2024, PMLR 235:14988--15005；其动作对固定硬约束可行，并平均近似满足附加时变约束 | 其目标是遗憾与累计约束违反，不是每个输出对未知真值的点态损害管，也不要求电池轨迹单调递归 | `NEAR_NEIGHBOR`；可用于说明研究邻域，不得据此声称其提供相同保证 |
| V341-C5 | 历史依赖的时变可行集与 causal invariance 已用于在线控制 | 背景事实/近邻 | Li et al., *Proc. ACM Meas. Anal. Comput. Syst.* 2021, DOI `10.1145/3460085`，摘要与问题式；可行集随历史状态和动作变化，结果依赖 causal invariance criterion | 研究的是双控制器、聚合可行性信息和动态遗憾；不涉及 PCHP 的绝对损失几何 | `NEAR_NEIGHBOR`；只支持“因果可行性条件有更广泛先例” |
| V341-C6 | 对 PCHP 的时变非对称管，所有候选序列保持可行当且仅当截断下端点非增 | 理论结果 | 本项目 V341 形式证明、冻结契约和独立验证报告；未在检索边界内找到相同对象与闭式条件 | 一般递归可行性属于先验；检索只显示 PCHP 特定标量递归、点态未知结局损害管和该充要条件的组合差异 | `SCOPED_GAP_IDENTIFIED` 且 `PRIORITY_PROOF_INSUFFICIENT`；写成“for the PCHP recursion, we derive...”，禁止 `first`/`unprecedented` |
| V341-C7 | 给定上一已发布输出，最小当前预算为 $\eta_{t,\min}=c_{\mathrm{under},t}(b_t-p_{t-1})_+$ | 理论结果 | 由当前区间非空条件直接求解；等号与紧邻下方已按预冻结规则分别验证 | 属于本形式对象的闭式紧下界，不依赖外部文献授权 | `SUPPORTED_EXACT`（由证明与验证支持）；表述必须限定为 directionally weighted absolute loss 和当前 PCHP 递归 |
| V341-C8 | 上侧成本不决定递归可行性，但改变向上损害半径与输出效用 | 理论边界 | 非对称损失恒等式、V341 证明和 `upper_cost_feasibility_invariance` 验证门 | 下端点仅由 $c_{\mathrm{under},t}$ 与 $\eta_t$ 决定；上端点仍影响可行区形状和投影结果 | `SUPPORTED_EXACT`；不能扩张成“上侧成本不重要” |

## 3. 可直接写入论文的有界定位

英文：

> Recursive feasibility is a longstanding concept in constrained control, and time-varying constraints have been studied in both predictive control and online optimization. Our result is narrower: for the scalar PCHP recursion induced by an outcome-uniform asymmetric loss tube, candidate-universal feasibility reduces exactly to monotonicity of the clipped lower endpoint, while realized-prefix feasibility admits a closed-form minimum current budget.

中文：

> 递归可行性是约束控制中的既有概念，预测控制与在线优化也已研究时变约束。本文的结果更为具体：对于未知结局一致非对称损失管所诱导的一维 PCHP 递归，候选序列无关可行性恰好等价于截断下端点非增，而给定已发布前缀时的当前最小预算具有闭式表达。

## 4. 禁止使用的表述

- “首次提出递归可行性”；
- “首次解决时变约束”；
- “现有方法无法处理时变约束”；
- “普遍适用于任意损失、任意动态系统”；
- “候选序列无关可行性是全新的控制概念”。

## 5. 核验结论

相邻理论没有推翻 V341 的 PCHP 特定结果，但明确排除了宽泛原创性包装。建议把该结果保留为同一 PCHP 核心的精确可行性推论与部署诊断，而不是新架构、第二核心贡献或普遍控制理论首创。

## 6. 原始证据入口

- Löfberg 2012 作者全文：https://people.isy.liu.se/rt/johanl/2011_AUT_OOPS.pdf
- Löfberg 2012 DOI：https://doi.org/10.1016/j.automatica.2011.12.003
- Knaup and Tsiotras 2026：https://doi.org/10.1016/j.automatica.2026.112957
- Garber and Kretzu 2024：https://proceedings.mlr.press/v235/garber24a.html
- Li et al. 2021：https://doi.org/10.1145/3460085

