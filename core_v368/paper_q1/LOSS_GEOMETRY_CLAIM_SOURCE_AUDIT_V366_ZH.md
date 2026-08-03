# PCHP 损失几何扩展主张—来源审计（V366）

审计日期：2026-08-02  
审计对象：拟议的度量损失恒等式、有界平方损失精确域、无界平方损失不可能性及其与前缀递归投影的组合。

## 1. 检索边界

本次检索覆盖以下方向及同义表达：

1. `safe prediction`、`no worse regression`、`worst-case performance gain`；
2. `pointwise loss guarantee`、`per-instance safe prediction`、`outcome-wise loss dominance`；
3. `baseline-relative loss`、`prediction projection`、`bounded squared loss`；
4. `prediction with expert advice`、`multiple losses`、`mixable loss`；
5. `metric loss`、`worst-case loss difference`、`outcome-uniform prediction`；
6. 安全弱监督学习、半监督回归、在线校准、统计决策论和安全控制等邻近领域。

检索优先使用正式论文、会议论文集、PubMed/出版社元数据页和作者完整论文。有限检索只能支持限定区别，不能证明普适优先权。

## 2. 最近邻来源

### 2.1 SAFER

Li、Zha 与 Zhou 的 SAFER 将监督回归预测投影到多个半监督回归器形成的凸集合。当完整真实标签向量位于该集合时，投影的向量平方误差不劣于监督基线；其最坏性能增益也在同一真值集合内定义。

- 原始论文：<https://ojs.aaai.org/index.php/AAAI/article/download/10856/10715>
- 证据位置：正文 Theorems 1–3 及几何投影解释。
- 状态：`SUPPORTED_EXACT`、`NEAR_NEIGHBOR`。
- 边界：真值被限制在回归器凸集合中，保护完整向量平方误差；不是每条记录在任意隐藏真实值下的损害域，也没有前缀非增递归。

### 2.2 SAFEW

Li、Guo 与 Zhou 的 SAFEW 对多种分类和回归凸损失采用最坏性能增益最大化，并在真实标签向量可以表示为基学习器凸组合时给出安全性；论文同时指出其一般结论是充分条件和性能下界。

- 元数据：<https://pubmed.ncbi.nlm.nih.gov/31199253/>
- 作者完整论文：<https://www.lamda.nju.edu.cn/liyf/paper/TPAMI19-SafeW.pdf>
- 证据位置：Section 1.1、Section 3.1、Theorems 1–2。
- 状态：`SUPPORTED_EXACT`、`NEAR_NEIGHBOR`。
- 边界：其多损失能力属于真值集合条件下的向量级弱监督集成，不是对所有隐藏结果成立的逐点精确域；也不研究电池前缀执行与轨迹递归。

### 2.3 预测与专家建议

Chernov 与 Vovk 允许不同专家使用不同的适当可混合损失，并给出相对于专家的累计损失界；真实结果在每轮预测后揭示，界的对象是累计遗憾。

- 原始论文：<https://arxiv.org/pdf/0902.4127>
- 证据位置：Theorems 1–2、Corollaries 1–3。
- 状态：`SUPPORTED_EXACT`、`NEAR_NEIGHBOR`。
- 边界：这是多损失、序贯结果揭示后的累计保证，不是当前结果揭示前相对指定基线的零概率逐点损害管。

### 2.4 度量恒等式

度量损失结论

\[
\sup_y[d(p,y)-d(b,y)]=d(p,b)
\]

直接由三角不等式上界和 \(y=b\) 取等得到。三角不等式是既有数学原理，因此不得把该恒等式本身包装成新的数学原语。

- 状态：`PRIOR_ART_INGREDIENT`、`NOT_NOVEL_ALONE`。
- 允许角色：作为统一 PCHP 损害几何和说明绝对损失并非孤立技巧的基础命题。

## 3. 拟议主张矩阵

| 主张 | 最近邻 | 状态 | 允许表述 |
|---|---|---|---|
| 任意度量损失的最坏结果一致损害等于预测间距离 | 三角不等式 | `PRIOR_ART_INGREDIENT` | “由度量性质可得统一恒等式”；不得称为首次 |
| 有界区间上平方损失具有基线位置相关的精确非对称可行域 | SAFER、SAFEW | `SCOPED_GAP_IDENTIFIED` | “本文推导用于该执行契约的精确闭式域”；不得宣称普适优先权 |
| 真实值遍历整个实数轴时，平方损失的任意非平凡更新具有无限最坏损害 | 安全回归与在线多损失工作 | `SCOPED_GAP_IDENTIFIED` | “该线性发散给出本文在整个实数结果空间上的可行性边界”；不得外推到任意无界真值子集 |
| 上述平方损失域与前缀因果、非增输出和递归可行投影的组合 | SAFER、SAFEW、专家建议 | `SCOPED_GAP_IDENTIFIED` | “在本次检索边界内未发现相同组合” |
| “本文首次建立结果一致损失几何” | 当前证据 | `PRIORITY_PROOF_INSUFFICIENT` | 删除 |

## 4. 审计结论

拟议 V366 扩展没有被最近邻工作直接覆盖，但其原创性不能放在三角不等式或“考虑平方损失”本身。可防守的增量是：在同一个结果未知执行契约下，给出损失依赖的精确可行性分界，并把有界平方损失的闭式域接入前缀因果、物理范围和递归非增投影。SAFER/SAFEW 的安全性依赖真值向量位于基学习器集合，多损失专家建议保护累计遗憾；这些信息边界均与 V366 拟议对象不同。

正式投稿前仍须按投稿日更新检索，并继续使用“在检索文献中未发现相同组合”的限定措辞。
