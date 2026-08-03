# PCHP 跨骨干可移植性审计预冻结协议（V375）

冻结日期：2026-08-02  
状态：`PREFROZEN_BEFORE_V375_EXECUTION`  
研究角色：当前 PCHP 的次级开发稳健性审计；不增加方法模块、不重新选择主模型，也不构成新外部确认。

## 1. 审稿风险与检验目标

V368 的理论保证与候选模型无关，但主要经验结果使用 ExtraTrees。审稿人仍可能提出一个经验层面的替代解释：开发效用来自某一个树模型的误差结构，而不是 PCHP 输出契约对不同学习候选都具有可用性。

V375 因此只检验：保持数据、特征合同、训练信息、每域源域选择的 (alpha)、损害预算和 PCHP 算子不变，仅替换回归骨干后，PCHP 相对各骨干自身受保护因果状态的域级效用是否仍为负向 MAE 差，并继续满足全部确定性证书。

该审计不比较哪个骨干最好，也不把骨干本身写成方法创新。

## 2. 冻结骨干与公平性合同

主骨干 ExtraTrees 保持 V368 不变。次级骨干及其既有固定配置为：

1. Ridge：源域内中位数插补与缺失指示、标准化、(ell_2) 正则系数 (10)；
2. HistGradientBoosting：绝对误差损失、学习率 (0.05)、(250) 次迭代、最多 (31) 个叶节点、叶节点最少 (20) 条记录；
3. LightGBM：L1 回归、(300) 棵树、学习率 (0.04)、(31) 个叶节点、最少 (20) 条叶内记录、行与列采样率 (0.9)、(ell_2) 正则系数 (1)。

全部随机设置沿用种子 (20{,}260{,}801)。每个外层目标域完整排除于训练；源域每个物理电芯最多保留 (80) 条时间均匀记录。每个骨干均训练两个模型：受保护模型使用原始早期充电特征并采用普通池化权重；候选模型增加投运参照绝对变化特征，并使每个源数据域贡献相同总权重。

次级骨干不做额外超参数搜索。对每个外层域，直接使用 V326 已由其余 (11) 个源域选择并冻结的 (alpha)；该 (alpha) 是为主骨干选择的，V375 不为次级骨干重新调参，从而检验零额外调参可移植性。

## 3. 冻结数据与身份

开发 roster：(12) 个已打开数据域、(586) 个物理电芯、(601{,}932) 条参考窗口后记录。外层独立单位为完整数据域；循环记录嵌套于电芯，电芯嵌套于数据域。

| 输入 | SHA-256 |
|---|---|
| `batterylife_early_charge_soh_v109.parquet` | `89ce9711f91b0ba7d4a14db7766055a76cce1223a553baff270abac08fed6c8e` |
| `batterylife_external_hust_rwth_v124.parquet` | `2765e6ee85b7ed4000140b55396e40d271cfc9ecf464663d391a607964cdd9d8` |
| `batterylife_sdu_early_charge_soh_v145.parquet` | `8db4f121dad4afb151327e85facfd6da1346f2b56409a9b487e7aaa345e33613` |
| `batterylife_matr_early_charge_soh_v151.parquet` | `00395c408410efa6d72c7e5936964993388739ec52ced6693fc4f3ff4b544542` |
| `nested_prefix_causal_selection_v326/nested_alpha_selections_v326.csv` | `7440327383458225bd8b92f2998620eec4d8d924b8e31cfab05428faaca81eab` |
| `nested_prefix_causal_selection_v326/nested_source_only_alpha_selection_v326_report.json` | `0197b5c8f2b2f42168d88ea4fa0625ca100a5c852a6af5df451f15dbfba02e8f` |
| `evaluate_backbone_generality_v314.py` | `888c3502909a7d10d4a9ba0279ef7a947be56f54960cf85ee89fea87c3888bcc` |
| `analyze_method_ablation_v315.py` | `60bef7d6449c4b668e63e8dbac1b4f8fe828e3c5b6161138a784eafda46f3f5d` |
| `prefix_causal_harm_projection_v321.py` | `ce7288a129c17114e1ca57432c6417beba7938d58db2b1fd0a87171c479eb54c` |

全部数据域历史上已经打开，故 V375 是协议锁定的回顾性开发稳健性证据。

## 4. 冻结估计目标与不确定性

对每个次级骨干 (m)，主要估计目标为

\[
\Delta_m
=
\frac{1}{12}\sum_{d=1}^{12}
\left(
\operatorname{MAE}^{\mathrm{cell}}_{d,m,\mathrm{PCHP}}
-
\operatorname{MAE}^{\mathrm{cell}}_{d,m,\mathrm{protected}}
\right),
\]

其中每个域先在每个电芯内对记录绝对误差取平均，再对该域电芯等权平均，最后对 (12) 个域等权平均。负值有利于 PCHP。

使用固定种子 (20{,}260{,}802) 的 (100{,}000) 次域聚类 bootstrap。三个次级骨干构成同一稳健性家族，报告 Bonferroni 调整后的双侧 (98.33\%) 百分位区间；等价分位点为 (0.008333\) 与 (0.991667)。该分析不做骨干间优劣检验。

## 5. 决策门与停止规则

只有三个次级骨干全部同时满足下列条件，才判定 `RETAIN`：

1. (Delta_m<0)；
2. Bonferroni 调整后的 (98.33\%) 区间上端点小于 (0)；
3. 至少 (9/12) 个外层域改善；
4. 全部记录满足 (lvert p-bvert\leq0.01)；
5. 最大观测绝对损失增量不超过 (0.01)；
6. 受保护状态和 PCHP 输出均非增、位于 ([0,1.3]) 且前缀重放差为 (0)；
7. 外层目标标签不进入拟合、(alpha) 选择、预处理或判定门。

若任一骨干的效用门失败，则判定 `REJECT`，不得删除该骨干、改用记录等权统计量、重新调参或把“多数骨干通过”改写为预冻结成功。确定性证书若失败，则属于实现失败，必须先修复后重新版本化，且不得读取修复后的效用结果作为原协议结论。

若 `RETAIN`，允许在补充材料中表述：“在三个结构不同且未额外调参的次级骨干上，PCHP 相对各自受保护状态均保持域级效用和确定性证书。”禁止表述为“所有模型普遍有效”“深度学习验证”或“骨干无关的经验定理”。

