# PCHP V342 项目交接记录

## 1. 权威稿件

- 英文源码与成稿：`paper_q1/rccp_causal_manuscript_v342/main_en.tex`、`main_en.pdf`。
- 中文源码与成稿：`paper_q1/rccp_causal_manuscript_v342/main_zh.tex`、`main_zh.pdf`。
- 英文补充材料：`supplement_en.tex`、`supplement_en.pdf`。
- 中文补充材料：`supplement_zh.tex`、`supplement_zh.pdf`。
- 作者姓名已固定为 Yuyang Wu / 吴雨阳；单位等元数据暂留待作者确认。

标题：

- 英文：*Risk-Controlled Cross-Domain Battery State-of-Health Estimation via Prefix-Causal Harm-Budget Projection*。
- 中文：基于前缀因果损害预算投影的风险可控跨域锂离子电池健康状态估计。

## 2. 方法与证据主线

方法核心不是重新训练一个更大的 SOH 网络，而是在受保护的前缀因果基线与目标标签自由候选之间，递归构造同时满足在线轨迹约束和最坏情形损害预算的可行域，并把候选投影到该域中。

三层证据必须保持区分：

1. 理论层：前缀因果性、递归可行性、损害管道、非对称成本与零预算不可能性。
2. 开发层：嵌套源域选择、候选控制、强无保护精度边界和源组敏感性。
3. 外部层：BaSyTec 结果隔离合同确认与 NASA 协议压力测试。

## 3. 关键数字

- 开发域：$12$ 个域、$586$ 个物理电芯、$601{,}932$ 条记录。
- PCHP 相对受保护基线的域等权差：$-0.00486$，$95\%$ 区间 $[-0.00670,-0.00301]$。
- 强无保护比较器相对基线的域等权差：$-0.02054$；PCHP 只保留 $23.7\%$ 增益，但强比较器在全部域与电芯上违反预算。
- BaSyTec 主分析：$45$ 个合格电芯、$2{,}969$ 个评分循环，PCHP 减基线为 $-0.00986$，区间 $[-0.01000,-0.00959]$，胜/平/负为 $45/0/0$。
- BaSyTec 强无保护比较器 MAE 为 $0.06172$，PCHP 效用保留为 $8.7\%$；事后固定 $+0.01$ 略优于 PCHP，因此外部自适应主张被禁止。
- NASA 压力测试：$33$ 个电池，保留相对效用和确定性损害控制，但绝对目标不兼容。

## 4. 复现包

目录：`paper_q1/rccp_reproducibility_v342`。

执行：

```powershell
python make_manifest_v342.py
python verify_reproducibility_v342.py --regenerate-figures --write-receipt
```

当前验证结果：`PCHP_V342_REVIEW_LITE_VERIFICATION_PASSED`，共 $619$ 项命名检查通过。包中不含第三方原始压缩包、BaSyTec 记录级结果表或拟合模型；只包含允许重分发的作者代码、协议、失败回执和聚合证据。

## 5. 编译与视觉检查

```powershell
latexmk -xelatex -interaction=nonstopmode -halt-on-error main_en.tex
latexmk -xelatex -interaction=nonstopmode -halt-on-error main_zh.tex
latexmk -xelatex -interaction=nonstopmode -halt-on-error supplement_en.tex
latexmk -xelatex -interaction=nonstopmode -halt-on-error supplement_zh.tex
```

四个 PDF 均已编译通过。主稿各 $25$ 页，补充材料各 $2$ 页；最终日志中没有未定义引用、版面溢出或欠满告警。全部主稿页面和补充材料页面已渲染为图像检查，新 BaSyTec 中英文图无重叠、裁切或乱码。

## 6. 不得回退的写作边界

- 不得声称 PCHP 精度优于强无保护比较器。
- 不得把 BaSyTec 称为现实前瞻性实验室试验。
- 不得把 BaSyTec 解释为外部自适应机制验证。
- 不得把 NASA 解释为高精度绝对 SOH 验证。
- 不得把循环记录数当作独立样本量。
- 不得删除两次模式失败、低参考容量排除或固定偏移诊断。

## 7. 下一步

1. 由吴雨阳和导师补齐作者与机构元数据。
2. 完成电池领域术语与物理合理性审读。
3. 选择方法定位匹配的一区 Top 目标期刊，再按其格式压缩和重排。
4. 选定代码许可并生成公开发布版本。
5. 除非重新冻结新的外部纳入标准和判定门，否则停止继续搜索数据集；当前最有价值的工作是投稿定位与审稿防守，而不是继续累积异质公开数据。
