# 非对称损害预算扩展预冻结决策（V340）

审计日期：2026-08-02  
最终决策：`RETAIN_AS_CORE_COROLLARY`  
预冻结记录：`ASYMMETRIC_HARM_EXTENSION_PREFREEZE_V340.json`

## 1. 决策对象

该扩展不是一套新模型，也不是第二个核心贡献。它检验同一 PCHP 输出空间契约能否在低估与高估后果不相等时保持精确、结果未知且递归可行。

定义方向加权绝对损失

$$
\ell_{\mathrm{asym}}(p,y)
=c_{\mathrm{under}}(y-p)_+
+c_{\mathrm{over}}(p-y)_+,
\qquad c_{\mathrm{under}},c_{\mathrm{over}}>0.
$$

预冻结待证结论为

$$
\sup_y\left[\ell_{\mathrm{asym}}(p,y)-\ell_{\mathrm{asym}}(b,y)\right]
=
\begin{cases}
c_{\mathrm{over}}(p-b), & p\ge b,\\
c_{\mathrm{under}}(b-p), & p<b,
\end{cases}
$$

以及损害预算 $\eta$ 下的精确可行域

$$
\mathcal H^{\mathrm{asym}}_{\eta}(b)
=\left[b-\frac{\eta}{c_{\mathrm{under}}},
b+\frac{\eta}{c_{\mathrm{over}}}\right].
$$

相应递归区间为

$$
I_t^{\mathrm{asym}}
=\left[
\max\left(0,b_t-\frac{\eta}{c_{\mathrm{under}}}\right),
\min\left(1.3,b_t+\frac{\eta}{c_{\mathrm{over}}},p_{t-1}\right)
\right].
$$

## 2. 预冻结门槛与结果

预冻结状态为 `PREFROZEN_BEFORE_IMPLEMENTATION`，失败规则是任一门槛失败即 `REJECT`，不得写入论文。

| 预冻结门槛 | 直接证据 | 结果 |
|---|---|---|
| 解析上确界与分段线性直接回放一致 | $10{,}000$ 组随机标量；最大误差 $3.552713678800501\times10^{-15}$ | PASS |
| 管内全部点满足预算、管外邻点存在构造性反例 | 精确内外边界检查 | PASS |
| 单位代价完全恢复原始 PCHP | $100$ 条随机轨迹逐元素相等 | PASS |
| 前缀不变、非增、物理范围与逐点预算同时成立 | 随机轨迹及多前缀检查 | PASS |
| 零预算恒等 | 输出与受保护状态逐元素相等 | PASS |
| 方向代价只收紧对应一侧半径 | 独立上下方向检查 | PASS |
| 非法代价、预算、基线与形状失败关闭 | 反例契约检查 | PASS |
| 原对称测试不回归 | 方法测试 $22/22$ 通过 | PASS |

机器报告：`asymmetric_harm_extension_v340/asymmetric_harm_extension_v340_report.json`，状态为 `ASYMMETRIC_HARM_EXTENSION_RETAINED`。

## 3. 原创性边界

方向加权绝对损失及其与分位数损失的关系是既有统计工具，不构成本文原创性。本文可主张的是：

1. 对该损失给出相对于指定受保护预测、对任意隐藏结果成立的精确逐记录损害上确界；
2. 由上确界得到必要且充分的最大方向相关可行域；
3. 将该可行域与前缀因果状态、物理范围和上一时刻已发布输出组合，并保持递归非空与非增；
4. 单位代价时精确退化为原始 PCHP，而不是另起方法分支。

因此，论文中只把它写成核心 PCHP 的一个推论。禁止将“非对称损失”本身称为新损失、第二贡献或新的电池架构。

## 4. 应用解释与未闭合边界

$c_{\mathrm{under}}$ 与 $c_{\mathrm{over}}$ 表示两类方向错误的相对后果，$\eta$ 以决策损失单位表示。区间只由代价比及与之共同缩放的预算决定。当前实验没有从维护、质保或运营数据估计这些代价，因此只支持形式化决策损失映射，不支持真实业务收益或最优代价权重主张。

阈值型、非线性或随时间变化的损失不由本推论覆盖，必须重新推导精确可行域与递归可行条件。

## 5. 权威工件与哈希

| 工件 | SHA-256 |
|---|---|
| `ASYMMETRIC_HARM_EXTENSION_PREFREEZE_V340.json` | `4BEA581E4B3E86B9DD35CAD054823E9204EA879CB410F551D0542896BCCC89A3` |
| `asymmetric_harm_extension_v340_report.json` | `76D58A04464176B05EC84B369185959261DB8760C08B2D527F2806004D9636AD` |
| `prefix_causal_harm_projection_v321.py` | `9F4297DB4C3B3C40995BFD656C901C4168470372CF3F203FF9852292D073A96F` |
| `test_prefix_causal_harm_projection_v321.py` | `5C9B3C76F377513E07790F3B0ACD8E79011481538787DF4881B4373FCAD3ADED` |
| `validate_asymmetric_harm_extension_v340.py` | `29379CF37AE77B0F10569683704902CA69C4D511C78C39815BA660E078B7072E` |

