# PCHP 理论—实现—证据契约（V340）

审计日期：2026-08-02  
核心实现：`prefix_causal_harm_projection_v321.py`  
方法测试：`test_prefix_causal_harm_projection_v321.py`

## 1. 唯一核心对象

PCHP 的核心对象是一个结果未知的在线输出空间投影契约，而不是某个特定神经网络。给定前缀因果、非增受保护状态 $b_t$、学习候选 $c_t$、已发布上一时刻输出 $p_{t-1}$ 和物理范围 $[0,1.3]$，对称绝对损失版本使用

$$
I_t=\left[\max(0,b_t-\delta),\min(1.3,b_t+\delta,p_{t-1})\right],
\qquad p_t=\Pi_{I_t}(c_t).
$$

方向加权绝对损失版本使用

$$
I_t^{\mathrm{asym}}
=\left[
\max\left(0,b_t-\frac{\eta}{c_{\mathrm{under}}}\right),
\min\left(1.3,b_t+\frac{\eta}{c_{\mathrm{over}}},p_{t-1}\right)
\right].
$$

## 2. 契约矩阵

| 形式结论 | 必要假设 | 实现位置 | 直接验证 | 允许措辞与边界 |
|---|---|---|---|---|
| 前缀不变性 | 当前状态和候选只依赖当前及历史记录 | `causal_nonincreasing_state`、两个投影函数 | 多个前缀长度逐元素相等；V325 未来扩展最大历史差为 $0$ | 可称已发布前缀不被未来记录修改；不可称输入数据无时间戳或泄漏风险 |
| 非增与物理范围 | $b_t$ 非增；常数非负预算；初始上界为 $1.3$ | 每步上下端点与上一输出交集 | 随机轨迹、全部开发域和 NASA 证书 | 可称输出轨迹非增；不可称真实电化学容量绝不恢复 |
| 对称精确上确界 | 标量点预测、绝对损失、任意实数结果 | `worst_case_absolute_loss_increase` | 解析恒等式与数值构造 | $\sup_y(|p-y|-|b-y|)=|p-b|$；不可推广到任意损失 |
| 对称最大损害域 | 同上，预算 $\delta\ge0$ | `prefix_causal_harm_projection` | 管内、管外和零预算测试 | $[b-\delta,b+\delta]$ 是声明保证的必要充分最大集合；不证明基线准确 |
| 非对称精确上确界 | $c_{\mathrm{under}},c_{\mathrm{over}}>0$ 为轨迹内常数；方向加权绝对损失 | `worst_case_asymmetric_absolute_loss_increase` | $10{,}000$ 组标量，最大误差 $3.552713678800501\times10^{-15}$ | 可称精确方向损害几何；不可称该损失函数原创 |
| 非对称最大损害域 | 同上，$\eta\ge0$ | `prefix_causal_asymmetric_harm_projection` | 管内满足、管外构造性违反、方向半径检查 | $[b-\eta/c_{\mathrm{under}},b+\eta/c_{\mathrm{over}}]$；实际代价权重尚未估计 |
| 递归非空 | 受保护状态非增，代价和预算为常数，上一输出由同一递归产生 | 两个投影函数逐步更新上端点 | 随机轨迹与下界可行测试 | 可称给定假设下递归可行；不可推广到任意时变预算或代价 |
| 唯一最接近候选 | 可行集为非空闭区间；目标为严格凸的 $\tfrac12(p-c_t)^2$ | `np.clip` 等价区间欧氏投影 | 每步最近可行值测试 | 可称完整声明约束下唯一最小候选失真解；不可称任意决策效用下全局最优 |
| 单位代价向后兼容 | $c_{\mathrm{under}}=c_{\mathrm{over}}=1$ 且 $\eta=\delta$ | 对称 API 委托给广义实现 | $100$ 条随机轨迹逐元素完全相等 | 非对称扩展是同一核心推论，不是方法分叉 |
| 非法契约失败关闭 | 数组有限、形状一致、基线非增、预算非负、代价正且有限 | `_validated_positive_scalar` 与输入验证 | 非正、非有限、负预算、形状错配及递增基线均抛错 | 可称实现失败关闭；不可称覆盖所有软件或硬件故障 |

## 3. 代码与测试身份

| 工件 | SHA-256 |
|---|---|
| `prefix_causal_harm_projection_v321.py` | `9F4297DB4C3B3C40995BFD656C901C4168470372CF3F203FF9852292D073A96F` |
| `test_prefix_causal_harm_projection_v321.py` | `5C9B3C76F377513E07790F3B0ACD8E79011481538787DF4881B4373FCAD3ADED` |
| `validate_asymmetric_harm_extension_v340.py` | `29379CF37AE77B0F10569683704902CA69C4D511C78C39815BA660E078B7072E` |
| `ASYMMETRIC_HARM_EXTENSION_PREFREEZE_V340.json` | `4BEA581E4B3E86B9DD35CAD054823E9204EA879CB410F551D0542896BCCC89A3` |
| `asymmetric_harm_extension_v340_report.json` | `76D58A04464176B05EC84B369185959261DB8760C08B2D527F2806004D9636AD` |

方法测试共 $22/22$ 通过。NASA 的四个既有 `test_*` 函数已直接导入并全部执行通过；该环境缺少 `pytest`，因此该项记录为直接函数执行而不是 `pytest` 运行。

## 4. 论文允许主张

允许主张：PCHP 把前缀执行、递归非增输出和相对于指定受保护状态的逐记录结果无关损害预算统一为精确输出空间约束；对称与方向加权分段线性损失分别具有精确最大损害域。

禁止主张：普遍安全、绝对准确、基线无偏、真实维护收益、最优业务代价、任意损失的统一保证、任意时变预算下递归可行，或非对称损失本身具有原创性。

