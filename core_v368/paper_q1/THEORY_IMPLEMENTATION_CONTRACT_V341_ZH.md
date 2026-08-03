# PCHP 理论—实现—证据契约（V341）

审计日期：2026-08-02  
权威实现：`prefix_causal_harm_projection_v321.py`  
权威测试：`test_prefix_causal_harm_projection_v321.py`

## 1. 唯一核心对象

PCHP 是结果未知条件下的在线输出空间契约，而不是新的神经网络结构。对非增受保护状态 $b_t$、候选预测 $c_t$、上一时刻已发布输出 $p_{t-1}$ 和物理范围 $[0,1.3]$，对称版本为

$$
I_t=\left[\max(0,b_t-\delta),\min(1.3,b_t+\delta,p_{t-1})\right],
\qquad p_t=\Pi_{I_t}(c_t).
$$

对时变方向代价与预算，定义

$$
L_t=\max\left(0,b_t-\frac{\eta_t}{c_{\mathrm{under},t}}\right),
\qquad
U_t=\min\left(1.3,b_t+\frac{\eta_t}{c_{\mathrm{over},t}}\right),
$$

$$
I_t^{\mathrm{tv}}=\left[L_t,\min(U_t,p_{t-1})\right].
$$

## 2. V341 新增的精确可行性结论

在 $b_t$ 非增、$\eta_t\ge 0$ 且 $c_{\mathrm{under},t},c_{\mathrm{over},t}>0$ 下：

1. 给定已经实现的前缀，递归区间非空当且仅当 $L_t\le p_{t-1}$；
2. 对每一条候选序列都保持递归可行，当且仅当对所有 $t\ge2$ 有 $L_t\le L_{t-1}$；
3. 给定 $p_{t-1}$，当前最小可行预算为

$$
\eta_t^{\min}=c_{\mathrm{under},t}(b_t-p_{t-1})_+;
$$

4. $c_{\mathrm{over},t}$ 改变上侧半径和投影结果，但不决定递归区间是否为空。

若日程预先声明，第二条给出执行前的全轨迹证书；若日程仅按前缀到达，系统逐步检查第一条并在违反时失败关闭，不得静默扩大预算。

## 3. 理论—代码—验证矩阵

| 形式结论 | 必要假设 | 实现位置 | 直接证据 | 允许措辞与边界 |
|---|---|---|---|---|
| 前缀不变性 | 当前状态与候选只依赖当前及历史记录 | `causal_nonincreasing_state` 与三种投影 API | 多前缀逐元素完全相等 | 可称已发布前缀不被未来记录修改；不可声称输入数据无泄漏 |
| 对称精确损害域 | 标量点预测、绝对损失、任意实数结果 | `prefix_causal_harm_projection` | 管内充分、管外构造性违反、零预算同一性 | $[b-\delta,b+\delta]$ 是该声明的必要充分最大集合；不可推广到任意损失 |
| 非对称精确损害域 | 方向加权绝对损失，正有限代价 | `prefix_causal_asymmetric_harm_projection` | $10{,}000$ 个标量重放及方向半径检查 | 损害域为 $[b-\eta/c_{\mathrm{under}},b+\eta/c_{\mathrm{over}}]$；损失形式本身不是原创 |
| 时变已实现前缀可行性 | 非增 $b_t$、非负预算、正有限代价 | `prefix_causal_time_varying_asymmetric_harm_projection` | $3{,}000$ 条轨迹，判定逐次等价 | 可称精确在线诊断；不可称任意非法日程也能被修复 |
| 时变候选普适可行性 | 同上；全轨迹证书还要求日程预先声明 | `time_varying_asymmetric_harm_tube_bounds` | $2{,}160$ 个穷举日程、$58{,}320$ 条候选轨迹；$1{,}000$ 个构造反例 | 可称下端点非增是充要条件；递归可行性概念不是本文首创 |
| 最小预算下限 | 给定当前 $b_t$、上一输出 $p_{t-1}$ 与低估代价 | `minimum_viable_asymmetric_budget` | $1{,}000$ 次等号可行、$1{,}000$ 次紧邻下方拒绝、$500$ 次零下限 | 可称精确可行性下限；不可称已完成业务成本标定 |
| 上侧代价不决定空集 | $b_t\in[0,1.3]$ 且独立损害管非空 | 广义投影算子 | $2{,}000$ 条只改变上侧代价的轨迹 | 只限空集判定；输出数值仍会变化 |
| 向后兼容 | 常数日程；单位代价恢复对称 API | 原 API 委托给广义算子 | $1{,}000$ 条轨迹、$2{,}000$ 次数组完全相等 | V341 是同一输出契约的精确推论，不是第二方法 |
| 非法契约失败关闭 | 数组有限、形状一致、预算非负、代价为正、基线非增 | 输入验证与空区间异常 | $11/11$ 类非法契约拒绝；全套 $28/28$ 测试通过 | 可称实现失败关闭；不可扩展为一般软硬件可靠性保证 |

## 4. 冻结身份

| 工件 | SHA-256 |
|---|---|
| `prefix_causal_harm_projection_v321.py` | `CE7288A129C17114E1CA57432C6417BEBA7938D58DB2B1FD0A87171C479EB54C` |
| `test_prefix_causal_harm_projection_v321.py` | `0509488DDFA66094987BA36207C7BB93656642CD571B1773FE36F142814CEE25` |
| `validate_time_varying_viability_v341.py` | `9D40E71875CFF1933623E149BB0A1BA8DE4B454143DE91030A6C01C83811A0D6` |
| `TIME_VARYING_VIABILITY_PREFREEZE_V341.json` | `80C3662681FBC71612470A16910E0A7199D1DB767CC8826405A77EF87A001127` |
| `time_varying_viability_v341_report.json` | `7D449A41813C74366A3FBCFD2ECB3CB2B06F6F7380EAFC13DF49A1F121C6F6CA` |

## 5. 正文主张边界

允许主张：PCHP 把前缀执行、递归非增输出和相对于指定受保护状态的逐记录结果无关损害预算统一为精确输出空间约束；对方向加权、时变契约给出精确损害管、递归可行性充要条件和最小预算下限。

禁止主张：普遍安全、绝对准确、基线无偏、真实维护收益、最优业务代价、任意损失保证、任意时变日程自动可行，或递归可行性与时变约束概念由本文首次提出。
