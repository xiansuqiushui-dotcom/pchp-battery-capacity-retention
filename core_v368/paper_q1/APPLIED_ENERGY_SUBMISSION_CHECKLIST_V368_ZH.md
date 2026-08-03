# Applied Energy 投稿就绪清单（V368）

核验日期：2026-08-02  
文章类型：Full Length Research Article  
当前稿件：`rccp_causal_manuscript_v368/`

## 已关闭的期刊适配项

| 项目 | 官方要求或编辑目的 | V368 证据 | 状态 |
|---|---|---|---|
| 期刊范围 | 储能、能源 AI、能源数据分析与决策方法 | `VENUE_ROUTE_AND_CONCURRENT_WORK_AUDIT_V368_ZH.md` | `PASS` |
| 英文题名 | 简洁、信息充分，尽量避免不常见缩写 | 题名直接给出跨域锂离子电池 SOH 问题和前缀因果损害预算投影 | `PASS` |
| 摘要长度 | 不超过 \(250\) 词 | 保守机械计数为 \(247\) 词 | `PASS` |
| Highlights | \(3\) 至 \(5\) 条，每条最多 \(85\) 字符 | `highlights.txt` 共 \(5\) 条，长度分别为 \(74,70,77,66,67\) | `PASS` |
| Graphical abstract | 建议横向，最低约 \(1328\times531\) 像素或同比例更高 | `graphical_abstract_applied_energy_v368.png` 为 \(2340\times900\) 像素、约 \(180\) dpi；另有 PDF 矢量版 | `PASS` |
| 同期工作 | 体现目标期刊近年门槛并避免过时原创性定位 | 新增 Applied Energy 多源适配、贝叶斯迁移及 Energy and AI 真实车队半监督工作 | `PASS` |
| 引用解析 | 正文引用与文献库一致 | 双语 BibTeX 重建后无未定义引用 | `PASS` |
| 生成式 AI 声明 | 使用生成式 AI 辅助稿件准备时必须披露 | 双语正文已加入具体工具、用途、人工核验责任和无生成图像说明 | `PASS_PENDING_AUTHOR_APPROVAL` |
| 数据与代码声明 | 说明数据来源、仓库与不可再分发原因 | 双语正文已给出 Zenodo、NASA、第三方条款和 review-lite 复现边界 | `PASS_WITH_LICENSE_GATE` |
| 编译与版面 | PDF 可读且无引用、溢出或重叠问题 | V373 修订后英文与中文正文均为 \(32\) 页，补充材料均为 \(3\) 页；日志扫描与关键页目视检查通过 | `PASS` |

## 尚未关闭的投稿门槛

| 门槛 | 当前状态 | 最小关闭动作 | 决策责任 |
|---|---|---|---|
| 作者单位和通讯信息 | `OPEN` | 填写单位、完整地址、邮箱、ORCID 和通讯作者 | 作者/导师 |
| CRediT 作者贡献 | `OPEN` | 按真实贡献列出 Conceptualization、Methodology、Software、Validation、Writing 等角色 | 全体作者 |
| 基金与致谢 | `OPEN` | 填写真实基金；无专项基金时使用期刊建议的无基金声明 | 作者/导师 |
| 利益冲突 | `OPEN` | 使用 Elsevier declarations 工具生成并上传声明 | 全体作者 |
| 作者代码许可证 | `OPEN_COMPLIANCE` | 由作者和机构确认许可证；替换正文中的未来时表述 | 作者/机构 |
| 复现包长期仓库 | `OPEN_COMPLIANCE` | 将最终包存入具有持久标识符的仓库并在正文链接 | 作者/导师 |
| 投稿信确认句 | `OPEN` | 全体作者确认原创、未一稿多投和作者顺序 | 全体作者 |
| 电池物理终审 | `OPEN_SCIENTIFIC` | 导师逐项确认 SOH 定义、物理范围、单调约束和运维解释 | 导师 |
| 预算 \(\delta\) 的实际代价解释 | `OPEN_EMPIRICAL_BOUNDARY` | 有可靠成本范围时预冻结敏感性；否则保持研究预算定位 | 作者/导师 |
| 新实验室验证 | `ACCEPTABLE_RISK_FOR_PRIMARY_ROUTE` | Applied Energy 官方范围未将其写为计算论文硬条件；若编辑要求，再寻求未访问新批次 | 作者/合作方 |
| 最新一区 Top 认定 | `OPEN_INSTITUTIONAL` | 按投稿时作者单位采用的中科院/JCR 清单核验 | 作者/单位 |

## 提交文件清单

- `main_en.tex` 及全部依赖图、表和 `references.bib`；
- 编译检查用 `main_en.pdf`；
- `supplement_en.tex` 与 `supplement_en.pdf`；
- `highlights.txt`；
- `graphical_abstract_applied_energy_v368.png` 或 PDF 版；
- `APPLIED_ENERGY_COVER_LETTER_V368.md` 完成占位符后的正式稿；
- Elsevier competing-interests 声明文件；
- 最终数据可得性、代码可得性和生成式 AI 使用声明；
- 最终长期存档的复现包链接。

## 编辑端最可能的三个问题

1. **这是否只是简单裁剪？** 当前答复证据是精确结果一致几何、精确在线可行核、零预算不可能性、唯一候选保真解、前缀不可回写和损失选择边界组成的统一契约；可行核证明每个核内输出均能在任意未来非增受保护状态路径下继续执行，而核外输出已经违反当前契约。不得把原创性放在裁剪或三角不等式本身。
2. **没有真实车队或新实验室是否缺少应用价值？** 当前答复是部分充电先于参考容量到达的明确运行问题、多域严格嵌套评估、基于留出公开数据的结果盲且协议锁定外部确认、强比较器机会成本和失败压力测试；同时明确不声称电化学安全，也不把外部确认误称为前瞻采集或独立实验室重复。该风险可以投稿，但不能说已彻底消除。
3. **为什么不能直接使用更准的无保护模型？** 当前答复是无保护比较器取得更低 MAE，却在所有开发域违反声明预算；PCHP 的贡献是可审计地约束风险，而不是无条件争夺最低平均误差。

当前结论：V368 已满足 Applied Energy 可由我们自行机械关闭的主要格式和叙事要求；剩余门槛需要作者、导师或机构提供真实信息，不能由稿件生成流程代填。
