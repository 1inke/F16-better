# 方法来源与核验范围

2026-10-03定向方法检索，非系统综述、非论文引用量审计。使用原论文的官方会议/出版方页面核对标题、作者、年份和内容。Semantic Scholar API一次访问返回Internal Error，未得到结构化校验结果；不称S2验证通过。官方主来源已确认存在。WSL镜像PDF404不作为证据；Roberts改用出版方原文核验。

以下来源支持方法设计，**不支持直接宣布当前F16或ETH/UCY的失败原因**。

1. Hewitt, J., & Liang, P. (2019). *Designing and Interpreting Probes with Control Tasks*. EMNLP-IJCNLP, 2733–2743. [官方ACL原文与元数据](https://aclanthology.org/D19-1275/)，DOI:10.18653/v1/D19-1275。
   - 用途：探针的读出准确率需配套记忆/容量控制。
   - 限制：原实验是NLP；本文分组置换和共同维度对照是轨迹任务的设计，不是原作者已验证的ETH/UCY结论。

2. Ravichander, A., Belinkov, Y., & Hovy, E. (2021). *Probing the Probing Paradigm: Does Probing Accuracy Entail Task Relevance?* EACL, 3363–3377. [官方ACL原文与元数据](https://aclanthology.org/2021.eacl-main.295/)，DOI:10.18653/v1/2021.eacl-main.295。
   - 用途：表示可读出某属性，不足以证明原任务使用该属性。
   - 限制：需要额外功能敏感性与任务响应证据；也不能将弱探针当作信息不存在的证明。

3. Roberts, D. R., et al. (2017). *Cross-validation strategies for data with temporal, spatial, hierarchical, or phylogenetic structure*. Ecography, 40(8), 913–929. [出版方原文](https://nsojournals.onlinelibrary.wiley.com/doi/10.1111/ecog.02881)，[机构元数据](https://epub.uni-regensburg.de/39299/)，DOI:10.1111/ecog.02881。
   - 用途：结构相关数据的验证不能忽略时间/层级相关；分块可能引入额外外推，必须报告分布变化。
   - 年份：在线发表2016-12-08，卷期2017；机构入库2019不是论文发表年份。
   - 限制：其原研究以生态学为主；分组方式需由本项目真实轨迹身份和时间元数据决定，不能凭论文保证现有样本独立。

4. Adebayo, J., Gilmer, J., Muelly, M., Goodfellow, I., Hardt, M., & Kim, B. (2018). *Sanity Checks for Saliency Maps*. NeurIPS 31. [官方论文与摘要](https://proceedings.neurips.cc/paper_files/paper/2018/hash/294a8ed24b1ad22ec2e7efea049b8737-Abstract.html)。
   - 用途：归因应做参数/数据相关的sanity checks，不能依赖漂亮热图。
   - 限制：通过sanity check仍不是真实世界因果证明；本调查不重训随机标签的原F16，只为新探针设计置换控制、对临时副本做可选归因检查。

5. Kooij, J. F. P., Flohr, F., Pool, E. A. I., & Gavrila, D. M. (2019). *Context-Based Path Prediction for Targets with Switching Dynamics*. IJCV, 127, 239–262. [出版方原文](https://link.springer.com/article/10.1007/s11263-018-1104-4)，DOI:10.1007/s11263-018-1104-4。
   - 用途：状态转换预测可使用观察动作、交互与环境等不同来源的上下文，动机上值得做输入可用性清单。
   - 年份：在线2018-07-02，卷期2019。
   - 限制：其车载视觉/路缘/动作cue和短时预测不能移植成当前坐标输入已具备的模态；本设计不承诺收益，也不自动新增视频/地图数据。

## 项目证据分级

- 已核实结构/数据schema：当前本地源码、已有cache/可视化audit及本轮只读数组检查。
- 历史描述动机：旧变化检测器、nearest-neighbor诊断、人类32例记录；不同checkpoint或选择偏差须单列。
- 新推断：本文待执行的分组探针、盲评、输入敏感性；尚无结果。
- 不可作结论：不可预测比例、表示信息论丢失、某模块为根因、部署泛化收益或major通过。

不直接用医学RCT层级给机器学习方法论文打分；按本任务的primary-source真实性、方法适用性和本地证据边界记录。未执行独立评审或完整六阶段研究，不伪造审核者意见。
