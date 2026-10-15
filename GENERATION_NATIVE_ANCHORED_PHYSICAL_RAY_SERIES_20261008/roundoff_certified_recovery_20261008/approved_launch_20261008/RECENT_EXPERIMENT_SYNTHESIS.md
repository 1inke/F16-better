# 近期实验整体分析与路线更新（2026-10-05—08）

## 核心判断

近期证据不支持“生成没有价值”，也不支持“已经大幅摆脱恒速外推并理解运动”。更准确的结论是：**有小幅可迁移候选互补；固定K20集成会损失少数原生独占支撑，简单小残差和平滑不能同时改善覆盖与运动学，强native-relative射线约束又把互补压回近零。**

当前优先分支是**原生支撑保护下的物理可行互补生成/输出几何**，不是提前改scorer、盲换softwinner，或只增加训练轮数。还没有大突破，baseline不升级。本报告按academic-research-suite证据/反方流程完成，AI辅助且同助手审核，非外部审稿。

## 1. 研究链条：每一轮究竟改变了什么

| 阶段 | 可信结果 | 不能推出的结论 |
|---|---|---|
| loss–完整覆盖/credit/motion诊断 | 旧persistent FIT loss↓19.01–21.56%，变化覆盖40.61→51.50–53.87%；CAL loss↑1.71–4.73%、覆盖34.71→30.86–34.57%，运动学有代价 | 不能说CAL loss也下降；已有good时hardwinner通常合理，不等未覆盖候选分配已最优 |
| 物理反馈三项 | P降低代价；固定路由未解决迁移；端点保持训练投影可控加速度，仍未同时满足所有质量/运动学 | 不能把不同容量/精度横向变化当单因素因果，也不能断言loss唯一错或共享head唯一干扰 |
| grouped→canonical协议统一 | 原scene与target/time实际算法不同，统一P恢复小幅grouped互补与CAL覆盖增益 | 未跑同canonical matched-random；各G样本量/outer/组成不同，不能纯归因grouping或宣布排除相关窗口依赖 |
| 原生20加4，K24 | 三grouping outer变化coverage +2.84/+2.77/+1.64pp；scene变化ADE/FDE收益3.25/5.25%，target1.90/3.87%，time1.49/2.20% | K增加与含原生集合天然lost0，不能叫等预算候选效率提高；没有同时5%双误差+全部motion |
| 固定K20 bottom-logit替换 | overall coverage各G +2.24–3.00pp；learned4独立K4优于removed4 | change ADE恶化8.67–15.52%、FDE9.57–18.82%；79/108全组motion失败，不是安全成功 |
| lost/rescue + LOO utility | 低rank仍有独占/高误差utility；实际lost路径与common16明显不近 | lost⇒removed首先是集合恒等式，不能当独立机制发现；GT utility仅事后审计，不能拿来部署挑slot |
| 支撑保护插入 | 27jobs全部零接纳，protected20 exact native20、motion安全 | 零loss也零rescue，不是集成成功，更不是固定K20不可能性定理 |
| 原生锚定三项 | native基底保留更多变化支撑；Bernstein残差jerk降；physical-ray五motion全通过但近原生 | 所有major FAIL，不是理解运动；三项累计fresh重复控制不当独立泛化 |

每行来源分别见：[loss诊断](C:/Users/linke/Documents/New project/F16_LOSS_QUALITY_CREDIT_MOTION_DIAGNOSIS_20261005/RESULT_ANALYSIS.md)、[物理三项总review](C:/Users/linke/Documents/New project/GENERATION_TRAINED_ACCEL_PROJECTION_SERIES_20261006/recovery_source_20261006/evaluation_recovery_20261006/PHYSICAL_FEEDBACK_STUDY_REVIEW.md)、[canonical归因闭环](C:/Users/linke/Documents/New project/PROTOCOL_RECONCILIATION_RUN_20261007/audit_recovery_20261007/RESULT_ANALYSIS.md)、[等K20](C:/Users/linke/Documents/New project/F16_CANONICAL_EQUAL_K20_REPLACEMENT_DIAGNOSIS_20261007/RESULT_ANALYSIS.md)、[tail/utility](C:/Users/linke/Documents/New project/F16_REMOVED_TAIL_UTILITY_AUDIT_20261007/RESULT_ANALYSIS.md)、[支撑插入](C:/Users/linke/Documents/New project/F16_NATIVE_SUPPORT_PROTECTED_INSERTION_DIAGNOSIS_20261008/RESULT_ANALYSIS.md)、[本组三项全review](C:/Users/linke/Documents/New project/GENERATION_NATIVE_ANCHORED_PHYSICAL_RAY_SERIES_20261008/roundoff_certified_recovery_20261008/approved_launch_20261008/NATIVE_ANCHOR_STUDY_REVIEW.md)。本整体报告引用已完成审核与不变来源，不重新训练、不声称重放所有历史模型/梯度。

## 2. CAL失败：当前证据改变了原来的判断

早期persistent架构FIT成功有明显物理代价，CAL final全部低于native，但constrained小增是反例。后续grouped训练中，旧FIT/CAL原划分是ordered而非random；target-disjoint不等context-disjoint，时间以source index核验而非随意秒/视频frame。不同算法P跟着grouping变化，导致跨研究归因不可辨识，Protocol Reconciliation确实是必要修复。

统一后的canonical full24在原CAL变化700窗口native243基础上：scene254/261/258、target257/260/261、time256/257/262，三个组都小增。所以不能再总结“CAL总是不能迁移”。这表明算法/表示/组合可以影响迁移结果，但不是证明已排除distribution shift、共享干扰或相关窗口依赖。它也没有隔离winnerloss/diversity/clip各项的因果贡献。

canonical三G采用相同算法，但train16337/18909/16104和outer1561/1035/566不同；相同P只消除了算法混杂，没有composition-controlled pureG。旧matched-random实验也不能与新的canonical结果强配对。缺少新的同P、同outer、同训练预算/组成的random–grouped控制，不能给场景/行人/时间难度做因果排序。只3scene/seed、相关窗口、F16已见旧pool、重复单university CAL均是限制。

## 3. 被删尾部承担了什么不可替代作用

tail审计258个lost行事件对应51个unique sample ID，变化组32个；CAL九head反复丢的是同样7个ID，不是63个独立行人。每个实际lost原生只有一条合格候选，Q与J winner均在removed4，coverage LOO utility=1。

归一化最近common距离D=min max(meanpath/.10,endpoint/.20)，所有lost最小2.4234，D≤1为0。CAL七例原生近GT ADE/FDE均值.150/.229m，learned J最近候选却约1.096–1.215/2.253–2.562m。这是少数难例的支撑缺失，不是只差一点跨门槛。

“压掉低速停留，只救高速”的推测没有得到普遍支持：lost全程低速0，末段停留却有11/51；CAL lost GT p95速度.881m/s而rescue .575–.708，scene/target方向相同，time不同。方向/弧长/停留/转角标签可重叠，不是排他根因分类。

速度尾部更可信的证据是集合组成：移除较慢removed4、加入较快learned4会改变ALL-K分布；common16本身可能比learned4更快。旧logit rank≠集合冗余rank，但logit并非完全无信息；个体LOO也不能保证联合删除安全。所有GT LOO utility只用于事后理解，不能以测试未来选择删除方案。[原审计](C:/Users/linke/Documents/New project/F16_REMOVED_TAIL_UTILITY_AUDIT_20261007/RESULT_ANALYSIS.md)

## 4. 为什么最近原生锚定路线没有取得突破

第一native-anchor变化outer110→111每seed，只净增1/253；native全部36组accelmean高2.61–3.05%，CV则多数speedp95失败。0.2m位置界不代表物理界，而且输出基底决定初始支撑与reachable set。

第二degree11 Bernstein使残差jerk降86.6–89.9%，却不改变任何paired二元coverage/rescue-lost集合；完整路径加速度仍高2.66–3.07%。所以“去高频就解决运动学”不足，持久修正、增长包络及native+residual几何仍有关，但未唯一归因。

第三训练内physical-ray将36组五motion全部守住，outer/inner/CAL变化严格覆盖仍110/166/243、救0/损0。53.76–56.82%候选alpha=0，median0，最大实际位置修正约0.04184mm。它相当于把有副作用的微弱修正压回近原生，不是学出新的安全变化。

第三31逐步native-relative条件**强于**原五pooled门槛。零/微小native加速度jerk预算可以令整条方向近退化。不能由此推断“所有可行候选都无法修”，更不能为了得到阳性结果改margin/epsilon/评价门槛。浮点推导certificate只负责工程可信度；研究ALL5 motion literal1%原样。

## 5. 按人类最初决策地图，现在位于哪个分支

- 不处于“质量目标对齐、运动学无害、只剩CAL泛化”的终端分支。早期FIT有motion代价，当前近identity无有效增益；不能先给loss或结构任何一方免责。
- 不处于“hardwinner普遍给错已有好候选，因此必须softmin”的明确分支。旧诊断good-present winnerbad比例很低，旧Gaussian witness零坐标credit不证明当前参数化可学/共享无反馈；softset/energy/mixture失败反证保留。
- 更接近“候选互补存在，但集合集成和修正可行几何未同时保护原生支撑、质量及运动学”的分支。它支持优先研究输出结构/可行方向，而不唯一证明共享head干扰或decoder根因。
- 还未进入scorer：生成端未稳定提高samecandidate完整覆盖并通过motion，scorer无法补出不存在的路径。旧nativeargmax/logits不是refined路径概率。
- 针对“恒速外推”的原问题，change标签并不等于已核验的CV-failure子集。minADE和minFDE可来自不同路径；当前证据只有有限互补，不够声称大幅摆脱CV偏好、更不是行人意图理解。

## 6. 下一大方向与禁止重复

优先问题：**在原K20、原生支撑和原五pooled运动学门槛下，能否构造非退化的可行互补输出方向？**

下一项应是单因素、结构级参数化比较，而不是继续扫alpha/margin/Bernstein degree/残差bound/LR/epoch/width/删除slot。保持当前合法newhead/现有FIT数据/原quality assignment、冻结旧F16/decoder/scorer，不用GT或CAL选择部署模式，不改评价门槛。预注册前应明确其非零可行跨度与失败条件：如果仍只有原生identity，判方法不可用，不无限增加诊断；也不能把更强native-relative包络与原门槛混为一谈。

真正启动前仍须source–ID–statistics–objective完整绑定、actualGPU梯度/paired identity/cachefinite、真实2epoch所有optimizer与RNG exactresume、资源/吞吐实测、唯一OS任务、唯一半程→实际完成节点。并按每3newformal强制review。没有自动增加新qualityloss/assignment、scorer、旧解冻、事件监督、test/fold/外部数据、云付费/messages/delete权限。

本次六小时窗口已结束；最新“开始”完成了单第三+全部复盘，无第四训练启动。不把时间截止当goal达成，也不重置旧03:30:18–09:30:18窗口。长期原目标仍未完成。

## 7. ARS反方2：避免积极叙事过头

最强反驳：“所谓新信息只是既有输入函数的重新组合，oracle小增可能主要来自增加K与样本复用，而不是未见运动理解。”结论因此用“有限互补信号”，不称新观察信息/理解。equalK总体coverage正是有价值反例，但变化误差与motion负结果同样保留，不能择取一个分母。tail的代数必然与真正附加几何证据分开。

## 8. ARS反方3：避免消极归因过头

最强反驳：“保护策略和ray比原门槛强得多，零接纳/近identity可能是研究者自身约束而非生成器无用。”承认且不否定generation。已有结果不足以唯一选loss、结构、distribution shift或shared干扰；新方向必须可辨识、非重复局部扫参。全部重大门槛、场景/window相关性、3seed充分性、不同协议/预算/群体限制保留。本复盘PASS限域，不等外部审核、baseline升级或SOTA。

## 9. 工程责任与可追溯性

之前把3.86e-17的浮点超额直接升级为待人类决定，是助手错误；当前已自主分解消减/舍入尺度、保留失败来源并复核，原science未放宽。大FD步长face切换、小步长收敛证据完整，不隐藏不利数字。当前18job独立saved质量/完整聚簇/五motion/36CI误差0，1193相关锁源未变，生产private2.792GiB。上述可信度不允许扩大成“全部历史模型重新复现”。所有父失败与报告修正保留，实验来源应能追到checkpoint和实际协议，而非只看到参数文件。

