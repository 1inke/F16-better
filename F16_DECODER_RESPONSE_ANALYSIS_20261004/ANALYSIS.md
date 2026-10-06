# F16：变化风险到正确轨迹响应的解码链分析

日期：2026-10-04。范围：依据既有结果和静态源码定位假设，不训练、不重放模型、不查询旧worker、不更改既有科学源码/模型/结果。本文件是新分析，不是新实验结果。使用 academic-research-suite 内联证据综合与反证检查；不是外部独立审稿。主要事实及反例先登记在 claim_manifest.json。

## 1. 判断：应优先检验解码链，但尚未证明“模型知道准确未来、只是没输出”

用户的观察有明确研究价值：一些候选会转弯或减速，却没有在正确时刻、方向、幅度上响应；终点接近不代表运动过程正确。现有证据支持把解码器列为优先研究对象，特别是时间过程与候选条件化生成。但“内部存在可被额外reader读出的变化风险”和“原模型准确捕获并有效利用具体变化”不能等同。

依据既有完成复核：decoder context64 的场景族留出线性风险 AP=.6820，未出现风险可读性坍塌。真实事件子集上的类型/粗时刻/左右转准确率分别57.30%/49.50%/61.55%，对应多数类基线49.48%/41.65%/52.23%。这些不是部署时检测加分类的联合准确率，也未测准连续变化幅度和完整路径。CAL AP E1/E2/E3=.7438/.7327/.7274，与留出reader逐阶段上升不一致，必须保留。<!-- cite: observable-interpretation#sections-2-3; claim:C01 -->

原1516DEV中的469个明确变化样例，按同一候选 ADE<=.20m AND FDE<=.40m：377无达标候选、87有但原argmax没选中、5最终达标，即80.38%/18.55%/1.07%。因此针对这一困难组，只换排序不能补救缺失候选；只改善生成也不能保证最终选对。387/469=82.52%有粗类型和时刻±.8s匹配的候选，却不能称变化预测正确：没有验证方向、幅度、完整形状，粗多事件匹配也有限制。<!-- cite: observable-interpretation#section-4; claims:C02,C03 -->

最稳妥的主结论：**风险线索、具体响应参数、完整候选路径、最终选择之间存在落差。当前证据不能唯一定位其因果来源，但足以优先检验解码时间表达与条件化生成。** 不把所有失败解释成完全不可预测，也不把所有失败解释成decoder故障。

## 2. 当前真实解码结构及可疑环节

当前诊断加载路径：run_formal_regime_selection.build_baseline -> JointTrainableV2Decoder -> FrozenControlledAnchorDecoder；诊断时整个加载的原模型冻结。类名中的Frozen或历史joint训练政策不能替代当前实际冻结审计。

| 环节 | 已核实结构 | 与失败现象相容的解释 | 尚不能下的结论 |
|---|---|---|---|
| context | 最后token96、历史mean token96、7维运动特征，经MLP得到context64；tokens本身已含时序/social信息 | 全局摘要可能不足以支持不同未来时刻读取不同细节；二分类风险保留不保证时间/方向细节保留 | 不能说历史被完全丢失或64维必然不足；E3风险AP没有坍塌 |
| mode/proposal | 20个学习式mode embedding、全局可学习三锚点模板；context给共用二维scale，mode给有界delta | 容易用已有运动原型表达“类似转弯/减速”，但缺少可解释的候选特异时刻/转向幅度约束 | 模式不是人工固定20种类别；模板可学习、delta可随观察变，不是只查表或显式CV外推 |
| interpolation | 锚点在step4/8/12，即1.6/3.2/4.8s；固定线性插值 | 各段内位移增量恒定，不能独立安排段内转向/加减速；相同端点不确定中间过程，事件时机可能被提前摊开或延后 | 不是所有变化都不能表示，也未证明这是当前误差的主要因果来源 |
| V2 refiner | 输入mode+source anchors，输出三个锚点的有界二维offset；当前max_offsets=.20/.25/.50m | 仍只能移动三锚点，无法改变分段内的独立时间过程；它不是完整12步轨迹细化器 | 不据此宣布放大offset上限可获益，也不混入后续残差实验的另一种velocity clip |
| score_head | 直接输入mode和修正前source anchors，之后才算refiner和最终轨迹 | 没有直接评估最终轨迹的速度、转向过程、观察一致性，存在质量与排序对齐问题的动机 | native评分与生成曾联合训练，仍可间接学习修正效果；不能仅凭顺序判定score“错误/过期” |

代码锚点：[原三锚点及插值](<C:/Users/linke/Documents/New project/HIERARCHICAL_DECODER_SERIES/hierarchical_decoder_models.py:41>)、[context和mode模板](<C:/Users/linke/Documents/New project/HIERARCHICAL_DECODER_SERIES/hierarchical_decoder_models.py:135>)、[当前V2完整forward](<C:/Users/linke/Documents/New project/CONTROLLED_ANCHOR_CORRECTION_SERIES/controlled_anchor_correction_models.py:79>)、[V2边界覆盖](<C:/Users/linke/Documents/New project/E2_F16_END_TO_END_COMPARISON_SERIES/joint_v2_adapter.py:50>)、[当前baseline加载](<C:/Users/linke/Documents/New project/REGIME_ADAPTIVE_SELECTION_FORMAL_SERIES/run_formal_regime_selection.py:134>)。<!-- cite: h3-decoder#lines-41-54-135-159-197-221; v2-decoder#lines-63-123; claims:C04,C05,C06 -->

一个直观例子：目标从预测起点起先走.8s再停。如果第一段必须连接起点与1.6s锚点，那么它可能把.8s的运动平均摊到1.6s；终点可对，但停止时机和中间位置错。该例说明表达限制，不是声称已量化这种错误占比。类似地，向左转与向右绕行可有近似终点，单独终点监督不能唯一决定中段几何。

## 3. 哪些模块值得改，优先级是什么

### 第一优先：时间过程生成/细化，而不是再次扩大锚点MLP

目标是把“到哪里”与“何时、以怎样的速度经过”分开表达。优先研究替换 interpolation + 三锚点-only refiner：保留终点/中间锚点作为几何约束或结构先验，增加候选特异的速度/转向随时间演化，让停止/转弯的时机不只由固定两个中间结点决定。

可考虑的实现假设是轻量的未来time-query状态生成器：每个候选、每个未来时刻读取冻结observed tokens和自身mode/state，输出位移增量或速度变化，再积分成轨迹。也可以研究可学习变化相位/时长的分段动力学。两者是待比较的参数化，不是已验证方案；不要求使用真实未来新观测，也不要求一定自回归。端点约束必须避免积分漂移，速度/加速度/jerk与原始观察的连续性必须实际检验，不能用“积分/平滑”直接宣称物理正确。

关键区别不是“6个数换成24个数”，而是让**观察线索按未来时刻、按候选状态参与生成正确响应**。初次对照应固定F16/K20/数据/训练损失和保护规则，以免同时更改生成、输入和监督而无法归因。扩大当前旧refiner、Top-M或输出维数不是默认有效方向。<!-- cite: decoder-claims#C01-C04-C09; claim:R01 -->

### 第二优先：候选模式到具体响应的参数化

检查 mode_mlp / proposal_delta / anchor_templates 的连接方式。模式向量目前是隐式运动原型，缺少直接约束“哪一个候选对应何种变化时刻、方向和强度”。可在新解码器中把候选latent组织成不同的变化响应假设，或让其控制未来状态演化；不要仅给所有候选加一个统一的二分类风险gate。

不把风险高的所有候选同时拉成一条GT：观察存在歧义时仍需合理多模态覆盖。事件标签可作为未来研究的训练辅助目标，但只允许训练/事后评估使用，推理时必须从观察预测；这是另一个监督干预，不能与第一个结构对照一起默认加入。既有context/mode dense、rank8、tangent更新没有取得major，说明“把conditioning再训练得更自由”不是现成答案。

### 第三优先：context读出方式，先有针对性地改，不默认重训整个F16

E3二分类风险可读性保留，因此不建议先扩F16或仅扩大context维数。可让未来time queries保留对observed token序列的访问，并明确传递末端速度、近期加速度等运动状态；检验其是否补充具体时机/方向的信息，而非只再提高风险AP。tokens已social-enriched，新增读出不是新数据或首次social建模。原token attention/ordered readout失败也是反例，未来“时间索引生成响应”需区别于那些只给共享锚点头补一个固定向量的试验。

### 第四优先：最终轨迹质量评分，和生成分开做

让score_head能够直接评估修正后完整路径及其与观测的一致性，而不是只看到修正前anchors。训练目标应与同一路径质量和实际Top1对齐，不能只看终点或把独立minADE/minFDE当一条轨迹。87个有好候选却没选中的清晰事件样例是这一方向的直接描述性依据；377个缺少好候选不能靠固定集合重排解决。

这个优先级针对困难运动变化组，不意味着总体scorer不重要：当前全1516的候选覆盖/筛选落差也很大。新增scorer训练超出这次分析，执行前需单独明确范围；本轮未改它。

### 并行核查项：物理邻居几何的输入路径

native.forward先分别中心化各agent历史，再给F16 set_context；relation_features的位置/distance/TTC/DCA来自这份中心化历史，不能直接当实际物理行人间距。原未分别中心化local同时经graph_loc进入图/结构路径，所以不等于整个F16缺物理几何。该路线可能影响具体避让方向/时机，应分支定位核查，但当前没有证明它是bug或解码失败主因。本轮不修父源码。<!-- cite: observable-interpretation#section-6; claim:C10 -->

## 4. 必须保留的反证：为什么不能说“换decoder必然解决”

1. **三锚点不是普遍不可达上限。** 旧FIT/CAL GT辅助可达性诊断在formal-change CAL700上，把native minADE/FDE .245932/.431965降到三锚点残差构造的.052755/.040895。它是native基底+残差构造，不是当前V2原offset范围内的最优拟合，也不是独立DEV469事件组；使用GT且不是learned/deployable/最优oracle。它说明很多样例若响应参数正确，三锚点残差已能显著改善，不能断言所有问题来自锚点数量。<!-- cite: reachability#cal-change-and-limits; claim:C07 -->
2. **学习与泛化的落差明确存在。** 固定FIT256诊断总体ADE/FDE改善15.32%/21.23%，CAL256总体伤害2.04%/1.29%；原正则/保护目标仍在且没有宣称完全记忆或收敛。Dense conditioning诊断FIT2048改善15.99%/24.77%，CAL2879伤害3.43%/5.04%。这些不是无法学习的证明，也不是单纯更多容量能解决的证据。<!-- cite: tiny-fit-learning#table; conditioning-learning#fit-cal; claim:C08 -->
3. **更灵活的轨迹输出已有失败历史。** EXP139D的Top8时间query细化没有净收益，粗oracle进入Top8仅69.2%General/53.6%Challenge；该实现还锁定终点。旧全速度残差的凸Bernstein混合也未优于对照，可能抑制急变化。新方向不得把“加time query/平滑/输出12步”当作全新、已充分论证的解决方案，应显式区分全K处理、真实观察条件化、事件时刻表达和不提前剪掉候选。不同旧协议不能当当前问题的因果否定。<!-- cite: h3-analysis#section-3-2; temporal-basis-report#verdict-and-metrics; claim:C09 -->

因此我支持优先改解码链的研究方向，但更精确的待检验命题是：**在保留已有观测表示和合理候选多样性的前提下，是否能更可靠地预测具体变化响应，并将它变成整条正确候选。** 这同时涉及响应参数预测、时间几何表达与泛化，不能只归因于一个MLP尺寸。

## 5. 如何判定下一结构是否真的解决了问题

这是未来执行标准建议，不是启动授权或已完成新诊断：

- 固定K20，在同一候选上同时看ADE/FDE，另报早/中/晚误差、转向方向、开始/停止时刻、速度变化幅度和曲线；端点近但中段错的样例需单列。
- 区分“更好的锚点”与“给定锚点时更好的过程”。可用FIT/CAL-only离线GT锚点替换作表达能力检查，明确GT不进部署，不能拿GT辅助结果当learned性能；它与当前残差可达性诊断不是同一个实验。
- 同等观察输入下比较候选生成结构，既看FIT也看CAL，所有checkpoint按原calibration规则固定，不按DEV选择；控制有效容量和训练计算差异并如实披露。
- 原regularizers/protection/calguards与数据范围先不改；更强时序/事件监督若需测试，应是另一个预注册单因素，不接着扫温度或系数。
- 改生成与改scorer分步评价，避免候选固定slot继承旧logits而把几何改善误报成筛选进步。
- 不以全部候选塌缩到实现GT、只增加K或加入起点之后真实观测作为解决方法；固定起点自回归同样不能获得真实未来新信息。
- 既往major保护门槛仍保留，当前诊断阈值也不借机放宽。反复ETHDEV/仅三场景族及原F16训练暴露限制独立泛化、因果和SOTA结论。

## 6. 本轮交付与审计边界

所有新增分析仅在本新文件夹；`.memory`只记录最新诊断判断和后续讨论范围。14份本轮直接引用的旧输入文件SHA在写报告前后比较，结果另存source_integrity.json；这不是重新核验全部18,208旧文件，也不是重新重算所有reader指标。数值沿用已完成的saved_result_review与既有报告。本轮未更改模型/科学源码/结果、任务状态、自动化或formal计数，不宣布major或baseline更新，等待人类决定具体实现方向。
