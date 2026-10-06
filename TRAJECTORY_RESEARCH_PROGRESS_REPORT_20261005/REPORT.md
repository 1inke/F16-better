# 行人轨迹预测研究进展报告

## 从首次候选可视化到最新候选状态实验

**整理日期：2026年10月5日；所有时间采用香港时间。**

本报告以2026年10月3日这轮“F16候选生成—评分—选择”可视化为起点，整理其后的人工发现、数据分类诊断、正式模型实验和方法演变。它不是整个项目自建立以来的实验总账。

最新已完成并分析的正式实验为 **Observed Discrepancy（相对观测运动偏差）**，于10月5日10:37:04结束。下一项 **Discrepancy Normalization（偏差表示标准化）** 在本次读取的记录中仍为设计待实施，未启动、未产生结果。后文严格区分“已完成”“诊断”和“仅设计”。[项目记录，2026：最新结果](<C:/Users/linke/Documents/New project/GENERATION_OBSERVED_DISCREPANCY_SERIES_20261005/RESULT_ANALYSIS.md>) <!--ref:S14--><!--anchor:section:对照与可靠性--> [保存状态](<C:/Users/linke/Documents/New project/GENERATION_DISCREPANCY_NORMALIZATION_SERIES_20261005/runtime_state.json>) <!--ref:S15--><!--anchor:section:status-->

## 一、先看结论

这一阶段最大的进展，是把一个笼统问题拆成了可以验证的几个环节：**过去有没有可检测线索、候选集合有没有正确响应、完整路径能不能表达出来、训练是否能学出来、是否迁移到其他样例，以及最终能不能选中。**

目前已经确认了以下事实：

- 原F16有明显的候选覆盖和选择缺口，不能仅看minADE/minFDE判断实际预测效果。
- 冻结特征中可以读出部分运动变化风险，但“知道可能变化”不等于“知道正确方向、时刻和完整路径”。
- 对一些重点失败样例，当前曲线族能够在知道GT的条件下构造好路径；因此不能把失败全部解释成曲线形式画不出来。
- 新响应头能学好一部分训练样例，但校准集并没有稳定获得净改善。
- 显式读取候选自身运动状态，增加了一些完整好候选的覆盖，是值得保留的探索性结果；副作用、种子不一致和不确定性尚未解决。
- 最新相对观测速度表示没有延续这个覆盖收益。到目前为止，这一阶段没有正式实验通过全部重大改进条件，也没有替换F16基线。

这些结论分别由后文的可视化、诊断和五项正式对照支持，不是由某张好看的图或某个平均值推导。全阶段证据的主要转折见[三项时序复盘](<C:/Users/linke/Documents/New project/GENERATION_PERSISTENT_RESPONSE_SERIES_20261005/RESPONSE_STUDY_REVIEW.md>) <!--ref:S12--><!--anchor:section:决策摘要--> 和[候选状态结果](<C:/Users/linke/Documents/New project/GENERATION_CANDIDATE_STATE_SERIES_20261005/RESULT_ANALYSIS.md>) <!--ref:S13--><!--anchor:section:同一高质量候选与去重：比独立min更重要-->。

## 二、研究口径：我们到底在改善什么

### 2.1 数据与几个容易混淆的词

原可视化基于既有ETH开发分析窗口，共1516个显式ID，每个窗口包含8步观察、12步预测，采样间隔0.4秒，未来预测跨度4.8秒，每次保留20条候选（K20）。这里的窗口不等于独立行人，同一轨迹可能贡献重叠窗口。它不是官方held-out测试结果。[项目记录，2026：可视化身份](<C:/Users/linke/Documents/New project/TRAJECTORY_SELECTION_VISUAL_REVIEW_20261003/README.md>) <!--ref:S02--><!--anchor:section:候选生成—模型筛选人工复核-->

| 名称 | 本报告中的含义 | 不能混同的内容 |
|---|---|---|
| GT | 已发生的真实未来轨迹，作训练目标或事后评估 | 推理时可获得的新观测 |
| FIT | 训练新修正头的25912个窗口 | 独立泛化评测 |
| CAL | 原协议的2879个校准窗口，用于约束选模 | 从未参与选择的独立测试 |
| DEV | 反复分析的1516个开发窗口 | 独立ETH/UCY整体成绩 |
| formal change | 原正式误差统计中的223个变化窗口 | 新持续事件规则的469个clear-event窗口 |
| oracle候选 | 用GT事后挑选的候选 | 模型推理时真正选择的轨迹 |
| constrained | 按原CAL约束选择的主checkpoint策略 | 为了DEV数字好看而事后选择final轮 |

另外，头学习诊断使用的是富集后的FIT256/CAL393，不是完整FIT/CAL，也不能与1516个DEV窗口混用分母。分组和数量来源见[最新正式结果](<C:/Users/linke/Documents/New project/GENERATION_OBSERVED_DISCREPANCY_SERIES_20261005/RESULT_ANALYSIS.md>) <!--ref:S14--><!--anchor:section:正式 constrained 结果-->、[头学习诊断](<C:/Users/linke/Documents/New project/F16_HEAD_LEARNING_DIAGNOSIS_20261005/RESULT_ANALYSIS.md>) <!--ref:S09--><!--anchor:section:这次具体比较什么-->。

### 2.2 从独立最小误差到完整候选质量

ADE是整段平均位置误差，FDE是终点位置误差。minADE和minFDE可以由不同候选取得：一条中段贴近GT，另一条只有终点贴近GT。两个最小值都好，不代表集合里存在一条完整好路径。

本阶段固定的严格位置质量标准是：**同一条完整候选同时ADE≤0.2米、FDE≤0.4米。** 这只是位置覆盖条件，还不代表满足全部运动学保护，更不是通用SOTA门槛。首次网页默认的0.3/0.5米是较宽的浏览口径，之后保留为敏感性分析，不能与0.2/0.4混用。[项目记录，2026：阈值敏感性](<C:/Users/linke/Documents/New project/CANDIDATE_SELECTION_VISUAL_AUDIT_20261003/REPORT.md>) <!--ref:S01--><!--anchor:section:阈值敏感性（原生整体）-->

人类明确后的研究优先级是：

1. **覆盖优先：** 尽量让原来没有好候选的窗口，至少出现一条完整好候选，并同时统计救回与损失。
2. **有效数量其次：** 在不依赖GT的几何去重后，考察真正不同的好候选，而不是让20条都挤到一起。
3. **选择最后：** 在生成质量稳定后，再单独研究如何选中好轨迹。

主去重先对全部K20做几何complete-link聚类，使用平均路径距离0.10米、终点距离0.20米；再统计含至少一条达标候选的簇。它是几何有效数量，不是心理意图种类，也不是越多越好。[项目记录，2026：去重与覆盖](<C:/Users/linke/Documents/New project/F16_CUE_RESPONSE_CONCORDANCE_20261004/recovery_memory_20261004/RESULT_ANALYSIS.md>) <!--ref:S05--><!--anchor:section:4. 去重后的有效好候选：先补“从0到1”-->

## 三、首次可视化带来的发现

### 3.1 工具做了什么

这轮先建立全量候选分类审计，再提供交互式人工复核。网页显示观察历史、GT、全部20条候选、实际原生选择、minADE/minFDE和联合最优参照，并联动ADE–FDE散点、候选表格和时间位置。人工可以按ID浏览、点击轨迹、修改诊断阈值和导出备注。不是只挑代表性好图，也没有训练或改变旧模型。[工具说明，2026](<C:/Users/linke/Documents/New project/TRAJECTORY_SELECTION_VISUAL_REVIEW_20261003/README.md>) <!--ref:S02--><!--anchor:section:候选生成—模型筛选人工复核-->

原F16全体1516窗口：平均minADE/minFDE为0.219154/0.399725米，实际选中轨迹的平均ADE/FDE为0.542432/1.197316米。严格0.2/0.4口径下的六类互斥统计如下。

| 类别 | 数量 | 对研究的意义 |
|---|---:|---|
| 选中合格完整路径 | 281 | 生成和选择在该窗口都达到位置标准 |
| 有合格路径，但没有选中 | 452 | 给定集合存在选择改进空间 |
| 两个min各自达标，但没有同路径达标 | 56 | 独立最小指标造成的质量错觉 |
| 仅minADE达标 | 77 | 全程平均较近，终点仍有问题 |
| 仅minFDE达标 | 191 | 终点较近，中间路径仍有问题 |
| 两个min均未达标 | 459 | 对这次GT的候选覆盖不足 |

因此，有完整好候选的窗口为733/1516，约48.35%；其中452个没有被原评分器选中。其余783个窗口没有完整达标候选，评分器不能凭排序补出不存在的路径。上述计数只说明给定GT和阈值的事后质量，不直接判定未来可预测性。[项目记录，2026：全量可视化统计](<C:/Users/linke/Documents/New project/CANDIDATE_SELECTION_VISUAL_AUDIT_20261003/REPORT.md>) <!--ref:S01--><!--anchor:section:阈值敏感性（原生整体）-->

### 3.2 人类提出的关键假设

人工观察把问题从“平均误差多少”推进到了轨迹形态：有的路径前面近、终点远；有的终点巧合接近、中间运动状态错误；两个指标都好也可能主要出现在近静止或运动平稳的简单案例。人类同时关注到候选之间较相似、原选择偏顺延，以及变化发生后是否存在更正确的运动响应。

这些观察被当作后续检验的假设，而没有直接转成“解码器一定有问题”“所有变化都可预知”或“平均值低就是理解了意图”。随后的风险诊断显示，469个明确持续变化窗口仅92个有完整好候选，而196个近静止窗口的位置覆盖率约98.47%。确实需要把简单状态与变化状态分开报告。[项目记录，2026：候选与原始筛选](<C:/Users/linke/Documents/New project/F16_OBSERVABLE_CUE_DIAGNOSIS_20261003/RESULT_ANALYSIS.md>) <!--ref:S03--><!--anchor:section:候选与原始筛选-->

## 四、分类诊断：逐步收缩问题范围

下列是诊断链，不计作正式模型对照。有的只读保存数据，有的训练独立分析reader，有的使用GT做表达构造。它们回答的是不同问题。

### 4.1 可观测线索诊断：输入中有变化风险信息吗？（10月3日）

冻结F16、原解码器和评分器，训练独立风险读出探针；比较原始历史、时序特征、社交整合特征和解码上下文。FIT按三个scene-family整组留出。在选定线性reader结果中，时序特征AUPRC约0.618，社交整合特征约0.670，解码上下文约0.682。

这提供“部分变化风险可读出”的证据，却不证明原模型有益利用了这些线索，也不证明正确方向、发生时刻、幅度和完整路径都能被推断。弱读出也不能证明信息不存在。三个场景族、原F16训练历史和邻居未来存在性资格偏差限制了泛化解释。[诊断结果，2026](<C:/Users/linke/Documents/New project/F16_OBSERVABLE_CUE_DIAGNOSIS_20261003/RESULT_ANALYSIS.md>) <!--ref:S03--><!--anchor:section:FIT场景整组留出的变化风险读出-->

### 4.2 低线索失败分层：多少变化暂时没有检测到预兆？（10月4日）

针对人类“不值得把主要资源投入完全无预兆变化”的担忧，使用观察期运动规则、邻居接近规则和已保存reader分数分层。

| 469个明确变化窗口 | 数量 | 比例 | 原K20无完整好候选 |
|---|---:|---:|---:|
| 命中观察运动／邻居规则 | 274 | 58.42% | 228 |
| 无规则，但至少一个reader信号高 | 95 | 20.26% | 66 |
| 无规则，reader不确定 | 79 | 16.84% | 62 |
| 无规则，所有reader分数低 | 21 | 4.48% | 21 |

**21例是当前定义下的低检测线索代理，不是已证明不可预测的比例。** 没有显式规则的195例也不能全部判为无信息。规则和reader会误报，结果对阈值与reader容量敏感；21窗口只对应11条可核验目标轨迹，不能算21个独立人或事件。

即使仅研究274个规则线索窗口，仍有228个无好候选、41个有好候选未选中、5个选中好候选。所以不能用“无预兆变化”解释掉全部失败，也没有正式删除这些困难样例。[分层结果，2026](<C:/Users/linke/Documents/New project/F16_LOW_CUE_FAILURE_STRATIFICATION_20261004/RESULT_ANALYSIS.md>) <!--ref:S04--><!--anchor:section:结论先行-->

### 4.3 线索—响应一致性：预兆真的对应未来变化吗？（10月4日）

进一步将观察期目标线索与未来GT事件的类型、方向对应，邻居接近只作为泛风险，不强推转向方向。274个规则线索窗口里，仅64个与GT第一事件的类型和方向相容；4个仅与更晚事件相容，其余包含泛风险、类型方向不相容和不清晰情况。这修正了“全部274都有正确预告”的过强解释。

更重要的是FIT/CAL证据：在第一事件相容组中，998个FIT窗口有791个无完整好候选，其中576个已有同一候选通过粗事件、类型、方向、时刻±2步和强度检查，但路径仍不合格。CAL对应137个相容窗口、106个无好候选，其中77个粗响应通过而完整路径失败，约占生成失败的72.64%。

这把优先问题缩到“粗响应集合已有覆盖，完整运动过程仍不准确”。它仍是GT辅助的事后集合描述，不等模型已经识别意图；±0.8秒容差与粗标签也可能掩盖重要误差。[一致性结果，2026](<C:/Users/linke/Documents/New project/F16_CUE_RESPONSE_CONCORDANCE_20261004/recovery_memory_20261004/RESULT_ANALYSIS.md>) <!--ref:S05--><!--anchor:section:3. FIT/CAL的主要证据：同一候选粗响应相符，完整路径不达标-->

### 4.4 曲线可达性：当前曲线究竟能不能画出好路径？（10月4日）

固定原native K20和修正范围，使用GT离线寻找曲线构造，不训练新模型、不把未来坐标送进预测前向。单脉冲函数族找到完整达标构造的比例为FIT 25673/25912（99.08%）、CAL 2843/2879（98.75%）；重点粗响应对、路径差组为FIT576/576、CAL77/77。

其余窗口分别记录更宽逐步函数有构造、必要几何下界排除当前修正范围，以及未知，不能把局部求解失败直接写成不可达。

**这个99%左右不是预测准确率。** 构造逐窗口知道未来并独立设置参数，实际共享头只能看过去。构造通过位置误差和几何限制，也没有证明通过全部运动学与训练保护。因此它只削弱“主要是曲线完全画不出来”的解释，不保证这些未来能被模型预测。[可达性结果，2026](<C:/Users/linke/Documents/New project/F16_CURVE_REACHABILITY_DIAGNOSIS_20261004/recovery_numeric_20261004/RESULT_ANALYSIS.md>) <!--ref:S06--><!--anchor:section:2. 四类表达证据：不只看成功样例-->

### 4.5 原目标兼容：损失会不会排斥好候选？（10月4日）

在上述576个FIT、77个CAL重点窗口中，只把一条候选替换为已验证的GT辅助完整好路径，其他19条保持原样。所有这些替换都降低了原总损失，说明在所查组里，好结果确实得到目标奖励。

但原质量项采用hard winner：直接质量坐标梯度给当前ADE+0.5FDE最小的候选。CAL77例中64例的可达见证位于另一个槽位，沿该替换方向的初始质量导数为零。这是局部反馈分配现象，不等83%的训练被阻断，因为共享参数还会同时影响多条候选。

因此损失值得继续检查，但没有证据把问题简单归为“损失不喜欢好轨迹”，也不能认为目标已经完全正确。奖励、直接反馈、共享头可学习性与迁移是不同层次。[目标兼容结果，2026](<C:/Users/linke/Documents/New project/F16_OBJECTIVE_RESPONSE_DIAGNOSIS_20261004/RESULT_ANALYSIS.md>) <!--ref:S07--><!--anchor:section:二、出现好结果与得到直接反馈是两件事-->

### 4.6 响应头学习：真实训练能不能学出来？（10月5日凌晨）

这是富集小样本诊断：FIT256、CAL393，两组响应头、三个新种子、每次120诊断epochs，共6次诊断训练；不是第三项正式模型实验。原F16、原decoder/scorer固定，检查实际参数梯度并保持原目标。

| 诊断集合 | 原F16完整好候选覆盖 | 学习clock后，seed7/8/9 |
|---|---:|---:|
| FIT256整体 | 77 | 154 / 148 / 155 |
| FIT重点失败128 | 0 | 58 / 54 / 57 |
| CAL393整体 | 169 | 167 / 161 / 165 |
| CAL重点失败77 | 0 | 11 / 12 / 11 |

响应头及主要attention权重有实际梯度，排除了已查路径上“梯度彻底断开”。FIT独立minADE/minFDE明显改善，CAL却恶化；CAL救回28/32/29，同时损失30/40/33，整体净减。平均加速度也明显上升。

所以这次证明了**部分失败样例确实能被训练救回**，但没有证明改善可稳定迁移，也不是一套可升级的模型。富集小集结果不能与自然总体比例等同。[头学习结果，2026](<C:/Users/linke/Documents/New project/F16_HEAD_LEARNING_DIAGNOSIS_20261005/RESULT_ANALYSIS.md>) <!--ref:S09--><!--anchor:section:首要目标：至少一个完整合格候选-->

## 五、五项正式模型对照及进展

正式训练与上面的诊断交错推进。前三项组成时序响应研究，已完成3/3并保存总复盘；后两项属于候选状态研究，目前完成2/3。

每项正式实验是两个arm、三个配对种子，共6次全新80epoch训练；保存constrained/legacy/final三种策略，共18个policy条目。五项合计30次正式训练运行、90个policy条目，**不是90次独立训练，也不是30次独立泛化验证**。部分新鲜控制运行与前一实验的相同架构、相同种子轨迹逐元素相同，不能重复累计为独立证据。[研究复盘，2026](<C:/Users/linke/Documents/New project/GENERATION_PERSISTENT_RESPONSE_SERIES_20261005/RESPONSE_STUDY_REVIEW.md>) <!--ref:S12--><!--anchor:section:三假设与各自新鲜对照--> [重复控制核对](<C:/Users/linke/Documents/New project/GENERATION_OBSERVED_DISCREPANCY_SERIES_20261005/RESULT_ANALYSIS.md>) <!--ref:S14--><!--anchor:section:对照与可靠性-->

### 5.1 主结果总表

下表均为预先指定constrained策略，change误差为三个种子的平均，单位米；覆盖是同一1516个窗口的三个种子各自计数。原F16正式change误差为0.372175/0.696326，严格全体覆盖733，clear-event469覆盖92。

| 正式实验与完成日期 | 主要受控比较 | change minADE / minFDE | 相对F16收益 | 全体完整好候选覆盖 | clear-event覆盖 | 判定 |
|---|---|---:|---:|---:|---:|---|
| TIME，10月4日 | 4步分段平均重复 vs 12步自由增量 | .371457 / .694880 | .193% / .208% | 734 / 735 / 734 | 90 / 91 / 93 | 未达major |
| PULSE，10月4日 | 固定脉冲位置跨度 vs 学习位置跨度 | .364915 / .686081 | 1.951% / 1.471% | 723 / 728 / 727 | 88 / 88 / 88 | 未达major |
| PERSISTENT，10月5日 | 学习clock的暂态Gaussian vs 持续sigmoid | .367385 / .686951 | 1.287% / 1.346% | 733 / 730 / 735 | 93 / 90 / 93 | 未达major |
| CANDIDATE STATE，10月5日 | 候选状态输入屏蔽 vs 实际读取 | .367063 / .686794 | 1.373% / 1.369% | 744 / 742 / 749 | 99 / 93 / 100 | 未达major |
| OBSERVED DISCREPANCY，10月5日 | 实际原始状态 vs 相对最后观测速度的偏差 | .371580 / .694748 | .160% / .227% | 725 / 730 / 728 | 88 / 92 / 86 | 未达major |

表中误差仍是两个独立min，完整好候选覆盖才是同一条路径的双阈值检验。不同实验架构有变化，跨行大小只能描述，不是受控的因果优势比较。来源：[时序三项复盘](<C:/Users/linke/Documents/New project/GENERATION_PERSISTENT_RESPONSE_SERIES_20261005/RESPONSE_STUDY_REVIEW.md>) <!--ref:S12--><!--anchor:section:同路径高质量覆盖与数量-->、[候选状态结果](<C:/Users/linke/Documents/New project/GENERATION_CANDIDATE_STATE_SERIES_20261005/RESULT_ANALYSIS.md>) <!--ref:S13--><!--anchor:section:主要均值和不确定性-->、[观测偏差结果](<C:/Users/linke/Documents/New project/GENERATION_OBSERVED_DISCREPANCY_SERIES_20261005/RESULT_ANALYSIS.md>) <!--ref:S14--><!--anchor:section:同完整候选覆盖：真正阶段优先项-->。

### 5.2 每项实验回答了什么

**TIME：单纯提高段内自由度没有带来稳定增益。** 治疗比新鲜分段对照误差更大，native与fresh对照的ADE/FDE/joint区间均跨0；完整覆盖净变化很小。平均加速度总体约增加3.90%、change约5.31%，违反原1%保护。它削弱了“只要锚点之间更自由就会明显改善”的充分性设想，不否定所有时序模型。[TIME结果，2026](<C:/Users/linke/Documents/New project/GENERATION_TIME_RESPONSE_SERIES_20261004/RESULT_ANALYSIS.md>) <!--ref:S08--><!--anchor:section:结论-->

**PULSE：学习响应发生时间，参数确实变化了，但没有救回更多窗口。** 相对原F16均值改善约1.95%/1.47%，相对固定时间新鲜对照的增量却很小且CI跨0，只有一个种子双指标更好。clear-event每个种子都从92减到88。加速度副作用明显，整体均值约增加14.60%、change约23.79%。更多动态变化不等更正确的变化。[PULSE结果，2026](<C:/Users/linke/Documents/New project/GENERATION_PULSE_RESPONSE_SERIES_20261004/RESULT_ANALYSIS.md>) <!--ref:S11--><!--anchor:section:副作用与学习/泛化-->

**PERSISTENT：持续响应降低了暂态脉冲的副作用，但仍没实现覆盖突破。** 该结构让响应可以持续影响后段和终点。全seed/group平均加速度增幅降到约1.17%–5.27%，明显低于Gaussian对照的11.45%–26.53%，但仍未通过严格1%。完整覆盖回到接近F16，误差不稳定优于fresh Gaussian。这个折中值得记录，却不是运动理解已明显提升。[三项复盘，2026](<C:/Users/linke/Documents/New project/GENERATION_PERSISTENT_RESPONSE_SERIES_20261005/RESPONSE_STUDY_REVIEW.md>) <!--ref:S12--><!--anchor:section:运动质量、校准和过拟合-->

**CANDIDATE STATE：目前最明确的覆盖增量来自直接读取候选自身状态。** 相对F16救回30/22/41个窗口，同时损失19/13/25，净增11/9/16。clear-event净增7/1/8。但所有主要native/fresh误差CI跨0，第三个种子FDE恶化，平均加速度增幅5.23%–10.66%。不能把局部覆盖提高升级为稳定新模型。[候选状态结果，2026](<C:/Users/linke/Documents/New project/GENERATION_CANDIDATE_STATE_SERIES_20261005/RESULT_ANALYSIS.md>) <!--ref:S13--><!--anchor:section:运动学、时序和学习迁移-->

**OBSERVED DISCREPANCY：更直观的相对运动表示并没有自动更好。** 治疗相对F16仅救回2/3/3、损失10/6/8；相对raw状态控制覆盖净减19/12/21。所有注册pooled运动学1%保护通过，但CAL选中的是epoch5/6/7的小修正，而raw为29/25/35。未触发原生fallback，却非常接近原生输出。安全了，并不代表正确响应学得更好。final80的change误差反而较原F16恶化约6.84%/10.95%，不能用延长训练当默认补救。[最新结果，2026](<C:/Users/linke/Documents/New project/GENERATION_OBSERVED_DISCREPANCY_SERIES_20261005/RESULT_ANALYSIS.md>) <!--ref:S14--><!--anchor:section:为什么运动学通过，却没有解决问题-->

### 5.3 为什么仍没有major

这一阶段没有放宽原重大改进标准：正式change组ADE/FDE各至少5%收益，overall/stable/transition误差不超过1%伤害，三种子一致，native和新鲜对照的ADE/FDE及同路径joint CI满足注册条件，同路径、覆盖、重复和硬运动检查通过；每个seed/group的pooled K20速度p95、加速度mean/p95、jerk mean/p95均不超过native 1%伤害，dt固定0.4秒。出现首次全项信号后，还需额外检查场景/窗口相关性和仅三个种子的充分性。

这是本项目的研究门槛，不是外部公认的“高质量候选”或SOTA标准。误差均值好、某个CI不跨0、运动学某几项通过，都不能抵消其他条件失败。[注册门槛，2026](<C:/Users/linke/Documents/New project/GENERATION_PERSISTENT_RESPONSE_SERIES_20261005/DESIGN.md>) <!--ref:S16--><!--anchor:section:Gates and audits-->

## 六、人工观察与数据核对：哪些判断得到支持

可视化工具随后扩展到PULSE的改善/恶化比较，以及头学习诊断的FIT/CAL逐例复核。头学习复核保留全部649个窗口、三个种子、20条轨迹、rescue/lost分类、ADE–FDE散点及误差/速度/加速度曲线，不把训练案例当泛化案例。

人类的38条批注全部对应CAL seed7，包含23个rescue、15个lost，是主动挑选的富集案例，不是随机总体抽样。核对后的结论如下。[人工反馈分析，2026](<C:/Users/linke/Documents/New project/HEAD_LEARNING_HUMAN_FEEDBACK_ANALYSIS_20261005/RESULT_ANALYSIS.md>) <!--ref:S10--><!--anchor:section:本轮新增核对-->

### 6.1 局部候选改善是真的，整体改善不能由好案例推断

确实存在后段明显接近GT的新路径，例如ID29201后半误差从约0.585降至0.161米，FDE从0.499降至0.175米。但完整CAL覆盖169变成167/161/165，仍无净改善。人眼感受到“部分候选更好了”与“全体质量没有升级”可以同时成立。

### 6.2 后半段和终点是重要失败位置，但不是所有失败的统一机制

38条批注中31条后半误差大于前半，只有5条相对F16属于“前半改善、后半恶化”。CAL seed7的30个lost中28个终点变差，前半平均误差约0.113→0.142米、后半0.189→0.355米、FDE 0.238→0.459米。说明后段代价较大，但该组前半也不是整体改善。

原F16在364/393窗口后半误差更大，新头为363/393，后段增长并不是新头独有缺陷。应逐例分辨终点漂移、持续变化和中段形状，而不能只凭误差随时域增长断言某个模块损坏。

### 6.3 “最终选择偏向顺延”必须区分原生选择与继承槽位

原F16的logit argmax是实际选择。当前修正实验没有重新训练或重新评分新几何，而是沿用原native的槽位。因此橙色选择线不能解释成“新head判断这个未来最可能”，原softmax也没有被验证为校准概率。

23个rescue批注中，只有2个继承选择同时合格，21个仍错过好路径。原选槽位接近恒速外推的几何倾向确实存在，但不能借此证明新头产生了新的保守决策。候选覆盖与评分一致性要分别研究。

### 6.4 min指标与有效数量都不能单独代表理解

头学习CAL中，有12/8/9个窗口两个独立min都达标，却没有同一条完整路径达标。最新实验第三种子比raw控制多101个有效好簇，却少覆盖21个窗口，说明好数量可以更集中在已成功样例。

因此，“理解运动状态”在当前研究里只能通过可观察行为评价：完整未来覆盖、响应类型/方向/时序、运动合理性、有效多样性和选择一致性。不能把oracle更近或更多轨迹弯曲直接当心理意图识别。上述人工核对见[后段与选择分析](<C:/Users/linke/Documents/New project/HEAD_LEARNING_HUMAN_FEEDBACK_ANALYSIS_20261005/RESULT_ANALYSIS.md>) <!--ref:S10--><!--anchor:section:二、后段缺口真实，但不是所有失败都属于“前改善后恶化”-->，最新数量反例见[有效好候选与选择](<C:/Users/linke/Documents/New project/GENERATION_OBSERVED_DISCREPANCY_SERIES_20261005/RESULT_ANALYSIS.md>) <!--ref:S14--><!--anchor:section:有效好候选与选择-->。

## 七、目前的方法：冻结原模型，再学习有界候选修正

### 7.1 原F16与新头的职责

本阶段并没有整体替换F16解码器。原F16继续生成不可变的20条原生未来路径，并提供冻结的观测特征、context和mode表征；可视化也可以比较原粗轨迹与refiner后轨迹。新的生成实验是在这些原生路径外增加修正头，原F16、完整原decoder和scorer的权重及buffer全部固定。

原模型先前的锚点、运动模式和细化结构仍是底座。本阶段真正改动的是“如何根据过去信息和候选条件，输出每步修正”，不是把GT预测节点后的信息输入模型，也不是自回归在线更新。只有8步观察及冻结observed tokens进入预测；GT用于训练loss与事后标签/评估。[原可视化结构边界](<C:/Users/linke/Documents/New project/TRAJECTORY_SELECTION_VISUAL_REVIEW_20261003/README.md>) <!--ref:S02--><!--anchor:section:候选生成—模型筛选人工复核--> [新头冻结范围](<C:/Users/linke/Documents/New project/GENERATION_PERSISTENT_RESPONSE_SERIES_20261005/DESIGN.md>) <!--ref:S16--><!--anchor:section:Unchanged protocol-->

### 7.2 当前候选状态修正头

最近两项均使用45798个可训练参数的新头，主要读取：原context64、原mode表征64、对8个observed tokens的attention摘要，以及候选预测运动状态32。状态编码器为32→32→32，主修正head为192→128→64→6。

状态32的内容是12步二维速度24维、平均速度2维、最后速度2维、终点位移2维、平均加速度幅值1维、首末预测朝向差1维。它来自原生候选，是原模型输出的确定函数，**不是新增现实观测或未来GT信息**。

最新实验比较原始state与减去最后观测速度参照的state：前30个速度/终点分量减去参照，最后2个统计不变。双方仍用共同raw FIT均值/标准差。这个变换可能改变优化几何，但不能解释为增加信息或纯因果语义改进。[当前对照说明](<C:/Users/linke/Documents/New project/GENERATION_OBSERVED_DISCREPANCY_SERIES_20261005/RESULT_ANALYSIS.md>) <!--ref:S14--><!--anchor:section:对照与可靠性-->

### 7.3 修正如何形成一条轨迹

新头输出6个数：二维漂移D、二维响应幅度A，以及曲线中心和跨度两个clock参数。当前使用可学习sigmoid持续profile，而不是此前去均值、零和的Gaussian暂态profile。

简化表达如下，u是每一步的位移增量修正，不是额外观测速度：

```text
profile(t) = sigmoid((t - center) / span)
L = 1.5 × (原协议speed + 0.03)
u(t) = clip[0.5 × L × (tanh(D) + tanh(A) × profile(t)), -L, L]
新候选(t) = 原生候选(t) + 从第1步到t的u累加
```

Gaussian零和暂态在相同codes下主要由drift决定终点修正；sigmoid持续响应的幅度、clock和累计终点相互耦合，更容易持续影响后段。但一个sigmoid不能保证表达多次转向、停止再起步等复杂过程，center/span也不是已识别的语义事件onset/duration。给不同训练模型同名参数，不等于最终终点匹配。[方法预注册](<C:/Users/linke/Documents/New project/GENERATION_PERSISTENT_RESPONSE_SERIES_20261005/DESIGN.md>) <!--ref:S16--><!--anchor:section:Sole implementation factor-->

### 7.4 训练目标与选择仍未改动

主要质量项为原hard winner的ADE+0.5FDE，加上既有样本平衡、多样性、残差、平滑和protection项；原FIT/CAL拆分、学习率顺序、80epoch预算、K20和common clip均保留。新头零输出初始化时与原生路径完全一致，以配对初始化对照改动。

训练质量winner使用GT挑选；原部署scorer用其logits选择。这是两个不同“选择”。当前修正后的候选只继承原评分槽位，尚未建立修正路径的重新评分模型。[训练协议](<C:/Users/linke/Documents/New project/GENERATION_PERSISTENT_RESPONSE_SERIES_20261005/DESIGN.md>) <!--ref:S16--><!--anchor:section:Unchanged protocol-->

## 八、关于损失函数：当前结论是什么

“FIT训练损失继续下降，而CAL变化样例没有改善”不能单独证明损失函数错误。这比较了不同数据和不同指标：加权综合loss可能因其他分项或简单状态改善而下降；也可能训练样例改善但迁移失败。

已有证据分别是：原目标在所查组奖励完整好路径；hard winner的直接质量反馈稀疏；真实新头主要梯度可达；富集小集FIT轨迹质量明显改善但CAL变差。它们更适合支持“奖励、反馈分配、共享条件映射与泛化之间有缺口”，不能把责任只归给loss或decoder。

这也没有排除损失与阶段覆盖目标不完全一致、不同正则/权重的作用，以及多候选训练反馈可能存在的问题。后续若专门检查目标，应对比同指标、同分组在FIT/CAL上的变化，再作可辨识对照，而不是仅凭总loss曲线直接换loss。[目标诊断](<C:/Users/linke/Documents/New project/F16_OBJECTIVE_RESPONSE_DIAGNOSIS_20261004/RESULT_ANALYSIS.md>) <!--ref:S07--><!--anchor:section:四、保留的反证--> [实际学习证据](<C:/Users/linke/Documents/New project/F16_HEAD_LEARNING_DIAGNOSIS_20261005/RESULT_ANALYSIS.md>) <!--ref:S09--><!--anchor:section:训练曲线与目标分项-->

## 九、可靠性与工程进展

### 9.1 已处理的工程问题

本阶段发生过进度文件原子替换权限失败、全量成对数组的内存分配失败、过于固定的数值校验断言及补充审核格式错误。相应失败代码、日志和来源被保留，基础设施或审核修复与正式模型负结果分开记录，没有把旧失败日志改成成功，也没有为通过检查而放宽科学门槛。

线索一致性诊断改为256行分块，将一个最大成对临时数组由约901MiB降到8.91MiB，跨分块结果与原算法一致。曲线诊断恢复验证了171个已完成块后复用，279个未完成块按原求解预算计算；数值校验改用高精度参考与逐候选舍入界，原位置误差、修正范围和见证margin未变。[内存恢复](<C:/Users/linke/Documents/New project/F16_CUE_RESPONSE_CONCORDANCE_20261004/recovery_memory_20261004/RESULT_ANALYSIS.md>) <!--ref:S05--><!--anchor:section:1. 工程恢复与完整审计--> [数值恢复](<C:/Users/linke/Documents/New project/F16_CURVE_REACHABILITY_DIAGNOSIS_20261004/recovery_numeric_20261004/RESULT_ANALYSIS.md>) <!--ref:S06--><!--anchor:section:1. 执行可靠性-->

### 9.2 实验可靠性不等于模型成功

正式实验完成要求worker和结果exit双SUCCESS；核验模型SHA、显式ID、有限K20×12×2预测、完整训练history、冻结权重/buffer，以及源码和旧结果SHA不变；从保存坐标重算指标、分组与注册CI。训练前还核验真实GPU梯度、原生零输出一致性、两arm实际小集2epoch连续/中断恢复的参数、optimizer及全部RNG一致。

这里的保存算术复核不是所有模型再次前向；同助手反方检查不是外部独立审稿。审核PASS表示“结果完整、来源与计算可追溯”，不表示“模型major通过”。最新实验正常完成，但覆盖退化，是可信的负结果。[最新可靠性记录](<C:/Users/linke/Documents/New project/GENERATION_OBSERVED_DISCREPANCY_SERIES_20261005/RESULT_ANALYSIS.md>) <!--ref:S14--><!--anchor:section:对照与可靠性-->

人类约17GB可用host RAM被作为长期资源约束。规划过程预算6GiB，并保留系统physical与commit各至少6GiB余量；按批次/分块内部检查，不修改pagefile、不关闭用户程序。最新正式实验峰值RSS约2.58GiB、private commit约3.37GiB，资源检查通过。[最新资源](<C:/Users/linke/Documents/New project/GENERATION_OBSERVED_DISCREPANCY_SERIES_20261005/RESULT_ANALYSIS.md>) <!--ref:S14--><!--anchor:section:资源与文件-->

运行流程保持：预检、实测吞吐与规模估计、独立唯一worker启动、启动后退出；在预计耗时50%时只作一次进度核验，按实际剩余修正完成节点，到截止再核验，不在对话里高频轮询。微基准和生产耗时有明显差异，未确认原因时不把速度变化解释为故障或优化成果。[注册流程](<C:/Users/linke/Documents/New project/GENERATION_PERSISTENT_RESPONSE_SERIES_20261005/DESIGN.md>) <!--ref:S16--><!--anchor:section:Resource and run workflow-->

## 十、目前到哪里，下一步是什么

### 10.1 当前状态

- 时序响应研究TIME/PULSE/PERSISTENT：3/3正常完成，完整总复盘已保存，停止第四个局部profile/clock变体。
- 候选状态研究：显式状态、相对运动偏差两项正式完成，2/3；没有任何一项升级为基线。
- 第三项偏差标准化：保存状态为设计待实施，未启动worker、未做正式训练，不能因存在设计文件或计划时间而称为running。

这些状态以本次读取的保存结果为准，不对已完成旧PID反复查询。[三项复盘](<C:/Users/linke/Documents/New project/GENERATION_PERSISTENT_RESPONSE_SERIES_20261005/RESPONSE_STUDY_REVIEW.md>) <!--ref:S12--><!--anchor:section:决策摘要--> [当前第三项状态](<C:/Users/linke/Documents/New project/GENERATION_DISCREPANCY_NORMALIZATION_SERIES_20261005/runtime_state.json>) <!--ref:S15--><!--anchor:section:status-->

### 10.2 已设计的下一项，只检验标准化因素

双方保持同一个centered state、同一新头与原协议，只比较套用raw FIT标准化与使用centered FIT自身均值/标准差。前30维改变，最后2维保持，标准差floor仍0.001，全部尺度仅由FIT计算，不按CAL/DEV效果选择。

理由是第二项centered输入继续套raw尺度后出现明显均值/尺度偏移，但该现象不是失败根因证明。下一项只能回答固定仿射预条件是否改善有限优化，不新增信息，也不能单独证明运动语义理解提升。第三项完成后必须完整复盘三项再选方向，不无限进行第四个标准化、宽度、系数或epoch变体。[设计依据](<C:/Users/linke/Documents/New project/GENERATION_OBSERVED_DISCREPANCY_SERIES_20261005/RESULT_ANALYSIS.md>) <!--ref:S14--><!--anchor:section:输入尺度证据与下一步边界-->

### 10.3 尚未解决的问题与权限边界

尚未分清的关键问题是：过去表征是否支持足够细的未来响应、共享修正映射是否在不同状态间相互干扰、原反馈分配是否足以驱动困难候选、时序/终点协调如何稳定泛化，以及修改后候选如何与评分一致。

不能用新scorer掩盖“没有好候选”，也不能在生成仍不稳定时一并改loss、decoder和scorer。当前持续授权覆盖合法newhead研究；新loss/事件监督、scorer训练、更多原模型解冻、独立测试/其他fold/外部数据等需要具体新决定。出现源锁、ID、资源、科学不可辨识或权限的实质阻碍时，应保留证据并中断计划等待人类决断。[当前边界与局限](<C:/Users/linke/Documents/New project/GENERATION_OBSERVED_DISCREPANCY_SERIES_20261005/RESULT_ANALYSIS.md>) <!--ref:S14--><!--anchor:section:反方检查3（同助手inline，非外审）-->

## 十一、结论边界和本报告自查

当前不是独立ETH/UCY泛化结论：DEV反复探索，CAL用于选模且部分诊断仅来自university，只有三个场景族与三个种子，窗口同轨迹相关，原F16已见过这些训练来源，既有邻居资格还有未来存在性选择偏差。未完成的人类past-only盲评不能冒称完成，也不能把未检出线索等同不可预测。[线索诊断限制](<C:/Users/linke/Documents/New project/F16_OBSERVABLE_CUE_DIAGNOSIS_20261003/RESULT_ANALYSIS.md>) <!--ref:S03--><!--anchor:section:人工材料与限制--> [总复盘限制](<C:/Users/linke/Documents/New project/GENERATION_PERSISTENT_RESPONSE_SERIES_20261005/RESPONSE_STUDY_REVIEW.md>) <!--ref:S12--><!--anchor:section:来源、反方检查与限制-->

本报告由AI辅助整理，采用academic-research-suite的来源核对、反例和写作检查；没有开展新训练、模型前向或新外部研究，也没有声称外部审稿。三次内联反方检查保留如下问题：

| 检查阶段 | 必须防止的错误 | 本报告处理方式 |
|---|---|---|
| 范围检查 | 把项目更早历史全部算入这轮，或遗漏诊断/正式身份 | 起点明确为本轮2026-10-03可视化，分开诊断与五项正式对照 |
| 证据检查 | 只看rescue、均值、正面图片，忽视lost与运动学 | 同时报净覆盖、退化、fresh对照、种子不一致和副作用 |
| 交付检查 | 把GT构造、旧slot、重复控制当预测/概率/独立验证 | 对oracle、继承选择、重复seed与保存算术均明确限定 |

最强反论仍然成立：GT已知时存在好曲线，并不能保证past-only共享模型能学会；局部新候选更近，并不能保证完整覆盖、运动合理性和真实选择共同提升。本阶段的价值是定位下一可检验环节，而不是提前宣布理解了行人意图、SOTA或解码器唯一根因。

## 附录：原始材料与可视化入口

以下链接指向本机项目原始材料，保留绝对路径；移动项目后需相应调整。

| 材料 | 入口 |
|---|---|
| 首次候选分类全量报告 | [REPORT](<C:/Users/linke/Documents/New project/CANDIDATE_SELECTION_VISUAL_AUDIT_20261003/REPORT.md>) |
| 原F16候选与选择交互工具 | [可视化](<C:/Users/linke/Documents/New project/TRAJECTORY_SELECTION_VISUAL_REVIEW_20261003/index.html>) |
| 可观测变化风险探针 | [结果](<C:/Users/linke/Documents/New project/F16_OBSERVABLE_CUE_DIAGNOSIS_20261003/RESULT_ANALYSIS.md>) |
| 低线索失败分层 | [结果](<C:/Users/linke/Documents/New project/F16_LOW_CUE_FAILURE_STRATIFICATION_20261004/RESULT_ANALYSIS.md>) |
| 线索—响应一致性，内存恢复后的正式解释 | [结果](<C:/Users/linke/Documents/New project/F16_CUE_RESPONSE_CONCORDANCE_20261004/recovery_memory_20261004/RESULT_ANALYSIS.md>) |
| GT辅助曲线可达性，数值恢复后的正式解释 | [结果](<C:/Users/linke/Documents/New project/F16_CURVE_REACHABILITY_DIAGNOSIS_20261004/recovery_numeric_20261004/RESULT_ANALYSIS.md>) |
| 原目标兼容诊断 | [结果](<C:/Users/linke/Documents/New project/F16_OBJECTIVE_RESPONSE_DIAGNOSIS_20261004/RESULT_ANALYSIS.md>) |
| TIME正式实验 | [结果](<C:/Users/linke/Documents/New project/GENERATION_TIME_RESPONSE_SERIES_20261004/RESULT_ANALYSIS.md>) |
| PULSE正式实验 | [结果](<C:/Users/linke/Documents/New project/GENERATION_PULSE_RESPONSE_SERIES_20261004/RESULT_ANALYSIS.md>) |
| PULSE改善与恶化可视化 | [可视化](<C:/Users/linke/Documents/New project/PULSE_RESPONSE_VISUAL_REVIEW_20261004/index.html>) |
| 新响应头小样本学习诊断 | [结果](<C:/Users/linke/Documents/New project/F16_HEAD_LEARNING_DIAGNOSIS_20261005/RESULT_ANALYSIS.md>) |
| 头学习改善与恶化可视化 | [可视化](<C:/Users/linke/Documents/New project/HEAD_LEARNING_VISUAL_REVIEW_20261005/index.html>) |
| 38条人工批注核对 | [结果](<C:/Users/linke/Documents/New project/HEAD_LEARNING_HUMAN_FEEDBACK_ANALYSIS_20261005/RESULT_ANALYSIS.md>) |
| PERSISTENT正式实验与三项总复盘 | [结果](<C:/Users/linke/Documents/New project/GENERATION_PERSISTENT_RESPONSE_SERIES_20261005/RESULT_ANALYSIS.md>) · [总复盘](<C:/Users/linke/Documents/New project/GENERATION_PERSISTENT_RESPONSE_SERIES_20261005/RESPONSE_STUDY_REVIEW.md>) |
| CANDIDATE STATE正式实验 | [结果](<C:/Users/linke/Documents/New project/GENERATION_CANDIDATE_STATE_SERIES_20261005/RESULT_ANALYSIS.md>) |
| OBSERVED DISCREPANCY最新已完成实验 | [结果](<C:/Users/linke/Documents/New project/GENERATION_OBSERVED_DISCREPANCY_SERIES_20261005/RESULT_ANALYSIS.md>) |
| 下一项标准化对照，仅设计 | [DESIGN](<C:/Users/linke/Documents/New project/GENERATION_DISCREPANCY_NORMALIZATION_SERIES_20261005/DESIGN.md>) · [状态](<C:/Users/linke/Documents/New project/GENERATION_DISCREPANCY_NORMALIZATION_SERIES_20261005/runtime_state.json>) |

**阶段总结：从“看起来预测差”推进到了“风险线索、粗响应、完整可达、实际学习、跨样例迁移和选择一致性”的逐层证据。部分候选能更好地覆盖GT，但完整、稳定、运动合理的生成提升仍未实现；这就是当前最重要的未完成目标。**
