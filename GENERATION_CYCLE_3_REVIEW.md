# 候选生成自主实验：第三周期总复盘

2026-10-02。三项全部正常完成且完整审计通过，全部未达到重大结果标准。18次新80轮训练，54policy条目，不是54次独立训练。所有原decoder/F16/scorer冻结、nativeK20/共同速度残差bound1.5，fit25912/cal2879/dev1516，同原loss+protection.1/cal协议。反复ETHdev与同3seed不等独立泛化。

## 三假设及主结果

| 校准选定治疗组三seed平均 |overall ADE/FDE|stable ADE/FDE|transition ADE/FDE|change ADE/FDE|
|---|---|---|---|---|
|Native|.219154/.399725|.175177/.313804|.275791/.512714|.372175/.696326|
|DIRECT_ANCHOR|0.218798/0.400823|0.172427/0.311370|0.280408/0.521312|0.378235/0.706719|
|ANCHOR_FEEDBACK|0.218374/0.402120|0.173752/0.314145|0.275106/0.513668|0.374389/0.710006|
|SLOT_HEAD|0.218866/0.396436|0.172828/0.306873|0.281366/0.514170|0.375808/0.705654|

1. DirectAnchor rawquery绕过原modeMLP特征：与fresh同context+mode control相比changeADE delta+.006852 CI[.000137,.014175]、FDE+.014867[-.002536,.032632]，同路径joint+.015353[.001531,.030839]。Rawquery不支持当前替代原mode表示；不能推断所有原mode层永久必要。对nativechange两CI跨0。
2. AnchorFeedback保留原mode+context增加原候选3anchor坐标：对freshzero changeADE+.002915 CI[-.004425,.010857]、FDE+.018416[.003348,.035736]，有探索性FDE增量伤害信号；对native两CI跨0。不是几何信息原则无价值，而是该支路没有把它转化为突变轨迹质量。相同参数但zero6有效容量低。
3. SlotHead解耦新最后输出层：对freshsharedchangeADE+.004424 CI[-.007183,.016307]、FDE+.013802[-.014530,.041483]；对native+.003633[-.007551,.015296]/+.009327[-.018320,.036482]，没有明确增量。稳定组ADE/FDE改善区间不跨0，但过渡组ADE保护失败。参数+29.46%，共享front仍耦合，不能把失败当作“梯度冲突不存在”的证明。

所有组、native与freshcontrol的完整效果/CI及全部门槛保存GENERATION_CYCLE_3_REVIEW.json（直接来自三份锁后summary，不只摘取有利组）。Cycle3sharedcontrol三seed再训练与DirectAnchorcontrol数值相同；它是fresh执行而非新随机独立验证。组别多重比较、仅3seed与场景相关性限制保留，任何小CI不跨0均是探索性观察。

## 同路径质量、覆盖与运动学

change同路径joint native.746631，rawquery.759034，feedback.752697，slot.752577；相应shared/zero control.743681/.743612/.743681。后两对control分别+1.22%/+1.20%超过samepath门槛；不得用独立minADE和minFDE两个oracle拼一条轨迹。

粗change family覆盖native91.03%，rawquery93.72%，feedback92.53%，slot92.68%：方向“存在”仍不等于转向时机、速度、位置准确。GT附近候选集中目标没有被原则否定，先前实现未同时改善oracle/coverage。三个overall加速度与jerk均值各约+7.39%/+10.24%/+13.99%。硬阈值通过不替代分布保护，不能放宽guard追求名义胜利。

## 选择/学习/干扰

治疗组选中epochs raw76/80/75、feedback58/31/72、slot26/23/24。Finalchange ADE/FDE分别.378614/.706288、.374654/.712406、.388912/.733070，均无突破；不同早晚选择削弱单一“轮数不够”解释。先前FIT/CAL诊断fit收益强于cal提示泛化差距，但不能仅凭final恶化证明唯一过拟合原因。
新共享头FIT/CAL固定128/group、3zero checkpoint梯度stable-change余弦fit[.100,-.164,-.122]cal[-.458,-.093,.016]；存在局部冲突，但组与mode不是同一概念、winner槽位重叠，解耦slot不能保证解除冲突。这是checkpoint局部描述，不是全训练历史或总体CI。所有gradient smoke只说明可达，不保证学习有效。

## 大方向调整

第三周期更换了输出参数化、候选几何输入和输出权重共享方式，仍没有突变5%双指标信号。结合前两周期historyGRU/集合pool/平滑/原geometry/gates/full原decoder负结果，停止这些输出侧局部变体及盲目系数/更多epoch扫参。没有证明F16不可用或未来意图不可辨识，也没有证明冻结主干是唯一瓶颈。

下一周期转向“生成器读取冻结观测时序表示”：当前新head依赖原context(last+mean tokens压缩)和原modeMLP。新分支直接读取8个原F16 observation tokens，比较mode-conditioned temporal cross-attention与uniform pooling（相同value/query/key参数名义数量，但uniform下q/k数据梯度零，有效capacity不同）。保留原context/mode输入与共同native安全参考，改的是输入读取不是扩大slothead/加入GT专家loss。曾经historyGRU读rawvelocity和SET_CONTEXT读candidatepool，不等于此fullfrozen-token readout已被测试。

假设是压缩/读取丢掉条件信息；尚未证明瓶颈，注意力不等意图理解。固定32维新读出、共享3anchorhead、原loss/cal/bound/K20；不去找dev最佳维数/注意力系数。6fresh80epoch3seed，对照均值也有直接tokens，不能只比较“有无额外数据”。需要FIT-only token方差/attention sensitivity/实际GPU新增参数梯度、完整冻结、init/nativeidentity/hash/splits、botharm中断EXACT，然后才开新cycle重置计数。新分支仍native-constrained correction不是独立绝对生成。保留ALL5%/1%/3seed/nativecontrolCI/运动学/覆盖门槛，首次major额外cluster/three-seed充分性审查。No test/crossfold/unfreeze/cloud/外部数据/messages/deletion；扩范围另问具体授权。

当前NEXT_DIRECTION_DESIGN，无新训练运行。下一设计文件GENERATION_TOKEN_READOUT_DESIGN.md；代码与预检通过才报实际估时、OS独立启动一次、60%时间查询再修正。不得用重大目标未达成为由伪称成功或创建重复goal。
