# 阶段二物理加速度保护：结果与下一步
日期：2026-10-05（香港时间）。本报告使用 academic-research-suite 的证据/推断分离及反方检查，由同一助手内联完成，不是外部独立审稿。

## 结论
实验正常完成，但 major 全项失败，F16 baseline 不变。固定 .1 加速度保护确实减少了加速度副作用，却没有同时建立稳定的完整好候选覆盖改善。不能据此说 loss 唯一有问题，也不能说 head 的结构性冲突已证明。
下一步选择合法 newhead 的候选条件化/共享映射对照，而不是继续扫加速度权重、soft winner、epoch 或 normalization。具体新预注册见 ../GENERATION_CANDIDATE_ROUTED_MAPPING_SERIES_20261005/DESIGN.md；目前仅设计，尚无预检或生产 worker。

## 可靠性与执行
真实独立 worker39304/service2728，18:55:41.279151→19:59:21.969362 HK，实际63分40.690秒。用户继续请求触发的本轮唯一 check_once 于20:20:15.8592548发现完整SUCCESS；这是迟到的第一次检查，不冒称19:35按时检查或观察到半程运行状态。不得再次查完成PID/重训，原20:15规划截止已失效。
worker与results exit双SUCCESS；6全新80epoch训练，18policy模型/预测（不是18独立训练）。显式1516 uniqueIDs、有限20×12×2；全部保存sample/nativeALLgroups、72perseed/24aggregate/96探索CI、quality完整GT无关complete-link聚簇/全局运动学、temporal/profile分解和36FITCAL物理population核验通过。484source/30298旧文件SHA不变，全部旧F16/原decoder/scorer权重及bool/scaler buffers固定。
完成补充 completion_review/REVIEW.json：仅NumPy保存数据，无torch/模型forward/optimizer；36population originaltotal/P重算最大误差0，18保存quality分组计数与6条80epoch诊断记录复核，18model/36population outputs/484source/30298old前后SHA通过。完整几何聚簇和所有CI由完成worker重算，本补充未第二次全部重放，非外部审稿。raw控制全部9policy所有数组与之前CandidateState实际state治疗bitwise EXACT，fresh重复不增加独立泛化证据。
生产peakRSS2.659/private3.463GiB，最低physical16.099/commit15.411GiB；补充private.052GiB。长期17GB可用规划仍按process6GiB/systemphysical+commit各保留6GiB，不修改OS/pagefile或停止用户程序。

## 单因素
zero：原total。mean_std：原total+.1 P_acc。双方相同active rawstate32/FITrawscalers、45798 persistent head，quality hardwinner/原所有其他loss/数据/CALguard/LR/budget/K20/nativeclip/旧F16-decoder-scorer冻结不变。
P是现有stable/transition/change minibatch组的ALL20×11绝对加速度均值native-relative1.01 hinge，按固定FIT组scale及n_g/B归一化。不是逐候选“不能加速”，也不是population硬门槛。未来GT/groups/logits不进入预测forward。
训练自然P首次激活在seed1/2/3第8/9/8epoch，证明不是全程inactive。第80epoch变化组active_present_fraction均1，added loss约.000329–.000367（batch-n加权），不把小标量误读为小梯度。first_batch_coordinate_P_gradient_norm .660–.687是输出坐标范数，不是参数范数。
physical_population是全split分母；objective_history是batch-n加权；legacy train_loss是batch均值。不得混用。

## 固定DEV1516：正式constrained政策
严格好候选是同一条完整轨迹 ADE≤.2m AND FDE≤.4m，不拼两个slot或前后段。
| 指标 | F16 | 无新增P控制 seed1/2/3 | 加P seed1/2/3 |
|---|---|---|---|
| overall完整覆盖（1516） | 733 | 744/742/749 | 727/732/737 |
| change完整覆盖（223） | 43 | 53/48/51 | 42/47/52 |
| overall有效good簇总数 | 929 | 938/942/828 | 927/931/820 |
| 旧argmax继承slot good | 281 | 285/283/289 | 281/279/283 |
加P overall原失败救回4/24/26、原成功丢失10/25/22，净-6/-1/+4。种子重复相同IDs，不把计数相加成独立人。change组救回0/9/12、丢失1/5/3。有效good不能抵消覆盖丢失，继承argmax不是新轨迹概率或scorer学习。

change平均minADE/FDE：F16 .372174740/.696326435；控制 .367063254/.686794281；加P .371380866/.690220416，相对F16收益仅 .2133%/.8769%，低于各5%。相对native与freshcontrol的ADE/FDE/joint探索CI均跨0。protected误差、hardmotion、重复等部分通过不等全部major通过。
加P DEV change全局pooled加速度均值害 seed1/2/3 +.340/+1.596/+2.568%，控制原+5.23至10.66%范围大幅降低；但seed2/3仍超1%。其他p95/jerk接近native不能覆盖这一失败。加Pconstrained的speedp95/accelp95/jerkmean,p95全seed-group在1%内；单seed1运动学通过不构成3seed成功。

## FIT/CAL同组：final并非部署结果
以下基于新增36保存population数组，changeFIT6410/CAL700：
| 时点/组 | F16 | 无P seed1/2/3 | 加P seed1/2/3 |
|---|---|---|---|
| constrained FIT good | 2603 | 2739/2716/2844 | 2606/2661/2709 |
| constrained CAL good | 243 | 242/253/246 | 244/247/246 |
| final FIT good | 2603 | 3345/3336/3398 | 3237/3254/3265 |
| final CAL good | 243 | 230/216/223 | 230/242/232 |
所以不是“加P把FIT学习完全消灭”：finalFIT仍50.50–50.94% vs40.61%native；但CAL32.86/34.57/33.14%仍未超34.71%native。constrained CAL变化组略改善是重要反例，不能说所有组/时点都恶化。
finalchange加速度害 FIT控制+15.50–15.97%→加P+1.72–1.88%；CAL控制+14.12–14.51%→加P+1.44–1.55%，仍超原1%保护。因此当前组合改善运动学，却未解决CAL迁移或正式覆盖收益。
population原total：加P final FIT .324483/.324271/.324058；CAL .413630/.415822/.416870。不能把训练下降称CAL下降，或与不同组分母loss混比。所有时点原total/P和全部组记录可查results/physical_population/summary.json和completion_review/REVIEW.json。

## 综合前面诊断
1. 原loss不是完全无完整覆盖信号：九final FIT变化覆盖40.61→51.50–53.87%，loss↓19–22%；CAL覆盖34.71→30.86–34.57%，CALloss↑1.71–4.73%，且运动学代价大。
2. 已有完整good时，训练quality winner通常good，finalFIT winnerbad<.8%、CAL最多1.67%；不是当前部署scorer对齐结论。Jclosest!=Qwinner不能等同“最可修候选被不给梯度”。旧GTGaussianwitness与persistent可学性不可互换。
3. constrained quality下降方向与accelmean/jerkmean冲突，原residualsmoothness不等绝对物理运动；finalCAL方向反例保留。真实physical feedback现在有效，但仍失败，说明单加一个保护项不足，不能由此单独排除权重/优化/结构。
4. 风险可读不是时刻方向路径可预测；旧freeincrement/timequery/velocitybasis/conditioninggeometry/mode/social/softset-energy-mixture失败仍是反证，不能夸大“理解意图”。

## 决策与反方2（分析内联）
决定测试候选条件化分支的路由策略，不修改质量assignment、loss权重或评分器。目标是检验共享修正与candidate-specific分工在固定物理反馈下是否有不同coverage/motion/FITCAL表现。
Major反例：P可能压制真正变化的候选，当前aggregate减少无法证明是共享参数干扰；新route是可检验假设而非根因结论。
Major：候选状态是native预测，不是新增现实信息；路由可能强化原错误。此前state/discrepancy失败不能忽略。
Major：两专家控制/治疗同nominal参数不等有效容量、梯度样本数或优化几何。只能称路线策略对照，不能纯归因“干扰被解除”。
PASS_EXPLORATORY_NEXT_DESIGN，仅在源/真实GPU/双实际2epochexactresume/资源全部预检通过后才可独立启动。

## 反方3（交付内联）
最强反驳：当前最稳定的效果可能只是缩小修正趋近native，没有扩大正确路径的可泛化空间；即使下一项更好也需全部原门槛，而非改阈值追认。
3scene/窗口轨迹相关/F16已见/CAL单university且用于选checkpoint/重复DEV/仅3seed限制，所有结果是探索，不SOTA/心理意图/不可预测率或decoder唯一根因。
现行严谨门槛不放宽。若下一设计的源/ID/映射/资源/科学可辨识或模块权限实质失败，STOP并保存证据等待人类，不无限诊断或局部变体。物理feedback study本项1/3，第三新formal完成前不得冒称总review完成；历史cycle11不重置，goal未完成。

