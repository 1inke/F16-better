# 候选生成自主实验：第四周期总复盘
2026-10-02。三项正常SUCCESS，分别18个模型SHA/18显式ID有限K20/全部存储指标独立重算/冻结审计PASS；本周期18次fresh80epoch训练、54个policy产物，不是54次独立训练。累计四个新周期12项均未达到重大标准。无需重跑旧PID44332/45988/43320。

## 三个假设与实际效果
共同native突变minADE/minFDE=.372175/.696326，joint=.746631。
1. TOKEN_READOUT：候选条件注意力读8个冻结tokens，对照均值投影。attention .368992/.691457 (约.86%/.70%收益)，uniform .367050/.688901。attention-native ADECI负、FDECI跨0；attention-uniformADE+.001942 CI[-.000046,.004164]、FDE+.002556[-.001016,.006996]，joint+.003681[.000121,.008112]探索性增量伤害。无注意力增量证据。39590名义参数双方相同，uniformquery/key数据梯度零，非等有效容量。
2. TOKEN_CONTRAST：均值保留、last4-minus-first4值差分 vszero。difference .367102/.689724 (1.36%/.95%)，zero .367399/.688230；对nativeADE-.005072[-.007264,-.003007]、FDE-.006602[-.013490,-.000966]探索性改善。对freshzeroADE-.000297[-.001772,.001207]、FDE+.001494[-.001865,.006069]跨0。36454名义参数相同，zero额外输入梯度零。没有差分有效增量证据。
3. TOKEN_SEQUENCE：均值保留、8步有序中心化value序列线性读出 vszero。sequence .367472/.690307 (约1.26%/.86%)，zero .367399/.688230；nativeADE-.004702[-.007097,-.002406]、FDE-.006019[-.013363,.000808]；freshzeroADE+.000073[-.002236,.002450]、FDE+.002077[-.002413,.006549]均跨0。44646名义参数相同，zero时间投影数据梯度零。完整有序读出仍无明确增量。
全部组别constrained效应/CI、其他policy诊断/epochs/源summary引用保存GENERATION_CYCLE_4_REVIEW.json，不只摘有利组。Contrast与Sequence控制组数值一致，fresh执行但同seed和有效架构，不能算独立重复/泛化。样本场景相关性、多组多次探索和仅3seed限制仍在，不能按最好一条CI建立结论。

## 同路径、覆盖、运动学与保护
三治疗组change joint分别.741141/.737608/.738922，控制.737460/.738073/.738073；所有samepath1%控制保护PASS，不能拼不同oracle成单条GT轨迹。三者change粗familycoverage均.905830vsnative.910314，未增加；coverage/duplicate/hardmotion通过不替代准确性。
三者overallaccel和jerk均值各约+2.40%/+2.24%/+2.06%，超过1%分布门槛；虽然比周期3伤害小，仍不能放宽门槛。三治疗组总体/稳定/过渡保护PASS，uniform-readout对照stableFDE+1.20%失败；其名义突变收益较大不是合法新baseline。三治疗组每seed native双oracle改善PASS，但全部5%/nativeANDfreshcontrolCI/motiondistributionFAIL。
各group速度p95/accelmean,p95/jerkmean,p95完整值及其他保护都在JSON；不把速度残差bound当GT误差或物理保证。

## 选择、学习与泛化
attention最佳epoch23/29/17，halfcontrast19/21/20，sequence19/18/18；控制23/29/26、23/21/30、23/21/30。Final突变attention .368497/.693493，halfcontrast .365006/.683394，sequence .362186/.689170；即使最后一轮ADE更低也未达5%双指标，且后期校准保护不可选。禁止依据dev事后换checkpoint、延长轮数或减少guard，final仅诊断。早期可选、后期不可选支持训练/校准权衡的担忧，不能据此单独证明过拟合为唯一原因；不能把梯度preflight当有效学习证明。
之前三周期集合/时间基底/历史GRU、原native几何decoder/门控、rawquery/原候选feedback/slot解耦均无major。本周期三种时序读取也未提供freshcontrol明确增量，不证明所有注意力无用、未来意图不可预测或F16主干一定是瓶颈。所有方法都保留nativebase、冻结旧主干/scorer，是受限修正而非独立生成或筛选改善。

## 大方向调整
停止继续时序pool、半段差分、完整序列的局部系数/维数/epoch变体。下一方向先做FIT/CAL-only的冻结空间交互表示检查：原cache social96来自agent_aware_trans_encoder.last_agent_features(focal行)，observations-only生成，当前nativeprepared把它清零、仅显式送past+8tokens。F16tokens已经经过social tokenfeedback，因此绝不宣称当前模型没有social信息；旧残差模型曾用该cachedsocial，新增直读是新anchor支路交互而非首次发明social。
先核验该tap来源/哈希/IDs、FIT/CAL方差/线性可重建性、实际frozen模型小批次复算对齐；若有不一致先排查，不用dev选择特征。支持后再锁定直接social32投影vszero32的新支路fresh对照，保留原context/mode和均值token读取、nativebase原loss/cal/bound、F16/scorer全冻结。仅测读取方式，不证明信息增益/意图。名义容量和zero分支有效容量差必须披露。若tap几乎重复或无法核验，不盲开，另选证据支持方向；实质解冻F16/heldtest/crossfold等需要人类具体授权。
当前NEXT_DIRECTION_DIAGNOSIS，无新正式训练。下一设计GENERATION_SOCIAL_TAP_DESIGN.md；预检/actualGPUgradient/frozenidentity/K20/hash/split/两组optimizerRNGresumeEXACT后才报估时OS独立启动，并在真实启动才重置cycle5计数。保留全部major标准及首次cluster/3seed充分性审查。自主授权持续，不把未达标称goalcomplete。
