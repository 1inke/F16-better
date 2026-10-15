# D4 / D5a 固定轻量探针

2026-10-09；在父T0算术结果之后预注册，在任何本探针结果之前锁定。既有五fold官方train bank / evaluation IDs，要求train/test target track无交集；每fold训练原有其他场景，测试被留场景，不新增数据。无神经模型训练、无旧E3恢复。

D4：GT-CV全24维残差在训练集中心化、PCA前2方向；仅训练basis/mean/std。自身特征=7步速度、6步加速度、7步速度模、6次转向sin/cos、静止比例、净位移；不含绝对位置。邻居特征=各valid非target邻居last相对位置/速度/CV最近距离与12步tca的均值和std，以及最近距离/邻居数；无GT未来输入。ridge以标准化特征训练协方差+固定0.01 I求解，截距不惩罚。均值、basis、weights全保存。不会在测试选alpha/特征数/PCA维。以CV为baseline重构完整path，同组all/P1/C1/C2分别报skill与top1差值。

D5a：只对原C3+awaydefined训练样本，目标g为原12step最近away方向投影GTendpoint-CVendpoint。相同fixedridge的self与self+social比较，用训练g均值作合法基线，另报相对0技能；savedprediction perID，track bootstrap2000。邻居量利用原观测+CV而非实际未来。

反方1：两维PCA/线性ridge不足不是信息上限或不可预测证明；添加特征无增益也不能证明数据无交互。特征条件性/regularization/有限样本/单场景目标皆可替代解释。只单固定线性探针不扫alpha。位置同场景留recording D4(c)不属于本次LOSO对照；未做，不混入跨场景成绩。辅助拟合秒级不是F16训练/原框架重训。任何资源/科学/ID gate失败保留failure停止，不放宽。
