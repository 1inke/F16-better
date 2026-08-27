# EXP-118 RGVA-G4

Relation-Grouped Vector Attention。保持 F16 的基础多头 Q/K score，再在每个 24 维 head 内划分 4 个通道组，根据分组 QK 差异和已有运动关系生成残差校正，使同一邻居—时刻能够对不同运动通道采用不同权重。

校正强度初始化为 0.10，relation 校正层零初始化；稀疏图、完整 value、relation value 和固定 K20 均不改变。首筛为 seed=1、scene 1/3、30 epochs。

