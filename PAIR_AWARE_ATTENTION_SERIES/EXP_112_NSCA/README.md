# EXP-112 NSCA

唯一变量：将 F16 temporal-social attention 的缩放点积相似度替换为 Q/K L2 归一化后的缩放余弦相似度。每个 head 增加一个可学习 logit scale，初始化为 `sqrt(head_dim)`；relation bias 在余弦分数后照常加入。

首筛使用当前跨场景主线：focal heading normalization、50% reflection/speed augmentation、Coverage 0.20、EMA 0.995、Kinematic K20。scene 1 和 scene 3 各运行 F16 同批控制及 NSCA，seed=1。

