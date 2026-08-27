# EXP-119 CDSA

Compute-Neutral Differential Sparse Attention。每个 head 的 24 维 Q/K 分为两个 12 维分支，分别生成 softmax 图，并使用归一化差分权重聚合完整 V。两次 12 维点积的总 QK 乘加量与基线一次 24 维点积近似相同。

每个 head 的差分系数初始化为 0.05，并限制不超过 0.20；运动学稀疏 mask、relation bias/value、完整 8 帧和固定 K20 均保留。首筛为 seed=1、scene 1/3、30 epochs。

