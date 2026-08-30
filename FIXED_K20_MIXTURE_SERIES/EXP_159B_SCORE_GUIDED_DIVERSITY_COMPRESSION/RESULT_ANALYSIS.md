# EXP-159B Result Analysis

## Decision: REJECTED

最佳配置为K20=4、V2=16、diversity weight=0.20，宏平均minADE/minFDE/same-path FDE为0.2301/0.3789/0.4516。相对matched K20，最坏折ADE/FDE仍退化9.65%/9.07%。

## Next step

分数引导优于纯几何压缩，但仍不满足跨折门槛；验证训练集监督的轻量Top-20选择器。
