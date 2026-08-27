# EXP-087 Augmentation Replication

对精度主线 `F16 + K20 + Coverage 0.20 + EMA + rotation/reflection/speed augmentation` 做正式稳定性验证：

1. scene 1 补齐 seed 2/3，与统一基线三种子对照；
2. seed 1 补齐内部 scene 0/2/3/4，与统一基线做跨场景加权比较；
3. 保留每个运行的日志、检查点、划分和指标。
