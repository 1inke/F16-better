# ETH/UCY Oracle Candidate Geometry Series

统一使用 F16 temporal-social cross encoder、K=20、恒速运动中心、原 ModeQueryLoss 与原评分结构。目标是降低 minADE20/minFDE20，而非优化候选选择。

- EXP-066：条件锚点旋转；
- EXP-067：单个中间曲率控制点；
- EXP-068：受约束可学习运动基底；
- EXP-069：软 Oracle 覆盖损失。
- EXP-070：覆盖损失强度筛选；
- EXP-071：覆盖损失前5轮渐进启用。
- EXP-072：coverage weight 0.20 的五场景泛化验证。
- EXP-073：逐样本尺度归一化的覆盖责任分配。

旧实验已经否定上下文锚点尺度和训练集聚类锚点，本系列不重复。
