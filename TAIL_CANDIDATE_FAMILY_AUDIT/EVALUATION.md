# 对 Tail-Candidate Audit 建议的评估

## 总体判断

建议是当前最合适的下一步。它不是继续增加loss，而是先验证tail coverage是否存在稳定、可学习的结构，符合CDA1之后的证据链。

## 必须修正的歧义

1. `oracle mode id > 10`不能表示tail。mode ID是持久身份，tail必须由每个样本的score rank定义。
2. ADE winner和FDE winner经常不同，必须分别计算tail gain和winner。
3. 未来轨迹聚类本身不是HCB1依据。必须加入随机全future聚类对照和锁定validation上的medoid replacement proxy。
4. cluster数量不能只按最好silhouette选择，还要检查不同初始化的ARI稳定性和最小簇占比。
5. HCB1不能在audit之前预先实现，否则容易根据validation结果反复修改候选结构。

## 晋级规则

每个split同时满足：

- material tail-needed不少于300个样本且占训练折至少5%；
- silhouette ≥0.25；
- 多初始化ARI ≥0.60；
- 最小cluster占比 ≥3%；
- 训练tail medoid在锁定validation proxy上使minADE20或minFDE20改善 ≥0.5%。

General与Challenge都通过才允许HCB1训练。该规则故意严格，因为HCB1会改变candidate family，验证成本和过拟合风险都高于loss消融。
