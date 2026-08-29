# 对DSA建议的评估

建议与TCA→TNPA形成合理的诊断链，值得执行。它不再尝试直接优化tail，而是先识别可迁移信号、domain-specific信号和混杂因素，能够减少继续获得微小且不稳定模型增益的风险。

本实现增加了以下约束：

1. General与Challenge都执行相同审计，General作为负控制。
2. Domain classifier使用5折out-of-fold概率，避免报告训练集拟合指标。
3. Domain两类下采样为1:1，AUROC/AUPRC和校准可直接比较。
4. 匹配分层边界只由训练折定义，不使用验证标签。
5. 匹配严格控制scene、speed、curvature和density，并报告匹配前后SMD及共同支持保留率。
6. Social晋级除点估计≥0.03外，要求bootstrap区间下界大于0且General同方向。

匹配后仍然存在未观测混杂，因此本审计支持“独立预测信息”而不是严格因果效应。
