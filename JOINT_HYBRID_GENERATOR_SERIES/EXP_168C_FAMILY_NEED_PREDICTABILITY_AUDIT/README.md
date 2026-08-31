# EXP-168C：Family Need Predictability Audit

## 目的

在不读取 outer-test 的条件下，判断 observation-only 特征能否预测 structured/free 哪个 family 包含最佳候选。

## 数据协议

- 使用 EXP-168A 每个 outer fold 已保存的 5% inner-validation indices；
- 不使用对应 outer-test；
- 轨迹模型 checkpoint 完全冻结；
- 目标定义为哪个 family 包含最小 `ADE + 0.25×FDE` 候选；
- 每个 outer fold 内做固定五折 stratified CV。

## 比较

- speed：只使用观测速度和位移；
- full：增加加速度、转向、非线性和邻居特征；
- 分类器固定为 class-balanced logistic regression，不搜索超参数。

## 晋级门槛

只有同时满足以下条件，才允许实现 observation-conditioned allocator：

1. full-feature macro balanced accuracy ≥ 0.60；
2. 每一 fold balanced accuracy ≥ 0.55；
3. macro ROC-AUC ≥ 0.65；
4. 相比训练折内决定的最佳 constant family，候选质量改善 ≥ 3%。

