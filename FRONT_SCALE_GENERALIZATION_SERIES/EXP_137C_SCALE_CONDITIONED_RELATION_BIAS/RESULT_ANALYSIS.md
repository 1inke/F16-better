# EXP-137C 结果分析

## 结论

不晋级，社会注意力尺度偏置暂时降为低优先级。

## 三种子结果

- General：minADE +0.02%，minFDE -0.77%，Top-1 ADE/FDE -0.90%/-0.99%。
- Challenge：minADE +0.27%，minFDE +0.03%，Top-1 ADE/FDE +0.44%/+0.06%。
- Challenge Same-path FDE 平均改善 0.18%，但整体 minADE/minFDE 没有改善。
- 参数仅增加 156（+0.055%），训练时间增加约 5.0%–5.6%。

## 解释

困难划分的低密度属性意味着许多目标缺少足够邻居，关系偏置无法修复主要的单体运动外推误差。General 上的局部 FDE 改善不足以证明域泛化。
