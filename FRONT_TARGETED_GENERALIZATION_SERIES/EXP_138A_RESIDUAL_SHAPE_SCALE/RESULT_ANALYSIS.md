# EXP-138A 结果分析

## 结论

本轮唯一保留候选，但当前版本不直接晋级；下一步校准残差幅度。

## 三种子结果

- General：minADE -0.17%，minFDE -0.96%，Top-1 ADE/FDE -2.18%/-1.70%。
- Challenge：minADE +0.27%，minFDE -0.22%，Same-path FDE -0.30%，Top-1 ADE/FDE -1.61%/-0.98%。
- General Top-1 两项 3/3 种子改善。
- Challenge minFDE、Same-path FDE、Top-1 两项均为 2/3 种子改善。
- Challenge minADE 只有 2/3 改善，seed 2 恶化 1.17%，导致均值未通过。
- 参数量 290,859，相比 Baseline 增加 2.18%。

## 判断

保留 Baseline robust state 后，EXP-137 替换式方案的域外失效明显缓解，说明尺度残差方向成立。但当前 0.25 残差上限略强或缺少约束，需要比较 0.10/0.15 幅度及 residual-norm 正则。
