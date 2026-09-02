# EXP-179A Candidate Perturbation-Stability Audit 结果分析

## 结论

**EXP-179A FAIL，停止 uncertainty scorer，不进入 EXP-179B。**

本实验冻结 EXP-173 `F16 + H3 + Joint-V2` 五折 seed1 权重，完整覆盖 34,161
个 ETH/UCY 测试样本。每个样本包含 original、P1+/P1-、P3+/P3- 五次主审计
前向，以及一次 P2 补充前向。mode identity 固定，所有轨迹均反变换至共同全局坐标
后比较。

## 三项预注册门槛

| 门槛 | 实际结果 | 要求 | 判定 |
|---|---:|---:|---|
| `trajectory_U` 五折 macro Spearman | -0.0366 | `abs(rho) >= 0.15` 且至少4折同号 | FAIL |
| Q4 相对 Q1 真实质量差 | 1.21% | >= 10% | FAIL |
| Top5 non-winner - winner uncertainty | +0.001149，5折均值同向 | 正向且至少4折 | 形式 PASS，但效应很弱 |

总门槛要求三项同时成立，因此最终决定为 `STOP_UNCERTAINTY_SCORER`。

## 主稳定性信号的跨场景表现

| Fold | 样本数 | Spearman(`trajectory_U`, residual quality) | Q4 vs Q1 quality gap |
|---|---:|---:|---:|
| ETH | 364 | -0.0803 | +3.96% |
| HOTEL | 1,197 | -0.0486 | +41.84% |
| UNIV | 24,334 | +0.1184 | +21.28% |
| ZARA01 | 2,356 | -0.1699 | -43.44% |
| ZARA02 | 5,910 | -0.0023 | +9.04% |

该信号并不具备跨场景可迁移性。UNIV 上“越不稳定、误差越高”符合原假设；ZARA01
却强烈反向，表示更敏感的候选反而质量更好。其余三折相关性很弱。由于 UNIV 占总样本
的 71.2%，样本加权结果会被 UNIV 主导，因此必须坚持五折等权宏平均，不能用 pooled
统计宣称有效。

Q1/Q4 的方向同样不稳定：HOTEL 和 UNIV 差异较大，但 ZARA01 完全翻转，使五折总体
差距只剩 1.21%。这不满足训练统一 uncertainty head 的前提。

## Top5 区分能力

所有折的“non-winner 平均不确定性减 winner 不确定性”均为正，因此形式门槛通过；
但样本级 winner 更低的比例分别为：

| ETH | HOTEL | UNIV | ZARA01 | ZARA02 |
|---:|---:|---:|---:|---:|
| 48.63% | 71.51% | 53.43% | 48.47% | 64.15% |

ETH 和 ZARA01 低于随机二分类意义上的 50%，UNIV 也仅为 53.43%。平均差异主要由
HOTEL 拉高，不能单独证明该信号足以改变最终排序。

## 其他稳定性特征

没有任何辅助特征达到 `abs(macro Spearman) >= 0.15`：

| 特征 | Macro Spearman | 跨折方向 |
|---|---:|---|
| endpoint U | -0.0477 | 4负1正 |
| teacher U（0.7轨迹+0.3终点） | -0.0421 | 4负1正 |
| logit variance | +0.0411 | 3正2负 |
| rank variance | -0.0129 | 3负2正 |
| V2 correction variance | -0.1074 | 5折均负 |
| endpoint direction variance | +0.0318 | 3正2负 |
| P2 trajectory U | -0.0201 | 3负2正 |
| P2 endpoint U | -0.0329 | 3负2正 |

V2 correction variance是最接近门槛且唯一5折同号的特征，但相关性仍不足，而且符号为负：
较大的修正变化更像模型进行有效自适应，而不是候选不可靠。不能直接把它作为惩罚项。

## 专业判断

轻微输入扰动测到的是“局部响应敏感性”，并不等价于“预测误差不确定性”。对当前多模态
解码器而言，一些弯曲或可调整的有效 mode 本来就会随观测发生更明显变化；刚性、泛化的
候选反而可能非常稳定但误差更大。这解释了多个折上的负相关和 ZARA01 的强烈翻转。

## 决策

- 不执行 EXP-179B Stability Distillation Scorer。
- 不补 seeds2/3，不扩大 uncertainty head，不扫描惩罚系数。
- 不使用多次前向稳定性进行在线重评分。
- 保留该实验为输入扰动敏感性不能跨场景预测候选误差的负结果。
- 按阶段指导转向独立的 EXP-180A Q3KV Runtime Stability Audit；该实验只解决部署
  延迟问题，不改变当前正式模型的精度结构。
