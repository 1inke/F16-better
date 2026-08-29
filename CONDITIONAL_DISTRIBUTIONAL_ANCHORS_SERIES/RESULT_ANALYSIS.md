# CDA1 正式结果分析

## 1. 最终结论

CDA1 **未达到 seed1 晋级标准，不进入 seed2/3，也不建议继续 CDA2 learned scale**。

- 相对严格 matched continuation：
  - General minADE20 改善 **0.28%**，minFDE20 退化 **0.36%**；
  - Challenge minADE20 改善 **0.20%**，minFDE20 基本不变（退化 0.01%）。
- 两个 split 的 K=2/5/10 均稳定改善，但收益在 K=20 明显衰减或反转。
- top-1 指标小幅、同向改善；候选多样性没有坍缩。
- social-needed 没有达到 2% 门槛，General social minFDE 反而退化 0.93%。
- 相对项目当前 source B0，CDA1 也没有形成稳定的 minADE/minFDE 双改善。

CDA1 证明条件混合似然可以改善候选排序和中等 K 覆盖，但单 GT 下的 posterior 仍主要训练已经接近真值的少数模式，不能补强 tail candidate family，因此不符合当前追求 minADE20/minFDE20 大幅提升的目标。

## 2. 严格 matched continuation 主结果

正数表示误差上升，负数表示改善。

| Split | 模型 | minADE20 | minFDE20 | same-path FDE | top-1 ADE | top-1 FDE |
|---|---|---:|---:|---:|---:|---:|
| General | matched control | 0.17897 | 0.31431 | 0.37916 | 0.46470 | 1.02097 |
| General | CDA1 | 0.17848 | 0.31544 | 0.37744 | 0.46399 | 1.01934 |
| General | 相对变化 | **-0.28%** | **+0.36%** | -0.45% | -0.15% | -0.16% |
| Challenge | matched control | 0.27875 | 0.47840 | 0.56757 | 0.75216 | 1.62190 |
| Challenge | CDA1 | 0.27819 | 0.47847 | 0.56844 | 0.75060 | 1.61866 |
| Challenge | 相对变化 | **-0.20%** | **+0.01%** | +0.15% | -0.21% | -0.20% |

minADE 的方向在两个 split 上一致，但绝对幅度只有 0.00049 和 0.00056，远低于预设 0.5% 相对改善门槛，也不接近用户期望的 10% 量级。

## 3. 与原 source B0 的关系

| Split | 指标 | source B0 | CDA1 | 相对变化 |
|---|---|---:|---:|---:|
| General | minADE20 | 0.17868 | 0.17848 | -0.11% |
| General | minFDE20 | 0.31407 | 0.31544 | +0.44% |
| Challenge | minADE20 | 0.27807 | 0.27819 | +0.04% |
| Challenge | minFDE20 | 0.47784 | 0.47847 | +0.13% |

因此 CDA1 不能替代当前 source B0。General 的 top-1 ADE/FDE 相对 source 改善 0.73%/0.66%，但这是候选选择端的小收益，不是 oracle candidate coverage 的提升。

## 4. ADE/FDE@K

| Split | K | ADE 变化 | FDE 变化 |
|---|---:|---:|---:|
| General | 1 | -0.15% | -0.16% |
| General | 2 | -0.54% | -0.42% |
| General | 5 | -0.61% | -0.34% |
| General | 10 | **-0.62%** | **-0.70%** |
| General | 20 | -0.28% | **+0.36%** |
| Challenge | 1 | -0.21% | -0.20% |
| Challenge | 2 | -0.63% | -0.61% |
| Challenge | 5 | -0.56% | -0.51% |
| Challenge | 10 | **-0.80%** | **-0.78%** |
| Challenge | 20 | -0.20% | +0.01% |

这是本轮最明确的信号：CDA1 对前 10 个候选有稳定价值，但没有改善完整 20 候选的尾部上限。它更接近 ranking/central-mode refinement，而不是 tail coverage 方法。

## 5. social-needed

| Split | minADE20 变化 | minFDE20 变化 | same-path FDE 变化 |
|---|---:|---:|---:|
| General | -0.26% | **+0.93%** | -1.70% |
| Challenge | -0.26% | -0.02% | -0.54% |

虽然 same-path 和部分 top-1 指标改善，但 social oracle 指标没有达到 2% 晋级线。CDA1 没有把社会交互信息转化为新的未来分支，只是重新调整现有模式的局部责任。

## 6. posterior 与 NLL

| Split | 模型 | NLL/维 | N_eff | posterior top-1 mass | ECE |
|---|---|---:|---:|---:|---:|
| General | control | 3.0225 | 2.208 | 0.765 | 0.0689 |
| General | CDA1 | 3.0454 | 2.217 | 0.763 | 0.0609 |
| Challenge | control | 10.5917 | 1.188 | 0.945 | 0.0775 |
| Challenge | CDA1 | 10.4747 | 1.192 | 0.943 | 0.0773 |

- General 的 ECE 有所改善，但 validation NLL 反而退化 0.76%。
- Challenge NLL 改善 1.11%，但 ECE 几乎不变。
- Challenge 的 train scale 校准时 `N_eff=2.21`，到 motion-challenge validation 降到约 1.19，posterior 几乎成为 hard winner。
- score top-1 与 anchor winner 的一致率仍很低：General 约 5.4%，Challenge 约 1.45%。

这说明固定 shared scale 对普通验证集尚可，但不能适应运动难度分布偏移。更重要的是，即使 NLL 改善，也没有稳定转化为 minADE20/minFDE20 改善。

## 7. 为什么 ADE 改善而 FDE 不改善

训练折得到的尺度随预测时刻增大：

- General：0.165 / 0.227 / 0.309；
- Challenge：0.150 / 0.205 / 0.279。

高斯似然的坐标梯度近似按 `1/sigma²` 加权，因此第 12 步终点由于 sigma 最大，实际权重低于第 4、8 步。这会更偏向改善早期和中段锚点，从而改善 ADE，却不足以改善 FDE。General 的 minFDE 反转正是该目标权衡的直接表现。

## 8. 多样性、mode usage 与 anchor coverage

- 整轨迹多样性相对 control：General 增加 0.82%，Challenge 增加 1.04%。
- anchor 多样性：General 增加 0.81%，Challenge 增加 0.94%。
- 20 个模式仍然全部成为过 winner，没有 mode collapse。
- General 三个 anchor coverage 均轻微退化 0.12%～0.53%。
- Challenge 三个 anchor coverage 改善约 0.55%～1.00%。

因此 CDA1 的失败不是 MA1 式收缩，而是目标与 minADE20/minFDE20 的对齐不足以及 tail modes 缺乏有效监督。

## 9. 训练过程

- General control 最优在 epoch 9，CDA1 最优在 epoch 12。
- Challenge control 最优在 epoch 6，CDA1 最优在 epoch 9；epoch 12 已回落。
- 不是简单增加训练轮数可以解决的问题。
- 参数量、可训练参数和推理结构与 B0 完全相同；mixture likelihood 仅训练期使用。
- 四次训练合计约 19 分钟。

## 10. 研究判断与下一步

### 当前决策

1. 不运行 CDA1 seed2/3。
2. 不继续 CDA2 learned shared scale。
3. 不扫描 lambda、sigma multiplier 或 endpoint weight。
4. CDA1 可保留为“top-10/ranking 有效但 full-set coverage 无效”的机制性消融。
5. 按预设建议，停止继续微调现有 20-mode/3-anchor decoder 的训练目标。

### 为什么不建议继续 learned scale

learned scale 可能缓解 Challenge 的 hard posterior，但它更直接优化的是 likelihood calibration，而不是产生当前候选集合缺失的稀有未来。它还可能通过放大 variance 降低 NLL，或让更多 mode 同追单 GT。结合当前不足 0.3% 的 minADE 收益，继续该链路很难接近目标中的大幅提升。

### 应保留的结论

CDA1 说明当前模型的前部候选仍有约 0.5%～0.8% 的可优化空间，但 K=20 family 已成为主要瓶颈。下一阶段若仍追求显著精度提升，需要改变未来信息来源或 candidate family，而不是继续调整同一个单 GT 下的 responsibility 公式。

本轮最重要的结论是：**概率责任可以改善已有候选的排序和局部几何，但无法从单条真实未来中恢复缺失的尾部分支。**
