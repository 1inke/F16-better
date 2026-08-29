# DSA正式结果分析

## 1. 预注册结论

```text
social_branch_gate.passed = false
motion_branch_gate_passed = false
scene_shortcut_detected = false
recommended_next_experiment = freeze_tail_specific_modeling_and_focus_general_encoder_robustness
```

不编写DRS1、DRM1或tail-specific gate。Challenge不是单一motion、单一social或scene-ID问题，而是速度尺度、运动非线性、局部密度与encoder特征幅值共同发生的大幅condition shift。

## 2. Domain classification

| Split | D0 Scene | D1 Individual | D2 Social | D3 Ind+Social | D4 Encoder | D5 All |
|---|---:|---:|---:|---:|---:|---:|
| General | 0.4970 | 0.4974 | 0.5138 | 0.5105 | 0.4974 | 0.4662 |
| Challenge | 0.4882 | 0.9086 | 0.9113 | 0.9693 | 0.9226 | 0.9930 |

General所有结果接近0.5，证明审计没有因为普通训练/验证划分或流程泄漏而人为制造domain可分性。

Challenge则几乎可以被全部可见信息完美识别。Scene-only仍为0.4882，因此不是scene ID shortcut。Individual与Social都超过0.90，二者组合达到0.9693，说明偏移是互补的motion-social condition shift。

## 3. 最大特征偏移

| 特征 | Train mean | Challenge mean | SMD |
|---|---:|---:|---:|
| last speed | 0.2131 | 0.4598 | +1.5348 |
| encoder individual norm | 19.4437 | 29.4806 | +1.5074 |
| mean speed | 0.2164 | 0.4504 | +1.4969 |
| social delta norm | 17.9290 | 25.7875 | +1.4780 |
| observation displacement | 1.4901 | 3.0983 | +1.4573 |
| mean acceleration | 0.0225 | 0.0521 | +1.2028 |
| neighbour count | 6.7229 | 4.0623 | −1.0509 |
| observation nonlinearity | 0.0583 | 0.1297 | +1.0282 |
| density r6 | 6.6236 | 3.9396 | −1.0193 |
| density r4 | 6.0267 | 3.2906 | −1.0023 |

Challenge样本不是简单“更密集”。它们运动更快、加速度和历史非线性更强，但平均邻居和局部密度反而更低。这说明motion-challenge split主要富集剧烈个体运动，同时改变了社会上下文分布。

Social domain classifier的主要系数来自neighbour count、social delta norm和density，而D3同时依赖neighbour count、acceleration、density和nonlinearity。这进一步支持“混合偏移”，而不是单一社会交互效应。

## 4. Matched-strata结果

### Union label

| Split | 匹配对数 | Val保留率 | M1 Individual | M2 Ind+Social | M2−M1 | 95% CI |
|---|---:|---:|---:|---:|---:|---:|
| General | 1515 | 99.93% | 0.6120 | 0.6139 | +0.0019 | [−0.0228, 0.0252] |
| Challenge | 1968 | 64.91% | 0.5626 | 0.5774 | +0.0148 | [−0.0061, 0.0370] |

Challenge点估计为正，但：

- 未达到预注册的+0.03；
- 500次bootstrap区间跨0；
- AUPRC只增加0.0065；
- General几乎没有增量。

ADE label上Challenge增量为+0.0163，FDE为−0.0007，也没有任何一个达到门槛。因此当前没有证据支持专门训练DRS1。

## 5. 匹配质量限制

| Challenge confounder | 匹配前SMD | 匹配后SMD |
|---|---:|---:|
| mean speed | 1.4969 | 0.5193 |
| mean curvature | 0.0666 | 0.1312 |
| density r4 | −1.0023 | −0.3408 |
| neighbour count | −1.0509 | −0.5336 |

粗分层明显缩小了偏移，但匹配后仍未达到常用的|SMD|<0.1，而且丢弃了35.09%的Challenge validation。这意味着：

> 本结果足以说明Social没有达到DRS1的晋级证据，但不足以证明Social的独立贡献严格为零。

不能把+0.0148解释为因果效应，也不能依据该点估计继续训练社会tail gate。

## 6. Encoder robustness

### General

- tail AUROC：Encoder P4 0.6704，Raw P5 0.6679；
- domain AUROC：Encoder D4 0.4974。

General下F16基本满足task-informative且domain-invariant。

### Challenge

- tail AUROC：Encoder P4 0.5837，Raw P5 0.5945；
- domain AUROC：Encoder D4 0.9226；
- 全部特征domain AUROC：0.9930。

Challenge下encoder表示非常容易暴露domain，却没有提高tail-risk预测。F16在普通随机划分上表现正常，但在困难运动分布上主要继承并放大了速度/条件尺度差异，没有形成鲁棒抽象。

## 7. 研究决策

1. 正式停止tail-specific decoder modeling，包括CAT1、DRS1和确定性tail gate。
2. 不把问题归因于scene ID，也不单独选择motion或social分支。
3. 保留K20与当前三锚点解码器。
4. 下一阶段转向通用encoder domain robustness，重点处理：
   - 绝对运动尺度与归一化运动形状的解耦；
   - social/relation特征的尺度和密度归一化；
   - speed×density平衡采样或增强；
   - 原始条件与encoder表示之间的跨尺度一致性。
5. 若以后需要声称Social具有独立贡献，必须使用更细的propensity/nearest-neighbour匹配并达到匹配后|SMD|<0.1；当前审计不能支持该强结论。

最终可靠结论不是“社会信息无用”，而是：

> 当前F16面对的是强烈的混合motion-social domain shift；已有社会信号不足以晋级专门分支，最有潜力的方向是让整个encoder学习尺度解耦、条件归一化且跨分布稳定的表示。
