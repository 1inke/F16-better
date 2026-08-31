# EXP-168A：联合 Hybrid K20（10/10）

## 目的

在固定总候选数 K=20 的条件下，检验结构化三锚点候选与自由运动候选能否在同一 coverage 目标下保留互补性。

- 共享编码器：F16 temporal-social cross
- Structured family：10 个 H3 三锚点候选 + 联合训练的 V2 有界修正，修正上限 `[0.20, 0.25, 0.50]`
- Free family：10 个 matched-K20 运动学自由候选
- 合并输出：始终为 K20，不存在 K40 后压缩或 family selector
- 损失：沿用锁定的 ModeQueryLoss 与 coverage loss；首轮不加 family balance/entropy/branch oracle
- 协议：官方 ETH/UCY 五折、seed 1、30 epochs

## 诊断

每折记录：winner family、soft responsibility、训练后期 100 batches responsibility、两分支梯度范数代理、活跃 mode 数、分支内多样性和跨分支距离。

## 晋级门槛

相对 EXP-164 seed-1 V2：

1. Macro J 改善不少于 2%；
2. 五折任一折 J 退化不超过 2%；
3. ETH 或 Univ 至少一折 J 改善不少于 4%；
4. 每一折训练后期两个 family 的 soft responsibility 均不少于 10%。

其中 `J = minADE20 + 0.5 × minFDE20`。

