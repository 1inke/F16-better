# 对 CFB2 建议的评估

## 总体评价

建议抓住了 CFB1 最可信的机制解释：future bank 含有有用替代未来，但固定权重、极端多样化选择与共享梯度共同损伤了 tail coverage。先做 PFRA，再决定是否训练，比直接扫描 pseudo loss 权重更合理。

## 值得保留的部分

- 允许每个样本使用 0/1/2 条 pseudo，而不是固定 3 条；
- 从最大 diversity 转向可靠性和邻居共识；
- 冻结 B0 主干并使用独立 adapter；
- 稳定 mode responsibility；
- 同时报告 K=1/2/5/10/20、tail winner 和主干偏移；
- PFRA 失败时立即终止路线。

## 原建议中需要进一步约束的部分

1. PFRA 若在验证集上挑 confidence 公式或阈值，会产生评估偏差。本实现只在训练折拟合与校准。
2. stop-gradient 之后，adapter 仍会替换两个原候选，因此不能声称 K=20 理论上绝不退化；只能保证其余 18 条候选逐值不变。
3. 只按 retrieval distance 阈值不足以识别伪未来可靠性。本实现同时使用未来邻居共识、cluster support 和观测运动尺度差。
4. 每 batch nearest-mode 仍会造成职责漂移。本实现使用固定两个 mode 槽，并按终点横向次序建立稳定对应。
5. 训练器自带的 oracle coverage loss 也会产生 GT 梯度，必须关闭，否则“伪监督隔离”不成立。

## 风险判断

CFB2 仍然不是高概率获得 10% 提升的方案。它首先是一次机制验证：判断 CFB1 的中 K 收益能否在严格隔离下保留。即使成功，合理预期仍是 0.5%～2% 级别；若 PFRA 相关性弱或 CFB2 再次损伤 K=20，应正式停止 retrieval-based pseudo-future supervision。
