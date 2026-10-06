# EXP-196 指导内容评估

## 总体评价

方案的科学顺序合理：先消除 EXP-195 的训练预算混杂，再做多 seed，随后分别扩展时间长度与邻居数量，最后才考虑 Mamba。它比直接把 Attention 换成 Mamba 更容易形成可信的因果证据。

## 可直接保留的设计

1. A1C/A2/A3 从相同 A1 checkpoint 继续相同 epoch，是必要的公平对照。
2. Long History 固定 10 Hz 和 future decoder，只改变真实历史长度，变量控制正确。
3. 使用 scenario-level paired bootstrap，能处理同一场景多个 target 的相关性。
4. `T`、`N` 分轴测试并只补两个 joint 点，成本合理。
5. Mamba 只替换 temporal mixer，保留 graph、decoder、loss 和 manifest，公平性正确。

## 必须修正或补充的部分

1. **长历史初始化未指定。** 本实现让 L1–L4 全部从同一个 A1 checkpoint 开始并训练相同 4 epochs，避免 L2/A3 预适配带来的偏置。
2. **500-target dense cohort 门槛不符合当前数据分布。** smoke 扫描约 5,000 个训练场景，在完整 5 秒历史条件下只有 2 个 `>=5` 邻居目标。Dense 主实验改用 32-step 完整历史池，S3 不足时合并 `S2+`，并报告真实样本数；joint 另建稀有样本池。
3. **碰撞指标需要邻居 future。** EXP-195 给邻居填的是静止伪 future，不能用于碰撞结论。本实现对有真实 future 的邻居保存有效标记，只在这些邻居上计算 collision/near-collision；无有效邻居的样本记为缺失而不是 0。
4. **效率测试不能只顺序运行一次。** EXP-195 两次正式 profiling 的 A3/A1 batch-64 latency 比率有明显波动。本实现随机交错配置顺序并默认重复 3 cycle。
5. **长历史运动 cohort 必须有可执行定义。** 当前实现预注册 near-stationary、acceleration、turning 和 curvature proxy 的阈值，并输出各 cohort 样本数；正式论文前仍应做阈值敏感性分析。

## 仍需谨慎解释

- 该实验是单数据子集上的非官方诊断，不可写成 AV2 leaderboard 结果。
- `top_k` 固定不等于实际边数固定；应同时报告 effective edges 和 selected-neighbor 分布。
- N=24 的计算压力包含 padding/mask 开销，accuracy 则只由真实有效 agent 决定，两者需分开解释。
- Joint dense 样本极少，适合压力测试与现象发现，不适合单独作为总体精度结论。
- 只有当 continued-A1、多 seed 和 bootstrap 同时支持效果时，才能把 EXP-195 的 2%–3% 差异解释为稳定的高分辨率收益。

## 推荐决策顺序

1. 先完成 196A/B；若 A3 对 A1C 的平均收益不足 1% 或 CI 跨零较多，停止扩大高频分支。
2. 196C 重点看 L4 对 L2 的 paired scenario improvement 与运动 cohort，而非只看总体均值。
3. 196D 先比较 D2/D4，再用 D3/D5、D4/D6 判断是 agent truncation 还是 top-k capacity。
4. 只有准确率已稳定，且 `T`/`N` 扩展至少两项资源指标明显恶化时，才进入 EXP-197。
