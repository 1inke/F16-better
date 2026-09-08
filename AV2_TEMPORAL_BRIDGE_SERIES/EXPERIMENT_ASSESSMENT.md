# EXP-195 方法评估与修订版执行口径

## 总评

EXP-195A 的研究问题清晰，采用相同 target manifest、相同未来标签、相同 decoder、相同训练预算，仅改变历史采样率，适合作为低成本桥接实验。它不构成标准 AV2 leaderboard 结果，但能够回答 F16 是否可迁移、历史分辨率是否有增益，以及高频 token 的代价是否值得。

结论：**可以执行，但原指导不能原样编码。** 以下修订是得到可解释结果的前置条件。

## 必须修订的问题

1. **未来时间索引必须明确。** AV2 当前观测为 timestep 49；12 个 2.5 Hz 标签应为 `53,57,...,97`，分别对应未来 0.4–4.8 s。若从 50 开始每隔 4 帧抽样，会变成 0.1–4.5 s，并与原 ETH/UCY decoder 的物理时刻错位。
2. **“8/16/32 个点覆盖相同 3.2 s”只是数据集惯用命名，不是严格采样跨度。** 三者的首末点跨度分别为 2.8、3.0、3.1 s。脚本保留 `8/16/32` 设计，但在报告中明确这一点，不把它表述为严格相同跨度。
3. **位置编码应在当前时刻端对齐。** A1 沿用预训练坐标 `0..7`；A2/A3 使用 `7 + Δt_seconds × 2.5`。这既保持物理秒一致，又避免 A1 的零样本表示被无意改变。
4. **解码器运动特征也必须换算。** 当前三锚点 decoder 使用逐 token 位移，原指导只提到 encoder/graph 的速度与加速度。脚本将所有历史速率的 decoder motion feature 换算为 2.5 Hz 参考步长，否则 A2/A3 的运动量会缩小 2/4 倍。
5. **缺失轨迹不能用普通零填充冒充观测。** 当前 F16 没有逐时刻 validity mask。第一轮因此只纳入在 A3 历史窗口完整的 pedestrian context，并只选择历史与未来标签完整的 target；padding agent 由 `num_valid` 排除。

## 需要降低结论强度的问题

- 单 seed 下 1%–2% 的差异可能落在训练噪声内。第一轮只能称“筛选信号”；达到 GO 条件后，至少对 A1 与赢家补 3 seeds。
- target cap 是 target 数而不是独立 scenario 数，同一 scenario 的多个 target 具有相关性。主结果应同时保存 scenario_id，正式统计时建议做 scenario-level bootstrap。
- 2 m 的 MR6、Top-6 及 12@2.5 Hz 都是非官方诊断，不可与 AV2 官方 50→60@10 Hz 数字横比。
- Attention 的复杂度增长应由实测 latency/FLOPs 判断。当前 F16 是 factorized temporal-social encoder，不能仅根据标准全注意力的理论二次复杂度宣称 Mamba 必然占优。

## 建议决策门

- A1 的 minADE20 和 minFDE20 均优于 CV：桥接可行。
- A2/A3 相对 A1 的 minADE20 或 minFDE20 改善至少 2%，且 Selected ADE 退化不超过 2%：进入长历史实验，并补多 seed。
- 精度在 ±1% 内而 A3 batch-1 latency 增长至少 50%：可形成 Attention scaling / Mamba 动机，但仍需相同输入、decoder、训练预算的直接 Mamba 对照。
- A2/A3 退化超过 5%：先审查采样索引、时间 PE、decoder motion scale、图 gamma 与 mask，不能立即解释为“高分辨率有害”。

## 时间估算依据

同一台 RTX 5070 Laptop GPU 8 GB 上，已有 F16 正式实验记录为 28,791 个训练样本、30 epochs、727.8 s。按样本数和 epoch 线性折算，A1 的纯训练下限约 26 s；A2/A3 因时间 token 增加会更慢。另一次本机短基准中，8 线程读取并物化 400 个 AV2 场景、63 个合格 target 用时约 26.8 s。按合格 target 命中率外推，正式数据准备约 55–100 min，完整 EXP-195A 约 1–2 h；实时杀毒、机械盘或 GPU 功耗限制下可到 2–3 h。
