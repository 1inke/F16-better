# F16 真实稀疏执行指导评估与执行方案

## 结论

实验方向正确，而且应当优先于 Sparse-Native 训练。当前正式 F16 确实是
“稀疏图结构 + dense relation/attention 执行”；已有 F41 只稀疏了 attention，
没有消除完整 `B×N×N×T×8` relation 构造与编码。本实验能够补齐轻量化论证，
但应命名为 EXP-197，不能复用已经被 AV2 系列占用的 EXP-196。

## 对原指导的修正

1. 正式 checkpoint 合同的总参数量是 355,048，不是 284,642；其中 trainable
   为 266,582，decoder 为 87,889。
2. 当前正式 F16 的 allowed graph 来自原有 mask/valid-agent 路径，不依赖完整
   relation encoder。因此无需近似重建 graph；先取得完全相同的 allowed，再
   gather edge relation 即可。
3. 总复杂度不能笼统写成完全从 `O(N²T)` 变成 `O(ET)`。构图所需的低维距离或
   mask 生成可能仍有 `O(N²)` 成本；准确表述是 relation encoder 与 temporal-social
   attention 的主 pair-time 分支从 `O(N²T)` 降为 `O(ET)`。
4. `density < 0.5` 不是充分成功条件。还必须同时报告有效 agent、self edge、
   非自环 edge、padding fallback、实际 edge ratio 和端到端占比。
5. “F16-Dense”容易歧义。正式模型应称 `F16-MaskedDenseExec`；全连接图版本只适合
   kernel/scaling 参照，因为它改变了模型语义，不能进入精度等价表。
6. 小 N 下 scatter/gather 的启动与索引开销可能使 EdgeExec 更慢。成功标准应以
   数值等价、显存增长曲线、pair MACs 和中大 N crossover 综合判定。
7. padded target 在原 dense softmax 中有可观察的均匀 fallback。本实现显式复现
   该边界行为，否则全张量等价会失败；有效目标仍只走 edge-first 路径。

## 已验证的实现 Gate

使用正式 EXP-154 seed-1 checkpoint 的快速验证结果：

- 两路参数量：355,048，完全一致；
- graph mismatch：0；
- raw relation feature max error：0；
- encoded relation max error：不高于 3.58e-7；
- encoder max error：不高于 7.16e-7；
- logit max error：不高于 1.43e-6；
- trajectory max error：不高于 1.91e-6。

因此当前实现已经通过第一轮零训练数值等价 Gate。

## 推荐正式协议

1. 先运行 20×16 batch 等价检查；任何一项超过 1e-5 就停止。
2. 对 ETH/UCY 五个 processed train folds 做 graph density；SDD 和 AV2 必须使用
   各自原生 config、尺度和 checkpoint 单独接入，不能直接套 ETH 的 scale/radius。
3. 在 batch 1/16/64 上做 50 warmup、200 iterations、10 interleaved cycles。
4. 在 N=4/8/16/32/64/96/128、T=8、d=96、H=4、k=4 上比较 full graph、
   masked dense execution 和 edge execution。
5. 只有上述 Gate 通过后，才补正式全验证集精度 checksum 和 profiler 分解；
   不重新训练。

## 预计时间（RTX 5070 Laptop GPU）

| 工作 | 预计时间 |
|---|---:|
| smoke + 20×16 数值等价 | 1–2 分钟 |
| ETH/UCY 每 fold 2,000 样本 density | 1–3 分钟 |
| batch 1/16/64 正式 interleaved benchmark | 4–7 分钟 |
| N=4–128 synthetic scaling | 3–6 分钟 |
| ETH/UCY 全量 density | 4–8 分钟 |
| SDD + AV2 原生接入与完整 density/checksum | 10–20 分钟 |
| module/operator profiler 与汇总 | 5–10 分钟 |

核心 EXP-197 正式脚本约 10–18 分钟；连同 SDD、AV2 和 profiler 的完整论文级
证据包约 25–45 分钟。这里不含代码开发时间，也不含训练时间，因为本阶段为 0 epoch。

## 快速结果的解释边界

缩短版计时只用于验证脚本可执行，不能直接写入论文。它已经显示预期形态：真实
ETH/UCY batch=1 未必加速；synthetic N=128 时 EdgeExec 才开始出现 latency crossover，
但在 N=128 的 peak allocated memory 从约 151.7 MB 降至约 8.3 MB。正式结论必须以
完整 warmup/iterations/cycles 重新测量。
