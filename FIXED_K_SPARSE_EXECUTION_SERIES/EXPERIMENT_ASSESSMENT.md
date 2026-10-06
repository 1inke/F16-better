# Fixed-K 指导评估

## 总体判断

方向正确，是 EXP-197 后最后一轮值得做的执行层实验。固定宽度布局能够去掉
`nonzero + scatter_reduce + scatter_add`，把主要 relation/attention 路径变为规则
gather、连续 softmax 和连续 reduction，直接检验 scatter overhead 是否是延迟失败原因。

## 必要修正

1. 新实验编号采用 EXP-198；EXP-197 已经用于上一轮 EdgeExec，不能再拆成 197A-E。
2. 指导把 `TOP_K=4` 误当成正式图的执行上限。正式配置是
   `adaptive_kinematic, K_MIN=2, K_MAX=6`；真实批次已观察到 6 个非自环邻居。
   因此严格等价布局必须是 social bound=6，加 self 后物理总宽度 7。
   Synthetic 协议才使用 social K=4、总宽度 5。
3. `allowed` 只保存布尔邻接，未保存原 top-k rank。Fixed-K 使用固定宽度 `topk`
   直接抽取有效槽，避免全量 source sort；softmax 语义对 slot 排列不敏感，数值
   等价仍由 1e-5 Gate 约束。
4. padding slot 必须执行 relation MLP 后再 mask，因而 Fixed-K FLOPs预计略高于动态
   EdgeExec；需要用 slot utilization 明确报告这项规则化代价。
5. 原 F16 对 padded target 存在均匀 softmax fallback，且会影响完整输出。实现保留该
   边界路径；`nonzero` 只存在于 padded fallback，不出现在有效 agent 的主 attention。
6. profiler-accounted FLOPs 并不计入全部 gather/index/softmax 开销，必须和 latency、
   memory、pair/slot count 并列解释，不能单独作为加速证据。

## 判定建议

沿用文档的 stop rule。Fixed-K 若在 B64 仍比 masked dense 慢超过 5%，或 N>=32
始终无 crossover，应停止通用 GPU sparse runtime 优化，不进入 Triton/CUDA；保留
数值等价、FLOPs 与 memory scaling 作为轻量化证据即可。

正式协议预计约 20–35 分钟；若加入 N=256，约增加 5–10 分钟。全程 0 epoch。
