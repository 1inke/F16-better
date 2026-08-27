# EXP-056 Low-Dimensional Q/K Attention

受 EfficientViT 参数重分配启发，保留 4 个独立并行 heads 和全维 V，只将每个 head 的 Q/K 维度从 24 压到 8。目标是降低 Q/K 投影参数与 QK 相似度 MAC，而不引入 Cascaded Group 的串行执行或 GQA 的 head 共享。
