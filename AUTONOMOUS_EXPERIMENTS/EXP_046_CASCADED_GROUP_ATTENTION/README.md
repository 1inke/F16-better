# EXP-046 Cascaded Group Attention

受 EfficientViT 启发，将通道分给 4 个 heads，各 head 只使用 head-dim QKV 投影，并将上一 head 的输出注入下一 head。目标是削减 QKV 投影参数和 MAC。
