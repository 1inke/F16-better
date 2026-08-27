# EXP-049 Identity-Value Attention

受 ICLR 2024 Simplifying Transformer Blocks 启发，删除可学习 V 投影，直接将 temporal feature 按 heads 拆分为 values，减少一次全量历史 token 线性投影。
